"""
HPOLightningExperiment — HPO Optuna + eval gold du best trial.

Flow :
    1. Optuna study avec nested MLflow runs (metrics par epoch)
    2. Sauvegarde du checkpoint du meilleur trial
    3. Chargement des poids → eval gold (GPU) → MLflow → promotion @champion

PAS de retrain. Les poids du best trial sont directement évalués sur le gold.
Inspiré du Population Based Training (Jaderberg et al. 2017, DeepMind) et du
ASHA scheduler (Li et al. 2020) : ne jamais retrainer from scratch après HPO.

Complète le strategy pattern :
    - SklearnExperiment : fit sklearn + eval gold
    - BaseLearnerExperiment : fit base learner + @active
    - LightningExperiment : fit Lightning + eval gold + @champion
    - HPOLightningExperiment : Optuna HPO → eval gold du best trial → @champion
"""
from __future__ import annotations

import logging
import os
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.experiments.strategies._metrics import expected_calibration_error


logger = logging.getLogger(__name__)


def _suggest_hp_from_yaml(trial: optuna.Trial, search_space: dict) -> dict:
    """Traduit le search space YAML en appels Optuna suggest_*."""
    hp = {}
    for name, spec in search_space.items():
        t = spec["type"]
        if t == "float":
            hp[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "int":
            hp[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "categorical":
            hp[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Type HPO inconnu : {t} pour {name}")
    return hp


class HPOLightningExperiment:
    """
    HPO Optuna → checkpoint best trial → eval gold → promotion.

    Args:
        model_factory: callable(config_dict) → LightningModule.
            Les base learners sont capturés dans la closure.
        dm: DataModule configuré. setup() appelé avant fit().
        config: dict avec sections hpo, model, trainer, mlflow, promotion.
    """

    def __init__(self, model_factory, dm, config: dict):
        self.model_factory = model_factory
        self.dm = dm
        self.config = config

        hpo_cfg = config.get("hpo", {})
        self.n_trials = hpo_cfg.get("n_trials", 10)
        self.study_name = hpo_cfg.get("study_name", "m3_hpo")
        self.search_space = hpo_cfg.get("search_space", {})

        self.model_cfg = config.get("model", {})
        self.trainer_cfg = config.get("trainer", {})

        mlflow_cfg = config.get("mlflow", {})
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "http://mlflow:5000")
        self.experiment_name = mlflow_cfg.get(
            "experiment_name", "M3_attention_fusion"
        )

        promotion_cfg = config.get("promotion", {})
        self.registry_model_name = promotion_cfg.get(
            "registry_model_name", "rakuten-m3-attention-fusion"
        )
        self.promotion_threshold = promotion_cfg.get("threshold", 0.005)
        self.n_classes = self.model_cfg.get("n_classes", 27)
        self.promotion_enabled = promotion_cfg.get("enabled", True)

    # ================================================================== #
    # FIT — séquence complète                                              #
    # ================================================================== #

    def fit(self) -> None:
        """
        1. Optuna study (nested MLflow runs)
        2. Load best trial checkpoint
        3. Eval gold + MLflow logging + promotion
        """
        logger.info("[HPO] START")
        start_time = time.time()

        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        gold_loader = self.dm.gold_dataloader()
        gold_labels = self.dm.get_gold_labels()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # Stocke le meilleur checkpoint par trial
        trial_checkpoints: dict[int, str] = {}

        # Répertoire temp pour les checkpoints
        ckpt_dir = tempfile.mkdtemp(prefix="hpo_ckpts_")

        # ============================================================== #
        # 1. Optuna study — nested MLflow runs                             #
        # ============================================================== #

        with mlflow.start_run(run_name="M3_HPO") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.log_param("hpo/n_trials", self.n_trials)
            mlflow.log_param("hpo/study_name", self.study_name)
            for k, v in self.search_space.items():
                mlflow.log_param(f"hpo/space/{k}", str(v))
            # T.2b — log retrain strategy params
            mlflow.log_params(self.dm.retrain_params())

            def objective(trial: optuna.Trial) -> float:
                hp = _suggest_hp_from_yaml(trial, self.search_space)

                d_model = hp.get("d_model", self.model_cfg.get("d_model", 512))
                n_heads = self.model_cfg.get("n_heads", 8)
                if d_model % n_heads != 0:
                    raise optuna.TrialPruned(
                        f"d_model={d_model} % n_heads={n_heads} != 0"
                    )

                trial_cfg = {**self.model_cfg, **hp}
                model = self.model_factory(trial_cfg)

                n_params = sum(p.numel() for p in model.fusion.parameters())

                with mlflow.start_run(
                    run_name=f"trial_{trial.number}", nested=True
                ) as child_run:
                    for k, v in hp.items():
                        mlflow.log_param(k, v)
                    mlflow.log_param("n_trainable_params", n_params)

                    mlf_logger = MLFlowLogger(
                        experiment_name=self.experiment_name,
                        tracking_uri=self.tracking_uri,
                        run_id=child_run.info.run_id,
                    )

                    # Checkpoint pour sauvegarder le meilleur état
                    trial_ckpt_dir = os.path.join(ckpt_dir, f"trial_{trial.number}")
                    ckpt_callback = ModelCheckpoint(
                        dirpath=trial_ckpt_dir,
                        monitor="val/f1_weighted",
                        mode="max",
                        save_top_k=1,
                        filename="best-{epoch}-{val_f1_weighted:.4f}",
                    )

                    pruning_cb = PyTorchLightningPruningCallback(
                        trial, monitor="val/f1_weighted"
                    )

                    callbacks = [
                        EarlyStopping(
                            monitor="val/f1_weighted",
                            patience=self.trainer_cfg.get("patience", 3),
                            mode="max",
                        ),
                        ckpt_callback,
                        pruning_cb,
                    ]

                    trainer = L.Trainer(
                        max_epochs=self.trainer_cfg.get("max_epochs", 15),
                        accelerator=self.trainer_cfg.get("accelerator", "auto"),
                        precision=self.trainer_cfg.get("precision", "16-mixed"),
                        callbacks=callbacks,
                        logger=mlf_logger,
                        enable_progress_bar=False,
                        enable_checkpointing=True,
                        log_every_n_steps=10,
                    )
                    trainer.fit(model, train_loader, val_loader)

                    val_f1 = trainer.callback_metrics.get("val/f1_weighted", 0.0)
                    if hasattr(val_f1, "item"):
                        val_f1 = val_f1.item()

                    mlflow.log_metric("final_val_f1", val_f1)

                    # Sauver le chemin du meilleur checkpoint
                    best_path = ckpt_callback.best_model_path
                    if best_path:
                        trial_checkpoints[trial.number] = best_path

                logger.info(
                    f"[HPO] Trial {trial.number} — {hp} — "
                    f"{n_params:,} params → val/f1 = {val_f1:.4f}"
                )
                return val_f1

            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,
                    n_warmup_steps=5,
                ),
            )
            study.optimize(objective, n_trials=self.n_trials)

            # Log summary sur le parent
            best = study.best_trial
            mlflow.log_metric("hpo/best_val_f1", best.value)
            mlflow.log_param("hpo/best_trial", best.number)
            for k, v in best.params.items():
                mlflow.log_param(f"hpo/best/{k}", v)

            trials_summary = []
            for t in study.trials:
                trials_summary.append({
                    "number": t.number,
                    "value": t.value,
                    "state": t.state.name,
                    **t.params,
                })
            mlflow.log_dict(
                {"best_trial": best.number, "trials": trials_summary},
                "hpo_summary.json",
            )

        # ============================================================== #
        # 2. Load best trial checkpoint                                    #
        # ============================================================== #

        print(f"\n{'='*60}")
        print(f"[HPO] BEST TRIAL #{best.number}")
        print(f"[HPO]   val/f1 = {best.value:.4f}")
        print(f"[HPO]   params = {best.params}")
        print(f"{'='*60}")

        best_ckpt = trial_checkpoints.get(best.number)
        if not best_ckpt:
            raise RuntimeError(
                f"Pas de checkpoint pour le trial {best.number}. "
                f"Checkpoints disponibles : {list(trial_checkpoints.keys())}"
            )

        logger.info(f"[HPO] Chargement checkpoint : {best_ckpt}")

        best_cfg = {**self.model_cfg, **best.params}
        model = self.model_factory(best_cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = type(model).load_from_checkpoint(
            best_ckpt,
            text_net=model.text_net,
            image_net=model.image_net,
            d_text=model.hparams.d_text,
            d_image=model.hparams.d_image,
            config=model.hparams.config,
        )
        model = model.to(device)
        logger.info(f"[HPO] Modèle chargé sur {device}")

        # ============================================================== #
        # 3. Eval gold + MLflow + promotion (run séparé)                   #
        # ============================================================== #

        logger.info("[HPO] Eval gold du best trial...")

        tags = self.config.setdefault("mlflow", {}).setdefault("tags", {})
        tags["hpo_best_trial"] = str(best.number)
        tags["hpo_best_val_f1"] = f"{best.value:.4f}"
        tags["hpo_n_trials"] = str(self.n_trials)
        tags["hpo_parent_run_id"] = parent_run_id
        tags["source"] = "hpo_checkpoint_reuse"
        for k, v in best.params.items():
            tags[f"hpo_{k}"] = str(v)

        with mlflow.start_run(
            run_name=f"M3_hpo_best_trial{best.number}"
        ) as eval_run:
            run_id = eval_run.info.run_id

            # Log config
            self._log_config_and_tags(best.params)

            # Eval gold
            y_pred, y_proba = self._predict_gold(model, gold_loader)
            f1_gold = self._log_gold_metrics(gold_labels, y_pred, y_proba)

            # Artifacts
            self._log_artifacts(gold_labels, y_pred)

            n_params = sum(p.numel() for p in model.fusion.parameters())
            mlflow.log_metric("model/n_trainable_params", n_params)
            mlflow.log_metric("fit_duration_s", time.time() - start_time)

            # Log model + promotion
            self._handle_promotion(model, f1_gold, run_id)

        total = time.time() - start_time
        logger.info(f"[HPO] Terminé en {total:.1f}s")

    # ================================================================== #
    # Composants internes                                                  #
    # ================================================================== #

    def _log_config_and_tags(self, best_params: dict) -> None:
        flat = {}
        for section, values in self.config.items():
            if section.startswith("_"):
                continue
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = v
            else:
                flat[section] = values
        for k, v in flat.items():
            mlflow.log_param(k, str(v)[:500])

        for k, v in best_params.items():
            mlflow.log_param(f"best/{k}", v)

        tags = self.config.get("mlflow", {}).get("tags", {})
        for k, v in tags.items():
            mlflow.set_tag(k, v)

    def _predict_gold(self, model, gold_loader):
        model.eval()
        device = next(model.fusion.parameters()).device

        all_logits = []
        with torch.no_grad():
            for batch in gold_loader:
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k != "label"
                }
                logits = model(batch)
                all_logits.append(logits.cpu())

        logits = torch.cat(all_logits, dim=0)
        proba = F.softmax(logits, dim=-1).numpy().astype(np.float32)
        preds = logits.argmax(dim=-1).numpy()
        return preds, proba

    def _log_gold_metrics(self, y_true, y_pred, y_proba) -> float:
        f1_w = f1_score(y_true, y_pred, average="weighted")
        f1_m = f1_score(y_true, y_pred, average="macro")
        acc = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_proba, labels=list(range(self.n_classes)))
        ece = expected_calibration_error(y_true, y_proba, n_bins=10)

        mlflow.log_metric("eval_gold/f1_weighted", f1_w)
        mlflow.log_metric("eval_gold/f1_macro", f1_m)
        mlflow.log_metric("eval_gold/accuracy", acc)
        mlflow.log_metric("eval_gold/log_loss", ll)
        mlflow.log_metric("eval_gold/ece", ece)

        logger.info(
            f"  eval_gold : f1_w={f1_w:.4f} f1_m={f1_m:.4f} "
            f"acc={acc:.4f} log_loss={ll:.4f} ece={ece:.4f}"
        )

        f1_per_class = f1_score(
            y_true, y_pred, labels=list(range(self.n_classes)),
            average=None, zero_division=0,
        )
        mlflow.log_dict(
            {"n_classes": self.n_classes, "f1_per_class": f1_per_class.tolist()},
            "eval_gold_f1_per_class.json",
        )
        return f1_w

    def _log_artifacts(self, y_true, y_pred) -> None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues", ax=ax,
            xticklabels=range(cm.shape[0]),
            yticklabels=range(cm.shape[0]),
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix — M3 HPO best trial")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report = classification_report(y_true, y_pred, digits=3)
        mlflow.log_text(report, "classification_report.txt")

    def _handle_promotion(self, model, f1_gold, run_id) -> None:
        client = MlflowClient(self.tracking_uri)

        mlflow.pytorch.log_model(
            model.fusion,
            artifact_path="model",
            registered_model_name=self.registry_model_name,
        )

        versions = client.search_model_versions(
            f"name='{self.registry_model_name}'"
        )
        latest_version = max(int(v.version) for v in versions)

        if not self.promotion_enabled:
            should_promote = False
            logger.info(
                "  promotion.enabled=false → modèle ENREGISTRÉ mais NON promu "
                "(décision déléguée à compare_and_promote)"
            )
        else:
            should_promote = True
            try:
                champion = client.get_model_version_by_alias(
                    self.registry_model_name, "champion"
                )
                champion_run = client.get_run(champion.run_id)
                f1_champion = champion_run.data.metrics.get(
                    "eval_gold/f1_weighted", 0.0
                )
                should_promote = (f1_gold - f1_champion) > self.promotion_threshold
                logger.info(
                    f"  F1 gold: {f1_gold:.4f} vs champion: {f1_champion:.4f} "
                    f"(delta={f1_gold - f1_champion:.4f})"
                )
            except Exception:
                logger.info("  Pas de @champion → promotion automatique")
        mlflow.log_param("promote_to_champion", should_promote)

        tags = self.config.get("mlflow", {}).get("tags", {})
        for key in ("base_text", "base_text_version",
                     "base_image", "base_image_version"):
            value = tags.get(key)
            if value is not None:
                client.set_model_version_tag(
                    self.registry_model_name, str(latest_version),
                    key, str(value),
                )

        if should_promote:
            client.set_registered_model_alias(
                self.registry_model_name, "champion", latest_version
            )
            logger.info(
                f"  Promu @champion : v{latest_version} (f1={f1_gold:.4f})"
            )
        else:
            logger.info(f"  Non promu (gain < {self.promotion_threshold})")
