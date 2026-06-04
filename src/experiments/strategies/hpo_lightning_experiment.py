"""
HPOLightningExperiment — strategy HPO Optuna pour modèles Lightning.

Complète le strategy pattern :
    - SklearnExperiment : fit sklearn + eval gold
    - BaseLearnerExperiment : fit base learner + @active
    - LightningExperiment : fit Lightning + eval gold + @champion
    - HPOLightningExperiment : Optuna HPO (nested runs) → LightningExperiment

Structure MLflow :
    Parent run "M3_HPO" :
        - Nested child runs pour chaque trial (metrics par epoch)
        - Artifact hpo_summary.json (tous les trials)
    Run séparé "M3_hpo_best_trialN" :
        - Retrain final avec best HP
        - Métriques complètes par epoch
        - Eval gold + promotion @champion

Les base learners frozen sont capturés dans model_factory (closure).
"""
from __future__ import annotations

import logging

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from src.experiments.strategies.lightning_experiment import LightningExperiment


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
    HPO Optuna avec nested MLflow runs + retrain final via LightningExperiment.

    Args:
        model_factory: callable(config_dict) → LightningModule.
        dm: DataModule configuré (set_m3_preprocessing appelé).
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
        self.experiment_name = mlflow_cfg.get("experiment_name", "M3_attention_fusion")

    def fit(self) -> None:
        """
        1. Parent run MLflow avec nested trials
        2. Retrain final (run séparé) via LightningExperiment
        """
        logger.info("[HPO] START")

        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        # ============================================================== #
        # Parent run — contient les nested trials                          #
        # ============================================================== #
        with mlflow.start_run(run_name="M3_HPO") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.log_param("hpo/n_trials", self.n_trials)
            mlflow.log_param("hpo/study_name", self.study_name)
            for k, v in self.search_space.items():
                mlflow.log_param(f"hpo/space/{k}", str(v))

            def objective(trial: optuna.Trial) -> float:
                hp = _suggest_hp_from_yaml(trial, self.search_space)

                # Guard : d_model % n_heads == 0
                d_model = hp.get("d_model", self.model_cfg.get("d_model", 512))
                n_heads = self.model_cfg.get("n_heads", 8)
                if d_model % n_heads != 0:
                    raise optuna.TrialPruned(
                        f"d_model={d_model} % n_heads={n_heads} != 0"
                    )

                trial_cfg = {**self.model_cfg, **hp}
                model = self.model_factory(trial_cfg)

                n_params = sum(p.numel() for p in model.fusion.parameters())

                # --- Nested child run pour ce trial ---
                with mlflow.start_run(
                    run_name=f"trial_{trial.number}", nested=True
                ) as child_run:
                    # Log les HP du trial
                    for k, v in hp.items():
                        mlflow.log_param(k, v)
                    mlflow.log_param("n_trainable_params", n_params)

                    mlf_logger = MLFlowLogger(
                        experiment_name=self.experiment_name,
                        tracking_uri=self.tracking_uri,
                        run_id=child_run.info.run_id,
                    )

                    pruning_cb = PyTorchLightningPruningCallback(
                        trial, monitor="val/f1_weighted"
                    )
                    callbacks = [
                        L.pytorch.callbacks.EarlyStopping(
                            monitor="val/f1_weighted",
                            patience=self.trainer_cfg.get("patience", 3),
                            mode="max",
                        ),
                        pruning_cb,
                    ]

                    trainer = L.Trainer(
                        max_epochs=self.trainer_cfg.get("max_epochs", 15),
                        accelerator=self.trainer_cfg.get("accelerator", "auto"),
                        precision=self.trainer_cfg.get("precision", "16-mixed"),
                        callbacks=callbacks,
                        logger=mlf_logger,
                        enable_progress_bar=False,
                        enable_checkpointing=False,
                        log_every_n_steps=10,
                    )
                    trainer.fit(model, train_loader, val_loader)

                    val_f1 = trainer.callback_metrics.get("val/f1_weighted", 0.0)
                    if hasattr(val_f1, "item"):
                        val_f1 = val_f1.item()

                    mlflow.log_metric("final_val_f1", val_f1)

                logger.info(
                    f"[HPO] Trial {trial.number} — {hp} — "
                    f"{n_params:,} params → val/f1 = {val_f1:.4f}"
                )
                return val_f1

            # --- Optuna study ---
            study = optuna.create_study(
                study_name=self.study_name,
                direction="maximize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=3,
                    n_warmup_steps=5,
                ),
            )
            study.optimize(objective, n_trials=self.n_trials)

            # --- Log summary sur le parent run ---
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
        # Résultats                                                        #
        # ============================================================== #
        print(f"\n{'='*60}")
        print(f"[HPO] BEST TRIAL #{best.number}")
        print(f"[HPO]   val/f1 = {best.value:.4f}")
        print(f"[HPO]   params = {best.params}")
        print(f"{'='*60}")

        # ============================================================== #
        # Retrain final — run séparé via LightningExperiment               #
        # ============================================================== #
        logger.info("[HPO] Retrain final avec best HP (run séparé)...")

        for k, v in best.params.items():
            self.config["model"][k] = v

        tags = self.config.setdefault("mlflow", {}).setdefault("tags", {})
        self.config["mlflow"]["run_name"] = f"M3_hpo_best_trial{best.number}"
        tags["hpo_best_trial"] = str(best.number)
        tags["hpo_best_val_f1"] = f"{best.value:.4f}"
        tags["hpo_n_trials"] = str(self.n_trials)
        tags["hpo_parent_run_id"] = parent_run_id
        for k, v in best.params.items():
            tags[f"hpo_{k}"] = str(v)

        best_cfg = {**self.model_cfg, **best.params}
        best_model = self.model_factory(best_cfg)

        final_experiment = LightningExperiment(
            model=best_model,
            dm=self.dm,
            config=self.config,
        )
        final_experiment.fit()

        logger.info("[HPO] Terminé.")
