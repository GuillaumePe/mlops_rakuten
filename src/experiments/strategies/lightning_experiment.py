"""
O.4 — LightningExperiment : strategy d'orchestration pour les modèles Lightning.
 
Complète le strategy pattern :
    - SklearnExperiment : modèles tabulaires (M2Baseline, M2Assembled)
    - BaseLearnerExperiment : extracteurs de features (TextCNN, ResNet, CamemBERT)
    - LightningExperiment : assemblés deep (M3 cross-attention, futurs M4+)
 
Responsabilités :
    1. MLflow run lifecycle (start_run, log params/tags, log model)
    2. Lightning Trainer (fit avec EarlyStopping, MLFlowLogger)
    3. Évaluation sur gold → métriques standard (f1, accuracy, ECE, etc.)
    4. Promotion @champion conditionnelle
    5. Artifacts (confusion matrix, classification report, F1 per class)
 
Ce qui est DEHORS :
    - Construction du modèle (fait par le builder dans runner.py)
    - Chargement des base learners (fait par le builder dans runner.py)
    - Preprocessing M3 (configuré sur le DataModule par le builder)
    - dm.setup() (fait dans cmd_fit_lightning, comme les autres cmd)
 
Usage (depuis runner.py) :
    # Builder
    dm = RakutenLightningDataModule(...)
    dm.set_m3_preprocessing(tokenizer, max_len, image_transform)
    model = M3AttentionFusion(text_net, image_net, d_text, d_image, config)
    experiment = LightningExperiment(model=model, dm=dm, config=config)
 
    # cmd
    dm.setup()
    experiment.fit()
"""
from __future__ import annotations
 
import logging
import time
from typing import Optional
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
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
 
 
class LightningExperiment:
    """
    Orchestration complète d'un cycle d'entraînement Lightning + MLflow.
 
    Args:
        model: LightningModule (M3AttentionFusion ou tout futur modèle).
        dm: DataModule déjà configuré (set_m3_preprocessing appelé).
            setup() sera appelé par cmd_fit_lightning AVANT fit().
        config: dict de configuration complète (sera loggé en MLflow).
            Clés attendues :
            - trainer.max_epochs (int, default 30)
            - trainer.patience (int, default 3)
            - trainer.precision (str, default "16-mixed")
            - trainer.accelerator (str, default "auto")
            - mlflow.tracking_uri (str, default "http://mlflow:5000")
            - mlflow.experiment_name (str)
            - mlflow.run_name (str)
            - mlflow.tags (dict)
            - promotion.registry_model_name (str)
            - promotion.threshold (float, default 0.005)
            - model.n_classes (int, default 27)
    """
 
    def __init__(
        self,
        model: L.LightningModule,
        dm,
        config: dict,
    ):

        self.model = model
        self.dm = dm
        self.config = config
 
        # --- Config shortcuts ---
        trainer_cfg = config.get("trainer", {})
        self.max_epochs = trainer_cfg.get("max_epochs", 30)
        self.patience = trainer_cfg.get("patience", 3)
        self.precision = trainer_cfg.get("precision", "16-mixed")
        self.accelerator = trainer_cfg.get("accelerator", "auto")
 
        mlflow_cfg = config.get("mlflow", {})
        self.tracking_uri = mlflow_cfg.get("tracking_uri", "http://mlflow:5000")
        self.experiment_name = mlflow_cfg.get("experiment_name", "m3_attention_fusion")
        self.run_name = mlflow_cfg.get("run_name", "m3_run")
        self.tags = mlflow_cfg.get("tags", {})
 
        promotion_cfg = config.get("promotion", {})
        self.registry_model_name = promotion_cfg.get(
            "registry_model_name", "rakuten-m3-attention-fusion"
        )
        self.promotion_threshold = promotion_cfg.get("threshold", 0.005)
 
        self.n_classes = config.get("model", {}).get("n_classes", 27)
 
    # ================================================================== #
    # FIT — séquence complète                                              #
    # ================================================================== #
 
    def fit(self) -> None:
        """
        Orchestre le cycle complet :
            1. Récupérer les DataLoaders depuis le DataModule
            2. MLflow run
            3. Trainer.fit (train + val avec early stopping)
            4. Eval gold → métriques
            5. Log model + artifacts
            6. Promotion @champion conditionnelle
 
        Prérequis : dm.setup() doit avoir été appelé avant (par cmd_fit_lightning).
        """

        logger.info("[LightningExperiment] Démarrage fit")
        start_time = time.time()

        # --- DataLoaders depuis le DataModule ---
        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        gold_loader = self.dm.gold_dataloader()
        gold_labels = self.dm.get_gold_labels()

 
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
 
        with mlflow.start_run(run_name=self.run_name) as run:
            run_id = run.info.run_id
            logger.info(f"  MLflow run_id={run_id}")
 
            # ========================================================== #
            # 1. Log config + tags                                         #
            # ========================================================== #
            self._log_config_and_tags()
 
            # ========================================================== #
            # 2. Trainer.fit                                               #
            # ========================================================== #
            logger.info("[LightningExperiment.2] Trainer.fit")
            trainer = self._build_trainer(run_id)
            trainer.fit(self.model, train_loader, val_loader)
            fit_duration = time.time() - start_time
            mlflow.log_metric("fit_duration_s", fit_duration)
            logger.info(f"  Fit terminé en {fit_duration:.1f}s")
 
            # Charger le meilleur checkpoint (early stopping)
            best_ckpt = trainer.checkpoint_callback.best_model_path
            if best_ckpt:
                logger.info(f"  Best checkpoint: {best_ckpt}")
                device = next(self.model.fusion.parameters()).device                
                self.model = type(self.model).load_from_checkpoint(
                    best_ckpt,
                    text_net=self.model.text_net,
                    image_net=self.model.image_net,
                    d_text=self.model.hparams.d_text,
                    d_image=self.model.hparams.d_image,
                    config=self.model.hparams.config,
                )
                self.model = self.model.to(device)
 
            # ========================================================== #
            # 3. Eval gold                                                 #
            # ========================================================== #
            logger.info("[LightningExperiment.3] Eval gold")
            y_pred, y_proba = self._predict_gold(gold_loader)
            self._log_gold_metrics(gold_labels, y_pred, y_proba)
 
            # ========================================================== #
            # 4. Log model + artifacts                                     #
            # ========================================================== #
            logger.info("[LightningExperiment.4] Log model + artifacts")
            self._log_artifacts(gold_labels, y_pred)
            n_params = sum(
                p.numel() for p in self.model.fusion.parameters()
            )
            mlflow.log_metric("model/n_trainable_params", n_params)
 
            # ========================================================== #
            # 5. Promotion @champion                                       #
            # ========================================================== #
            logger.info("[LightningExperiment.5] Promotion @champion")
            f1_gold = f1_score(gold_labels, y_pred, average="weighted")
            self._handle_promotion(f1_gold, run_id)
 
        total_duration = time.time() - start_time
        logger.info(
            f"[LightningExperiment] Terminé en {total_duration:.1f}s"
        )
 
    # ================================================================== #
    # Composants internes                                                  #
    # ================================================================== #
 
    def _log_config_and_tags(self) -> None:
        """Log la config (flat) et les tags en MLflow."""
        # Flat config pour MLflow params (pas de nested dicts)
        flat = {}
        for section, values in self.config.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = v
            else:
                flat[section] = values
        # MLflow limite à 500 chars par valeur
        for k, v in flat.items():
            mlflow.log_param(k, str(v)[:500])
 
        for k, v in self.tags.items():
            mlflow.set_tag(k, v)
 
    def _build_trainer(self, run_id: str) -> L.Trainer:
        """Construit le Lightning Trainer avec callbacks standard."""
        mlf_logger = MLFlowLogger(
            experiment_name=self.experiment_name,
            tracking_uri=self.tracking_uri,
            run_id=run_id,
        )
 
        callbacks = [
            EarlyStopping(
                monitor="val/loss",
                patience=self.patience,
                mode="min",
                verbose=True,
            ),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                filename="best-{epoch}-{val/loss:.4f}",
            ),
        ]
 
        return L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            precision=self.precision,
            callbacks=callbacks,
            logger=mlf_logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
 
    def _predict_gold(self, gold_loader) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward sur le gold set → (predictions, probabilities).
 
        Itère manuellement sur le gold_loader au lieu d'utiliser
        Trainer.predict() pour éviter l'overhead (DDP, etc.) et
        garder le contrôle sur le device.
        """
        self.model.eval()
        device = next(self.model.fusion.parameters()).device
 
        all_logits = []
        with torch.no_grad():
            for batch in gold_loader:
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k != "label"
                }
                logits = self.model(batch)
                all_logits.append(logits.cpu())
 
        logits = torch.cat(all_logits, dim=0)
        proba = F.softmax(logits, dim=-1).numpy().astype(np.float32)
        preds = logits.argmax(dim=-1).numpy()
 
        return preds, proba
 
    def _log_gold_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> None:
        """Log les métriques standard sur le gold set."""
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
 
        # F1 par classe (artifact JSON, pas en metric pour ne pas
        # polluer l'UI MLflow avec 27 lignes)
        f1_per_class = f1_score(
            y_true, y_pred, labels=list(range(self.n_classes)),
            average=None, zero_division=0,
        )
        mlflow.log_dict(
            {
                "n_classes": self.n_classes,
                "f1_per_class": f1_per_class.tolist(),
            },
            "eval_gold_f1_per_class.json",
        )
 
    def _log_artifacts(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """Confusion matrix + classification report."""
        # --- Confusion matrix ---
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm, annot=False, fmt="d", cmap="Blues", ax=ax,
            xticklabels=range(cm.shape[0]),
            yticklabels=range(cm.shape[0]),
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {self.run_name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
 
        # --- Classification report ---
        report = classification_report(y_true, y_pred, digits=3)
        mlflow.log_text(report, "classification_report.txt")
 
    def _handle_promotion(self, f1_gold: float, run_id: str) -> None:
        """
        Promotion conditionnelle @champion.
 
        Compare le F1 gold du run courant avec le @champion actuel.
        Promeut si gain > threshold (défaut 0.005).
        """
        client = MlflowClient(self.tracking_uri)
 
        # Enregistre le modèle dans le registry
        mlflow.pytorch.log_model(
            self.model.fusion,  # On log uniquement le module de fusion
            artifact_path="model",
            registered_model_name=self.registry_model_name,
        )
 
        # Récupère la version qui vient d'être loggée
        versions = client.search_model_versions(
            f"name='{self.registry_model_name}'"
        )
        latest_version = max(int(v.version) for v in versions)
 
        # Décision de promotion
        should_promote = True  # défaut si pas de @champion existant
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
                f"(Δ={f1_gold - f1_champion:.4f}, threshold={self.promotion_threshold})"
            )
        except Exception:
            logger.info("  Pas de @champion existant → promotion automatique")
 
        mlflow.log_param("promote_to_champion", should_promote)
 
        # Tag le registered model avec les refs des base learners
        # pour reconstruction à l'inférence sans chercher le run
        for key in ("base_text", "base_text_version",
                     "base_image", "base_image_version"):
            value = self.tags.get(key)
            if value is not None:
                client.set_model_version_tag(
                    self.registry_model_name, str(latest_version), key, str(value)
                )
 
        if should_promote:
            client.set_registered_model_alias(
                self.registry_model_name, "champion", latest_version
            )
            logger.info(
                f"  ✓ Promu @champion : v{latest_version} "
                f"(f1_gold={f1_gold:.4f})"
            )
        else:
            logger.info(
                f"  Non promu (gain < {self.promotion_threshold})"
            )
