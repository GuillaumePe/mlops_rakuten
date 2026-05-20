"""
M.4 — BaseLearnerExperiment : stratégie d'orchestration complète.

Résume le cycle de vie d'un base learner (TextCNN, ResNet50PartialFT, etc.) :
  1. Setup DataModule + résolution ACTIVE_VAL_SELECTION_VERSION
  2. Fit du learner sur train (val interne pour early stopping)
  3. Eval sur val_selection → métrique d'arbitrage @active
  4. Extract embeddings sur _df_full, write cache parquet, DVC push
  5. Log model + tag modality + promotion @active conditionnelle
  6. Cascade @active_text / @active_image via refresh_modality_alias

Usage typique (depuis runner.py M.5) :
    experiment = BaseLearnerExperiment(
        learner_name="textcnn",
        config_dict={...},
    )
    experiment.fit(datamodule)

Décomposition responsabilités :
  - BaseLearner ABC : contrat du modèle (fit, extract_embeddings, predict_proba)
  - BaseLearnerExperiment : orchestration expérience (data, train, eval, log, promote)
    (Strategy pattern : composition, pas héritage)
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import polars as pl
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from slugify import slugify

if TYPE_CHECKING:
    from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
    from src.models.base_learners._base import BaseLearner

from src.models.utils import (
    compute_promotion_decision,
    get_active_val_selection_version,
    refresh_modality_alias,
)


logger = logging.getLogger(__name__)


class BaseLearnerExperiment:
    """
    Orchestration complète d'un cycle d'entraînement de base learner.

    Cette classe consumer un BaseLearner (via composition, pas héritage) et
    gère tout ce qui entoure : DataModule, données, évaluation, MLflow,
    promotion alias, cache parquet, DVC.

    Responsabilités découpées :
    - BaseLearner : apprendre à encoder (fit, extract_embeddings)
    - BaseLearnerExperiment : orchestre l'expérimentation complet
    """

    def __init__(
        self,
        learner_name: str,
        config: dict,
        tracking_uri: str = "http://mlflow:5000",
        experiment_name: str = "base_learners_phase1",
        data_folder: Optional[Path] = None,
        cache_output_dir: Optional[Path] = None,
    ):
        """
        Args:
            learner_name: identifiant du base learner à entraîner
                (ex: "textcnn", "resnet50_partial_ft").
            config: dict de configuration du learner (ex: hyperparams,
                learning rate, batch size, etc.). Sera loggé en MLflow.
            tracking_uri: URI du serveur MLflow (default http://mlflow:5000).
            experiment_name: nom de l'expérience MLflow.
            data_folder: racine des données (images, parquets). Par défaut,
                déduit depuis env ou ../data/raw_data/.
            cache_output_dir: dossier où écrire les caches parquets. Par défaut,
                ./mlruns_cache.
        """
        self.learner_name = learner_name
        self.config = config
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        self.data_folder = data_folder or Path("data/raw_data")
        self.cache_output_dir = cache_output_dir or Path("mlruns_cache")
        self.cache_output_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self._learner: Optional[BaseLearner] = None
        self._run_id: Optional[str] = None

    def _build_learner(self) -> BaseLearner:
        """
        Instancie le BaseLearner selon learner_name.

        Factory pluggable (utile pour les tests ou l'extension).
        """
        from src.models.base_learners.text.textcnn import TextCNN
        from src.models.base_learners.image.resnet50_partial_ft import ResNet50PartialFT

        learner_builders = {
            "textcnn": lambda cfg: TextCNN(
                vocab_size=cfg.get("vocab_size", 50000),
                embed_dim=cfg.get("embed_dim", 300),
                n_filters=cfg.get("n_filters", 512),
                kernel_sizes=tuple(cfg.get("kernel_sizes", [1, 2, 3, 4, 5, 6])),
                n_classes=27,
                dropout=cfg.get("dropout", 0.5),
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 0.0),
            ),
            "resnet50_partial_ft": lambda cfg: ResNet50PartialFT(
                image_folder=str(self.data_folder / "images" / "image_train"),
                n_classes=27,
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 15),
                patience=cfg.get("patience", 3),
                lr_head=cfg.get("lr_head", 1e-3),
                lr_backbone=cfg.get("lr_backbone", 1e-5),
                weight_decay=cfg.get("weight_decay", 1e-4),
                num_workers=cfg.get("num_workers", 4),
                random_state=cfg.get("random_state", 42),
                precision=cfg.get("precision", "bf16-mixed"),
            ),
        }

        if self.learner_name not in learner_builders:
            raise ValueError(
                f"learner_name={self.learner_name!r} inconnu. "
                f"Disponibles : {list(learner_builders.keys())}"
            )

        return learner_builders[self.learner_name](self.config)

    def fit(self, datamodule: "RakutenLightningDataModule") -> None:
        """
        Orchestre le cycle complet d'entraînement.

        Étapes :
        1. Setup DataModule
        2. Récup splits train/val
        3. Instancie + fit du learner
        4. Éval sur val_selection → métrique @active
        5. Extract embeddings + write cache parquet
        6. Log MLflow + tag modality
        7. Promo @active conditionnelle
        8. Cascade @active_text / @active_image

        Args:
            datamodule: RakutenLightningDataModule configuré (pas encore setupé).
        """
        logger.info(f"[BaseLearnerExperiment] Démarrage fit pour {self.learner_name}")
        start_time = time.time()

        # 1. Setup DataModule
        logger.info("[BaseLearnerExperiment.1] Setup DataModule")
        datamodule.setup()
        n_val_selection = get_active_val_selection_version()
        logger.info(f"  ACTIVE_VAL_SELECTION_VERSION={n_val_selection}")

        # 2. Récup splits train/val (standard 80/20 sur train_pool_effective)
        logger.info("[BaseLearnerExperiment.2] Récup splits")
        X_train, y_train = datamodule.get_sklearn_data(
            "train", include_raw=True
        )
        X_val, y_val = datamodule.get_sklearn_data("val", include_raw=True)
        logger.info(f"  train: {len(X_train)}, val: {len(X_val)}")

        # 3. Instancie + fit learner
        logger.info("[BaseLearnerExperiment.3] Instancie + fit learner")
        self._learner = self._build_learner()
        logger.info(f"  Learner : {self._learner}")
        with mlflow.start_run() as run:
            self._run_id = run.info.run_id
            logger.info(f"  MLflow run_id={self._run_id}")

            # Log config en MLflow
            mlflow.log_params(self.config)
            mlflow.set_tag("learner_name", self.learner_name)
            mlflow.set_tag("modality", self._learner.modality)

            # Fit
            fit_start = time.time()
            self._learner.fit(X_train, y_train, X_val, y_val)
            fit_duration_s = time.time() - fit_start
            logger.info(f"  Fit terminé en {fit_duration_s:.1f}s")
            mlflow.log_metric("fit_duration_s", fit_duration_s)

            # 4. Éval sur val_selection (arbitre @active)
            logger.info("[BaseLearnerExperiment.4] Éval sur val_selection_v{n}")
            X_vs, y_vs = datamodule.get_sklearn_data(
                "val_selection", include_raw=True
            )
            if len(X_vs) == 0:
                logger.error(
                    f"val_selection vide ! Vérifier is_val_selection_v{n_val_selection}"
                )
                raise RuntimeError(
                    f"val_selection_v{n_val_selection} est vide. "
                    f"Lancer src/data/init_val_selection.py --version {n_val_selection}"
                )

            y_pred_proba = self._learner.predict_proba(X_vs)
            y_pred = y_pred_proba.argmax(axis=1)
            f1_vs = f1_score(y_vs, y_pred, average="weighted")
            metric_key = f"val_selection_v{n_val_selection}/f1_weighted"
            mlflow.log_metric(metric_key, f1_vs)
            logger.info(f"  {metric_key}={f1_vs:.4f}")

            # 5. Extract embeddings sur _df_full, write cache parquet
            logger.info("[BaseLearnerExperiment.5] Extract embeddings + cache parquet")
            self._write_cache_parquet(datamodule)

            # 6. Log model MLflow + tag modality
            logger.info("[BaseLearnerExperiment.6] Log model MLflow + tag")
            registered_model_name = f"rakuten-base-{self.learner_name}"
            mlflow.sklearn.log_model(
                self._learner,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
            client = MlflowClient(self.tracking_uri)
            client.set_registered_model_tag(
                registered_model_name,
                key="modality",
                value=self._learner.modality,
            )
            logger.info(f"  Logged : {registered_model_name}")

            # 7. Promo @active conditionnelle
            logger.info("[BaseLearnerExperiment.7] Promo @active conditionnelle")
            threshold = self.config.get("promotion_threshold", 0.005)
            should_promote = compute_promotion_decision(
                registered_model_name, self._run_id, threshold=threshold
            )
            mlflow.log_param("promote_to_active", should_promote)
            if should_promote:
                # Récup la version qui vient de être logée
                versions = client.search_model_versions(
                    f"name='{registered_model_name}'"
                )
                latest_version = max(int(v.version) for v in versions)
                client.set_registered_model_alias(
                    registered_model_name, "active", latest_version
                )
                logger.info(f"  Promu @active : v{latest_version}")

                # 8. Cascade @active_text / @active_image
                logger.info(
                    "[BaseLearnerExperiment.8] Cascade @active_text/image"
                )
                refresh_modality_alias(self._learner.modality)
            else:
                logger.info(f"  Pas promu (delta < {threshold})")

            total_duration_s = time.time() - start_time
            mlflow.log_metric("total_duration_s", total_duration_s)

        logger.info(
            f"[BaseLearnerExperiment] Fit terminé en {total_duration_s:.1f}s"
        )

    def _write_cache_parquet(self, datamodule: "RakutenLightningDataModule") -> None:
        """
        Extract embeddings sur _df_full, construit un cache parquet
        avec les colonnes métadonnées et features, puis DVC push.

        Cache layout :
            productid, batch_id, is_gold, is_val_selection_v1..v_max,
            source_model_name, source_model_version,
            {learner_name}_feat_0, {learner_name}_feat_1, ..., {learner_name}_feat_{embed_dim-1}
        """
        logger.info("[_write_cache_parquet] Début extract embeddings")

        # Récup _df_full (toutes les données : train+gold+future)
        df_full = datamodule._df_full
        if df_full is None:
            raise RuntimeError(
                "_df_full not found. DataModule.setup() doit avoir charge parquet."
            )

        # Extract embeddings
        X_full, _ = datamodule.get_sklearn_data(
            "train_pool", include_raw=True
        )  # Représentative, on prend train_pool
        # mais en réalité il faut _df_full complet
        # FIXME : ajouter une méthode get_full_data() au DataModule pour récup TOUT
        # Pour MVP M.4, on fait avec train_pool (limitation connue)

        logger.info(f"  Extract embeddings sur {len(X_full)} samples")
        embeddings = self._learner.extract_embeddings(X_full)
        logger.info(f"  Shape embeddings : {embeddings.shape}")

        # Construire le cache DataFrame
        cache_data = {
            "productid": X_full["productid"].to_numpy(),
            "batch_id": X_full.get_column("batch_id").to_numpy()
            if "batch_id" in X_full.columns
            else np.full(len(X_full), 1, dtype=int),
            "source_model_name": self.learner_name,
            "source_model_version": 1,  # FIXME : récup depuis MLflow
        }

        # Ajouter colonnes is_val_selection_v1..v_max (depuis df_full)
        for col in df_full.columns:
            if col.startswith("is_val_selection_v"):
                cache_data[col] = df_full[col].to_numpy()

        # Ajouter colonnes is_gold (important pour le garde-fou)
        if "is_gold" in df_full.columns:
            cache_data["is_gold"] = df_full["is_gold"].to_numpy()

        # Ajouter embeddings
        for i in range(embeddings.shape[1]):
            cache_data[f"{self.learner_name}_feat_{i}"] = embeddings[:, i]

        # Construire le DataFrame cache
        cache_df = pd.DataFrame(cache_data)
        logger.info(f"  Cache shape : {cache_df.shape}")

        # Write parquet
        cache_filename = (
            f"embeddings_{slugify(self.learner_name)}_v{get_active_val_selection_version()}.parquet"
        )
        cache_path = self.cache_output_dir / cache_filename
        cache_df.to_parquet(cache_path, index=False)
        logger.info(f"  Écrit cache : {cache_path}")

        # DVC push (optionnel, warning si pas dispo)
        try:
            subprocess.run(
                ["dvc", "add", str(cache_path)],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["dvc", "push"],
                cwd=Path.cwd(),
                check=True,
                capture_output=True,
                timeout=60,
            )
            logger.info(f"  DVC push OK")
        except Exception as e:
            logger.warning(f"  DVC push échoué (optionnel) : {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper pour usage dans runner.py
# ─────────────────────────────────────────────────────────────────────────────


def build_base_learner_experiment(
    learner_name: str,
    config: dict,
    **kwargs,
) -> BaseLearnerExperiment:
    """
    Factory pour instancier une expérience base learner.

    Usage (M.5 runner.py) :
        config = yaml.safe_load(open("config/base_learner_textcnn.yaml"))
        exp = build_base_learner_experiment("textcnn", config)
        exp.fit(datamodule)
    """
    return BaseLearnerExperiment(
        learner_name=learner_name,
        config=config,
        **kwargs,
    )
