"""
M.4 + M.4bis — BaseLearnerExperiment : stratégie d'orchestration complète.

Résume le cycle de vie d'un base learner (TextCNN, ResNet50PartialFT, etc.) :
  1. Setup DataModule + résolution ACTIVE_VAL_SELECTION_VERSION
  2. Fit du learner sur train (val interne pour early stopping)
  3. Eval sur val_selection → métrique d'arbitrage @active
  4. Log model PyFunc + tag modality + récup version MLflow
  5. Décision promotion @active conditionnelle
  6. Si promu : (a) set alias @active (b) cascade @active_text/image
                (c) extract embeddings + write cache parquet + DVC push

INVARIANT M.4bis (séquençage critique) :
  Le cache parquet est écrit UNIQUEMENT après promotion @active réussie.
  Cela garantit que `embeddings_{name}_v{N}.parquet` contient toujours les
  embeddings du modèle pointé par `rakuten-base-{name} @active`. Cet invariant
  est vérifié par le guard-fou M.7 dans DataModule._load_base_learner_embeddings
  qui compare `source_model_version` (dans le parquet) à `@active.version`
  (en MLflow). Sans ce séquençage, un run non-promu écraserait le cache avec
  des embeddings désynchronisés → RuntimeError au prochain `mode=m2_benchmark`.

Persistence PyFunc (M.4bis) :
  Au lieu de mlflow.sklearn.log_model (qui ne fonctionne pas pour les
  BaseLearner deep contenant des nn.Module + état Python comme le vocab),
  on utilise mlflow.pyfunc.log_model avec :
    - BaseLearner.save_pretrained() → sauve state_dict + métadonnées
    - BaseLearnerPyfunc → wrapper générique qui reconstruit via from_pretrained
  Cf. src/models/base_learners/_pyfunc_wrapper.py pour le wrapper.

Décomposition responsabilités :
  - BaseLearner ABC : contrat du modèle (fit, extract_embeddings, predict_proba,
                      save_pretrained, from_pretrained)
  - BaseLearnerExperiment : orchestration expérience (data, train, eval, log,
                            promote, cache parquet, DVC)
  (Strategy pattern : composition, pas héritage)

Usage typique (depuis runner.py M.5) :
    experiment = BaseLearnerExperiment(
        learner_name="textcnn",
        config={...},
        tracking_uri=...,
        cache_output_dir=Path(os.getenv("DATA_ROOT", ".")) / "data/cache",
    )
    experiment.fit(datamodule)
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import polars as pl
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, log_loss
from lightning.pytorch.loggers import MLFlowLogger
from src.models.base_learners._pyfunc_wrapper import BaseLearnerPyfunc
from src.experiments.strategies._metrics import expected_calibration_error
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

    Cette classe consume un BaseLearner (via composition, pas héritage) et
    gère tout ce qui entoure : DataModule, données, évaluation, MLflow,
    promotion alias, cache parquet, DVC.

    Responsabilités découpées :
    - BaseLearner : apprendre à encoder (fit, extract_embeddings) + se
                    sérialiser/reconstruire (save_pretrained, from_pretrained)
    - BaseLearnerExperiment : orchestre l'expérimentation complète
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
            config: dict de configuration du learner (hyperparams,
                learning rate, batch size, etc.). Sera loggé en MLflow.
            tracking_uri: URI du serveur MLflow (default http://mlflow:5000).
                Le runner override avec l'env var MLFLOW_TRACKING_URI pour les
                pods cloud (Tailscale IP).
            experiment_name: nom de l'expérience MLflow.
            data_folder: racine des données (images, parquets). Par défaut,
                "data/raw_data".
            cache_output_dir: dossier où écrire les caches parquets. Par défaut,
                "mlruns_cache" (à OVERRIDER côté runner avec un chemin sur le
                volume persistant cloud, sinon le cache est perdu au cleanup
                du pod). Recommandé :
                    Path(os.getenv("DATA_ROOT", ".")) / "data/cache"
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

        self._learner: Optional["BaseLearner"] = None
        self._run_id: Optional[str] = None

    def _build_learner(self) -> "BaseLearner":
        """
        Instancie le BaseLearner selon learner_name.

        Factory pluggable (utile pour les tests ou l'extension).
        """
        from src.models.base_learners.text.textcnn import TextCNN
        from src.models.base_learners.text.camembert_lora import CamembertLoRA
        from src.models.base_learners.image.resnet50_partial_ft import ResNet50PartialFT
        from src.models.base_learners.image.resnet18_full_ft import ResNet18FullFT

        learner_builders = {
            "textcnn": lambda cfg: TextCNN(
                vocab_size=cfg.get("vocab_size", 50000),
                embed_dim=cfg.get("embed_dim", 300),
                n_filters=cfg.get("n_filters", 512),
                kernel_sizes=tuple(cfg.get("kernel_sizes", [1, 2, 3, 4, 5, 6])),
                n_classes=27,
                dropout=cfg.get("dropout", 0.5),
                lr=float(cfg.get("lr", 1e-3)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                patience=cfg.get("patience", 3),
            ),
            "resnet50_partial_ft": lambda cfg: ResNet50PartialFT(
                image_folder=str(self.data_folder / "images" / "image_train"),
                n_classes=27,
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 15),
                patience=cfg.get("patience", 3),
                lr_head=float(cfg.get("lr_head", 1e-3)),
                lr_backbone=float(cfg.get("lr_backbone", 1e-5)),
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
                num_workers=cfg.get("num_workers", 4),
                random_state=cfg.get("random_state", 42),
                precision=cfg.get("precision", "bf16-mixed"),
                augmentation_level=cfg.get("augmentation_level", "soft"),
            ),
            "camembert_lora": lambda cfg: CamembertLoRA(
                model_name=cfg.get("model_name", "camembert-base"),
                n_classes=cfg.get("n_classes", 27),
                max_len=cfg.get("max_len", 128),
                lora_rank=cfg.get("lora_rank", 16),
                lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05),
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 10),
                patience=cfg.get("patience", 2),
                lr_lora=float(cfg.get("lr_lora", 5e-4)),
                lr_head=float(cfg.get("lr_head", 1e-3)),
                weight_decay=float(cfg.get("weight_decay", 0.01)),
                warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
                num_workers=cfg.get("num_workers", 2),
                random_state=cfg.get("random_state", 42),
                precision=cfg.get("precision", "bf16-mixed"),
            ),
            "resnet18_full_ft": lambda cfg: ResNet18FullFT(
                image_folder=str(self.data_folder / "images" / "image_train"),
                n_classes=cfg.get("n_classes", 27),
                batch_size=cfg.get("batch_size", 32),
                max_epochs=cfg.get("max_epochs", 15),
                patience=cfg.get("patience", 3),
                lr_head=float(cfg.get("lr_head", 1e-3)),
                lr_backbone=float(cfg.get("lr_backbone", 1e-5)),
                weight_decay=float(cfg.get("weight_decay", 1e-2)),
                num_workers=cfg.get("num_workers", 4),
                random_state=cfg.get("random_state", 42),
                precision=cfg.get("precision", "bf16-mixed"),
                augmentation_level=cfg.get("augmentation_level", "medium"),
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

        Séquence M.4bis (invariant @active ↔ parquet) :
            1. Setup DataModule (skip si déjà setupé)
            2. Récup splits train/val standard
            3. Fit du learner avec MLflow run open
            4. Eval sur val_selection (arbitre @active)
            5. Log PyFunc model + tag modality + récup version
            6. Décision promotion (compute_promotion_decision)
            7. Si promu :
               a. Set alias @active
               b. Cascade @active_text/@active_image
               c. Extract embeddings + write cache parquet + DVC push
               Sinon : log skip, parquet conservé tel quel (correspond à
                       @active actuel)

        Args:
            datamodule: RakutenLightningDataModule configuré.
        """
        logger.info(f"[BaseLearnerExperiment] Démarrage fit pour {self.learner_name}")
        start_time = time.time()

        # ============================================================== #
        # 1. Setup DataModule (idempotent)                                 #
        # ============================================================== #
        logger.info("[BaseLearnerExperiment.1] Setup DataModule")
        if getattr(datamodule, "_df_full", None) is None:
            datamodule.setup()
        else:
            logger.info("  DataModule déjà setupé, skip")
        n_val_selection = get_active_val_selection_version()
        logger.info(f"  ACTIVE_VAL_SELECTION_VERSION={n_val_selection}")

        # ============================================================== #
        # 2. Récup splits train/val (standard 80/20 sur train_pool_effective)
        # ============================================================== #
        logger.info("[BaseLearnerExperiment.2] Récup splits train/val")
        X_train, y_train = datamodule.get_sklearn_data("train", include_raw=True)
        X_val, y_val = datamodule.get_sklearn_data("val", include_raw=True)
        logger.info(f"  train: {len(X_train)}, val: {len(X_val)}")

        # ============================================================== #
        # 3. Instancie + fit learner (dans un run MLflow)                 #
        # ============================================================== #
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
            lightning_logger = MLFlowLogger(
				experiment_name=self.experiment_name,
                tracking_uri=self.tracking_uri,
                run_id=self._run_id,
            )
            logger.info(
                f"  MLFlowLogger initialisé, run_id partagé={self._run_id}"
            )

            self._learner.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                lightning_logger=lightning_logger,
            )
            fit_duration_s = time.time() - fit_start
            logger.info(f"  Fit terminé en {fit_duration_s:.1f}s")
            mlflow.log_metric("fit_duration_s", fit_duration_s)

            # ========================================================== #
            # 4. Eval sur val_selection (arbitre @active)                  #
            # ========================================================== #
            logger.info(
                f"[BaseLearnerExperiment.4] Eval sur val_selection_v{n_val_selection}"
            )
            X_vs, y_vs = datamodule.get_sklearn_data("val_selection", include_raw=True)
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
            # ---------------------------------------------------------- #
            # M.0d : métriques étendues sur val_selection                #
            # ---------------------------------------------------------- #
            # Métriques scalaires loggées en MLflow pour comparaison entre runs.
            # Justification statistique de chaque métrique :
            #
            # - f1_weighted : métrique principale Rakuten (déséquilibre 27 classes).
            #     Pondération par support de classe → respecte la distribution.
            # - f1_macro : F1 non pondéré, sensible aux classes minoritaires.
            #     Gap (f1_weighted - f1_macro) signale un biais sur les rares.
            # - accuracy : repère intuitif, mais trompeur sur multi-classe déséquilibré.
            # - log_loss : NLL = MLE catégorielle. Mesure la calibration + justesse.
            #     log_loss faible = probas bien étalonnées ET prédictions justes.
            # - ECE : Expected Calibration Error (Naeini et al. 2015).
            #     Mesure pure de calibration, indépendante de l'accuracy.
            #     ECE ≤ 0.02 = excellent, ≤ 0.05 = acceptable, > 0.1 = problème.
            #
            # Tous loggés avec préfixe `val_selection_v{n}/` pour respecter la
            # convention de namespacing déjà en place pour f1_weighted.
            
            n_classes = int(max(y_vs.max() + 1, y_pred_proba.shape[1]))
            f1_w = f1_score(y_vs, y_pred, average="weighted")
            f1_m = f1_score(y_vs, y_pred, average="macro")
            acc = accuracy_score(y_vs, y_pred)
            ll = log_loss(y_vs, y_pred_proba, labels=list(range(n_classes)))
            ece = expected_calibration_error(y_vs, y_pred_proba, n_bins=10)

            prefix = f"val_selection_v{n_val_selection}"
            mlflow.log_metric(f"{prefix}/f1_weighted", f1_w)
            mlflow.log_metric(f"{prefix}/f1_macro", f1_m)
            mlflow.log_metric(f"{prefix}/accuracy", acc)
            mlflow.log_metric(f"{prefix}/log_loss", ll)
            mlflow.log_metric(f"{prefix}/ece", ece)
            logger.info(
                f"  {prefix} : f1_w={f1_w:.4f} f1_m={f1_m:.4f} "
                f"acc={acc:.4f} log_loss={ll:.4f} ece={ece:.4f}"
            )

            # Artifact JSON : F1 par classe (27 valeurs).
            # Pas en metric (sinon 27 lignes MLflow par run, pollue l'UI).
            # Permet l'analyse fine "quelle classe a tiré le F1 vers le bas ?"
            # et la comparaison f1_per_class entre base learners (texte vs image).
            f1_per_class = f1_score(
                y_vs, y_pred, labels=list(range(n_classes)),
                average=None, zero_division=0,
            )
            mlflow.log_dict(
                {
                    "val_selection_version": n_val_selection,
                    "n_classes": n_classes,
                    "f1_per_class": f1_per_class.tolist(),
                },
                f"{prefix}_f1_per_class.json",
            )

            # Garde-fou : la métrique d'arbitrage @active reste f1_weighted.
            # On expose ici la variable pour compute_promotion_decision.
            f1_vs = f1_w


            # ========================================================== #
            # 5. Log PyFunc model + tag modality + récup version          #
            # ========================================================== #
            logger.info("[BaseLearnerExperiment.5] Log PyFunc model + tag modality")
            registered_model_name = f"rakuten-base-{self.learner_name}"

            # Path complet de la classe pour reconstruction dynamique
            learner_class_path = (
                f"{self._learner.__class__.__module__}."
                f"{self._learner.__class__.__name__}"
            )

            # Sauve les artefacts du learner dans un tmpdir, puis log_model
            # déplace tout vers le store MLflow.
            with tempfile.TemporaryDirectory() as tmp:
                learner_dir = Path(tmp) / "learner"
                self._learner.save_pretrained(learner_dir)
                logger.info(
                    f"  save_pretrained → {learner_dir} : "
                    f"{[p.name for p in learner_dir.iterdir()]}"
                )

                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=BaseLearnerPyfunc(
                        learner_class_path=learner_class_path
                    ),
                    artifacts={"learner_dir": str(learner_dir)},
                    registered_model_name=registered_model_name,
                )

            client = MlflowClient(self.tracking_uri)
            client.set_registered_model_tag(
                registered_model_name,
                key="modality",
                value=self._learner.modality,
            )
            logger.info(f"  Logged : {registered_model_name}")

            # Récup la version du modèle qui vient d'être loggé
            versions = client.search_model_versions(
                f"name='{registered_model_name}'"
            )
            latest_version = max(int(v.version) for v in versions)
            logger.info(f"  Version loggée : v{latest_version}")

            # ========================================================== #
            # 6. Décision promotion @active (AVANT cache parquet !)        #
            # ========================================================== #
            logger.info("[BaseLearnerExperiment.6] Décision promotion @active")
            threshold = self.config.get("promotion_threshold", 0.005)
            should_promote = compute_promotion_decision(
                registered_model_name, self._run_id, threshold=threshold
            )
            mlflow.log_param("promote_to_active", should_promote)

            # ========================================================== #
            # 7. Si promu : alias @active + cascade + write parquet        #
            # ========================================================== #
            if should_promote:
                promotion_success = False
                try:
                    # 7a. Set alias @active
                    client.set_registered_model_alias(
                        registered_model_name, "active", latest_version
                    )
                    logger.info(f"[BaseLearnerExperiment.7a] ✓ Promu @active : v{latest_version}")
 
                    # 7b. Cascade @active_text / @active_image
                    logger.info("[BaseLearnerExperiment.7b] Cascade @active_text/image")
                    refresh_modality_alias(self._learner.modality)
                    logger.info("[BaseLearnerExperiment.7b] ✓ Cascade OK")
 
                    # 7c. Write cache parquet (UNIQUEMENT si promu — invariant M.4bis)
                    logger.info(
                        "[BaseLearnerExperiment.7c] Extract embeddings + write cache parquet"
                    )
                    self._write_cache_parquet(
                        datamodule, source_model_version=latest_version
                    )
                    logger.info("[BaseLearnerExperiment.7c] ✓ Cache parquet écrit")
                    promotion_success = True
 
                except Exception as e:
                    logger.error(
                        f"[BaseLearnerExperiment.7] ✗ ERREUR bloc promotion !\n"
                        f"  Exception : {type(e).__name__}: {e}\n"
                        f"  Le modèle v{latest_version} est entraîné et loggé,\n"
                        f"  mais l'alias @active et/ou le cache parquet peuvent être\n"
                        f"  incohérents. Vérifier manuellement.",
                        exc_info=True,
                    )
                    mlflow.log_param("promotion_error", f"{type(e).__name__}: {str(e)[:200]}")
 
                mlflow.log_param("promotion_success", promotion_success)
            else:
                logger.info(
                    f"[BaseLearnerExperiment.7] Non promu (gain < {threshold}). "
                    f"Cache parquet conservé tel quel."
                )


    def _write_cache_parquet(
        self,
        datamodule: "RakutenLightningDataModule",
        source_model_version: int | None = None,
    ) -> None:
        """
        Extract embeddings sur _df_full (tous productids), construit un cache
        parquet avec colonnes métadonnées + features.

        Cache layout :
            productid, batch_id, is_gold, is_val_selection_v1..v_max,
            source_model_name, source_model_version,
            {learner_name}_feat_0, ..., {learner_name}_feat_{embed_dim-1}

        Filename (cohérence avec DataModule._load_base_learner_embeddings) :
            embeddings_{learner_name}_v{ACTIVE_VAL_SELECTION_VERSION}.parquet

        Args:
            datamodule: RakutenLightningDataModule (setupé).
            source_model_version: version MLflow @active (REQUIS pour guard-fou M.7).

        """
        logger.info("[_write_cache_parquet] Début extract embeddings")

        df_full = datamodule._df_full
        if df_full is None:
            raise RuntimeError(
                "_df_full non chargé. DataModule.setup() doit avoir été appelé."
            )

        # Extract embeddings sur _df_full (tous productids : train + gold + shadow)
        # Corrige FIXME 2 : le cache doit couvrir 100% des samples pour éviter
        # les pertes au inner join dans DataModule.setup() (extra_embedding_caches).
        X_full, _ = datamodule.get_full_data(include_raw=True)
        logger.info(f"  Extract embeddings sur {len(X_full)} samples (full coverage)")

        embeddings = self._learner.extract_embeddings(X_full)
        logger.info(f"  Shape embeddings : {embeddings.shape}")

        # Construire le cache DataFrame
        cache_data = {
            "productid": X_full["productid"].to_numpy(),
            "batch_id": (
                X_full.get_column("batch_id").to_numpy()
                if "batch_id" in X_full.columns
                else np.full(len(X_full), 1, dtype=int)
            ),
            "source_model_name": self.learner_name,
            "source_model_version": source_model_version or 1,
        }

        # Ajouter colonnes is_val_selection_v1..v_max, is_gold, batch_id depuis df_full.
        # X_full est issu de get_full_data() donc couvre 100% de df_full, dans le
        # même ordre. On jointe quand même sur productid pour la robustesse.
        all_pids = X_full["productid"].to_list()
        df_meta = df_full.join(
            pl.DataFrame({"productid": all_pids,
                          "_order": list(range(len(all_pids)))}),
            on="productid",
            how="inner",
        ).sort("_order")


        for col in df_meta.columns:
            if col.startswith("is_val_selection_v"):
                cache_data[col] = df_meta[col].to_numpy()
        if "is_gold" in df_meta.columns:
            cache_data["is_gold"] = df_meta["is_gold"].to_numpy()

        # Ajouter embeddings
        for i in range(embeddings.shape[1]):
            cache_data[f"{self.learner_name}_feat_{i}"] = embeddings[:, i]

        cache_df = pl.DataFrame(cache_data)
        logger.info(f"  Cache shape : {cache_df.shape}")

        # Write parquet — filename SANS slugify (cohérence avec
        # DataModule._load_base_learner_embeddings qui attend le learner_name brut)
        cache_filename = (
            f"embeddings_{self.learner_name}_"
            f"v{get_active_val_selection_version()}.parquet"
        )
        cache_path = self.cache_output_dir / cache_filename
        cache_df.write_parquet(cache_path)
        logger.info(f"  Cache écrit : {cache_path}")

                # ── Push vers R2 via boto3 (backup hors volume persistant) ───
        # dvc add échoue sur le pod car data/cache est un symlink vers
        # le volume persistant (DVC refuse les symlinked directories).
        # On pousse directement sur R2 via boto3.
        try:
            import boto3
 
            r2_endpoint = os.getenv("R2_ENDPOINT_URL")
            r2_key = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
            r2_secret = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
            r2_bucket = os.getenv("R2_BUCKET", "rakuten-mlops-dvc")
 
            if not all([r2_endpoint, r2_key, r2_secret]):
                logger.warning("  R2 credentials manquantes, skip push R2")
            else:
                s3 = boto3.client(
                    "s3",
                    endpoint_url=r2_endpoint,
                    aws_access_key_id=r2_key,
                    aws_secret_access_key=r2_secret,
                )
                r2_key_name = f"embedding_caches/{cache_filename}"
                s3.upload_file(str(cache_path), r2_bucket, r2_key_name)
                logger.info(f"  ✓ Push R2 OK : s3://{r2_bucket}/{r2_key_name}")
        except Exception as e:
            logger.warning(f"  Push R2 échoué (non bloquant) : {type(e).__name__}: {e}")



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
