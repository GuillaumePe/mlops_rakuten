"""
P.1 — RakutenScorer : dispatcher de scoring par model_family.

Résout TD-003 : le scoring n'est plus hardcodé sur pca_lgbm_pipeline.
Reconstruit le pipeline correct à partir du champion dans le registry MLflow,
en dispatchant sur le tag `model_family` (M2 / M3 / M3_2).

Deux chemins de scoring fondamentalement différents :

    M2 (stacking tabulaire)
        champion pyfunc → predict(DataFrame d'embeddings)
        Les embeddings sont extraits par les base learners @active_text/@active_image
        chargés séparément (forward CPU).

    M3 / M3.2 (fusion par attention)
        champion = fusion module seul (mlflow.pytorch)
        Les encodeurs frozen sont reconstruits depuis les base learners
        identifiés par les version tags (base_text, base_image + versions).
        Le modèle complet est assemblé en mémoire → forward end-to-end.

Usage (depuis une action runner ou l'API) :

    scorer = RakutenScorer.from_champion(
        model_name="rakuten-m2-best",
        tracking_uri="http://mlflow:5000",
    )
    results = scorer.score(raw_df)
    # results = {"predictions": np.ndarray, "probas": np.ndarray,
    #            "model_name": str, "model_version": str, "model_family": str}

Contrat d'entrée : raw_df est un polars DataFrame avec les colonnes :
    productid, designation, description, imageid, image_path
(= le schéma brut issu de X_to_predict / X_raw_data_batches Mongo).
La concaténation designation + description → text + clean_description()
est effectuée en interne par le scorer (symétrie train/predict garantie).
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from mlflow.tracking import MlflowClient
from src.models.utils import ensure_device
from torch.utils.data import DataLoader
from src.experiments.datamodule.datasets import MultimodalDataset
from src.features.utils import clean_description
from src.experiments.datamodule.tabular_features import extract_text_tabular, extract_image_tabular, TEXT_TABULAR_COLS, IMAGE_TABULAR_COLS


logger = logging.getLogger(__name__)

# Mapping learner_name → préfixe de colonne dans le DataFrame d'embeddings M2.
# Doit rester synchronisé avec src/models/assembled/m2_assembled.py::_LEGACY_COL_PREFIX.
_COL_PREFIX = {
    "textcnn": "textcnn_feat",
    "camembert_frozen": "camembert_frozen_feat",
    "camembert_lora": "camembert_lora_feat",
    "resnet50_partial_ft": "resnet50_partial_ft_feat",
    "resnet18_full_ft": "resnet18_full_ft_feat",
    "resnet18_frozen": "resnet18_frozen_feat",
    "siglip2": "siglip2_feat",
}

# Colonnes tabulaires : importées depuis le DataModule au besoin dans _build_tabular_features.


def _prepare_text_column(raw_df: pl.DataFrame) -> pl.DataFrame:
    """
    Ajoute la colonne `text` au DataFrame brut.

    Reproduit exactement la logique du DataModule :
        full_text = f"{designation}. {description}" if description else designation
        text = clean_description(full_text)

    Garantit la symétrie train/predict (même preprocessing textuel).
    """


    designation = raw_df["designation"].to_list()
    description = (
        raw_df["description"].to_list()
        if "description" in raw_df.columns
        else [""] * len(designation)
    )

    texts = []
    for desig, desc in zip(designation, description):
        desig = str(desig) if desig is not None else ""
        desc = str(desc) if desc is not None else ""
        full_text = f"{desig}. {desc}" if desc.strip() else desig
        texts.append(clean_description(full_text))

    return raw_df.with_columns(pl.Series("text", texts))


@dataclass
class ScoringResult:
    """Résultat structuré d'un appel score()."""
    predictions: np.ndarray          # (n,) int — classe prédite
    probas: np.ndarray               # (n, 27) float32 — probas par classe
    model_name: str
    model_version: str
    model_family: str
    n_scored: int = 0


class RakutenScorer:
    """
    Scorer dispatché par model_family.

    Ne pas instancier directement — utiliser la classmethod from_champion().
    """

    def __init__(
        self,
        model_family: str,
        model_name: str,
        model_version: str,
        tracking_uri: str,
        champion_meta: dict,
    ):
        self.model_family = model_family
        self.model_name = model_name
        self.model_version = model_version
        self.tracking_uri = tracking_uri
        self.champion_meta = champion_meta  # tags du run/version
        self._client = MlflowClient(tracking_uri)

        # Chargé paresseusement au premier score()
        self._ready = False
        self._scorer_impl: Optional[_ScorerM2 | _ScorerM3] = None

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_champion(
        cls,
        model_name: str,
        tracking_uri: str = "",
        alias: str = "champion",
    ) -> "RakutenScorer":
        """
                Résout l'alias champion du modèle nommé et construit le scorer.

        P.3c — alias paramétrable : 'champion' (Phase 1, défaut),
        'champion_stateless' / 'champion_stateful' (Phase 3).

        Args:
            model_name: nom du registered model MLflow
                (ex: "rakuten-m2-best", "rakuten-m3-attention-fusion").
            tracking_uri: URI du serveur MLflow.

        Raises:
            ValueError: si pas de @champion ou model_family inconnu.
        """
        import os
        if not tracking_uri:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri)

        # 1. Résoudre @champion
        try:
            version_info = client.get_model_version_by_alias(
                model_name, alias
            )
        except Exception as e:
            raise ValueError(
                f"Pas de @{alias} pour '{model_name}'. "
                f"Vérifier le registry MLflow. Cause : {e}"
            ) from e

        version = version_info.version
        run_id = version_info.run_id
        logger.info(
            f"[RakutenScorer] Champion résolu : {model_name} "
            f"v{version} (run_id={run_id})"
        )

        # 2. Récupérer les tags (version tags > run tags en fallback)
        _mv_tags = client.get_model_version(model_name, version).tags or {}
        version_tags = {k: v for k, v in _mv_tags.items() if v}

        run_tags = client.get_run(run_id).data.tags
        meta = {**run_tags, **version_tags}  # version tags priment

        # 3. Déterminer model_family
        model_family = meta.get("model_family", "")
        if not model_family:
            raise ValueError(
                f"Tag 'model_family' absent pour {model_name} v{version}. "
                f"Impossible de dispatcher le scoring."
            )
        logger.info(f"[RakutenScorer] model_family={model_family}")

        return cls(
            model_family=model_family,
            model_name=model_name,
            model_version=version,
            tracking_uri=tracking_uri,
            champion_meta=meta,
        )

    # ------------------------------------------------------------------ #
    # Score — point d'entrée principal                                     #
    # ------------------------------------------------------------------ #

    def score(self, raw_df: pl.DataFrame) -> ScoringResult:
        """
        Score un DataFrame brut (productid, designation, imageid, image_path).

        Lazy-load le pipeline au premier appel.
        """
        if not self._ready:
            self._load()

        preds, probas = self._scorer_impl.predict(raw_df)

        return ScoringResult(
            predictions=preds,
            probas=probas,
            model_name=self.model_name,
            model_version=self.model_version,
            model_family=self.model_family,
            n_scored=len(preds),
        )

    # ------------------------------------------------------------------ #
    # Chargement paresseux (dispatch)                                      #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """Instancie le bon scorer interne selon model_family."""
        family = self.model_family.upper().replace(".", "")

        if family.startswith("M2"):
            self._scorer_impl = _ScorerM2(
                model_name=self.model_name,
                model_version=self.model_version,
                meta=self.champion_meta,
                client=self._client,
                tracking_uri=self.tracking_uri,
            )
        elif family in ("M3", "M3_2", "M32"):
            self._scorer_impl = _ScorerM3(
                model_name=self.model_name,
                model_version=self.model_version,
                meta=self.champion_meta,
                client=self._client,
                tracking_uri=self.tracking_uri,
                model_family=self.model_family,
            )
        else:
            raise ValueError(
                f"model_family '{self.model_family}' non supporté. "
                f"Familles connues : M2, M2.1, M3, M3_2."
            )

        self._scorer_impl.load()
        self._ready = True
        logger.info(
            f"[RakutenScorer] Pipeline {self.model_family} chargé "
            f"({self.model_name} v{self.model_version})"
        )

    # ------------------------------------------------------------------ #
    # Introspection                                                        #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "ready" if self._ready else "not loaded"
        return (
            f"RakutenScorer(family={self.model_family}, "
            f"model={self.model_name}@v{self.model_version}, {status})"
        )


# ====================================================================== #
# Scorer interne M2 (stacking tabulaire)                                  #
# ====================================================================== #


class _ScorerM2:
    """
    Chemin M2 : base learners → embeddings → DataFrame → champion.predict(df).

    Le champion M2 attend un DataFrame polars avec :
    - colonnes d'embeddings texte ({prefix}_0 .. {prefix}_{dim-1})
    - colonnes d'embeddings image (idem)
    - colonnes tabulaires (n_words, n_chars, etc.)
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        meta: dict,
        client: MlflowClient,
        tracking_uri: str,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.meta = meta
        self.client = client
        self.tracking_uri = tracking_uri

        self._champion_model = None   # sklearn model direct (M2Assembled)
        self._text_learner = None     # BaseLearner texte (unwrapped)
        self._image_learner = None    # BaseLearner image (unwrapped)
        self._text_name: str = ""
        self._image_name: str = ""

    def load(self) -> None:
        """Charge le champion M2 + les base learners actifs.

        On bypass la couche pyfunc pour les 3 modèles :
        - champion : accès direct via sklearn_model (attend Polars)
        - base learners : accès direct via python_model.learner
          (expose extract_embeddings(pl.DataFrame) → np.ndarray)
        """
        # 1. Champion : charger le pyfunc puis extraire le sklearn_model
        champion_uri = f"models:/{self.model_name}/{self.model_version}"
        logger.info(f"[_ScorerM2] Chargement champion : {champion_uri}")
        champion_pyfunc = mlflow.pyfunc.load_model(champion_uri)
        self._champion_model = champion_pyfunc._model_impl.sklearn_model
        if self._champion_model is None:
            raise RuntimeError(
                f"Impossible d'extraire sklearn_model depuis {champion_uri}. "
                f"Le modèle a-t-il été logué via mlflow.sklearn.log_model ?"
            )

        # 2. Identifier les base learners depuis les tags
        self._text_name = self.meta.get("base_text", "")
        self._image_name = self.meta.get("base_image", "")
        if not self._text_name or not self._image_name:
            raise ValueError(
                f"Tags 'base_text'/'base_image' manquants pour "
                f"{self.model_name} v{self.model_version}. "
                f"Tags disponibles : {list(self.meta.keys())}"
            )

        # 3. Charger les base learners et extraire le learner sous-jacent
        self._text_learner = self._load_learner(
            f"rakuten-base-{self._text_name}", alias="active"
        )
        self._image_learner = self._load_learner(
            f"rakuten-base-{self._image_name}", alias="active"
        )

        logger.info(
            f"[_ScorerM2] Prêt : text={self._text_name}, "
            f"image={self._image_name}"
        )

    @staticmethod
    def _load_learner(registered_name: str, alias: str):
        """Charge un base learner pyfunc puis extrait le BaseLearner sous-jacent."""
        uri = f"models:/{registered_name}@{alias}"
        logger.info(f"[_ScorerM2] Chargement base learner : {uri}")
        pyfunc = mlflow.pyfunc.load_model(uri)
        python_model = getattr(
            getattr(pyfunc, "_model_impl", None), "python_model", None
        )
        if python_model is None:
            raise RuntimeError(f"Pas de python_model sous-jacent pour {uri}")
        learner = getattr(python_model, "learner", None)
        ensure_device(learner)
        if learner is None:
            raise RuntimeError(f"BaseLearnerPyfunc.learner est None pour {uri}")
        return learner

    def predict(self, raw_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Score un batch de samples bruts.

        Chemin :
        1. Préparer colonne `text` (concat + clean_description)
        2. Extraire embeddings via learner.extract_embeddings(df) — Polars natif
        3. Construire le DataFrame d'embeddings + tabulaires
        4. Appeler champion_model.predict(scoring_pl) — Polars natif
        """
        # 0. Préparer la colonne `text` (concaténation + clean_description)
        raw_df = _prepare_text_column(raw_df)

        # 1. Extraire les embeddings via les base learners (bypass pyfunc)
        # extract_embeddings attend un pl.DataFrame avec colonne "text"
        text_emb = self._text_learner.extract_embeddings(raw_df)   # (n, d_text)
        image_emb = self._image_learner.extract_embeddings(raw_df)  # (n, d_image)

        # 2. Construire le DataFrame d'embeddings attendu par M2
        text_prefix = _COL_PREFIX.get(self._text_name, f"{self._text_name}_feat")
        image_prefix = _COL_PREFIX.get(self._image_name, f"{self._image_name}_feat")

        data = {}
        data["productid"] = raw_df["productid"].to_list()

        for i in range(text_emb.shape[1]):
            data[f"{text_prefix}_{i}"] = text_emb[:, i]

        for i in range(image_emb.shape[1]):
            data[f"{image_prefix}_{i}"] = image_emb[:, i]

        data.update(self._build_tabular_features(raw_df))

        scoring_pl = pl.DataFrame(data)

        # 3. Prédiction directe sur le sklearn_model (Polars natif)
        preds = self._champion_model.predict(scoring_pl)
        preds = np.asarray(preds, dtype=np.int64)

        # 4. Probas
        probas = self._try_predict_proba(scoring_pl, n_samples=len(preds))

        return preds, probas

    def _try_predict_proba(
        self, scoring_pl: pl.DataFrame, n_samples: int
    ) -> np.ndarray:
        """Tente d'extraire les probas du champion. Fallback : one-hot."""
        try:
            if hasattr(self._champion_model, "predict_proba"):
                return np.asarray(
                    self._champion_model.predict_proba(scoring_pl),
                    dtype=np.float32,
                )
        except Exception as e:
            logger.warning(f"[_ScorerM2] predict_proba indisponible : {e}")

        preds = self._champion_model.predict(scoring_pl)
        n_classes = 27
        probas = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i, p in enumerate(preds):
            probas[i, int(p)] = 1.0
        return probas

    @staticmethod
    def _build_tabular_features(raw_df: pl.DataFrame) -> dict:
        """
        Construit les features tabulaires brutes depuis designation/description + image.

        Réutilise les fonctions du DataModule pour garantir les mêmes noms
        de colonnes (tab_text_length, tab_image_width, etc.).
        """


        text_records = []
        image_records = []

        for row in raw_df.iter_rows(named=True):
            designation = str(row.get("designation", "") or "")
            description = str(row.get("description", "") or "")
            image_path = row.get("image_path", "")

            text_records.append(extract_text_tabular(designation, description))
            image_records.append(extract_image_tabular(image_path))

        result = {}
        for col in TEXT_TABULAR_COLS:
            result[col] = [r[col] for r in text_records]
        for col in IMAGE_TABULAR_COLS:
            result[col] = [r[col] for r in image_records]

        return result


# ====================================================================== #
# Scorer interne M3 / M3.2 (fusion par attention)                        #
# ====================================================================== #


class _ScorerM3:
    """
    Chemin M3/M3.2 : fusion module + base learners → modèle assemblé → forward.

    Le champion dans le registry ne contient que le module de fusion
    (AttentionFusionModule ou V2). Les encodeurs frozen sont reconstruits
    depuis les base learners identifiés par les version tags.
    """

    def __init__(
        self,
        model_name: str,
        model_version: str,
        meta: dict,
        client: MlflowClient,
        tracking_uri: str,
        model_family: str,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.meta = meta
        self.client = client
        self.tracking_uri = tracking_uri
        self.model_family = model_family

        self._fusion_module = None
        self._text_learner = None   # BaseLearner reconstruit
        self._image_learner = None  # BaseLearner reconstruit
        self._tokenizer = None
        self._image_transform = None
        self._max_len: int = 128
        self._device = torch.device("cpu")

    def load(self) -> None:
        """Charge le fusion module + reconstruit les base learners."""
        # 1. Charger le fusion module (nn.Module)
        fusion_uri = f"models:/{self.model_name}/{self.model_version}"
        logger.info(f"[_ScorerM3] Chargement fusion : {fusion_uri}")
        self._fusion_module = mlflow.pytorch.load_model(fusion_uri)
        self._fusion_module.eval()

        # Device : rester sur CPU pour le scoring (frugalité, pas de GPU requis
        # pour l'inférence d'un seul batch).
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._fusion_module = self._fusion_module.to(self._device)

        # 2. Identifier les base learners depuis les version tags
        text_name = self.meta.get("base_text", "")
        image_name = self.meta.get("base_image", "")
        text_version = self.meta.get("base_text_version", "")
        image_version = self.meta.get("base_image_version", "")

        if not text_name or not image_name:
            raise ValueError(
                f"Tags 'base_text'/'base_image' manquants pour "
                f"{self.model_name} v{self.model_version}."
            )

        # 3. Charger les base learners comme des objets Python (pas pyfunc)
        # pour accéder à .net, .tokenizer, ._eval_transform
        self._text_learner = self._load_base_learner(
            f"rakuten-base-{text_name}", text_version
        )
        self._image_learner = self._load_base_learner(
            f"rakuten-base-{image_name}", image_version
        )

        # 4. Extraire tokenizer + transform
        self._tokenizer = self._text_learner.tokenizer
        self._max_len = getattr(self._text_learner, "max_len", 128)

        # _eval_transform : méthode spécifique aux base learners image
        if hasattr(self._image_learner, "_eval_transform"):
            self._image_transform = self._image_learner._eval_transform
        elif hasattr(self._image_learner, "eval_transform"):
            self._image_transform = self._image_learner.eval_transform
        else:
            raise AttributeError(
                f"Image learner '{image_name}' n'expose pas de transform d'évaluation."
            )

        # 5. Placer les encodeurs sur le bon device, mode eval
        self._text_learner.net.eval()
        self._text_learner.net.requires_grad_(False)
        self._text_learner.net.to(self._device)
        

        self._image_learner.net.eval()
        self._image_learner.net.requires_grad_(False)
        self._image_learner.net.to(self._device)

        logger.info(
            f"[_ScorerM3] Prêt : family={self.model_family}, "
            f"text={text_name} v{text_version}, "
            f"image={image_name} v{image_version}, "
            f"device={self._device}"
        )

    def _load_base_learner(
        self, registered_name: str, version: str
    ) -> "BaseLearner":
        """
        Charge un base learner comme objet Python (pas pyfunc wrapper).

        Accède aux artefacts MLflow, puis appelle from_pretrained.
        """
        # Si version spécifiée, charger cette version ; sinon @active
        if version:
            uri = f"models:/{registered_name}/{version}"
        else:
            uri = f"models:/{registered_name}@active"

        logger.info(f"[_ScorerM3] Chargement base learner : {uri}")

        # Charger le pyfunc, puis extraire le learner sous-jacent
        pyfunc_model = mlflow.pyfunc.load_model(uri)

        # Accéder au BaseLearnerPyfunc sous-jacent
        # MLflow 2.x : _model_impl.python_model est le PythonModel
        python_model = getattr(
            getattr(pyfunc_model, "_model_impl", None),
            "python_model", None
        )
        if python_model is None:
            raise RuntimeError(
                f"Impossible d'accéder au PythonModel sous-jacent pour {uri}. "
                f"Le modèle a-t-il été logué via mlflow.pyfunc.log_model ?"
            )

        # Le BaseLearnerPyfunc a été initialisé par load_context lors du load
        learner = getattr(python_model, "learner", None)
        if learner is None:
            raise RuntimeError(
                f"BaseLearnerPyfunc.learner est None pour {uri}. "
                f"load_context n'a pas été appelé ou a échoué."
            )
        ensure_device(learner)
        return learner

    def predict(
        self, raw_df: pl.DataFrame, batch_size: int = 64, num_workers: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Score un batch de samples bruts via le modèle assemblé M3/M3.2.

        Mini-batché via MultimodalDataset + DataLoader (le MÊME Dataset que le
        DataModule à l'entraînement → symétrie train/inférence, zéro duplication
        de tokenize/transform). Empreinte mémoire O(batch_size · L²), constante
        en n → robuste à la croissance du gold set.
        """

        # 0. Préparer la colonne `text` (concaténation + clean_description)
        raw_df = _prepare_text_column(raw_df)

        texts = raw_df["text"].to_list()
        image_paths = raw_df["image_path"].to_list()

        dataset = MultimodalDataset(
            texts=texts,
            image_paths=image_paths,
            labels=None,  # inférence
            tokenizer=self._tokenizer,
            max_len=self._max_len,
            image_transform=self._image_transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,          # ordre préservé → alignement predictions↔raw_df
            num_workers=num_workers,
            pin_memory=(self._device.type == "cuda"),
        )

        all_probas = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self._device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self._device, non_blocking=True)
                images = batch["image"].to(self._device, non_blocking=True)

                # Encodeurs frozen → token embeddings + spatial feature map
                text_tokens, text_mask = self._text_learner.net._token_level_features(
                    input_ids, attention_mask
                )
                image_patches = self._image_learner.net._spatial_feature_map(images)

                # Fusion → logits (B, n_classes)
                logits = self._fusion_module(text_tokens, text_mask, image_patches)
                all_probas.append(F.softmax(logits, dim=-1).cpu())

        probas = torch.cat(all_probas, dim=0).numpy().astype(np.float32)
        preds = probas.argmax(axis=-1).astype(np.int64)
        return preds, probas
