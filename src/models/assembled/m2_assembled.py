"""
M2Assembled : wrapper assembled paramétrique pour stacking multimodal (Phase 1).

Généralise M2Baseline en acceptant n'importe quelle paire de base learners
dont les embeddings sont pré-calculés et cachés en parquet dans le DataFrame
d'entrée. Chaque base learner — qu'il soit frozen (CamemBERT, ResNet18),
entraîné from scratch (TextCNN) ou fine-tuné (CamembertLoRA, ResNet18FullFT,
ResNet50PartialFT) — est traité comme un extracteur d'embeddings frozen au
niveau du méta-learner.

Architecture :
    emb_text(n, d_text)  ──► LogReg OOF ──► p_text ∈ Δ^27
    emb_image(n, d_image) ──► LogReg OOF ──► p_image ∈ Δ^27
    tab(n, d_tab)         ─────────────────────────────────┐
                                                           │
    [p_text, p_image, tab] ──► LightGBM méta ──► ŷ

Le contrat est identique à M2Baseline pour SklearnExperiment :
    fit(X, y) → self
    predict(X) → np.ndarray
    predict_proba(X) → np.ndarray
    + toutes les propriétés d'introspection (best_params_, cv_scores_, etc.)

Registre des préfixes de colonnes (convention du cache parquet) :
    - camembert_frozen  → text_feat_{i}       (héritage M2 v4)
    - resnet18_frozen   → image_feat_{i}      (héritage M2 v4)
    - textcnn           → textcnn_feat_{i}
    - resnet50_partial_ft → resnet50_partial_ft_feat_{i}
    - camembert_lora    → camembert_lora_feat_{i}
    - resnet18_full_ft  → resnet18_full_ft_feat_{i}

Usage :
    # M2.2 benchmark (TextCNN + ResNet50)
    model = M2Assembled(
        tabular_cols=dm.tabular_cols,
        text_learner_name="textcnn",
        text_embed_dim=3072,
        image_learner_name="resnet50_partial_ft",
        image_embed_dim=2048,
    )

    # M2.1 frugal FT (CamembertLoRA + ResNet18FullFT)
    model = M2Assembled(
        tabular_cols=dm.tabular_cols,
        text_learner_name="camembert_lora",
        text_embed_dim=768,
        image_learner_name="resnet18_full_ft",
        image_embed_dim=512,
    )

    # M2 baseline (drop-in replacement de M2Baseline)
    model = M2Assembled(
        tabular_cols=dm.tabular_cols,
        text_learner_name="camembert_frozen",
        text_embed_dim=768,
        image_learner_name="resnet18_frozen",
        image_embed_dim=512,
    )
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import polars as pl

from src.models.fusion.stacking_lgbm import StackingLGBM


# --------------------------------------------------------------------------- #
# Résolution du préfixe de colonnes parquet                                   #
# --------------------------------------------------------------------------- #
# Les frozen de base (M2 v4) utilisent text_feat / image_feat par héritage.
# Tous les autres base learners entraînés suivent la convention {name}_feat.

_LEGACY_COL_PREFIX = {
    "camembert_frozen": "text_feat",
    "resnet18_frozen": "image_feat",
}


def _resolve_col_prefix(learner_name: str) -> str:
    """Retourne le préfixe de colonnes pour un base learner donné."""
    return _LEGACY_COL_PREFIX.get(learner_name, f"{learner_name}_feat")


def _build_cols(learner_name: str, embed_dim: int) -> list[str]:
    """Construit la liste ordonnée des noms de colonnes d'embeddings."""
    prefix = _resolve_col_prefix(learner_name)
    return [f"{prefix}_{i}" for i in range(embed_dim)]


class M2Assembled:
    """
    Modèle assemblé paramétrique : text_learner + image_learner → StackingLGBM.

    Les embeddings sont lus directement depuis le DataFrame (cache parquet).
    Aucun forward pass n'est effectué — le wrapper est purement tabulaire.

    Args:
        tabular_cols: noms des features tabulaires Rakuten dans le DataFrame
        text_learner_name: nom du base learner texte (ex: "textcnn", "camembert_lora")
        text_embed_dim: dimension d'embedding texte (3072 pour TextCNN, 768 pour CamemBERT)
        image_learner_name: nom du base learner image (ex: "resnet50_partial_ft")
        image_embed_dim: dimension d'embedding image (2048 pour ResNet50, 512 pour ResNet18)
        n_classes: nombre de classes Rakuten (27)
        n_folds: K du StratifiedKFold pour OOF + HPO
        n_trials: nombre de trials Optuna
        random_state: seed reproductibilité
        n_jobs_optuna: parallélisme des trials Optuna
        warm_start_params: dict optionnel d'hyperparams à enqueue en trial #0
    """

    def __init__(
        self,
        tabular_cols: list[str],
        text_learner_name: str,
        text_embed_dim: int,
        image_learner_name: str,
        image_embed_dim: int,
        n_classes: int = 27,
        n_folds: int = 5,
        n_trials: int = 30,
        random_state: int = 42,
        n_jobs_optuna: int = 3,
        warm_start_params: dict | None = None,
    ):
        self.tabular_cols = tabular_cols
        self.n_classes = n_classes
        self.random_state = random_state

        # Identité des base learners (pour metadata + traçabilité MLflow)
        self._text_learner_name = text_learner_name
        self._text_embed_dim = text_embed_dim
        self._image_learner_name = image_learner_name
        self._image_embed_dim = image_embed_dim

        # Résolution des colonnes d'embeddings
        self._text_cols = _build_cols(text_learner_name, text_embed_dim)
        self._image_cols = _build_cols(image_learner_name, image_embed_dim)

        # Stacking méta
        self.stacking = StackingLGBM(
            text_cols=self._text_cols,
            image_cols=self._image_cols,
            tabular_cols=tabular_cols,
            n_classes=n_classes,
            n_folds=n_folds,
            n_trials=n_trials,
            random_state=random_state,
            n_jobs_optuna=n_jobs_optuna,
            warm_start_params=warm_start_params,
        )

    # ------------------------------------------------------------------ #
    # Validation colonnes (fail-fast)                                     #
    # ------------------------------------------------------------------ #

    def _validate_columns(self, X: pl.DataFrame) -> None:
        """Vérifie que toutes les colonnes d'embeddings attendues sont présentes."""
        all_expected = self._text_cols + self._image_cols
        missing = [c for c in all_expected if c not in X.columns]
        if missing:
            n_miss = len(missing)
            examples = missing[:5]
            raise ValueError(
                f"M2Assembled : {n_miss} colonnes d'embeddings manquantes dans X.\n"
                f"  Text learner '{self._text_learner_name}' attend {len(self._text_cols)} colonnes "
                f"(préfixe '{_resolve_col_prefix(self._text_learner_name)}').\n"
                f"  Image learner '{self._image_learner_name}' attend {len(self._image_cols)} colonnes "
                f"(préfixe '{_resolve_col_prefix(self._image_learner_name)}').\n"
                f"  Premières manquantes : {examples}\n"
                f"  Le cache parquet des base learners a-t-il été produit et chargé "
                f"par le DataModule (extra_embedding_caches) ?"
            )

    # ------------------------------------------------------------------ #
    # API sklearn-style (consommée par SklearnExperiment)                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> "M2Assembled":
        """
        Fit du StackingLGBM sur les embeddings pré-calculés.

        Étapes :
        1. Validation fail-fast des colonnes
        2. Délégation au StackingLGBM (K-Fold OOF + Optuna HPO + refit final)

        Pas de forward pass des base learners — les embeddings sont déjà
        dans le DataFrame X sous forme de colonnes {prefix}_feat_{i}.
        """
        self._validate_columns(X)
        self.stacking.fit(X, y)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.stacking.predict(X)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return self.stacking.predict_proba(X)

    def _to_numpy(self, X: pl.DataFrame, cols: list[str]) -> np.ndarray:
        """Délégué à StackingLGBM (compatibilité SklearnExperiment._log_stacking_analysis)."""
        return self.stacking._to_numpy(X, cols)

    # ------------------------------------------------------------------ #
    # Propriétés déléguées à StackingLGBM                                 #
    # (contrat SklearnExperiment — pas de rupture d'interface)            #
    # ------------------------------------------------------------------ #

    @property
    def text_cols(self) -> list[str]:
        return self._text_cols

    @property
    def image_cols(self) -> list[str]:
        return self._image_cols

    @property
    def best_params_(self) -> Optional[dict]:
        return self.stacking.best_params_

    @property
    def best_score_(self) -> Optional[float]:
        return self.stacking.best_score_

    @property
    def cv_scores_(self) -> Optional[list[float]]:
        return self.stacking.cv_scores_

    @property
    def oof_p_text_(self) -> Optional[np.ndarray]:
        return self.stacking.oof_p_text_

    @property
    def oof_p_image_(self) -> Optional[np.ndarray]:
        return self.stacking.oof_p_image_

    @property
    def y_train_(self) -> Optional[np.ndarray]:
        return self.stacking.y_train_

    @property
    def meta_(self):
        return self.stacking.meta_

    @property
    def f_text_(self):
        return self.stacking.f_text_

    @property
    def f_image_(self):
        return self.stacking.f_image_

    @property
    def feature_importances_lgbm_(self) -> np.ndarray:
        return self.stacking.feature_importances_lgbm_

    @property
    def feature_importances_logreg_text_(self) -> np.ndarray:
        return self.stacking.feature_importances_logreg_text_

    @property
    def feature_importances_logreg_image_(self) -> np.ndarray:
        return self.stacking.feature_importances_logreg_image_

    # ------------------------------------------------------------------ #
    # Métadonnées (MLflow tags + dashboard Phase 4)                       #
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> dict:
        return {
            "model_family": "M2",
            "model_variant": "assembled",
            "base_text": self._text_learner_name,
            "base_image": self._image_learner_name,
            "fusion": "stacking_lgbm",
            "text_embed_dim": self._text_embed_dim,
            "image_embed_dim": self._image_embed_dim,
            "n_tabular": len(self.tabular_cols),
        }

    def __repr__(self) -> str:
        return (
            f"M2Assembled("
            f"text={self._text_learner_name}[{self._text_embed_dim}], "
            f"image={self._image_learner_name}[{self._image_embed_dim}], "
            f"tab={len(self.tabular_cols)})"
        )
