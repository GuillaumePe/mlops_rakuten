"""
M2Baseline : reproduction de M2 v4 dans l'architecture modulaire Phase 1.

Assemble :
    CamembertFrozen   ──► text_feat_0..767  ──┐
    ResNet18Frozen    ──► image_feat_0..511 ──┼──► StackingLGBM ──► ŷ
    (features tabulaires Rakuten)         ───┘

Pour les base learners frozen, les embeddings sont produits par
RakutenLightningDataModule.prepare_data() et déjà présents dans le DataFrame
d'entrée. Le fit() des BaseLearners est no-op ; tout le travail (K-Fold OOF,
Optuna HPO, refit) est dans StackingLGBM.

Test d'intégration critique (L.5) : ce modèle doit reproduire les chiffres
M2 v4 à variance d'init près :
    F1 weighted gold ∈ [0.795, 0.805]
    ECE ≤ 0.02
    cv_score_std ≤ 0.01
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from src.models.base_learners.text.camembert_frozen import CamembertFrozen
from src.models.base_learners.image.resnet18_frozen import ResNet18Frozen
from src.models.fusion.stacking_lgbm import StackingLGBM


class M2Baseline:
    """
    Modèle assemblé baseline : CamemBERT frozen + ResNet18 frozen + StackingLGBM.

    Args:
        tabular_cols: noms des features tabulaires Rakuten dans le DataFrame
        text_embed_dim: dimension d'embedding texte (768 pour camembert-base)
        image_embed_dim: dimension d'embedding image (512 pour ResNet18)
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
        text_embed_dim: int = 768,
        image_embed_dim: int = 512,
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

        # Base learners frozen (passe-plat sur le cache parquet)
        self.text_learner = CamembertFrozen(embed_dim=text_embed_dim)
        self.image_learner = ResNet18Frozen(embed_dim=image_embed_dim)

        # Stacking méta : reçoit les noms de colonnes via les BaseLearners
        self.stacking = StackingLGBM(
            text_cols=self.text_learner.cols,
            image_cols=self.image_learner.cols,
            tabular_cols=tabular_cols,
            n_classes=n_classes,
            n_folds=n_folds,
            n_trials=n_trials,
            random_state=random_state,
            n_jobs_optuna=n_jobs_optuna,
            warm_start_params=warm_start_params,
        )

    # ------------------------------------------------------------------ #
    # API sklearn-style (consommée par SklearnExperiment)                 #
    # ------------------------------------------------------------------ #

    def fit(self, X: pl.DataFrame, y: np.ndarray) -> "M2Baseline":
        """
        Pipeline complet : fit base learners (no-op pour frozen) → stacking.

        Pour M2.1/M2.2, la méthode équivalente fera un vrai entraînement deep
        des BaseLearners, puis enrichira X avec les colonnes d'embeddings
        produites par extract_embeddings(), avant d'appeler stacking.fit().
        """
        # 1. Fit des BaseLearners (no-op pour frozen, vérifie présence des colonnes)
        self.text_learner.fit(X, y)
        self.image_learner.fit(X, y)

        # 2. Fit du StackingLGBM (K-Fold OOF + Optuna + refit final)
        self.stacking.fit(X, y)
        return self

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        return self.stacking.predict(X)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        return self.stacking.predict_proba(X)
    def _to_numpy(self, X: pl.DataFrame, cols: list[str]) -> np.ndarray:
        """
        Délégué à StackingLGBM. Exposé pour compatibilité avec
        SklearnExperiment._log_stacking_analysis (couplage hérité de M2Stacking
        qui exposait cette méthode "privée" via convention).
        """
        return self.stacking._to_numpy(X, cols)
    
    # ------------------------------------------------------------------ #
    # Délégation des propriétés utiles pour SklearnExperiment             #
    # (les chemins de logging dans MLflow restent identiques à M2 v4)     #
    # ------------------------------------------------------------------ #

    @property
    def text_cols(self) -> list[str]:
        return self.text_learner.cols

    @property
    def image_cols(self) -> list[str]:
        return self.image_learner.cols

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
    # Métadonnées (pour MLflow tags + dashboard Phase 4)                  #
    # ------------------------------------------------------------------ #

    @property
    def metadata(self) -> dict:
        return {
            "model_family": "M2",
            "model_variant": "baseline",
            "base_text": self.text_learner.name,
            "base_image": self.image_learner.name,
            "fusion": "stacking_lgbm",
            "text_embed_dim": self.text_learner.embed_dim,
            "image_embed_dim": self.image_learner.embed_dim,
            "n_tabular": len(self.tabular_cols),
        }