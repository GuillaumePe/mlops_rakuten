"""
ResNet18Frozen : BaseLearner image pour M2 baseline.

Wrapper minimaliste autour des embeddings ResNet18 frozen produits par
RakutenLightningDataModule. Ne fait PAS le forward pass : il lit directement
les colonnes image_feat_0..511 du DataFrame d'entrée.

Pour M2.1 (ResNet18 full fine-tune), voir `resnet18_full_ft.py`.
Pour M2.2 (ResNet50 partial FT), voir `resnet50_partial_ft.py`.

Le LogReg qui s'applique aux embeddings est géré en aval par StackingLGBM.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from src.models.base_learners._base import BaseLearner


class ResNet18Frozen(BaseLearner):
    """
    BaseLearner image ResNet18 frozen (passe-plat sur cache parquet).

    Args:
        embed_dim: dimension d'embedding (512 pour ResNet18 sans tête FC)
        col_prefix: préfixe des colonnes dans le DataFrame d'entrée
    """

    def __init__(
        self,
        embed_dim: int = 512,
        col_prefix: str = "image_feat",
    ):
        self._embed_dim = embed_dim
        self.col_prefix = col_prefix
        self._fitted = False

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "image"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def name(self) -> str:
        return "resnet18_frozen"

    @property
    def cols(self) -> list[str]:
        return [f"{self.col_prefix}_{i}" for i in range(self._embed_dim)]

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        val_split: float = 0.2,
    ) -> "ResNet18Frozen":
        """No-op (cf. CamembertFrozen.fit)."""
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"ResNet18Frozen.fit : {len(missing)} colonnes manquantes dans X. "
                f"Premières manquantes : {missing[:3]}. "
                f"Le cache parquet du RakutenLightningDataModule a-t-il été produit ?"
            )
        self._fitted = True
        return self

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        if not self._fitted:
            missing = [c for c in self.cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"ResNet18Frozen.extract_embeddings : colonnes manquantes. "
                    f"Appeler fit() d'abord ou produire le cache parquet."
                )
        return X.select(self.cols).to_numpy().astype(np.float32)

 