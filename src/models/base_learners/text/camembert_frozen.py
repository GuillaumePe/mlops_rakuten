"""
CamembertFrozen : BaseLearner texte pour M2 baseline.

Wrapper minimaliste autour des embeddings CamemBERT frozen produits par
RakutenLightningDataModule. Ne fait PAS le forward pass : il lit directement
les colonnes text_feat_0..767 du DataFrame d'entrée (déjà extraites via
sentence-transformers et cachées en parquet).

Pour M2.1 (CamemBERT + LoRA), voir `camembert_lora.py` qui aura un fit() réel
et un extract_embeddings() qui fait un vrai forward pass.

Le LogReg qui s'applique aux embeddings est géré en aval par StackingLGBM
(K-Fold OOF + refit final). Le `fit()` de CamembertFrozen est donc no-op.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from src.models.base_learners._base import BaseLearner


class CamembertFrozen(BaseLearner):
    """
    BaseLearner texte CamemBERT frozen (passe-plat sur cache parquet).

    Args:
        embed_dim: dimension d'embedding (768 pour `dangvantuan/sentence-camembert-base`)
        col_prefix: préfixe des colonnes dans le DataFrame d'entrée (par défaut "text_feat")
    """

    def __init__(
        self,
        embed_dim: int = 768,
        col_prefix: str = "text_feat",
    ):
        self._embed_dim = embed_dim
        self.col_prefix = col_prefix
        self._fitted = False

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "text"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def name(self) -> str:
        return "camembert_frozen"

    @property
    def cols(self) -> list[str]:
        """Liste des colonnes attendues dans le DataFrame d'entrée."""
        return [f"{self.col_prefix}_{i}" for i in range(self._embed_dim)]

    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        val_split: float = 0.2,
    ) -> "CamembertFrozen":
        """
        No-op pour un learner frozen.

        Les embeddings sont produits par RakutenLightningDataModule (cache
        parquet). Le LogReg qui transforme ces embeddings en probas est géré
        par StackingLGBM en aval (K-Fold OOF + refit final).

        On vérifie quand même que X contient bien les colonnes attendues —
        échec rapide si le cache n'a pas été produit.
        """
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"CamembertFrozen.fit : {len(missing)} colonnes manquantes dans X. "
                f"Premières manquantes : {missing[:3]}. "
                f"Le cache parquet du RakutenLightningDataModule a-t-il été produit ?"
            )
        self._fitted = True
        return self

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """
        Lit les colonnes text_feat_* du DataFrame.

        Returns:
            np.ndarray (n, embed_dim) float32, dans l'ordre des lignes de X.
        """
        if not self._fitted:
            # On accepte un appel sans fit préalable (frozen = pas d'état à apprendre)
            # mais on vérifie quand même que les colonnes sont là
            missing = [c for c in self.cols if c not in X.columns]
            if missing:
                raise ValueError(
                    f"CamembertFrozen.extract_embeddings : colonnes manquantes. "
                    f"Appeler fit() d'abord ou produire le cache parquet."
                )

        return X.select(self.cols).to_numpy().astype(np.float32)
