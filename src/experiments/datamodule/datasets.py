"""
Datasets PyTorch utilisés par RakutenLightningDataModule.

Deux modes :
- EmbeddingsDataset : retourne (text_emb, image_emb, label) depuis le cache parquet.
  Utilisé par M2 (sklearn LightGBM) et M4 (linear probe / zero-shot).
- RawMultimodalDataset : retourne (image_tensor, text_tokens, label) à la volée.
  Utilisé par M3 (fine-tune BERT + ResNet).

Les deux respectent le contrat torch.utils.data.Dataset : __len__ et __getitem__.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


# === Mode embeddings (M2, M4-zero-shot, M4-linear-probe) =====================

class EmbeddingsDataset(Dataset):
    """
    Retourne des embeddings pré-calculés depuis un DataFrame Polars.
    
    Le DataFrame doit contenir :
    - colonnes `text_feat_*` (dim variable selon le modèle texte)
    - colonnes `image_feat_*` (dim variable selon le modèle image)
    - colonne `label` (int, déjà encodée)
    
    Args:
        df: DataFrame Polars déjà filtré (sous-ensemble train ou val ou test)
        text_cols: liste des colonnes d'embedding texte
        image_cols: liste des colonnes d'embedding image
        label_col: nom de la colonne label
    """

    def __init__(
        self,
        df: pl.DataFrame,
        text_cols: list[str],
        image_cols: list[str],
        label_col: str = "label",
    ):
        # On matérialise en numpy une fois pour toutes : Polars est rapide à
        # filtrer/joindre, mais l'accès ligne-à-ligne dans __getitem__ doit
        # passer par numpy pour rester en O(1) sans overhead de conversion.
        self.text_emb = df.select(text_cols).to_numpy().astype(np.float32)
        self.image_emb = df.select(image_cols).to_numpy().astype(np.float32)
        self.labels = df.get_column(label_col).to_numpy().astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.text_emb[idx]),
            torch.from_numpy(self.image_emb[idx]),
            torch.tensor(self.labels[idx]),
        )


# === Mode raw (M3 fine-tune) ==================================================

class RawMultimodalDataset(Dataset):
    """
    Retourne (image_tensor, text_tokens, label) à la volée pour fine-tune.
    
    Pas de cache : à chaque epoch, on re-charge l'image depuis disque et on
    re-tokenize le texte. C'est intentionnel — pour M3, le BERT et le ResNet
    sont en cours de fine-tune, donc leurs embeddings changent à chaque step,
    cacher n'aurait aucun sens.
    
    """

    def __init__(
        self,
        df: pl.DataFrame,
        tokenizer,
        image_transform,
        image_root: Path,
        max_text_length: int = 256,
    ):
        # Polars n'a pas de reset_index — on travaille par row index implicite via slicing.
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.image_root = Path(image_root)
        self.max_text_length = max_text_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        raise NotImplementedError(
            "RawMultimodalDataset à implémentr."
        )