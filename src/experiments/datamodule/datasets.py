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
from PIL import Image

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

class MultimodalDataset(Dataset):
    """
    Dataset multimodal pour M3 : tokenise le texte et transforme l'image.
 
    Chaque __getitem__ retourne un dict de tensors prêts pour le forward
    de M3AttentionFusion. Le collate_fn par défaut de DataLoader stack
    les tensors automatiquement.
 
    Args:
        texts: liste de textes bruts (designation + description concaténées).
        image_paths: liste de chemins absolus vers les images.
        labels: array de labels entiers (None pour l'inférence).
        tokenizer: tokenizer HuggingFace du text encoder
            (ex: CamembertTokenizer). Extrait du base learner par le
            builder dans runner.py.
        max_len: longueur max de tokenisation (padding/truncation).
        image_transform: torchvision transform de l'image encoder
            (ex: eval_transform de ResNet18). Extrait du base learner
            par le builder dans runner.py.
    """
 
    def __init__(
        self,
        texts: list[str],
        image_paths: list[str],
        labels: np.ndarray | None,
        tokenizer,
        max_len: int,
        image_transform,
    ):
        assert len(texts) == len(image_paths), (
            f"texts ({len(texts)}) et image_paths ({len(image_paths)}) "
            f"doivent avoir la même longueur."
        )
        if labels is not None:
            assert len(labels) == len(texts), (
                f"labels ({len(labels)}) et texts ({len(texts)}) "
                f"doivent avoir la même longueur."
            )
 
        self.texts = texts
        self.image_paths = image_paths
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transform = image_transform
 
    def __len__(self) -> int:
        return len(self.texts)
 
    def __getitem__(self, idx: int) -> dict:
        # --- Texte → input_ids + attention_mask ---
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
 
        # --- Image → tensor (3, 224, 224) ---
        img = Image.open(self.image_paths[idx]).convert("RGB")
        item["image"] = self.image_transform(img)
 
        # --- Label (optionnel) ---
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
 
        return item
 
