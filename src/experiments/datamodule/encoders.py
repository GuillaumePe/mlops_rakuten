"""
Encoders frozen utilisés par Pipline LigthGbm (et potentiellement Clip-distillation).

Chaque encoder est un wrapper fin autour d'une lib externe (sentence-transformers,
torchvision). Pas d'apprentissage ici — uniquement du forward pass batché.

"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# === Texte =====================================================================

class TextEncoder:
    """
    Wrapper sentence-transformers. Frozen, batch-encode sur GPU si disponible.

    Exemple:
        enc = TextEncoder("dangvantuan/sentence-camembert-base")
        embeddings = enc.encode(["texte 1", "texte 2"])  # shape (2, 768)
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # NOTE: sentence-transformers gère déjà le batching et le passage GPU.
        # On laisse `convert_to_numpy=True` pour rester en float32 CPU en sortie
        # (le LightGBM aval n'utilise pas le GPU).
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=False,
        )
        return embeddings.astype(np.float32)


# === Image =====================================================================

class _ImagePathDataset(Dataset):
    """Dataset minimal qui charge une image depuis disque et applique les transforms."""

    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Conversion RGB systématique : certaines images Rakuten sont en RGBA ou L.
        # Sans ça, le tensor en sortie aurait un nombre de canaux variable et
        # casserait le batching.
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class ImageEncoder:
    """
    Wrapper torchvision (ResNet par défaut). Frozen, retourne les features
    de l'avant-dernière couche (sortie de l'avg pooling, avant la classification head).

    Exemple:
        enc = ImageEncoder("resnet18")
        embeddings = enc.encode([Path("img1.jpg"), Path("img2.jpg")])  # shape (2, 512)
    """

    # Dictionnaire des architectures supportées : nom → (constructor, weights, dim).
    # Ajouter ici si on veut tester ResNet50 ou EfficientNet en Phase 1.
    _ARCHS = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(self, model_name: str = "resnet18", device: str | None = None):
        if model_name not in self._ARCHS:
            raise ValueError(f"Unsupported model {model_name}. Available: {list(self._ARCHS)}")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        constructor, weights, dim = self._ARCHS[model_name]
        self.embedding_dim = dim

        # On charge le modèle pré-entraîné, puis on remplace la dernière FC par
        # une Identity : la sortie devient l'embedding de l'avg pooling (dim 512 pour R18).
        model = constructor(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(self.device)

        # Les transforms sont fournies par les weights → garantit la cohérence
        # avec le pré-entraînement ImageNet (resize, crop, normalize correct).
        self.transform = weights.transforms()

    @torch.inference_mode()
    def encode(
        self,
        image_paths: list[Path],
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> np.ndarray:
        dataset = _ImagePathDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )

        chunks = []
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            features = self.model(batch)
            chunks.append(features.cpu().numpy())

        return np.concatenate(chunks, axis=0).astype(np.float32)


# === Helpers de naming pour le cache ==========================================

def slugify_model_name(name: str) -> str:
    """
    Transforme un nom de modèle (ex: 'dangvantuan/sentence-camembert-base') 
    en slug compatible filesystem (ex: 'sentence-camembert-base').
    
    Utilisé pour nommer le cache : embeddings_{text_slug}_{image_slug}_v{N}.parquet
    """
    return name.split("/")[-1].lower()