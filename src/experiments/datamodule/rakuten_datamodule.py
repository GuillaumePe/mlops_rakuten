"""
RakutenLightningDataModule : orchestre l'extraction d'embeddings, le cache 
parquet incrémental, et le split stratifié train/val/test.

Modes :
- "m2_embeddings" : retourne embeddings + tabulaires depuis cache parquet (M2)
- "raw_for_finetune" : retourne (image, text_tokens, label) à la volée (M3)
- "m4_embeddings" : cache CLIP/SigLIP (Phase 2, NotImplementedError)

Cache parquet :
- Filename = embeddings_<text_slug>_<image_slug>_v<N>.parquet
- Indexé par productid → permet l'incrémental : on calcule uniquement les
  productid manquants à chaque appel de prepare_data()
- Changement de modèle = nouveau filename = nouveau cache (pas d'écrasement
  silencieux qui produirait des embeddings hétérogènes)

Le DataModule reste agnostique du modèle aval. Il fournit embeddings +
tabulaires bruts. Les OOF predictions et base learners finaux sont calculés
par M2Stacking.fit().
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import os
import json
import numpy as np
import polars as pl
import pytorch_lightning as pl_lightning
import torch
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.experiments.datamodule.encoders import TextEncoder, ImageEncoder, slugify_model_name
from src.experiments.datamodule.datasets import EmbeddingsDataset, RawMultimodalDataset
from src.experiments.datamodule.tabular_features import (
    extract_text_tabular,
    extract_image_tabular,
    TEXT_TABULAR_COLS,
    IMAGE_TABULAR_COLS,
)
from src.features.utils import clean_description  # legacy, à refondre en Bloc F


# Constantes Mongo et chemins data
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = "MAR25_CMLOPS_RAKUTEN"
IMAGE_FOLDER_TRAIN = Path("data/raw_data/images/image_train")
CACHE_DIR = Path("data/cache")


class RakutenLightningDataModule(pl_lightning.LightningDataModule):
    """
    DataModule unifié pour M2/M3/M4.

    Args:
        mode: "m2_embeddings" | "raw_for_finetune" | "m4_embeddings"
        text_model: nom HuggingFace du modèle texte
        image_model: nom torchvision (ex: "resnet18")
        cache_version: int incrémenté manuellement si la logique d'extraction change
        batch_size: pour les DataLoader (M3 surtout)
        num_workers: parallélisme IO
        val_size: fraction du total pour validation
        test_size: fraction du total pour test
        random_state: seed du split
    """

    def __init__(
        self,
        mode: Literal["m2_embeddings", "raw_for_finetune", "m4_embeddings"] = "m2_embeddings",
        text_model: str = "dangvantuan/sentence-camembert-base",
        image_model: str = "resnet18",
        cache_version: int = 1,
        batch_size: int = 64,
        num_workers: int = 4,
        val_size: float = 0.20,
        test_size: float = 0.10,
        random_state: int = 42,
        limit: int | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.text_model = text_model
        self.image_model = image_model
        self.cache_version = cache_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.limit = limit 

        # Chemin du cache, déterministe à partir des modèles + version
        self.cache_path = CACHE_DIR / (
            f"embeddings_{slugify_model_name(text_model)}_"
            f"{slugify_model_name(image_model)}_v{cache_version}.parquet"
        )

        # Placeholders remplis dans setup()
        self._df_full: pl.DataFrame | None = None
        self._df_train: pl.DataFrame | None = None
        self._df_val: pl.DataFrame | None = None
        self._df_test: pl.DataFrame | None = None
        self._text_cols: list[str] | None = None
        self._image_cols: list[str] | None = None
        self._tabular_cols: list[str] | None = None

    # --- prepare_data : extraction + cache incrémental --------------------

    def prepare_data(self):
        """
        Garantit que le cache contient embeddings + tabulaires pour tous les
        productid actuellement en base. Calcule uniquement les manquants.
        """
        if self.mode != "m2_embeddings":
            raise NotImplementedError(
                f"prepare_data pour mode={self.mode} sera implémenté en Phase 1+"
            )

        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        # 1. Récupérer tous les productid en base (X et label)
        
        cursor  = db["X_raw_data_batches"].find(
            {}, {"_id": 0, "productid": 1, "imageid": 1, "designation": 1, "description": 1}
        )
        if self.limit is not None:
            cursor = cursor.limit(self.limit)
        all_docs = list(cursor)
        print(f"[DataModule] {len(all_docs)} docs lus depuis Mongo (limit={self.limit})")
        all_productids = {d["productid"] for d in all_docs}

        # 2. Lire les productid déjà en cache
        if self.cache_path.exists():
            cached_pids = set(
                pl.read_parquet(self.cache_path, columns=["productid"])
                  .get_column("productid")
                  .to_list()
            )
        else:
            cached_pids = set()
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        missing_pids = all_productids - cached_pids
        if not missing_pids:
            print(f"[DataModule] Cache à jour ({len(cached_pids)} embeddings de pids)")
            return

        print(f"[DataModule] Cache : {len(cached_pids)} existants, {len(missing_pids)} à calculer")

        # 3. Filtrer les docs à traiter
        docs_to_process = [d for d in all_docs if d["productid"] in missing_pids]

        # 4. Charger labels depuis Y_raw_data_batches
        y_docs = list(db["Y_raw_data_batches"].find(
            {"productid": {"$in": list(missing_pids)}},
            {"_id": 0, "productid": 1, "prdtypecode": 1},
        ))
        raw_labels_map  = {d["productid"]: d["prdtypecode"] for d in y_docs}
        all_codes = sorted(
            set(d["prdtypecode"] for d in db["Y_raw_data_batches"].find({}, {"prdtypecode": 1, "_id": 0}))
        )
        self._code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
        self._idx_to_code = {idx: code for code, idx in self._code_to_idx.items()}
        labels_map = {pid: self._code_to_idx[code] for pid, code in raw_labels_map.items()}

        # 5. Préparer textes + chemins images + extraire features tabulaires
        texts = []
        image_paths = []
        productids_ordered = []
        imageids_ordered = []
        labels_ordered = []
        text_tab_records = []
        image_tab_records = []

        for d in docs_to_process:
            pid = d["productid"]
            iid = d["imageid"]
            if pid not in labels_map:
                # Sample sans label (rare, défensif) : on skip
                continue

            designation = d.get("designation", "") or ""
            description = d.get("description", "") or ""
            full_text = f"{designation}. {description}" if description else designation

            texts.append(clean_description(full_text))
            img_path = IMAGE_FOLDER_TRAIN / f"image_{iid}_product_{pid}.jpg"
            image_paths.append(img_path)
            productids_ordered.append(pid)
            imageids_ordered.append(iid)
            labels_ordered.append(labels_map[pid])

            # Tabulaires extraits dans la même boucle (texte = quasi-instantané,
            # image = ~1ms par sample, négligeable)
            text_tab_records.append(extract_text_tabular(designation, description))
            image_tab_records.append(extract_image_tabular(img_path))

        # 6. Encoders (forward GPU si disponible)
        print(f"[DataModule] Encoding texte ({len(texts)} samples)...")
        text_encoder = TextEncoder(self.text_model)
        text_emb = text_encoder.encode(texts, batch_size=64)

        print(f"[DataModule] Encoding images ({len(image_paths)} samples)...")
        image_encoder = ImageEncoder(self.image_model)
        image_emb = image_encoder.encode(image_paths, batch_size=64, num_workers=self.num_workers)

        # 7. Assembler en DataFrame Polars : ids + label + embeddings + tabulaires
        new_data = {
            "productid": productids_ordered,
            "imageid": imageids_ordered,
            "label": labels_ordered,
        }
        for i in range(text_emb.shape[1]):
            new_data[f"text_feat_{i}"] = text_emb[:, i].tolist()
        for i in range(image_emb.shape[1]):
            new_data[f"image_feat_{i}"] = image_emb[:, i].tolist()
        for col in TEXT_TABULAR_COLS:
            new_data[col] = [r[col] for r in text_tab_records]
        for col in IMAGE_TABULAR_COLS:
            new_data[col] = [r[col] for r in image_tab_records]

        new_df = pl.DataFrame(new_data)

        # 8. Append au cache existant (incrémental)
        if self.cache_path.exists():
            existing_df = pl.read_parquet(self.cache_path)
            full_df = pl.concat([existing_df, new_df], how="vertical")
        else:
            full_df = new_df

        full_df.write_parquet(self.cache_path)
        print(f"[DataModule] Cache écrit : {len(full_df)} embeddings → {self.cache_path}")

    # --- setup : load + split stratifié -----------------------------------

    def setup(self, stage: str | None = None):
        """
        Charge le cache et fait le split train/val/test stratifié sur le label.
        Le split est fait AVANT toute opération preprocessing (anti-fuite).
        """
        if self.mode != "m2_embeddings":
            raise NotImplementedError(f"setup pour mode={self.mode} non encore supporté")

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Cache introuvable : {self.cache_path}. Lance prepare_data() d'abord."
            )

        df = pl.read_parquet(self.cache_path)
        mapping_path = self.cache_path.with_suffix(".labels.json")
        if mapping_path.exists():
            data = json.loads(mapping_path.read_text())
            self._idx_to_code = {int(k): v for k, v in data["idx_to_code"].items()}
            self._code_to_idx = {int(k): v for k, v in data["code_to_idx"].items()}
        if self.limit is not None and len(df) > self.limit:
            df = df.head(self.limit)
            print(f"[DataModule] Cache limité à {self.limit} samples pour ce run")

        self._text_cols = [c for c in df.columns if c.startswith("text_feat_")]
        self._image_cols = [c for c in df.columns if c.startswith("image_feat_")]
        self._tabular_cols = [c for c in df.columns if c.startswith("tab_")]

        labels = df.get_column("label").to_numpy()
        indices = np.arange(len(df))

        # Split stratifié à 3 (train, val, test).
        # On ajuste la fraction val pour avoir test_size + val_size du TOTAL.
        idx_trainval, idx_test = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state,
        )
        labels_trainval = labels[idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size=self.val_size / (1 - self.test_size),
            stratify=labels_trainval,
            random_state=self.random_state,
        )

        self._df_full = df
        self._df_train = df[idx_train.tolist()]
        self._df_val = df[idx_val.tolist()]
        self._df_test = df[idx_test.tolist()]

        print(
            f"[DataModule] Split : train={len(self._df_train)}, "
            f"val={len(self._df_val)}, test={len(self._df_test)}"
        )

    # --- Interface Lightning (DataLoaders pour M3/M4) ---------------------

    def _make_loader(self, df: pl.DataFrame, shuffle: bool) -> DataLoader:
        ds = EmbeddingsDataset(df, self._text_cols, self._image_cols)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self._df_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self._df_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self._df_test, shuffle=False)

    # --- Interface sklearn (DataFrames bruts pour M2) ---------------------

    def get_sklearn_data(
        self, split: Literal["train", "val", "test"]
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) pour usage sklearn-style (M2Stacking).
        X est le DataFrame Polars avec embeddings texte + image + tabulaires.
        y est un numpy array de labels.
        """
        if self.mode != "m2_embeddings":
            raise ValueError("get_sklearn_data est uniquement pour mode='m2_embeddings'")

        df_map = {"train": self._df_train, "val": self._df_val, "test": self._df_test}
        df = df_map[split]
        feat_cols = self._text_cols + self._image_cols + self._tabular_cols
        X = df.select(feat_cols)
        y = df.get_column("label").to_numpy()
        return X, y

    # --- Properties pour exposer les listes de colonnes -------------------

    @property
    def text_cols(self) -> list[str]:
        return self._text_cols

    @property
    def image_cols(self) -> list[str]:
        return self._image_cols

    @property
    def tabular_cols(self) -> list[str]:
        return self._tabular_cols
    
    @property
    def idx_to_code(self) -> dict[int, int]:
        return self._idx_to_code