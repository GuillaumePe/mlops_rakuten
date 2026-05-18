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

from dotenv import load_dotenv
load_dotenv()

# Constantes Mongo et chemins data
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = "MAR25_CMLOPS_RAKUTEN"

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TRAIN = DATA_ROOT / "data/raw_data/images/image_train"
CACHE_DIR = DATA_ROOT / "data/cache"

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
        val_size: float = 0.10,
        random_state: int = 42,
        limit: int | None = None,
        train_batches: list[int] = (1, 2, 3),
        exclude_gold: bool = True,
    ):
        super().__init__()
        self.mode = mode
        self.text_model = text_model
        self.image_model = image_model
        self.cache_version = cache_version
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.random_state = random_state
        self.limit = limit
        self.train_batches = list(train_batches)
        self.exclude_gold = exclude_gold

        # Chemin du cache, déterministe à partir des modèles + version
        self.cache_path = CACHE_DIR / (
            f"embeddings_{slugify_model_name(text_model)}_"
            f"{slugify_model_name(image_model)}_v{cache_version}.parquet"
        )

        # Placeholders remplis dans setup()
        self._df_full: pl.DataFrame | None = None
        self._df_train_pool: pl.DataFrame | None = None
        self._df_train: pl.DataFrame | None = None
        self._df_val: pl.DataFrame | None = None
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
            {}, {"_id": 0, "productid": 1, "imageid": 1, "designation": 1, "description": 1, "batch_id": 1, "is_gold": 1}
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
        batch_ids_ordered = []
        is_gold_ordered = []

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
            batch_ids_ordered.append(d.get("batch_id"))
            is_gold_ordered.append(d.get("is_gold", False))

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

        # 7. Assembler en DataFrame Polars : ids + label + embeddings + tabulaires + text
        new_data = {
            "productid": productids_ordered,
            "imageid": imageids_ordered,
            "label": labels_ordered,
            "batch_id": batch_ids_ordered,
            "is_gold": is_gold_ordered,
            "text": texts,  # texte nettoyé (déjà passé par clean_description plus haut)
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
        Charge le cache, applique les filtres (batches autorisés, gold exclu),
        et expose train_pool + ensembles d'évaluation (gold, shadow batches).

        Migration douce : si le cache parquet n'a pas batch_id/is_gold, on les
        rapatrie depuis Mongo (one-shot, écrit dans le parquet pour les runs suivants).

        Le split train/val interne (pour early stopping en M3, par exemple) est
        un split 90/10 du train_pool, utilisé par les DataLoaders Lightning.
        """
        if self.mode != "m2_embeddings":
            raise NotImplementedError(f"setup pour mode={self.mode} non encore supporté")

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Cache introuvable : {self.cache_path}. Lance prepare_data() d'abord."
            )

        df = pl.read_parquet(self.cache_path)

        # Migration douce : si batch_id ou is_gold manquent, on les rapatrie depuis Mongo
        missing_cols = [c for c in ("batch_id", "is_gold", "text") if c not in df.columns]
        if missing_cols:
            print(f"[DataModule] Cache sans {missing_cols}, migration auto depuis Mongo...")
            client = MongoClient(MONGO_URI)
            db = client[DB_NAME]
            pid_list = df.get_column("productid").to_list()
            # On demande à Mongo uniquement les champs dont on a besoin
            mongo_fields = {"_id": 0, "productid": 1}
            if "batch_id" in missing_cols:
                mongo_fields["batch_id"] = 1
            if "is_gold" in missing_cols:
                mongo_fields["is_gold"] = 1
            if "text" in missing_cols:
                mongo_fields["designation"] = 1
                mongo_fields["description"] = 1
            mongo_docs = db["X_raw_data_batches"].find(
                {"productid": {"$in": pid_list}}, mongo_fields,
            )
            mapping = {d["productid"]: d for d in mongo_docs}
            new_cols = []
            if "batch_id" in missing_cols:
                new_cols.append(
                    pl.Series("batch_id", [mapping.get(pid, {}).get("batch_id") for pid in pid_list])
                )
            if "is_gold" in missing_cols:
                new_cols.append(
                    pl.Series("is_gold", [mapping.get(pid, {}).get("is_gold", False) for pid in pid_list])
                )
            if "text" in missing_cols:
                def _build_text(pid):
                    d = mapping.get(pid, {})
                    designation = d.get("designation") or ""
                    description = d.get("description") or ""
                    full_text = f"{designation}. {description}" if description else designation
                    return clean_description(full_text)
                new_cols.append(
                    pl.Series("text", [_build_text(pid) for pid in pid_list])
                )
            df = df.with_columns(new_cols)
            df.write_parquet(self.cache_path)
            print(f"[DataModule] Cache enrichi : {self.cache_path}")

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
        self._df_full = df

        # Train pool : batches autorisés AND not gold (ceinture + bretelles)
        mask_pool = pl.col("batch_id").is_in(self.train_batches)
        if self.exclude_gold:
            mask_pool = mask_pool & (~pl.col("is_gold"))
        self._df_train_pool = df.filter(mask_pool)

        n_total = df.height
        n_pool = self._df_train_pool.height
        n_gold = df.filter(pl.col("is_gold")).height
        n_shadow = n_total - n_pool - n_gold
        print(
            f"[DataModule] Total={n_total} | train_pool={n_pool} "
            f"(batches={self.train_batches}, gold_excluded={self.exclude_gold}) | "
            f"gold={n_gold} | shadow={n_shadow}"
        )

        # Split train/val interne sur le pool (pour DataLoaders Lightning M3+)
        labels_pool = self._df_train_pool.get_column("label").to_numpy()
        indices_pool = np.arange(n_pool)
        idx_train, idx_val = train_test_split(
            indices_pool,
            test_size=self.val_size,
            stratify=labels_pool,
            random_state=self.random_state,
        )
        self._df_train = self._df_train_pool[idx_train.tolist()]
        self._df_val = self._df_train_pool[idx_val.tolist()]
        print(
            f"[DataModule] Split du train_pool : train={len(self._df_train)}, "
            f"val={len(self._df_val)} (val_size={self.val_size})"
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
        raise NotImplementedError(
            "Pas de test_dataloader local : utilise get_eval_data('gold') ou "
            "get_eval_data('shadow', batch_id=N) selon ton besoin d'évaluation."
        )

    # --- Interface principale (M2 sklearn + évaluations) ------------------
    def get_sklearn_data(
    self,
    split: Literal["train", "val"],
    include_raw: bool = False,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) du split interne 90/10 du train_pool.
        - "train" : 90% du pool, utilisé pour fit (en M2 le K-Fold consomme ça)
        - "val" : 10% du pool, utilisé pour diagnostics post-fit (calibration,
              confusion, stacking analysis). Ce n'est PAS du test métier.

        Args:
            include_raw: si True, ajoute les colonnes 'text', 'imageid', 'productid'
                     au DataFrame. Utile pour les BaseLearners non-frozen (TextCNN,
                     ResNet50PartialFT) qui ont besoin de ces données brutes.
                     Default False : comportement compatible avec M2 baseline.
        """
        if self._df_train is None or self._df_val is None:
            raise RuntimeError("setup() doit être appelé avant get_sklearn_data()")

        df_map = {"train": self._df_train, "val": self._df_val}
        if split not in df_map:
            raise ValueError(
                f"split={split} non supporté. Utiliser 'train' ou 'val' pour le "
                "split interne du pool, ou get_eval_data('gold'/'shadow') pour "
                "l'évaluation métier."
            )

        feat_cols = self._text_cols + self._image_cols + self._tabular_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = df_map[split].select(feat_cols)
        y = df_map[split].get_column("label").to_numpy()
        return X, y
    
    def get_train_pool(self, include_raw: bool = False) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) du pool d'entraînement : batches autorisés, gold exclu.
        Chaque algo gère son propre split interne (K-Fold pour M2, train/val pour M3).

        Args:
            include_raw: si True, ajoute 'text', 'imageid', 'productid' au DataFrame.
        """
        if self._df_train_pool is None:
            raise RuntimeError("setup() doit être appelé avant get_train_pool()")
        feat_cols = self._text_cols + self._image_cols + self._tabular_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = self._df_train_pool.select(feat_cols)
        y = self._df_train_pool.get_column("label").to_numpy()
        return X, y

    def get_eval_data(
    self,
    kind: Literal["gold", "shadow"],
    batch_id: int | None = None,
    include_raw: bool = False,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) pour évaluation.
        - "gold" : test set transverse (arbitre @champion)
        - "shadow" : batch hors train_batches (simule nouvelles données arrivées)

        Args:
            include_raw: si True, ajoute 'text', 'imageid', 'productid' au DataFrame.
        """
        if self._df_full is None:
            raise RuntimeError("setup() doit être appelé avant get_eval_data()")

        if kind == "gold":
            df_eval = self._df_full.filter(pl.col("is_gold"))
        elif kind == "shadow":
            if batch_id is None:
                raise ValueError("batch_id requis pour kind='shadow'")
            if batch_id in self.train_batches:
                raise ValueError(
                    f"batch_id={batch_id} est dans train_batches={self.train_batches}, "
                    "pas un shadow batch valide"
                )
            df_eval = self._df_full.filter(
                (~pl.col("is_gold")) & (pl.col("batch_id") == batch_id)
            )
        else:
            raise ValueError(f"kind doit être 'gold' ou 'shadow', pas '{kind}'")

        if df_eval.height == 0:
            raise ValueError(f"Ensemble eval ({kind}, batch_id={batch_id}) vide")

        feat_cols = self._text_cols + self._image_cols + self._tabular_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = df_eval.select(feat_cols)
        y = df_eval.get_column("label").to_numpy()
        print(f"[DataModule] get_eval_data({kind}, batch_id={batch_id}, include_raw={include_raw}) : n={len(y)}")
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

    @property
    def all_batch_ids(self) -> list[int]:
        """
        Liste des batch_id uniques (non-null) présents dans le dataset.
        Permet de dériver dynamiquement les shadow batches : all_batch_ids - train_batches.
        """
        if self._df_full is None:
            raise RuntimeError("setup() doit être appelé avant all_batch_ids")
        return sorted(
            self._df_full.get_column("batch_id").drop_nulls().unique().to_list()
        )