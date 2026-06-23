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
from src.data.mongo_utils import get_mongo_client, get_db
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import mlflow
from src.experiments.datamodule.encoders import TextEncoder, ImageEncoder, slugify_model_name
from src.experiments.datamodule.datasets import EmbeddingsDataset, MultimodalDataset
from src.experiments.datamodule.tabular_features import (
    extract_text_tabular,
    extract_image_tabular,
    TEXT_TABULAR_COLS,
    IMAGE_TABULAR_COLS,
)
from src.features.utils import clean_description  # legacy, à refondre en Bloc F
from src.models.utils import get_active_val_selection_version

import logging
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
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
        extra_embedding_caches: list[str] | None = None,
    ):
        super().__init__()
        self.mode = mode
        # Pour raw_for_finetune, text_model et image_model sont ignorés
        if mode != "raw_for_finetune":
            if text_model is None or image_model is None:
                raise ValueError(f"text_model et image_model requis pour mode={mode}")
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
        self.extra_embedding_caches = list(extra_embedding_caches or [])

        # Chemin du cache, déterministe à partir des modèles + version
        # (n'existe pas pour raw_for_finetune)
        if mode == "raw_for_finetune":
            self.cache_path = None
        else:
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
        self._extra_cols: list[str] = [] # colonnes d'embeddings des caches extra

        # Placeholders M.0 (val_selection versionné)
        self._df_train_pool_effective: pl.DataFrame | None = None
        self._df_val_selection: pl.DataFrame | None = None
        self._val_selection_version: int | None = None

        self._base_learner_caches: dict[str, pl.DataFrame] | None = None
    
    def set_m3_preprocessing(
        self,
        tokenizer,
        max_len: int,
        image_transform,
    ) -> None:
        """
        Configure le preprocessing multimodal pour M3.
 
        Appelé par le builder dans runner.py APRÈS chargement des base
        learners, AVANT setup(). Stocke les références — pas besoin de
        données pour cette étape.
 
        Args:
            tokenizer: tokenizer HuggingFace du text encoder.
            max_len: longueur max de tokenisation (300 pour CamemBERT LoRA).
            image_transform: torchvision transform de l'image encoder.
        """
        self._m3_tokenizer = tokenizer
        self._m3_max_len = max_len
        self._m3_image_transform = image_transform    
    
    # --- prepare_data : extraction + cache incrémental --------------------

    def prepare_data(self):
        """
        Garantit que le cache contient embeddings + tabulaires pour tous les
        productid actuellement en base. Calcule uniquement les manquants.
        """
        if self.mode == "raw_for_finetune":
            logger.info("[DataModule] prepare_data() ignoré pour mode=raw_for_finetune")
            return
        
        if self.mode != "m2_embeddings":
            raise NotImplementedError(
                f"prepare_data pour mode={self.mode} sera implémenté en Phase 1+"
            )

        db = get_db()
        

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
    def _setup_raw_for_finetune(self):
        """
        M.5 — Charge les données BRUTES directement de MongoDB (pas de cache parquet).
    
        Utilisé pour l'entraînement des base learners (TextCNN, ResNet50PartialFT).
    
        Colonnes exposées :
        - productid, imageid : pour identifier les samples
        - text : designation + description merged
        - label : classe cible
        - batch_id, is_gold : metadata pour filtrage
    
        Split :
        - train_pool : batches autorisés, gold exclu (tous les samples dispos)
        - val_selection : ~10% mis de côté pour arbitrer @active
        - train_pool_effective : train_pool - val_selection
        - train / val : split 80/20 stratifié sur train_pool_effective
        """
        logger.info("[DataModule._setup_raw_for_finetune] Chargement données brutes Mongo...")
    
        db = get_db()
    
        # 1. Charger X_raw_data_batches (texte + metadata)
        x_docs = list(db["X_raw_data_batches"].find(
            {},
            {
                "_id": 0,
                "productid": 1,
                "imageid": 1,
                "designation": 1,
                "description": 1,
                "batch_id": 1,
                "is_gold": 1,
            }
        ))
        if self.limit is not None:
            x_docs = x_docs[:self.limit]
    
        logger.info(f"[DataModule._setup_raw_for_finetune] {len(x_docs)} docs chargés depuis X_raw")
    
        # 2. Charger Y_raw_data_batches (labels)
        productid_list = [d["productid"] for d in x_docs]
        y_docs = list(db["Y_raw_data_batches"].find(
            {"productid": {"$in": productid_list}},
            {"_id": 0, "productid": 1, "prdtypecode": 1}
        ))
    
        labels_map = {d["productid"]: d["prdtypecode"] for d in y_docs}
        all_codes = sorted(
            set(d["prdtypecode"] for d in db["Y_raw_data_batches"].find({}, {"prdtypecode": 1, "_id": 0}))
        )
        self._code_to_idx = {code: idx for idx, code in enumerate(all_codes)}
        self._idx_to_code = {idx: code for code, idx in self._code_to_idx.items()}
    
        # 3. Assembler en DataFrame Polars
        data = {
            "productid": [],
            "imageid": [],
            "image_path": [],
            "text": [],
            "label": [],
            "batch_id": [],
            "is_gold": [],
        }
    
        for doc in x_docs:
            pid = doc["productid"]
            if pid not in labels_map:
                # Sample sans label : skip
                continue
        
            designation = doc.get("designation") or ""
            description = doc.get("description") or ""
            full_text = f"{designation}. {description}" if description else designation
        
            data["productid"].append(pid)
            data["imageid"].append(doc["imageid"])
            data["image_path"].append(str(IMAGE_FOLDER_TRAIN / f"image_{doc['imageid']}_product_{pid}.jpg"))
            data["text"].append(clean_description(full_text))
            data["label"].append(self._code_to_idx[labels_map[pid]])
            data["batch_id"].append(doc.get("batch_id"))
            data["is_gold"].append(doc.get("is_gold", False))
    
        self._df_full = pl.DataFrame(data)
        logger.info(f"[DataModule._setup_raw_for_finetune] DataFrame assemblé : {len(self._df_full)} samples")
    
        # 4. Résoudre val_selection version
        self._val_selection_version = get_active_val_selection_version()
        val_sel_col = f"is_val_selection_v{self._val_selection_version}"
    
        # Lire le val_selection versionné depuis Mongo
        val_sel_col = f"is_val_selection_v{self._val_selection_version}"
        pid_list = self._df_full.get_column("productid").to_list()
        val_sel_map = self._fetch_val_selection_from_mongo(pid_list)
        self._df_full = self._df_full.with_columns(
            pl.Series(val_sel_col, [val_sel_map.get(pid, False) for pid in pid_list])
        )
        val_sel_productids = {pid for pid, v in val_sel_map.items() if v}
        logger.info(
            f"[DataModule._setup_raw_for_finetune] {val_sel_col} : "
            f"{len(val_sel_productids)} val_selection depuis Mongo"
        )

    
        # 5. Appliquer les masks
        mask_pool = pl.col("batch_id").is_in(self.train_batches)
        if self.exclude_gold:
            mask_pool = mask_pool & (~pl.col("is_gold"))
    
        self._df_train_pool = self._df_full.filter(mask_pool)
    
        mask_val_selection = mask_pool & pl.col("productid").is_in(val_sel_productids)
        self._df_val_selection = self._df_full.filter(mask_val_selection)
    
        # train_pool_effective : pool - val_selection
        mask_pool_effective = mask_pool & (~pl.col("productid").is_in(val_sel_productids))
        self._df_train_pool_effective = self._df_full.filter(mask_pool_effective)
    
        # 6. Split train/val standard 80/20 sur train_pool_effective
        n_pool_eff = len(self._df_train_pool_effective)
        labels_pool_eff = self._df_train_pool_effective.get_column("label").to_numpy()
        indices_pool_eff = np.arange(n_pool_eff)
    
        idx_train, idx_val = train_test_split(
            indices_pool_eff,
            test_size=self.val_size,
            stratify=labels_pool_eff,
            random_state=self.random_state,
        )
    
        self._df_train = self._df_train_pool_effective[idx_train.tolist()]
        self._df_val = self._df_train_pool_effective[idx_val.tolist()]
    
        # 7. Exposer colonnes (raw_for_finetune n'a pas d'embeddings)
        self._text_cols = []
        self._image_cols = []
        self._tabular_cols = []
    
        # Log summary
        n_total = len(self._df_full)
        n_pool = len(self._df_train_pool)
        n_pool_eff = len(self._df_train_pool_effective)
        n_val_sel = len(self._df_val_selection)
        n_gold = len(self._df_full.filter(pl.col("is_gold")))
    
        logger.info(
            f"[DataModule._setup_raw_for_finetune] "
            f"Total={n_total} | train_pool={n_pool} | train_pool_effective={n_pool_eff} | "
            f"val_selection≈10%={n_val_sel} | gold={n_gold} | "
            f"train={len(self._df_train)} | val={len(self._df_val)}"
        )

    def _fetch_val_selection_from_mongo(self, pid_list: list) -> dict[int, bool]:
        """
        Lit is_val_selection_v{n} depuis Mongo.

        Returns:
            {productid: True/False} pour chaque pid dans pid_list.
        """
        val_sel_col = f"is_val_selection_v{self._val_selection_version}"
        db = get_db()
        docs = db["X_raw_data_batches"].find(
            {"productid": {"$in": pid_list}},
            {"_id": 0, "productid": 1, val_sel_col: 1},
        )
        return {d["productid"]: bool(d.get(val_sel_col, False)) for d in docs}


    def setup(self, stage: str | None = None):
        """
        Charge le cache, applique les filtres (batches autorisés, gold exclu),
        et expose train_pool + ensembles d'évaluation (gold, shadow batches).

        Migration douce : si le cache parquet n'a pas batch_id/is_gold, on les
        rapatrie depuis Mongo (one-shot, écrit dans le parquet pour les runs suivants).

        Le split train/val interne (pour early stopping en M3, par exemple) est
        un split 90/10 du train_pool, utilisé par les DataLoaders Lightning.
        """
        if self.mode == "raw_for_finetune":
            self._setup_raw_for_finetune()
            return
    
        if self.mode not in ("m2_embeddings"):
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
            db = get_db()
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

        # ─────────────────────────────────────────────────────────────────
        # M.7 — Jointure des caches extra base learners
        # ─────────────────────────────────────────────────────────────────
        # Chaque cache parquet (produit par fit_base_learner) est jointé
        # sur `productid` dans _df_full. Les colonnes d'embeddings se
        # retrouvent alors dans tous les splits dérivés (train, val, etc.)
        # → M2Assembled les consomme directement via _validate_columns().
        self._extra_cols = []
        if self.extra_embedding_caches:
            logger.info(
                f"[DataModule.M7] {len(self.extra_embedding_caches)} cache(s) extra à joindre"
            )
            for cache_filename in self.extra_embedding_caches:
                cache_path = CACHE_DIR / cache_filename
                if not cache_path.exists():
                    raise FileNotFoundError(
                        f"Cache extra introuvable : {cache_path}\n"
                        f"Vérifier que le base learner a été entraîné et promu @active,\n"
                        f"et que le parquet a été généré via fit_base_learner + DVC pull."
                    )
                df_extra = pl.read_parquet(cache_path)
                logger.info(
                    f"[DataModule.M7]   Chargé {cache_filename} : "
                    f"{df_extra.shape[0]} samples, {df_extra.shape[1]} colonnes"
                )
 
                # Identifier les colonnes d'embeddings (convention {name}_feat_{i})
                feat_cols_extra = [
                    c for c in df_extra.columns
                    if c.endswith(("_feat_0",)) or  # test rapide premier feat
                    ("_feat_" in c and c.split("_feat_")[-1].isdigit())
                ]
                if not feat_cols_extra:
                    raise ValueError(
                        f"Cache {cache_filename} : aucune colonne *_feat_* trouvée. "
                        f"Colonnes disponibles : {df_extra.columns[:10]}..."
                    )
 
                # Sélectionner uniquement productid + colonnes feat pour le join
                join_cols = ["productid"] + feat_cols_extra
                df_join = df_extra.select(
                    [c for c in join_cols if c in df_extra.columns]
                )
 
                # Inner join sur productid — les samples absents du cache
                # seront droppés (ex: gold si le cache n'a que train_pool).
                n_before = self._df_full.height
                self._df_full = self._df_full.join(
                    df_join, on="productid", how="inner"
                )
                n_after = self._df_full.height
                n_dropped = n_before - n_after
 
                self._extra_cols.extend(feat_cols_extra)
 
                logger.info(
                    f"[DataModule.M7]   Jointé {len(feat_cols_extra)} colonnes "
                    f"({feat_cols_extra[0]}..{feat_cols_extra[-1]}). "
                    f"Samples : {n_before} → {n_after} "
                    f"({n_dropped} droppés par inner join)"
                )
                if n_dropped > 0:
                    logger.warning(
                        f"[DataModule.M7]   ⚠ {n_dropped} samples droppés ! "
                        f"Le cache {cache_filename} ne couvre pas tous les productids "
                        f"du cache principal. Vérifier FIXME 2 dans base_learner_experiment.py."
                    )
 
            # Réassigner df pour la suite du setup (val_selection, splits, etc.)
            df = self._df_full
            logger.info(
                f"[DataModule.M7] Jointure terminée. "
                f"DataFrame final : {df.shape[0]} samples, {df.shape[1]} colonnes "
                f"(dont {len(self._extra_cols)} colonnes extra)"
            )


        # ─────────────────────────────────────────────────────────────────────
        # M.0 — Résolution du val_selection versionné
        # ─────────────────────────────────────────────────────────────────────
        val_sel_col = f"is_val_selection_v{self._val_selection_version}"
        if val_sel_col not in df.columns:
            logger.info(
                f"[DataModule] {val_sel_col} absente du cache, "
                f"migration depuis Mongo..."
            )
            pid_list = df.get_column("productid").to_list()
            val_sel_map = self._fetch_val_selection_from_mongo(pid_list)
            df = df.with_columns(
                pl.Series(val_sel_col, [val_sel_map.get(pid, False) for pid in pid_list])
            )
            df.write_parquet(self.cache_path)
            self._df_full = df
            n_val = sum(1 for v in val_sel_map.values() if v)
            logger.info(
                f"[DataModule] {val_sel_col} migrée depuis Mongo : "
                f"{n_val} val_selection. Cache mis à jour."
            )


        # Niveau 1 — train_pool : batches autorisés AND not gold (inchangé)
        # Utilisé par les assembled qui n'arbitrent pas @active (M2, M3, ...).
        mask_pool = pl.col("batch_id").is_in(self.train_batches)
        if self.exclude_gold:
            mask_pool = mask_pool & (~pl.col("is_gold"))
        self._df_train_pool = df.filter(mask_pool)

        # Niveau 2a — val_selection : ~10% mis de côté pour arbitrer @active.
        # Consommé uniquement par BaseLearnerExperiment.
        mask_val_selection = mask_pool & pl.col(val_sel_col).cast(pl.Boolean)
        self._df_val_selection = df.filter(mask_val_selection)

        # Niveau 2b — train_pool_effective : sur-ensemble du split 80/20 standard.
        # train_pool moins val_selection.
        mask_pool_effective = mask_pool & (~pl.col(val_sel_col).cast(pl.Boolean))
        self._df_train_pool_effective = df.filter(mask_pool_effective)

        n_total = df.height
        n_pool = self._df_train_pool.height
        n_pool_eff = self._df_train_pool_effective.height
        n_val_sel = self._df_val_selection.height
        n_gold = df.filter(pl.col("is_gold")).height
        n_shadow = n_total - n_pool - n_gold
        print(
            f"[DataModule] Total={n_total} | "
            f"train_pool={n_pool} (batches={self.train_batches}, gold_excluded={self.exclude_gold}) "
            f"| train_pool_effective={n_pool_eff} | val_selection_v{self._val_selection_version}={n_val_sel} "
            f"| gold={n_gold} | shadow={n_shadow}"
        )

        # Garde-fou : val_selection orthogonal à gold
        n_leak_gold = df.filter(pl.col(val_sel_col).cast(pl.Boolean) & pl.col("is_gold")).height
        if n_leak_gold != 0:
            raise RuntimeError(
                f"Bug critique : {n_leak_gold} samples sont à la fois val_selection_v"
                f"{self._val_selection_version} ET gold. Cache parquet corrompu."
            )

        # Niveau 3 — split train/val standard 80/20 stratifié sur train_pool_effective.
        # Sert : - aux BaseLearners deep (via X_val/y_val passés à fit())
        #        - aux DataLoaders Lightning (train_dataloader/val_dataloader)
        labels_pool_eff = self._df_train_pool_effective.get_column("label").to_numpy()
        indices_pool_eff = np.arange(n_pool_eff)
        idx_train, idx_val = train_test_split(
            indices_pool_eff,
            test_size=self.val_size,
            stratify=labels_pool_eff,
            random_state=self.random_state,
        )
        self._df_train = self._df_train_pool_effective[idx_train.tolist()]
        self._df_val = self._df_train_pool_effective[idx_val.tolist()]
        print(
            f"[DataModule] Split standard du train_pool_effective : "
            f"train={len(self._df_train)}, val={len(self._df_val)} (val_size={self.val_size})"
        )
        


    # --- Interface Lightning (DataLoaders pour M3/M4) ---------------------
    def _make_multimodal_dataset(
        self, df: pl.DataFrame, include_labels: bool = True
    ) -> "MultimodalDataset":
        """Construit un MultimodalDataset depuis un DataFrame interne."""
        return MultimodalDataset(
            texts=df["text"].to_list(),
            image_paths=df["image_path"].to_list(),
            labels=df["label"].to_numpy() if include_labels else None,
            tokenizer=self._m3_tokenizer,
            max_len=self._m3_max_len,
            image_transform=self._m3_image_transform,
        )
    
    def _make_loader(self, df: pl.DataFrame, shuffle: bool) -> DataLoader:
        if self.mode == "raw_for_finetune" and hasattr(self, "_m3_tokenizer"):
            ds = self._make_multimodal_dataset(df)
        else:
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
    

    def gold_dataloader(self) -> DataLoader:
        """
        DataLoader pour le gold test set.
 
        En mode M3 (set_m3_preprocessing appelé) : MultimodalDataset.
        Sinon : EmbeddingsDataset.
        """
        df_gold = self._df_full.filter(pl.col("is_gold"))
        if df_gold.height == 0:
            raise ValueError("Gold set vide.")
        return self._make_loader(df_gold, shuffle=False)
 
    def get_gold_labels(self) -> np.ndarray:
        """Labels du gold set, même ordre que gold_dataloader()."""
        df_gold = self._df_full.filter(pl.col("is_gold"))
        return df_gold["label"].to_numpy()
 
    def get_gold_productids(self) -> np.ndarray:
        """productid du gold, même ordre que gold_dataloader() et get_gold_labels()."""
        df_gold = self._df_full.filter(pl.col("is_gold"))
        return df_gold["productid"].to_numpy()
    
    # --- Interface principale (M2 sklearn + évaluations) ------------------
    def get_sklearn_data(
        self,
        split: Literal["train", "val", "train_pool", "train_pool_effective", "val_selection"],
        include_raw: bool = False,
    ) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) selon le split demandé.
 
        Splits standard d'entraînement (80/20 stratifié sur train_pool_effective) :
        - "train" : 80% pour fit (consommé par BaseLearner.fit(X_train, y_train, ...))
        - "val"   : 20% pour validation interne (early stopping, monitor)
 
        Splits "structurels" (sans tirage aléatoire) :
        - "train_pool" : tout sauf gold (assembled qui n'arbitrent pas @active)
        - "train_pool_effective" : tout sauf gold ni val_selection
        - "val_selection" : les ~10% mis de côté pour arbitrer les promotions @active
 
        Pour gold ou shadow batches, utiliser get_eval_data().
 
        Args:
            include_raw: si True, ajoute les colonnes 'text', 'imageid', 'productid'
                         au DataFrame. Utile pour les BaseLearners non-frozen
                         (TextCNN, ResNet50PartialFT, etc.).
        """
        if self._df_train is None or self._df_val is None:
            raise RuntimeError("setup() doit être appelé avant get_sklearn_data()")
        df_map = {
            "train": self._df_train,
            "val": self._df_val,
            "train_pool": self._df_train_pool,
            "train_pool_effective": self._df_train_pool_effective,
            "val_selection": self._df_val_selection,
        }
        if split not in df_map:
            raise ValueError(
                f"split={split!r} non supporté. Valides : {list(df_map.keys())}. "
                f"Pour gold ou shadow, utiliser get_eval_data()."
            )
        feat_cols = self._text_cols + self._image_cols + self._tabular_cols + self._extra_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = df_map[split].select(feat_cols)
        y = df_map[split].get_column("label").to_numpy()
        return X, y

    def _load_base_learner_embeddings(self, learner_name: str) -> pl.DataFrame:
        """
        M.7 — Charge le cache parquet d'un base learner avec validation guard-fou.
 
            Valide que le cache a été produit par la version @active courante du modèle.
 
        Args:
            learner_name: "textcnn", "resnet50_partial_ft", etc.
 
        Returns:
            DataFrame avec les colonnes d'embeddings du learner.
 
        Raises:
            FileNotFoundError: cache inexistant
            RuntimeError: désync entre cache et MLflow @active
        """
   
 
        # Résoudre le chemin du cache
        cache_filename = (
            f"embeddings_{learner_name}_"
            f"v{self._val_selection_version}.parquet"
        )
        cache_path = CACHE_DIR / cache_filename
 
        # 1. Vérifier que le cache existe
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache base learner introuvable : {cache_path}\n"
                f"Action requise :\n"
                f"    python -m src.experiments.runner "
                f"--experiment base_learner_{learner_name} "
                f"--action fit_base_learner"
            )
 
        # 2. Lire le cache
        df_cache = pl.read_parquet(cache_path)
        logger.debug(
            f"[DataModule.M7._load_base_learner_embeddings] "
            f"Cache chargé : {cache_path} ({len(df_cache)} samples)"
        )
 
        # 3. Vérifier les métadonnées de traçabilité
        if "source_model_name" not in df_cache.columns:
            raise ValueError(
                f"Cache {cache_filename} corrompu : colonne 'source_model_name' manquante"
            )
        if "source_model_version" not in df_cache.columns:
            raise ValueError(
                f"Cache {cache_filename} corrompu : colonne 'source_model_version' manquante"
            )
 
        # 4. Vérifier unicité
        unique_names = df_cache.get_column("source_model_name").unique().to_list()
        unique_versions = (
            df_cache.get_column("source_model_version").unique().to_list()
        )
 
        if len(unique_names) > 1:
            raise RuntimeError(
                f"Cache {cache_filename} hétérogène : "
                f"multiple source_model_name = {unique_names}"
            )
        if len(unique_versions) > 1:
            raise RuntimeError(
                f"Cache {cache_filename} hétérogène : "
                f"multiple source_model_version = {unique_versions}"
            )
 
        source_model_name = unique_names[0]
        source_model_version = int(unique_versions[0])
 
        # 5. Guard-fou : vérifier @active en MLflow
        client = mlflow.tracking.MlflowClient()
        try:
            active_mv = client.get_model_version_by_alias(source_model_name, "active")
            active_version = int(active_mv.version)
        except mlflow.exceptions.MlflowException as e:
            raise RuntimeError(
                f"Modèle {source_model_name} n'a pas d'alias @active en MLflow. "
                f"Guard-fou M.7 échoue.\n"
                f"Lancer fit_base_learner pour {learner_name} d'abord."
            ) from e
 
        # 6. Comparaison : version du cache vs @active courant
        if source_model_version != active_version:
            raise RuntimeError(
                f"DESYNC CRITIQUE — Cache vs MLflow @active :\n"
                f"Cache {cache_filename}:\n"
                f"source_model_name: {source_model_name}\n"
                f"source_model_version: {source_model_version}\n"
                f"MLflow @active:\n"
                f"{source_model_name} @active v{active_version}\n"
                f"\n"
                f"Action :\n"
                f"    python -m src.experiments.runner "
                f"--experiment base_learner_{learner_name} "
                f"--action fit_base_learner"
            )
 
        logger.debug(
            f"[DataModule.M7._load_base_learner_embeddings] "
            f"Guard-fou OK : {source_model_name} v{active_version} @active"
        )
 
        return df_cache

    def get_train_pool(self, include_raw: bool = False) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) du pool d'entraînement : batches autorisés, gold exclu.
        Chaque algo gère son propre split interne (K-Fold pour M2, train/val pour M3).

        Args:
            include_raw: si True, ajoute 'text', 'imageid', 'productid' au DataFrame.
        """
        if self._df_train_pool is None:
            raise RuntimeError("setup() doit être appelé avant get_train_pool()")
        feat_cols = self._text_cols + self._image_cols + self._tabular_cols + self._extra_cols
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

        feat_cols = self._text_cols + self._image_cols + self._tabular_cols + self._extra_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = df_eval.select(feat_cols)
        y = df_eval.get_column("label").to_numpy()
        print(f"[DataModule] get_eval_data({kind}, batch_id={batch_id}, include_raw={include_raw}) : n={len(y)}")
        return X, y

    def get_full_data(self, include_raw: bool = False) -> tuple[pl.DataFrame, np.ndarray]:
        """
        Retourne (X, y) sur l'INTÉGRALITÉ de _df_full (tous batches, gold inclus).
 
        Usage principal : BaseLearnerExperiment._write_cache_parquet() pour
        extraire les embeddings sur tous les productids, y compris gold et
        val_selection. Garantit que le cache parquet produit couvre 100% des
        samples → pas de perte au inner join dans le DataModule.
 
        ⚠ NE PAS utiliser pour le training (fuite gold → val → train).
        Uniquement pour l'extraction d'embeddings post-fit (model.eval()).
 
        Args:
            include_raw: si True, ajoute 'text', 'imageid', 'productid'.
        """
        if self._df_full is None:
            raise RuntimeError("setup() doit être appelé avant get_full_data()")
 
        feat_cols = self._text_cols + self._image_cols + self._tabular_cols + self._extra_cols
        if include_raw:
            feat_cols = feat_cols + ["text", "imageid", "productid"]
        X = self._df_full.select(feat_cols)
        y = self._df_full.get_column("label").to_numpy()
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
    


    