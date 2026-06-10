#!/usr/bin/env python
"""
Voie B — Termine le 7c interrompu : écrit `embeddings_siglip2_v1.parquet`
(+ push R2) SANS re-train ni toucher aux alias.

Contexte : le pod a été tué par le timeout PENDANT l'étape 7c
(_write_cache_parquet) de BaseLearnerExperiment. Les alias @active / @active_image
ont été posés (7a/7b) mais le cache parquet n'a jamais été écrit → m2_best casse
(garde-fou M.7 : cache introuvable / désync version).

Pourquoi une instance fraîche suffit (et reproduit v1 à l'identique) :
    v1 est FROZEN (lora_enabled=False). `extract_embeddings` = pooler_output du
    backbone SigLIP gelé, AVANT la tête. La tête entraînée n'intervient PAS dans
    l'extraction. Une instance fraîche (même model_name, même preprocessing,
    mêmes images) produit donc les mêmes embeddings que v1.
    ⚠ Si le learner était LoRA (lora_enabled=True), il faudrait charger les
    poids adaptés via Siglip2.from_pretrained(<artefacts @active>).

Prérequis :
    - GPU (sinon extraction CPU ~1h). Le script force net.to("cuda").
    - DataModule opérationnel : data/raw_data + images + cache principal présent.
    - Env : MLFLOW_TRACKING_URI (résolution version alias),
            ACTIVE_VAL_SELECTION_VERSION=1,
            R2_ENDPOINT_URL / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET
            (chargés depuis .env via load_dotenv), DATA_ROOT (défaut ".").

Usage :
    MLFLOW_TRACKING_URI=http://localhost:5000 ACTIVE_VAL_SELECTION_VERSION=1 \
        python scripts/complete_siglip2_cache.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
from src.models.base_learners.image.siglip2 import Siglip2

load_dotenv()

# --- Constantes (alignées sur le run v1 : cf. params MLflow chill-perch-521) ---
LEARNER_NAME = "siglip2"
REGISTERED_MODEL = "rakuten-base-siglip2"
ALIAS = "active_image"                       # alias que m2_best résout pour l'image
MODEL_NAME = "google/siglip2-base-patch16-224"
EMBED_DIM = 768
DATA_FOLDER = Path("data/raw_data")
IMAGE_FOLDER = DATA_FOLDER / "images" / "image_train"
NUM_WORKERS = 8                              # EPYC 64 cœurs : on ouvre le décodage JPEG


def resolve_source_version() -> int:
    """Version pointée par @active_image (= ce que le cache doit matcher, garde-fou M.7)."""
    try:
        mv = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL, ALIAS)
        v = int(mv.version)
        print(f"[voie_b] {REGISTERED_MODEL}@{ALIAS} -> v{v}")
        return v
    except Exception as e:
        print(f"[voie_b] WARN: résolution alias échouée ({type(e).__name__}: {e}), fallback v1")
        return 1


def push_to_r2(cache_path: Path, cache_filename: str) -> None:
    """Réplique exacte du push R2 de _write_cache_parquet (clé embedding_caches/...)."""
    try:
        import boto3

        endpoint = os.getenv("R2_ENDPOINT_URL")
        key = os.getenv("R2_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("R2_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
        bucket = os.getenv("R2_BUCKET", "rakuten-mlops-dvc")
        if not all([endpoint, key, secret]):
            print("[voie_b] WARN: creds R2 manquantes, skip push "
                  "(le parquet local suffit si tu lances m2_best en local).")
            return
        s3 = boto3.client(
            "s3", endpoint_url=endpoint,
            aws_access_key_id=key, aws_secret_access_key=secret,
        )
        r2_key_name = f"embedding_caches/{cache_filename}"
        s3.upload_file(str(cache_path), bucket, r2_key_name)
        print(f"[voie_b] ✓ Push R2 : s3://{bucket}/{r2_key_name}")
    except Exception as e:
        print(f"[voie_b] WARN: push R2 échoué (non bloquant) : {type(e).__name__}: {e}")


def main() -> None:
    val_sel_version = int(os.getenv("ACTIVE_VAL_SELECTION_VERSION", "1"))
    source_model_version = resolve_source_version()

    # 1. DataModule -> _df_full + get_full_data (couverture 100% : train + gold + shadow)
    dm = RakutenLightningDataModule(
        mode="raw_for_finetune",
        batch_size=64,
        num_workers=NUM_WORKERS,
        val_size=0.2,
        random_state=42,
        train_batches=[1],
        exclude_gold=True,
    )
    dm.setup()
    X_full, _ = dm.get_full_data(include_raw=True)
    print(f"[voie_b] X_full : {len(X_full)} samples (couverture complète)")

    # 2. Instance Siglip2 FROZEN (même config que v1) forcée sur GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[voie_b] WARN: pas de GPU -> extraction CPU LENTE (~1h). "
              "Lance ce script sur une machine GPU.")
    learner = Siglip2(
        image_folder=str(IMAGE_FOLDER),
        model_name=MODEL_NAME,
        embed_dim=EMBED_DIM,
        lora_enabled=False,
        num_workers=NUM_WORKERS,
    )
    learner.net = learner._build_net()
    learner.net.to(device)        # force GPU même sans le fix _forward_in_batches
    learner.net.eval()

    # 3. Extraction (backbone gelé -> embeddings identiques à v1)
    print("[voie_b] Extraction des embeddings (GPU)...")
    embeddings = learner.extract_embeddings(X_full)
    print(f"[voie_b] embeddings : {embeddings.shape}")
    assert embeddings.shape[1] == EMBED_DIM, (
        f"embed_dim attendu {EMBED_DIM}, obtenu {embeddings.shape[1]}"
    )

    # 4. Construire le cache (schéma IDENTIQUE à _write_cache_parquet)
    cache_data = {
        "productid": X_full["productid"].to_numpy(),
        "source_model_name": LEARNER_NAME,
        "source_model_version": source_model_version,
    }
    # Métadonnées depuis _df_full, ré-ordonnées comme X_full (join sur productid)
    all_pids = X_full["productid"].to_list()
    df_meta = dm._df_full.join(
        pl.DataFrame({"productid": all_pids, "_order": list(range(len(all_pids)))}),
        on="productid", how="inner",
    ).sort("_order")
    cache_data["batch_id"] = (
        df_meta["batch_id"].to_numpy() if "batch_id" in df_meta.columns
        else np.full(len(all_pids), 1, dtype=int)
    )
    for col in df_meta.columns:
        if col.startswith("is_val_selection_v"):
            cache_data[col] = df_meta[col].to_numpy()
    if "is_gold" in df_meta.columns:
        cache_data["is_gold"] = df_meta["is_gold"].to_numpy()
    for i in range(embeddings.shape[1]):
        cache_data[f"{LEARNER_NAME}_feat_{i}"] = embeddings[:, i]

    cache_df = pl.DataFrame(cache_data)
    print(f"[voie_b] cache shape : {cache_df.shape}")

    # 5. Write parquet (même dossier que celui lu par le DataModule)
    cache_dir = Path(os.getenv("DATA_ROOT", ".")) / "data/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = f"embeddings_{LEARNER_NAME}_v{val_sel_version}.parquet"
    cache_path = cache_dir / cache_filename
    cache_df.write_parquet(cache_path)
    print(f"[voie_b] ✓ Cache écrit : {cache_path}")

    # 6. Push R2 (backup hors volume persistant)
    push_to_r2(cache_path, cache_filename)

    print("[voie_b] Terminé. Vérifie avec :")
    print(f"  python -c \"import polars as pl; d=pl.read_parquet('{cache_path}'); "
          f"print(d.shape); print(d['source_model_version'].unique())\"")


if __name__ == "__main__":
    main()
