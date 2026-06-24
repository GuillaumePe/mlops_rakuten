"""
P.2 — Action predict_pending : score les samples en attente dans X_to_predict.

Remplace le chemin legacy (DistilBERT features + pca_lgbm_pipeline) par
RakutenScorer.from_champion() qui dispatche sur model_family.

Appelable par :
- PythonOperator dans le DAG Predict
- runner.py --action predict_pending (debug CLI)

Flow :
    1. Lire la file X_to_predict dans Mongo
    2. Vérifier le seuil (no-op si queue trop petite)
    3. Capturer les productid au début (protection race condition)
    4. Construire le raw_df (designation, description, image_path)
    5. Scorer via RakutenScorer.from_champion
    6. Écrire les résultats dans Prediction (avec model@version)
    7. Supprimer les samples scorés de X_to_predict ($in ciblé)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from src.data.mongo_utils import get_db
from src.models.rakuten_scorer import RakutenScorer
from src.data.label_encoding import decode_labels
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Constantes                                                          #
# ------------------------------------------------------------------ #


DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TEST = DATA_ROOT / "data/raw_data_test/images_test"

# Seuil par défaut — écrasable via Airflow Variable ou kwarg
DEFAULT_THRESHOLD = 50

# Modèle champion par défaut (registered model name dans MLflow)
DEFAULT_MODEL_NAME = "rakuten-m2-best"


def run_predict_pending(
    threshold: int = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
    mongo_uri: str = "",
    **kwargs,
) -> dict:
    """
    Score les samples en attente dans X_to_predict.

    Args:
        threshold: nombre minimum de samples pour déclencher le scoring.
            Si la queue est plus petite, no-op.
        model_name: nom du registered model MLflow à utiliser.
        mongo_uri: URI MongoDB. Défaut: MONGO_URI env var.
        **kwargs: ignoré (PythonOperator passe context, etc.)

    Returns:
        dict avec message, nb scored, model@version, etc.
    """
    db = get_db(uri=mongo_uri) if mongo_uri else get_db()

    # ------------------------------------------------------------ #
    # 1. Vérifier le seuil                                          #
    # ------------------------------------------------------------ #
    queue_size = db["X_to_predict"].count_documents({})
    logger.info(f"[predict_pending] Queue size: {queue_size}, threshold: {threshold}")

    if queue_size < threshold:
        msg = f"Queue trop petite ({queue_size} < {threshold}), no-op."
        logger.info(f"[predict_pending] {msg}")
        return {
            "message": msg,
            "queue_size": queue_size,
            "threshold": threshold,
            "scored": 0,
        }

    # ------------------------------------------------------------ #
    # 2. Capturer les productid à scorer (snapshot avant scoring)   #
    # ------------------------------------------------------------ #
    pending_docs = list(db["X_to_predict"].find(
        {},
        {"_id": 0, "productid": 1, "designation": 1, "description": 1, "imageid": 1},
    ))
    pending_ids = [d["productid"] for d in pending_docs]
    logger.info(f"[predict_pending] {len(pending_ids)} samples à scorer")

    # ------------------------------------------------------------ #
    # 3. Construire le raw_df                                       #
    # ------------------------------------------------------------ #
    raw_df = pl.DataFrame([
        {
            "productid": d["productid"],
            "designation": d.get("designation", "") or "",
            "description": d.get("description", "") or "",
            "imageid": d.get("imageid", 0),
            "image_path": str(
                IMAGE_FOLDER_TEST
                / f"image_{d.get('imageid', 0)}_product_{d['productid']}.jpg"
            ),
        }
        for d in pending_docs
    ])

    # ------------------------------------------------------------ #
    # 4. Scorer                                                     #
    # ------------------------------------------------------------ #
    scorer = RakutenScorer.from_champion(model_name)
    result = scorer.score(raw_df)

    logger.info(
        f"[predict_pending] Scoré {result.n_scored} samples "
        f"avec {result.model_name}@v{result.model_version} ({result.model_family})"
    )

    # ------------------------------------------------------------ #
    # 5. Écrire dans Prediction                                     #
    # ------------------------------------------------------------ #
    now = datetime.now().isoformat()
    model_tag = f"{result.model_name}@v{result.model_version}"

    prediction_records = []
    for i, row in enumerate(raw_df.iter_rows(named=True)):
        prediction_records.append({
            "productid": row["productid"],
            "designation": row["designation"],
            "imageid": row["imageid"],
            "prediction": int(decode_labels([result.predictions[i]])[0]),
            "confidence": float(result.probas[i].max()),
            "date_pred": now,
            "model": model_tag,
            "model_family": result.model_family,
        })

    if prediction_records:
        db["Prediction"].insert_many(prediction_records)
        logger.info(f"[predict_pending] {len(prediction_records)} prédictions écrites dans Prediction")

    # ------------------------------------------------------------ #
    # 6. Purger la queue (suppression ciblée)                       #
    # ------------------------------------------------------------ #
    delete_result = db["X_to_predict"].delete_many(
        {"productid": {"$in": pending_ids}}
    )
    logger.info(f"[predict_pending] {delete_result.deleted_count} samples retirés de X_to_predict")

    return {
        "message": f"{len(prediction_records)} prédictions faites.",
        "scored": len(prediction_records),
        "model": model_tag,
        "model_family": result.model_family,
        "timestamp": now,
        "queue_size_before": queue_size,
        "deleted_from_queue": delete_result.deleted_count,
    }
