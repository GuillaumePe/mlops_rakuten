"""
I.1 — Action ingest_batch : valide et prépare un batch dans Mongo.

En mode simulation pédagogique, les batches sont pré-chargés dans Mongo
(X_raw_data_batches avec batch_id) par upload_to_mongodb.py.
Cette action "révèle" le batch n en :
    1. Vérifiant que le batch existe et est complet (labels, classes)
    2. Posant le flag is_gold sur chaque document (hash MD5 déterministe)
    3. Nettoyant le texte (designation + description → champ text)
    4. Loggant les stats dans MLflow

Mongo est la source unique de vérité pour les données brutes et les splits.

Appelable par :
    - PythonOperator dans le DAG Ingestion
    - runner.py --action ingest_batch --batch n (debug CLI)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

from src.data.gold_test import is_gold
from src.features.utils import clean_description

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "MAR25_CMLOPS_RAKUTEN"

VALID_CLASSES = {
    10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302,
    1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582,
    2583, 2585, 2705, 2905,
}


def run_ingest_batch(
    batch_id: int,
    mongo_uri: str = "",
    tracking_uri: str = "",
    **kwargs,
) -> dict:
    """
    Valide et prépare le batch batch_id dans Mongo.

    Args:
        batch_id: numéro du batch (1, 2, 3).
        mongo_uri: URI MongoDB.
        tracking_uri: URI MLflow.

    Returns:
        dict avec n_samples, n_gold, n_classes, etc.
    """
    mongo_uri = mongo_uri or MONGO_URI
    if not tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]

    col = db["X_raw_data_batches"]
    col.create_index("productid", unique=True)
    col.create_index("batch_id")
    # ------------------------------------------------------------ #
    # 1. Vérifier que le batch existe                               #
    # ------------------------------------------------------------ #
    n_batch = db["X_raw_data_batches"].count_documents({"batch_id": batch_id})
    if n_batch == 0:
        raise ValueError(
            f"Aucun sample pour batch_id={batch_id} dans X_raw_data_batches. "
            f"Lancer upload_to_mongodb.py d'abord."
        )
    logger.info(f"[ingest_batch] batch_id={batch_id} : {n_batch} samples")

    # ------------------------------------------------------------ #
    # 2. Vérifier les labels                                        #
    # ------------------------------------------------------------ #
    pids = [d["productid"] for d in db["X_raw_data_batches"].find(
        {"batch_id": batch_id}, {"productid": 1, "_id": 0}
    )]

    y_docs = {
        d["productid"]: d["prdtypecode"]
        for d in db["Y_raw_data_batches"].find(
            {"productid": {"$in": pids}},
            {"_id": 0, "productid": 1, "prdtypecode": 1},
        )
    }

    missing_labels = [pid for pid in pids if pid not in y_docs]
    if missing_labels:
        raise ValueError(f"{len(missing_labels)} productid sans label. Premiers : {missing_labels[:5]}")

    classes = set(y_docs.values())
    invalid = classes - VALID_CLASSES
    if invalid:
        raise ValueError(f"Classes invalides : {invalid}")

    # ------------------------------------------------------------ #
    # 3. Poser is_gold + text nettoyé sur chaque document           #
    # ------------------------------------------------------------ #
    x_docs = list(db["X_raw_data_batches"].find({"batch_id": batch_id}, {"_id": 0}))

    bulk_ops = []
    n_gold = 0
    for doc in x_docs:
        pid = doc["productid"]
        gold = is_gold(pid)
        if gold:
            n_gold += 1

        designation = doc.get("designation") or ""
        description = doc.get("description") or ""
        full_text = f"{designation}. {description}" if description.strip() else designation
        text = clean_description(full_text)

        bulk_ops.append(UpdateOne(
            {"productid": pid, "batch_id": batch_id},
            {"$set": {"is_gold": gold, "text": text}},
        ))

    # Chunked bulk_write pour éviter les timeouts Atlas
    CHUNK_SIZE = 5000
    total_modified = 0
    for i in range(0, len(bulk_ops), CHUNK_SIZE):
        chunk = bulk_ops[i:i + CHUNK_SIZE]
        result = db["X_raw_data_batches"].bulk_write(chunk)
        total_modified += result.modified_count
        logger.info(f"[ingest_batch] bulk_write chunk {i//CHUNK_SIZE + 1} : {result.modified_count} modifiés")
    logger.info(f"[ingest_batch] bulk_write total : {total_modified} modifiés")

    # ------------------------------------------------------------ #
    # 4. Log MLflow                                                 #
    # ------------------------------------------------------------ #
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ingestion")

    with mlflow.start_run(run_name=f"ingest_batch_{batch_id}"):
        mlflow.log_param("batch_id", batch_id)
        mlflow.log_param("action", "ingest_batch")
        mlflow.log_metric("n_samples", n_batch)
        mlflow.log_metric("n_gold", n_gold)
        mlflow.log_metric("n_classes", len(classes))

    summary = {
        "batch_id": batch_id,
        "n_samples": n_batch,
        "n_gold": n_gold,
        "n_classes": len(classes),
    }
    logger.info(f"[ingest_batch] Terminé : {summary}")
    return summary
