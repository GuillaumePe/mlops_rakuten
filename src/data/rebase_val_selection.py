"""
I.2 — Action rebase_val_selection : crée is_val_selection_v{n} dans Mongo.

Lit les samples non-gold des batches 1..n depuis X_raw_data_batches,
applique un split stratifié 10% (seed=42), et pose le flag booléen
is_val_selection_v{n} directement sur chaque document Mongo.

Les versions antérieures (v1..v{n-1}) ne sont pas touchées.

Appelable par :
    - PythonOperator dans le DAG Ingestion
    - runner.py --action rebase_val_selection --version n (debug CLI)
"""
from __future__ import annotations

import logging
import os

import mlflow
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "MAR25_CMLOPS_RAKUTEN"

VAL_SELECTION_FRACTION = 0.10
VAL_SELECTION_SEED = 42


def run_rebase_val_selection(
    version: int,
    mongo_uri: str = "",
    tracking_uri: str = "",
    **kwargs,
) -> dict:
    """
    Crée le flag is_val_selection_v{version} dans X_raw_data_batches.

    Args:
        version: numéro de version (= dernier batch ingéré).
        mongo_uri: URI MongoDB.
        tracking_uri: URI MLflow.

    Returns:
        dict avec stats du split.
    """
    mongo_uri = mongo_uri or MONGO_URI
    if not tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    client = MongoClient(mongo_uri)
    db = client[DB_NAME]
    col = db["X_raw_data_batches"]
    field_name = f"is_val_selection_v{version}"

    # ------------------------------------------------------------ #
    # 1. Charger le sur-ensemble : batch_1..n, non-gold             #
    # ------------------------------------------------------------ #
    super_set_filter = {
        "batch_id": {"$lte": version},
        "is_gold": False,
    }
    # Vérifier que is_gold a bien été posé (par ingest_batch)
    n_missing_gold = col.count_documents({
        "batch_id": {"$lte": version},
        "is_gold": {"$exists": False},
    })
    if n_missing_gold > 0:
        raise RuntimeError(
            f"{n_missing_gold} docs sans champ is_gold dans batch_id <= {version}. "
            f"Lancer ingest_batch pour chaque batch d'abord."
        )

    docs = list(col.find(
        super_set_filter,
        {"_id": 0, "productid": 1, "prdtypecode": 1},
    ))

    # Si prdtypecode n'est pas dans X_raw, le lire depuis Y_raw
    if docs and "prdtypecode" not in docs[0]:
        pids = [d["productid"] for d in docs]
        y_map = {
            d["productid"]: d["prdtypecode"]
            for d in db["Y_raw_data_batches"].find(
                {"productid": {"$in": pids}},
                {"_id": 0, "productid": 1, "prdtypecode": 1},
            )
        }
        for d in docs:
            d["prdtypecode"] = y_map.get(d["productid"])

    n_super = len(docs)
    if n_super == 0:
        raise RuntimeError(f"Sur-ensemble vide pour v{version}.")

    logger.info(f"[rebase_val_selection] Sur-ensemble v{version} : {n_super} samples")

    # ------------------------------------------------------------ #
    # 2. Split stratifié 10%                                        #
    # ------------------------------------------------------------ #
    productids = np.array([d["productid"] for d in docs])
    labels = np.array([d["prdtypecode"] for d in docs])

    _, idx_val = train_test_split(
        np.arange(n_super),
        test_size=VAL_SELECTION_FRACTION,
        stratify=labels,
        random_state=VAL_SELECTION_SEED,
    )
    pids_val = set(productids[idx_val].tolist())
    n_val = len(pids_val)

    logger.info(
        f"[rebase_val_selection] Split seed={VAL_SELECTION_SEED} : "
        f"val={n_val}, train_residuel={n_super - n_val}"
    )

    # ------------------------------------------------------------ #
    # 3. Écrire le flag dans Mongo (bulk update)                    #
    # ------------------------------------------------------------ #
    # D'abord remettre tout à False (idempotence)
    col.update_many(
        {"batch_id": {"$lte": version}},
        {"$set": {field_name: False}},
    )

    # Puis True pour les val_selection
    bulk_ops = [
        UpdateOne(
            {"productid": pid},
            {"$set": {field_name: True}},
        )
        for pid in pids_val
    ]
    CHUNK_SIZE = 5000
    total_modified = 0
    for i in range(0, len(bulk_ops), CHUNK_SIZE):
        chunk = bulk_ops[i:i + CHUNK_SIZE]
        result = db["X_raw_data_batches"].bulk_write(chunk)
        total_modified += result.modified_count
        logger.info(f"[ingest_batch] bulk_write chunk {i//CHUNK_SIZE + 1} : {result.modified_count} modifiés")
    logger.info(f"[ingest_batch] bulk_write total : {total_modified} modifiés")

    # ------------------------------------------------------------ #
    # 4. Sanity checks                                              #
    # ------------------------------------------------------------ #
    # Vérifier orthogonalité gold ↔ val_selection
    n_overlap = col.count_documents({
        field_name: True,
        "is_gold": True,
    })
    if n_overlap != 0:
        raise RuntimeError(f"Bug : {n_overlap} samples gold ET val_selection_v{version}")

    # Vérifier le count
    n_val_check = col.count_documents({field_name: True})
    assert n_val_check == n_val, f"Incohérence : {n_val_check} vs {n_val} attendus"

    # Vérifier que les versions antérieures sont intactes
    for v in range(1, version):
        prev_field = f"is_val_selection_v{v}"
        n_prev = col.count_documents({prev_field: True})
        if n_prev > 0:
            logger.info(f"  {prev_field} : {n_prev} docs (inchangé)")

    logger.info("[rebase_val_selection] Sanity checks OK")

    # ------------------------------------------------------------ #
    # 5. Log MLflow                                                 #
    # ------------------------------------------------------------ #
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ingestion")

    with mlflow.start_run(run_name=f"rebase_val_selection_v{version}"):
        mlflow.log_param("version", version)
        mlflow.log_param("seed", VAL_SELECTION_SEED)
        mlflow.log_param("fraction", VAL_SELECTION_FRACTION)
        mlflow.log_param("super_set_batches", list(range(1, version + 1)))
        mlflow.log_metric("super_set_size", n_super)
        mlflow.log_metric("val_selection_size", n_val)
        mlflow.log_metric("train_residuel_size", n_super - n_val)

    summary = {
        "version": version,
        "super_set_size": n_super,
        "val_selection_size": n_val,
        "train_residuel_size": n_super - n_val,
    }
    logger.info(f"[rebase_val_selection] Terminé : {summary}")
    return summary
