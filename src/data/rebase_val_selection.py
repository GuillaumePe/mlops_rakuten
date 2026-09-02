"""
I.2 — Action rebase_val_selection : crée is_val_selection_v{n} dans Mongo.

CONSTRUCTION MONOTONE [ADR-003] :

    v_n = v_{n-1} ∪ split_stratifié_10%(batch_n \\ gold)

Le split porte UNIQUEMENT sur le batch n ; les productids de v_{n-1} sont
hérités tels quels. Un produit entré dans val_selection n'en sort jamais.

⚠ POURQUOI (défaut corrigé le 2026-09-01). L'implémentation d'origine faisait
un train_test_split GLOBAL sur les batches 1..n. Même graine, mais population
différente à chaque appel → tirage entièrement renouvelé : v2 ∩ v3 = 673 sur
5728. Conséquence : 5031 produits qui étaient dans le TRAIN au batch 2
basculaient en validation au batch 3. Les modèles titulaires les avaient
mémorisés, les challengers non → la porte de promotion comparait un score
gonflé à un score honnête. Preuve : textcnn v3 (train_batches=[1,2]) obtient
0.8127 sur v2 et 0.9219 sur v3 — modèle identique, +10.9 pts, ~20σ. Aucun base
learner n'a pu être promu depuis le batch 1.

La monotonie garantit que la part batches 1..n-1 de v_n est identique à
v_{n-1} — un ensemble que le DataModule exclut du train pool depuis toujours,
donc jamais entraîné par personne. C'est ce qui rend la comparaison
titulaire/challenger valide.

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
    # 1. Charger le sur-ensemble : batch n SEUL, non-gold           #
    #    (et non 1..n : c'est le cœur du correctif ADR-003)         #
    # ------------------------------------------------------------ #
    super_set_filter = {
        "batch_id": version,
        "is_gold": False,
    }

    # Héritage : les productids de v_{n-1}, repris tels quels.
    # Dépendance DURE et volontaire : sans v_{n-1}, on ne peut pas construire
    # v_n de façon monotone. On lève plutôt que de retomber sur un tirage
    # global — ce fallback silencieux est exactement le défaut corrigé ici.
    inherited: set[int] = set()
    if version > 1:
        prev_field = f"is_val_selection_v{version - 1}"
        inherited = {
            d["productid"]
            for d in col.find({prev_field: True}, {"_id": 0, "productid": 1})
        }
        if not inherited:
            raise RuntimeError(
                f"{prev_field} vide ou absent : impossible de construire "
                f"v{version} par union monotone. Lancer rebase_val_selection "
                f"pour les versions antérieures d'abord."
            )
        logger.info(
            f"[rebase_val_selection] Hérité de v{version - 1} : "
            f"{len(inherited)} productids"
        )
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
    # 2. Split stratifié 10% SUR LE BATCH n, puis union             #
    # ------------------------------------------------------------ #
    productids = np.array([d["productid"] for d in docs])
    labels = np.array([d["prdtypecode"] for d in docs])

    # Garde-fou : la stratification devient LOCALE au batch. Sur un batch
    # petit ou déséquilibré, une classe rare peut avoir < 2 individus —
    # train_test_split lève alors un message obscur, ou rend 0 échantillon de
    # validation pour cette classe. Le tirage global masquait ce risque par
    # mutualisation. Échouer bruyamment ici dit que le découpage en batches
    # est trop fin pour un held-out à 10 %.
    uniq, counts = np.unique(labels, return_counts=True)
    too_rare = [(int(c), int(k)) for c, k in zip(uniq, counts) if k < 2]
    if too_rare:
        raise RuntimeError(
            f"Batch {version} : {len(too_rare)} classe(s) avec < 2 individus "
            f"{too_rare[:5]} — stratification impossible. Le découpage en "
            f"batches est trop fin pour un held-out à "
            f"{VAL_SELECTION_FRACTION:.0%}."
        )

    _, idx_val = train_test_split(
        np.arange(n_super),
        test_size=VAL_SELECTION_FRACTION,
        stratify=labels,
        random_state=VAL_SELECTION_SEED,
    )
    pids_new = set(productids[idx_val].tolist())

    # UNION monotone — l'invariant de l'ADR-003.
    pids_val = inherited | pids_new
    n_val = len(pids_val)

    logger.info(
        f"[rebase_val_selection] Split seed={VAL_SELECTION_SEED} sur batch "
        f"{version} ({n_super} samples) : +{len(pids_new)} nouveaux, "
        f"{len(inherited)} hérités → v{version} = {n_val} productids"
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

    # INVARIANT STRUCTURANT [ADR-003] : v_{n-1} ⊆ v_n.
    # C'est cette propriété, et elle seule, qui garantit qu'aucun produit ne
    # retourne du held-out vers le train pool — donc que titulaire et
    # challenger sont comparés sur des données qu'aucun des deux n'a vues.
    if version > 1:
        prev_field = f"is_val_selection_v{version - 1}"
        n_lost = col.count_documents({prev_field: True, field_name: False})
        if n_lost != 0:
            raise RuntimeError(
                f"MONOTONIE VIOLÉE : {n_lost} produits de {prev_field} ne sont "
                f"pas dans {field_name}. Ces produits retourneraient au train "
                f"pool et fuiteraient dans l'évaluation du prochain cycle."
            )
        logger.info(
            f"[rebase_val_selection] Invariant v{version - 1} ⊆ v{version} : OK"
        )

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
        mlflow.log_param("split_batch", version)
        mlflow.log_param("construction", "monotone_union")
        mlflow.log_metric("super_set_size", n_super)
        mlflow.log_metric("val_selection_size", n_val)
        mlflow.log_metric("inherited_size", len(inherited))
        mlflow.log_metric("new_size", len(pids_new))
        mlflow.log_metric("train_residuel_size", n_super - len(pids_new))

    summary = {
        "version": version,
        "super_set_size": n_super,
        "val_selection_size": n_val,
        "inherited_size": len(inherited),
        "new_size": len(pids_new),
        "train_residuel_size": n_super - len(pids_new),
    }
    logger.info(f"[rebase_val_selection] Terminé : {summary}")
    return summary
