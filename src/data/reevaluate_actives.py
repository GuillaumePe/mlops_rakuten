"""
I.3 — Action reevaluate_actives : re-score les @active sur val_selection_v{n}.

Lit les samples val_selection_v{n}=True depuis Mongo, charge chaque base
learner @active, et calcule f1_weighted SANS ré-entraînement.

Appelable par :
    - POST /reevaluate_actives sur l'API FastAPI
    - runner.py --action reevaluate_actives --version n (debug CLI)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from dotenv import load_dotenv

import torch
from sklearn.metrics import f1_score

from src.features.utils import clean_description
from src.data.mongo_utils import get_db
from src.data.label_encoding import encode_labels
from src.models.utils import ensure_device

load_dotenv()

logger = logging.getLogger(__name__)


DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TRAIN = DATA_ROOT / "data/raw_data/images/image_train"

BASE_LEARNER_PREFIX = "rakuten-base-"


def run_reevaluate_actives(
    version: int,
    mongo_uri: str = "",
    tracking_uri: str = "",
    aliases: tuple[str, ...] = ("active", "active_stateless", "active_stateful"),
    **kwargs,
) -> dict:
    """
    Re-score les base learners actifs sur val_selection_v{version}.

   P.1d — multi-lignées : re-baseline TOUS les alias actifs, pas seulement
   @active. Sans ça, compute_promotion_decision(alias="active_stateful")
   comparerait le challenger du batch n+1 à une métrique calculée sur
   val_selection_v{n-1} → décision de promotion fausse.

   Les (name, version) sont dédupliqués : si @active et @active_stateless
   pointent sur la même version (pont legacy), un seul forward est fait.

   Args:
       aliases: alias à re-baseliner. Défaut : le nu (Phase 1) + les deux
           lignées (Phase 3). Un alias absent est simplement ignoré.
    """
    
    if not tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking_uri)
    client_mlflow = mlflow.tracking.MlflowClient()

    field_name = f"is_val_selection_v{version}"

    # ------------------------------------------------------------ #
    # 1. Charger val_selection depuis Mongo                         #
    # ------------------------------------------------------------ #
    db = get_db(uri=mongo_uri) if mongo_uri else get_db()

    n_val = db["X_raw_data_batches"].count_documents({field_name: True})
    if n_val == 0:
        raise RuntimeError(
            f"Aucun document avec {field_name}=True dans X_raw_data_batches. "
            f"Lancer rebase_val_selection --version {version} d'abord."
        )

    val_docs = list(db["X_raw_data_batches"].find(
        {field_name: True},
        {"_id": 0, "productid": 1, "imageid": 1, "text": 1,
         "designation": 1, "description": 1},
    ))

    # Labels
    pids = [d["productid"] for d in val_docs]
    y_map = {
        d["productid"]: d["prdtypecode"]
        for d in db["Y_raw_data_batches"].find(
            {"productid": {"$in": pids}},
            {"_id": 0, "productid": 1, "prdtypecode": 1},
        )
    }

    # Construire le DataFrame pour les base learners
    records = []
    labels = []
    for doc in val_docs:
        pid = doc["productid"]
        iid = doc.get("imageid", 0)

        # Utiliser le champ text pré-calculé par ingest_batch, sinon recalculer
        text = doc.get("text")
        if not text:
            designation = doc.get("designation") or ""
            description = doc.get("description") or ""
            full_text = f"{designation}. {description}" if description.strip() else designation
            text = clean_description(full_text)

        records.append({
            "productid": pid,
            "imageid": iid,
            "text": text,
            "image_path": str(IMAGE_FOLDER_TRAIN / f"image_{iid}_product_{pid}.jpg"),
        })
        labels.append(y_map.get(pid, -1))

    df_val = pl.DataFrame(records)
    y_val = np.array(labels)

    logger.info(f"[reevaluate_actives] val_selection_v{version} : {n_val} samples")

    # ------------------------------------------------------------ #
    # 2. Lister les base learners @active                           #
    # ------------------------------------------------------------ #
    active_learners = _list_active_learners(client_mlflow, aliases)
    if not active_learners:
        logger.warning(f"[reevaluate_actives] Aucun base learner pour {aliases}.")
        return {"version": version, "results": {}, "n_val": n_val}

    logger.info(
        f"[reevaluate_actives] {len(active_learners)} version(s) à re-scorer : "
        f"{[f'{a[chr(39)]}' for a in []] or [(a['name'], a['version'], a['aliases']) for a in active_learners]}"
    )

    # ------------------------------------------------------------ #
    # 3. Évaluer chaque @active                                     #
    # ------------------------------------------------------------ #
    results = {}
    metric_key = f"val_selection_v{version}/f1_weighted"

    for info in active_learners:
        name = info["name"]
        run_id = info["run_id"]
        model_version = info["version"]
        key = f"{name}:v{model_version}"

        try:
            f1 = _evaluate_learner(name, model_version, df_val, y_val)
            results[key] = {
                "f1_weighted": round(f1, 4),
                "version": model_version,
                "run_id": run_id,
                "aliases": info["aliases"],
            }

            # Log sur le run d'origine
            client_mlflow.log_metric(run_id, metric_key, f1)
            logger.info(
                f"  {name} v{model_version} @{info['aliases']} : {metric_key}={f1:.4f} "
                f"(loggé sur run {run_id[:8]}...)"
            )

        except Exception as e:
            logger.error(f"  {name} : échec — {e}")
            results[key] = {"error": str(e)}

    # ------------------------------------------------------------ #
    # 4. Log MLflow run récapitulatif                               #
    # ------------------------------------------------------------ #
    mlflow.set_experiment("ingestion")
    with mlflow.start_run(run_name=f"reevaluate_actives_v{version}"):
        mlflow.log_param("version", version)
        mlflow.log_param("n_val", n_val)
        mlflow.log_param("aliases_rebaselined", list(aliases))
        mlflow.log_param(
            "active_learners",
            [f"{a['name']}:v{a['version']}" for a in active_learners],
        )

        for key, res in results.items():
            if "f1_weighted" in res:
                safe_name = key.replace(BASE_LEARNER_PREFIX, "").replace(":", "_")
                mlflow.log_metric(f"{safe_name}/f1_weighted", res["f1_weighted"])

    # ------------------------------------------------------------ #
    # 5. Set ACTIVE_VAL_SELECTION_VERSION                           #
    # ------------------------------------------------------------ #
    os.environ["ACTIVE_VAL_SELECTION_VERSION"] = str(version)

    summary = {"version": version, "n_val": n_val, "results": results}
    logger.info(f"[reevaluate_actives] Terminé : {summary}")
    return summary


def _list_active_learners(
    client: mlflow.tracking.MlflowClient,
    aliases: tuple[str, ...] = ("active",),
) -> list[dict]:
    """
    Liste les (registered model, version) portant au moins un des `aliases`.

    Déduplique par (name, version) : si plusieurs alias pointent sur la même
    version (ex. @active et @active_stateless via le pont legacy), un seul
    forward sera fait. Les alias qui n'existent pas sont ignorés.

    Returns:
        [{"name", "version", "run_id", "aliases": [alias, ...]}, ...]
    """
    seen: dict[tuple[str, int], dict] = {}
    for rm in client.search_registered_models():
        if not rm.name.startswith(BASE_LEARNER_PREFIX):
            continue
        for alias in aliases:
            try:
                mv = client.get_model_version_by_alias(rm.name, alias)
            except Exception:
                continue  # cet alias n'existe pas sur ce modèle
            key = (rm.name, int(mv.version))
            if key in seen:
                seen[key]["aliases"].append(alias)  # même version, alias en plus
            else:
                seen[key] = {
                    "name": rm.name,
                    "version": int(mv.version),
                    "run_id": mv.run_id,
                    "aliases": [alias],
                }
    return list(seen.values())


def _evaluate_learner(
    registered_name: str,
    model_version: int,
    df_val: pl.DataFrame,
    y_val: np.ndarray,
) -> float:
    """
    Charge une VERSION précise, predict_proba sur df_val, retourne f1_weighted.

    Chargement par version (pas par alias) : en multi-lignées, un même learner
    porte des versions différentes selon la stratégie (@active_stateless → v5,
    @active_stateful → v9). L'alias seul serait ambigu.
    """
    uri = f"models:/{registered_name}/{model_version}"
    logger.info(f"  Chargement {uri}...")

    pyfunc = mlflow.pyfunc.load_model(uri)
    python_model = getattr(
        getattr(pyfunc, "_model_impl", None), "python_model", None
    )
    if python_model is None:
        raise RuntimeError(f"Pas de python_model pour {uri}")

    learner = getattr(python_model, "learner", None)
    
    if learner is None:
        raise RuntimeError(f"learner est None pour {uri}")
    ensure_device(learner)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(learner, "net") and learner.net is not None:
        learner.net.to(device)
        learner.net.eval()
        logger.info(f"    {registered_name} → {device}")

    y_pred_proba = learner.predict_proba(df_val)
    y_pred = y_pred_proba.argmax(axis=1)

    # y_val contient des prdtypecodes bruts (10, 40, ...) depuis Mongo
    # y_pred contient des indices (0-26) depuis predict_proba.argmax
    y_val_encoded = np.array(encode_labels(y_val))

    return f1_score(y_val_encoded, y_pred, average="weighted")
