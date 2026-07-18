"""
T.3c — Action eval_gold_champion : re-score le @champion sur le gold set COURANT.

Forward-only (pas de ré-entraînement) du modèle servi (@champion) sur l'intégralité
du gold courant, et log de eval_gold/f1_weighted dans un run MLflow dédié.

Raison d'être : le gold grandit à chaque batch (hash md5). La F1 stockée du champion
a été mesurée sur un gold plus ancien → incomparable aux challengers évalués sur le
gold courant. Cette action ré-évalue le champion sur le MÊME gold que les challengers,
pour une comparaison iso dans compare_and_promote (T.3d).

Réutilise RakutenScorer (P.1) : dispatch automatique M2/M3.2 par model_family.

Appelable par :
    - BashOperator(submit_cloud) / PythonOperator dans le DAG Training
    - runner.py --action eval_gold_champion --experiment <name> [--batch n]
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import mlflow
import polars as pl
from dotenv import load_dotenv
from sklearn.metrics import f1_score

from src.data.mongo_utils import get_db
from src.data.label_encoding import encode_labels
from src.models.rakuten_scorer import RakutenScorer

load_dotenv()

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TRAIN = DATA_ROOT / "data/raw_data/images/image_train"


def run_eval_gold_champion(
    model_name: str,
    batch_id: int | None = None,
    mongo_uri: str = "",
    tracking_uri: str = "",
    experiment_name: str = "training_compare",
    champion_alias: str = "champion",
    **kwargs,
) -> dict:
    """
    Re-score le @champion de `model_name` sur le gold set courant.

    Args:
        model_name: registered model name (ex: "rakuten-m3-2-coadaptation").
        batch_id: numéro de batch courant (nommage du run). Optionnel.
        mongo_uri / tracking_uri: overridables, sinon env vars.
        experiment_name: experiment MLflow où logger — DOIT être le même que
            les challengers pour que compare_and_promote retrouve tout.

    Returns:
        dict : model_name, model_version, model_family, f1_weighted, n_gold, run_id.
    """
    if not model_name:
        raise ValueError(
            "model_name requis (résolu depuis promotion.registry_model_name)."
        )

    if not tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # 1. Charger le gold COURANT depuis Mongo (is_gold=True)
    db = get_db(uri=mongo_uri) if mongo_uri else get_db()
    n_gold_docs = db["X_raw_data_batches"].count_documents({"is_gold": True})
    if n_gold_docs == 0:
        raise RuntimeError(
            "Aucun document is_gold=True dans X_raw_data_batches. "
            "Lancer ingest_batch d'abord."
        )
    gold_docs = list(db["X_raw_data_batches"].find(
        {"is_gold": True},
        {"_id": 0, "productid": 1, "imageid": 1, "designation": 1, "description": 1},
    ))

    # 2. Labels gold (prdtypecode), garder uniquement les docs labellisés
    pids = [d["productid"] for d in gold_docs]
    y_map = {
        d["productid"]: d["prdtypecode"]
        for d in db["Y_raw_data_batches"].find(
            {"productid": {"$in": pids}},
            {"_id": 0, "productid": 1, "prdtypecode": 1},
        )
    }
    gold_docs = [d for d in gold_docs if d["productid"] in y_map]
    if not gold_docs:
        raise RuntimeError("Aucun sample gold labellisé dans Y_raw_data_batches.")

    # 3. raw_df attendu par RakutenScorer (schéma brut + image_path résolu)
    raw_df = pl.DataFrame([
        {
            "productid": d["productid"],
            "designation": d.get("designation", "") or "",
            "description": d.get("description", "") or "",
            "imageid": d.get("imageid", 0),
            "image_path": str(
                IMAGE_FOLDER_TRAIN
                / f"image_{d.get('imageid', 0)}_product_{d['productid']}.jpg"
            ),
        }
        for d in gold_docs
    ])

    # 4. Labels encodés dans le MÊME ordre que raw_df (alignement predictions↔labels)
    y_codes = [y_map[d["productid"]] for d in gold_docs]
    y_gold = encode_labels(y_codes)  # → indices 0-26

    # 5. Forward via RakutenScorer (dispatch M2/M3.2 par model_family, ordre préservé)
    scorer = RakutenScorer.from_champion(model_name, tracking_uri=tracking_uri, alias=champion_alias)
    result = scorer.score(raw_df)

    # 6. F1 weighted (predictions = indices, y_gold = indices → comparable)
    f1 = float(f1_score(y_gold, result.predictions, average="weighted"))
    logger.info(
        f"[eval_gold_champion] {result.model_name}@v{result.model_version} "
        f"({result.model_family}) : eval_gold/f1_weighted={f1:.4f} sur n={len(y_gold)}"
    )

    # 7. Log MLflow (run dédié, findable par compare_and_promote)
    mlflow.set_experiment(experiment_name)
    run_name = (
        f"eval_gold_champion_b{batch_id}" if batch_id is not None
        else "eval_gold_champion"
    )
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.set_tag("role", "champion_regold")
        mlflow.set_tag("model_family", result.model_family)
        mlflow.set_tag("rescored_model", f"{result.model_name}@{champion_alias}")
        mlflow.set_tag("champion_alias", champion_alias)
        mlflow.log_param("champion_version", result.model_version)
        mlflow.log_param("n_gold", len(y_gold))
        mlflow.log_metric("eval_gold/f1_weighted", f1)

    return {
        "model_name": result.model_name,
        "model_version": result.model_version,
        "model_family": result.model_family,
        "f1_weighted": f1,
        "n_gold": len(y_gold),
        "run_id": run_id,
        "run_name": run_name,
        "experiment_name": experiment_name,
    }