"""
P4.5a — Scoring complet de X_test pour soumission concours (ENS challenge 35).

Score la TOTALITÉ de X_to_predict_pool (le jeu de test Rakuten, immuable) avec
le modèle @production courant, et écrit les prédictions dans Prediction_test.

Diffère de predict_pending (superséded) :
  - source = X_to_predict_pool (tout X_test), PAS la file X_to_predict
  - pas de seuil, pas de purge de la source (le pool est immuable)
  - sortie dans une collection dédiée Prediction_test
  - idempotent par batch : re-scorer un batch REMPLACE ses lignes

Le forward tourne sur GPU cloud (doctrine + artefact M3 CUDA-natif). model_name
reste None → résolution @production SUR LE POD (MLflow via Tailscale).

Les prédictions stockées sont les prdtypecode Rakuten DÉCODÉS (pas les labels
internes 0-26), prêts pour l'assemblage de soumission (build_submission, P4.5c).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from src.data.mongo_utils import get_db
from src.models.rakuten_scorer import RakutenScorer
from src.data.label_encoding import decode_labels
from src.models.utils import resolve_production_model
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TEST = DATA_ROOT / "data/raw_data_test/images_test"

POOL_COLLECTION = "X_to_predict_pool"
OUTPUT_COLLECTION = "Prediction_test"


def run_predict_test_pool(
    model_name: str | None = None,
    mongo_uri: str = "",
    batch_id: int | None = None,
    limit: int | None = None,
    **kwargs,
) -> dict:
    """
    Score tout X_to_predict_pool avec @production, écrit dans Prediction_test.

    Args:
        model_name: registered model à scorer. None (défaut) → résolution du
            @production (vainqueur du tournament, version épinglée). Chaîne
            explicite → chemin debug (alias @champion du modèle nommé).
        mongo_uri: URI MongoDB. Défaut: MONGO_URI env var.
        batch_id: batch venant d'être entraîné, estampillé sur chaque prédiction
            ET clé d'idempotence (re-scorer ce batch remplace ses lignes).
        limit: si fourni, ne score que les N premiers documents du pool.
            Sert au sanity local et au premier run pod (test avant les 13812).
            None (défaut) → tout le pool.
        **kwargs: ignoré (PythonOperator passe context, etc.)

    Returns:
        dict : message, scored, model@version, model_family, alias_source,
            batch_id, collection, timestamp.
    """
    db = get_db(uri=mongo_uri) if mongo_uri else get_db()

    # ------------------------------------------------------------ #
    # 1. Charger le pool (X_test complet, immuable)                #
    # ------------------------------------------------------------ #
    cursor = db[POOL_COLLECTION].find(
        {},
        {"_id": 0, "productid": 1, "designation": 1, "description": 1, "imageid": 1},
    )
    if limit is not None:
        cursor = cursor.limit(int(limit))
    pool_docs = list(cursor)

    if not pool_docs:
        msg = f"{POOL_COLLECTION} vide, no-op."
        logger.warning(f"[predict_test_pool] {msg}")
        return {"message": msg, "scored": 0, "batch_id": batch_id}

    logger.info(
        f"[predict_test_pool] {len(pool_docs)} samples à scorer "
        f"(batch_id={batch_id}, limit={limit})"
    )

    # ------------------------------------------------------------ #
    # 2. Construire le raw_df                                      #
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
        for d in pool_docs
    ])

    # ------------------------------------------------------------ #
    # 3. Scorer avec @production (ou modèle nommé en debug)        #
    # ------------------------------------------------------------ #
    if model_name is None:
        # Production : vainqueur du tournament cross-model (@production, exclusif).
        # Résolu UNE fois → chargé par version épinglée.
        prod_name, prod_version = resolve_production_model()
        scorer = RakutenScorer.from_champion(prod_name, version=prod_version)
        alias_source = "production"
    else:
        # Debug/CLI : modèle nommé explicitement (alias @champion).
        scorer = RakutenScorer.from_champion(model_name)
        alias_source = "manual"
    result = scorer.score(raw_df)

    logger.info(
        f"[predict_test_pool] Scoré {result.n_scored} samples "
        f"avec {result.model_name}@v{result.model_version} ({result.model_family})"
    )

    # ------------------------------------------------------------ #
    # 4. Écrire dans Prediction_test (idempotent par batch)       #
    # ------------------------------------------------------------ #
    now = datetime.now().isoformat()
    model_tag = f"{result.model_name}@v{result.model_version}"
    # decode_labels : lookup dict de module, appelé UNE fois sur tout le vecteur.
    decoded = decode_labels(result.predictions)

    records = []
    for i, row in enumerate(raw_df.iter_rows(named=True)):
        records.append({
            "productid": row["productid"],
            "imageid": row["imageid"],
            "prediction": int(decoded[i]),
            "confidence": float(result.probas[i].max()),
            "date_pred": now,
            "model": model_tag,
            "model_family": result.model_family,
            "alias_source": alias_source,
            "batch_id": batch_id,
        })

    # Idempotence : re-scorer un batch REMPLACE ses lignes (pas d'accumulation).
    db[OUTPUT_COLLECTION].delete_many({"batch_id": batch_id})
    db[OUTPUT_COLLECTION].insert_many(records)
    logger.info(
        f"[predict_test_pool] {len(records)} prédictions écrites dans "
        f"{OUTPUT_COLLECTION} (batch_id={batch_id})"
    )

    return {
        "message": f"{len(records)} prédictions écrites.",
        "scored": len(records),
        "model": model_tag,
        "model_family": result.model_family,
        "alias_source": alias_source,
        "batch_id": batch_id,
        "collection": OUTPUT_COLLECTION,
        "timestamp": now,
    }
