"""
P4.5c — Assemblage de la soumission ENS (challenge 35) à partir de Prediction_test.

Étape LOCALE (aucun GPU, aucune dépendance ML) : lit les prédictions du batch N
dans Prediction_test, les joint sur `productid` avec X_test_update.csv, et écrit
`data/submissions/submission_batch{N}.csv` au format ENS.

Format ENS (identique à Y_train_update.csv) :
    ,prdtypecode
    0,2583
    1,1560
    ...
En-tête = virgule + "prdtypecode" ; colonne 1 = l'index entier du CSV d'origine
(celui que `df.drop("")` a jeté à l'upload Mongo → seul le CSV le porte encore) ;
colonne 2 = le prdtypecode Rakuten. `Prediction_test.prediction` est DÉJÀ décodé
(pas un label interne 0-26) → recopie directe, aucun decode ici.

Doctrine d'exhaustivité : la soumission ENS doit couvrir TOUT X_test. Le CSV est
la référence d'itération ; toute ligne sans prédiction fait ÉCHOUER la fonction
(pas de soumission partielle silencieuse).

Dépendances volontairement minimales (csv stdlib + pymongo) : la fonction tourne
dans le conteneur Airflow mince, qui n'embarque pas Polars. La jointure est un
dict lookup sur n=13812 → O(n), ~1 Mo.

Appelable par :
    - PythonOperator local dans le DAG Predict_Y_test (P4.5d)
    - CLI : python -m src.models.build_submission --batch-id 2
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src.data.mongo_utils import get_db

load_dotenv()

logger = logging.getLogger(__name__)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
X_TEST_CSV = DATA_ROOT / "data/raw_data_test/X_test_update.csv"
SUBMISSION_DIR = DATA_ROOT / "data/submissions"

PREDICTION_COLLECTION = "Prediction_test"

# Les 27 classes Rakuten (mêmes valeurs que src/data/ingest_batch.VALID_CLASSES).
VALID_CLASSES = {
    10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302,
    1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582,
    2583, 2585, 2705, 2905,
}

_MAX_REPORTED = 5  # nb d'exemples cités dans les messages d'erreur


def _read_test_index(csv_path: Path) -> list[tuple[str, int]]:
    """
    Lit X_test_update.csv et retourne [(index_ens, productid), ...] dans l'ordre
    du fichier.

    `index_ens` est conservé en STR (recopie verbatim de la colonne d'origine) :
    aucune reformatation, donc aucun risque de divergence de représentation.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} introuvable. Lancer "
            f"`dvc pull data/raw_data_test/X_test_update.csv.dvc`."
        )

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        if header[0] != "":
            raise ValueError(
                f"En-tête inattendu dans {csv_path.name} : première colonne = "
                f"{header[0]!r}, attendu '' (colonne d'index sans nom)."
            )
        if "productid" not in header:
            raise ValueError(
                f"Colonne 'productid' absente de {csv_path.name} : {header}"
            )

        pid_col = header.index("productid")

        rows: list[tuple[str, int]] = []
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            idx_raw = row[0].strip()
            if not idx_raw.lstrip("-").isdigit():
                raise ValueError(
                    f"{csv_path.name} ligne {line_no} : index ENS non entier "
                    f"({idx_raw!r})."
                )
            rows.append((idx_raw, int(row[pid_col])))

    return rows


def _load_predictions(db, batch_id: int) -> tuple[dict[int, int], list[str]]:
    """
    Charge {productid: prdtypecode} pour le batch demandé.

    Retourne aussi la liste des `model` distincts rencontrés : l'idempotence de
    predict_test_pool (delete_many par batch_id) garantit normalement un modèle
    unique par batch — un pluriel signale une écriture concurrente ou un pool
    scoré en deux passes, ce qui invalide la traçabilité de la soumission.
    """
    docs = list(
        db[PREDICTION_COLLECTION].find(
            {"batch_id": batch_id},
            {"_id": 0, "productid": 1, "prediction": 1, "model": 1},
        )
    )
    if not docs:
        raise RuntimeError(
            f"Aucune prédiction pour batch_id={batch_id} dans "
            f"{PREDICTION_COLLECTION}. Lancer predict_test_pool (cloud GPU) d'abord."
        )

    preds: dict[int, int] = {}
    duplicates: list[int] = []
    models: set[str] = set()

    for d in docs:
        pid = int(d["productid"])
        if pid in preds:
            duplicates.append(pid)
        preds[pid] = int(d["prediction"])
        models.add(str(d.get("model", "?")))

    if duplicates:
        raise RuntimeError(
            f"{len(duplicates)} productid dupliqués dans {PREDICTION_COLLECTION} "
            f"(batch_id={batch_id}), ex. {duplicates[:_MAX_REPORTED]}. "
            f"Purger le batch et re-scorer."
        )

    return preds, sorted(models)


def run_build_submission(
    batch_id: int,
    mongo_uri: str = "",
    csv_path: str | Path = "",
    output_dir: str | Path = "",
    **kwargs,
) -> dict:
    """
    Assemble submission_batch{batch_id}.csv à partir de Prediction_test.

    Args:
        batch_id: batch venant d'être entraîné (= estampille des prédictions).
        mongo_uri: URI MongoDB. Défaut : MONGO_URI env var.
        csv_path: override du chemin X_test_update.csv (tests).
        output_dir: override du dossier de sortie (tests).
        **kwargs: ignoré (PythonOperator passe context, etc.)

    Returns:
        dict : message, path, n_rows, batch_id, model, timestamp.

    Raises:
        RuntimeError si la soumission serait incomplète (≥1 ligne de X_test sans
        prédiction) ou si un prdtypecode est hors des 27 classes valides.
    """
    if batch_id is None:
        raise ValueError(
            "batch_id requis (pas de défaut : l'estampille est la clé de jointure)."
        )
    batch_id = int(batch_id)

    csv_file = Path(csv_path) if csv_path else X_TEST_CSV
    out_dir = Path(output_dir) if output_dir else SUBMISSION_DIR

    db = get_db(uri=mongo_uri) if mongo_uri else get_db()

    # ------------------------------------------------------------ #
    # 1. Référence = le CSV d'origine (porteur de l'index ENS)     #
    # ------------------------------------------------------------ #
    test_rows = _read_test_index(csv_file)
    logger.info(f"[build_submission] {len(test_rows)} lignes dans {csv_file.name}")

    # ------------------------------------------------------------ #
    # 2. Prédictions du batch                                      #
    # ------------------------------------------------------------ #
    preds, models = _load_predictions(db, batch_id)
    logger.info(
        f"[build_submission] {len(preds)} prédictions (batch_id={batch_id}), "
        f"modèle(s) : {models}"
    )
    if len(models) > 1:
        raise RuntimeError(
            f"Plusieurs modèles dans le batch {batch_id} : {models}. "
            f"Soumission non traçable — purger le batch et re-scorer."
        )

    # ------------------------------------------------------------ #
    # 3. Jointure + garde-fous d'exhaustivité et de validité       #
    # ------------------------------------------------------------ #
    missing: list[int] = []
    invalid: list[tuple[int, int]] = []
    out_rows: list[tuple[str, int]] = []

    for idx_ens, pid in test_rows:
        code = preds.get(pid)
        if code is None:
            missing.append(pid)
            continue
        if code not in VALID_CLASSES:
            invalid.append((pid, code))
        out_rows.append((idx_ens, code))

    if missing:
        raise RuntimeError(
            f"Soumission INCOMPLÈTE : {len(missing)}/{len(test_rows)} lignes de "
            f"{csv_file.name} sans prédiction (batch_id={batch_id}), "
            f"ex. productid {missing[:_MAX_REPORTED]}. "
            f"Causes probables : run predict_test_pool avec --limit, ou "
            f"X_to_predict_pool filtré par images disponibles à l'upload. "
            f"Aucun fichier écrit."
        )
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} prdtypecode hors des 27 classes valides, "
            f"ex. {invalid[:_MAX_REPORTED]}. Aucun fichier écrit."
        )

    extra = len(preds) - len(out_rows)
    if extra > 0:
        logger.warning(
            f"[build_submission] {extra} prédictions n'ont aucune ligne "
            f"correspondante dans {csv_file.name} (ignorées)."
        )

    # ------------------------------------------------------------ #
    # 4. Écriture (après validation uniquement)                    #
    # ------------------------------------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"submission_batch{batch_id}.csv"

    # newline="" + lineterminator="\n" : LF strict, pas de CRLF.
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["", "prdtypecode"])
        writer.writerows(out_rows)

    now = datetime.now().isoformat()
    logger.info(
        f"[build_submission] {out_path} écrit : {len(out_rows)} lignes "
        f"(+ en-tête), modèle {models[0]}"
    )

    return {
        "message": f"{len(out_rows)} lignes écrites.",
        "path": str(out_path),
        "n_rows": len(out_rows),
        "batch_id": batch_id,
        "model": models[0],
        "timestamp": now,
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Assemble la soumission ENS d'un batch depuis Prediction_test."
    )
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--csv-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--mongo-uri", default="")
    args = parser.parse_args()

    result = run_build_submission(
        batch_id=args.batch_id,
        mongo_uri=args.mongo_uri,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))
