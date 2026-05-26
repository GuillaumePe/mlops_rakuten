"""
Crée la colonne is_val_selection_v{N} dans le cache parquet _df_full.

Usage Phase 1 :
    python src/data/init_val_selection.py --version 1

Usage Phase 2 (après ingestion d'un nouveau batch) :
    python src/data/init_val_selection.py --version 2

Comportement :
    - Charge le cache parquet du DataModule (chemin auto-résolu via CACHE_DIR
      + slugify des modèles configurés en défaut DataModule)
    - Construit le sur-ensemble :
        v1 : batch_1 non-gold
        v2 : batch_1 ∪ batch_2 non-gold  (Phase 2)
        v3 : batch_1 ∪ batch_2 ∪ batch_3 non-gold  (Phase 2)
    - Applique train_test_split stratifié (test_size=0.10, seed=42) sur la
      colonne `label` (= prdtypecode encodé)
    - Ajoute (ou écrase si déjà présente) la colonne booléenne
      `is_val_selection_v{N}` au parquet
    - Push DVC (optionnel via --no-dvc-push)
    - Log un run MLflow dédié (expériment "init_val_selection") avec stats
      + CSV des productids val_selection en artifact pour audit

⚠ ONE-SHOT par version. Toute ré-exécution avec le même --version écrasera la
colonne existante : décision intentionnelle pour permettre de corriger une
init bugée avant qu'aucun fit_base_learner n'ait été lancé, MAIS attention
à ne pas relancer sur une version déjà en production de fits — sinon les
modèles déjà entraînés voient un val_selection différent de celui utilisé
pour les promotions @active passées.

⚠ Si le cache parquet est régénéré (bump de cache_version dans le DataModule
suite à un changement de modèle d'embedding), il faut re-lancer ce script.
Le flag is_val_selection_v{N} ne survit pas à une régénération du parquet.

Note architecture : contrairement à init_gold_test_set qui écrit dans Mongo
(parce que is_gold est une fonction déterministe f(productid)), le
val_selection est un échantillonnage stratifié figé à un instant t. On écrit
directement dans le parquet, source de vérité pour le DataModule.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

# Imports projet — la racine du repo doit être dans PYTHONPATH (cas standard
# quand on lance `python src/data/init_val_selection.py` depuis la racine)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.experiments.datamodule.rakuten_datamodule import CACHE_DIR  # noqa: E402
from src.experiments.datamodule.encoders import slugify_model_name   # noqa: E402


# --- Defaults alignés sur RakutenLightningDataModule ----------------------- #
DEFAULT_TEXT_MODEL = "dangvantuan/sentence-camembert-base"
DEFAULT_IMAGE_MODEL = "resnet18"
DEFAULT_CACHE_VERSION = 1

VAL_SELECTION_FRACTION = 0.10
VAL_SELECTION_SEED = 42
MAX_CLASS_DRIFT_PCT = 0.01  # seuil sanity check stratification


def _resolve_cache_path(text_model: str, image_model: str, cache_version: int) -> Path:
    """Reproduit la logique de RakutenLightningDataModule.cache_path."""
    return CACHE_DIR / (
        f"embeddings_{slugify_model_name(text_model)}_"
        f"{slugify_model_name(image_model)}_v{cache_version}.parquet"
    )


def _resolve_super_set_mask(version: int) -> pl.Expr:
    """Expression Polars du masque sur-ensemble pour val_selection_v{version}.

    v1 : batch_id == 1 ∧ !is_gold
    v2 : batch_id ∈ {1, 2} ∧ !is_gold
    v3 : batch_id ∈ {1, 2, 3} ∧ !is_gold
    """
    allowed_batches = list(range(1, version + 1))
    return pl.col("batch_id").is_in(allowed_batches) & (~pl.col("is_gold"))


def _md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _stratification_check(
    df: pl.DataFrame,
    is_val_col: str,
    version: int,
    label_col: str = "label",
) -> dict:
    """Vérifie que la distribution des classes dans val_selection est proche
    de celle du résiduel du sur-ensemble (écart absolu max par classe < seuil).

    Compare p_val (proportion d'une classe dans val_selection) vs p_train
    (proportion dans le sur-ensemble \\ val_selection).
    """
    super_set_mask = _resolve_super_set_mask(version)
    df_val = df.filter(super_set_mask & pl.col(is_val_col).cast(pl.Boolean))
    df_train = df.filter(super_set_mask & (~pl.col(is_val_col).cast(pl.Boolean)))

    if df_val.height == 0 or df_train.height == 0:
        return {"max_drift_observed": float("nan"), "drift_per_class": {}, "passed": False}

    p_val = (
        df_val.group_by(label_col)
        .len()
        .with_columns((pl.col("len") / df_val.height).alias("p_val"))
        .select(label_col, "p_val")
    )
    p_train = (
        df_train.group_by(label_col)
        .len()
        .with_columns((pl.col("len") / df_train.height).alias("p_train"))
        .select(label_col, "p_train")
    )
    joined = p_val.join(p_train, on=label_col, how="full", coalesce=True).fill_null(0.0)
    joined = joined.with_columns(
        (pl.col("p_val") - pl.col("p_train")).abs().alias("drift_abs")
    )
    drift_per_class = {
        int(row[label_col]): float(row["drift_abs"])
        for row in joined.iter_rows(named=True)
    }
    max_drift = max(drift_per_class.values()) if drift_per_class else 0.0
    return {
        "max_drift_observed": max_drift,
        "drift_per_class": drift_per_class,
        "passed": max_drift < MAX_CLASS_DRIFT_PCT,
    }


def init_val_selection(
    version: int,
    cache_path: Path,
    push_dvc: bool = True,
    mlflow_experiment: str = "init_val_selection",
) -> None:
    """Pipeline complet M.0b."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache parquet introuvable : {cache_path}\n"
            f"As-tu lancé `prepare_data()` du DataModule au moins une fois "
            f"pour générer ce cache ?"
        )

    print(f"[init_val_selection] Charge {cache_path}")
    df = pl.read_parquet(cache_path)
    print(f"[init_val_selection] _df_full : {df.height} samples, {len(df.columns)} cols")

    # Vérifications préalables — colonnes critiques
    for col in ("productid", "batch_id", "is_gold", "label"):
        if col not in df.columns:
            raise ValueError(
                f"Colonne {col!r} absente du parquet. "
                f"Le DataModule a-t-il bien été setup() au moins une fois "
                f"pour appliquer la migration douce (batch_id, is_gold) ?"
            )

    # 1. Sur-ensemble
    super_set_mask = _resolve_super_set_mask(version)
    df_super = df.filter(super_set_mask)
    n_super = df_super.height
    print(f"[init_val_selection] Sur-ensemble v{version} : {n_super} samples")
    if n_super == 0:
        raise RuntimeError(
            f"Sur-ensemble vide pour v{version}. Vérifie que batch_id ∈ "
            f"[1..{version}] contient bien des samples non-gold dans le parquet."
        )

    # 2. Split stratifié 10%
    labels = df_super.get_column("label").to_numpy()
    productids_super = df_super.get_column("productid").to_numpy()
    _, idx_val = train_test_split(
        np.arange(n_super),
        test_size=VAL_SELECTION_FRACTION,
        stratify=labels,
        random_state=VAL_SELECTION_SEED,
    )
    productids_val_selection = set(productids_super[idx_val].tolist())
    print(
        f"[init_val_selection] Split stratifié seed={VAL_SELECTION_SEED} : "
        f"val={len(idx_val)}, train_residuel={n_super - len(idx_val)}"
    )

    # 3. Construire la colonne booléenne is_val_selection_v{version}
    col_name = f"is_val_selection_v{version}"
    productids_val_list = list(productids_val_selection)
    is_val_col = (
        pl.col("productid").is_in(productids_val_list)
         .alias(col_name)
     )

    if col_name in df.columns:
        print(f"[init_val_selection] ⚠ Colonne {col_name} déjà présente, écrasement.")
        df = df.drop(col_name)
    df = df.with_columns(is_val_col)

    # 4. Sanity checks orthogonalité
    n_overlap_gold = df.filter(pl.col(col_name) & pl.col("is_gold")).height
    if n_overlap_gold != 0:
        raise RuntimeError(
            f"Bug : {n_overlap_gold} samples sont simultanément "
            f"val_selection_v{version} et gold."
        )

    n_val_final = df.filter(pl.col(col_name)).height
    assert n_val_final == len(idx_val), (
        f"Incohérence taille val_selection : {n_val_final} dans df vs "
        f"{len(idx_val)} attendus"
    )

    # 5. Distribution par classe (sanity check stratification)
    strat_report = _stratification_check(df, col_name, version)
    verdict = "OK" if strat_report["passed"] else "WARN"
    print(
        f"[init_val_selection] Stratification : "
        f"max_drift={strat_report['max_drift_observed']:.4f} "
        f"(seuil {MAX_CLASS_DRIFT_PCT}) → {verdict}"
    )

    # 6. Écriture du parquet
    df.write_parquet(cache_path)
    print(f"[init_val_selection] Parquet écrit : {cache_path}")
    md5 = _md5_of_file(cache_path)
    print(f"[init_val_selection] MD5 nouveau parquet : {md5}")

    # 7. Push DVC (best effort)
    if push_dvc:
        import subprocess
        try:
            subprocess.run(["dvc", "add", str(cache_path)], check=True)
            subprocess.run(["dvc", "push", str(cache_path) + ".dvc"], check=True)
            print("[init_val_selection] DVC push OK")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[init_val_selection] ⚠ DVC push échoué : {e}. Continue.")

    # 8. Log MLflow
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name=f"init_val_selection_v{version}") as run:
        mlflow.log_param("version", version)
        mlflow.log_param("seed", VAL_SELECTION_SEED)
        mlflow.log_param("fraction", VAL_SELECTION_FRACTION)
        mlflow.log_param("super_set_batches", list(range(1, version + 1)))
        mlflow.log_param("cache_path", str(cache_path))
        mlflow.log_param("parquet_md5", md5)

        mlflow.log_metric("super_set_size", n_super)
        mlflow.log_metric("val_selection_size", n_val_final)
        mlflow.log_metric("train_residuel_size", n_super - len(idx_val))
        mlflow.log_metric("max_class_drift", strat_report["max_drift_observed"])
        mlflow.log_metric("stratification_passed", int(strat_report["passed"]))

        distrib_df = pl.DataFrame({
            "label": list(strat_report["drift_per_class"].keys()),
            "drift_abs": list(strat_report["drift_per_class"].values()),
        })
        distrib_csv = Path(f"/tmp/val_selection_v{version}_distrib.csv")
        distrib_df.write_csv(distrib_csv)
        mlflow.log_artifact(str(distrib_csv))

        productids_csv = Path(f"/tmp/val_selection_v{version}_productids.csv")
        pl.DataFrame({"productid": sorted(productids_val_selection)}).write_csv(
            productids_csv
        )
        mlflow.log_artifact(str(productids_csv))

        print(f"[init_val_selection] MLflow run : {run.info.run_id}")

    print(f"\n[init_val_selection] ✅ Terminé.")
    print(f"    Colonne {col_name} créée dans {cache_path}")
    print(f"    Set env var ACTIVE_VAL_SELECTION_VERSION={version} avant le prochain fit.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--cache-version", type=int, default=DEFAULT_CACHE_VERSION)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Chemin explicite du cache (override l'auto-résolution).",
    )
    parser.add_argument(
        "--no-dvc-push",
        action="store_true",
        help="Skip dvc add/push (utile en debug local).",
    )
    parser.add_argument("--mlflow-experiment", default="init_val_selection")
    args = parser.parse_args()

    cache_path = args.cache_path or _resolve_cache_path(
        args.text_model, args.image_model, args.cache_version
    )

    init_val_selection(
        version=args.version,
        cache_path=cache_path,
        push_dvc=not args.no_dvc_push,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == "__main__":
    main()
