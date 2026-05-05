"""
Runner CLI pour les expériences M2/M3/M4.

Usage:
    python -m src.experiments.runner --experiment m2 --action fit
    python -m src.experiments.runner --experiment m2 --action evaluate
    python -m src.experiments.runner --experiment m2 --action prepare_data

Le runner :
1. Charge la config YAML de l'expérience
2. Instancie le DataModule selon le mode
3. Instancie l'Experiment selon la stratégie (sklearn / lightning)
4. Exécute l'action demandée
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import mlflow
import yaml

from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
from src.experiments.models.m2.m2 import M2Stacking
from src.experiments.strategies.sklearn_experiment import SklearnExperiment


CONFIG_DIR = Path("src/experiments/config")


def load_config(experiment_name: str) -> dict:
    """Charge la config YAML correspondant à l'expérience."""
    config_path = CONFIG_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_m2_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """Assemble DataModule + M2Stacking + SklearnExperiment depuis une config M2."""
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.15),
        test_size=dm_cfg.get("test_size", 0.15),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
    )

    model_cfg = config["model"]

    def m2_factory(_optuna_callback_unused):
        # Le callback n'est plus utilisé (logging direct dans l'objective)
        # mais SklearnExperiment l'expose pour compat ABC future
        return M2Stacking(
            text_cols=dm.text_cols,
            image_cols=dm.image_cols,
            tabular_cols=dm.tabular_cols,
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    experiment = SklearnExperiment(
        model_factory=m2_factory,
        run_name=config["mlflow"]["run_name"],
        tags=config["mlflow"].get("tags", {}),
    )

    return dm, experiment


# Registry des constructeurs par expérience.
# Phase 1+ : ajouter "m3", "m4" ici.
EXPERIMENT_BUILDERS = {
    "m2": build_m2_experiment,
}


def cmd_prepare_data(dm: RakutenLightningDataModule):
    """Extrait/met à jour le cache d'embeddings. Étape lourde (GPU recommandé)."""
    print(f"[Runner] prepare_data() — cache: {dm.cache_path}")
    dm.prepare_data()


def cmd_fit(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Setup + fit avec tracking MLflow."""
    print("[Runner] setup()...")
    dm.setup()
    print("[Runner] fit() avec tracking MLflow...")
    experiment.fit(dm)


def cmd_evaluate(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Évalue le modèle sur le test set. Suppose que fit() a déjà été appelé."""
    if experiment.model is None:
        raise RuntimeError(
            "Le modèle n'est pas fitté. Lance d'abord `--action fit` "
            "dans le même process, ou implémente le rechargement depuis MLflow."
        )
    print("[Runner] setup()...")
    dm.setup()
    results = experiment.evaluate(dm)
    print(f"[Runner] Résultats sur test : {results}")
    return results


def main():
    parser = argparse.ArgumentParser(description="MLOps experiment runner")
    parser.add_argument(
        "--experiment", required=True, choices=list(EXPERIMENT_BUILDERS),
        help="Nom de l'expérience (correspond à src/experiments/config/<name>.yaml)",
    )
    parser.add_argument(
        "--action", required=True,
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate"],
        help="Action à exécuter",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="URI MLflow (override la config). Ex: http://mlflow:5000",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limiter le nombre de samples (debug/test rapide). None = full dataset.",
    )
    args = parser.parse_args()

    # Charger la config
    config = load_config(args.experiment)
    if args.limit is not None:
        config["limit"] = args.limit
    print(f"[Runner] Config chargée : {args.experiment} (limit={args.limit})")


    # MLflow tracking URI : CLI > config > défaut
    tracking_uri = (
        args.mlflow_tracking_uri
        or config.get("mlflow", {}).get("tracking_uri")
        or "http://mlflow:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[Runner] MLflow tracking URI : {tracking_uri}")

    # Set experiment MLflow (groupe les runs par modèle)
    experiment_name = config["mlflow"].get("experiment_name", args.experiment)
    mlflow.set_experiment(experiment_name)

    # Construire les composants
    builder = EXPERIMENT_BUILDERS[args.experiment]
    dm, experiment = builder(config)

    # Dispatch action
    if args.action == "prepare_data":
        cmd_prepare_data(dm)
    elif args.action == "fit":
        cmd_prepare_data(dm)  # idempotent : ne refait rien si cache à jour
        cmd_fit(dm, experiment)
    elif args.action == "evaluate":
        cmd_evaluate(dm, experiment)
    elif args.action == "fit_and_evaluate":
        cmd_prepare_data(dm)
        cmd_fit(dm, experiment)
        cmd_evaluate(dm, experiment)


if __name__ == "__main__":
    sys.exit(main())