"""
Callback Optuna qui log chaque trial en nested run MLflow et trace le best score
dans le run parent.

Doit être instancié par SklearnExperiment qui fournit le contexte MLflow.
"""
from __future__ import annotations
from typing import Callable

import mlflow
import optuna


def make_mlflow_optuna_callback() -> Callable:
    """
    Retourne un callback (study, trial) → None qui :
    - ouvre un nested MLflow run par trial Optuna
    - log les params du trial et les metrics (mean, std, mean-std)
    - ferme le nested run

    Doit être appelé pendant qu'un run MLflow parent est actif (with mlflow.start_run).
    """

    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return  # skip pruned/failed trials

        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            nested=True,
        ):
            mlflow.log_params(trial.params)
            mlflow.log_metric("optuna_objective", trial.value)
            # Récupérer les user_attrs stockés par M2Stacking
            mean = trial.user_attrs.get("f1_weighted_mean")
            std = trial.user_attrs.get("f1_weighted_std")
            if mean is not None:
                mlflow.log_metric("f1_weighted_mean", mean)
            if std is not None:
                mlflow.log_metric("f1_weighted_std", std)

    return callback