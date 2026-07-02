"""
T.3d — Action compare_and_promote : arbitrage de promotion @champion sur gold COURANT.

Tourne en LOCAL (lecture MLflow + swap d'alias = léger). Suppose que :
  - chaque challenger a loggé eval_gold/f1_weighted (gold courant) pendant son fit,
    SANS s'auto-promouvoir (promotion.enabled=false, cf. T.3b) ;
  - le run de ré-évaluation du champion (eval_gold_champion, T.3c) a déjà tourné et
    loggé la f1 du champion sur le MÊME gold courant.

Invariant de comparabilité : tous les candidats (challengers + champion) sont comparés
sur la même réalisation du gold → l'écart de f1 ne mesure que la différence de modèle,
pas la variance d'échantillonnage du test.

Décision : promeut le meilleur challenger ssi (f1_best - f1_champion) >= epsilon.
Sinon le champion reste. Aucun @champion existant → promotion directe du meilleur.

Ne réutilise PAS evaluate_promotion_via_logged_metrics (qui lit la f1 FIGÉE du champion).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import mlflow
from dotenv import load_dotenv

from src.models.utils import promotion_exclusive_best_model_to_production

load_dotenv()
logger = logging.getLogger(__name__)


def _read_run_metric(client, experiment_id, run_name, metric_key):
    """(run_id, metric) du run le plus récent nommé run_name, sinon (None, None)."""
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None, None
    run = runs[0]
    return run.info.run_id, run.data.metrics.get(metric_key)


def run_compare_and_promote(
    registry_model_name: str,
    challenger_run_names: list[str],
    champion_run_name: str,
    batch_id: Optional[int] = None,
    experiment_name: str = "training_compare",
    metric_key: str = "eval_gold/f1_weighted",
    epsilon: float = 0.005,
    tracking_uri: str = "",
    **kwargs,
) -> dict:
    """
    Compare les challengers au champion re-scoré sur le gold courant et promeut le meilleur.

    Args:
        registry_model_name: registered model MLflow (ex: "rakuten-m3-2-coadaptation").
        challenger_run_names: run_names des challengers (ex: ["m3_2_stateless_b2",
            "m3_2_stateful_b2"]). Le DAG les construit — il connaît le fan-out.
        champion_run_name: run_name du run eval_gold_champion (ex: "eval_gold_champion_b2").
        batch_id: pour nommer le run récapitulatif.
        experiment_name: experiment MLflow commun à TOUS les runs comparés.
        metric_key / epsilon: métrique et marge de décision.

    Returns:
        dict de décision (promoted, best_challenger, gain, f1 de chacun, etc.).
    """
    if not tracking_uri:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri)

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' introuvable.")
    exp_id = exp.experiment_id

    # 1. F1 des challengers (gold courant)
    challengers = {}
    for rn in challenger_run_names:
        run_id, f1 = _read_run_metric(client, exp_id, rn, metric_key)
        if run_id is None:
            logger.warning(f"[compare_and_promote] Challenger '{rn}' introuvable, ignoré.")
            continue
        if f1 is None:
            logger.warning(
                f"[compare_and_promote] Challenger '{rn}' sans '{metric_key}', ignoré."
            )
            continue
        challengers[rn] = {"run_id": run_id, "f1": float(f1)}

    if not challengers:
        raise RuntimeError("Aucun challenger valide avec métrique. Rien à promouvoir.")

    # 2. Meilleur challenger
    best_rn = max(challengers, key=lambda k: challengers[k]["f1"])
    best = challengers[best_rn]
    best_f1 = best["f1"]

    # 3. F1 champion RE-SCORÉ (gold courant) + existence d'un @champion
    _, champ_f1 = _read_run_metric(client, exp_id, champion_run_name, metric_key)
    try:
        champion_mv = client.get_model_version_by_alias(registry_model_name, "champion")
        champion_exists = True
        champion_version = champion_mv.version
    except mlflow.exceptions.MlflowException:
        champion_exists = False
        champion_version = None

    # 4. Version registry du meilleur challenger (l'enregistrement n'est PAS gaté par T.3b)
    mvs = client.search_model_versions(
        f"name='{registry_model_name}' AND run_id='{best['run_id']}'"
    )
    if not mvs:
        raise RuntimeError(
            f"Aucune version registry pour '{best_rn}' (run_id={best['run_id']}) "
            f"sous '{registry_model_name}'. log_model a-t-il bien eu lieu ?"
        )
    best_version = mvs[0].version

    # 5. Décision
    gain = None
    if not champion_exists or champ_f1 is None:
        reason = (
            "first_champion (aucun @champion existant)"
            if not champion_exists
            else "champion re-score absent → traité comme first_champion"
        )
        promotion_exclusive_best_model_to_production(registry_model_name, best_version)
        promoted = True
    else:
        gain = best_f1 - float(champ_f1)
        if gain >= epsilon:
            promotion_exclusive_best_model_to_production(registry_model_name, best_version)
            promoted = True
            reason = f"gain {gain:+.4f} >= epsilon {epsilon:+.4f}"
        else:
            promoted = False
            reason = f"gain {gain:+.4f} < epsilon {epsilon:+.4f} → champion conservé"

    # 6. Run récapitulatif
    mlflow.set_experiment(experiment_name)
    rec_name = (
        f"compare_and_promote_b{batch_id}" if batch_id is not None
        else "compare_and_promote"
    )
    with mlflow.start_run(run_name=rec_name):
        mlflow.set_tag("role", "compare_and_promote")
        for rn, d in challengers.items():
            mlflow.log_metric(f"challenger/{rn}", d["f1"])
        if champ_f1 is not None:
            mlflow.log_metric("champion/regold_f1", float(champ_f1))
        mlflow.log_param("best_challenger", best_rn)
        mlflow.log_param("best_challenger_version", best_version)
        mlflow.log_param("promoted", promoted)
        mlflow.log_param("reason", reason)
        if gain is not None:
            mlflow.log_metric("gain_vs_champion", gain)

    result = {
        "promoted": promoted,
        "reason": reason,
        "best_challenger": best_rn,
        "best_challenger_version": best_version,
        "best_challenger_f1": best_f1,
        "champion_regold_f1": (float(champ_f1) if champ_f1 is not None else None),
        "champion_existed": champion_exists,
        "champion_version_before": champion_version,
        "gain": gain,
        "epsilon": epsilon,
        "challengers": {rn: d["f1"] for rn, d in challengers.items()},
    }
    logger.info(f"[compare_and_promote] {result}")
    return result