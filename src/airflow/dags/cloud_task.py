"""
Helper d'orchestration : make_cloud_task.

Fabrique une tâche Airflow (PythonOperator) qui soumet une action cloud via
`python -m src.cloud.submit` (entrypoint léger, sans deps ML — option B), et
traduit le CODE DE SORTIE
(contrat défini en 0.c) en sémantique de retry Airflow :

    exit 0                → succès
    exit EXIT_NO_CAPACITY → AirflowException      → RETRY (backoff exponentiel)
    exit ≠ 0 (autre)      → AirflowFailException   → FAIL-FAST (aucun retry)

Le BashOperator ne sait pas distinguer pénurie (retryable) de bug (fatal) :
il retente sur tout exit≠0. Ce helper est le remplaçant qui consomme le
contrat de sortie de runner.py.

Placé dans le dossier dags/ (importable en sibling : `from cloud_task import
make_cloud_task`), car le PYTHONPATH du scheduler n'inclut pas forcément la
racine projet au parse-time. `src` n'est donc importé qu'au RUNTIME, dans le
callable, après ajout de la racine projet au sys.path.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from airflow.exceptions import AirflowException, AirflowFailException
from airflow.operators.python import PythonOperator

# src/airflow/dags/cloud_task.py → parents[3] = racine projet
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Interpréteur du subprocess runner (override possible sans toucher au code)
# Racine projet montée dans le conteneur Airflow (option B). Posée par le
# compose (RAKUTEN_PROJECT_ROOT=/opt/project). Fallback parents[3] pour une
# exécution hors conteneur (ex. tests sur l'hôte).
_PROJECT_ROOT = Path(
    os.getenv("RAKUTEN_PROJECT_ROOT") or Path(__file__).resolve().parents[3]
)

def _run_cloud_action(
    *,
    experiment: str,
    cloud_action: str,
    gpu_types: list[str] | None,
    cloud_image: str | None,
    cloud_timeout: int,
    overrides: list[str],
    extra_args: list[str],
) -> None:
    """
    Lance `runner.py --action submit_cloud` en subprocess et traduit le code
    de sortie en sémantique de retry Airflow. Exécuté DANS le worker Airflow.
    """
    # src peut ne pas être sur le PYTHONPATH du process Airflow → on l'ajoute
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.cloud.exceptions import EXIT_NO_CAPACITY  # source de vérité unique

    cmd = [
        _PYTHON_BIN, "-m", "src.cloud.submit",
        "--experiment", experiment,
        "--cloud-action", cloud_action,
        "--cloud-timeout", str(cloud_timeout),
    ]
    if gpu_types:  # None → le runner utilise sa cascade par défaut (pas de duplication)
        cmd += ["--gpu-types", *gpu_types]
    if cloud_image:
        cmd += ["--cloud-image", cloud_image]
    if overrides:
        cmd += ["--set", *overrides]
    if extra_args:
        cmd += list(extra_args)

    env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
    print(f"[make_cloud_task] cwd = {_PROJECT_ROOT}")
    print(f"[make_cloud_task] cmd = {' '.join(cmd)}")

    try:
        rc = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, check=False).returncode
    except FileNotFoundError as e:
        # python/module introuvable → bug de config déterministe, pas une pénurie
        raise AirflowFailException(
            f"[make_cloud_task] impossible de lancer {cmd[0]} : {e}"
        ) from e

    if rc == 0:
        print(f"[make_cloud_task] {cloud_action} : succès (exit 0)")
        return
    if rc == EXIT_NO_CAPACITY:
        # Pénurie GPU transitoire → RETRYABLE (backoff piloté par l'operator)
        raise AirflowException(
            f"[make_cloud_task] {cloud_action} : pénurie GPU (exit {rc}) "
            f"→ retry programmé"
        )
    # Toute autre sortie ≠ 0 → erreur déterministe → FAIL-FAST (aucun retry)
    raise AirflowFailException(
        f"[make_cloud_task] {cloud_action} : échec fatal (exit {rc}) "
        f"→ pas de retry"
    )


def make_cloud_task(
    task_id: str,
    experiment: str,
    cloud_action: str,
    *,
    gpu_types: list[str] | None = None,
    cloud_image: str | None = None,
    cloud_timeout: int = 3600,
    overrides: list[str] | None = None,
    extra_args: list[str] | None = None,
    retries: int = 24,
    retry_delay: timedelta = timedelta(minutes=10),
    max_retry_delay: timedelta = timedelta(minutes=30),
    execution_timeout: timedelta | None = None,
    pool: str = "training_pool",
    **operator_kwargs,
) -> PythonOperator:
    """
    Construit un PythonOperator qui soumet `cloud_action` au cloud, avec
    politique de relance « attendre qu'un GPU se libère » :

    - retries / retry_delay / max_retry_delay avec backoff exponentiel :
      défaut 24 × (10min→30min cappé) ≈ fenêtre d'attente ~11h.
    - retry UNIQUEMENT sur pénurie (exit 42) ; fail-fast sur tout autre bug.

    Args:
        overrides : passés en `--set KEY=VALUE` (templatés : peuvent contenir
            du Jinja, ex. "version={{ var.value.batch_id }}").
        extra_args : args CLI bruts additionnels (ex. warm-start).
        gpu_types : None → cascade par défaut du runner (source unique).
        cloud_timeout : borne côté runner (stop du pod). execution_timeout
            (Airflow) reste None par défaut pour ne pas doublonner ; si fixé,
            le mettre > cloud_timeout comme filet de sécurité.
    """
    return PythonOperator(
        task_id=task_id,
        python_callable=_run_cloud_action,
        op_kwargs={
            "experiment": experiment,
            "cloud_action": cloud_action,
            "gpu_types": gpu_types,
            "cloud_image": cloud_image,
            "cloud_timeout": cloud_timeout,
            "overrides": overrides or [],
            "extra_args": extra_args or [],
        },
        pool=pool,
        retries=retries,
        retry_delay=retry_delay,
        retry_exponential_backoff=True,
        max_retry_delay=max_retry_delay,
        execution_timeout=execution_timeout,
        **operator_kwargs,
    )