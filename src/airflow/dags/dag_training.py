"""
T.1 — DAG Training : fan-out des 5 base learners stateless.

Version squelette (T.1) : ne couvre que la lignée STATELESS des base learners.
Les 7 fusions stateless seront ajoutées en T.2 ; la lignée stateful en T.3 ;
le comparateur + promotions en T.4 ; le trigger auto par Ingestion en T.5.

Doctrine :
    - DAG mince / Python épais : chaque tâche = un submit_cloud → RunPod.
      Aucune logique métier locale.
    - Retry pénurie / fail-fast bug : géré par make_cloud_task
      (exit 42 → retry backoff exponentiel ; exit ≠0 → fail-fast).
    - Pool training_pool (3 slots GPU) : concurrence bornée.

Paramètres (via dag_run.conf ou l'UI "Trigger DAG w/ config") :
    - batch_id (int|null, défaut = Variable Airflow `batch_id`)
    - retrain_strategy (str, défaut "compare") — informatif à T.1 ;
      utilisé pour brancher la topologie à partir de T.3.

Convention run_name (§3.4 du plan) :
    {experiment}_stateless_b{batch_id}
"""
from datetime import timedelta

from airflow.decorators import dag
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from cloud_task import make_cloud_task


# 5 base learners de Phase 1, tous entraînés en stateless à chaque batch.
# Noms exacts = clés de EXPERIMENT_BUILDERS dans src/experiments/runner.py.
BASE_LEARNERS = [
    "base_learner_textcnn",
    "base_learner_camembert_lora",
    "base_learner_siglip2",
    "base_learner_resnet18_full_ft",
    "base_learner_resnet50_partial_ft",
]

# batch_id : priorité dag_run.conf > Variable Airflow.
# Jinja pré-rendu au runtime dans les overrides templatés de make_cloud_task.
BATCH_ID_JINJA = "{{ params.batch_id or var.value.batch_id }}"


@dag(
    dag_id="Training",
    schedule=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "training", "phase3"],
    catchup=False,
    max_active_runs=1,
    params={
        "batch_id": Param(
            None,
            type=["null", "integer"],
            description="Batch à réentraîner. None → lit Variable Airflow `batch_id`.",
        ),
        "retrain_strategy": Param(
            "compare",
            type="string",
            enum=["stateless", "stateful", "compare"],
            description=(
                "Trajectoire de retraining. À T.1 : informatif "
                "(seule la lignée stateless est câblée). "
                "Sera consommé par la topologie à partir de T.3."
            ),
        ),
    },
    doc_md=__doc__,
)
def training_dag():

    # ---------------------------------------------------------------- #
    # Fan-out : 5 base learners stateless en parallèle.                #
    # Concurrence effective = min(5, slots(training_pool)=3).          #
    # ---------------------------------------------------------------- #
    fit_tasks = []
    for experiment in BASE_LEARNERS:
        short = experiment.removeprefix("base_learner_")
        task = make_cloud_task(
            task_id=f"fit_{short}_stateless",
            experiment=experiment,
            cloud_action="fit_base_learner",
            cloud_timeout=7200,  # 2h — borne haute uniforme (validée avec user)
            overrides=[
                "retrain_strategy=stateless",
                f"mlflow.run_name={experiment}_stateless_b{BATCH_ID_JINJA}",
            ],
            # Filet Airflow > cloud_timeout : évite un worker bloqué si le
            # pod cloud ne rend jamais la main.
            execution_timeout=timedelta(hours=3),
        )
        fit_tasks.append(task)

    # Point de convergence : ancre pour T.2 (resolve_active + fan-out fusions).
    join_base_learners_stateless = EmptyOperator(
        task_id="join_base_learners_stateless"
    )

    fit_tasks >> join_base_learners_stateless


training_instance = training_dag()