"""
I.4 — DAG Ingestion : pipeline d'ingestion d'un nouveau batch de données.

Enchaîne les 3 actions du BLOC I :
    1. ingest_batch(n)         → PythonOperator (local, léger)
    2. rebase_val_selection(n) → PythonOperator (local, léger)
    3. reevaluate_actives(n)   → BashOperator → submit_cloud → RunPod GPU
    4. increment_batch_id      → PythonOperator (incrémente batch_id si succès)

Le batch_id est lu depuis la Variable Airflow `batch_id`.
L'incrémentation n'a lieu QUE si les 3 étapes précédentes réussissent.
Si le DAG échoue, batch_id reste inchangé → relance idempotente.

Schedule : None (déclenché manuellement ou par SimulateTrainDataArrival).
Pool : training_pool pour reevaluate (évite conflit avec un training en cours).

Variables Airflow :
    - batch_id : int — numéro du batch courant
"""
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from cloud_task import make_cloud_task

@dag(
    dag_id="Ingestion",
    schedule=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "ingestion"],
    catchup=False,
    max_active_runs=1,
    doc_md=__doc__,
)
def ingestion_dag():

    @task()
    def ingest_batch(**context):
        """I.1 — Valide le batch, pose is_gold + text dans Mongo."""
        batch_id = int(Variable.get("batch_id", default_var=1))
        from src.data.ingest_batch import run_ingest_batch
        result = run_ingest_batch(batch_id=batch_id)
        print(f"[Ingestion DAG] ingest_batch({batch_id}) : {result}")
        return result

    @task()
    def rebase_val_selection(**context):
        """I.2 — Crée is_val_selection_v{n} dans Mongo (split 10% stratifié)."""
        version = int(Variable.get("batch_id", default_var=1))
        from src.data.rebase_val_selection import run_rebase_val_selection
        result = run_rebase_val_selection(version=version)
        print(f"[Ingestion DAG] rebase_val_selection(v{version}) : {result}")
        return result

    # I.3 — reevaluate_actives sur RunPod (GPU forward des base learners).
    # Migré de BashOperator vers make_cloud_task (T.5) :
    # exit 42 → retry pénurie GPU, exit ≠0 → fail-fast.
    reevaluate = make_cloud_task(
        task_id="reevaluate_actives",
        experiment="m2_best",
        cloud_action="reevaluate_actives",
        cloud_timeout=1800,
        overrides=["version={{ var.value.batch_id }}"],
        execution_timeout=timedelta(minutes=45),
    )
    # I.4 — Trigger Training [D-T5.1]
    # batch_id passé en conf AVANT l'incrément : la Variable va bumper
    # juste après, mais le Training reçoit la valeur figée.
    # wait_for_completion=False : Ingestion finit, Training tourne en parallèle.
    # max_active_runs=1 sur Training sérialise les trainings.
    
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="Training",
        conf={
            "batch_id": "{{ var.value.batch_id | int }}",
            "retrain_strategy": "compare",
        },
        wait_for_completion=False,
    )

    @task(trigger_rule="all_success")
    def increment_batch_id(**context):
        """Incrémente batch_id seulement si les 3 étapes ont réussi."""
        current = int(Variable.get("batch_id", default_var=1))
        new_id = current + 1
        Variable.set("batch_id", str(new_id))
        print(f"[Ingestion DAG] batch_id incrémenté : {current} → {new_id}")
        return {"previous": current, "new": new_id}

    ingest_batch() >> rebase_val_selection() >> reevaluate >> increment_batch_id()


ingestion_instance = ingestion_dag()
