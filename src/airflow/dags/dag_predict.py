"""
P.3 — DAG Predict (système neuf).

Remplace le DAG Predict legacy qui appelait POST /predict_pending via l'API.
Appelle directement run_predict_pending() via PythonOperator.

Schedule : */10 * * * * (identique au legacy).
Pool : training_pool (évite conflit avec un training en cours).
Doctrine : DAG mince / Python épais — toute la logique est dans l'action.

Variables Airflow (optionnelles, avec défauts) :
    - predict_queue_threshold : int (défaut 50)
    - champion_model_name : str (défaut "rakuten-m2-best")
"""
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.utils.dates import days_ago
from src.models.predict_pending import run_predict_pending

@dag(
    dag_id="Predict",
    schedule="*/10 * * * *",
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "predict", "phase3"],
    catchup=False,
    max_active_runs=1,
    doc_md=__doc__,
)
def predict_dag():

    @task(pool="training_pool")
    def score_pending(**context):
        """Score les samples en attente via RakutenScorer."""
        

        threshold = int(Variable.get("predict_queue_threshold", default_var=50))
        model_name = Variable.get("champion_model_name", default_var="rakuten-m2-best")

        result = run_predict_pending(
            threshold=threshold,
            model_name=model_name,
        )
        print(f"[Predict DAG] {result}")
        return result

    score_pending()


predict_instance = predict_dag()
