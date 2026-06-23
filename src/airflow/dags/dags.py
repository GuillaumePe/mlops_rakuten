from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.exceptions import AirflowSkipException
import requests

from simulator import inject_random_batch

# Appel au endpoint /login pour récupérer le token
def get_auth_headers():
    username = Variable.get("api_username")
    password = Variable.get("api_password")
    
    response = requests.post(
        "http://api:8000/login",
        data={"username": username, "password": password}
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}




# === DAG Training (manuel) ===
@dag(
    dag_id="Training",
    schedule_interval=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN"],
    catchup=False
)
def batch_training():

    @task(pool="training_pool", priority_weight=10)
    def set_batch_id_and_preprocessing():
        headers = get_auth_headers()
        batch_id = int(Variable.get("batch_id", 0)) + 1
        Variable.set("batch_id", str(batch_id))
        payload = {"batch_id": batch_id}
        requests.post("http://api:8000/preprocessing", json=payload,headers=headers)

    @task(pool="training_pool", priority_weight=10)
    def trigger_training():
        headers = get_auth_headers()
        requests.post("http://api:8000/training",headers=headers)
        
        
    
    @task(pool="training_pool", priority_weight=10)
    def rescore_all():
        headers = get_auth_headers()
        requests.post("http://api:8000/rescore_all", headers=headers)

    set_batch_id_and_preprocessing() >> trigger_training() >> rescore_all()

batch_training_instance = batch_training()


# === DAG SimulateDataArrival (simulation) ===
# injecte de la donnéee de manière random de X_to_predict_pool à X_to_predict ppur simuler l'arrivée de nouvelle données à prédire
@dag(
    dag_id="SimulateDataArrival",
    schedule="*/5 * * * *",
    start_date=days_ago(1),
    tags=["simulation", "MAR25_CMLOPS_RAKUTEN"],
    catchup=False,
)
def simulate_data_arrival():

    @task()
    def inject():
        result = inject_random_batch(min_size=20, max_size=100)
        print(f"Injection : {result}")
        return result

    inject()


simulate_arrival_instance = simulate_data_arrival()
