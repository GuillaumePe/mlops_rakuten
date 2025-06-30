from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.models import Variable
import requests
from pymongo import MongoClient
import subprocess
from datetime import timedelta
import random

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
# DAG 1: Batch Training toutes les 2 heures
@dag(
    dag_id="Training",
    schedule_interval="0 */2 * * *",
    start_date=days_ago(1),
    tags=["Datascientest","MAR25_CMLOPS_RAKUTEN"],
    catchup=False
)
def batch_training():

    @task()
    def set_batch_id_and_preprocessing():
        headers = get_auth_headers()
        batch_id = int(Variable.get("batch_id", 0)) + 1
        Variable.set("batch_id", str(batch_id))
        payload = {"batch_id": batch_id}
        requests.post("http://api:8000/preprocessing", json=payload,headers=headers)

    @task()
    def trigger_training():
        headers = get_auth_headers()
        try:
            Variable.set("is_training_running", "true")
            requests.post("http://api:8000/training",headers=headers)
        finally:
            Variable.set("is_training_running", "false")
        
    
    @task()
    def predict_full_after_training():
        headers = get_auth_headers()
        client = MongoClient("mongodb://mongodb:27017")
        db = client["MAR25_CMLOPS_RAKUTEN"]
        all_ids = [doc["productid"] for doc in db["X_to_predict"].find({}, {"productid":1})]
        if all_ids:
            payload = {"productid": all_ids}
            requests.post("http://api:8000/predict", json=payload,headers=headers)

    set_batch_id_and_preprocessing() >> trigger_training() >> predict_full_after_training()

batch_training_instance = batch_training()

# DAG 2: Predict toutes les 30 min
@dag(
    dag_id="Predict",
    schedule_interval="*/15 * * * *",
    start_date=days_ago(1),
    tags=["Datascientest","MAR25_CMLOPS_RAKUTEN"],
    catchup=False
)
def predict():

    @task()
    def run_predict():
        headers = get_auth_headers()
        if Variable.get("is_training_running", default_var="false") == "true":
            return

        client = MongoClient("mongodb://mongodb:27017")
        db = client["MAR25_CMLOPS_RAKUTEN"]

        # Récupére tous les productid de X_to_predict
        x_to_predict_ids = set(doc["productid"] for doc in db["X_to_predict"].find({}, {"productid":1}))
        # Récupére les productid déjà prédits
        predicted_ids = set(doc["productid"] for doc in db["Prediction"].find({}, {"productid":1}))

        # Différence = les productid encore disponibles
        available_ids = list(x_to_predict_ids - predicted_ids)

        sample_size = int(Variable.get("predict_sample_size", 100))
        selected_ids = random.sample(available_ids, min(sample_size, len(available_ids)))

        if selected_ids:
            payload = {"productid": selected_ids}
            requests.post("http://api:8000/predict", json=payload,header=headers)

    run_predict()

predict_instance = predict()