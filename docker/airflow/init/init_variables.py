from airflow.models import Variable, Pool
from airflow.models import Pool
import os

# Init uniquement si pas déjà défini
if Variable.get("api_username", default_var=None) is None:
    Variable.set("api_username", os.getenv("API_USERNAME", "admin"))
if Variable.get("api_password", default_var=None) is None:
    api_pwd = os.getenv("API_PASSWORD")
    if not api_pwd:
        raise RuntimeError("API_PASSWORD env var is not set.")
    Variable.set("api_password", api_pwd)
if Variable.get("predict_queue_threshold", default_var=None) is None:
    Variable.set("predict_queue_threshold", "50")


# Définition d'un pool Airflow afin 
@provide_session
def init_pool(session=None):
    if not session.query(Pool).filter(Pool.pool == "training_pool").first():
        Pool.create_or_update_pool(
            name="training_pool",
            slots=1,
            description="Mutex pour entraînement",
            include_deferred=False,
            session=session,
        )