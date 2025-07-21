from airflow.models import Variable

# Init uniquement si pas déjà défini
if Variable.get("api_username", default_var=None) is None:
    Variable.set("api_username", "admin")

if Variable.get("api_password", default_var=None) is None:
    Variable.set("api_password", "123admin")

if Variable.get("is_training_running", default_var=None) is None:
    Variable.set("is_training_running", "false")

