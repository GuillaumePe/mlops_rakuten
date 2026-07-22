# Image Airflow mince pour l'option B : orchestration + soumission RunPod.
# AUCUNE dep ML (torch/lightning/mlflow) — le calcul lourd reste sur le pod.
FROM apache/airflow:2.8.1-python3.10

# Deps du chemin submit (src.cloud.submit) + pymongo (déjà utilisé par les DAGs).
# Pinne les versions à celles de ton .venv hôte pour la parité (voir note).
# mlflow-skinny (T.2a, [D-T2.1]) : client registry léger (~sans deps lourdes)
# pour les tâches LOCALES resolve_active_* (T.2) et compare_and_promote (T.4).
# Version alignée sur le SERVEUR MLflow (3.11.1) — jamais torch ici (piège #9).

RUN pip install --no-cache-dir \
        runpod==1.9.0 \
        python-dotenv==1.2.2 \
        pymongo==3.12.0 \
        mlflow-skinny==3.11.1