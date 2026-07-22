# Pin explicite ([D-T2.4]) : `latest` flottant = risque de migration de schéma
# PostgreSQL silencieuse et irréversible au premier démarrage post-rebuild.
# Version = celle effectivement en prod au moment du pin (2026-07).
FROM ghcr.io/mlflow/mlflow:v3.11.1

USER root
RUN apt-get update && apt-get install -y git gcc libpq-dev netcat curl &&  rm -rf /var/lib/apt/lists/*

RUN pip install psycopg2-binary

#script pour attendre que postegre se deploie
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

RUN useradd -ms /bin/bash mlflow
USER mlflow
