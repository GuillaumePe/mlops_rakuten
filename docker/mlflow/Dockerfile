FROM ghcr.io/mlflow/mlflow:latest

USER root
RUN apt-get update && apt-get install -y git gcc libpq-dev netcat curl &&  rm -rf /var/lib/apt/lists/*

RUN pip install psycopg2-binary

#script pour attendre que postegre se deploie
COPY wait-for-it.sh /wait-for-it.sh
RUN chmod +x /wait-for-it.sh

RUN useradd -ms /bin/bash mlflow
USER mlflow
