DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'airflow') THEN
      CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow';
   END IF;
END
$$;

CREATE DATABASE airflow_db OWNER airflow;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO airflow;

-- Créer le rôle et la base pour MLflow
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mlflow') THEN
      CREATE ROLE mlflow WITH LOGIN PASSWORD 'mlflowpwd';
   END IF;
END
$$;

DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db') THEN
      CREATE DATABASE mlflow_db OWNER mlflow;
   END IF;
END
$$;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;

-- Donner les permissions sur les schémas pour Airflow
\connect airflow_db
GRANT ALL ON SCHEMA public TO airflow;
ALTER SCHEMA public OWNER TO airflow;

-- Donner les permissions sur les schémas pour MLflow
\connect mlflow_db
GRANT ALL ON SCHEMA public TO mlflow;
ALTER SCHEMA public OWNER TO mlflow;