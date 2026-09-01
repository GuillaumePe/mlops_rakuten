"""
P4.5d — DAG Predict_Y_test : scoring complet de X_test pour soumission ENS.

Déclenché EVENT-BASED par la fin de Training (TriggerDagRunOperator posé après
le tournament), une fois `@production` réattribué au vainqueur cross-model.
Ne fait PAS partie de la chaîne de trigger du cycle : il produit une soumission,
il ne relance pas Ingestion.

Topologie (P4.5d-1) :
    check_production (LOCAL, gratuit) → score_test (CLOUD GPU)

`build_submission` (tâche locale, P4.5d-2) sera greffée en aval une fois réglé
le point d'écriture disque (le projet est monté :ro dans le conteneur Airflow).

Pourquoi un garde LOCAL avant le pod :
    resolve_production_model() lève si 0 ou >1 porteur de @production. Sans
    garde, cette erreur ne se manifeste QUE sur le pod, après pull d'image
    (~2 min) et pull DVC des images de test (~3-5 min) — soit ~8 min de GPU
    facturé pour un échec déterministe détectable en 200 ms en local, via le
    mlflow-skinny déjà présent dans l'image Airflow. Le garde est du fail-fast
    à coût nul.

Paramètres (dag_run.conf ou UI "Trigger DAG w/ config") :
    - batch_id (int|null) : batch venant d'être entraîné. Estampille des
      prédictions dans Prediction_test ET clé d'idempotence.

⚠ Fallback Variable `batch_id` : la Variable est déjà à n+1 quand Training(n)
  tourne (Ingestion l'incrémente juste après le trigger). Pour un run MANUEL,
  passer batch_id explicitement en conf ; le fallback Variable est un filet,
  pas la voie normale (warning loggé).

⚠ Image trainer : `cloud_image=None` → submit.py utilise `GHCR_IMAGE_TRAINER`
  (env du compose). Cette variable DOIT pointer sur le tag contenant P4.5a/b
  (`predict_test_pool` + handler runner), pas sur un tag antérieur. RunPod
  cache par tag : un tag stale = handler absent = exit ≠ 0.
"""
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.models.param import Param
from airflow.utils.dates import days_ago
from cloud_task import make_cloud_task

# Cascade identique à dag_training.BATCH_ID_JINJA :
# dag_run.conf (trigger depuis Training) → params (UI) → Variable (filet).
BATCH_ID_JINJA = (
    "{{ dag_run.conf.get('batch_id', params.batch_id) or var.value.batch_id }}"
)

# Experiment porteur : les actions "data" ne dépendent pas de la config du
# modèle (early-return dans le runner avant tout build d'experiment), mais
# --experiment est requis par l'argparse. Convention reprise de dag_ingestion
# (reevaluate_actives) pour ne pas multiplier les conventions.
CARRIER_EXPERIMENT = "m2_best"

# Le pod n'a besoin QUE des images de test : predict_test_pool lit les samples
# dans Mongo (via tunnel), pas dans les CSV. Ce target pilote aussi
# SKIP_IMAGE_EXTRACT=true côté submit.py (pas d'archive train à extraire).
DVC_TARGETS_TEST = ["data/raw_data_test/images_test.dvc"]


@dag(
    dag_id="Predict_Y_test",
    schedule=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "predict", "phase4"],
    catchup=False,
    max_active_runs=1,
    params={
        "batch_id": Param(
            None,
            # "string" accepté volontairement : Jinja rend TOUJOURS du texte,
            # sauf si le DAG appelant déclare render_template_as_native_obj.
            # dag_ingestion l'a, dag_training ne l'a pas — son trigger envoie
            # donc conf={"batch_id": "3"} et un Param strictement integer
            # rejette le run AVANT toute exécution. Les consommateurs font
            # int() de leur côté (check_production, override YAML du pod).
            type=["null", "integer", "string"],
            description=(
                "Batch venant d'être entraîné. None → Variable Airflow "
                "`batch_id` (⚠ déjà à n+1 pendant Training(n))."
            ),
        ),
    },
    doc_md=__doc__,
)
def predict_y_test_dag():

    @task()
    def check_production(**context):
        """
        Garde LOCAL : vérifie qu'un @production unique existe avant de payer un pod.

        Ne charge PAS le modèle (impossible en local : artefact M3 CUDA-natif),
        il ne fait que résoudre l'alias — lecture MLflow pure.
        """
        import os
        import sys

        project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import mlflow
        from src.models.utils import resolve_production_model

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )

        conf = context["dag_run"].conf or {}
        from_conf = conf.get("batch_id") or context["params"].get("batch_id")
        if from_conf is None:
            print(
                "[check_production] ⚠ batch_id absent de la conf → fallback "
                "Variable. Si ce run n'a PAS été déclenché par Training, la "
                "Variable peut valoir n+1 (off-by-one sur l'estampille)."
            )
        batch_id = int(from_conf or Variable.get("batch_id"))

        name, version = resolve_production_model()
        print(f"[check_production] @production = {name} v{version} "
              f"| batch_id ciblé = {batch_id}")

        return {"model_name": name, "version": int(version), "batch_id": batch_id}

    # ---------------------------------------------------------------- #
    # Scoring CLOUD : forward M3 sur les 13812 samples de X_test.       #
    # Pas de --limit → tout le pool (une soumission ENS partielle est   #
    # invalide ; build_submission la refuserait de toute façon).        #
    # ---------------------------------------------------------------- #
    score_test = make_cloud_task(
        task_id="score_test",
        experiment=CARRIER_EXPERIMENT,
        cloud_action="predict_test_pool",
        # Forward seul, mais borne haute : pull image + pull DVC images_test
        # + 13812 forwards multimodaux. 1h30 laisse de la marge sans autoriser
        # un pod zombie.
        cloud_timeout=5400,
        overrides=[f"batch_id={BATCH_ID_JINJA}"],
        extra_args=["--cloud-dvc-targets", *DVC_TARGETS_TEST],
        # Filet Airflow > cloud_timeout (convention dag_training).
        execution_timeout=timedelta(hours=2, minutes=30),
    )

    @task()
    def build_submission(guard: dict, **context):
        """
        Assemble le CSV de soumission ENS (LOCAL, ~1 s).

        Jointure stdlib entre Prediction_test (Mongo) et X_test_update.csv,
        seul porteur de l'index ENS (drop("") a l'upload). Le CSV est la
        reference d'iteration : toute ligne sans prediction fait echouer la
        tache plutot que de produire une soumission partielle, qui serait
        rejetee par l'ENS.

        batch_id vient de l'XCom de check_production : meme valeur que celle
        passee au pod, donc aucune divergence possible entre l'estampille des
        predictions et celle de la soumission.
        """
        import os
        import sys

        project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.models.build_submission import run_build_submission

        # Chemins explicites : DATA_ROOT vaut "." par defaut et le cwd du
        # worker Airflow n'est PAS /opt/project. Seul data/submissions est
        # monte en rw (bind imbrique surchargeant le :ro parent).
        result = run_build_submission(
            batch_id=guard["batch_id"],
            csv_path=f"{project_root}/data/raw_data_test/X_test_update.csv",
            output_dir=f"{project_root}/data/submissions",
        )
        print(
            f"[build_submission] {result['n_rows']} lignes -> {result['path']} "
            f"| modele {result['model']}"
        )
        return result

    guard = check_production()
    submission = build_submission(guard)

    # guard >> score_test : fail-fast avant le pod.
    # score_test >> submission : l'XCom seul ne suffirait pas, submission
    # depend de guard et non de score_test — sans cette arete elle partirait
    # en parallele du scoring et lirait un Prediction_test vide.
    guard >> score_test >> submission


predict_y_test_instance = predict_y_test_dag()
