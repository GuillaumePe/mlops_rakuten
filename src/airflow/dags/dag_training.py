"""
T.1 — DAG Training : fan-out des 5 base learners stateless.

Version squelette (T.1) : ne couvre que la lignée STATELESS des base learners.
Les 7 fusions stateless seront ajoutées en T.2 ; la lignée stateful en T.3 ;
le comparateur + promotions en T.4 ; le trigger auto par Ingestion en T.5.

Doctrine :
    - DAG mince / Python épais : chaque tâche = un submit_cloud → RunPod.
      Aucune logique métier locale.
    - Retry pénurie / fail-fast bug : géré par make_cloud_task
      (exit 42 → retry backoff exponentiel ; exit ≠0 → fail-fast).
    - Pool training_pool (3 slots GPU) : concurrence bornée.

Paramètres (via dag_run.conf ou l'UI "Trigger DAG w/ config") :
    - batch_id (int|null, défaut = Variable Airflow `batch_id`)
    - retrain_strategy (str, défaut "compare") — informatif à T.1 ;
      utilisé pour brancher la topologie à partir de T.3.

Convention run_name (§3.4 du plan) :
    {experiment}_stateless_b{batch_id}
"""
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

from cloud_task import make_cloud_task


# 5 base learners de Phase 1, tous entraînés en stateless à chaque batch.
# Noms exacts = clés de EXPERIMENT_BUILDERS dans src/experiments/runner.py.
BASE_LEARNERS = [
    "base_learner_textcnn",
    "base_learner_camembert_lora",
    "base_learner_siglip2",
    "base_learner_resnet18_full_ft",
    "base_learner_resnet50_partial_ft",
]

# batch_id : priorité dag_run.conf > Variable Airflow.
# Jinja pré-rendu au runtime dans les overrides templatés de make_cloud_task.
BATCH_ID_JINJA = "{{ params.batch_id or var.value.batch_id }}"

# 7 fusions, indexées par (experiment_name, cloud_action).
# M2 = SklearnExperiment (action "fit"), M3 = LightningExperiment ("fit_lightning").
FUSIONS_STATELESS = {
    # experiment             cloud_action
    "m2_benchmark":          "fit",
    "m2_frugal_ft":          "fit",
    "m2_best":               "fit",
    "m3_attention_fusion":      "fit_lightning",
    "m3_attention_fusion_best": "fit_lightning",
    "m3_hpo_best":              "fit_lightning",
    "m3_2_coadaptation":        "fit_lightning",
}

# Helper : référence Jinja vers un champ du XCom de resolve_active_stateless.
_XCOM_PREFIX = "{{ ti.xcom_pull(task_ids='resolve_active_stateless')"

# Stateful : mêmes fusions, TextCNN exclu des base learners (stateless-only).
BASE_LEARNERS_STATEFUL = [
    bl for bl in BASE_LEARNERS if bl != "base_learner_textcnn"
]
FUSIONS_STATEFUL = dict(FUSIONS_STATELESS)

# Registry model names par fusion — pour construire l'URI de warm-start
# stateful (models:/REGISTRY@champion_stateful). Fallback cross-lignée
# géré par apply_warm_start [D-T3.5].
FUSION_REGISTRY = {
    "m2_benchmark":             "rakuten-m2-benchmark",
    "m2_frugal_ft":             "rakuten-m2-frugal-ft",
    "m2_best":                  "rakuten-m2-best",
    "m3_attention_fusion":      "rakuten-m3-attention-fusion",
    "m3_attention_fusion_best": "rakuten-m3-attention-fusion",
    "m3_hpo_best":              "rakuten-m3-attention-fusion",
    "m3_2_coadaptation":        "rakuten-m3-2-coadaptation",
}

def _xcom_ref(modality: str, key: str) -> str:
    """Construit une expression Jinja vers resolve_active_stateless XCom."""
    return _XCOM_PREFIX + "['" + modality + "']['" + key + "'] }}"

@dag(
    dag_id="Training",
    schedule=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "training", "phase3"],
    catchup=False,
    max_active_runs=1,
    params={
        "batch_id": Param(
            None,
            type=["null", "integer"],
            description="Batch à réentraîner. None → lit Variable Airflow `batch_id`.",
        ),
        "retrain_strategy": Param(
            "compare",
            type="string",
            enum=["stateless", "stateful", "compare"],
            description=(
                "Trajectoire de retraining. À T.1 : informatif "
                "(seule la lignée stateless est câblée). "
                "Sera consommé par la topologie à partir de T.3."
            ),
        ),
    },
    doc_md=__doc__,
)
def training_dag():

    # ---------------------------------------------------------------- #
    # Branching [D-T3.1] : sélection des lignées.                      #
    # batch_id == 1 → stateful exclue (pas d'ancre @active_stateful).  #
    # Fallback cross-lignée [D-T3.5] en warm_start.py si batch 2.      #
    # ---------------------------------------------------------------- #
    @task.branch(task_id="select_lineages")
    def select_lineages(**context):
        from airflow.models import Variable

        strategy = context["params"]["retrain_strategy"]
        batch_id = int(
            context["params"].get("batch_id") or Variable.get("batch_id")
        )

        branches = []
        if strategy in ("stateless", "compare"):
            branches.append("gate_stateless")
        if strategy in ("stateful", "compare") and batch_id > 1:
            branches.append("gate_stateful")

        if not branches:
            raise ValueError(
                f"Aucune lignée à lancer : strategy={strategy}, "
                f"batch_id={batch_id}. "
                f"Stateful requiert batch_id > 1."
            )

        print(f"[select_lineages] strategy={strategy}, batch_id={batch_id} "
              f"→ {branches}")
        return branches

    branching = select_lineages()

    gate_stateless = EmptyOperator(task_id="gate_stateless")
    gate_stateful = EmptyOperator(task_id="gate_stateful")

    branching >> [gate_stateless, gate_stateful]

    # ---------------------------------------------------------------- #
    # Fan-out : 5 base learners stateless en parallèle.                #
    # Concurrence effective = min(5, slots(training_pool)=3).          #
    # ---------------------------------------------------------------- #
    fit_tasks = []
    for experiment in BASE_LEARNERS:
        short = experiment.removeprefix("base_learner_")
        fit_task  = make_cloud_task(
            task_id=f"fit_{short}_stateless",
            experiment=experiment,
            cloud_action="fit_base_learner",
            cloud_timeout=7200,  # 2h — borne haute uniforme (validée avec user)
            overrides=[
                "retrain_strategy=stateless",
                f"mlflow.run_name={experiment}_stateless_b{BATCH_ID_JINJA}",
            ],
            # Filet Airflow > cloud_timeout : évite un worker bloqué si le
            # pod cloud ne rend jamais la main.
            execution_timeout=timedelta(hours=3),
        )
        fit_tasks.append(fit_task)

    # Point de convergence : ancre pour T.2 (resolve_active + fan-out fusions).
    join_base_learners_stateless = EmptyOperator(
        task_id="join_base_learners_stateless"
    )

    gate_stateless >> fit_tasks >> join_base_learners_stateless

    @task()
    def resolve_active_stateless(**context):
        """
        Résout le meilleur base learner par modalité pour la lignée stateless.

        Appelle resolve_active_for_fusion("stateless") et normalise le retour
        (tuples) en dicts JSON-safe pour XCom [D-T2.2].

        Returns (XCom push) :
            {"text":  {"registry_name": "rakuten-base-camembert-lora",
                       "name": "camembert_lora", "version": 9, "embed_dim": 768},
             "image": {"registry_name": "rakuten-base-siglip2",
                       "name": "siglip2", "version": 4, "embed_dim": 768}}
        """
        import os
        import sys
    
        project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import mlflow
        from src.models.utils import resolve_active_for_fusion
        from src.experiments.runner import LEARNER_EMBED_DIM

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )

        raw = resolve_active_for_fusion("stateless")
        result = {}
        for mod, info in raw.items():
            reg_name, ver = info[0], info[1]
            short = reg_name.replace("rakuten-base-", "").replace("-", "_")
            result[mod] = {
                "registry_name": reg_name,
                "name": short,
                "version": ver,
                "embed_dim": LEARNER_EMBED_DIM[short],
            }
        print(f"[resolve_active_stateless] {result}")
        return result
    
    resolved_stateless = resolve_active_stateless()
    join_base_learners_stateless >> resolved_stateless

    # ---------------------------------------------------------------- #
    # Fan-out : 7 fusions stateless, consommant les BL épinglés.       #
    # Overrides XCom → versions déterministes (pas de re-résolution    #
    # d'alias au runtime du pod) [D-T2.5].                             #
    # promotion.enabled=false : la promotion est gérée en T.4          #
    # (eval_gold_champion + compare_and_promote en local).             #
    # ---------------------------------------------------------------- #
    fusion_tasks = []
    for experiment, cloud_action in FUSIONS_STATELESS.items():
        overrides = [
            "retrain_strategy=stateless",
            f"mlflow.run_name={experiment}_stateless_b{BATCH_ID_JINJA}",
            "promotion.enabled=false",
            # Pin base learners depuis XCom resolve_active_stateless
            f"base_learners.text.registry_name={_xcom_ref('text', 'registry_name')}",
            f"base_learners.text.name={_xcom_ref('text', 'name')}",
            f"base_learners.text.version={_xcom_ref('text', 'version')}",
            f"base_learners.text.embed_dim={_xcom_ref('text', 'embed_dim')}",
            f"base_learners.image.registry_name={_xcom_ref('image', 'registry_name')}",
            f"base_learners.image.name={_xcom_ref('image', 'name')}",
            f"base_learners.image.version={_xcom_ref('image', 'version')}",
            f"base_learners.image.embed_dim={_xcom_ref('image', 'embed_dim')}",
        ]
        fusion_task = make_cloud_task(
            task_id=f"fit_{experiment}_stateless",
            experiment=experiment,
            cloud_action=cloud_action,
            cloud_timeout=7200,
            overrides=overrides,
            execution_timeout=timedelta(hours=3),
        )
        fusion_tasks.append(fusion_task)

    # Point de convergence fusions stateless — ancre pour T.4
    # (eval_gold_champion + compare_and_promote).
    join_fusions_stateless = EmptyOperator(
        task_id="join_fusions_stateless"
    )

    resolved_stateless >> fusion_tasks >> join_fusions_stateless

    # ================================================================ #
    # LIGNÉE STATEFUL                                                  #
    # 4 BL (TextCNN exclu), warm-start BL depuis @active_stateful,     #
    # warm-start fusions depuis @champion_stateful.                    #
    # Fallback cross-lignée [D-T3.5] si alias inexistant (batch 2).    #
    # ================================================================ #
    fit_tasks_sf = []
    for experiment in BASE_LEARNERS_STATEFUL:
        short = experiment.removeprefix("base_learner_")
        fit_task_sf = make_cloud_task(
            task_id=f"fit_{short}_stateful",
            experiment=experiment,
            cloud_action="fit_base_learner",
            cloud_timeout=7200,
            overrides=[
                "retrain_strategy=stateful",
                f"mlflow.run_name={experiment}_stateful_b{BATCH_ID_JINJA}",
                f"warm_start_from=models:/rakuten-base-{short}@active_stateful",
            ],
            execution_timeout=timedelta(hours=3),
        )
        fit_tasks_sf.append(fit_task_sf)

    join_base_learners_stateful = EmptyOperator(
        task_id="join_base_learners_stateful"
    )
    gate_stateful >> fit_tasks_sf >> join_base_learners_stateful

    # ---- Resolve active stateful ---- #
    @task()
    def resolve_active_stateful(**context):
        """Même logique que resolve_active_stateless, pour la lignée stateful."""
        import os
        import sys

        project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import mlflow
        from src.models.utils import resolve_active_for_fusion
        from src.experiments.runner import LEARNER_EMBED_DIM

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )

        raw = resolve_active_for_fusion("stateful")
        result = {}
        for mod, info in raw.items():
            reg_name, ver = info[0], info[1]
            short = reg_name.replace("rakuten-base-", "").replace("-", "_")
            result[mod] = {
                "registry_name": reg_name,
                "name": short,
                "version": ver,
                "embed_dim": LEARNER_EMBED_DIM[short],
            }
        print(f"[resolve_active_stateful] {result}")
        return result

    resolved_stateful = resolve_active_stateful()
    join_base_learners_stateful >> resolved_stateful

    # ---- Fan-out 7 fusions stateful ---- #
    _XCOM_SF = "{{ ti.xcom_pull(task_ids='resolve_active_stateful')"

    def _xcom_sf_ref(modality: str, key: str) -> str:
        return _XCOM_SF + "['" + modality + "']['" + key + "'] }}"

    fusion_tasks_sf = []
    for experiment, cloud_action in FUSIONS_STATEFUL.items():
        registry = FUSION_REGISTRY[experiment]
        fusion_task_sf = make_cloud_task(
            task_id=f"fit_{experiment}_stateful",
            experiment=experiment,
            cloud_action=cloud_action,
            cloud_timeout=7200,
            overrides=[
                "retrain_strategy=stateful",
                f"mlflow.run_name={experiment}_stateful_b{BATCH_ID_JINJA}",
                "promotion.enabled=false",
                f"warm_start_from=models:/{registry}@champion_stateful",
                f"base_learners.text.registry_name={_xcom_sf_ref('text', 'registry_name')}",
                f"base_learners.text.name={_xcom_sf_ref('text', 'name')}",
                f"base_learners.text.version={_xcom_sf_ref('text', 'version')}",
                f"base_learners.text.embed_dim={_xcom_sf_ref('text', 'embed_dim')}",
                f"base_learners.image.registry_name={_xcom_sf_ref('image', 'registry_name')}",
                f"base_learners.image.name={_xcom_sf_ref('image', 'name')}",
                f"base_learners.image.version={_xcom_sf_ref('image', 'version')}",
                f"base_learners.image.embed_dim={_xcom_sf_ref('image', 'embed_dim')}",
            ],
            execution_timeout=timedelta(hours=3),
        )
        fusion_tasks_sf.append(fusion_task_sf)

    join_fusions_stateful = EmptyOperator(
        task_id="join_fusions_stateful"
    )
    resolved_stateful >> fusion_tasks_sf >> join_fusions_stateful
    
training_instance = training_dag()