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
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
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
# Cascade : dag_run.conf (TriggerDagRunOperator depuis Ingestion)
#         → params (UI "Trigger DAG w/ config")
#         → Variable Airflow (fallback)
BATCH_ID_JINJA = "{{ dag_run.conf.get('batch_id', params.batch_id) or var.value.batch_id }}"
# [D-T5.2] Override commun à TOUTES les tâches fit : pilote la dérivation
# train_batches=[1..n] dans le runner (et le run_name l'utilise déjà).
BATCH_ID_OVERRIDE = f"batch_id={BATCH_ID_JINJA}"

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

# Experiment MLflow par fusion — pour que eval_gold et challenger
# soient dans le même experiment [D-T4.1].
# Valeur = config["mlflow"]["experiment_name"] de chaque YAML.
FUSION_EXPERIMENT = {
    "m2_benchmark":             "M2_benchmark_phase1",
    "m2_frugal_ft":             "M2_frugal_ft_phase1",
    "m2_best":                  "M2_best_phase1",
    "m3_attention_fusion":      "M3_attention_fusion",
    "m3_attention_fusion_best": "M3_attention_fusion",
    "m3_hpo_best":              "M3_attention_fusion",
    "m3_2_coadaptation":        "M3_2_coadaptation",
}
# Fusions dont les base learners sont FIXES (définis dans leur YAML) :
# le DAG ne doit PAS override base_learners.* — elles réentraînent leurs
# propres BL (textcnn/resnet50 pour benchmark, camembert_lora/resnet18 pour
# frugal) sur les données croissantes, pour mesurer l'effet du volume à
# architecture constante. Overrider les transformerait en clones de m2_best.
FUSIONS_BL_FIXES = {"m2_benchmark", "m2_frugal_ft"}


STRATEGIES = ("stateless", "stateful")

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
        

        strategy = context["params"]["retrain_strategy"]
        # Cascade conf → params → Variable (cohérent avec BATCH_ID_JINJA)
        conf = context["dag_run"].conf or {}
        batch_id = int(
            conf.get("batch_id")
            or context["params"].get("batch_id")
            or Variable.get("batch_id")
        )
        strategy = conf.get("retrain_strategy", strategy)

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
                BATCH_ID_OVERRIDE,
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

        conf = context["dag_run"].conf or {}
        batch_id = int(
            conf.get("batch_id")
            or context["params"].get("batch_id")
            or Variable.get("batch_id")
        )

        os.environ["ACTIVE_VAL_SELECTION_VERSION"] = str(batch_id)
        import mlflow
        from src.models.utils import resolve_active_for_fusion
        from src.models.learner_registry import LEARNER_EMBED_DIM

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
            BATCH_ID_OVERRIDE,
            "retrain_strategy=stateless",
            f"mlflow.run_name={experiment}_stateless_b{BATCH_ID_JINJA}",
            "promotion.enabled=false",
        ]
        # BL dynamiques (m2_best, M3) : pin depuis XCom resolve_active_stateless.
        # BL fixes (benchmark/frugal) : gardent leur YAML statique, pas d'override.
        if experiment not in FUSIONS_BL_FIXES:
            overrides += [
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
                BATCH_ID_OVERRIDE,
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

        conf = context["dag_run"].conf or {}
        batch_id = int(
            conf.get("batch_id")
            or context["params"].get("batch_id")
            or Variable.get("batch_id")
        )
        
        os.environ["ACTIVE_VAL_SELECTION_VERSION"] = str(batch_id)
        import mlflow
        from src.models.utils import resolve_active_for_fusion
        from src.models.learner_registry import LEARNER_EMBED_DIM

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
        overrides_sf = [
            BATCH_ID_OVERRIDE,
            "retrain_strategy=stateful",
            f"mlflow.run_name={experiment}_stateful_b{BATCH_ID_JINJA}",
            "promotion.enabled=false",
            f"warm_start_from=models:/{registry}@champion_stateful",
        ]
        # Idem stateless : BL fixes gardent leur YAML, dynamiques pinnés XCom.
        if experiment not in FUSIONS_BL_FIXES:
            overrides_sf += [
                f"base_learners.text.registry_name={_xcom_sf_ref('text', 'registry_name')}",
                f"base_learners.text.name={_xcom_sf_ref('text', 'name')}",
                f"base_learners.text.version={_xcom_sf_ref('text', 'version')}",
                f"base_learners.text.embed_dim={_xcom_sf_ref('text', 'embed_dim')}",
                f"base_learners.image.registry_name={_xcom_sf_ref('image', 'registry_name')}",
                f"base_learners.image.name={_xcom_sf_ref('image', 'name')}",
                f"base_learners.image.version={_xcom_sf_ref('image', 'version')}",
                f"base_learners.image.embed_dim={_xcom_sf_ref('image', 'embed_dim')}",
            ]
        fusion_task_sf = make_cloud_task(
            task_id=f"fit_{experiment}_stateful",
            experiment=experiment,
            cloud_action=cloud_action,
            cloud_timeout=7200,
            overrides=overrides_sf,
            execution_timeout=timedelta(hours=3),
        )
        fusion_tasks_sf.append(fusion_task_sf)

    join_fusions_stateful = EmptyOperator(
        task_id="join_fusions_stateful"
    )
    resolved_stateful >> fusion_tasks_sf >> join_fusions_stateful

    # ================================================================ #
    # EVAL GOLD + COMPARE & PROMOTE (×14 : 7 fusions × 2 lignées)     #
    # eval_gold = cloud (forward pass), compare = local (MLflow read). #
    # ================================================================ #
    all_promote_joins = []

    for strategy in STRATEGIES:
        join_fusions = (
            join_fusions_stateless if strategy == "stateless"
            else join_fusions_stateful
        )

        eval_tasks = []
        for experiment in FUSIONS_STATELESS:
            eval_task = make_cloud_task(
                task_id=f"eval_gold_{experiment}_{strategy}",
                experiment=experiment,
                cloud_action="eval_gold_champion",
                cloud_timeout=3600,  # forward only, 1h suffit
                overrides=[
                    f"promotion.champion_alias=champion_{strategy}",
                    f"mlflow.eval_gold_run_name=eval_gold_{experiment}_{strategy}_b{BATCH_ID_JINJA}",
                ],
                execution_timeout=timedelta(hours=2),
            )
            eval_tasks.append(eval_task)

        join_eval = EmptyOperator(
            task_id=f"join_eval_{strategy}",
            trigger_rule="none_failed_min_one_success",
        )
        join_fusions >> eval_tasks >> join_eval

        # ---- Compare & promote (local, mlflow-skinny) ---- #
        @task(task_id=f"compare_promote_{strategy}")
        def compare_promote(strategy=strategy, **context):
            """Compare chaque challenger à son champion et promeut si gain > epsilon."""
            import os
            import sys

            project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            import mlflow
            from src.models.compare_and_promote import run_compare_and_promote

            mlflow.set_tracking_uri(
                os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
            )

            # Cascade conf → params → Variable (cohérent avec select_lineages).
            # CRITIQUE : la Variable est déjà à n+1 quand Training(n) tourne
            # (incrémentée par Ingestion juste après le trigger).
            conf = context["dag_run"].conf or {}
            batch_id = int(
                conf.get("batch_id")
                or context["params"].get("batch_id")
                or Variable.get("batch_id")
            )

            results = {}
            for exp, registry in FUSION_REGISTRY.items():
                exp_name = FUSION_EXPERIMENT[exp]
                challenger_rn = f"{exp}_{strategy}_b{batch_id}"
                champion_rn = f"eval_gold_{exp}_{strategy}_b{batch_id}"

                try:
                    r = run_compare_and_promote(
                        registry_model_name=registry,
                        challenger_run_names=[challenger_rn],
                        champion_run_name=champion_rn,
                        batch_id=batch_id,
                        experiment_name=exp_name,
                        champion_alias=f"champion_{strategy}",
                    )
                    results[exp] = r
                    print(f"[compare_promote_{strategy}] {exp}: "
                          f"promoted={r['promoted']}, reason={r['reason']}")
                except Exception as e:
                    print(f"[compare_promote_{strategy}] {exp}: ERROR {e}")
                    results[exp] = {"error": str(e)}

            return results

        promote_task = compare_promote(strategy=strategy)
        join_eval >> promote_task
        all_promote_joins.append(promote_task)

    # ================================================================ #
    # TOURNAMENT — best-of-champions → @production [D-T4.4]            #
    # Cross-lignée, cross-archi. Alias @production posé sur le         #
    # registered model du vainqueur (Option C).                        #
    # ================================================================ #
    join_all = EmptyOperator(
        task_id="join_all_promotions",
        trigger_rule="none_failed_min_one_success",
    )
    all_promote_joins >> join_all

    @task()
    def tournament(**context):
        """Scanne tous les @champion_*, trouve le meilleur F1 gold, pose @production."""
        import os
        import sys

        project_root = os.getenv("RAKUTEN_PROJECT_ROOT", "/opt/project")
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri)

        from mlflow.exceptions import MlflowException

        # Cascade conf → params → Variable (identique à compare_promote).
        # CRITIQUE : la Variable vaut déjà n+1 pendant Training(n).
        conf = context["dag_run"].conf or {}
        batch_id = int(
            conf.get("batch_id")
            or context["params"].get("batch_id")
            or Variable.get("batch_id")
        )

        def _absent(exc: MlflowException) -> bool:
            """
            Alias/modèle inexistant (normal) vs panne MLflow (anormal).

            ⚠ MLflow ne signale PAS un alias manquant avec
            RESOURCE_DOES_NOT_EXIST mais avec :
                INVALID_PARAMETER_VALUE: Registered model alias X not found.
            Un code qui suggère une erreur d'appel là où il s'agit d'une
            absence nominale (@champion_stateful n'existe pas sur toutes les
            archis). C'est exactement ce que masquait l'ancien
            `except Exception: continue`.

            On accepte donc les deux codes, mais on exige « not found » dans
            le message pour INVALID_PARAMETER_VALUE : une panne de tracking
            ou un modèle inconnu produisent d'autres messages et doivent
            continuer de remonter — @production détermine ce qui est servi
            et soumis à l'ENS, l'échec silencieux reste interdit.
            """
            code = getattr(exc, "error_code", "") or ""
            msg = str(exc).lower()
            if "RESOURCE_DOES_NOT_EXIST" in code:
                return True
            return "INVALID_PARAMETER_VALUE" in code and "not found" in msg

        # ------------------------------------------------------------ #
        # 1. Scan des champions — classement COMPLET, pas seulement le  #
        #    vainqueur : c'est l'écart entre lignées et entre archis qui #
        #    porte l'information expérimentale, et il n'existe nulle     #
        #    part ailleurs (les alias MLflow n'ont pas d'historique).    #
        # ------------------------------------------------------------ #
        ranking = []
        for registry in sorted(set(FUSION_REGISTRY.values())):
            for strategy in STRATEGIES:
                alias = f"champion_{strategy}"
                try:
                    mv = client.get_model_version_by_alias(registry, alias)
                    run = client.get_run(mv.run_id)
                except MlflowException as e:
                    # Un champion peut légitimement ne pas exister (lignée
                    # stateful au batch 1). Une PANNE MLflow, non : la laisser
                    # passer élirait un vainqueur parmi un sous-ensemble
                    # arbitraire, et @production détermine ce qui est servi ET
                    # soumis à l'ENS. Échec silencieux interdit ici.
                    if _absent(e):
                        print(f"[tournament] {registry}@{alias} absent — ignoré.")
                        continue
                    raise

                f1 = run.data.metrics.get("eval_gold/f1_weighted")
                if f1 is None:
                    print(f"[tournament] {registry}@{alias} v{mv.version} "
                          f"sans eval_gold/f1_weighted — écarté.")
                    continue

                f1 = float(f1)
                print(f"[tournament] {registry}@{alias} v{mv.version} → F1={f1:.4f}")
                ranking.append({
                    "registry": registry,
                    "alias": alias,
                    "strategy": strategy,
                    "version": int(mv.version),
                    "f1": f1,
                })

        ranking.sort(key=lambda d: d["f1"], reverse=True)

        # ------------------------------------------------------------ #
        # 2. Porteur SORTANT de @production, capturé AVANT suppression.  #
        #    set_registered_model_alias écrase sans archiver : sans      #
        #    cette capture, la transition n → n+1 est irrécupérable.     #
        # ------------------------------------------------------------ #
        previous = None
        for registry in sorted(set(FUSION_REGISTRY.values())):
            try:
                mv_prod = client.get_model_version_by_alias(registry, "production")
                previous = f"{registry}@v{mv_prod.version}"
            except MlflowException as e:
                if not _absent(e):
                    raise

        def _log_tournament(winner: dict | None) -> None:
            """Run MLflow récapitulatif — même rôle que compare_and_promote."""
            mlflow.set_experiment("training_compare")
            with mlflow.start_run(run_name=f"tournament_b{batch_id}"):
                mlflow.set_tag("role", "tournament")
                mlflow.log_param("batch_id", batch_id)
                mlflow.log_param("n_candidates", len(ranking))
                mlflow.log_param("previous_production", previous or "none")
                for c in ranking:
                    key = f"ranking/{c['registry']}__{c['strategy']}"
                    mlflow.log_metric(key, c["f1"])
                    mlflow.log_param(f"version/{c['registry']}__{c['strategy']}",
                                     c["version"])
                if winner is None:
                    mlflow.log_param("production_set", False)
                    return
                mlflow.log_param("production_set", True)
                mlflow.log_param("winner_model", winner["registry"])
                mlflow.log_param("winner_version", winner["version"])
                mlflow.log_param("winner_strategy", winner["strategy"])
                mlflow.log_metric("production/f1_weighted", winner["f1"])
                if len(ranking) > 1:
                    # Marge sur le dauphin : si elle est sous le bruit
                    # d'échantillonnage du gold (σ_F1 ≈ 0.004), le choix du
                    # servi n'est pas statistiquement fondé — information
                    # décisive pour lire la courbe, et perdue sans ce log.
                    mlflow.log_metric("margin_vs_second",
                                      winner["f1"] - ranking[1]["f1"])

        if not ranking:
            print("[tournament] Aucun champion trouvé → skip")
            _log_tournament(None)
            return {"production_set": False}

        best = ranking[0]

        # Retirer @production de l'ancien porteur (s'il existe)
        for registry in set(FUSION_REGISTRY.values()):
            try:
                client.delete_registered_model_alias(registry, "production")
            except MlflowException as e:
                if not _absent(e):
                    raise

        # Poser @production sur le vainqueur [D-T4.4a Option C]
        client.set_registered_model_alias(
            best["registry"], "production", best["version"]
        )
        print(f"[tournament] @production → {best['registry']} v{best['version']} "
              f"(F1={best['f1']:.4f}, source={best['alias']}) "
              f"| sortant : {previous or 'aucun'}")

        _log_tournament(best)

        return {
            "production_set": True,
            "model_name": best["registry"],
            "version": int(best["version"]),
            "f1": best["f1"],
            "source_alias": best["alias"],
            "previous_production": previous,
            "ranking": ranking,
        }

    tournament_task = tournament()
    join_all >> tournament_task

    # ================================================================ #
    # P4.5d - Trigger Predict_Y_test (soumission ENS)                  #
    # En aval du tournament : @production vient d'etre (re)attribue au #
    # vainqueur cross-model, c'est LUI qui doit scorer X_test.         #
    # HORS chaine de cycle : ce trigger ne relance pas Ingestion, il   #
    # produit un livrable. wait_for_completion=False -> Training rend  #
    # la main tout de suite (le scoring dure ~20-40 min sur pod).      #
    # batch_id fige dans la conf : la Variable passera a n+1 juste     #
    # apres, l'estampille des predictions doit rester le batch         #
    # entraine.                                                        #
    # ================================================================ #
    trigger_predict_test = TriggerDagRunOperator(
        task_id="trigger_predict_y_test",
        trigger_dag_id="Predict_Y_test",
        conf={"batch_id": BATCH_ID_JINJA},
        wait_for_completion=False,
    )
    tournament_task >> trigger_predict_test

training_instance = training_dag()