"""
[BLOC L] Experiment_Cycle — meta-DAG d'orchestration de l'experience finale.

PROBLEME. On veut enchainer les cycles de vie MLOps sans declenchement manuel :
    Ingestion(n) -> Training(n) -> Predict_Y_test(n) -> Ingestion(n+1) -> ...
La solution naive (une arete retour de Training vers Ingestion) introduit un
cycle dans le graphe des DAGs. Le graphe des TACHES de chaque run reste
acyclique, donc Airflow l'accepte — mais la TERMINAISON devient une propriete
d'execution au lieu d'une propriete structurelle : plus rien dans le code ne
garantit l'arret, seule une Variable mutable lue au runtime le fait. Pour une
boucle dont chaque iteration coute une douzaine de pods GPU, c'est un mauvais
echange.

SOLUTION. La borne est connue a l'ecriture du code, donc la boucle se DEROULE :

    check_b3 -> trigger_ingestion_b3 -> wait_cycle_b3
    (puis check_b4 -> trigger_ingestion_b4 -> wait_cycle_b4, etc.)

Le graphe est acyclique par construction, sans arete retour. Le nombre de
cycles est fini et lisible dans le code. Ni kill switch ni garde anti-boucle :
ils n'existaient que pour compenser la fragilite de l'arete retour.

POURQUOI UN CAPTEUR PLUTOT QU'UNE CASCADE D'ATTENTES.
L'alternative etait de faire remonter le signal de fin par transitivite :
Predict finit -> Training se debloque -> Ingestion se debloque -> le meta-DAG
se debloque. Trois maillons, dont deux DAGs bloques uniquement pour transmettre
un signal qui ne les concerne pas — Ingestion aurait dure aussi longtemps que
Training, et « une ingestion de trois heures » est un contresens qui piege le
prochain lecteur. Ici le meta-DAG observe la fin de Predict la ou elle se
produit. AUCUN DAG METIER N'EST MODIFIE : ils ignorent qu'ils sont orchestres.

MODE RESCHEDULE. Le capteur rend son slot de worker entre deux pokes (etat
`up_for_reschedule`) au lieu de le monopoliser plusieurs heures. Meme effet
que `deferrable=True` sans exiger le service `triggerer`.

PILOTAGE DU BATCH. Ingestion lit la Variable `batch_id`, pas une conf. Apres
le cycle n, `increment_batch_id` l'a deja portee a n+1 — exactement ce dont
Ingestion(n+1) a besoin. Le meta-DAG n'a donc rien a piloter, mais il VERIFIE
avant chaque declenchement (`check_batch_ready`). Le plan d'experience reste
partiellement porte par un etat mutable ; le controle rend toute incoherence
bruyante au lieu de silencieuse.

USAGE
    airflow dags trigger Experiment_Cycle
Prerequis : Variable `batch_id` == EXPERIMENT_BATCHES[0], donnees du batch
presentes dans X_raw_data_batches.
"""
from datetime import timedelta

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models import DagRun, Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.base import PokeReturnValue
from airflow.utils.dates import days_ago
from airflow.utils.state import DagRunState

# ------------------------------------------------------------------ #
# PLAN D'EXPERIENCE — la boucle deroulee.                            #
#                                                                     #
# Batch 1 = bootstrap manuel, batch 2 = deja execute (doctrine du     #
# projet), donc l'experience finale ne couvre que le batch 3.         #
# Pour rejouer plusieurs cycles : [2, 3] et remettre la Variable      #
# `batch_id` a 2 au prealable.                                        #
#                                                                     #
# /!\ DETTE BLOQUANTE avant d'ajouter un batch 4 :                    #
#     get_active_val_selection_version leve hors de {1, 2, 3}.        #
# ------------------------------------------------------------------ #
EXPERIMENT_BATCHES = [3]

# Duree max d'un cycle complet (ingestion + zoo de 12 fits + scoring).
# Un cycle nominal tourne en 3-5 h ; 10 h laisse de la marge pour les
# retries de penurie GPU sans laisser un capteur tourner indefiniment.
CYCLE_TIMEOUT = timedelta(hours=10)
POKE_INTERVAL = 60


def _conf_batch_id(dag_run: DagRun) -> str | None:
    """
    Extrait batch_id de la conf d'un run, normalise en str.

    BATCH_ID_JINJA est rendu en CHAINE ('3') dans dag_training, alors que
    dag_ingestion utilise `| int` avec render_template_as_native_obj.
    Comparer en str evite de dependre de ce detail.
    """
    conf = dag_run.conf or {}
    value = conf.get("batch_id")
    return None if value is None else str(value)


def _find_runs(dag_id: str, batch_id: int, since) -> list[DagRun]:
    """
    Runs de `dag_id` estampilles batch_id, demarres apres `since`.

    Le filtre temporel n'est PAS cosmetique : Predict_Y_test(2) a deja
    reussi avant cette experience. Sans borne, le capteur serait satisfait
    instantanement par un run historique et le meta-DAG enchainerait le
    cycle suivant alors que rien n'a tourne.
    """
    runs = DagRun.find(dag_id=dag_id)
    out = []
    for r in runs:
        if _conf_batch_id(r) != str(batch_id):
            continue
        started = r.start_date or r.execution_date
        if started is None or since is None or started >= since:
            out.append(r)
    return out


@dag(
    dag_id="Experiment_Cycle",
    schedule=None,
    start_date=days_ago(1),
    tags=["MAR25_CMLOPS_RAKUTEN", "experiment", "phase4"],
    catchup=False,
    max_active_runs=1,
    doc_md=__doc__,
)
def experiment_cycle_dag():

    @task()
    def check_batch_ready(batch_id: int, **context):
        """
        Verifie que la Variable `batch_id` correspond au cycle attendu.

        Ingestion lit cette Variable : si elle ne vaut pas `batch_id`, on
        ingererait le mauvais batch — silencieusement, et on ne s'en
        apercevrait qu'en lisant les metriques d'un training de 3 h.
        """
        current = int(Variable.get("batch_id", default_var=1))
        if current != batch_id:
            raise AirflowFailException(
                f"Variable batch_id={current}, attendu {batch_id}. "
                f"Ingestion ingererait le mauvais batch. Corriger la "
                f"Variable avant de relancer ce cycle."
            )
        print(f"[check_batch_ready] Variable batch_id={current} — OK pour le cycle {batch_id}.")
        return batch_id

    @task.sensor(
        poke_interval=POKE_INTERVAL,
        timeout=CYCLE_TIMEOUT.total_seconds(),
        mode="reschedule",
        soft_fail=False,
    )
    def wait_cycle(batch_id: int, **context):
        """
        Attend la fin reelle du cycle : Predict_Y_test(batch_id) en succes.

        Predict est le DERNIER maillon (Ingestion -> Training -> Predict),
        donc sa reussite atteste que la soumission du batch est produite et
        que @production ne bougera plus pour ce cycle.

        Diagnostic amont : si Training ou Ingestion echoue, Predict ne sera
        JAMAIS declenche et le capteur tournerait jusqu'au timeout — le
        message d'erreur dirait « delai depasse » la ou la cause reelle est
        un training plante trois heures plus tot. On detecte donc l'echec
        amont explicitement.
        """
        since = context["dag_run"].start_date

        # --- Echecs amont : diagnostic exact plutot que timeout muet ---
        for upstream in ("Ingestion", "Training"):
            failed = [
                r for r in _find_runs(upstream, batch_id, since)
                if r.state == DagRunState.FAILED
            ]
            if failed:
                raise AirflowFailException(
                    f"{upstream}(batch {batch_id}) a echoue "
                    f"(run_id={failed[-1].run_id}). Predict ne sera jamais "
                    f"declenche — cycle interrompu."
                )

        runs = _find_runs("Predict_Y_test", batch_id, since)
        if not runs:
            print(f"[wait_cycle {batch_id}] aucun run de Predict_Y_test encore visible.")
            return PokeReturnValue(is_done=False)

        failed = [r for r in runs if r.state == DagRunState.FAILED]
        if failed:
            raise AirflowFailException(
                f"Predict_Y_test(batch {batch_id}) a echoue "
                f"(run_id={failed[-1].run_id}). Pas de soumission — "
                f"cycle interrompu."
            )

        success = [r for r in runs if r.state == DagRunState.SUCCESS]
        if success:
            run_id = success[-1].run_id
            print(f"[wait_cycle {batch_id}] cycle termine (run_id={run_id}).")
            return PokeReturnValue(is_done=True, xcom_value={"batch_id": batch_id, "run_id": run_id})

        states = sorted({str(r.state) for r in runs})
        print(f"[wait_cycle {batch_id}] Predict_Y_test en cours : {states}")
        return PokeReturnValue(is_done=False)

    # ---------------------------------------------------------------- #
    # Deroulage : un maillon par batch, chaines en serie.              #
    # Aucune arete retour — la terminaison est structurelle.           #
    # ---------------------------------------------------------------- #
    previous = None
    for n in EXPERIMENT_BATCHES:
        check = check_batch_ready.override(task_id=f"check_batch_ready_b{n}")(n)

        # wait_for_completion=False : Ingestion rend la main tout de suite,
        # elle garde sa semantique d'origine (« j'ai ingere »). C'est le
        # capteur, et lui seul, qui porte l'attente du cycle complet.
        trigger = TriggerDagRunOperator(
            task_id=f"trigger_ingestion_b{n}",
            trigger_dag_id="Ingestion",
            wait_for_completion=False,
            reset_dag_run=True,
        )

        wait = wait_cycle.override(task_id=f"wait_cycle_b{n}")(n)

        check >> trigger >> wait
        if previous is not None:
            previous >> check
        previous = wait


experiment_cycle_instance = experiment_cycle_dag()
