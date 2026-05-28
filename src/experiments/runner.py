"""
Runner CLI pour les expériences M2/M3/M4.

Usage:
    python -m src.experiments.runner --experiment m2 --action fit
    python -m src.experiments.runner --experiment m2 --action evaluate
    python -m src.experiments.runner --experiment m2 --action prepare_data

Le runner :
1. Charge la config YAML de l'expérience
2. Instancie le DataModule selon le mode
3. Instancie l'Experiment selon la stratégie (sklearn / lightning)
4. Exécute l'action demandée
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # charge .env automatiquement

import mlflow
import yaml

from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
from src.experiments.models.m2.m2 import M2Stacking
from src.experiments.strategies.sklearn_experiment import SklearnExperiment
from src.experiments.strategies.base_learner_experiment import BaseLearnerExperiment
from src.models.assembled.m2_baseline import M2Baseline
from src.models.assembled.m2_assembled import M2Assembled
import os


CONFIG_DIR = Path("src/experiments/config")


def load_config(experiment_name: str) -> dict:
    """Charge la config YAML correspondant à l'expérience."""
    config_path = CONFIG_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_local_tailscale_ip() -> str:
    """
    Récupère l'IP Tailscale (100.x.x.x) de la machine locale.
    Utilisée pour construire automatiquement MLFLOW_TRACKING_URI vu côté pod
    quand on submit un job cloud sans URI explicite.

    Raises:
        RuntimeError: si `tailscale` n'est pas installé ou pas connecté au tailnet.
    """
    try:
        output = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True, timeout=5
        ).strip()
    except FileNotFoundError as e:
        raise RuntimeError(
            "`tailscale` introuvable. Installe-le et lance `sudo tailscale up`."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`tailscale ip -4` a échoué (code={e.returncode}). "
            f"Vérifie que tu es connecté au tailnet (`tailscale status`)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("`tailscale ip -4` a timeout (>5s).") from e

    # Une machine peut avoir plusieurs IPs Tailscale (rare); on prend la première
    ip = output.split("\n")[0].strip()
    if not ip.startswith("100."):
        raise RuntimeError(f"IP Tailscale inattendue : '{ip}' (devrait commencer par 100.)")
    return ip

def build_m2_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """Assemble DataModule + M2Stacking + SklearnExperiment depuis une config M2."""
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1, 2, 3]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )

    model_cfg = config["model"]
    def m2_factory(_optuna_callback_unused):
        return M2Stacking(
            text_cols=dm.text_cols,
            image_cols=dm.image_cols,
            tabular_cols=dm.tabular_cols,
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    # Fusion des tags YAML + tags promotion (étape 5)
    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-stacking"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }

    experiment = SklearnExperiment(
        model_factory=m2_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_m2_baseline_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """
    Assemble DataModule + M2Baseline + SklearnExperiment depuis une config.

    Nouvelle architecture Phase 1 (modulaire) : équivalent fonctionnel de
    build_m2_experiment, mais via CamembertFrozen + ResNet18Frozen + StackingLGBM.
    Sert au test d'intégration L.5 (reproduction M2 v4).
    """
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )

    model_cfg = config["model"]

    def m2_baseline_factory(_optuna_callback_unused):
        return M2Baseline(
            tabular_cols=dm.tabular_cols,
            text_embed_dim=768,
            image_embed_dim=512,
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-stacking"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }

    experiment = SklearnExperiment(
        model_factory=m2_baseline_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_m2_assembled_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """
    M.9 / N.5 — Assemble DataModule + M2Assembled + SklearnExperiment.

    Builder générique pour tout stacking utilisant des base learners dont les
    embeddings sont pré-calculés en parquet. Lit la section `base_learners`
    du YAML pour résoudre (text_learner_name, text_embed_dim, image_learner_name,
    image_embed_dim) → M2Assembled.

    Configurations couvertes :
      - m2_benchmark  : TextCNN(3072) + ResNet50PartialFT(2048)
      - m2_frugal_ft  : CamembertLoRA(768) + ResNet18FullFT(512)
      - toute future combinaison de base learners
    """
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
        extra_embedding_caches=dm_cfg.get("extra_embedding_caches", []),
    )

    model_cfg = config["model"]
    bl_cfg = config["base_learners"]

    def m2_assembled_factory(_optuna_callback_unused):
        return M2Assembled(
            tabular_cols=dm.tabular_cols,
            text_learner_name=bl_cfg["text"]["name"],
            text_embed_dim=bl_cfg["text"]["embed_dim"],
            image_learner_name=bl_cfg["image"]["name"],
            image_embed_dim=bl_cfg["image"]["embed_dim"],
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-assembled"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }

    experiment = SklearnExperiment(
        model_factory=m2_assembled_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_base_learner_experiment(config: dict) -> tuple[RakutenLightningDataModule, BaseLearnerExperiment]:
    """
    M.5 — Assemble DataModule + BaseLearnerExperiment pour un base learner (TextCNN, ResNet50, etc.).
 
    Config attendue :
    ```yaml
    datamodule:
      mode: "base_learners"  # Mode où on récupère les features brutes
      ...
    learner:
      name: "textcnn" ou "resnet50_partial_ft"
      config: {...}  # hyperparams du learner
    mlflow:
      experiment_name: "base_learners_phase1"
      run_name: "textcnn_run_1"
      ...
    ```
    """
    print("[DEBUG] build_base_learner_experiment START")
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "raw_for_finetune"),
        text_model=dm_cfg.get("text_model", None),
        image_model=dm_cfg.get("image_model", None),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )
    print("[DEBUG] DataModule instantiated")
    learner_cfg = config["learner"]
    print(f"[DEBUG] learner_cfg: {learner_cfg}")
    learner_name = learner_cfg["name"]
    learner_config = learner_cfg.get("config", {})
 
    mlflow_cfg = config["mlflow"]
    # Priorité : env var (set par submit_cloud) > CLI > config YAML > default
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or mlflow_cfg.get("tracking_uri")
        or "http://mlflow:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "base_learners_phase1")
    print("[DEBUG] Creating BaseLearnerExperiment...")
    # Instancier BaseLearnerExperiment (Strategy pattern)
    experiment = BaseLearnerExperiment(
        learner_name=learner_name,
        config=learner_config,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        data_folder=Path(dm_cfg.get("data_folder", "data/raw_data")),
        cache_output_dir=Path(os.getenv("DATA_ROOT", ".")) / "data/cache",
    )
    print("[DEBUG] BaseLearnerExperiment instantiated")
    return dm, experiment

# Registry des constructeurs par expérience.
EXPERIMENT_BUILDERS = {
    "m2": build_m2_experiment,                  # legacy M2Stacking (à déprécier après L.5 validé)
    "m2_baseline": build_m2_baseline_experiment,  # nouvelle archi modulaire Phase 1
    "m2_benchmark": build_m2_assembled_experiment,
    "m2_frugal_ft": build_m2_assembled_experiment,
    "base_learner_textcnn": build_base_learner_experiment,
    "base_learner_resnet50_partial_ft": build_base_learner_experiment,
    "base_learner_camembert_lora": build_base_learner_experiment,
    "base_learner_resnet18_full_ft": build_base_learner_experiment,

}

# ─────────────────────────────────────────────────────────────────────────────
# Commands (actions)
# ─────────────────────────────────────────────────────────────────────────────
 

def cmd_prepare_data(dm: RakutenLightningDataModule):
    """Extrait/met à jour le cache d'embeddings. Étape lourde (GPU recommandé)."""
    print(f"[Runner] prepare_data() — cache: {dm.cache_path if hasattr(dm, 'cache_path') else 'N/A'}")
    if hasattr(dm, 'prepare_data'):
        dm.prepare_data()
    else:
        print("[Runner] DataModule n'a pas de prepare_data() (OK pour base_learners)")



def cmd_fit(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Setup + fit avec tracking MLflow."""
    print("[Runner] setup()...")
    dm.setup()
    print("[Runner] fit() avec tracking MLflow...")
    experiment.fit(dm)


def cmd_evaluate(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Évalue le modèle sur le test set. Suppose que fit() a déjà été appelé."""
    if experiment.model is None:
        raise RuntimeError(
            "Le modèle n'est pas fitté. Lance d'abord `--action fit` "
            "dans le même process, ou implémente le rechargement depuis MLflow."
        )
    print("[Runner] setup()...")
    dm.setup()
    results = experiment.evaluate(dm)
    print(f"[Runner] Résultats sur test : {results}")
    return results

def cmd_fit_base_learner(dm: RakutenLightningDataModule, experiment):
    """M.5 — Action pour fit un base learner (TextCNN, ResNet50PartialFT, etc.)."""
    print("[Runner.M5] fit_base_learner() — orchestration BaseLearnerExperiment")
    print("[Runner.M5] setup()...")
    dm.setup()
    print("[Runner.M5] fit() avec MLflow tracking + alias promotion...")
    experiment.fit(dm)
    print("[Runner.M5] fit_base_learner() terminé")


def cmd_smoke_tailscale():
    """
    Smoke test : valide la chaîne pod → Tailscale → MLflow local.

    Log un run minimal dans l'experiment '_smoke_tailscale' avec un param,
    une metric, et un artefact. Si tout apparaît dans l'UI MLflow locale,
    la chaîne complète est opérationnelle (incluant les uploads multipart).

    Cette action est destinée à tourner sur le pod cloud (--cloud-action smoke_tailscale).
    """
    import socket
    import tempfile

    print(f"[smoke] Hostname pod   : {socket.gethostname()}")
    print(f"[smoke] MLFLOW_TRACKING_URI : {os.environ.get('MLFLOW_TRACKING_URI', '<not set>')}")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI non défini, impossible de smoke-tester MLflow")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("_smoke_tailscale")

    with mlflow.start_run(run_name=f"smoke_{socket.gethostname()}") as run:
        print(f"[smoke] Run ID : {run.info.run_id}")
        mlflow.log_param("hostname", socket.gethostname())
        mlflow.log_param("pod_id", os.environ.get("RUNPOD_POD_ID", "unknown"))
        mlflow.log_metric("test_metric", 42.0)

        # Test d'upload d'artefact (chemin critique : exerce le multipart upload HTTP
        # qui peut échouer sur certains tunnels même quand /health répond)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"smoke test from {socket.gethostname()}\n")
            artifact_path = f.name
        mlflow.log_artifact(artifact_path)
        os.unlink(artifact_path)

        print("[smoke] OK : run + param + metric + artifact loggés")

def cmd_fetch_logs(args):
    """Récupère les logs d'un pod cloud depuis R2."""
    import subprocess
    print("[fetch_logs] Logs disponibles sur R2 :")
    subprocess.run([sys.executable, "scripts/r2_logs.py", "list"], check=True)
    if args.job_id:
        # Cherche le log le plus récent contenant le job_id
        # (le job_id RunPod ≠ pod_id mais souvent corrélés, donc on liste et l'user choisit)
        print(f"\n[fetch_logs] Pour télécharger un log, lance :")
        print(f"    python scripts/r2_logs.py download <key_au_dessus> /tmp/<key>")

def cmd_submit_cloud(args, config: dict):
    """
    Soumet un job au provider cloud avec fallback sur la liste de GPUs.
    
    Le pod cloud va exécuter `runner.py --action <cloud-action>` après
    avoir pull les données via DVC. Il push les nouveaux artefacts à la fin.
    """
    import time
    from src.cloud.factory import get_cloud_provider
    from src.cloud.base import JobConfig, GPUSpec, VolumeMount, JobStatus
    from src.cloud.exceptions import JobSubmissionError
    
    if args.cloud_action is None:
        raise ValueError("--cloud-action requis pour --action submit_cloud")
    
    # Image Docker (priorité : CLI > env > défaut)
    image = (
        args.cloud_image
        or os.getenv("GHCR_IMAGE_TRAINER")
        or f"ghcr.io/{os.getenv('GITHUB_USER', 'guillaumepe').lower()}/mlops-rakuten-trainer:latest"
    )
    
    # Commande à exécuter dans le pod
    pod_command = [
        "python", "-m", "src.experiments.runner",
        "--experiment", args.experiment,
        "--action", args.cloud_action,
    ]
    if args.limit is not None:
        pod_command += ["--limit", str(args.limit)]
    
    # Résolution MLflow tracking URI :
    # - Si fourni explicitement (CLI ou env) ET hors localhost : on garde tel quel
    # - Sinon : auto-construction depuis l'IP Tailscale locale (cas standard)
    mlflow_uri_override = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
    local_hosts = ("localhost", "127.0.0.1", "mlflow:5000")
    if mlflow_uri_override and not any(h in mlflow_uri_override for h in local_hosts):
        mlflow_uri_for_pod = mlflow_uri_override
        print(f"[submit_cloud] MLflow URI explicite : {mlflow_uri_for_pod}")
    else:
        ts_ip = get_local_tailscale_ip()
        mlflow_uri_for_pod = f"http://{ts_ip}:5000"
        print(f"[submit_cloud] MLflow URI auto via Tailscale : {mlflow_uri_for_pod}")
    
    # Tailscale auth key : obligatoire pour le pod
    tailscale_authkey = os.getenv("TAILSCALE_AUTHKEY", "")
    if not tailscale_authkey:
        raise RuntimeError(
            "TAILSCALE_AUTHKEY manquante dans .env. "
            "Génère une auth key pod (reusable=true, ephemeral=true) dans le dashboard Tailscale."
        )

    # On passe toujours l'URI résolu au pod, qu'il ait été fourni explicitement ou auto
    # (utile pour les actions qui lisent args.mlflow_tracking_uri côté pod)
    pod_command += ["--mlflow-tracking-uri", mlflow_uri_for_pod]
    # Env vars critiques à passer au pod
    pod_env = {
        # R2 (DVC remote)
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        # DVC-S3 lit AWS_*, pas R2_* — mapping nécessaire
        "AWS_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        # Pour l'upload des logs vers R2
        "R2_ENDPOINT_URL": os.getenv("R2_ENDPOINT_URL", ""),
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "rakuten-mlops-dvc"),
        # MongoDB Atlas
        "MONGO_URI": os.getenv("MONGO_URI", ""),
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN"),
        # Tailscale (overlay network vers MLflow local)
        "TAILSCALE_AUTHKEY": tailscale_authkey,
        # MLflow
        "MLFLOW_TRACKING_URI": mlflow_uri_for_pod,
        # Self-terminate
        "RUNPOD_API_KEY": os.getenv("RUNPOD_API_KEY", ""),
        # DATA_ROOT pour les paths
        "DATA_ROOT": "/workspace",
        # DVC auto-push
        "DVC_AUTO_PUSH": "true",
    }
    
    # Targets DVC à puller
    dvc_targets = args.cloud_dvc_targets or [
        "data/raw_data/X_train_update.csv.dvc",
        "data/raw_data/Y_train_update.csv.dvc",
        "data/raw_data/images/image_train.tar.zst.dvc",
    ]
    pod_env["DVC_PULL_TARGETS"] = " ".join(dvc_targets)
    
    # Vérifier les vars critiques
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "MONGO_URI"):
        if not pod_env[key]:
            raise RuntimeError(f"{key} manquante dans .env")
    
    # Volume RunPod cache (optionnel)
    volumes = []
    volume_id = os.getenv("RUNPOD_VOLUME_ID")
    if volume_id:
        volumes.append(VolumeMount(
            volume_id=volume_id,
            mount_path=os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace/cache"),
        ))
        print(f"[submit_cloud] Volume cache attaché : {volume_id}")
    else:
        print("[submit_cloud] Pas de volume cache (RUNPOD_VOLUME_ID non défini)")
    
    print(f"[submit_cloud] Image      : {image}")
    print(f"[submit_cloud] GPUs cibles : {args.gpu_types}")
    print(f"[submit_cloud] Commande   : {' '.join(pod_command)}")
    # Timeout adaptatif par action si pas explicitement override en CLI
    TIMEOUT_BY_ACTION = {
        "smoke_tailscale": 300,
        "prepare_data": 3600,
        "fit": 7200,
        "promote": 600,
    }
    DEFAULT_TIMEOUT = 3600  # le défaut du parser CLI
    if args.cloud_timeout == DEFAULT_TIMEOUT:
        # Pas overridé par l'utilisateur → on prend le défaut spécifique à l'action
        args.cloud_timeout = TIMEOUT_BY_ACTION.get(args.cloud_action, DEFAULT_TIMEOUT)
        print(f"[submit_cloud] Timeout auto pour action '{args.cloud_action}' : {args.cloud_timeout}s")
    print(f"[submit_cloud] Timeout    : {args.cloud_timeout}s")
    
    # Submit avec fallback sur la liste de GPUs
    provider = get_cloud_provider()
    print(f"[submit_cloud] Provider   : {provider.name}")
    
    handle = None
    last_error = None
    for gpu_type in args.gpu_types:
        print(f"[submit_cloud] Tentative GPU : {gpu_type}")
        job_config = JobConfig(
            image=image,
            command=pod_command,
            env=pod_env,
            gpu=GPUSpec(gpu_type=gpu_type, count=1),
            volumes=volumes,
            name=f"rakuten-{args.experiment}-{args.cloud_action}",
        )
        try:
            handle = provider.submit_job(job_config)
            print(f"[submit_cloud] ✓ Pod provisionné avec {gpu_type}")
            print(f"[submit_cloud] Job ID     : {handle.job_id}")
            break
        except JobSubmissionError as e:
            print(f"[submit_cloud] ✗ {gpu_type} indispo : {e}")
            last_error = e
            continue
    
    if handle is None:
        raise RuntimeError(
            f"Aucun GPU dispo dans la liste {args.gpu_types}. "
            f"Dernière erreur : {last_error}"
        )
    
    # Wait avec polling visible (debug)
    print(f"[submit_cloud] Attente de la fin du job...")
    start = time.time()
    poll_count = 0
    last_status = JobStatus.UNKNOWN
    
    try:
        while True:
            poll_count += 1
            last_status = provider.get_status(handle)
            elapsed = int(time.time() - start)
            print(f"[submit_cloud] [t={elapsed}s] Poll #{poll_count} : status={last_status.value}")
            
            if last_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                break
            
            if elapsed > args.cloud_timeout:
                print(f"[submit_cloud] Timeout dépassé, stop du pod")
                provider.stop(handle)
                raise RuntimeError(f"Timeout après {elapsed}s")
            
            time.sleep(10)
        
        duration = time.time() - start
        print(f"[submit_cloud] Job terminé : {last_status.value}")
        print(f"[submit_cloud] Durée      : {duration:.1f}s")
    
    except Exception as e:
        print(f"[submit_cloud] Erreur wait : {e}")
        print(f"[submit_cloud] Tentative de stop du pod...")
        try:
            provider.stop(handle)
        except Exception as stop_err:
            print(f"[submit_cloud] Stop échec : {stop_err}")
        raise
    
    return last_status

def main():
    parser = argparse.ArgumentParser(description="MLOps experiment runner")
    parser.add_argument(
        "--experiment", required=True, choices=list(EXPERIMENT_BUILDERS),
        help="Nom de l'expérience (correspond à src/experiments/config/<name>.yaml)",
    )
    parser.add_argument(
        "--action", required=True,
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate", "fit_base_learner", "submit_cloud", "smoke_tailscale","fetch_logs"],
        help="Action à exécuter",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="URI MLflow (override la config). Ex: http://mlflow:5000",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limiter le nombre de samples (debug/test rapide). None = full dataset.",
    )
    parser.add_argument(
        "--cloud-action",
        default=None,
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate", "smoke_tailscale","fit_base_learner"],
        help="(submit_cloud only) Quelle action le pod cloud doit exécuter",
    )
    parser.add_argument(
        "--gpu-types",
        nargs="+",
        default=["rtx_5090","rtx_4090", "rtx_3090", "rtx_4080", "rtx_a5000", "rtx_a6000","rtx_a4000", "a40", "l40", "l40s", "a100_40gb","rtx_pro_4500"],
        help="(submit_cloud) Liste de GPUs à essayer en cascade (du préféré au fallback)",
)
    parser.add_argument(
        "--cloud-image",
        default=None,
        help="(submit_cloud only) Image Docker. Si None, lit GHCR_IMAGE_TRAINER ou défaut.",
    )
    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=3600,
        help="(submit_cloud only) Timeout en secondes (défaut 1h)",
    )
    parser.add_argument(
    "--cloud-dvc-targets",
        nargs="+",
        default=None,
        help="(submit_cloud) Liste des .dvc à puller. Défaut : X_train + Y_train + images.",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="(fetch_logs only) Job ID RunPod du pod dont on veut les logs",
    )
    
    args = parser.parse_args()

    # Charger la config
    config = load_config(args.experiment)
    if args.limit is not None:
        config["limit"] = args.limit
    print(f"[Runner] Config chargée : {args.experiment} (limit={args.limit})")


    # MLflow tracking URI : CLI > config > défaut
    tracking_uri = (
        args.mlflow_tracking_uri
        or config.get("mlflow", {}).get("tracking_uri")
        or "http://mlflow:5000"
    )

    # Dispatch action
    if args.action == "submit_cloud":
        # Pas d'init MLflow local : le tracking se fera côté pod
        cmd_submit_cloud(args, config)
        return

    # smoke_tailscale : test de bout en bout sans construire DataModule/Experiment
    if args.action == "smoke_tailscale":
        cmd_smoke_tailscale()
        return

    # Init MLflow seulement pour les actions qui en ont besoin
    if args.action in ("fit", "evaluate", "fit_and_evaluate"):
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[Runner] MLflow tracking URI : {tracking_uri}")
        experiment_name = config["mlflow"].get("experiment_name", args.experiment)
        mlflow.set_experiment(experiment_name)
    else:
        print(f"[Runner] Action '{args.action}' : pas d'init MLflow nécessaire")

    # Construire les composants
    builder = EXPERIMENT_BUILDERS[args.experiment]
    dm, experiment = builder(config)

    if args.action == "prepare_data":
        cmd_prepare_data(dm)
    elif args.action == "fit":
        cmd_prepare_data(dm)
        cmd_fit(dm, experiment)
    elif args.action == "evaluate":
        cmd_evaluate(dm, experiment)
    elif args.action == "fit_and_evaluate":
        cmd_prepare_data(dm)
        cmd_fit(dm, experiment)
        cmd_evaluate(dm, experiment)
    elif args.action == "fit_base_learner":
        cmd_fit_base_learner(dm, experiment)
    elif args.action == "fetch_logs":
        cmd_fetch_logs(args)


if __name__ == "__main__":
    sys.exit(main())