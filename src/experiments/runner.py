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
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # charge .env automatiquement

import mlflow
import yaml

from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
from src.experiments.models.m2.m2 import M2Stacking
from src.experiments.strategies.sklearn_experiment import SklearnExperiment
import os


CONFIG_DIR = Path("src/experiments/config")


def load_config(experiment_name: str) -> dict:
    """Charge la config YAML correspondant à l'expérience."""
    config_path = CONFIG_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


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
        val_size=dm_cfg.get("val_size", 0.15),
        test_size=dm_cfg.get("test_size", 0.15),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
    )

    model_cfg = config["model"]

    def m2_factory(_optuna_callback_unused):
        # Le callback n'est plus utilisé (logging direct dans l'objective)
        # mais SklearnExperiment l'expose pour compat ABC future
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

    experiment = SklearnExperiment(
        model_factory=m2_factory,
        run_name=config["mlflow"]["run_name"],
        tags=config["mlflow"].get("tags", {}),
    )

    return dm, experiment


# Registry des constructeurs par expérience.
# Phase 1+ : ajouter "m3", "m4" ici.
EXPERIMENT_BUILDERS = {
    "m2": build_m2_experiment,
}


def cmd_prepare_data(dm: RakutenLightningDataModule):
    """Extrait/met à jour le cache d'embeddings. Étape lourde (GPU recommandé)."""
    print(f"[Runner] prepare_data() — cache: {dm.cache_path}")
    dm.prepare_data()


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
    if args.mlflow_tracking_uri:
        pod_command += ["--mlflow-tracking-uri", args.mlflow_tracking_uri]
    
    # Env vars critiques à passer au pod
    pod_env = {
        # R2 (DVC remote)
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        # MongoDB Atlas
        "MONGO_URI": os.getenv("MONGO_URI", ""),
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN"),
        # MLflow
        "MLFLOW_TRACKING_URI": args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", ""),
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
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate", "submit_cloud"],
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
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate"],
        help="(submit_cloud only) Quelle action le pod cloud doit exécuter",
    )
    parser.add_argument(
    "--gpu-types",
    nargs="+",
    default=["rtx_4090", "rtx_3090", "rtx_4080", "rtx_a5000", "rtx_a6000","rtx_a4000", "a40", "l40", "l40s", "a100_40gb"],
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


if __name__ == "__main__":
    sys.exit(main())