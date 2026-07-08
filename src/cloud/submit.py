"""
Point d'entrée LÉGER de soumission cloud.

Extrait de runner.cmd_submit_cloud pour être importable et exécutable SANS les
dépendances lourdes (torch / lightning / mlflow) — ce qui permet de lancer une
soumission RunPod depuis le conteneur Airflow mince (option B). Le pod, lui,
recharge sa config et fait tout le calcul.

Dépendances : stdlib + python-dotenv + src.cloud.* uniquement.

Usage standalone :
    python -m src.cloud.submit --experiment m3_2_coadaptation \\
        --cloud-action eval_gold_champion --gpu-types rtx_5090

Codes de sortie (contrat 0.c, consommé par make_cloud_task) :
    0                 succès
    EXIT_NO_CAPACITY  pénurie GPU (cascade épuisée) → RETRYABLE
    1                 pod FAILED/STOPPED, ou toute erreur déterministe → FAIL-FAST
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from dotenv import load_dotenv

from src.cloud.base import GPUSpec, JobConfig, JobStatus, VolumeMount
from src.cloud.exceptions import (
    EXIT_NO_CAPACITY,
    JobSubmissionError,
    NoCapacityError,
)
from src.cloud.factory import get_cloud_provider

# Cascade GPU par défaut (frugal → fallback). Source de vérité unique :
# runner.py pointera son argparse dessus en 0.e.2.
DEFAULT_GPU_TYPES = [
    "rtx_5090", "rtx_4090", "rtx_3090", "rtx_4080", "rtx_a5000",
    "rtx_a6000", "rtx_a4000", "a40", "l40", "l40s", "a100_40gb", "rtx_pro_4500",
]

# Timeout par défaut spécifique à chaque action (si non overridé en CLI)
TIMEOUT_BY_ACTION = {
    "smoke_tailscale": 300,
    "prepare_data": 3600,
    "fit": 7200,
    "promote": 600,
}
DEFAULT_TIMEOUT = 3600


def get_local_tailscale_ip() -> str:
    """
    Récupère l'IP Tailscale (100.x.x.x) de la machine locale via le binaire
    `tailscale`. Utilisé en fallback quand aucune IP n'est fournie par l'env.

    Raises:
        RuntimeError: si `tailscale` est absent ou pas connecté au tailnet.
    """
    try:
        output = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True, timeout=5
        ).strip()
    except FileNotFoundError as e:
        raise RuntimeError(
            "`tailscale` introuvable. Installe-le et lance `sudo tailscale up`, "
            "ou fournis IP_Tailscale / HOST_TAILSCALE_IP en variable d'env "
            "(cas conteneur Airflow)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`tailscale ip -4` a échoué (code={e.returncode}). "
            f"Vérifie la connexion au tailnet (`tailscale status`)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("`tailscale ip -4` a timeout (>5s).") from e

    ip = output.split("\n")[0].strip()
    if not ip.startswith("100."):
        raise RuntimeError(f"IP Tailscale inattendue : '{ip}' (devrait commencer par 100.)")
    return ip


def _resolve_host_tailscale_ip() -> str:
    """
    Résout l'IP tailscale de l'HÔTE (où tournent MLflow + Mongo).

    Env d'abord (conteneur Airflow, sans binaire tailscale — option B),
    fallback shell (exécution directe sur l'hôte). Le compose expose déjà
    IP_Tailscale (utilisé par le service mlflow).
    """
    return (
        os.getenv("IP_Tailscale")
        or os.getenv("HOST_TAILSCALE_IP")
        or get_local_tailscale_ip()
    )


def submit_cloud(
    *,
    experiment: str,
    cloud_action: str,
    gpu_types: list[str],
    cloud_image: str | None = None,
    cloud_timeout: int = DEFAULT_TIMEOUT,
    limit: int | None = None,
    overrides: list[str] | None = None,
    warm_start_from: str | None = None,
    mlflow_tracking_uri: str | None = None,
    dvc_targets: list[str] | None = None,
) -> JobStatus:
    """
    Soumet un job au provider cloud (cascade GPU) et attend sa fin.

    Retourne le JobStatus final. Lève NoCapacityError si toute la cascade est
    en pénurie (retryable), ou une JobSubmissionError fatale (fail-fast).

    Port fidèle de runner.cmd_submit_cloud, dédupliqué et sans dépendance ML.
    Le paramètre `config` de l'ancienne signature est supprimé (jamais utilisé).
    """
    if cloud_action is None:
        raise ValueError("cloud_action requis")

    # ── Tailscale : résolution unique (env-first pour le conteneur) ──
    ts_ip = _resolve_host_tailscale_ip()
    tailscale_authkey = os.getenv("TAILSCALE_AUTHKEY", "")
    if not tailscale_authkey:
        raise RuntimeError(
            "TAILSCALE_AUTHKEY manquante. Génère une auth key pod "
            "(reusable=true, ephemeral=true) dans le dashboard Tailscale."
        )

    # URIs dérivées pour le pod (tout passe par Tailscale)
    mlflow_uri_for_pod = f"http://{ts_ip}:5000"
    mongo_uri_for_pod = f"mongodb://{ts_ip}:27017"

    # Override MLflow si URI explicite non-locale fournie (CLI ou env) — une seule fois
    mlflow_uri_override = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
    local_hosts = ("localhost", "127.0.0.1", "mlflow:5000")
    if mlflow_uri_override and not any(h in mlflow_uri_override for h in local_hosts):
        mlflow_uri_for_pod = mlflow_uri_override

    print(f"[submit_cloud] Host tailscale IP : {ts_ip}")
    print(f"[submit_cloud] MLflow pod        : {mlflow_uri_for_pod}")
    print(f"[submit_cloud] Mongo pod         : {mongo_uri_for_pod} (via SOCKS5)")

    # Image Docker
    image = (
        cloud_image
        or os.getenv("GHCR_IMAGE_TRAINER")
        or f"ghcr.io/{os.getenv('GITHUB_USER', 'guillaumepe').lower()}/mlops-rakuten-trainer:latest"
    )

    # Commande pod (--mlflow-tracking-uri posé UNE seule fois — fix duplication)
    pod_command = [
        "python", "-m", "src.experiments.runner",
        "--experiment", experiment,
        "--action", cloud_action,
        "--mlflow-tracking-uri", mlflow_uri_for_pod,
    ]
    if limit is not None:
        pod_command += ["--limit", str(limit)]
    if overrides:
        pod_command += ["--set", *overrides]
    if warm_start_from:
        pod_command += ["--warm-start-from", warm_start_from]

    pod_env = {
        # R2 / DVC
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        "AWS_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        "R2_ENDPOINT_URL": os.getenv("R2_ENDPOINT_URL", ""),
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "rakuten-mlops-dvc"),
        # MongoDB local via Tailscale
        "MONGO_URI": mongo_uri_for_pod,
        "MONGO_PROXY_HOST": "localhost",
        "MONGO_PROXY_PORT": "1055",
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN"),
        # Tailscale
        "TAILSCALE_AUTHKEY": tailscale_authkey,
        # MLflow
        "MLFLOW_TRACKING_URI": mlflow_uri_for_pod,
        # RunPod
        "RUNPOD_API_KEY": os.getenv("RUNPOD_API_KEY", ""),
        "DATA_ROOT": "/workspace",
        "DVC_AUTO_PUSH": "true",
    }

    targets = dvc_targets or [
        "data/raw_data/X_train_update.csv.dvc",
        "data/raw_data/Y_train_update.csv.dvc",
        "data/raw_data/images/image_train.tar.zst.dvc",
    ]
    pod_env["DVC_PULL_TARGETS"] = " ".join(targets)

    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "MONGO_URI"):
        if not pod_env[key]:
            raise RuntimeError(f"{key} manquante dans l'environnement")

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

    # Timeout adaptatif par action si non overridé (variable locale, pas de mutation)
    if cloud_timeout == DEFAULT_TIMEOUT:
        cloud_timeout = TIMEOUT_BY_ACTION.get(cloud_action, DEFAULT_TIMEOUT)
        print(f"[submit_cloud] Timeout auto pour '{cloud_action}' : {cloud_timeout}s")

    print(f"[submit_cloud] Image      : {image}")
    print(f"[submit_cloud] GPUs cibles : {gpu_types}")
    print(f"[submit_cloud] Commande   : {' '.join(pod_command)}")
    print(f"[submit_cloud] Timeout    : {cloud_timeout}s")

    # ── Cascade GPU (0.c : pénurie → GPU suivant ; fatal → fail-fast) ──
    provider = get_cloud_provider()
    print(f"[submit_cloud] Provider   : {provider.name}")

    handle = None
    last_capacity_error = None
    for gpu_type in gpu_types:
        print(f"[submit_cloud] Tentative GPU : {gpu_type}")
        job_config = JobConfig(
            image=image,
            command=pod_command,
            env=pod_env,
            gpu=GPUSpec(gpu_type=gpu_type, count=1),
            volumes=volumes,
            name=f"rakuten-{experiment}-{cloud_action}",
        )
        try:
            handle = provider.submit_job(job_config)
            print(f"[submit_cloud] ✓ Pod provisionné avec {gpu_type}")
            print(f"[submit_cloud] Job ID     : {handle.job_id}")
            break
        except NoCapacityError as e:
            print(f"[submit_cloud] ✗ {gpu_type} pénurie : {e}")
            last_capacity_error = e
            continue
        except JobSubmissionError as e:
            print(f"[submit_cloud] ✗ erreur fatale (fail-fast, pas de retry) : {e}")
            raise

    if handle is None:
        raise NoCapacityError(
            f"Aucun GPU dispo dans la liste {gpu_types} (tous en pénurie). "
            f"Dernière erreur : {last_capacity_error}"
        )

    # ── Polling jusqu'à terminaison ou timeout ──
    print("[submit_cloud] Attente de la fin du job...")
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
            if elapsed > cloud_timeout:
                print("[submit_cloud] Timeout dépassé, stop du pod")
                provider.stop(handle)
                raise RuntimeError(f"Timeout après {elapsed}s")
            time.sleep(10)

        duration = time.time() - start
        print(f"[submit_cloud] Job terminé : {last_status.value}")
        print(f"[submit_cloud] Durée      : {duration:.1f}s")
    except Exception as e:
        print(f"[submit_cloud] Erreur wait : {e}")
        print("[submit_cloud] Tentative de stop du pod...")
        try:
            provider.stop(handle)
        except Exception as stop_err:
            print(f"[submit_cloud] Stop échec : {stop_err}")
        raise

    return last_status


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Soumission cloud légère (sans deps ML).")
    p.add_argument("--experiment", required=True)
    p.add_argument("--cloud-action", required=True)
    p.add_argument("--gpu-types", nargs="+", default=None,
                   help="Cascade GPU. Défaut : DEFAULT_GPU_TYPES.")
    p.add_argument("--cloud-image", default=None)
    p.add_argument("--cloud-timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--warm-start-from", default=None)
    p.add_argument("--mlflow-tracking-uri", default=None)
    p.add_argument("--cloud-dvc-targets", nargs="+", default=None)
    return p


def main() -> int:
    """Entrée standalone : mappe le JobStatus final sur un code de sortie (contrat 0.c)."""
    load_dotenv()  # no-op dans le conteneur (env via compose), charge .env sur l'hôte
    args = _build_arg_parser().parse_args()
    try:
        final_status = submit_cloud(
            experiment=args.experiment,
            cloud_action=args.cloud_action,
            gpu_types=args.gpu_types or DEFAULT_GPU_TYPES,
            cloud_image=args.cloud_image,
            cloud_timeout=args.cloud_timeout,
            limit=args.limit,
            overrides=args.overrides,
            warm_start_from=args.warm_start_from,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            dvc_targets=args.cloud_dvc_targets,
        )
    except NoCapacityError as e:
        print(f"[submit] Pénurie GPU (retryable, exit {EXIT_NO_CAPACITY}) : {e}")
        return EXIT_NO_CAPACITY
    if final_status == JobStatus.SUCCEEDED:
        return 0
    print(f"[submit] Pod terminé en '{final_status.value}' → échec (exit 1)")
    return 1


if __name__ == "__main__":
    sys.exit(main())