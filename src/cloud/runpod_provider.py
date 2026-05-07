"""
Implémentation RunPod du CloudProvider.

Utilise le SDK Python `runpod` officiel.
"""
from __future__ import annotations
import os
import time
from typing import Optional

import runpod

from src.cloud.base import (
    CloudProvider, GPUSpec, JobConfig, JobHandle, JobResult, JobStatus,
)
from src.cloud.exceptions import (
    CloudConfigError, JobFailedError, JobSubmissionError, JobTimeoutError,
)


# Mapping aliases logiques → IDs GPU RunPod (à compléter selon la dispo réelle)
# Liste consultable via runpod.get_gpus() une fois authentifié
RUNPOD_GPU_MAP = {
    "rtx_a4000": "NVIDIA RTX A4000",
    "rtx_3090": "NVIDIA GeForce RTX 3090",
    "rtx_4090": "NVIDIA GeForce RTX 4090",
    "a100_40gb": "NVIDIA A100 80GB PCIe",  # 40GB peut être indispo selon les régions
}


# Mapping statuts RunPod → JobStatus universel
RUNPOD_STATUS_MAP = {
    "PENDING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "EXITED": JobStatus.SUCCEEDED,    # à distinguer via exit code
    "TERMINATED": JobStatus.STOPPED,
    "FAILED": JobStatus.FAILED,
}


class RunPodProvider(CloudProvider):
    """Provider cloud RunPod."""

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not api_key:
            raise CloudConfigError(
                "RUNPOD_API_KEY env var manquante. Définis-la dans .env."
            )
        runpod.api_key = api_key

    @property
    def name(self) -> str:
        return "runpod"

    # --- submit -----------------------------------------------------------

    def submit_job(self, config: JobConfig) -> JobHandle:
        """
        Crée un pod RunPod éphémère qui exécute la commande puis s'arrête.
        """
        gpu_id = self._resolve_gpu_id(config.gpu)
        # Volume : RunPod ne supporte qu'un seul volume attaché par pod
        volume_kwargs = {}
        if len(config.volumes) > 1:
            raise JobSubmissionError(
                "RunPod ne supporte qu'un seul volume attaché par pod."
            )
        v = config.volumes[0]
        volume_kwargs = {
            "network_volume_id": v.volume_id,
            "volume_mount_path": v.mount_path,
        }
        try:
            pod = runpod.create_pod(
                name=config.name or "mlops-rakuten-job",
                image_name=config.image,
                gpu_type_id=gpu_id,
                gpu_count=config.gpu.count,
                container_disk_in_gb=20,       # disk éphémère du container
                volume_in_gb=0,                # pas de volume attaché ici (DVC gère)
                env=config.env,
                docker_args=" ".join(config.command),  # commande de démarrage
                cloud_type="SECURE",          # ou "COMMUNITY" si moins cher dispo
                **volume_kwargs,
            )
        except Exception as e:
            raise JobSubmissionError(f"RunPod create_pod failed: {e}") from e

        if not pod or "id" not in pod:
            raise JobSubmissionError(f"RunPod retour invalide: {pod}")

        return JobHandle(
            provider_name=self.name,
            job_id=pod["id"],
            metadata={"pod_info": pod},
        )

    # --- status / logs ----------------------------------------------------

    def get_status(self, handle: JobHandle) -> JobStatus:
        try:
            pod = runpod.get_pod(handle.job_id)
        except Exception as e:
            raise JobFailedError(f"RunPod get_pod failed: {e}") from e

        if not pod:
            return JobStatus.UNKNOWN

        runtime_status = pod.get("desiredStatus", "UNKNOWN")
        return RUNPOD_STATUS_MAP.get(runtime_status, JobStatus.UNKNOWN)

    def fetch_logs(self, handle: JobHandle, follow: bool = False) -> str:
        """
        Récupère les logs du pod.
        
        Note RunPod : il n'y a pas d'API pure logs côté SDK Python officiel
        à ce jour. Pour follow=True on poll en boucle. Pour une seule fetch,
        on récupère le snapshot courant.
        """
        # NOTE: l'API logs de RunPod n'est pas exposée par le SDK Python
        # officiel à date. On utilise donc l'endpoint web ou on tombera 
        # éventuellement sur un wrapper communauté. Pour l'instant on retourne
        # une string indicative ; à enrichir si besoin.
        if follow:
            self._wait_until_terminal(handle, poll_interval_seconds=10)
        return f"(Logs RunPod non disponibles via SDK pour le pod {handle.job_id} — consulter la web UI RunPod)"

    # --- wait -------------------------------------------------------------

    def wait(self, handle: JobHandle, timeout_seconds: Optional[int] = None, poll_interval_seconds: int = 10,) -> JobResult:
        start = time.time()
        last_status = JobStatus.UNKNOWN

        while True:
            last_status = self.get_status(handle)
            if last_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                break

            if timeout_seconds is not None and (time.time() - start) > timeout_seconds:
                raise JobTimeoutError(
                    f"Job {handle.job_id} a dépassé le timeout de {timeout_seconds}s "
                    f"(dernier statut : {last_status})"
                )
            time.sleep(poll_interval_seconds)

        duration = time.time() - start
        logs = self.fetch_logs(handle, follow=False)

        # Stop explicit du pod pour arrêter la facturation
        # (RunPod ne stoppe pas automatiquement les pods éphémères)
        try:
            self.stop(handle)
        except Exception as e:
            print(f"[RunPodProvider] WARN: stop a échoué : {e}")

        return JobResult(
            handle=handle,
            status=last_status,
            logs=logs,
            exit_code=0 if last_status == JobStatus.SUCCEEDED else 1,
            duration_seconds=duration,
        )

    def _wait_until_terminal(self, handle: JobHandle, poll_interval_seconds: int = 10):
        while True:
            status = self.get_status(handle)
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                return
            time.sleep(poll_interval_seconds)

    # --- stop -------------------------------------------------------------

    def stop(self, handle: JobHandle) -> None:
        try:
            runpod.terminate_pod(handle.job_id)
        except Exception as e:
            raise CloudConfigError(f"RunPod terminate_pod failed: {e}") from e

    # --- helpers ----------------------------------------------------------

    def _resolve_gpu_id(self, gpu: GPUSpec) -> str:
        """Mappe l'alias logique vers l'ID GPU RunPod."""
        if gpu.gpu_type not in RUNPOD_GPU_MAP:
            raise CloudConfigError(
                f"Alias GPU '{gpu.gpu_type}' inconnu. "
                f"Connus : {list(RUNPOD_GPU_MAP)}"
            )
        return RUNPOD_GPU_MAP[gpu.gpu_type]