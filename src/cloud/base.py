"""
Abstraction provider cloud GPU.

Tous les providers (RunPod, Lambda Labs, Vast.ai, etc.) implémentent
l'interface CloudProvider. Le runner ne dépend que de cette interface.

Le LCD (lowest common denominator) est :
- Lance une image Docker avec une commande
- Attache des env vars
- Choisit un type de GPU
- Récupère les logs et le status
- Stoppe le job

Les volumes persistants, le storage, les sécurités sont à la charge
du code applicatif (DVC + R2 ici), pas de l'interface cloud.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    PENDING = "pending"        # En attente de provisioning
    RUNNING = "running"        # En cours d'exécution
    SUCCEEDED = "succeeded"    # Terminé avec exit code 0
    FAILED = "failed"          # Terminé avec exit code != 0
    STOPPED = "stopped"        # Stoppé manuellement
    UNKNOWN = "unknown"


@dataclass
class GPUSpec:
    """
    Spécification du GPU demandé. Les noms sont des aliases logiques
    abstraits que chaque provider mappe vers sa nomenclature interne.
    """
    gpu_type: str = "rtx_a4000"   # alias logique : "rtx_a4000", "rtx_3090", "rtx_4090", "a100_40gb"
    count: int = 1


@dataclass
class JobConfig:
    """
    Config d'un job à soumettre.

    image : image Docker (ex: "ghcr.io/user/mlops-rakuten-trainer:latest")
    command : commande à exécuter dans le container
    env : variables d'environnement passées au container
    gpu : spec GPU
    name : nom lisible du job (pour logs/UI)
    """
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    gpu: GPUSpec = field(default_factory=GPUSpec)
    name: Optional[str] = None
    volumes: list[VolumeMount] = field(default_factory=list)


@dataclass
class JobHandle:
    """
    Référence opaque vers un job soumis. Le provider l'utilise en interne
    pour fetch logs, status, stop. L'utilisateur ne doit pas en dépendre
    structurellement.
    """
    provider_name: str
    job_id: str
    metadata: dict = field(default_factory=dict)   # pour info debug


@dataclass
class JobResult:
    """Résultat d'un job terminé."""
    handle: JobHandle
    status: JobStatus
    logs: str
    exit_code: Optional[int] = None
    duration_seconds: Optional[float] = None

@dataclass
class VolumeMount:
    """
    Volume persistant à attacher au container.
    
    Si le provider ne supporte pas les volumes persistants, le job tournera
    sans (et les caches seront perdus à chaque pod).
    """
    volume_id: str          # ID du volume côté provider (créé hors du code)
    mount_path: str         # chemin de montage dans le container (ex: /workspace/cache)


class CloudProvider(ABC):
    """Interface commune des providers cloud GPU."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nom du provider (ex: 'runpod')."""

    @abstractmethod
    def submit_job(self, config: JobConfig) -> JobHandle:
        """
        Soumet un job au provider. Retourne immédiatement (le job tourne en
        arrière-plan). Lève JobSubmissionError en cas d'échec.
        """

    @abstractmethod
    def get_status(self, handle: JobHandle) -> JobStatus:
        """Récupère le statut courant du job."""

    @abstractmethod
    def fetch_logs(self, handle: JobHandle, follow: bool = False) -> str:
        """
        Récupère les logs.
        Si follow=True, stream en continu jusqu'à ce que le job termine.
        """

    @abstractmethod
    def wait(
        self,
        handle: JobHandle,
        timeout_seconds: Optional[int] = None,
        poll_interval_seconds: int = 10,
    ) -> JobResult:
        """
        Bloque jusqu'à ce que le job termine ou que timeout soit atteint.
        Lève JobTimeoutError si timeout.
        """

    @abstractmethod
    def stop(self, handle: JobHandle) -> None:
        """Arrête le job (graceful)."""

    

