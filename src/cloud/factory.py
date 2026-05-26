"""
Factory : retourne le provider cloud en fonction de la config.

Permet d'ajouter d'autres providers (Lambda Labs, Vast.ai...) sans modifier
le code utilisateur.
"""
from __future__ import annotations
import os

from src.cloud.base import CloudProvider
from src.cloud.exceptions import CloudConfigError
from src.cloud.runpod_provider import RunPodProvider


def get_cloud_provider(name: str | None = None) -> CloudProvider:
    """
    Retourne une instance du provider cloud.
    
    Args:
        name: nom du provider. Si None, lu depuis CLOUD_PROVIDER env var.
              Default : 'runpod'.
    
    Raises:
        CloudConfigError si le provider est inconnu.
    """
    name = name or os.getenv("CLOUD_PROVIDER", "runpod")
    name = name.lower()

    if name == "runpod":
        return RunPodProvider()

    raise CloudConfigError(
        f"Provider cloud '{name}' inconnu. "
        f"Connus : runpod. (Lambda/VastAI à implémenter en Phase 1+)"
    )
