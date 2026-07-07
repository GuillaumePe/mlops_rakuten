"""Exceptions du module cloud."""


class CloudProviderError(Exception):
    """Erreur générique d'un provider cloud."""


class JobSubmissionError(CloudProviderError):
    """Échec lors de la soumission du job (creds, image introuvable, GPU saturé...)."""


class JobFailedError(CloudProviderError):
    """Le job a tourné mais s'est terminé en erreur."""


class JobTimeoutError(CloudProviderError):
    """Le job a dépassé le timeout configuré."""


class CloudConfigError(CloudProviderError):
    """Configuration cloud invalide (env vars manquantes, etc.)."""

class NoCapacityError(JobSubmissionError):
    """
    Sous-cas de JobSubmissionError : aucun GPU disponible côté provider
    (pénurie de capacité), par opposition à une erreur fatale (creds
    invalides, image introuvable, config volume erronée).

    Distinction CENTRALE pour l'orchestration :
    - NoCapacityError    → RETRYABLE (attendre qu'un GPU se libère)
    - JobSubmissionError → FAIL-FAST (retry inutile, l'erreur est déterministe)

    Hérite de JobSubmissionError → rétro-compatible : tout `except
    JobSubmissionError` existant (ex. la cascade GPU de cmd_submit_cloud)
    continue de l'attraper.
    """

