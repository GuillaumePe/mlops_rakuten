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