"""
Registre des dimensions d'embedding par base learner.

Constante partagée entre le runner (pods GPU) et les tâches Airflow locales
(resolve_active_*). Extraite de runner.py [D-T6.4] pour que l'image Airflow
mince puisse l'importer SANS tirer la chaîne lourde du runner (optuna, torch).

Vérité du code (propriété d'architecture), pas config d'environnement :
camembert-base = 768 dims sur toute machine. Ne PAS mettre dans .env.
"""

# {nom court du learner: dimension du vecteur extract_embeddings}
LEARNER_EMBED_DIM = {
    "textcnn": 3072,
    "camembert_lora": 768,
    "camembert_frozen": 768,
    "resnet50_partial_ft": 2048,
    "resnet18_full_ft": 512,
    "resnet18_frozen": 512,
    "siglip2": 768,
}