"""
T.1 — Warm-start pour retraining stateful.

Dispatcher unique pour 3 types de modèles :

    1. LightningModule (M3, M3.2)
       → load_state_dict partiel (DoRA adapters + fusion head)
       → source : @champion du registered model fusion

    2. BaseLearner (CamemBERT-LoRA, SigLIP2, TextCNN, ResNet*)
       → load_state_dict du .net interne
       → source : @active du registered model base learner

    3. StackingLGBM (M2)
       → init_model LightGBM (continue le boosting)
       → coefs LogReg level-1 (warm_start sklearn)
       → source : @champion du registered model M2

Contrat stateful complet (batch n>1) :
    - Base learners : warm-start depuis @active, train sur batch_n + replay(1..n-1)
    - Fusion M3/M3.2 : warm-start depuis @champion, train sur batch_n + replay(1..n-1)
    - Meta M2 : warm-start depuis @champion, OOF sur embeddings des BL warm-startés

Le warm-start est appliqué AVANT le .fit() de chaque modèle.
Sans warm-start → comportement stateless inchangé (init fraîche).

Usage :
    from src.models.warm_start import apply_warm_start

    stats = apply_warm_start(model_or_learner, "models:/name@alias")
    model_or_learner.fit(...)  # part des poids chargés
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ====================================================================== #
# Dispatcher                                                              #
# ====================================================================== #


def apply_warm_start(model, warm_start_uri: str) -> dict:
    """
    Détecte le type de modèle et applique le mécanisme de warm-start adapté.

    Args:
        model: LightningModule, BaseLearner, ou StackingLGBM (non-fitté).
        warm_start_uri: URI MLflow du modèle source.
            Ex: "rakuten-m3-2-coadaptation@champion"
                "rakuten-base-camembert_lora@active"
                "rakuten-m2-best@champion"

    Returns:
        dict avec stats de chargement (type, loaded, total, ...).
    """
    if not warm_start_uri.startswith("models:/"):
        warm_start_uri = f"models:/{warm_start_uri}"

    import lightning as L

    # 1. LightningModule direct (M3, M3.2 fusion)
    if isinstance(model, L.LightningModule):
        return _warm_start_lightning(model, warm_start_uri)

    # 2. BaseLearner (a un .net + predict_proba)
    if hasattr(model, "net") and hasattr(model, "predict_proba"):
        return _warm_start_base_learner(model, warm_start_uri)

    # 3. StackingLGBM (a _base_text_template)
    if hasattr(model, "_base_text_template"):
        return _warm_start_sklearn(model, warm_start_uri)

    raise ValueError(
        f"Warm-start non supporté pour {type(model).__name__}. "
        f"Types supportés : LightningModule, BaseLearner, StackingLGBM."
    )


# ====================================================================== #
# 1. Lightning (M3, M3.2)                                                #
# ====================================================================== #


def _warm_start_lightning(model, warm_start_uri: str) -> dict:
    """
    Charge les poids entraînables (DoRA + fusion head) depuis le champion.

    Le champion est un M3/M3.2 complet sauvé via mlflow.pytorch.log_model.
    On extrait son state_dict, on filtre aux paramètres requires_grad=True
    du modèle cible, et on injecte avec strict=False.

    Les backbones gelés (text_net, image_net) restent intacts — ils viennent
    des @active base learners chargés par le builder, pas du champion.
    """
    import mlflow.pytorch

    logger.info(f"[warm-start/lightning] Chargement {warm_start_uri}...")

    # 1. Charger le champion complet
    champion = mlflow.pytorch.load_model(warm_start_uri)
    champion_state = champion.state_dict()

    # 2. Identifier les clés entraînables du modèle cible
    trainable_keys = {
        name for name, param in model.named_parameters()
        if param.requires_grad
    }

    # 3. Filtrer : clés entraînables + présentes dans le champion + même shape
    filtered = {}
    mismatched = []
    missing = []
    for key in trainable_keys:
        if key not in champion_state:
            missing.append(key)
        elif champion_state[key].shape != model.state_dict()[key].shape:
            mismatched.append(key)
        else:
            filtered[key] = champion_state[key]

    # 4. Injecter
    if filtered:
        model.load_state_dict(filtered, strict=False)

    # 5. Diagnostic : norme L2 des DoRA (>0 = warm-start effectif)
    dora_norm = sum(
        p.data.norm().item()
        for name, p in model.named_parameters()
        if p.requires_grad and "lora" in name.lower()
    )

    # 6. Cleanup VRAM
    del champion, champion_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "type": "lightning",
        "loaded": len(filtered),
        "trainable": len(trainable_keys),
        "missing_in_champion": len(missing),
        "shape_mismatched": len(mismatched),
        "dora_l2_norm": round(dora_norm, 4),
    }
    logger.info(
        f"[warm-start/lightning] {stats['loaded']}/{stats['trainable']} "
        f"poids chargés. DoRA L2={dora_norm:.4f}"
    )
    if mismatched:
        logger.warning(f"[warm-start/lightning] Shape mismatch sur : {mismatched}")
    return stats


# ====================================================================== #
# 2. BaseLearner (CamemBERT-LoRA, SigLIP2, TextCNN, ResNet*)            #
# ====================================================================== #


def _warm_start_base_learner(learner, warm_start_uri: str) -> dict:
    """
    Charge un BaseLearner @active et injecte ses poids dans le learner cible.

    Le @active a été entraîné sur batch_n-1. Le learner cible est une
    instance fraîche (même architecture, même backbone). On injecte
    TOUS les poids du .net (backbone + head) pour partir de l'état
    entraîné au lieu de l'init aléatoire.

    Pour les learners avec LoRA (CamemBERT-LoRA), les poids LoRA
    entraînés remplacent l'init aléatoire fraîche.
    """
    import mlflow.pyfunc

    logger.info(f"[warm-start/base_learner] Chargement {warm_start_uri}...")

    # 1. Charger le pyfunc et extraire le learner
    pyfunc = mlflow.pyfunc.load_model(warm_start_uri)
    python_model = getattr(
        getattr(pyfunc, "_model_impl", None), "python_model", None
    )
    if python_model is None:
        raise RuntimeError(f"Pas de python_model pour {warm_start_uri}")

    source_learner = getattr(python_model, "learner", None)
    if source_learner is None:
        raise RuntimeError(f"learner est None pour {warm_start_uri}")

    # 2. State dict du net source
    source_state = source_learner.net.state_dict()
    target_state = learner.net.state_dict()

    # 3. Filtrer aux clés communes + même shape
    filtered = {}
    mismatched = []
    for key in target_state:
        if key in source_state:
            if source_state[key].shape == target_state[key].shape:
                filtered[key] = source_state[key]
            else:
                mismatched.append(key)

    # 4. Injecter
    if filtered:
        learner.net.load_state_dict(filtered, strict=False)

    # 5. Cleanup
    del pyfunc, python_model, source_learner

    stats = {
        "type": "base_learner",
        "loaded": len(filtered),
        "total_target": len(target_state),
        "total_source": len(source_state),
        "shape_mismatched": len(mismatched),
    }
    logger.info(
        f"[warm-start/base_learner] {stats['loaded']}/{stats['total_target']} "
        f"poids chargés depuis {warm_start_uri}"
    )
    if mismatched:
        logger.warning(f"[warm-start/base_learner] Shape mismatch : {mismatched}")
    return stats


# ====================================================================== #
# 3. Sklearn (M2 StackingLGBM)                                          #
# ====================================================================== #


def _warm_start_sklearn(model, warm_start_uri: str) -> dict:
    """
    Charge le champion M2 et stocke les composants pour warm-start.

    Trois composants extraits et stockés comme attributs du modèle.
    Le StackingLGBM.fit() consulte ces attributs pendant le refit :

        model._warm_start_init_model
            LightGBM Booster du champion. Passé à
            LGBMClassifier.fit(..., init_model=booster) pour continuer le
            boosting au lieu de repartir de zéro.

        model._warm_start_f_text
            Pipeline (Scaler + LogReg) texte fitté du champion.
            Le LogReg level-1 du refit est initialisé avec ces coefs
            (sklearn warm_start=True + coef_/intercept_ copy).

        model._warm_start_f_image
            Idem pour la modalité image.

    Attributs optionnels : si un composant est absent du champion,
    le refit se fait avec init fraîche (fallback gracieux).
    """
    import mlflow.pyfunc

    logger.info(f"[warm-start/sklearn] Chargement {warm_start_uri}...")

    # 1. Charger le champion via pyfunc → sklearn_model
    pyfunc = mlflow.pyfunc.load_model(warm_start_uri)
    champion = pyfunc._model_impl.sklearn_model
    if champion is None:
        raise RuntimeError(f"Pas de sklearn_model dans {warm_start_uri}")

    components = []

    # 2. Meta LightGBM booster
    if hasattr(champion, "meta_") and champion.meta_ is not None:
        model._warm_start_init_model = champion.meta_.booster_
        n_trees = champion.meta_.n_estimators
        components.append(f"meta_lgbm({n_trees}_arbres)")
        logger.info(f"[warm-start/sklearn] Meta LGBM booster : {n_trees} arbres")

    # 3. LogReg texte fitté
    if hasattr(champion, "f_text_") and champion.f_text_ is not None:
        model._warm_start_f_text = champion.f_text_
        components.append("f_text_logreg")
        logger.info("[warm-start/sklearn] LogReg texte extrait")

    # 4. LogReg image fitté
    if hasattr(champion, "f_image_") and champion.f_image_ is not None:
        model._warm_start_f_image = champion.f_image_
        components.append("f_image_logreg")
        logger.info("[warm-start/sklearn] LogReg image extrait")

    # 5. Cleanup
    del pyfunc, champion

    stats = {
        "type": "sklearn",
        "loaded": len(components),
        "components": components,
    }
    logger.info(
        f"[warm-start/sklearn] {len(components)} composants chargés : {components}"
    )
    return stats
