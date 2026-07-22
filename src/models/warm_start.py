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
import mlflow.pyfunc
logger = logging.getLogger(__name__)

# ====================================================================== #
# Fallback cross-lignée [D-T3.5]                                         #
# ====================================================================== #


def _cross_lineage_fallback(uri: str) -> str | None:
    """
    Si l'alias stateful n'existe pas (batch 2, premier run),
    tente la version stateless comme seed.

    models:/name@active_stateful   → models:/name@active_stateless
    models:/name@champion_stateful → models:/name@champion_stateless
    Autre → None (pas de fallback)
    """
    if "@active_stateful" in uri:
        return uri.replace("@active_stateful", "@active_stateless")
    if "@champion_stateful" in uri:
        return uri.replace("@champion_stateful", "@champion_stateless")
    return None

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
        Si l'URI n'existe pas et qu'un fallback cross-lignée est trouvé,
        retente avec l'alias de l'autre lignée [D-T3.5].
        Si aucun fallback → cold start gracieux (pas de crash).
    """
    if not warm_start_uri.startswith("models:/"):
        warm_start_uri = f"models:/{warm_start_uri}"

    def _dispatch(uri):
        import lightning as L

        if isinstance(model, L.LightningModule):
            return _warm_start_lightning(model, uri)
        if hasattr(model, "net") and hasattr(model, "predict_proba"):
            return _warm_start_base_learner(model, uri)
        if hasattr(model, "_base_text_template"):
            return _warm_start_sklearn(model, uri)

        raise ValueError(
            f"Warm-start non supporté pour {type(model).__name__}. "
            f"Types supportés : LightningModule, BaseLearner, StackingLGBM."
        )

    try:
        return _dispatch(warm_start_uri)
    except Exception as primary_err:
        # Fallback cross-lignée [D-T3.5]
        fallback_uri = _cross_lineage_fallback(warm_start_uri)
        if fallback_uri:
            logger.warning(
                f"[warm-start] {warm_start_uri} introuvable "
                f"({type(primary_err).__name__}: {primary_err}) → "
                f"fallback cross-lignée : {fallback_uri}"
            )
            try:
                stats = _dispatch(fallback_uri)
                stats["fallback_from"] = warm_start_uri
                stats["fallback_to"] = fallback_uri
                return stats
            except Exception as fallback_err:
                logger.warning(
                    f"[warm-start] Fallback {fallback_uri} également échoué "
                    f"({type(fallback_err).__name__}) → cold start"
                )

        # Aucun fallback ou fallback échoué → cold start gracieux
        logger.warning(
            f"[warm-start] {warm_start_uri} introuvable, "
            f"pas de fallback disponible → cold start"
        )
        return {
            "type": "cold_start",
            "reason": f"{type(primary_err).__name__}: {str(primary_err)[:200]}",
            "uri_attempted": warm_start_uri,
        }

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
    Charge un BaseLearner source et STOCKE son state_dict sur le learner
    cible. L'injection réelle est DIFFÉRÉE à learner.fit().

    Raison : le .net d'un base learner fraîchement construit par
    _build_learner() vaut None (il est bâti paresseusement DANS fit()).
    On ne peut donc pas charger les poids ici. On mémorise l'état source
    (déplacé sur CPU pour ne pas retenir de VRAM) dans
    learner._warm_start_net_state, et fit() le consomme via
    self.apply_warm_start_state() juste après avoir bâti self.net.

    Contrat : sans appel à apply_warm_start_state() dans fit(), le
    warm-start est un no-op silencieux.
    """
    

    logger.info(f"[warm-start/base_learner] Chargement {warm_start_uri}...")

    # 1. Charger le pyfunc et extraire le learner source
    pyfunc = mlflow.pyfunc.load_model(warm_start_uri)
    python_model = getattr(
        getattr(pyfunc, "_model_impl", None), "python_model", None
    )
    if python_model is None:
        raise RuntimeError(f"Pas de python_model pour {warm_start_uri}")

    source_learner = getattr(python_model, "learner", None)
    if source_learner is None:
        raise RuntimeError(f"learner est None pour {warm_start_uri}")
    if getattr(source_learner, "net", None) is None:
        raise RuntimeError(
            f"source_learner.net est None pour {warm_start_uri} "
            f"(modèle source non-fitté ?)"
        )

    # 2. Extraire le state_dict source sur CPU, puis libérer la source (VRAM)
    source_state = {
        k: v.detach().cpu()
        for k, v in source_learner.net.state_dict().items()
    }
    learner._warm_start_net_state = source_state

    del pyfunc, python_model, source_learner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "type": "base_learner",
        "deferred": True,
        "total_source": len(source_state),
    }
    logger.info(
        f"[warm-start/base_learner] {stats['total_source']} clés source "
        f"stockées ; injection différée à fit() (apply_warm_start_state)"
    )
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
