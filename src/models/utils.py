from __future__ import annotations
import mlflow
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from dataclasses import dataclass
import torch
import logging
import os
from typing import Optional
 
logger = logging.getLogger(__name__)
 

def get_or_create_experiment(experiment_name):
  """
  Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

  This function checks if an experiment with the given name exists within MLflow.
  If it does, the function returns its ID. If not, it creates a new experiment
  with the provided name and returns its ID.

  Parameters:
  - experiment_name (str): Name of the MLflow experiment.

  Returns:
  - str: ID of the existing or newly created MLflow experiment.
  """

  if experiment := mlflow.get_experiment_by_name(experiment_name):
      return experiment.experiment_id
  else:
      return mlflow.create_experiment(experiment_name)
  
  # override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'


def champion_callback(study, frozen_trial):
  """
  Logging callback that will report when a new trial iteration improves upon existing
  best trial values.

  Note: This callback is not intended for use in distributed computing systems such as Spark
  or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
  workers or agents.
  The race conditions with file system state management for distributed trials will render
  inconsistent values with this callback.
  """

  winner = study.user_attrs.get("winner", None)

  if study.best_value and winner != study.best_value:
      study.set_user_attr("winner", study.best_value)
      if winner:
          improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
          print(
              f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
              f"{improvement_percent: .4f}% improvement"
          )
      else:
          print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


#fonction ojective pour l'optimisation par optuna
def objective_wrapper_pca(split_operator, X_train, y_train,num_class, metric, hyperparameters):
    
    def objective(trial):

      with mlflow.start_run(nested=True):
    
        text_feat_cols = [col for col in X_train.columns if col.startswith("text_feat_")]
        image_feat_cols = [col for col in X_train.columns if col.startswith("image_feat_")]
    
        #Search Space
        all_params = {}

        for param_name, param_value in hyperparameters.items():
          if param_name=="lgbm__num_leaves" and isinstance(param_value, tuple) and len(param_value) == 3:
            max_num_leaves = max(min(2 ** all_params["lgbm__max_depth"], param_value[2]), param_value[1])
            all_params["lgbm__num_leaves"] = trial.suggest_int("lgbm__num_leaves", param_value[1], max_num_leaves)
          
          elif isinstance(param_value, tuple) and len(param_value) == 3:
              if param_value[0] == "int":
                all_params[param_name] = trial.suggest_int(param_name, param_value[1], param_value[2])
              elif param_value[0] == "float":
                all_params[param_name] = trial.suggest_float(param_name, param_value[1], param_value[2], log=True)

        all_params.update({
                "lgbm__num_class": num_class,
                "random_state": 42,
                "verbosity": -1})
        
        lgbm__params = {
                    "lgbm__num_leaves": all_params["lgbm__num_leaves"],
                    "lgbm__max_depth": all_params["lgbm__max_depth"],
                    "lgbm__learning_rate": all_params["lgbm__learning_rate"],
                    "lgbm__n_estimators": all_params["lgbm__n_estimators"],
                    "lgbm__subsample": all_params["lgbm__subsample"],
                    "lgbm__colsample_bytree": all_params["lgbm__colsample_bytree"],
                    "lgbm__scale_pos_weight":all_params["lgbm__scale_pos_weight"],
                    "lgbm__min_split_gain":all_params["lgbm__min_split_gain"],
                    "lgbm__num_class":all_params["lgbm__num_class"],
                    "random_state": all_params["random_state"],
                    "verbosity": all_params["verbosity"]
                  }
        
        # Pipelines
        text_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=all_params["preprocessor__text__pca__n_components"]))])
        image_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=all_params["preprocessor__image__pca__n_components"]))])
        preprocessor = ColumnTransformer([
             ("text", text_pipeline, text_feat_cols),
             ("image", image_pipeline, image_feat_cols)])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("lgbm", LGBMClassifier(**lgbm__params))])

        f1_scores = []

        for train_idx, val_idx in split_operator.split(X_train, y_train):
          X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
          y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]


          pipeline.fit(X_tr, y_tr)
          preds = pipeline.predict(X_val)
          f1 = metric(y_val, preds, average="weighted")
          f1_scores.append(f1)

        # Log to MLflow
        mlflow.log_params(all_params)
        mlflow.log_metric("meaned weighted f1 score", np.mean(f1_scores))
        mlflow.log_metric("std weighted f1 score", np.std(f1_scores))

        return np.mean(f1_scores)-np.std(f1_scores)
    return objective



def objective_wrapper_lgbm(split_operator, X_train, y_train,num_class, metric,):
    
    def objective(trial):

      with mlflow.start_run(nested=True):
        
        max_depth = trial.suggest_int("max_depth", 3, 20)
        max_num_leaves = min(2**max_depth, 200)
        num_leaves = trial.suggest_int("num_leaves", 50, max_num_leaves)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
        n_estimators= trial.suggest_int("n_estimators", 100, 500)
        min_split_gain = trial.suggest_float("min_split_gain",0,1)
        subsample = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 20, 80)
        params = {
            "num_leaves": max_depth,
            "max_depth": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_split_gain": min_split_gain,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "num_class":num_class,
            "random_state": 42,
           "verbosity": -1
          }
        
        pipeline = LGBMClassifier(**params)

        f1_scores = []

        for train_idx, val_idx in split_operator.split(X_train, y_train):
          X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
          y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]


          pipeline.fit(X_tr, y_tr)
          preds = pipeline.predict(X_val)
          f1 = metric(y_val, preds, average="weighted")
          f1_scores.append(f1)

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("meaned weighted f1 score", np.mean(f1_scores))
        mlflow.log_metric("std weighted f1 score", np.std(f1_scores))

        return np.mean(f1_scores)-np.std(f1_scores)
    return objective

###fonction permettant d'établir automatiquement des nom de run incrémentale en fonctipon des exp déjà existante dans mlflow
def ordinal(n):
    return ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"][n-1] if n <= 10 else f"{n}th"

def get_next_run_name(base_name="attempt",experiment_name="GP_optuna_lightgbm_stratified_ops"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)  # remplace par le nom réel
    if not experiment:
        return f"first_{base_name}"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    existing_names = [run.data.tags.get("mlflow.runName") for run in runs]

    # Trouver le prochain suffixe
    i = 1
    while True:
        candidate = f"{ordinal(i)}_{base_name}"
        if candidate not in existing_names:
            return candidate
        i += 1




#fonction d'evaluation permettant de comparer une liste de versions d'un modèle enregistré dans le registre MLflow
def compare_model_versions(model_name, versions, X_test, y_test, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")):
    best_score = -float('inf')
    best_version = None

    for version in versions:
        # chargement des modèles depuis le registre
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # predictions
        predictions = model.predict(X_test)

        # métrique
        score = metric(y_test, predictions)
        print(f"Model {model_name} version {version} - Score: {score}")

        # miettre à jour le meilleur modèle si besoin
        if score > best_score:
            best_score = score
            best_version = version
    
    # Add a tag to the best model in the registry
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(model_name, "best_model_version", best_version)
    print(f"Best model version: {best_version} with score: {best_score}")
    return best_version


def compare_best_models(model_names: list, X_test, y_test, metric=lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted")):
    
    best_score = -float('inf')
    best_model = None

    client = mlflow.tracking.MlflowClient()

    for model_name in model_names:
        # Get the latest two versions of the model
        versions = [v.version for v in client.search_model_versions(f"name='{model_name}'")[-2:]]
        
        # Use the compare_models function to determine the best version among the two latest
        best_version = compare_model_versions(model_name, versions, X_test, y_test, metric)

        # Load and evaluate the best version
        model_uri = f"models:/{model_name}/{best_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(X_test)
        score = metric(y_test, predictions)
        print(f"Best model for {model_name} (version {best_version}) - Score: {score}")

        if score > best_score:
            best_score = score
            best_model = model_name

    print(f"Overall best model: {best_model}, version :{best_version}, with score: {best_score}")
    return {"best_model":best_model, "best_version":best_version}


#Fonction permettant de promouvoir le meilleur modèle en production

def promotion_exclusive_best_model_to_production(model_name, model_version, alias="champion"):
    """
    Promeut model_version sous l'alias donné (supprime l'alias d'une autre version
    s'il existe déjà, puis le pose sur model_version).

    P.3a — alias paramétrable : 'champion' (Phase 1, défaut), ou
    'champion_stateless' / 'champion_stateful' (Phase 3).
    """
    client = mlflow.tracking.MlflowClient()

    # Supprimer l'alias "champion" s'il existe déjà
    try:
        current_alias_info = client.get_model_version_by_alias(model_name, alias)
        current_version = current_alias_info.version
        if current_version != model_version:
            print(f"Suppression de l'alias '{alias}' actuellement sur version {current_version}")
            client.delete_registered_model_alias(model_name, alias)
    except mlflow.exceptions.MlflowException:
        print(f"Alias '{alias}' non trouvé, aucun conflit à résoudre.")

    # Ajouter alias "champion" à la nouvelle version
    print(f"Ajout de l'alias '{alias}' à {model_name} v{model_version}")
    client.set_registered_model_alias(model_name, alias=alias, version=model_version)

    model_uri = f"models:/{model_name}@{alias}"
    return model_uri

def get_f1_score_from_model_uri(model_uri: str, metric_name: str = "f1_score") -> float:
    try:
        if not model_uri.startswith("models:/") or "@" not in model_uri:
            raise ValueError(f"Format de model_uri non supporté : {model_uri}")

        # Extraire nom du modèle et alias dynamiquement
        uri_body = model_uri[len("models:/"):]
        model_name, alias = uri_body.split("@")

        # Accéder au run ID via le model registry
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(model_name, alias)
        run_id = version_info.run_id

        # Récupérer la métrique depuis le run
        run_data = client.get_run(run_id).data.metrics
        f1_score = run_data.get(metric_name)

        if f1_score is None:
            raise ValueError(f"Métrique '{metric_name}' non trouvée dans le run {run_id}")
        
        return f1_score

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la récupération du f1-score depuis {model_uri} : {e}")


@dataclass
class PromotionResult:
    """Résultat de l'évaluation de promotion @champion."""
    promoted: bool
    reason: str
    model_name: str
    candidate_version: str
    candidate_score: float
    champion_version: Optional[str] = None
    champion_score: Optional[float] = None
    gain: Optional[float] = None
    epsilon: float = 0.0

    def __str__(self) -> str:
        head = "✓ PROMOTED" if self.promoted else "✗ NOT PROMOTED"
        out = f"{head} : {self.model_name} v{self.candidate_version}"
        out += f"\n  reason: {self.reason}"
        out += f"\n  candidate_score: {self.candidate_score:.4f}"
        if self.champion_score is not None:
            out += f"\n  champion_score: {self.champion_score:.4f} (v{self.champion_version})"
            out += f"\n  gain: {self.gain:+.4f} (threshold: {self.epsilon:+.4f})"
        return out


def evaluate_promotion_via_logged_metrics(
    model_name: str,
    candidate_version: str,
    metric_key: str = "eval_gold/f1_weighted",
    epsilon: float = 0.005,
    champion_alias: str = "champion",
) -> PromotionResult:
    """
    Compare un candidat au champion actuel via les métriques MLflow déjà loggées
    (pas de prédiction live).

    Logique :
    - Alias absent → promotion automatique comme baseline
    - Alias existant → promotion si (candidate - champion) >= epsilon

    P.3a — champion_alias paramétrable : 'champion' (Phase 1, défaut),
    'champion_stateless' / 'champion_stateful' (Phase 3). Chaque lignée
    se compare à SA propre référence.

    Note statistique : avec n_gold ≈ 8500 et F1 ≈ 0.85, σ_F1 ≈ 0.004.
    Un epsilon = 0.005 ≈ 1.3σ correspond à ~90% de confiance unilatérale.
    Pour 95% de confiance unilatérale, prendre epsilon = 0.007.

    Args:
        model_name: nom du modèle dans le registry
        candidate_version: version du candidat fraîchement enregistrée
        metric_key: clé MLflow de la métrique de comparaison
        epsilon: seuil de gain minimal pour promotion

    Returns:
        PromotionResult avec décision et justification (et exécution de la promotion si applicable).
    """
    import mlflow
    client = mlflow.tracking.MlflowClient()

    # 1. Score du candidat (lit la métrique loggée pendant le fit)
    candidate_mv = client.get_model_version(model_name, candidate_version)
    candidate_run = client.get_run(candidate_mv.run_id)
    candidate_score = candidate_run.data.metrics.get(metric_key)
    if candidate_score is None:
        raise ValueError(
            f"Candidate v{candidate_version} n'a pas la métrique '{metric_key}'. "
            "Vérifier que _log_eval_matrix() est bien appelé pendant le fit."
        )

    # 2. Score du champion actuel (s'il existe)
    try:
        champion_uri = f"models:/{model_name}@{champion_alias}"
        champion_score = get_f1_score_from_model_uri(champion_uri, metric_name=metric_key)
        champion_mv = client.get_model_version_by_alias(model_name, champion_alias)
        champion_version = champion_mv.version
    except (mlflow.exceptions.MlflowException, RuntimeError):
       # Alias absent : promotion automatique (démarrage de lignée)
        promotion_exclusive_best_model_to_production(model_name, candidate_version, alias=champion_alias)
        return PromotionResult(
            promoted=True,
            reason=f"first_{champion_alias} (no previous {champion_alias} in registry)",
            model_name=model_name,
            candidate_version=candidate_version,
            candidate_score=candidate_score,
            epsilon=epsilon,
        )

    # 3. Comparaison avec seuil ε
    gain = candidate_score - champion_score
    if gain >= epsilon:
        promotion_exclusive_best_model_to_production(model_name, candidate_version, alias=champion_alias)
        return PromotionResult(
            promoted=True,
            reason=f"gain {gain:+.4f} >= epsilon {epsilon:+.4f} (significant)",
            model_name=model_name,
            candidate_version=candidate_version,
            candidate_score=candidate_score,
            champion_version=champion_version,
            champion_score=champion_score,
            gain=gain,
            epsilon=epsilon,
        )
    else:
        return PromotionResult(
            promoted=False,
            reason=f"gain {gain:+.4f} < epsilon {epsilon:+.4f} (not significant)",
            model_name=model_name,
            candidate_version=candidate_version,
            candidate_score=candidate_score,
            champion_version=champion_version,
            champion_score=champion_score,
            gain=gain,
            epsilon=epsilon,
        )
    
# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Convention val_selection versionné (Bloc M.0)
# ═════════════════════════════════════════════════════════════════════════════
#
# Le val_selection est un 3ème split orthogonal au gold, qui sert d'arbitre
# pour les promotions @active (au niveau base learner) et @active_text /
# @active_image (au niveau modalité). Il est versionné car re-créé à chaque
# ingestion d'un nouveau batch (v1 = batch_1, v2 = batch_1∪batch_2, etc.).
#
# La variable d'environnement ACTIVE_VAL_SELECTION_VERSION détermine quelle
# version est utilisée pour résoudre les colonnes is_val_selection_v{N} dans
# _df_full et pour piloter les promotions @active.
#
# Voir checklist_phase_1.md (section "Conventions Phase 1") pour la spec
# complète. Détails M.0 dans scripts/init_val_selection.py.
# ═════════════════════════════════════════════════════════════════════════════
 
ACTIVE_VAL_SELECTION_VERSION_ENV = "ACTIVE_VAL_SELECTION_VERSION"
DEFAULT_VAL_SELECTION_VERSION = 1  # Phase 1 démarrage : v1 obligatoire après init
 
 
def get_active_val_selection_version() -> int:
    """
    Retourne la version courante du val_selection à utiliser pour les évaluations
    et les promotions @active.
 
    Sources de résolution (premier non-None gagne) :
      1. Variable d'environnement ACTIVE_VAL_SELECTION_VERSION
      2. Airflow Variable du même nom (si Airflow disponible dans le contexte)
      3. Valeur par défaut DEFAULT_VAL_SELECTION_VERSION (= 1)
 
    Le retour est int ∈ {1, 2, 3} (Phase 1 limite à 3 batches).
 
    Raises:
        ValueError: si la valeur résolue n'est pas un entier dans [1, 3].
 
    Examples:
        >>> os.environ["ACTIVE_VAL_SELECTION_VERSION"] = "2"
        >>> get_active_val_selection_version()
        2
        >>> del os.environ["ACTIVE_VAL_SELECTION_VERSION"]
        >>> get_active_val_selection_version()
        1
    """
    raw: Optional[str] = os.environ.get(ACTIVE_VAL_SELECTION_VERSION_ENV)
 
    if raw is None:
        raw = _try_read_airflow_variable(ACTIVE_VAL_SELECTION_VERSION_ENV)
 
    if raw is None:
        logger.debug(
            f"{ACTIVE_VAL_SELECTION_VERSION_ENV} non défini, utilise défaut "
            f"= {DEFAULT_VAL_SELECTION_VERSION}"
        )
        return DEFAULT_VAL_SELECTION_VERSION
 
    try:
        version = int(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{ACTIVE_VAL_SELECTION_VERSION_ENV}={raw!r} n'est pas un entier valide. "
            f"Attendu : 1, 2 ou 3."
        ) from e
 
    if version not in (1, 2, 3):
        raise ValueError(
            f"{ACTIVE_VAL_SELECTION_VERSION_ENV}={version} hors plage. "
            f"Attendu : 1, 2 ou 3 (Phase 1)."
        )
 
    return version
 
 
def _try_read_airflow_variable(key: str) -> Optional[str]:
    """Lecture best-effort d'une Airflow Variable, retourne None si Airflow
    n'est pas dans le contexte d'exécution (ex: lancement local hors DAG)."""
    try:
        from airflow.models import Variable  # noqa: WPS433 (lazy import volontaire)
        return Variable.get(key, default_var=None)
    except Exception:  # ImportError, AirflowException, etc.
        return None
 
 
# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Helpers MLflow alias @active (Bloc M.3) — STUBS
# ═════════════════════════════════════════════════════════════════════════════
#
# La mécanique @active / @active_text / @active_image complète sera implémentée
# en M.3 (cf. checklist_phase_1.md, section "Conventions Phase 1"). Pour M.0,
# on expose juste les stubs pour que les imports compilent. L'implémentation
# détaillée arrive avec M.4 (BaseLearnerExperiment).
#
# ⚠ NE PAS CONFONDRE avec les helpers @champion existants plus haut dans ce
# fichier (promotion_exclusive_best_model_to_production, etc.). Les deux
# mécaniques coexistent :
#   - @champion : arbitre = gold, niveau assembled (Bloc P)
#   - @active   : arbitre = val_selection, niveau base learner (Bloc M.4)
# ═════════════════════════════════════════════════════════════════════════════
 
def resolve_active_modality(modality: str) -> tuple[str, int]:
    """
    Retourne (registered_model_name, version) du base learner promu
    @active_<modality>.
 
    Scan toutes les registered models dont le nom commence par 'rakuten-base-',
    cherche celui qui porte l'alias 'active_<modality>'.
 
    Args:
        modality: 'text' | 'image' | 'tabular'
 
    Returns:
        (registered_model_name, version) du base learner @active_<modality>
 
    Raises:
        ValueError: si modality invalide.
        RuntimeError: si aucun base learner ne porte cet alias (Phase 1 démarrage
            ou avant le premier fit_base_learner d'une modalité).
    """
    if modality not in ("text", "image", "tabular"):
        raise ValueError(
            f"modality={modality!r} invalide. Attendu : 'text', 'image' ou 'tabular'."
        )
 
    client = mlflow.tracking.MlflowClient()
    alias_name = f"active_{modality}"
 
    for rm in client.search_registered_models():
        if not rm.name.startswith("rakuten-base-"):
            continue
        try:
            mv = client.get_model_version_by_alias(rm.name, alias_name)
            return (rm.name, int(mv.version))
        except mlflow.exceptions.MlflowException:
            continue
 
    raise RuntimeError(
        f"Aucun base learner ne porte l'alias @{alias_name}. "
        f"Lancer fit_base_learner pour un modèle de modalité {modality!r} d'abord."
    )
 
 
def refresh_modality_alias(modality: str) -> None:
    """
    Cascade auto : pose @active_<modality> sur le base learner dont le @active
    a le meilleur F1 sur le val_selection actif. Retire l'alias des autres.
 
    Cette fonction est appelée à la fin de BaseLearnerExperiment.fit() pour
    propager une promotion @active au niveau modalité (cf. checklist Phase 1
    Bloc M.3).
 
    Args:
        modality: 'text' | 'image' | 'tabular'
 
    Raises:
        ValueError: si modality invalide.
 
    Note:
        Si aucun candidat valide (= aucun base learner taggé modality={modality}
        avec un @active porteur de la métrique val_selection_v{n}), log un
        warning et retourne sans rien faire.
    """
    if modality not in ("text", "image", "tabular"):
        raise ValueError(
            f"modality={modality!r} invalide. Attendu : 'text', 'image' ou 'tabular'."
        )
 
    client = mlflow.tracking.MlflowClient()
    n = get_active_val_selection_version()
    metric_key = f"val_selection_v{n}/f1_weighted"
    alias_name = f"active_{modality}"
 
    best_name: Optional[str] = None
    best_version: Optional[int] = None
    best_f1: float = -1.0
    candidates: list[str] = []
 
    for rm in client.search_registered_models():
        if not rm.name.startswith("rakuten-base-"):
            continue
        if rm.tags.get("modality") != modality:
            continue
        try:
            mv = client.get_model_version_by_alias(rm.name, "active")
        except mlflow.exceptions.MlflowException:
            continue
        candidates.append(rm.name)
        try:
            run = client.get_run(mv.run_id)
            f1 = run.data.metrics.get(metric_key)
            if f1 is None:
                continue
        except mlflow.exceptions.MlflowException:
            continue
 
        if f1 > best_f1:
            best_f1, best_name, best_version = f1, rm.name, int(mv.version)
 
    if best_name is None:
        logger.warning(
            f"refresh_modality_alias({modality!r}): aucun candidat avec @active "
            f"+ métrique '{metric_key}'. Pas de promotion @{alias_name}."
        )
        return
 
    # Retirer @active_{modality} des autres candidats (s'il existait)
    for cand_name in candidates:
        if cand_name == best_name:
            continue
        try:
            existing_mv = client.get_model_version_by_alias(cand_name, alias_name)
            client.delete_registered_model_alias(cand_name, alias_name)
            logger.info(
                f"Retiré @{alias_name} de {cand_name} v{existing_mv.version}"
            )
        except mlflow.exceptions.MlflowException:
            pass
 
    # Poser @active_{modality} sur le meilleur
    client.set_registered_model_alias(best_name, alias_name, best_version)
    logger.info(
        f"Posé @{alias_name} sur {best_name} v{best_version} "
        f"(F1 val_selection_v{n}={best_f1:.4f})"
    )
 
 
def compute_promotion_decision(
    name: str,
    run_id_new: str,
    threshold: float = 0.005,
    alias: str = "active",
) -> bool:
    """
    True ssi la nouvelle version bat l'alias courant de > threshold sur le
    val_selection actif.
 
    Lit ACTIVE_VAL_SELECTION_VERSION pour résoudre la métrique correcte :
    'val_selection_v{n}/f1_weighted'.
 
    Args:
        name: registered_model_name (ex: 'rakuten-base-textcnn').
        run_id_new: run_id du fit qu'on évalue.
        threshold: marge minimale de gain F1 (défaut 0.005 ≈ 1.3σ à n_gold≈8500).
        alias: alias de référence à battre. Défaut 'active' (rétro-compat
            Phase 1). En Phase 3 multi-lignées, la lignée passe son propre
            alias : 'active_stateless' ou 'active_stateful', de sorte que
            chaque trajectoire se compare à SA propre référence et non à
            celle de l'autre lignée.
 
    Returns:
        True ssi la promotion doit être effectuée sur l'alias de référence.
        Cas particuliers :
        - Alias de référence inexistant → True (démarrage de lignée)
        - Alias de référence sans métrique val_selection_v{n} → True (incomparable,
          on remplace par sécurité avec un warning)
 
    Raises:
        ValueError: si run_id_new n'a pas la métrique 'val_selection_v{n}/f1_weighted'.
    """
    client = mlflow.tracking.MlflowClient()
    n = get_active_val_selection_version()
    metric_key = f"val_selection_v{n}/f1_weighted"
 
    # 1. Score du candidat
    new_run = client.get_run(run_id_new)
    new_f1 = new_run.data.metrics.get(metric_key)
    if new_f1 is None:
        raise ValueError(
            f"Run {run_id_new} n'a pas la métrique '{metric_key}'. "
            f"Vérifier que BaseLearnerExperiment.fit() loggue bien cette métrique."
        )
 
    # 2. Score de l'@active courant (si existe)
    try:
        current_mv = client.get_model_version_by_alias(name, alias)
        current_run = client.get_run(current_mv.run_id)
        current_f1 = current_run.data.metrics.get(metric_key)
        if current_f1 is None:
            logger.warning(
                f"@{alias} de {name} (v{current_mv.version}) n'a pas '{metric_key}'. "
                f"Promotion par défaut (incomparable)."
            )
            return True
    except mlflow.exceptions.MlflowException:
        # Pas d'alias courant : first promotion (démarrage de lignée)
        return True
 
    # 3. Comparaison
    delta = new_f1 - current_f1
    return delta > threshold

def embedding_cache_filename(
    learner_name: str,
    version: int,
    strategy: str = "stateless",
) -> str:
    """
    Nom canonique du cache parquet d'embeddings d'un base learner.

    Source de vérité UNIQUE du naming (write : _write_cache_parquet ;
    read : DataModule._load_base_learner_embeddings, resolve dynamique M2).

    Convention lignées (P.2) :
      - stateless → nom historique Phase 1 : embeddings_{name}_v{n}.parquet
        (rétro-compat : les caches existants SONT la lignée stateless,
        aucune régénération GPU requise)
      - stateful  → embeddings_{name}_stateful_v{n}.parquet
        (les deux lignées n'écrivent jamais le même fichier → pas de
        contamination par le cache en mode compare)

    Args:
        learner_name: nom court ('camembert_lora', ...).
        version: version du val_selection (ACTIVE_VAL_SELECTION_VERSION).
        strategy: 'stateless' | 'stateful'.
    """
    if strategy not in ("stateless", "stateful"):
        raise ValueError(
            f"strategy={strategy!r} invalide. Attendu 'stateless' ou 'stateful'."
        )
    suffix = "" if strategy == "stateless" else "_stateful"
    return f"embeddings_{learner_name}{suffix}_v{version}.parquet"
 
def ensure_device(model_or_learner, device=None):
    """
    Migre un modèle/learner vers GPU si disponible.
    
    Gère les deux patterns :
    - BaseLearner avec .net (CamemBERT, ResNet, SigLIP, TextCNN)
    - Module PyTorch direct (M3AttentionFusion, M32CoAdaptation)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Pattern BaseLearner (net interne)
    if hasattr(model_or_learner, "net") and model_or_learner.net is not None:
        model_or_learner.net.to(device)
        model_or_learner.net.eval()
    
    # Pattern Module PyTorch direct (M3, M3.2)
    elif hasattr(model_or_learner, "to") and hasattr(model_or_learner, "eval"):
        model_or_learner.to(device)
        model_or_learner.eval()
    
    return device


def resolve_active_for_fusion(
    strategy: str,
    modalities: tuple[str, ...] = ("text", "image"),
) -> dict[str, tuple[str, int]]:
    """
    Sélectionne, pour chaque modalité, le meilleur base learner de la lignée
    `strategy` — SANS poser d'alias.

    Remplace la cascade @active_{modality} sur le chemin Phase 3 : au lieu de
    persister un pointeur EXCLUSIF (qui serait partagé entre lignées, donc
    source de contamination en mode compare), on CALCULE la sélection à la
    volée et on la transmet en versions épinglées (XCom → --set).

    Scan : tous les 'rakuten-base-*' taggés modality={m} qui portent l'alias
    @active_{strategy} ; garde celui dont le F1 sur le val_selection actif est
    le meilleur.

    Args:
        strategy: 'stateless' ou 'stateful'. Détermine l'alias lu :
            @active_{strategy}.
        modalities: modalités à résoudre (défaut : text + image).

    Returns:
        {modality: (registered_model_name, version)}
        ex: {"text": ("rakuten-base-camembert-lora", 9),
             "image": ("rakuten-base-siglip2", 4)}

    Raises:
        ValueError: si strategy invalide.
        RuntimeError: si une modalité n'a aucun candidat (aucun base learner de
            cette modalité ne porte @active_{strategy} avec la métrique
            val_selection_v{n} — typiquement au batch 1 d'une lignée stateful).
    """
    if strategy not in ("stateless", "stateful"):
        raise ValueError(
            f"strategy={strategy!r} invalide. Attendu 'stateless' ou 'stateful'."
        )

    client = mlflow.tracking.MlflowClient()
    n = get_active_val_selection_version()
    metric_key = f"val_selection_v{n}/f1_weighted"
    alias_name = f"active_{strategy}"

    result: dict[str, tuple[str, int]] = {}

    for modality in modalities:
        best_name: Optional[str] = None
        best_version: Optional[int] = None
        best_f1: float = -1.0

        for rm in client.search_registered_models():
            if not rm.name.startswith("rakuten-base-"):
                continue
            if rm.tags.get("modality") != modality:
                continue
            try:
                mv = client.get_model_version_by_alias(rm.name, alias_name)
            except mlflow.exceptions.MlflowException:
                continue  # ce learner n'a pas (encore) de version dans cette lignée
            try:
                run = client.get_run(mv.run_id)
            except mlflow.exceptions.MlflowException:
                continue
            f1 = run.data.metrics.get(metric_key)
            if f1 is None:
                logger.warning(
                    f"{rm.name} @{alias_name} (v{mv.version}) n'a pas '{metric_key}' "
                    f"→ écarté de la sélection {modality}."
                )
                continue

            if f1 > best_f1:
                best_f1, best_name, best_version = f1, rm.name, int(mv.version)

        if best_name is None:
            raise RuntimeError(
                f"Aucun base learner de modalité {modality!r} ne porte "
                f"@{alias_name} avec la métrique '{metric_key}'. "
                f"La lignée {strategy!r} a-t-elle déjà tourné sur ce batch ?"
            )

        result[modality] = (best_name, best_version)
        logger.info(
            f"resolve_active_for_fusion({strategy!r}) : {modality} → "
            f"{best_name} v{best_version} (F1 {metric_key}={best_f1:.4f})"
        )

    return result