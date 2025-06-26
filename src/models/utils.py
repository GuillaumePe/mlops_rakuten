import mlflow
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

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

def promotion_exclusive_best_model_to_production(model_name, model_version):
    client = mlflow.tracking.MlflowClient()

    # Supprimer l'alias "champion" s'il existe déjà
    try:
        current_alias_info = client.get_model_version_by_alias(model_name, "champion")
        current_version = current_alias_info.version
        if current_version != model_version:
            print(f"Suppression de l'alias 'champion' actuellement sur version {current_version}")
            client.delete_registered_model_alias(model_name, "champion")
    except mlflow.exceptions.RestException:
        print("Alias 'champion' non trouvé, aucun conflit à résoudre.")

    # Ajouter alias "champion" à la nouvelle version
    print(f"Ajout de l'alias 'champion' à {model_name} v{model_version} pour mise en Production")
    client.set_registered_model_alias(model_name, alias="champion", version=model_version)

    model_uri = f"models:/{model_name}@champion"
    return model_uri
