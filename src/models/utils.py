import mlflow
import optuna
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier


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

def objective_wrapper_pca(split_operator, X_train, y_train,num_class, metric,):
    
    def objective(trial):

      with mlflow.start_run(nested=True):
    
        text_feat_cols = [col for col in X_train.columns if col.startswith("text_feat_")]
        image_feat_cols = [col for col in X_train.columns if col.startswith("image_feat_")]
    
        #Search Space
        preprocessor__text__pca__n_components = trial.suggest_int("preprocessor__text__pca__n_components", 100, 150)
        preprocessor__image__pca__n_components = trial.suggest_int("preprocessor__image__pca__n_components",100, 150)
        
        lgbm__max_depth = trial.suggest_int("max_depth", 3, 20)
        max_num_leaves = max(min(2**lgbm__max_depth, 200),50)
        lgbm__num_leaves = trial.suggest_int("num_leaves", 50, max_num_leaves)
        lgbm__learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
        lgbm__n_estimators= trial.suggest_int("n_estimators", 100, 500)
        lgbm__minc_split_gain = trial.suggest_float("min_split_gain",0,1)
        lgbm__subsample = trial.suggest_float("subsample", 0.6, 1.0),
        lgbm__colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0)
        lgbm__scale_pos_weight = trial.suggest_float("scale_pos_weight", 20, 80)
        params = {
            "lgbm__num_leaves": lgbm__num_leaves,
            "lgbm__max_depth": lgbm__max_depth,
            "lgbm__learning_rate": lgbm__learning_rate,
            "lgbm__n_estimators": lgbm__n_estimators,
            "lgbm__subsample": lgbm__subsample,
            "lgbm__colsample_bytree": lgbm__colsample_bytree,
            "lgbm__scale_pos_weight":lgbm__scale_pos_weight,
            "lgbm__minc_split_gain":lgbm__minc_split_gain,
            "lgbm__num_class":num_class,
            "random_state": 42,
           "verbosity": -1
          }
        preprocessing_params = {"preprocessor__text__pca__n_components": preprocessor__text__pca__n_components,
                                "preprocessor__image__pca__n_components": preprocessor__image__pca__n_components}

        all_params = {**preprocessing_params, **params}
    
        # Pipelines
        text_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=preprocessor__text__pca__n_components))])
        image_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=preprocessor__image__pca__n_components))])
        preprocessor = ColumnTransformer([
             ("text", text_pipeline, text_feat_cols),
             ("image", image_pipeline, image_feat_cols)])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("lgbm", LGBMClassifier(**params))])

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

