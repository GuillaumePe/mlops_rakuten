import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import dagshub
from utils import get_or_create_experiment, objective_wrapper_pca, champion_callback


#Paramètres
repo_owner='GuillaumePe'
repo_name='mar25_cmlops_rakuten'
X_train_path = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/X_train_processed_final.parquet"
Y_train_Path = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/Y_train_final.parquet"
LIST_ID_COLUMNS = ["imageid", "productid"]
TARGET_COLUMN = "prdtypecode" 
n_trials_bayesian_search = 10


# Chargement des données Polars 
X = pd.read_parquet(X_train_path)

y = pd.read_parquet(Y_train_Path)
X = X.sort_values(by=LIST_ID_COLUMNS)
y = y.sort_values(by=LIST_ID_COLUMNS)[TARGET_COLUMN]
num_class = y.nunique()
X = X.drop(columns=LIST_ID_COLUMNS, errors="raise") 
# Identification des colonnes text et image
text_feat_cols = [col for col in X.columns if col.startswith("text_feat_")]
image_feat_cols = [col for col in X.columns if col.startswith("image_feat_")]


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)
# Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialisation Dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

#MLflow 
experiment_id = get_or_create_experiment("GP_optuna_lightgbm_stratified")
run_name = "second_attempt"

mlflow.set_experiment(experiment_id=experiment_id)

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
  # Initialize the Optuna study
  study = optuna.create_study(direction="maximize")

  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
  study.optimize(objective_wrapper_pca(
        split_operator=skf,
        X_train=X_train,
        y_train=y_train,
        num_class=num_class,
        metric=f1_score),
    n_trials=n_trials_bayesian_search,
    callbacks=[champion_callback])

  mlflow.log_params(study.best_params)
  mlflow.log_metric("mean-std of weighted f1 score", study.best_value)
  

  # Log tags
  mlflow.set_tags(
      tags={
          "project": "MA25_CMLOPS_RAKUTEN",
          "optimizer_engine": "optuna",
          "Pipline": "distilbert et resnet18 suivis de PCA et regroupé en entrée d'un LightGBM",
          "feature_set_version": 1,
      }
  )
  best_params = study.best_params.copy()
  lgbm_params = {k.replace("lgbm__", ""): v for k, v in best_params.items() if k.startswith("lgbm__")}

  # Final Pipeline
  text_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=best_params.pop("preprocessor__text__pca__n_components")))])
  image_pipeline = Pipeline([
             ("scaler", StandardScaler()),
             ("pca", PCA(n_components=best_params.pop("preprocessor__image__pca__n_components")))])
  preprocessor = ColumnTransformer([
             ("text", text_pipeline, text_feat_cols),
             ("image", image_pipeline, image_feat_cols)])
  final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("lgbm", LGBMClassifier(**lgbm_params))])

  final_pipeline.fit(X_test, y_test)
  y_pred_finale = final_pipeline.predict(X_test)
  f1_finale = f1_score(y_test, y_pred_finale, average="weighted")

  mlflow.log_metric("optuna_best_model", f1_finale)

 # Confusion matrix
  cm = confusion_matrix(y_test, y_pred_finale)
  plt.figure(figsize=(12, 10))
  sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
  plt.title("Confusion Matrix")
  plt.xlabel("Predicted")
  plt.ylabel("True")
  plt.tight_layout()
  plt.savefig("confusion_matrix.png")
  mlflow.log_artifact("confusion_matrix.png")

 # Classification report
  report = classification_report(y_test, y_pred_finale, output_dict=True)
  with open("classification_report.json", "w") as f:
    json.dump(report, f, indent=4)
  mlflow.log_artifact("classification_report.json")

  mlflow.log_params(best_params)

