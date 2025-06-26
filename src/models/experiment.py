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
from pymongo import MongoClient
from src.models.utils import get_or_create_experiment, objective_wrapper_pca, champion_callback, get_next_run_name


# Paramètres
repo_owner = 'GuillaumePe'
repo_name = 'mar25_cmlops_rakuten'
LIST_ID_COLUMNS = ["imageid", "productid"]
TARGET_COLUMN = "prdtypecode"
n_trials_bayesian_search = 2
ml_flow_experiment_name= "GP_optuna_lightgbm_stratified_ops"

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]

# Chargement X
X = pd.DataFrame(list(db.X_train_final.find({}, {"_id": 0})))
product_ids_in_X = set(X["productid"])

# Chargement complet de Y
Y_all = pd.DataFrame(list(db.Y_train_final.find({}, {"_id": 0, "productid": 1, TARGET_COLUMN: 1})))
#print(len(Y_all))
# Filtrage côté Pandas
Y = Y_all[Y_all["productid"].isin(product_ids_in_X)]
#print(len(Y_all))
# Alignement via tri sur les ID
X = X.sort_values(by=LIST_ID_COLUMNS)
Y = Y.sort_values(by="productid")
y = Y[TARGET_COLUMN]
#print(len(y))
# Drop colonnes ID de X
X = X.drop(columns=LIST_ID_COLUMNS, errors="raise")


# Séparation des colonnes
text_feat_cols = [col for col in X.columns if col.startswith("text_feat_")]
image_feat_cols = [col for col in X.columns if col.startswith("image_feat_")]
num_class = y.nunique()

# Split des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.20, random_state=42
)
# Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialisation Dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

#MLflow 
experiment_id = get_or_create_experiment(ml_flow_experiment_name)
run_name = get_next_run_name("attempt")

mlflow.set_experiment(experiment_id=experiment_id)

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
  # Initialize the Optuna study
  study = optuna.create_study(direction="maximize")
  
  hyperparameters = {
    "preprocessor__text__pca__n_components": ("int", 150, 250),
    "preprocessor__image__pca__n_components": ("int", 150, 250),
    "lgbm__max_depth": ("int", 3, 15),
    "lgbm__num_leaves": ("int", 50, 150),
    "lgbm__learning_rate": ("float", 0.005, 0.2),
    "lgbm__n_estimators": ("int", 400, 800),
    "lgbm__min_split_gain": ("float", 0.5, 1),
    "lgbm__subsample": ("float", 0.6, 0.8),
    "lgbm__colsample_bytree": ("float", 0.3, 0.8),
    "lgbm__scale_pos_weight": ("float", 20, 80)
}

  # Execute the hyperparameter optimization trials.
  # Note the addition of the `champion_callback` inclusion to control our logging
  study.optimize(objective_wrapper_pca(
        split_operator=skf,
        X_train=X_train,
        y_train=y_train,
        num_class=num_class,
        metric=f1_score,
        hyperparameters=hyperparameters),
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

  final_pipeline.fit(X_train, y_train)
  y_pred_finale = final_pipeline.predict(X_test)
  f1_finale = f1_score(y_test, y_pred_finale, average="weighted")
  
  mlflow.sklearn.log_model(
      sk_model=final_pipeline,
      artifact_path="model",
      registered_model_name="pca_lgbm_pipeline"
  )
  mlflow.log_metric("optuna_best_model", f1_finale)

 # Confusion matrix
  cm = confusion_matrix(y_test, y_pred_finale)
  plt.figure(figsize=(12, 12))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
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

