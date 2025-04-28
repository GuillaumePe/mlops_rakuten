import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import dagshub


#Paramètres
repo_owner='GuillaumePe'
repo_name='mar25_cmlops_rakuten'

# Identifiants de l'expérience & du run Optuna
EXPERIMENT_NAME = "GP_optuna_lightgbm_stratified"
RUN_NAME        = "second_attempt"

# Chemins vers vos données pré-traitées
X_train_path = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/X_train_processed_final.parquet"
Y_train_Path = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/Y_train_final.parquet"

# Colonnes ID et cible
LIST_ID_COLUMNS = ["imageid", "productid"]
TARGET_COLUMN   = "prdtypecode"

#Chargement des données
# Chargement des données Polars 
X_train = pd.read_parquet(X_train_path)
y_train = pd.read_parquet(Y_train_Path)

X_train = X_train.sort_values(by=LIST_ID_COLUMNS)
y_train = y_train.sort_values(by=LIST_ID_COLUMNS)

y_train = y_train[TARGET_COLUMN]
num_class = y_train.nunique()
X_train = X_train.drop(columns=LIST_ID_COLUMNS, errors="raise") 

text_cols  = [c for c in X_train.columns if c.startswith("text_feat_")]
image_cols = [c for c in X_train.columns if c.startswith("image_feat_")]


#Initialisation Dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# Récupérer l'expérience par nom
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise ValueError(f"Expérience '{EXPERIMENT_NAME}' introuvable dans MLflow")

# Chercher le run par son tag 'mlflow.runName'
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string=f"tags.mlflow.runName = '{RUN_NAME}'",
    max_results=1
)
if runs.empty:
    raise ValueError(f"Aucun run nommé '{RUN_NAME}' dans l'expérience '{EXPERIMENT_NAME}'")


# Extraction des best paramètres 
best_run = runs.iloc[0]
run_id = best_run["run_id"]
best_run_obj = mlflow.get_run(run_id)
params = best_run_obj.data.params

print(params)

# Paramètres PCA
n_text_pca  = int(params["preprocessor__text__pca__n_components"])
n_img_pca   = int(params["preprocessor__image__pca__n_components"])

# Paramètres LightGBM
lgbm_params = {}
for k, v in params.items():
    if k.startswith("lgbm__"):
        key = k.replace("lgbm__", "")
        # caster automatiquement selon le nom du param
        if key in {"max_depth", "num_leaves", "n_estimators","num_class"}:
            lgbm_params[key] = int(v)
        else:
            lgbm_params[key] = float(v)

# Ajouter quelques paramètres fixes
lgbm_params.update({
    "random_state": 42,
    "verbosity":    -1
})


# CONSTRUCTION DU PIPELINE FINAL

# Pipelines de prétraitement
text_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=n_text_pca))
])
img_pipeline  = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=n_img_pca))
])

preprocessor = ColumnTransformer([
    ("text",  text_pipeline,  text_cols),
    ("image", img_pipeline,   image_cols),
])

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("lgbm",         LGBMClassifier(**lgbm_params))
])

# entrainement et export

with mlflow.start_run(run_name=EXPERIMENT_NAME+"_"+"RUN_NAME"+"_"+"final_model_training_for_predicting"):
    # Entraînement
    final_pipeline.fit(X_train, y_train)
    
    # On logge le modèle (incluant tout le pipeline) au format MLflow
    mlflow.sklearn.log_model(
        sk_model     = final_pipeline,
        artifact_path= "model_for_predict"
    )
    
    print("Modèle final entraîné et enregistré dans MLflow sous 'model_for_predict'")

    # Facultatif : logger quelques métriques sur l'entraînement complet
    preds = final_pipeline.predict(X_train)
    f1    = f1_score(y_train, preds, average="weighted")
    mlflow.log_metric("f1_train", f1)
    print(f"F1 (train) = {f1:.4f}")
