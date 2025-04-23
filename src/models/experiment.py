import pandas as pd
import mlflow
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import json
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

#Paramètres
repo_owner='GuillaumePe'
repo_name='mar25_cmlops_rakuten'
X_train_path = "mar25_cmlops_rakuten/data/preprocessed/final/X_train_processed_final.parquet"
Y_train_Path = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/Y_train_final.parquet"
LIST_ID_COLUMNS = ["imageid", "productid"]
TARGET_COLUMN = "prdtypecode" 

dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# === Chargement des données Polars ===
X_train = pd.read_parquet(X_train_path)
y_train = pd.read_csv(Y_train_Path)
X_train = X_train.sort_values(by=LIST_ID_COLUMNS)
y_train = y_train.sort_values(by=LIST_ID_COLUMNS)[TARGET_COLUMN]
# === Définition de l'espace de recherche ===
search_space = {
    "num_leaves": Integer(20, 150),
    "max_depth": Integer(3, 15),
    "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "n_estimators": Integer(50, 500),
    "subsample": Real(0.5, 1.0),
    "colsample_bytree": Real(0.5, 1.0),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

mlflow.set_experiment("GP_exp1_lightgbm_bayesian_search")

def mlflow_callback(search_result):
    with mlflow.start_run(nested=True):
        mlflow.log_params(search_result.cv_results_.params)
        mlflow.log_metric({"mean_weighted_f1":search_result.cv_results_.mean_test_score,
                           "std_weighted_f1":search_result.cv_results_.std_test_score})

opt = BayesSearchCV(
    estimator=LGBMClassifier(random_state=42),
    search_spaces=search_space,
    cv=cv,
    n_iter=30,
    n_jobs=-1,
    scoring="f1_weighted",
    verbose=0,
    random_state=42,
    refit=True
)

with mlflow.start_run(run_name="lgbm_bayesian_weighted_f1"):
    opt.fit(X_train, y_train, callback=mlflow_callback)

    mlflow.log_params(opt.best_params_)
    mlflow.log_metric("best_weighted_f1", opt.best_score_)

    y_pred = opt.predict(X_train)

    f1 = f1_score(y_train, y_pred, average="weighted")
    report = classification_report(y_train, y_pred, output_dict=True)
    cm = confusion_matrix(y_train, y_pred)

    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("classification_report.json")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

print("Best Params:", opt.best_params_)