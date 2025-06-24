import mlflow
from utils import compare_model_versions, compare_best_models, promotion_exclusive_best_model_to_production
import dagshub
import pandas as pd
from pymongo import MongoClient
repo_owner='GuillaumePe'
repo_name='mar25_cmlops_rakuten'

list_models_name = ["pca_lgbm_pipeline"]

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]
TARGET_COLUMN = "prdtypecode"

X_test = pd.DataFrame(list(db.X_test_final.find({}, {"_id": 0})))
y_test_df = pd.DataFrame(list(db.Y_test_final.find({}, {"_id": 0, "productid": 1, TARGET_COLUMN: 1})))
y_test = y_test_df[TARGET_COLUMN]


dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

best_model_dict = compare_best_models(X_test=X_test, y_test=y_test, model_names=list_models_name)

promotion_exclusive_best_model_to_production(best_model_dict["best_model"],best_model_dict["best_version"])
