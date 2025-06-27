import mlflow
#import dagshub
import pandas as pd
from pymongo import MongoClient
from src.models.utils import compare_best_models, promotion_exclusive_best_model_to_production

def select_and_promote_best_model(list_models_name= ["pca_lgbm_pipeline"], repo_owner='GuillaumePe', repo_name='mar25_cmlops_rakuten'):
    # Init MLflow via DagsHub
    #dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri("http://mlflow:5000")
    # Connexion MongoDB
    client = MongoClient("mongodb://localhost:27017")
    db = client["MAR25_CMLOPS_RAKUTEN"]
    TARGET_COLUMN = "prdtypecode"

    # Données de test
    X_test = pd.DataFrame(list(db.X_test_final.find({}, {"_id": 0})))
    y_test_df = pd.DataFrame(list(db.Y_test_final.find({}, {"_id": 0, "productid": 1, TARGET_COLUMN: 1})))
    y_test = y_test_df[TARGET_COLUMN]

    # Comparaison et promotion
    best_model_dict = compare_best_models(X_test=X_test, y_test=y_test, model_names=list_models_name)
    model_uri = promotion_exclusive_best_model_to_production(
        best_model_dict["best_model"],
        best_model_dict["best_version"]
    )
    return model_uri