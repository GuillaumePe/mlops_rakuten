from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import subprocess
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from mlflow.tracking import MlflowClient
from src.features.build_images_features import build_images_features_func_from_mongo
from src.features.build_text_features import build_text_features_func_from_mongo
from src.models.model_selection import select_and_promote_best_model 
from pymongo import MongoClient

# Configuration
SECRET_KEY = "rakuten_secret_key"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
repo_owner = 'GuillaumePe'
repo_name = 'mar25_cmlops_rakuten'
db_client = MongoClient("mongodb://mongodb:27017")
db = db_client["MAR25_CMLOPS_RAKUTEN"]
Mlflow_tracking_uri = "http://mlflow:5000"

#User DB
users_db = {
    "admin": {
        "username": "admin",
        "password": "123admin"  
    }
}

#Définition des auth models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

#=Fonction pour l'authentification
def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or user["password"] != password:
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users_db:
            raise credentials_exception
        return users_db[username]
    except JWTError:
        raise credentials_exception

# ==== APP ====
app = FastAPI()
mlflow.set_tracking_uri(Mlflow_tracking_uri)

@app.get("/")
def read_root():
    return {"message": "API modèle ML Rakuten online"}

# Initialisation Dagshub
#dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

def get_champion_model():
    try:
        client = MlflowClient()
        all_models = client.search_registered_models()

        for model_info in all_models:
            for v in model_info.latest_versions:
                if "champion" in v.aliases:
                    model_uri = f"models:/{model_info.name}@champion"
                    loaded_model = mlflow.pyfunc.load_model(model_uri)
                    print(f"Modèle chargé depuis {model_uri}")
                    return loaded_model, model_info.name, v.version

        raise ValueError("Aucun modèle avec l'alias 'champion' trouvé (pas de modèle en prod)")
    
    except Exception as e:
        print(f"Erreur chargement modèle champion : {e}")
        return None, None, None

# Class pour le schéma
class PredictRequest_ids(BaseModel):
    productid: list[int]

# création des endpoints

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Identifiants invalides")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

model, model_name_loaded, model_version = get_champion_model()
@app.post("/predict")
def predict(request_ids: PredictRequest_ids, user: dict = Depends(get_current_user)):
    global model
    try:
        if model is None:
            raise ValueError("Modèle champion non chargé.")
        #Récupération des porduct_id présent dans la base à scorer
        docs_in_db = list(db["X_to_predict"].find({}, {"_id": 0, "productid": 1, "imageid": 1, "designation": 1}))
        df_in_db = pd.DataFrame(docs_in_db)
        request_ids_set = set(request_ids.productid)
        filtered_df = df_in_db[df_in_db["productid"].isin(request_ids_set)]
        filtered_ids_list = filtered_df["productid"].tolist()

        # Liste des productid exclus
        excluded_productids = list(request_ids_set - set(filtered_ids_list))

        # Message si certains IDs sont manquants
        if excluded_productids:
            print(f"{len(excluded_productids)} productid sur {len(request_ids_set)} ne se trouvent pas dans la base X_to_predict")

        #Pré-processing
        build_images_features_func_from_mongo(for_predicting=True, list_id=filtered_ids_list, source="raw_data_for_prediction", IMAGE_FOLDER="data/raw_data_test/images_test")
        build_text_features_func_from_mongo(for_predicting=True, list_id=filtered_ids_list, source="raw_data_for_prediction")
        
            # Texte
        print("Filtrage des features texte...")
        text_docs = db["text_features_to_predict"].find({"productid": {"$in": filtered_ids_list}}, {"_id": 0})
        text_df = pd.DataFrame(list(text_docs))

            # Images
        print("Filtrage des features image...")
        image_docs = db["image_features_to_predict"].find({"productid": {"$in": filtered_ids_list}}, {"_id": 0})
        image_df = pd.DataFrame(list(image_docs))

        if text_df.empty or image_df.empty:
            raise HTTPException(status_code=400, detail="Données incomplètes pour certaines images ou textes.")

        #Construction du data input pour prédiction
        joined_df = text_df.merge(image_df, on="productid", how="inner")
        
        #Prédiction
        preds = model.predict(joined_df)
        
        #Enregistrement des prédictions dans la base MongoDB
        now = datetime.now().isoformat()
        model_name_version = f"{model_name_loaded}_{model_version}"
        prediction_records = []
        
        for productid, pred in zip(joined_df["productid"], preds):
            match_row = filtered_df[filtered_df["productid"] == productid]
            if not match_row.empty:
                record = {
                "productid": productid,
                "designation": match_row.iloc[0]["designation"],
                "imageid": match_row.iloc[0]["imageid"],
                "prediction": int(pred),
                "date_pred": now,
                "model": model_name_version,
            }
            prediction_records.append(record)

            if prediction_records:
                db["Prediction"].insert_many(prediction_records)
        return {
            "message": f"{len(prediction_records)} prédictions faites sur {len(filtered_ids_list)} produits.",
            "nb_exclus": len(excluded_productids),
            "model": model_name_version,
            "timestamp": now
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la prédiction : {str(e)}")


@app.post("/retrain")
def retrain(user: dict = Depends(get_current_user)):
    global model
    try:
        # Réentraînement du modèle
        subprocess.run(["python", "-m", "src.models.experiment"], check=True)
        # Sélection + Promotion + récupération du modèle champion
        model_uri = select_and_promote_best_model(list_models_name=["pca_lgbm_pipeline"])
        model = mlflow.pyfunc.load_model(model_uri)

        return {"message": f"Réentraînement terminé. Modèle champion rechargé depuis {model_uri}"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))