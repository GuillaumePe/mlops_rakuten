from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import subprocess
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import dagshub
from mlflow.tracking import MlflowClient

# Configuration
SECRET_KEY = "rakuten_secret_key"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
repo_owner = 'GuillaumePe'
repo_name = 'mar25_cmlops_rakuten'
#User DB
users_db = {
    "admin": {
        "username": "123admin",
        "password": "123",  
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
MODEL_NAME = "pca_lgbm_pipeline"
CHAMPION_URI = f"models:/{MODEL_NAME}@champion"

@app.get("/")
def read_root():
    return {"message": "API modèle ML Rakuten online"}

# Initialisation Dagshub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

model = None

try:
    client = MlflowClient()
    all_models = client.search_registered_models()

    champion_model_uri = None
    for models in all_models:
        for v in models.latest_versions:
            if "champion" in v.aliases:
                champion_model_uri = f"models:/{models.name}@champion"
                break
        if champion_model_uri:
            break

    if champion_model_uri is None:
        raise ValueError("Aucun modèle avec l'alias 'champion' trouvé")

    model = mlflow.pyfunc.load_model(champion_model_uri)
    print(f"Modèle chargé depuis {champion_model_uri}")
except Exception as e:
    print(f"Erreur chargement modèle champion : {e}")

# Class pour le schéma
class PredictRequest(BaseModel):
    data: list[dict]

# création des endpoints

@app.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Identifiants invalides")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/predict")
def predict(request: PredictRequest, user: dict = Depends(get_current_user)):
    global model
    try:
        if model is None:
            raise ValueError("Modèle champion non chargé.")
        df = pd.DataFrame(request.data)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain(user: dict = Depends(get_current_user)):
    global model
    try:
        subprocess.run(["python", "src/models/experiment.py"], check=True)
        subprocess.run(["python", "src/models/model_selection.py"], check=True)
        model = mlflow.pyfunc.load_model(CHAMPION_URI)
        return {"message": "Réentraînement terminé et modèle rechargé"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erreur pipeline : {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


