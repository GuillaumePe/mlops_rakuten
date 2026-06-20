"""Sanity check P.2 — predict_pending end-to-end."""
import polars as pl
from pymongo import MongoClient

# 1. Injecter 5 samples dans X_to_predict depuis le CSV train
print("=== Injection de 5 samples dans X_to_predict ===")
x_df = pl.read_csv("data/raw_data/X_train_update.csv", n_rows=5)

client = MongoClient("mongodb://localhost:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]

# Nettoyer la queue avant le test
db["X_to_predict"].delete_many({})

docs = []
for row in x_df.iter_rows(named=True):
    docs.append({
        "productid": row["productid"],
        "designation": row["designation"],
        "description": row.get("description"),
        "imageid": row["imageid"],
    })
db["X_to_predict"].insert_many(docs)
print(f"  X_to_predict: {db['X_to_predict'].count_documents({})} docs")

# 2. Monkey-patch le dossier image (on utilise image_train, pas images_test)
import src.models.predict_pending as pp
from pathlib import Path
pp.IMAGE_FOLDER_TEST = Path("data/raw_data/images/image_train")
print(f"  IMAGE_FOLDER override: {pp.IMAGE_FOLDER_TEST}")

# 3. Lancer predict_pending avec threshold=3
print("\n=== Lancement predict_pending ===")
result = pp.run_predict_pending(threshold=3, model_name="rakuten-m2-best")
print(f"\nRésultat: {result}")

# 4. Vérifier les résultats
print("\n=== Vérification ===")
n_predictions = db["Prediction"].count_documents({"model": {"$regex": "rakuten-m2-best"}})
print(f"Prediction docs (rakuten-m2-best): {n_predictions}")

n_queue = db["X_to_predict"].count_documents({})
print(f"X_to_predict restants: {n_queue}")

# Afficher un record
sample = db["Prediction"].find_one({"model": {"$regex": "rakuten-m2-best"}}, {"_id": 0})
if sample:
    print(f"Sample prediction: {sample}")

assert result["scored"] == 5, f"Expected 5 scored, got {result['scored']}"
assert n_queue == 0, f"Expected 0 in queue, got {n_queue}"
print("\n✅ Sanity check P.2 passed")
