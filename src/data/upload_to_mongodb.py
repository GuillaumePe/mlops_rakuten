"""
Upload Rakuten data to MongoDB Atlas.

Architecture des batches (simulation MLOps cycle de vie) :
- batch_id=1 : 50% du train (avec stratification sur les 27 classes)
- batch_id=2 : 25% du train (stratifié)
- batch_id=3 : 25% du train (stratifié)

Le DAG Training filtre par batch_id <= current_batch_id et déclenche un
retraining incrémental.

Collections produites :
- X_raw_data_batches : tous les samples train avec leur batch_id
- Y_raw_data_batches : labels train (mêmes productids)
- X_to_predict_pool : tous les samples test (réserve immuable pour simulation)
"""
from __future__ import annotations
import os
import sys

import numpy as np
import polars as pl
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

print("Script upload_to_mongodb.py lancé")
print(f"Python {sys.version}")

load_dotenv()

# === Connexion MongoDB Atlas ===
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI manquante dans .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# === Chemins locaux des CSV (récupérés via dvc pull) ===
X_TRAIN_CSV = "data/raw_data/X_train_update.csv"
Y_TRAIN_CSV = "data/raw_data/Y_train_update.csv"
X_TEST_CSV = "data/raw_data_test/X_test_update.csv"

# === Stratégie batches train ===
BATCH_FRACTIONS = [0.50, 0.25, 0.25]  # batch 1, 2, 3
RANDOM_STATE = 42


def assign_batch_ids(labels: np.ndarray, fractions: list[float], seed: int) -> np.ndarray:
    """
    Assigne un batch_id à chaque sample, avec stratification sur les classes.
    
    On fait un split stratifié récursif : d'abord on isole batch 1, puis on
    split le reste pour batch 2 et 3.
    
    Args:
        labels: array des labels (prdtypecode)
        fractions: liste des fractions par batch (somme = 1)
        seed: random state
    
    Returns:
        array de batch_id (1-indexé) de même longueur que labels
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Sum des fractions doit être 1.0, got {sum(fractions)}")
    
    n = len(labels)
    indices = np.arange(n)
    batch_ids = np.zeros(n, dtype=np.int32)
    
    remaining_idx = indices.copy()
    remaining_labels = labels.copy()
    cumulative = 0.0
    
    for batch_num, frac in enumerate(fractions, start=1):
        cumulative += frac
        if batch_num < len(fractions):
            # Fraction du remaining qui va dans ce batch
            # On veut que (batch_size / total_initial) = frac
            # donc batch_size_in_remaining = frac * n
            # et test_size_relative = batch_size_in_remaining / len(remaining)
            target_size = int(round(frac * n))
            test_size = target_size / len(remaining_idx)
            
            this_batch_idx, remaining_idx_new = train_test_split(
                remaining_idx,
                train_size=test_size,
                stratify=remaining_labels,
                random_state=seed + batch_num,
            )
            batch_ids[this_batch_idx] = batch_num
            # Update remaining
            mask = np.isin(remaining_idx, remaining_idx_new)
            remaining_idx = remaining_idx[mask]
            remaining_labels = remaining_labels[mask]
        else:
            # Dernier batch : tout le reste
            batch_ids[remaining_idx] = batch_num
    
    return batch_ids

def insert_in_chunks(collection, docs, chunk_size=5000):
        """Insert en chunks pour éviter les timeouts Atlas free tier."""
        n = len(docs)
        for i in range(0, n, chunk_size):
            chunk = docs[i:i + chunk_size]
            collection.insert_many(chunk, ordered=False)
            print(f"  inserted {min(i + chunk_size, n)}/{n}")

def upload_train_data():
    """Charge X_train et Y_train, assigne les batch_ids, upload."""
    print(f"\n[Upload train] Lecture des CSV...")
    df_x = pl.read_csv(X_TRAIN_CSV)
    df_y = pl.read_csv(Y_TRAIN_CSV)
    
    # Drop la colonne d'index sans nom
    df_x = df_x.drop("")
    df_y = df_y.drop("")
    
    # X et Y se correspondent par position : on ajoute productid à Y depuis X
    if len(df_x) != len(df_y):
        raise RuntimeError(
            f"X_train ({len(df_x)}) et Y_train ({len(df_y)}) ont des tailles différentes !"
        )
    
    df_y = df_y.with_columns(productid=df_x.get_column("productid"))
    
    # Maintenant on peut joindre
    df_xy = df_x.join(df_y, on="productid", how="inner")
    print(f"[Upload train] {len(df_xy)} samples avec label après jointure")
    
    labels = df_xy.get_column("prdtypecode").to_numpy()
    batch_ids = assign_batch_ids(labels, BATCH_FRACTIONS, RANDOM_STATE)
    
    # Distribution par batch
    print(f"[Upload train] Distribution batches :")
    for b in [1, 2, 3]:
        count = int((batch_ids == b).sum())
        print(f"  batch_id={b} : {count} samples ({100*count/len(labels):.1f}%)")
    
    # Vérification stratification : combien de classes uniques par batch ?
    print(f"[Upload train] Classes uniques par batch :")
    for b in [1, 2, 3]:
        n_classes = len(np.unique(labels[batch_ids == b]))
        print(f"  batch_id={b} : {n_classes} classes")
    
    # X_raw_data_batches : X columns + batch_id (label séparé dans Y)
    df_x_with_batch = df_x.join(
        pl.DataFrame({
            "productid": df_xy.get_column("productid"),
            "batch_id": batch_ids.tolist(),
        }),
        on="productid",
        how="inner",
    )
    
    print(f"[Upload train] Insertion X_raw_data_batches...")
    db["X_raw_data_batches"].delete_many({})
    insert_in_chunks(db["X_raw_data_batches"], df_x_with_batch.to_dicts())
    print(f"  → {db['X_raw_data_batches'].count_documents({})} docs")

    print(f"[Upload train] Insertion Y_raw_data_batches...")
    db["Y_raw_data_batches"].delete_many({})
    insert_in_chunks(db["Y_raw_data_batches"], df_y.to_dicts())
    print(f"  → {db['Y_raw_data_batches'].count_documents({})} docs")


def upload_test_pool():
    """Charge X_test, filtre par images disponibles, upload dans X_to_predict_pool."""
    print(f"\n[Upload test pool] Lecture du CSV...")
    df = pl.read_csv(X_TEST_CSV)
    df = df.drop("") 
    # Filtrer par images disponibles (logique reprise de l'ancien upload)
    image_dir = "data/raw_data_test/images_test"
    if os.path.isdir(image_dir):
        image_filenames = os.listdir(image_dir)
        available_productids = {
            filename.split("_product_")[1].split(".")[0]
            for filename in image_filenames
            if "_product_" in filename
        }
        df = df.filter(pl.col("productid").cast(pl.Utf8).is_in(available_productids))
        print(f"[Upload test pool] {len(df)} samples avec image disponible")
    else:
        print(f"[Upload test pool] WARN : {image_dir} introuvable, pas de filtrage")
    
    print(f"[Upload test pool] Insertion X_to_predict_pool...")
    db["X_to_predict_pool"].delete_many({})
    insert_in_chunks(db["X_to_predict_pool"], df.to_dicts())
    print(f"  → {db['X_to_predict_pool'].count_documents({})} docs")


if __name__ == "__main__":
    upload_train_data()
    upload_test_pool()
    print("\n✓ Upload terminé.")