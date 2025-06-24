import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os

# Paramètres
X_PATH = "data/raw_data/X_train.csv"
Y_PATH = "data/raw_data/Y_train.csv"
OUTPUT_DIR = "data/raw_data/batches"
NUM_SPLITS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chargement des données
df_X = pl.read_csv(X_PATH)
df_Y = pl.read_csv(Y_PATH)

# Fusion des labels sur l'index
if df_X.height != df_Y.height:
    raise ValueError("X_train et Y_train n'ont pas le même nombre de lignes.")

# Fusionner temporairement pour garder alignement
df_merged = df_X.with_columns(df_Y["prdtypecode"])

# Labels pour stratification
labels = df_merged["prdtypecode"].to_numpy()
skf = StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True, random_state=42)

for i, (_, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
    df_X_batch = df_merged[test_index].drop("prdtypecode")
    df_Y_batch = df_merged[test_index].select(["imageid", "productid", "prdtypecode"])

    df_X_batch = df_X_batch.with_columns(pl.lit(i).alias("batch_id"))
    df_Y_batch = df_Y_batch.with_columns(pl.lit(i).alias("batch_id"))

    df_X_batch.write_csv(os.path.join(OUTPUT_DIR, f"X_train_batch_{i}.csv"))
    df_Y_batch.write_csv(os.path.join(OUTPUT_DIR, f"Y_train_batch_{i}.csv"))

    print(f"Batch {i} : {df_X_batch.shape[0]} lignes X et {df_Y_batch.shape[0]} lignes Y")