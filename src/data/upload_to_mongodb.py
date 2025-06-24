from pymongo import MongoClient
import polars as pl
import os

client = MongoClient("mongodb://localhost:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]

def insert_file_to_collection(file_path, collection_name):
    if file_path.endswith(".csv"):
        df = pl.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        df = pl.read_parquet(file_path)
    else:
        raise ValueError("Format de fichier non supporté : doit être .csv ou .parquet")
    
    if collection_name in {"text_features", "image_features"}:
        if "imageid" in df.columns and "productid" in df.columns:
            df = df.with_columns(
                (pl.col("imageid").cast(pl.Utf8) + "_" + pl.col("productid").cast(pl.Utf8)).alias("key")
            )
    records = df.to_dicts()
    db[collection_name].insert_many(records)
    print(f"{len(records)} inserted in {collection_name}")

# Insert les raw batches
db["X_raw_data_batches"].delete_many({})
for i in range(1, 5):
    insert_file_to_collection(f"data/raw_data/batches/X_train_batch_{i}.csv", "X_raw_data_batches")

db["Y_raw_data_batches"].delete_many({})
for i in range(1, 5):
    insert_file_to_collection(f"data/raw_data/batches/Y_train_batch_{i}.csv", "Y_raw_data_batches")

db["text_features"].delete_many({})
# Insert text features
for f in sorted(os.listdir("data/preprocessed/chunked_text_files")):
    insert_file_to_collection(f"data/preprocessed/chunked_text_files/" + f, "text_features")

db["image_features"].delete_many({})
# Insert image features
for f in sorted(os.listdir("data/preprocessed/chunked_image_files")):
    insert_file_to_collection(f"data/preprocessed/chunked_image_files/" + f, "image_features")

# Insert X final
#insert_file_to_collection("data/preprocessed/final/X_train_processed_final.parquet", "X_train_final")

#  Insert Y final
#insert_file_to_collection("data/preprocessed/final/Y_train_final.parquet", "Y_train_final")