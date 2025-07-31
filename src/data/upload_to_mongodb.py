print("Script upload_to_mongodb.py lancé")
import sys
print(sys.version)
from pymongo import MongoClient
import polars as pl
import os
client = MongoClient("mongodb://mongodb:27017")
db = client["MAR25_CMLOPS_RAKUTEN"]

def insert_file_to_collection(file_path, collection_name):
    if collection_name == "X_to_predict":
        df = pl.read_csv(file_path)

        # Extraire les productid des fichiers images disponibles
        image_dir = "data/raw_data_test/images_test"
        image_filenames = os.listdir(image_dir)
        available_productids = {
            filename.split("_product_")[1].split(".")[0]
            for filename in image_filenames
            if "_product_" in filename
        }

        # Filtrer les lignes du CSV selon les productid présents dans les images
        df = df.filter(pl.col("productid").cast(pl.Utf8).is_in(available_productids))

    else:
        if file_path.endswith(".csv"):
            df = pl.read_csv(file_path)
        elif file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        else:
            raise ValueError("Format de fichier non supporté : doit être .csv ou .parquet")

    # Ajouter une clé si nécessaire
    if collection_name in {"text_features", "image_features"}:
        if "imageid" in df.columns and "productid" in df.columns:
            df = df.with_columns(
                (pl.col("imageid").cast(pl.Utf8) + "_" + pl.col("productid").cast(pl.Utf8)).alias("key")
            )

    # Insertion dans MongoDB
    records = df.to_dicts()
    db[collection_name].insert_many(records)
    print(f"{len(records)} inserted in {collection_name}")

# Insert les raw batches
db["X_raw_data_batches"].delete_many({})
for i in range(1, 5):
    insert_file_to_collection(f"data/raw_data/batches/X_train_batch_{i}.csv", "X_raw_data_batches")
print(f"{db['X_raw_data_batches'].count_documents({})} documents dans X_raw_data_batches")

db["Y_raw_data_batches"].delete_many({})
for i in range(1, 5):
    insert_file_to_collection(f"data/raw_data/batches/Y_train_batch_{i}.csv", "Y_raw_data_batches")
print(f"{db['Y_raw_data_batches'].count_documents({})} documents dans Y_raw_data_batches")

db["text_features"].delete_many({})
# Insert text features
for f in sorted(os.listdir("data/preprocessed/chunked_text_files")):
    insert_file_to_collection(f"data/preprocessed/chunked_text_files/" + f, "text_features")
print(f"{db['text_features'].count_documents({})} documents dans text_features")

db["image_features"].delete_many({})
# Insert image features
for f in sorted(os.listdir("data/preprocessed/chunked_image_files")):
    insert_file_to_collection(f"data/preprocessed/chunked_image_files/" + f, "image_features")
print(f"{db['image_features'].count_documents({})} documents dans image_features")

db["X_to_predict"].delete_many({})
insert_file_to_collection("data/raw_data_test/X_test_update.csv", "X_to_predict")
print(f"{db['X_to_predict'].count_documents({})} documents dans X_to_predict")


# Insert X final
#insert_file_to_collection("data/preprocessed/final/X_train_processed_final.parquet", "X_train_final")

#  Insert Y final
#insert_file_to_collection("data/preprocessed/final/Y_train_final.parquet", "Y_train_final")