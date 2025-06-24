import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import time
import polars as pl
import torch
from pymongo import MongoClient
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from features.utils import log_progress, extract_images_features

BATCH_ID =1  

def build_images_features_func_from_mongo(batch_id,IMAGE_FOLDER="data/raw_data/images/image_train", model=None, preprocess=None, batch_size=10000):
    # Connexion Mongo
    client = MongoClient("mongodb://localhost:27017")
    db = client["MAR25_CMLOPS_RAKUTEN"]

    # Chargement du batch depuis Mongo
    raw_docs = db["raw_data_batches"].find({"batch_id": batch_id})
    df_raw = pl.DataFrame(list(raw_docs)).select(["imageid", "productid"])
    df_raw = df_raw.sort(["imageid", "productid"])

    # Génère les chemins d'image
    df_raw = df_raw.with_columns(
        ("image_" + pl.col("imageid").cast(pl.Utf8) + "_product_" + pl.col("productid").cast(pl.Utf8) + ".jpg").alias("image_path")
    )

    # Filtrage : on retire les ID déjà présents dans image_features
    existing_ids = db["image_features"].distinct("productid")
    existing_productids = set(existing_ids)
    df_filtered = df_raw.filter(~pl.col("productid").is_in(existing_productids))

    if df_filtered.is_empty():
        print(f"Rien à faire pour batch_id={batch_id}, tous les productid sont déjà traités.")
        return

    # Modèle si non fourni
    if model is None:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Retirer FC layer
        model.eval()

    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    total_batches = (df_filtered.height + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(0, df_filtered.height, batch_size):
        sub_data = df_filtered.slice(i, batch_size)
        ids_df = sub_data.select(["imageid", "productid"])
        image_paths = sub_data["image_path"].to_list()

        # Extraction features
        all_embeddings = extract_images_features(input_dir=IMAGE_FOLDER, image_paths=image_paths, model=model, preprocess=preprocess)
        columns = [f"image_feat_{j}" for j in range(len(all_embeddings[0]))]
        features_pl = pl.DataFrame(all_embeddings, schema=columns)

        final_df = pl.concat([ids_df, features_pl], how="horizontal")
        final_df = final_df.with_columns(pl.lit(batch_id).alias("batch_id"))

        # Insertion dans Mongo
        db["image_features"].insert_many(final_df.to_dicts())

        print(f"Batch {batch_id} - {len(final_df)} images insérées dans Mongo")
        log_progress((i + batch_size) // batch_size, total_batches, start_time)

        del sub_data, ids_df, features_pl, final_df, all_embeddings
        gc.collect()

    print(f"Terminé pour batch {batch_id}.")

if __name__ == "__main__":
    build_images_features_func_from_mongo(
        batch_id=BATCH_ID 
    )
