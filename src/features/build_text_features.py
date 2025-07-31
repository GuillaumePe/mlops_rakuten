import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gc
import time
import polars as pl
from pymongo import MongoClient
from transformers import DistilBertTokenizerFast, DistilBertModel
from features.utils import extract_text_features_in_batches, clean_description, log_progress
import argparse
import numpy as np
#BATCH_ID =batch_id


def build_text_features_func_from_mongo(for_predicting=False,batch_id=None,list_id=None, source="X_raw_data_batches", batch_size=10000):
    # Connexion Mongo
    client = MongoClient("mongodb://mongodb:27017")
    db = client["MAR25_CMLOPS_RAKUTEN"]

     # Chargement du batch/liste depuis Mongo
    if batch_id is not None:
        raw_docs = db[source].find({"batch_id": batch_id})
    elif list_id is not None:
        raw_docs = db[source].find({"productid": {"$in": list_id}})
    else:
        raw_docs = db[source].find()

    df_raw = pl.DataFrame(list(raw_docs)).select(["imageid", "productid", "designation", "description"])
    df_raw = df_raw.sort(["imageid", "productid"])

    # Nettoyage et concat texte
    df_raw = df_raw.with_columns(
        pl.concat_str([pl.col("designation"), pl.col("description")], separator=". ", ignore_nulls=True).alias("full_text")
    )
    df_raw = df_raw.with_columns(
        pl.col("full_text").map_elements(lambda text: clean_description(text), return_dtype=pl.Utf8)
    )

    # Filtrage : exclude productid déjà extraits
    if for_predicting:
        existing_ids = db["text_features_to_predict"].distinct("productid")
    else:
        existing_ids = db["text_features"].distinct("productid")
    existing_productids =  set(existing_ids)
    df_filtered = df_raw.filter(~pl.col("productid").is_in(existing_productids))

    if df_filtered.is_empty():
        print(f"Rien à faire pour batch_id={batch_id}, tous les productid sont déjà traités.")
        return

    # Modèle DistilBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    total_batches = (df_filtered.height + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(0, df_filtered.height, batch_size):
        sub_data = df_filtered.slice(i, batch_size)
        ids_df = sub_data.select(["imageid", "productid"])
        texts = sub_data["full_text"].to_list()

        embeddings = extract_text_features_in_batches(texts, tokenizer=tokenizer, model=model)
        print(f"[DEBUG] Nb de textes à encoder : {len(texts)}")
        print(f"[DEBUG] Shape embeddings texte : {embeddings.shape}")

        nan_rows = [i for i in range(embeddings.shape[0]) if np.isnan(embeddings[i]).any()]
        if nan_rows:
            print(f"[ALERTE] {len(nan_rows)} embeddings texte contiennent des NaN !")

        columns = [f"text_feat_{j}" for j in range(embeddings.shape[1])]
        features_pl = pl.DataFrame(embeddings, schema=columns)

        final_df = pl.concat([ids_df, features_pl], how="horizontal")
        final_df = final_df.with_columns(pl.lit(batch_id).alias("batch_id"))
        
        # Insertion dans Mongo
        if for_predicting:
            db["text_features_to_predict"].insert_many(final_df.to_dicts())
        else:
            db["text_features"].insert_many(final_df.to_dicts())
        print(f"Batch {batch_id} - {len(final_df)} textes insérés dans Mongo")
        log_progress((i + batch_size) // batch_size, total_batches, start_time)

        del sub_data, ids_df, features_pl, final_df, embeddings
        gc.collect()

    print(f"Terminé pour batch {batch_id}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_id', type=int, default=1)
    args = parser.parse_args()

    batch_id = args.batch_id
    build_text_features_func_from_mongo(batch_id=batch_id)