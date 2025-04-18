import polars as pl
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
import gc
from utils import extract_features_in_batches, clean_description, log_progress
import time
import os

# Paramètres
INPUT_CSV = "/home/ubuntu/mar25_cmlops_rakuten/data/raw_data/X_train.csv"
LIST_TEXT_COLUMN = ["designation","description"]
LIST_ID_COLUMNs = ["imageid","productid"]
OUTPUT_DIR = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/chunked_text_files "

# Chargement des données et nettoyages
data = pl.read_csv(INPUT_CSV)
data = data.with_columns(pl.concat_str([pl.col(LIST_TEXT_COLUMN[0]),pl.col(LIST_TEXT_COLUMN[1])],separator=". ",ignore_nulls=True).alias('_'.join(LIST_TEXT_COLUMN)))
data = data.with_columns(pl.col('_'.join(LIST_TEXT_COLUMN)).map_elements(lambda text:clean_description(text), return_dtype=pl.Utf8)).drop(LIST_TEXT_COLUMN)
data = data.sort(LIST_ID_COLUMNs)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

start = time.time()
batch_size = 10000

for i in range(0, data.height, batch_size):
    sub_data = data.slice(i, batch_size)
    # extrait les ids
    ids_df = sub_data.select(LIST_ID_COLUMNs)
    # extrait les textes
    texts = sub_data.select('_'.join(LIST_TEXT_COLUMN)).to_series().to_list()

    # extraction des embeddings pour les textes
    embeddings = extract_features_in_batches(texts,tokenizer=tokenizer,model=model)

    # Converti les embeddings en DataFrame Polars
    columns = [f"text_feat_{i}" for i in range(embeddings.shape[1])]
    features_pl = pl.DataFrame(embeddings, schema=columns)

    # Ajoute les colonnes productid et imageid 
    final_sub_df = pl.concat([ids_df, features_pl],how="horizontal")
    output_file = os.path.join(OUTPUT_DIR, f"features_text_chunk_{i//batch_size:04d}.parquet")
    final_sub_df.write_parquet(output_file) 
    del sub_data, texts, embeddings, features_pl, ids_df, final_sub_df 
    gc.collect()
    log_progress(
            current=(i + batch_size) // batch_size,
            total=(len(texts) + batch_size - 1) // batch_size,
            start_time=start
        )

