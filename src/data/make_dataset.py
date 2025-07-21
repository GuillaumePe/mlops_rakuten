import polars as pl
from pymongo import MongoClient
import gc
from sklearn.model_selection import train_test_split
import argparse

LIST_ID_COLUMNS = ["imageid", "productid"]
#BATCH_ID = batch_id

def make_dataset_from_batch(batch_id):
    client = MongoClient("mongodb://mongodb:27017")
    db = client["MAR25_CMLOPS_RAKUTEN"]

    print(f"Chargement des IDs pour batch_id={batch_id}...")
    raw_docs = db["X_raw_data_batches"].find({"batch_id": batch_id}, {"_id": 0, "imageid": 1, "productid": 1})
    batch_ids = list(raw_docs)
    if not batch_ids:
        print("Aucun ID trouvé pour ce batch.")
        return

    keys = [f"{row['imageid']}_{row['productid']}" for row in batch_ids]

    # Texte
    print("Filtrage des features texte...")
    text_docs = db["text_features"].find({"key": {"$in": keys}}, {"_id": 0})
    text_df = pl.DataFrame(list(text_docs))

    # Images
    print("Filtrage des features image...")
    image_docs = db["image_features"].find({"key": {"$in": keys}}, {"_id": 0})
    image_df = pl.DataFrame(list(image_docs))

    if text_df.is_empty() or image_df.is_empty():
        print("Aucun feature à joindre pour ce batch.")
        return

    print("Jointure des features texte & image...")
    joined_df = text_df.join(image_df, on=LIST_ID_COLUMNS, how="inner")
    
    # Création du DataFrame cible
    print("Récupération des labels...")
    target_docs = db["Y_raw_data_batches"].find(
        {"batch_id": batch_id},
        {"_id": 0, "imageid": 1, "productid": 1, "prdtypecode": 1}
    )   
    target_df = pl.DataFrame(list(target_docs))

    # Jointure avec les IDs réellement présents dans X (joined_df)
    df_full = joined_df.join(target_df, on=LIST_ID_COLUMNS, how="inner")

    # Vérification
    if df_full.is_empty():
        print("Aucune ligne à insérer après jointure.")
        return

    # Split stratifié
    df_full_pd = df_full.to_pandas()
    X_df = df_full_pd.drop(columns=["prdtypecode"])
    y_series = df_full_pd["prdtypecode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, stratify=y_series, test_size=0.15, random_state=42
    )

    # Insertion dans MongoDB
    print(f"Insertion de {len(X_train)} lignes dans X_train_final")
    db["X_train_final"].insert_many(X_train.to_dict(orient="records"))
    print(f"Insertion de {len(y_train)} lignes dans Y_train_final")
    db["Y_train_final"].insert_many(
        [{**row, "prdtypecode": label} for row, label in zip(X_train[LIST_ID_COLUMNS].to_dict(orient="records"), y_train)]
    )   

    print(f"Insertion de {len(X_test)} lignes dans X_test_final")
    db["X_test_final"].insert_many(X_test.to_dict(orient="records"))
    print(f"Insertion de {len(y_test)} lignes dans Y_test_final")
    db["Y_test_final"].insert_many(
        [{**row, "prdtypecode": label} for row, label in zip(X_test[LIST_ID_COLUMNS].to_dict(orient="records"), y_test)]
    )

    del text_df, image_df, joined_df
    gc.collect()
    print("Terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_id', type=int, default=1)
    args = parser.parse_args()

    batch_id = args.batch_id
    make_dataset_from_batch(batch_id=batch_id)