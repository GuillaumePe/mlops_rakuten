import os
import gc
import time
import polars as pl
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from utils import log_progress, extract_images_features

# Paramètres
INPUT_CSV = "/home/ubuntu/mar25_cmlops_rakuten/data/raw_data/X_train.csv"
LIST_ID_COLUMNS = ["imageid", "productid"]
IMAGE_FOLDER = "/home/ubuntu/mar25_cmlops_rakuten/data/raw_data/images/image_train"
OUTPUT_DIR = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/chunked_image_files"

def build_images_features_func(INPUT_CSV,LIST_ID_COLUMNS,IMAGE_FOLDER,OUTPUT_DIR):
   
    #chargement de ResNet18 pré-entrainé
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Retirer la couche Fully Connected
    model.eval()
     # Chargement des données et nettoyages
    data_text = pl.read_csv(INPUT_CSV)
    data_text = data_text.select(LIST_ID_COLUMNS).sort(LIST_ID_COLUMNS)

    data_image = data_text.with_columns(("image_"+pl.col(LIST_ID_COLUMNS[0]).cast(pl.Utf8)+"_product_"+pl.col(LIST_ID_COLUMNS[1]).cast(pl.Utf8)+".jpg").alias("image_path"))




    #preprocess des images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_size = 10000
    total_batches = (data_image.height + batch_size - 1) // batch_size
    start_time = time.time()

    for i in range(0, data_image.height, batch_size):
        
        batch_num = i // batch_size
        output_file = os.path.join(OUTPUT_DIR, f"features_image_chunk_{batch_num:04d}.parquet")

        if os.path.exists(output_file):
            continue

        print(f"[Batch {batch_num}/{total_batches}] Processing...")
        sub_data = data_image.slice(i, batch_size)
        # extrait les ids
        ids_df = sub_data.select(LIST_ID_COLUMNS)
        image_path = sub_data["image_path"].to_list()

        all_embeddings = extract_images_features(input_dir=IMAGE_FOLDER,image_paths=image_path,model=model,preprocess=preprocess)
        #Construction du DataFrame Polars 
        columns = [f"image_feat_{i}" for i in range(len(all_embeddings[0]))]
        features_pl = pl.DataFrame(all_embeddings, schema=columns)

        final_sub_df = pl.concat([ids_df, features_pl], how="horizontal")

        if not os.path.exists(OUTPUT_DIR):
            print(f"Dossier {OUTPUT_DIR} supprimé pendant l'exécution ! Je le recrée.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        final_sub_df.write_parquet(output_file)

        del sub_data, ids_df, features_pl, final_sub_df, all_embeddings
        gc.collect()

        log_progress(batch_num + 1, total_batches, start_time)

    return f"images features exctract in {OUTPUT_DIR}"

if __name__ == "__main__":
    build_images_features_func(INPUT_CSV, LIST_ID_COLUMNS, IMAGE_FOLDER, OUTPUT_DIR)
