import polars as pl
import os
import re
import gc

# Paramètres
TEXT_FOLDER = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/chunked_text_files"
IMAGE_FOLDER = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/chunked_image_files"
OUTPUT_DIR = "/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final"
LIST_ID_COLUMNS = ["imageid", "productid"]

list_text_features_files = sorted(os.listdir(TEXT_FOLDER))
list_image_features_files = sorted(os.listdir(IMAGE_FOLDER))

num_file_text_init = str(re.search(r'_(\d+)\.parquet$', list_text_features_files[0]).group(1))
num_file_image_init = str(re.search(r'_(\d+)\.parquet$', list_image_features_files[0]).group(1))

print(num_file_text_init)
print(num_file_image_init)
if num_file_text_init != num_file_image_init:
    raise ValueError("les fichiers parquet text features et images features ne correspondent pas")

x_train_text_features_init = pl.read_parquet(os.path.join(TEXT_FOLDER,list_text_features_files[0]))
x_train_image_features_init = pl.read_parquet(os.path.join(IMAGE_FOLDER,list_image_features_files[0]))

X_train_processed_stack = x_train_text_features_init.join(
    x_train_image_features_init,
    on=LIST_ID_COLUMNS,
    how="inner"
    )

X_train_processed_stack.write_parquet(os.path.join(OUTPUT_DIR, f"X_train_processed_chunk_{num_file_text_init}.parquet"))

for text_features_file, image_features_file in zip(list_text_features_files[1:],list_image_features_files[1:]):
    
    num_file_text = str(re.search(r'_(\d+)\.parquet$', text_features_file).group(1))
    num_file_image = str(re.search(r'_(\d+)\.parquet$', image_features_file).group(1))
    if num_file_text != num_file_image:
        raise ValueError("les fichiers parquet text features et images features ne correspondent pas")

    x_train_text_features_chunk = pl.read_parquet(os.path.join(TEXT_FOLDER,text_features_file))
    x_train_image_features_chunk = pl.read_parquet(os.path.join(IMAGE_FOLDER,image_features_file))
    X_train_processed_chunk = x_train_text_features_chunk.join(
        x_train_image_features_chunk,
    on=LIST_ID_COLUMNS,
    how="inner"
    )
    X_train_processed_chunk.write_parquet(os.path.join(OUTPUT_DIR, f"X_train_processed_chunk_{num_file_text}.parquet"))

    X_train_processed_stack = pl.concat([X_train_processed_stack,X_train_processed_chunk], how="vertical")
    del num_file_text, x_train_text_features_chunk, x_train_image_features_chunk, X_train_processed_chunk
    gc.collect()
    
X_train_processed_stack.write_parquet(os.path.join(OUTPUT_DIR, f"X_train_processed_final.parquet"))




