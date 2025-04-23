import polars as pl


LIST_ID_COLUMNS = ["imageid", "productid"]
LIST_ID_COLUMNS.append("prdtypecode")
X_train = pl.read_csv("/home/ubuntu/mar25_cmlops_rakuten/data/raw_data/X_train.csv")
Y_train = pl.read_csv("/home/ubuntu/mar25_cmlops_rakuten/data/raw_data/Y_train.csv").select("prdtypecode")

Y_train_new = pl.concat([X_train,Y_train],how="horizontal")

Y_train_new = Y_train_new.select(LIST_ID_COLUMNS)
Y_train_new.write_parquet("/home/ubuntu/mar25_cmlops_rakuten/data/preprocessed/final/Y_train_final.parquet")