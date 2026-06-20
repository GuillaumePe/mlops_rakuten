"""Sanity check P.1 — scoring réel via RakutenScorer (M2 path)."""
import polars as pl
from src.models.rakuten_scorer import RakutenScorer

# 1. Scorer
scorer = RakutenScorer.from_champion("rakuten-m2-best")
print(scorer)

# 2. Charger 3 vrais samples depuis les CSV bruts
x_df = pl.read_csv("data/raw_data/X_train_update.csv", n_rows=3)
print(f"Colonnes CSV: {x_df.columns}")
print(x_df.head())

# Construire le raw_df attendu par le scorer
raw = x_df.select([
    "productid",
    "designation",
    "description",
    "imageid",
]).with_columns(
    pl.concat_str([
        pl.lit("data/raw_data/images/image_train/image_"),
        pl.col("imageid").cast(pl.Utf8),
        pl.lit("_product_"),
        pl.col("productid").cast(pl.Utf8),
        pl.lit(".jpg"),
    ]).alias("image_path")
)
print(f"\nInput shape: {raw.shape}")
print(raw.select(["productid", "designation"]))

# 3. Score
result = scorer.score(raw)
print(f"\nPredictions: {result.predictions}")
print(f"Probas shape: {result.probas.shape}")
print(f"Probas sum per sample: {result.probas.sum(axis=1)}")  # doit être ~1.0
print(f"Model: {result.model_name} v{result.model_version} ({result.model_family})")
print(f"N scored: {result.n_scored}")
print("\n✅ Sanity check P.1 passed")