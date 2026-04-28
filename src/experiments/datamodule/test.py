import polars as pl
import numpy as np
from src.experiments.datamodule.datasets import EmbeddingsDataset

n = 10
df = pl.DataFrame({
    "productid": range(n),
    **{f"text_feat_{i}": np.random.randn(n).tolist() for i in range(3)},
    **{f"image_feat_{i}": np.random.randn(n).tolist() for i in range(2)},
    "label": np.random.randint(0, 27, n).tolist(),
})

text_cols = [c for c in df.columns if c.startswith("text_feat_")]
image_cols = [c for c in df.columns if c.startswith("image_feat_")]

ds = EmbeddingsDataset(df, text_cols, image_cols)
print(f"Length: {len(ds)}")
text, image, label = ds[0]
print(f"Shapes: text={text.shape}, image={image.shape}, label={label}")
print(f"Types: text={text.dtype}, image={image.dtype}, label={label.dtype}")