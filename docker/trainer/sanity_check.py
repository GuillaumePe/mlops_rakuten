"""Sanity check au build de l'image trainer."""
import torch

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

archs = torch.cuda.get_arch_list()
print(f"Compiled CUDA architectures: {archs}")

# get_arch_list() est vide si aucun GPU détecté (cas du build).
# Vérification indirecte : la wheel doit être cu128+ pour supporter Blackwell.
assert "+cu12" in torch.__version__ or torch.version.cuda.startswith("12.8"), (
    f"torch CUDA version doit etre cu128+ pour Blackwell sm_120, "
    f"got {torch.__version__}"
)

print("[OK] Build PyTorch CUDA 12.8 - Blackwell sm_120 supporte")

# Verifier les imports clefs
import sentence_transformers, lightgbm, polars, mlflow, dvc  # noqa: F401
import src.experiments.runner  # noqa: F401

print("[OK] src package + dependencies importable")