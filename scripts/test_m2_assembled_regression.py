"""
Test de non-régression — vérifie toute la chaîne M2Assembled sans lancer de training.

Usage :
    MLFLOW_TRACKING_URI=http://localhost:5000 ACTIVE_VAL_SELECTION_VERSION=1 \
    python tests/test_m2_assembled_regression.py

Vérifie :
    1. Chargement des configs YAML (m2_benchmark, m2_frugal_ft, m2_best)
    2. EXPERIMENT_BUILDERS contient les bonnes clés
    3. Builders s'instancient sans crash
    4. Résolution dynamique @active_text / @active_image (m2_best)
    5. LEARNER_EMBED_DIM cohérent
    6. M2Assembled s'instancie et valide les colonnes
    7. DataModule accepte extra_embedding_caches
    8. Val_selection fallback (colonne absente → pas de crash)
    9. Bloc 7 try/except présent dans base_learner_experiment
   10. Push R2 boto3 présent (pas dvc add)
"""
import os
import sys
import traceback

# Forcer les env vars de test
os.environ.setdefault("ACTIVE_VAL_SELECTION_VERSION", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

PASS = 0
FAIL = 0
SKIP = 0

def test(name):
    """Décorateur de test."""
    def decorator(fn):
        global PASS, FAIL, SKIP
        try:
            fn()
            print(f"  ✓ {name}")
            PASS += 1
        except AssertionError as e:
            print(f"  ✗ {name} — ASSERTION: {e}")
            FAIL += 1
        except Exception as e:
            print(f"  ✗ {name} — {type(e).__name__}: {e}")
            FAIL += 1
        return fn
    return decorator


print("=" * 60)
print("TEST DE NON-RÉGRESSION — M2Assembled pipeline")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
print("\n[1] Chargement des configs YAML")
# ─────────────────────────────────────────────────────────────

@test("m2_benchmark.yaml charge OK")
def _():
    from src.experiments.runner import load_config
    cfg = load_config("m2_benchmark")
    assert "base_learners" in cfg, "base_learners manquant"
    assert cfg["base_learners"]["text"]["name"] == "textcnn"
    assert cfg["base_learners"]["text"]["embed_dim"] == 3072
    assert cfg["base_learners"]["image"]["name"] == "resnet50_partial_ft"
    assert cfg["base_learners"]["image"]["embed_dim"] == 2048
    assert "extra_embedding_caches" in cfg["datamodule"]

@test("m2_frugal_ft.yaml charge OK")
def _():
    from src.experiments.runner import load_config
    cfg = load_config("m2_frugal_ft")
    assert cfg["base_learners"]["text"]["name"] == "camembert_lora"
    assert cfg["base_learners"]["image"]["name"] == "resnet18_full_ft"

@test("m2_best.yaml charge OK (pas de base_learners)")
def _():
    from src.experiments.runner import load_config
    cfg = load_config("m2_best")
    assert "base_learners" not in cfg, "m2_best ne devrait PAS avoir base_learners"
    assert "model" in cfg
    assert "mlflow" in cfg

# ─────────────────────────────────────────────────────────────
print("\n[2] EXPERIMENT_BUILDERS")
# ─────────────────────────────────────────────────────────────

@test("EXPERIMENT_BUILDERS contient m2_benchmark, m2_frugal_ft, m2_best")
def _():
    from src.experiments.runner import EXPERIMENT_BUILDERS
    for key in ["m2_benchmark", "m2_frugal_ft", "m2_best"]:
        assert key in EXPERIMENT_BUILDERS, f"{key} manquant dans EXPERIMENT_BUILDERS"

@test("m2_benchmark et m2_best utilisent le même builder")
def _():
    from src.experiments.runner import EXPERIMENT_BUILDERS
    assert EXPERIMENT_BUILDERS["m2_benchmark"] is EXPERIMENT_BUILDERS["m2_best"]

# ─────────────────────────────────────────────────────────────
print("\n[3] LEARNER_EMBED_DIM")
# ─────────────────────────────────────────────────────────────

@test("LEARNER_EMBED_DIM existe et contient les 6 learners")
def _():
    from src.experiments.runner import LEARNER_EMBED_DIM
    expected = {
        "textcnn": 3072,
        "camembert_lora": 768,
        "camembert_frozen": 768,
        "resnet50_partial_ft": 2048,
        "resnet18_full_ft": 512,
        "resnet18_frozen": 512,
    }
    for name, dim in expected.items():
        assert name in LEARNER_EMBED_DIM, f"{name} manquant"
        assert LEARNER_EMBED_DIM[name] == dim, f"{name}: attendu {dim}, trouvé {LEARNER_EMBED_DIM[name]}"

# ─────────────────────────────────────────────────────────────
print("\n[4] Résolution dynamique @active_text / @active_image")
# ─────────────────────────────────────────────────────────────

@test("resolve_active_base_learners retourne text + image + extra_caches")
def _():
    from src.experiments.runner import resolve_active_base_learners
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    result = resolve_active_base_learners(tracking_uri)
    assert "text" in result, "text manquant"
    assert "image" in result, "image manquant"
    assert "extra_caches" in result, "extra_caches manquant"
    assert "name" in result["text"]
    assert "embed_dim" in result["text"]
    assert "version" in result["text"]
    assert result["text"]["name"] in ("textcnn", "camembert_lora", "camembert_frozen")
    assert result["image"]["name"] in ("resnet50_partial_ft", "resnet18_full_ft", "resnet18_frozen")
    assert len(result["extra_caches"]) == 2
    print(f"      → text={result['text']['name']} v{result['text']['version']}, "
          f"image={result['image']['name']} v{result['image']['version']}")

# ─────────────────────────────────────────────────────────────
print("\n[5] M2Assembled instanciation")
# ─────────────────────────────────────────────────────────────

@test("M2Assembled s'instancie avec textcnn + resnet50")
def _():
    from src.models.assembled.m2_assembled import M2Assembled
    model = M2Assembled(
        tabular_cols=["tab_0", "tab_1"],
        text_learner_name="textcnn",
        text_embed_dim=3072,
        image_learner_name="resnet50_partial_ft",
        image_embed_dim=2048,
    )
    assert len(model.text_cols) == 3072
    assert len(model.image_cols) == 2048
    assert model.text_cols[0] == "textcnn_feat_0"
    assert model.image_cols[0] == "resnet50_partial_ft_feat_0"

@test("M2Assembled s'instancie avec camembert_lora + resnet50 (m2_best)")
def _():
    from src.models.assembled.m2_assembled import M2Assembled
    model = M2Assembled(
        tabular_cols=["tab_0"],
        text_learner_name="camembert_lora",
        text_embed_dim=768,
        image_learner_name="resnet50_partial_ft",
        image_embed_dim=2048,
    )
    assert model.text_cols[0] == "camembert_lora_feat_0"
    assert model.text_cols[-1] == "camembert_lora_feat_767"
    assert model.metadata["base_text"] == "camembert_lora"

@test("M2Assembled legacy frozen (drop-in M2Baseline)")
def _():
    from src.models.assembled.m2_assembled import M2Assembled
    model = M2Assembled(
        tabular_cols=["tab_0"],
        text_learner_name="camembert_frozen",
        text_embed_dim=768,
        image_learner_name="resnet18_frozen",
        image_embed_dim=512,
    )
    assert model.text_cols[0] == "text_feat_0", f"Attendu text_feat_0, trouvé {model.text_cols[0]}"
    assert model.image_cols[0] == "image_feat_0"

# ─────────────────────────────────────────────────────────────
print("\n[6] DataModule extra_embedding_caches")
# ─────────────────────────────────────────────────────────────

@test("DataModule accepte extra_embedding_caches dans __init__")
def _():
    from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
    dm = RakutenLightningDataModule(
        extra_embedding_caches=["embeddings_textcnn_v1.parquet"],
    )
    assert dm.extra_embedding_caches == ["embeddings_textcnn_v1.parquet"]
    assert dm._extra_cols == []

@test("DataModule sans extra_embedding_caches → liste vide")
def _():
    from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
    dm = RakutenLightningDataModule()
    assert dm.extra_embedding_caches == []

# ─────────────────────────────────────────────────────────────
print("\n[7] Val_selection fallback (colonne absente)")
# ─────────────────────────────────────────────────────────────

@test("Val_selection fallback : code présent dans setup()")
def _():
    import inspect
    from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
    source = inspect.getsource(RakutenLightningDataModule.setup)
    # Vérifier que le raise RuntimeError a été remplacé par un fallback
    assert "pl.lit(False)" in source or "lit(False)" in source, \
        "Le fallback val_selection (pl.lit(False)) n'est pas dans setup()"

# ─────────────────────────────────────────────────────────────
print("\n[8] Base learner experiment — bloc 7 robuste")
# ─────────────────────────────────────────────────────────────

@test("Bloc 7 a try/except + promotion_success")
def _():
    import inspect
    from src.experiments.strategies.base_learner_experiment import BaseLearnerExperiment
    source = inspect.getsource(BaseLearnerExperiment.fit)
    assert "promotion_success" in source, "promotion_success manquant dans fit()"
    assert "promotion_error" in source, "promotion_error manquant dans fit()"
    assert "except Exception" in source, "try/except manquant dans bloc 7"

@test("_write_cache_parquet utilise boto3, pas dvc add")
def _():
    import inspect
    from src.experiments.strategies.base_learner_experiment import BaseLearnerExperiment
    source = inspect.getsource(BaseLearnerExperiment._write_cache_parquet)
    assert "boto3" in source, "boto3 manquant dans _write_cache_parquet"
    assert 'dvc", "add"' not in source and "dvc add" not in source.replace("# ", "").lower()[:50], \
        "dvc add encore présent dans _write_cache_parquet"

@test("_write_cache_parquet utilise get_full_data (pas train_pool)")
def _():
    import inspect
    from src.experiments.strategies.base_learner_experiment import BaseLearnerExperiment
    source = inspect.getsource(BaseLearnerExperiment._write_cache_parquet)
    assert "get_full_data" in source, "get_full_data manquant"
    assert 'get_sklearn_data("train_pool"' not in source, "Encore get_sklearn_data(train_pool)"

# ─────────────────────────────────────────────────────────────
print("\n[9] Builder m2_benchmark s'instancie (dry run)")
# ─────────────────────────────────────────────────────────────

@test("build_m2_assembled_experiment avec m2_benchmark config → dm + experiment")
def _():
    from src.experiments.runner import load_config, build_m2_assembled_experiment
    cfg = load_config("m2_benchmark")
    dm, experiment = build_m2_assembled_experiment(cfg)
    assert dm is not None
    assert experiment is not None
    assert dm.extra_embedding_caches == [
        "embeddings_textcnn_v1.parquet",
        "embeddings_resnet50_partial_ft_v1.parquet",
    ]

@test("build_m2_assembled_experiment avec m2_best config → résolution dynamique")
def _():
    from src.experiments.runner import load_config, build_m2_assembled_experiment
    cfg = load_config("m2_best")
    dm, experiment = build_m2_assembled_experiment(cfg)
    assert dm is not None
    assert len(dm.extra_embedding_caches) == 2
    # Les caches doivent correspondre aux @active_text et @active_image
    cache_names = dm.extra_embedding_caches
    assert any("camembert_lora" in c or "textcnn" in c for c in cache_names), \
        f"Pas de cache text dans {cache_names}"
    assert any("resnet50" in c or "resnet18" in c for c in cache_names), \
        f"Pas de cache image dans {cache_names}"

# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RÉSULTATS : {PASS} ✓  {FAIL} ✗  {SKIP} skip")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
else:
    print("\nTous les tests passent. Prêt pour le rebuild + lancement.")
    sys.exit(0)
