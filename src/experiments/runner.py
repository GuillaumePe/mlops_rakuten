"""
Runner CLI pour les expériences M2/M3/M4.

Usage:
    python -m src.experiments.runner --experiment m2 --action fit
    python -m src.experiments.runner --experiment m2 --action evaluate
    python -m src.experiments.runner --experiment m2 --action prepare_data

Le runner :
1. Charge la config YAML de l'expérience
2. Instancie le DataModule selon le mode
3. Instancie l'Experiment selon la stratégie (sklearn / lightning)
4. Exécute l'action demandée
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # charge .env automatiquement

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import yaml

from src.experiments.datamodule.rakuten_datamodule import RakutenLightningDataModule
from src.experiments.models.m2.m2 import M2Stacking
from src.experiments.strategies.sklearn_experiment import SklearnExperiment
from src.experiments.strategies.base_learner_experiment import BaseLearnerExperiment
from src.models.assembled.m2_baseline import M2Baseline
from src.models.assembled.m2_assembled import M2Assembled
import os
from src.experiments.strategies.lightning_experiment import LightningExperiment
from src.models.assembled.m3_attention_fusion import M3AttentionFusion
from src.experiments.datamodule.datasets import MultimodalDataset
from src.models.base_learners._pyfunc_wrapper import BaseLearnerPyfunc
from torch.utils.data import DataLoader
from src.models.assembled.m3_2_coadaptation import M32CoAdaptationFusion
from src.experiments.strategies.hpo_lightning_experiment import HPOLightningExperiment

# Registre des dimensions d'embeddings par base learner.
# Utilisé par build_m2_best_experiment pour résoudre embed_dim
# à partir du learner_name trouvé via @active_text/@active_image.
LEARNER_EMBED_DIM = {
    "textcnn": 3072,
    "camembert_lora": 768,
    "camembert_frozen": 768,
    "resnet50_partial_ft": 2048,
    "resnet18_full_ft": 512,
    "resnet18_frozen": 512,
    "siglip2": 768,    
}


CONFIG_DIR = Path("src/experiments/config")


def load_config(experiment_name: str) -> dict:
    """Charge la config YAML correspondant à l'expérience."""
    config_path = CONFIG_DIR / f"{experiment_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config introuvable : {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Applique des overrides 'a.b.c=value' au dict config (in-place + retour).

    Typage : yaml.safe_load ('8'->int, 'true'->bool, '[1,2]'->list, ...), AVEC
    rattrapage float pour la notation scientifique sans point que yaml laisse en
    str ('2e-4', '1e5' -> float). Crée les clés intermédiaires manquantes.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override invalide (attendu KEY=VALUE) : {item!r}")
        key_path, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        if isinstance(value, str):          # rattrape '2e-4' -> 0.0002
            try:
                value = float(value)
            except ValueError:
                pass                        # vraie chaîne (ex: 'attention')
        keys = key_path.split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
        print(f"[Runner] override : {key_path} = {value!r} ({type(value).__name__})")
    return config

def get_local_tailscale_ip() -> str:
    """
    Récupère l'IP Tailscale (100.x.x.x) de la machine locale.
    Utilisée pour construire automatiquement MLFLOW_TRACKING_URI vu côté pod
    quand on submit un job cloud sans URI explicite.

    Raises:
        RuntimeError: si `tailscale` n'est pas installé ou pas connecté au tailnet.
    """
    try:
        output = subprocess.check_output(
            ["tailscale", "ip", "-4"], text=True, timeout=5
        ).strip()
    except FileNotFoundError as e:
        raise RuntimeError(
            "`tailscale` introuvable. Installe-le et lance `sudo tailscale up`."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"`tailscale ip -4` a échoué (code={e.returncode}). "
            f"Vérifie que tu es connecté au tailnet (`tailscale status`)."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("`tailscale ip -4` a timeout (>5s).") from e

    # Une machine peut avoir plusieurs IPs Tailscale (rare); on prend la première
    ip = output.split("\n")[0].strip()
    if not ip.startswith("100."):
        raise RuntimeError(f"IP Tailscale inattendue : '{ip}' (devrait commencer par 100.)")
    return ip

def resolve_active_base_learners(tracking_uri: str) -> dict:
    """
    Résout @active_text et @active_image depuis MLflow.
 
    Returns:
        {
            "text": {"name": "camembert_lora", "embed_dim": 768, "version": 4},
            "image": {"name": "resnet50_partial_ft", "embed_dim": 2048, "version": 8},
            "extra_caches": ["embeddings_camembert_lora_v1.parquet", ...],
        }
    """
    import mlflow
    from src.models.utils import get_active_val_selection_version
 
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    n_val = get_active_val_selection_version()
 
    result = {}
    for alias, modality in [("active_text", "text"), ("active_image", "image")]:
        found = False
        for rm in client.search_registered_models():
            if not rm.name.startswith("rakuten-base-"):
                continue
            try:
                mv = client.get_model_version_by_alias(rm.name, alias)
                learner_name = rm.name.replace("rakuten-base-", "").replace("-", "_")
                if learner_name not in LEARNER_EMBED_DIM:
                    raise RuntimeError(
                        f"Learner '{learner_name}' pas dans LEARNER_EMBED_DIM."
                    )
                result[modality] = {
                    "name": learner_name,
                    "embed_dim": LEARNER_EMBED_DIM[learner_name],
                    "version": int(mv.version),
                }
                found = True
                break
            except mlflow.exceptions.MlflowException:
                continue
        if not found:
            raise RuntimeError(
                f"Aucun registered model n'a l'alias @{alias}. "
                f"Lancer les base learners et vérifier les promotions."
            )
 
    result["extra_caches"] = [
        f"embeddings_{result['text']['name']}_v{n_val}.parquet",
        f"embeddings_{result['image']['name']}_v{n_val}.parquet",
    ]
    return result

def _load_base_learner_for_m3(registry_name: str, tracking_uri: str,) -> tuple:
    """
    Charge un base learner depuis MLflow registry via @active.
 
    Returns:
        (learner, version) : BaseLearner reconstruit + numéro de version
    """

    client = MlflowClient(tracking_uri)
    model_uri = f"models:/{registry_name}@active"
 
    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
    learner = pyfunc_model.unwrap_python_model().learner
 
    mv = client.get_model_version_by_alias(registry_name, "active")
    version = int(mv.version)
    print(f"[_load_base_learner_for_m3] {registry_name} @active → v{version}")
 
    return learner, version

def _resolve_m3_base_learners(config: dict, tracking_uri: str) -> dict:
    """
    Résout les base learners pour M3, statique ou dynamique.
 
    Statique (section base_learners dans le YAML) :
        Charge les learners nommés explicitement.
 
    Dynamique (pas de section base_learners) :
        Résout @active_text / @active_image depuis MLflow,
        même pattern que build_m2_assembled pour m2_best.
 
    Returns:
        dict avec les clés :
            text_encoder, text_version, text_name, text_embed_dim,
            image_encoder, image_version, image_name, image_embed_dim,
            max_len
    """
 
    if "base_learners" in config:
        # --- Mode statique : noms explicites dans le YAML ---
        bl_cfg = config["base_learners"]
 
        text_encoder, text_version = _load_base_learner_for_m3(
            registry_name=bl_cfg["text"]["registry_name"],
            tracking_uri=tracking_uri,
        )
        image_encoder, image_version = _load_base_learner_for_m3(
            registry_name=bl_cfg["image"]["registry_name"],
            tracking_uri=tracking_uri,
        )
 
        return {
            "text_encoder": text_encoder,
            "text_version": text_version,
            "text_name": bl_cfg["text"]["name"],
            "text_embed_dim": text_encoder.embed_dim,
            "image_encoder": image_encoder,
            "image_version": image_version,
            "image_name": bl_cfg["image"]["name"],
            "image_embed_dim": image_encoder.embed_dim,
            "max_len": bl_cfg["text"].get("max_len", 300),
        }
 
    else:
        # --- Mode dynamique : @active_text / @active_image ---
        print("[build_m3] Pas de base_learners dans le YAML → résolution dynamique")
        client = MlflowClient(tracking_uri)
 
        # Trouver le registered model qui porte @active_text
        # Convention : les alias @active_text et @active_image sont posés
        # par BaseLearnerExperiment sur le registered model du learner promu
        result = {}
        for modality, alias in [("text", "active_text"), ("image", "active_image")]:
            # Chercher dans tous les registered models lequel porte cet alias
            found = False
            for rm in client.search_registered_models():
                try:
                    mv = client.get_model_version_by_alias(rm.name, alias)
                    # Charger le modèle
                    model_uri = f"models:/{rm.name}@{alias}"
                    pyfunc_model = mlflow.pyfunc.load_model(model_uri)
                    learner = pyfunc_model.unwrap_python_model().learner
                    version = int(mv.version)
 
                    # Extraire le nom court depuis le registry name
                    # "rakuten-base-camembert-lora" → "camembert_lora"
                    short_name = rm.name.replace("rakuten-base-", "").replace("-", "_")
 
                    result[f"{modality}_encoder"] = learner
                    result[f"{modality}_version"] = version
                    result[f"{modality}_name"] = short_name
                    result[f"{modality}_embed_dim"] = learner.embed_dim
 
                    print(
                        f"[build_m3] @{alias} → {rm.name} v{version} "
                        f"({learner.embed_dim}d)"
                    )
                    found = True
                    break
                except Exception:
                    continue
 
            if not found:
                raise RuntimeError(
                    f"Aucun registered model ne porte l'alias @{alias}. "
                    f"Lancer les base learners et vérifier les promotions."
                )
 
        result["max_len"] = getattr(result["text_encoder"], "max_len", 300)
        return result


def build_m2_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """Assemble DataModule + M2Stacking + SklearnExperiment depuis une config M2."""
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1, 2, 3]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )

    model_cfg = config["model"]
    def m2_factory(_optuna_callback_unused):
        return M2Stacking(
            text_cols=dm.text_cols,
            image_cols=dm.image_cols,
            tabular_cols=dm.tabular_cols,
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    # Fusion des tags YAML + tags promotion (étape 5)
    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-stacking"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }

    experiment = SklearnExperiment(
        model_factory=m2_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_m2_baseline_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """
    Assemble DataModule + M2Baseline + SklearnExperiment depuis une config.

    Nouvelle architecture Phase 1 (modulaire) : équivalent fonctionnel de
    build_m2_experiment, mais via CamembertFrozen + ResNet18Frozen + StackingLGBM.
    Sert au test d'intégration L.5 (reproduction M2 v4).
    """
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )

    model_cfg = config["model"]

    def m2_baseline_factory(_optuna_callback_unused):
        return M2Baseline(
            tabular_cols=dm.tabular_cols,
            text_embed_dim=768,
            image_embed_dim=512,
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
        )

    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-stacking"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }

    experiment = SklearnExperiment(
        model_factory=m2_baseline_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_m2_assembled_experiment(config: dict) -> tuple[RakutenLightningDataModule, SklearnExperiment]:
    """
    Builder générique pour M2Assembled.
 
    Deux modes :
      - YAML a base_learners: → lecture statique (m2_benchmark, m2_frugal_ft)
      - YAML sans base_learners: → résolution dynamique @active_text/@active_image (m2_best)
 
    Configurations couvertes :
      - m2_benchmark  : TextCNN(3072) + ResNet50PartialFT(2048)
      - m2_frugal_ft  : CamembertLoRA(768) + ResNet18FullFT(512)
      - m2_best       : @active_text + @active_image (dynamique)
    """
    dm_cfg = config["datamodule"]
    mlflow_cfg = config["mlflow"]
 
    # Résoudre tracking URI (pour la résolution dynamique)
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or mlflow_cfg.get("tracking_uri")
        or "http://mlflow:5000"
    )
 
    # Résoudre les base learners : statique (YAML) ou dynamique (MLflow)
    if "base_learners" in config:
        bl_cfg = config["base_learners"]
    else:
        print("[build_m2_assembled] Pas de base_learners dans le YAML → résolution dynamique")
        bl_info = resolve_active_base_learners(tracking_uri)
        bl_cfg = {
            "text": {"name": bl_info["text"]["name"], "embed_dim": bl_info["text"]["embed_dim"]},
            "image": {"name": bl_info["image"]["name"], "embed_dim": bl_info["image"]["embed_dim"]},
        }
        # Injecter les caches dans la config datamodule
        dm_cfg["extra_embedding_caches"] = bl_info["extra_caches"]
        print(
            f"[build_m2_assembled]   text  = {bl_cfg['text']['name']} ({bl_cfg['text']['embed_dim']}d, v{bl_info['text']['version']})\n"
            f"[build_m2_assembled]   image = {bl_cfg['image']['name']} ({bl_cfg['image']['embed_dim']}d, v{bl_info['image']['version']})\n"
            f"[build_m2_assembled]   caches = {dm_cfg['extra_embedding_caches']}"
        )
 
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "m2_embeddings"),
        text_model=dm_cfg["text_model"],
        image_model=dm_cfg["image_model"],
        cache_version=dm_cfg.get("cache_version", 1),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
        extra_embedding_caches=dm_cfg.get("extra_embedding_caches", []),
    )
 
    model_cfg = config["model"]
 
    def m2_assembled_factory(_optuna_callback_unused):
        return M2Assembled(
            tabular_cols=dm.tabular_cols,
            text_learner_name=bl_cfg["text"]["name"],
            text_embed_dim=bl_cfg["text"]["embed_dim"],
            image_learner_name=bl_cfg["image"]["name"],
            image_embed_dim=bl_cfg["image"]["embed_dim"],
            n_classes=model_cfg.get("n_classes", 27),
            n_folds=model_cfg.get("n_folds", 5),
            n_trials=model_cfg.get("n_trials", 30),
            random_state=model_cfg.get("random_state", 42),
            n_jobs_optuna=model_cfg.get("n_jobs_optuna", 4),
            logreg_C_text=model_cfg.get("logreg_C_text", 0.01),
            logreg_C_image=model_cfg.get("logreg_C_image", 0.1),

        )
 
    promotion_cfg = config.get("promotion", {})
    yaml_tags = config["mlflow"].get("tags", {})
    combined_tags = {
        **yaml_tags,
        "base_text": bl_cfg["text"]["name"],
        "base_image": bl_cfg["image"]["name"],
        "registry_model_name": promotion_cfg.get("registry_model_name", "rakuten-m2-assembled"),
        "promotion_epsilon": str(promotion_cfg.get("epsilon", 0.005)),
    }
 
    experiment = SklearnExperiment(
        model_factory=m2_assembled_factory,
        run_name=config["mlflow"]["run_name"],
        tags=combined_tags,
    )
    return dm, experiment

def build_base_learner_experiment(config: dict) -> tuple[RakutenLightningDataModule, BaseLearnerExperiment]:
    """
    M.5 — Assemble DataModule + BaseLearnerExperiment pour un base learner (TextCNN, ResNet50, etc.).
 
    Config attendue :
    ```yaml
    datamodule:
      mode: "base_learners"  # Mode où on récupère les features brutes
      ...
    learner:
      name: "textcnn" ou "resnet50_partial_ft"
      config: {...}  # hyperparams du learner
    mlflow:
      experiment_name: "base_learners_phase1"
      run_name: "textcnn_run_1"
      ...
    ```
    """
    print("[DEBUG] build_base_learner_experiment START")
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "raw_for_finetune"),
        text_model=dm_cfg.get("text_model", None),
        image_model=dm_cfg.get("image_model", None),
        batch_size=dm_cfg.get("batch_size", 64),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )
    print("[DEBUG] DataModule instantiated")
    learner_cfg = config["learner"]
    print(f"[DEBUG] learner_cfg: {learner_cfg}")
    learner_name = learner_cfg["name"]
    learner_config = learner_cfg.get("config", {})
 
    mlflow_cfg = config["mlflow"]
    # Priorité : env var (set par submit_cloud) > CLI > config YAML > default
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or mlflow_cfg.get("tracking_uri")
        or "http://mlflow:5000"
    )
    experiment_name = mlflow_cfg.get("experiment_name", "base_learners_phase1")
    print("[DEBUG] Creating BaseLearnerExperiment...")
    # Instancier BaseLearnerExperiment (Strategy pattern)
    experiment = BaseLearnerExperiment(
        learner_name=learner_name,
        config=learner_config,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        data_folder=Path(dm_cfg.get("data_folder", "data/raw_data")),
        cache_output_dir=Path(os.getenv("DATA_ROOT", ".")) / "data/cache",
    )
    print("[DEBUG] BaseLearnerExperiment instantiated")
    return dm, experiment

def build_m3_experiment(config: dict) -> tuple:
    """
    Builder pour M3 cross-attention fusion.
 
    Supporte deux modes :
        - Statique : base_learners explicites dans le YAML
        - Dynamique : résolution @active_text / @active_image (m3_best)
    """
    print("[build_m3] START")
 
    # --- Tracking URI ---
    mlflow_cfg = config["mlflow"]
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or mlflow_cfg.get("tracking_uri")
        or "http://mlflow:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)
    #config["mlflow"]["tracking_uri"] = tracking_uri 
    # --- Résoudre les base learners ---
    bl = _resolve_m3_base_learners(config, tracking_uri)
    text_encoder = bl["text_encoder"]
    image_encoder = bl["image_encoder"]
 
    print(
        f"[build_m3] Base learners résolus:\n"
        f"  text  = {bl['text_name']} v{bl['text_version']} "
        f"({bl['text_embed_dim']}d)\n"
        f"  image = {bl['image_name']} v{bl['image_version']} "
        f"({bl['image_embed_dim']}d)"
    )
 
    # --- Tags traçabilité ---
    tags = config.setdefault("mlflow", {}).setdefault("tags", {})
    tags["base_text"] = bl["text_name"]
    tags["base_text_version"] = str(bl["text_version"])
    tags["base_image"] = bl["image_name"]
    tags["base_image_version"] = str(bl["image_version"])
 
    # --- DataModule ---
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "raw_for_finetune"),
        batch_size=dm_cfg.get("batch_size", 32),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )
    # Configure le preprocessing M3 (stocke tokenizer + transform,
    # pas besoin de setup() pour ça)
    dm.set_m3_preprocessing(
        tokenizer=text_encoder.tokenizer,
        max_len=bl["max_len"],
        image_transform=image_encoder._eval_transform,
    )
 
    # --- M3 ---
    model_cfg = config.get("model", {})
    model = M3AttentionFusion(
        text_net=text_encoder.net,
        image_net=image_encoder.net,
        d_text=bl["text_embed_dim"],
        d_image=bl["image_embed_dim"],
        config=model_cfg,
    )
    n_params = sum(p.numel() for p in model.fusion.parameters())
    print(f"[build_m3] M3 instancié — {n_params:,} params entraînables")
 
    # --- LightningExperiment ---
    experiment = LightningExperiment(
        model=model,
        dm=dm,
        config=config,
    )

    print("[build_m3] LightningExperiment instancié")
 
    return dm, experiment

def build_m3_hpo_experiment(config: dict) -> tuple:
    """
    Builder pour HPO M3.
 
    Charge les base learners, configure le DataModule, et crée un
    HPOLightningExperiment avec une model_factory qui capture les
    base learners dans une closure.
    """
    print("[build_m3_hpo] START")
 
    tracking_uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
 
    # Charger base learners UNE FOIS
    bl = _resolve_m3_base_learners(config, tracking_uri)
    text_encoder = bl["text_encoder"]
    image_encoder = bl["image_encoder"]
 
    print(
        f"[build_m3_hpo] Base learners :\n"
        f"  text  = {bl['text_name']} v{bl['text_version']} ({bl['text_embed_dim']}d)\n"
        f"  image = {bl['image_name']} v{bl['image_version']} ({bl['image_embed_dim']}d)"
    )
 
    # Tags traçabilité
    tags = config.setdefault("mlflow", {}).setdefault("tags", {})
    tags["base_text"] = bl["text_name"]
    tags["base_text_version"] = str(bl["text_version"])
    tags["base_image"] = bl["image_name"]
    tags["base_image_version"] = str(bl["image_version"])
 
    # DataModule (configuré, PAS setup)
    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "raw_for_finetune"),
        batch_size=dm_cfg.get("batch_size", 32),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )
    dm.set_m3_preprocessing(
        tokenizer=text_encoder.tokenizer,
        max_len=bl["max_len"],
        image_transform=image_encoder._eval_transform,
    )
 
    # model_factory : closure qui capture les base learners
    # Appelée par HPOLightningExperiment à chaque trial avec les HP du trial
    def model_factory(trial_model_cfg: dict) -> M3AttentionFusion:
        return M3AttentionFusion(
            text_net=text_encoder.net,
            image_net=image_encoder.net,
            d_text=bl["text_embed_dim"],
            d_image=bl["image_embed_dim"],
            config=trial_model_cfg,
        )
 
    experiment = HPOLightningExperiment(
        model_factory=model_factory,
        dm=dm,
        config=config,
    )
 
    print("[build_m3_hpo] HPOLightningExperiment instancié")
    return dm, experiment

def build_m3_2_experiment(config: dict) -> tuple:
    """
    Builder pour M3.2 — fusion par attention + co-adaptation DoRA.

    Même squelette que build_m3_experiment (résolution base learners →
    set_m3_preprocessing → modèle → LightningExperiment), mais :
      - modèle = M32CoAdaptationFusion (DoRA sur les dernières couches),
      - d_tab passé explicitement (B.2 ; 0 tant que le tabulaire n'est pas
        injecté dans le MultimodalDataset — cf. commit 4-bis),
      - image_transform = siglip._eval_transform (méthode β).
    Résolution STATIQUE attendue (section base_learners dans le YAML).
    """
    print("[build_m3_2] START")

    mlflow_cfg = config["mlflow"]
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI")
        or mlflow_cfg.get("tracking_uri")
        or "http://mlflow:5000"
    )
    mlflow.set_tracking_uri(tracking_uri)

    bl = _resolve_m3_base_learners(config, tracking_uri)
    text_encoder = bl["text_encoder"]
    image_encoder = bl["image_encoder"]
    print(
        f"[build_m3_2] Base learners résolus:\n"
        f"  text  = {bl['text_name']} v{bl['text_version']} ({bl['text_embed_dim']}d)\n"
        f"  image = {bl['image_name']} v{bl['image_version']} ({bl['image_embed_dim']}d)"
    )

    tags = config.setdefault("mlflow", {}).setdefault("tags", {})
    tags["base_text"] = bl["text_name"]
    tags["base_text_version"] = str(bl["text_version"])
    tags["base_image"] = bl["image_name"]
    tags["base_image_version"] = str(bl["image_version"])

    dm_cfg = config["datamodule"]
    dm = RakutenLightningDataModule(
        mode=dm_cfg.get("mode", "raw_for_finetune"),
        batch_size=dm_cfg.get("batch_size", 16),
        num_workers=dm_cfg.get("num_workers", 4),
        val_size=dm_cfg.get("val_size", 0.10),
        random_state=dm_cfg.get("random_state", 42),
        limit=config.get("limit"),
        train_batches=dm_cfg.get("train_batches", [1]),
        exclude_gold=dm_cfg.get("exclude_gold", True),
    )
    dm.set_m3_preprocessing(
        tokenizer=text_encoder.tokenizer,
        max_len=bl["max_len"],
        image_transform=image_encoder._eval_transform,   # SigLIP β
    )

    model_cfg = config.get("model", {})
    d_tab = model_cfg.get("d_tab", 0)
    model = M32CoAdaptationFusion(
        text_net=text_encoder.net,
        image_net=image_encoder.net,
        d_text=bl["text_embed_dim"],
        d_image=bl["image_embed_dim"],
        d_tab=d_tab,
        config=model_cfg,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build_m3_2] M3.2 instancié — {n_params:,} params entraînables (d_tab={d_tab})")

    experiment = LightningExperiment(model=model, dm=dm, config=config)
    print("[build_m3_2] LightningExperiment instancié")
    return dm, experiment

# Registry des constructeurs par expérience.
EXPERIMENT_BUILDERS = {
    "m2": build_m2_experiment,                  # legacy M2Stacking (à déprécier après L.5 validé)
    "m2_baseline": build_m2_baseline_experiment,  # nouvelle archi modulaire Phase 1
    "m2_benchmark": build_m2_assembled_experiment,
    "m2_frugal_ft": build_m2_assembled_experiment,
    "m2_best": build_m2_assembled_experiment,
    "m3_attention_fusion": build_m3_experiment,
    "m3_attention_fusion_best": build_m3_experiment,
    "m3_hpo": build_m3_hpo_experiment,
    "m3_hpo_best": build_m3_experiment,
    "m3_2_coadaptation": build_m3_2_experiment,
    "base_learner_textcnn": build_base_learner_experiment,
    "base_learner_resnet50_partial_ft": build_base_learner_experiment,
    "base_learner_camembert_lora": build_base_learner_experiment,
    "base_learner_resnet18_full_ft": build_base_learner_experiment,
    "base_learner_siglip2": build_base_learner_experiment   
}

# ─────────────────────────────────────────────────────────────────────────────
# Commands (actions)
# ─────────────────────────────────────────────────────────────────────────────
 

def cmd_prepare_data(dm: RakutenLightningDataModule):
    """Extrait/met à jour le cache d'embeddings. Étape lourde (GPU recommandé)."""
    print(f"[Runner] prepare_data() — cache: {dm.cache_path if hasattr(dm, 'cache_path') else 'N/A'}")
    if hasattr(dm, 'prepare_data'):
        dm.prepare_data()
    else:
        print("[Runner] DataModule n'a pas de prepare_data() (OK pour base_learners)")



def cmd_fit(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Setup + fit avec tracking MLflow."""
    print("[Runner] setup()...")
    dm.setup()
    print("[Runner] fit() avec tracking MLflow...")
    experiment.fit(dm)


def cmd_evaluate(dm: RakutenLightningDataModule, experiment: SklearnExperiment):
    """Évalue le modèle sur le test set. Suppose que fit() a déjà été appelé."""
    if experiment.model is None:
        raise RuntimeError(
            "Le modèle n'est pas fitté. Lance d'abord `--action fit` "
            "dans le même process, ou implémente le rechargement depuis MLflow."
        )
    print("[Runner] setup()...")
    dm.setup()
    results = experiment.evaluate(dm)
    print(f"[Runner] Résultats sur test : {results}")
    return results

def cmd_fit_base_learner(dm: RakutenLightningDataModule, experiment: BaseLearnerExperiment):
    """M.5 — Action pour fit un base learner (TextCNN, ResNet50PartialFT, etc.)."""
    print("[Runner.M5] fit_base_learner() — orchestration BaseLearnerExperiment")
    print("[Runner.M5] setup()...")
    dm.setup()
    print("[Runner.M5] fit() avec MLflow tracking + alias promotion...")
    experiment.fit(dm)
    print("[Runner.M5] fit_base_learner() terminé")

def cmd_fit_lightning(dm: RakutenLightningDataModule, experiment: LightningExperiment):
    """Action pour fit un modèle Lightning (M3, futurs M4+)."""
    print("[Runner] setup()...")
    dm.setup()
    print("[Runner] fit_lightning()...")
    experiment.fit()
    print("[Runner] fit_lightning() terminé")

def cmd_hpo_lightning(dm: RakutenLightningDataModule, experiment: HPOLightningExperiment):
    """HPO Optuna pour M3 — même pattern que les autres cmd."""
    print("[Runner] setup()...")
    dm.setup()
    print("[Runner] hpo_lightning()...")
    experiment.fit()
    print("[Runner] hpo_lightning() terminé")


def cmd_smoke_tailscale():
    """
    Smoke test : valide la chaîne pod → Tailscale → MLflow local.

    Log un run minimal dans l'experiment '_smoke_tailscale' avec un param,
    une metric, et un artefact. Si tout apparaît dans l'UI MLflow locale,
    la chaîne complète est opérationnelle (incluant les uploads multipart).

    Cette action est destinée à tourner sur le pod cloud (--cloud-action smoke_tailscale).
    """
    import socket
    import tempfile

    print(f"[smoke] Hostname pod   : {socket.gethostname()}")
    print(f"[smoke] MLFLOW_TRACKING_URI : {os.environ.get('MLFLOW_TRACKING_URI', '<not set>')}")

    if not os.environ.get("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI non défini, impossible de smoke-tester MLflow")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("_smoke_tailscale")

    with mlflow.start_run(run_name=f"smoke_{socket.gethostname()}") as run:
        print(f"[smoke] Run ID : {run.info.run_id}")
        mlflow.log_param("hostname", socket.gethostname())
        mlflow.log_param("pod_id", os.environ.get("RUNPOD_POD_ID", "unknown"))
        mlflow.log_metric("test_metric", 42.0)

        # Test d'upload d'artefact (chemin critique : exerce le multipart upload HTTP
        # qui peut échouer sur certains tunnels même quand /health répond)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"smoke test from {socket.gethostname()}\n")
            artifact_path = f.name
        mlflow.log_artifact(artifact_path)
        os.unlink(artifact_path)

        print("[smoke] OK : run + param + metric + artifact loggés")

def cmd_fetch_logs(args):
    """Récupère les logs d'un pod cloud depuis R2."""
    import subprocess
    print("[fetch_logs] Logs disponibles sur R2 :")
    subprocess.run([sys.executable, "scripts/r2_logs.py", "list"], check=True)
    if args.job_id:
        # Cherche le log le plus récent contenant le job_id
        # (le job_id RunPod ≠ pod_id mais souvent corrélés, donc on liste et l'user choisit)
        print(f"\n[fetch_logs] Pour télécharger un log, lance :")
        print(f"    python scripts/r2_logs.py download <key_au_dessus> /tmp/<key>")

def cmd_complete_cache(dm, experiment):
    """
    Voie B — termine le 7c interrompu : réécrit le cache parquet du base learner
    @active SANS re-train ni toucher aux alias. Réutilise _write_cache_parquet.
    Learner frozen → embeddings = backbone gelé → instance fraîche = embeddings @active.
    """
    import torch
    from mlflow.tracking import MlflowClient

    print("[complete_cache] setup()...")
    dm.setup()

    learner = experiment._build_learner()        # méthode contenant learner_builders
    learner.net = learner._build_net()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learner.net.to(device)                        # force GPU (pas besoin du fix _forward_in_batches)
    learner.net.eval()
    experiment._learner = learner

    name = f"rakuten-base-{experiment.learner_name}"
    client = MlflowClient()
    try:
        version = int(client.get_model_version_by_alias(name, "active_image").version)
    except Exception:
        version = int(client.get_model_version_by_alias(name, "active").version)
    print(f"[complete_cache] {name} @active -> v{version} | extraction sur {device}")

    experiment._write_cache_parquet(dm, source_model_version=version)
    print("[complete_cache] ✓ Cache réécrit + push R2")

def cmd_submit_cloud(args, config: dict):
    """
    Soumet un job au provider cloud avec fallback sur la liste de GPUs.
    
    Le pod cloud va exécuter `runner.py --action <cloud-action>` après
    avoir pull les données via DVC. Il push les nouveaux artefacts à la fin.
    """
    import time
    from src.cloud.factory import get_cloud_provider
    from src.cloud.base import JobConfig, GPUSpec, VolumeMount, JobStatus
    from src.cloud.exceptions import JobSubmissionError
    
    if args.cloud_action is None:
        raise ValueError("--cloud-action requis pour --action submit_cloud")
    
    # Image Docker (priorité : CLI > env > défaut)
    image = (
        args.cloud_image
        or os.getenv("GHCR_IMAGE_TRAINER")
        or f"ghcr.io/{os.getenv('GITHUB_USER', 'guillaumepe').lower()}/mlops-rakuten-trainer:latest"
    )
    
    # Commande à exécuter dans le pod
    pod_command = [
        "python", "-m", "src.experiments.runner",
        "--experiment", args.experiment,
        "--action", args.cloud_action,
    ]
    if args.limit is not None:
        pod_command += ["--limit", str(args.limit)]
    if getattr(args, "overrides", None):
        pod_command += ["--set", *args.overrides]
    
    # Résolution MLflow tracking URI :
    # - Si fourni explicitement (CLI ou env) ET hors localhost : on garde tel quel
    # - Sinon : auto-construction depuis l'IP Tailscale locale (cas standard)
    mlflow_uri_override = args.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
    local_hosts = ("localhost", "127.0.0.1", "mlflow:5000")
    if mlflow_uri_override and not any(h in mlflow_uri_override for h in local_hosts):
        mlflow_uri_for_pod = mlflow_uri_override
        print(f"[submit_cloud] MLflow URI explicite : {mlflow_uri_for_pod}")
    else:
        ts_ip = get_local_tailscale_ip()
        mlflow_uri_for_pod = f"http://{ts_ip}:5000"
        print(f"[submit_cloud] MLflow URI auto via Tailscale : {mlflow_uri_for_pod}")
    
    # Tailscale auth key : obligatoire pour le pod
    tailscale_authkey = os.getenv("TAILSCALE_AUTHKEY", "")
    if not tailscale_authkey:
        raise RuntimeError(
            "TAILSCALE_AUTHKEY manquante dans .env. "
            "Génère une auth key pod (reusable=true, ephemeral=true) dans le dashboard Tailscale."
        )

    # On passe toujours l'URI résolu au pod, qu'il ait été fourni explicitement ou auto
    # (utile pour les actions qui lisent args.mlflow_tracking_uri côté pod)
    pod_command += ["--mlflow-tracking-uri", mlflow_uri_for_pod]
    # Env vars critiques à passer au pod
    pod_env = {
        # R2 (DVC remote)
        "R2_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "R2_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        # DVC-S3 lit AWS_*, pas R2_* — mapping nécessaire
        "AWS_ACCESS_KEY_ID": os.getenv("R2_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.getenv("R2_SECRET_ACCESS_KEY", ""),
        # Pour l'upload des logs vers R2
        "R2_ENDPOINT_URL": os.getenv("R2_ENDPOINT_URL", ""),
        "R2_BUCKET_NAME": os.getenv("R2_BUCKET_NAME", "rakuten-mlops-dvc"),
        # MongoDB Atlas
        "MONGO_URI": os.getenv("MONGO_URI", ""),
        "MONGO_DB_NAME": os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN"),
        # Tailscale (overlay network vers MLflow local)
        "TAILSCALE_AUTHKEY": tailscale_authkey,
        # MLflow
        "MLFLOW_TRACKING_URI": mlflow_uri_for_pod,
        # Self-terminate
        "RUNPOD_API_KEY": os.getenv("RUNPOD_API_KEY", ""),
        # DATA_ROOT pour les paths
        "DATA_ROOT": "/workspace",
        # DVC auto-push
        "DVC_AUTO_PUSH": "true",
    }
    
    # Targets DVC à puller
    dvc_targets = args.cloud_dvc_targets or [
        "data/raw_data/X_train_update.csv.dvc",
        "data/raw_data/Y_train_update.csv.dvc",
        "data/raw_data/images/image_train.tar.zst.dvc",
    ]
    pod_env["DVC_PULL_TARGETS"] = " ".join(dvc_targets)
    
    # Vérifier les vars critiques
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "MONGO_URI"):
        if not pod_env[key]:
            raise RuntimeError(f"{key} manquante dans .env")
    
    # Volume RunPod cache (optionnel)
    volumes = []
    volume_id = os.getenv("RUNPOD_VOLUME_ID")
    if volume_id:
        volumes.append(VolumeMount(
            volume_id=volume_id,
            mount_path=os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace/cache"),
        ))
        print(f"[submit_cloud] Volume cache attaché : {volume_id}")
    else:
        print("[submit_cloud] Pas de volume cache (RUNPOD_VOLUME_ID non défini)")
    
    print(f"[submit_cloud] Image      : {image}")
    print(f"[submit_cloud] GPUs cibles : {args.gpu_types}")
    print(f"[submit_cloud] Commande   : {' '.join(pod_command)}")
    # Timeout adaptatif par action si pas explicitement override en CLI
    TIMEOUT_BY_ACTION = {
        "smoke_tailscale": 300,
        "prepare_data": 3600,
        "fit": 7200,
        "promote": 600,
    }
    DEFAULT_TIMEOUT = 3600  # le défaut du parser CLI
    if args.cloud_timeout == DEFAULT_TIMEOUT:
        # Pas overridé par l'utilisateur → on prend le défaut spécifique à l'action
        args.cloud_timeout = TIMEOUT_BY_ACTION.get(args.cloud_action, DEFAULT_TIMEOUT)
        print(f"[submit_cloud] Timeout auto pour action '{args.cloud_action}' : {args.cloud_timeout}s")
    print(f"[submit_cloud] Timeout    : {args.cloud_timeout}s")
    
    # Submit avec fallback sur la liste de GPUs
    provider = get_cloud_provider()
    print(f"[submit_cloud] Provider   : {provider.name}")
    
    handle = None
    last_error = None
    for gpu_type in args.gpu_types:
        print(f"[submit_cloud] Tentative GPU : {gpu_type}")
        job_config = JobConfig(
            image=image,
            command=pod_command,
            env=pod_env,
            gpu=GPUSpec(gpu_type=gpu_type, count=1),
            volumes=volumes,
            name=f"rakuten-{args.experiment}-{args.cloud_action}",
        )
        try:
            handle = provider.submit_job(job_config)
            print(f"[submit_cloud] ✓ Pod provisionné avec {gpu_type}")
            print(f"[submit_cloud] Job ID     : {handle.job_id}")
            break
        except JobSubmissionError as e:
            print(f"[submit_cloud] ✗ {gpu_type} indispo : {e}")
            last_error = e
            continue
    
    if handle is None:
        raise RuntimeError(
            f"Aucun GPU dispo dans la liste {args.gpu_types}. "
            f"Dernière erreur : {last_error}"
        )
    
    # Wait avec polling visible (debug)
    print(f"[submit_cloud] Attente de la fin du job...")
    start = time.time()
    poll_count = 0
    last_status = JobStatus.UNKNOWN
    
    try:
        while True:
            poll_count += 1
            last_status = provider.get_status(handle)
            elapsed = int(time.time() - start)
            print(f"[submit_cloud] [t={elapsed}s] Poll #{poll_count} : status={last_status.value}")
            
            if last_status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED):
                break
            
            if elapsed > args.cloud_timeout:
                print(f"[submit_cloud] Timeout dépassé, stop du pod")
                provider.stop(handle)
                raise RuntimeError(f"Timeout après {elapsed}s")
            
            time.sleep(10)
        
        duration = time.time() - start
        print(f"[submit_cloud] Job terminé : {last_status.value}")
        print(f"[submit_cloud] Durée      : {duration:.1f}s")
    
    except Exception as e:
        print(f"[submit_cloud] Erreur wait : {e}")
        print(f"[submit_cloud] Tentative de stop du pod...")
        try:
            provider.stop(handle)
        except Exception as stop_err:
            print(f"[submit_cloud] Stop échec : {stop_err}")
        raise
    
    return last_status

def main():
    parser = argparse.ArgumentParser(description="MLOps experiment runner")
    parser.add_argument(
        "--experiment", required=True, choices=list(EXPERIMENT_BUILDERS),
        help="Nom de l'expérience (correspond à src/experiments/config/<name>.yaml)",
    )
    parser.add_argument(
        "--action", required=True,
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate", "fit_base_learner","fit_lightning", "submit_cloud", "smoke_tailscale","fetch_logs","hpo_lightning","complete_cache"],
        help="Action à exécuter",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="URI MLflow (override la config). Ex: http://mlflow:5000",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limiter le nombre de samples (debug/test rapide). None = full dataset.",
    )
    parser.add_argument(
        "--cloud-action",
        default=None,
        choices=["prepare_data", "fit", "evaluate", "fit_and_evaluate", "smoke_tailscale","fit_base_learner","fit_lightning","hpo_lightning","complete_cache"],
        help="(submit_cloud only) Quelle action le pod cloud doit exécuter",
    )
    parser.add_argument(
        "--gpu-types",
        nargs="+",
        default=["rtx_5090","rtx_4090", "rtx_3090", "rtx_4080", "rtx_a5000", "rtx_a6000","rtx_a4000", "a40", "l40", "l40s", "a100_40gb","rtx_pro_4500"],
        help="(submit_cloud) Liste de GPUs à essayer en cascade (du préféré au fallback)",
)
    parser.add_argument(
        "--cloud-image",
        default=None,
        help="(submit_cloud only) Image Docker. Si None, lit GHCR_IMAGE_TRAINER ou défaut.",
    )
    parser.add_argument(
        "--cloud-timeout",
        type=int,
        default=3600,
        help="(submit_cloud only) Timeout en secondes (défaut 1h)",
    )
    parser.add_argument(
    "--cloud-dvc-targets",
        nargs="+",
        default=None,
        help="(submit_cloud) Liste des .dvc à puller. Défaut : X_train + Y_train + images.",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="(fetch_logs only) Job ID RunPod du pod dont on veut les logs",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Overrides config en notation pointée, sans rebuild. Ex: "
             "--set model.dora_rank=8 model.dora_last_n=2 model.lr_dora=2e-4 "
             "trainer.max_epochs=15. Typage via yaml.",
    )
    
    args = parser.parse_args()

    # Charger la config
    config = load_config(args.experiment)
    if args.limit is not None:
        config["limit"] = args.limit
    if args.overrides:
        apply_overrides(config, args.overrides)
    print(f"[Runner] Config chargée : {args.experiment} (limit={args.limit})")


    # MLflow tracking URI : CLI > config > défaut
    tracking_uri = (
        args.mlflow_tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or config.get("mlflow", {}).get("tracking_uri")
        or "http://mlflow:5000"
    )

    # Injecter dans la config pour que tous les builders/experiments l'utilisent
    config.setdefault("mlflow", {})["tracking_uri"] = tracking_uri
    
    # Dispatch action
    if args.action == "submit_cloud":
        # Pas d'init MLflow local : le tracking se fera côté pod
        cmd_submit_cloud(args, config)
        return

    # smoke_tailscale : test de bout en bout sans construire DataModule/Experiment
    if args.action == "smoke_tailscale":
        cmd_smoke_tailscale()
        return

    # Init MLflow seulement pour les actions qui en ont besoin
    if args.action in ("fit", "evaluate", "fit_and_evaluate"):
        mlflow.set_tracking_uri(tracking_uri)
        print(f"[Runner] MLflow tracking URI : {tracking_uri}")
        experiment_name = config["mlflow"].get("experiment_name", args.experiment)
        mlflow.set_experiment(experiment_name)
    else:
        print(f"[Runner] Action '{args.action}' : pas d'init MLflow nécessaire")

    # Construire les composants
    builder = EXPERIMENT_BUILDERS[args.experiment]
    dm, experiment = builder(config)

    if args.action == "prepare_data":
        cmd_prepare_data(dm)
    elif args.action == "fit":
        cmd_prepare_data(dm)
        cmd_fit(dm, experiment)
    elif args.action == "evaluate":
        cmd_evaluate(dm, experiment)
    elif args.action == "fit_and_evaluate":
        cmd_prepare_data(dm)
        cmd_fit(dm, experiment)
        cmd_evaluate(dm, experiment)
    elif args.action == "fit_base_learner":
        cmd_fit_base_learner(dm, experiment)
    elif args.action == "complete_cache":
        cmd_complete_cache(dm, experiment)
    elif args.action == "fit_lightning":
        cmd_fit_lightning(dm, experiment)
    elif args.action == "hpo_lightning":
        cmd_hpo_lightning(dm, experiment)

    elif args.action == "fetch_logs":
        cmd_fetch_logs(args)


if __name__ == "__main__":
    sys.exit(main())