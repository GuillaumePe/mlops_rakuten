"""
Smoke test HPO Lightning — 27 samples synthétiques, 2 trials Optuna.

Valide le pipeline complet :
    1. HPOLightningExperiment instanciation
    2. Optuna study avec nested MLflow runs
    3. Retrain final avec best HP
    4. Métriques gold loggées

Utilise un MLflow tracking local (temp dir), pas de serveur.
Zéro dépendance externe (pas de MongoDB, pas de base learners réels).

Usage :
    python tests/smoke_test_m3_hpo.py
"""
import os
import shutil
import tempfile

import torch
import torch.nn as nn
import numpy as np
import lightning as L
import mlflow
from torch.utils.data import DataLoader, Dataset

from src.models.fusion.attention_fusion import AttentionFusionModule
from src.models.assembled.m3_attention_fusion import M3AttentionFusion
from src.experiments.strategies.hpo_lightning_experiment import HPOLightningExperiment


# ─────────────────────────────────────────────────────────────────────
# Mock encoders
# ─────────────────────────────────────────────────────────────────────

class MockTextNet(nn.Module):
    def __init__(self, d_hidden=768, seq_len=128):
        super().__init__()
        self.d_hidden = d_hidden
        self.seq_len = seq_len
        self._dummy = nn.Parameter(torch.zeros(1))

    def _token_level_features(self, input_ids, attention_mask):
        B = input_ids.size(0)
        hidden = torch.randn(B, self.seq_len, self.d_hidden, device=input_ids.device)
        return hidden, attention_mask


class MockImageNet(nn.Module):
    def __init__(self, d_hidden=512, n_patches=49):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_patches = n_patches
        self._dummy = nn.Parameter(torch.zeros(1))

    def _spatial_feature_map(self, x):
        B = x.size(0)
        return torch.randn(B, self.n_patches, self.d_hidden, device=x.device)


# ─────────────────────────────────────────────────────────────────────
# Dataset synthétique + Mock DataModule
# ─────────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    def __init__(self, n=27, seq_len=128):
        self.n = n
        self.seq_len = seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randint(0, 30000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "image": torch.randn(3, 224, 224),
            "label": torch.tensor(idx % 27, dtype=torch.long),
        }


class MockDataModule:
    """Simule le DataModule avec les méthodes attendues par HPOLightningExperiment."""

    def __init__(self):
        self.ds = SyntheticDataset(n=27)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=27, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=27, shuffle=False)

    def gold_dataloader(self):
        return DataLoader(self.ds, batch_size=27, shuffle=False)

    def get_gold_labels(self):
        return np.arange(27)


# ─────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────

def test_hpo_full_pipeline():
    """Pipeline HPO complet : 2 trials + retrain final."""

    # MLflow tracking dans un temp dir
    tmp_dir = tempfile.mkdtemp(prefix="mlflow_smoke_hpo_")
    tracking_uri = f"file://{tmp_dir}"

    print(f"[smoke] MLflow tracking : {tracking_uri}")

    # Mock base learners
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    # model_factory (closure capturant les base learners)
    def model_factory(trial_cfg):
        return M3AttentionFusion(
            text_net=text_net,
            image_net=image_net,
            d_text=768,
            d_image=512,
            config=trial_cfg,
        )

    # Mock DataModule
    dm = MockDataModule()
    dm.setup()

    # Config
    config = {
        "model": {
            "n_classes": 27,
            "n_heads": 8,
            "warmup_ratio": 0.1,
        },
        "hpo": {
            "n_trials": 2,
            "study_name": "smoke_hpo",
            "search_space": {
                "d_model": {"type": "categorical", "choices": [256, 512]},
                "n_layers": {"type": "int", "low": 1, "high": 2},
                "dropout": {"type": "float", "low": 0.2, "high": 0.5},
                "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
                "weight_decay": {"type": "float", "low": 1e-3, "high": 0.05, "log": True},
                "dim_ff_factor": {"type": "categorical", "choices": [2, 4]},
            },
        },
        "trainer": {
            "max_epochs": 2,
            "patience": 2,
            "precision": "32-true",
            "accelerator": "cpu",
        },
        "mlflow": {
            "experiment_name": "smoke_hpo_test",
            "run_name": "smoke_hpo",
            "tracking_uri": tracking_uri,
            "tags": {"test": "smoke"},
        },
        "promotion": {
            "registry_model_name": "smoke-m3-hpo",
            "threshold": 0.0,
        },
    }

    # Lancer HPO
    experiment = HPOLightningExperiment(
        model_factory=model_factory,
        dm=dm,
        config=config,
    )
    experiment.fit()

    # Vérifications
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name("smoke_hpo_test")
    assert exp is not None, "Experiment MLflow non trouvé"

    runs = client.search_runs(exp.experiment_id)
    print(f"\n[smoke] {len(runs)} runs trouvés dans MLflow")

    # On doit avoir : 1 parent + 2 trials nested + 1 retrain final = 4 runs
    # (les nested sont des child runs du parent)
    run_names = [r.data.tags.get("mlflow.runName", "") for r in runs]
    print(f"[smoke] Run names : {run_names}")

    # Vérifier le parent run
    parent_runs = [r for r in runs if r.data.tags.get("mlflow.runName") == "M3_HPO"]
    assert len(parent_runs) == 1, f"Expected 1 parent run, got {len(parent_runs)}"
    print("  ✓ Parent run M3_HPO trouvé")

    # Vérifier les nested trials
    trial_runs = [r for r in runs if "trial_" in r.data.tags.get("mlflow.runName", "")]
    assert len(trial_runs) == 2, f"Expected 2 trial runs, got {len(trial_runs)}"
    print(f"  ✓ {len(trial_runs)} trial runs trouvés")

    # Vérifier le retrain final
    final_runs = [r for r in runs if "hpo_best" in r.data.tags.get("mlflow.runName", "")]
    assert len(final_runs) == 1, f"Expected 1 final run, got {len(final_runs)}"
    final_run = final_runs[0]
    print(f"  ✓ Final retrain run trouvé : {final_run.data.tags.get('mlflow.runName')}")

    # Vérifier les métriques gold sur le final run
    gold_f1 = final_run.data.metrics.get("eval_gold/f1_weighted")
    assert gold_f1 is not None, "eval_gold/f1_weighted manquant sur le final run"
    print(f"  ✓ eval_gold/f1_weighted = {gold_f1:.4f}")

    # Vérifier les tags HPO
    assert "hpo_best_trial" in final_run.data.tags
    assert "hpo_best_val_f1" in final_run.data.tags
    print(f"  ✓ Tags HPO présents")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST HPO : PASS")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_hpo_full_pipeline()
