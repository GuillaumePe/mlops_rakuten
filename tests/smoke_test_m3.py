"""
Smoke test M3 — valide le pipeline complet avec 27 samples synthétiques.

Zéro dépendance externe (pas de MLflow, pas de MongoDB, pas de base learners réels).
Mocks les encoders frozen avec des modules qui retournent des tensors random de la
bonne shape. Valide :
    1. Instanciation M3AttentionFusion
    2. Forward pass (shapes correctes)
    3. Backward pass (gradients uniquement sur fusion, pas sur encoders)
    4. 3 epochs de training via Lightning Trainer
    5. Prédiction (predict_proba-like)

Usage :
    python -m tests.smoke_test_m3
"""
import torch
import torch.nn as nn
import numpy as np
import lightning as L
from torch.utils.data import DataLoader, Dataset

from src.models.fusion.attention_fusion import AttentionFusionModule
from src.models.assembled.m3_attention_fusion import M3AttentionFusion


# ─────────────────────────────────────────────────────────────────────
# Mock encoders (remplacent CamemBERT + ResNet18 frozen)
# ─────────────────────────────────────────────────────────────────────

class MockTextNet(nn.Module):
    """Simule _CamembertLoRALightning._token_level_features."""

    def __init__(self, d_hidden: int = 768, seq_len: int = 300):
        super().__init__()
        self.d_hidden = d_hidden
        self.seq_len = seq_len
        # Un paramètre bidon pour que .parameters() ne soit pas vide
        self._dummy = nn.Parameter(torch.zeros(1))

    def _token_level_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        hidden = torch.randn(B, self.seq_len, self.d_hidden, device=input_ids.device)
        return hidden, attention_mask


class MockImageNet(nn.Module):
    """Simule _ResNet18FullFTLightning._spatial_feature_map."""

    def __init__(self, d_hidden: int = 512, n_patches: int = 49):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_patches = n_patches
        self._dummy = nn.Parameter(torch.zeros(1))

    def _spatial_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        return torch.randn(B, self.n_patches, self.d_hidden, device=x.device)


# ─────────────────────────────────────────────────────────────────────
# Dataset synthétique (27 samples, 1 par classe)
# ─────────────────────────────────────────────────────────────────────

class SyntheticMultimodalDataset(Dataset):
    """27 samples avec des tensors random, 1 par classe."""

    def __init__(self, n_classes: int = 27, seq_len: int = 300):
        self.n_classes = n_classes
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n_classes

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": torch.randint(0, 30000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "image": torch.randn(3, 224, 224),
            "label": torch.tensor(idx % self.n_classes, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_1_instantiation():
    """M3 s'instancie sans erreur."""
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    model = M3AttentionFusion(
        text_net=text_net,
        image_net=image_net,
        d_text=768,
        d_image=512,
        config={"d_model": 512, "n_heads": 8, "n_layers": 2, "n_classes": 27},
    )

    n_fusion = sum(p.numel() for p in model.fusion.parameters())
    n_frozen_text = sum(p.numel() for p in model.text_net.parameters())
    n_frozen_image = sum(p.numel() for p in model.image_net.parameters())
    n_trainable = sum(p.numel() for p in model.fusion.parameters() if p.requires_grad)

    print(f"  fusion params     : {n_fusion:,}")
    print(f"  frozen text params: {n_frozen_text:,}")
    print(f"  frozen image      : {n_frozen_image:,}")
    print(f"  trainable         : {n_trainable:,}")

    assert n_trainable == n_fusion, "Tous les params fusion doivent être trainable"
    assert not any(p.requires_grad for p in model.text_net.parameters()), \
        "text_net doit être frozen"
    assert not any(p.requires_grad for p in model.image_net.parameters()), \
        "image_net doit être frozen"
    print("  ✓ PASS")


def test_2_forward_shapes():
    """Les shapes de sortie sont correctes."""
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    model = M3AttentionFusion(
        text_net=text_net,
        image_net=image_net,
        d_text=768,
        d_image=512,
        config={"d_model": 512, "n_heads": 8, "n_layers": 2, "n_classes": 27},
    )

    batch = {
        "input_ids": torch.randint(0, 30000, (4, 300)),
        "attention_mask": torch.ones(4, 300, dtype=torch.long),
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 2, 3]),
    }

    logits = model(batch)
    assert logits.shape == (4, 27), f"Expected (4, 27), got {logits.shape}"
    print(f"  logits shape: {logits.shape} ✓")
    print("  ✓ PASS")


def test_3_backward_gradients():
    """Le backward ne propage PAS dans les encoders frozen."""
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    model = M3AttentionFusion(
        text_net=text_net,
        image_net=image_net,
        d_text=768,
        d_image=512,
        config={"d_model": 512, "n_heads": 8, "n_layers": 2, "n_classes": 27},
    )

    batch = {
        "input_ids": torch.randint(0, 30000, (4, 300)),
        "attention_mask": torch.ones(4, 300, dtype=torch.long),
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 2, 3]),
    }

    loss = model.training_step(batch, 0)
    loss.backward()

    # Fusion a des gradients
    fusion_grads = [
        p.grad for p in model.fusion.parameters() if p.grad is not None
    ]
    assert len(fusion_grads) > 0, "Fusion doit avoir des gradients"

    # Encoders n'en ont pas
    for p in model.text_net.parameters():
        assert p.grad is None, "text_net ne doit PAS avoir de gradient"
    for p in model.image_net.parameters():
        assert p.grad is None, "image_net ne doit PAS avoir de gradient"

    print(f"  fusion grads: {len(fusion_grads)} tensors ✓")
    print(f"  text_net grads: 0 ✓")
    print(f"  image_net grads: 0 ✓")
    print("  ✓ PASS")


def test_4_lightning_training():
    """3 epochs de training via Lightning Trainer (CPU, pas de MLflow)."""
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    model = M3AttentionFusion(
        text_net=text_net,
        image_net=image_net,
        d_text=768,
        d_image=512,
        config={
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 2,
            "n_classes": 27,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
        },
    )

    ds = SyntheticMultimodalDataset(n_classes=27)
    loader = DataLoader(ds, batch_size=27, shuffle=True)

    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, loader, loader)

    # Vérifier que le training a tourné
    assert trainer.current_epoch == 3
    print(f"  3 epochs completed ✓")
    print("  ✓ PASS")


def test_5_prediction():
    """Predict sur 27 samples → probas (27, 27) sommant à 1."""
    text_net = MockTextNet(d_hidden=768)
    image_net = MockImageNet(d_hidden=512)

    model = M3AttentionFusion(
        text_net=text_net,
        image_net=image_net,
        d_text=768,
        d_image=512,
        config={"d_model": 512, "n_heads": 8, "n_layers": 2, "n_classes": 27},
    )

    ds = SyntheticMultimodalDataset(n_classes=27)
    loader = DataLoader(ds, batch_size=27, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            proba = torch.softmax(logits, dim=-1)

    assert proba.shape == (27, 27), f"Expected (27, 27), got {proba.shape}"
    row_sums = proba.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(27), atol=1e-5), \
        f"Probas doivent sommer à 1, got {row_sums}"
    print(f"  proba shape: {proba.shape} ✓")
    print(f"  row sums ≈ 1.0 ✓")
    print("  ✓ PASS")


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("1. Instanciation", test_1_instantiation),
        ("2. Forward shapes", test_2_forward_shapes),
        ("3. Backward gradients", test_3_backward_gradients),
        ("4. Lightning training (3 epochs)", test_4_lightning_training),
        ("5. Prediction probas", test_5_prediction),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"{'='*60}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"RÉSULTAT: {passed}/{len(tests)} tests passés")
    print(f"{'='*60}")

    if passed < len(tests):
        exit(1)
