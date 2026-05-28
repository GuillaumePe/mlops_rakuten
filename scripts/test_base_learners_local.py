#!/usr/bin/env python3
"""
Sanity check : teste l'import et l'instantiation des 4 base learners.

Utile pour détecter les erreurs de syntaxe, imports manquants, ou dépendances
cassées AVANT de relancer en cloud.

Usage:
    python test_base_learners_local.py [--verbose]

Exit code:
    0 = OK (tous les tests passent)
    1 = ERREUR (au moins un test échoue)
"""
import sys
import traceback
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_textcnn():
    """Test TextCNN import et instantiation."""
    print("\n" + "="*70)
    print("TEST 1 : TextCNN (Bloc M — text)")
    print("="*70)
    try:
        from src.models.base_learners.text.textcnn import TextCNN
        
        # Instanciation avec defaults
        learner = TextCNN(
            vocab_size=50000,
            max_len=128,
            embed_dim=300,
            n_filters=512,
            kernel_sizes=(1, 2, 3, 4, 5, 6),
            n_classes=27,
            dropout=0.5,
            batch_size=32,
            max_epochs=15,
            patience=2,
            lr=1e-3,
            weight_decay=0.0,
            text_col="text",
        )
        
        # Vérifications
        assert learner.name == "textcnn", f"name={learner.name}"
        assert learner.modality == "text", f"modality={learner.modality}"
        assert learner.embed_dim == 3072, f"embed_dim={learner.embed_dim}"
        assert learner.n_classes == 27, f"n_classes={learner.n_classes}"
        
        print("✅ TextCNN : PASS")
        print(f"   name={learner.name}, modality={learner.modality}, embed_dim={learner.embed_dim}")
        return True
        
    except Exception as e:
        print(f"❌ TextCNN : FAIL")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_resnet50_partial_ft():
    """Test ResNet50PartialFT import et instantiation."""
    print("\n" + "="*70)
    print("TEST 2 : ResNet50PartialFT (Bloc M — image)")
    print("="*70)
    try:
        from src.models.base_learners.image.resnet50_partial_ft import ResNet50PartialFT
        
        # Instanciation avec defaults
        learner = ResNet50PartialFT(
            image_folder="/tmp",
            n_classes=27,
            batch_size=32,
            max_epochs=15,
            patience=3,
            lr_head=1e-3,
            lr_backbone=1e-5,
            weight_decay=1e-4,
            num_workers=4,
            random_state=42,
            precision="bf16-mixed",
        )
        
        # Vérifications
        assert learner.name == "resnet50_partial_ft", f"name={learner.name}"
        assert learner.modality == "image", f"modality={learner.modality}"
        assert learner.embed_dim == 2048, f"embed_dim={learner.embed_dim}"
        assert learner.n_classes == 27, f"n_classes={learner.n_classes}"
        
        print("✅ ResNet50PartialFT : PASS")
        print(f"   name={learner.name}, modality={learner.modality}, embed_dim={learner.embed_dim}")
        return True
        
    except Exception as e:
        print(f"❌ ResNet50PartialFT : FAIL")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_resnet18_full_ft():
    """Test ResNet18FullFT import et instantiation (BLOC N.1)."""
    print("\n" + "="*70)
    print("TEST 3 : ResNet18FullFT (Bloc N.1 — image)")
    print("="*70)
    try:
        from src.models.base_learners.image.resnet18_full_ft import ResNet18FullFT
        
        # Instanciation avec defaults
        learner = ResNet18FullFT(
            image_folder="/tmp",
            n_classes=27,
            batch_size=32,
            max_epochs=15,
            patience=3,
            lr=1e-4,
            weight_decay=1e-4,
            num_workers=4,
            random_state=42,
            precision="bf16-mixed",
            augmentation_level="medium",
        )
        
        # Vérifications
        assert learner.name == "resnet18_full_ft", f"name={learner.name}"
        assert learner.modality == "image", f"modality={learner.modality}"
        assert learner.embed_dim == 512, f"embed_dim={learner.embed_dim}"
        assert learner.n_classes == 27, f"n_classes={learner.n_classes}"
        assert learner._augmentation_level == "medium", f"augmentation_level={learner._augmentation_level}"
        
        print("✅ ResNet18FullFT : PASS")
        print(f"   name={learner.name}, modality={learner.modality}, embed_dim={learner.embed_dim}")
        print(f"   augmentation_level={learner._augmentation_level}")
        return True
        
    except Exception as e:
        print(f"❌ ResNet18FullFT : FAIL")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_camembert_lora():
    """Test CamembertLoRA import et instantiation (BLOC N.2)."""
    print("\n" + "="*70)
    print("TEST 4 : CamembertLoRA (Bloc N.2 — text)")
    print("="*70)
    try:
        from src.models.base_learners.text.camembert_lora import CamembertLoRA
        
        # Instanciation avec defaults
        learner = CamembertLoRA(
            model_name="camembert-base",
            n_classes=27,
            max_len=128,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.05,
            batch_size=32,
            max_epochs=10,
            patience=2,
            lr_lora=5e-4,
            lr_head=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            text_col="text",
            num_workers=2,
            random_state=42,
            precision="bf16-mixed",
        )
        
        # Vérifications
        assert learner.name == "camembert_lora", f"name={learner.name}"
        assert learner.modality == "text", f"modality={learner.modality}"
        assert learner.embed_dim == 768, f"embed_dim={learner.embed_dim}"
        assert learner.n_classes == 27, f"n_classes={learner.n_classes}"
        assert learner._lora_rank == 16, f"lora_rank={learner._lora_rank}"
        assert learner._lr_lora == 5e-4, f"lr_lora={learner._lr_lora}"
        
        print("✅ CamembertLoRA : PASS")
        print(f"   name={learner.name}, modality={learner.modality}, embed_dim={learner.embed_dim}")
        print(f"   lora_rank={learner._lora_rank}, lr_lora={learner._lr_lora}")
        return True
        
    except Exception as e:
        print(f"❌ CamembertLoRA : FAIL")
        print(f"   Error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*70)
    print("█  SANITY CHECK — 4 Base Learners (Local Compile Test)")
    print("█"*70)
    
    results = {
        "TextCNN": test_textcnn(),
        "ResNet50PartialFT": test_resnet50_partial_ft(),
        "ResNet18FullFT": test_resnet18_full_ft(),
        "CamembertLoRA": test_camembert_lora(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n{passed}/{total} tests passed:")
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status:10} {name}")
    
    if passed == total:
        print("\n✅ All tests passed! Safe to launch cloud runs.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Fix errors before cloud launch.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
