"""
ResNet18FullFT : BaseLearner image pour M2.1 (Bloc N).

Variante 'full fine-tune' du ResNet18 : TOUS les poids défreezés
(~11.7M trainable / 11.7M total). Contraste avec ResNet50PartialFT (M2.2)
qui ne défreeze que layer3+layer4+head.

Justification mathématique du full FT :
- ResNet18 petit (11.7M) → ratio n/d = 38k/11.7M ≈ 0.003 (mauvais)
  Mais early stopping (patience=3 sur val_f1_weighted) régularise implicitement
  (équivalent L2 — Bishop 1995, "Regularization and Training Set Size").
- Vs partial FT : full FT a moins de biais (tout adapté au domaine Rakuten),
  variance contrôlée par early stopping. Pour ResNet18 (petit), le risque
  d'overfit est gérable. Pour ResNet50 (35M), on préfère partial FT.
- Vs ResNet50 partial FT (M2.2) : ResNet18 full FT a 4x moins de paramètres
  trainables (11.7M vs 14M), donc plus frugal en VRAM et temps GPU,
  tout en gardant un biais inductif fort grâce à pretrained ImageNet.

Niveau d'augmentation paramétrable (augmentation_level) :
- "soft"   : Resize+RRCrop(0.85-1.0)+HFlip — alignement benchmark Rakuten,
             approprié pour partial FT, faible régularisation
- "medium" : ajoute Rotation(10°)+ColorJitter modéré — DÉFAUT pour full FT,
             régularisation augmentée pour compenser les degrés de liberté
- "hard"   : ajoute Perspective+ColorJitter aggressif+RandomErasing —
             régularisation maximale, à activer si overfit manifeste à medium

Justification (Hernández-García & König 2018, "Data augmentation as explicit
regularization") : l'augmentation est équivalente à un terme de régularisation
sur l'output. Plus la distribution des samples augmentés s'éloigne de la
distribution originale, plus la régularisation est forte. Trade-off :
- Trop peu → faible variance batch → faible régularisation → overfit risk
- Trop → biais introduit (samples trop éloignés du test set) → underfit

Workflow :
- fit(X, y) : split 80/20 (fallback si X_val=None), fine-tune avec early
  stopping patience=3
- extract_embeddings(X) : forward jusqu'à AvgPool → (n, 512)
- predict_proba(X) : softmax du forward complet → (n, 27)

Le DataFrame d'entrée doit contenir au minimum les colonnes 'imageid' et
'productid' (utilisées pour reconstruire le chemin d'image).

Hyperparamètres de référence (cf. base_learner_resnet18_full_ft.yaml) :
- LR=1e-4 (unifié, pas de LR différentiel — full FT cohérent)
- weight_decay=1e-4 (régularisation L2 légère)
- batch_size=32 (variance gradient manageable, alignement ResNet50PartialFT)
- max_epochs=15, patience=3 (cohérent benchmark)
- precision="bf16-mixed" (cohérent ResNet50PartialFT)
- augmentation_level="medium" (cohérent full FT, ablation possible à "hard")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightning.pytorch as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.models.base_learners._base import BaseLearner


AugmentationLevel = Literal["soft", "medium", "hard"]


# ====================================================================== #
# Dataset : images chargées depuis disque                                 #
# ====================================================================== #


class _ImageDataset(Dataset):
    """
    Dataset minimal pour images Rakuten.

    On gère ici la conversion RGB systématique (certaines images Rakuten sont RGBA/L)
    pour éviter un crash silencieux dans le DataLoader.
    """

    def __init__(self, image_paths: list[Path], labels: np.ndarray | None, transform):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels  # None autorisé pour predict (placeholder retourné)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_t = self.transform(img)
        if self.labels is not None:
            return img_t, int(self.labels[idx])
        return img_t, 0  # placeholder pour predict


# ====================================================================== #
# Helper : transforms par défaut (paramétrables par level)                #
# ====================================================================== #


# Normalisation ImageNet partagée par tous les transforms train + eval
_IMAGENET_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _default_eval_transform():
    """Transforms ImageNet standard : resize 232 + center crop 224 + normalize."""
    return models.ResNet18_Weights.IMAGENET1K_V1.transforms()


def _default_train_transform(level: AugmentationLevel = "medium") -> transforms.Compose:
    """
    Construit le transform d'augmentation train selon le niveau demandé.

    Trois presets calibrés pour Rakuten :

    - "soft"   : alignement benchmark Rakuten (= ResNet50PartialFT).
        Resize(232) + RandomResizedCrop(224, 0.85-1.0) + HFlip + normalize.
        Régularisation faible. Approprié si overfit pas un problème
        (partial FT, ou full FT sur grand n).

    - "medium" : default pour full FT. Ajoute géométrie modérée + couleur modérée.
        Resize(232) + RandomResizedCrop(224, 0.75-1.0) + HFlip
        + Rotation(10°) + ColorJitter(bright=0.2, contrast=0.2, sat=0.1) + normalize.
        Régularisation modérée. Bon compromis biais-variance pour Rakuten
        (catalogue produits : objets souvent centrés, donc petites rotations OK,
        et couleurs parfois mal balancées entre photos → ColorJitter justifié).

    - "hard"   : régularisation forte. À tester si overfit observé sur medium.
        Resize(232) + RandomResizedCrop(224, 0.6-1.0) + HFlip
        + Rotation(20°) + Perspective(0.2, p=0.3)
        + ColorJitter(bright=0.4, contrast=0.4, sat=0.3, hue=0.1)
        + ToTensor + RandomErasing(p=0.25) + normalize.
        Attention : peut introduire du biais (samples trop différents du test).
        Surveiller la courbe val_loss : si elle plafonne haut → trop hard,
        revenir à medium.

    Args:
        level: "soft" | "medium" | "hard"

    Returns:
        transforms.Compose prêt à passer au DataLoader train.
    """
    if level == "soft":
        return transforms.Compose([
            transforms.Resize(232),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            _IMAGENET_NORMALIZE,
        ])

    if level == "medium":
        return transforms.Compose([
            transforms.Resize(232),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            _IMAGENET_NORMALIZE,
        ])

    if level == "hard":
        return transforms.Compose([
            transforms.Resize(232),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            # RandomErasing s'applique APRÈS ToTensor (opère sur tensor)
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
            _IMAGENET_NORMALIZE,
        ])

    raise ValueError(
        f"augmentation_level={level!r} invalide. "
        f"Valeurs supportées : 'soft', 'medium', 'hard'."
    )


# ====================================================================== #
# Lightning module : ResNet18 full FT                                     #
# ====================================================================== #


class _ResNet18FullFTLightning(L.LightningModule):
    """
    ResNet18 avec TOUS les poids défreezés (~11.7M trainable / 11.7M total).

    Différence avec _ResNet50PartialFTLightning :
    - Pas de freeze sélectif : tous les params sont trainable
    - Un seul LR (lr) au lieu de lr_head/lr_backbone
    - embed_dim = 512 (avgpool ResNet18) au lieu de 2048 (ResNet50)
    """

    def __init__(
        self,
        n_classes: int = 27,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        #self.save_hyperparameters()

        # Backbone ResNet18 pré-entraîné ImageNet1K_V1
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Remplace la tête FC : 512 → n_classes
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, n_classes)

        # FULL FT : aucun freeze, tout est trainable
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.lr = lr
        self.weight_decay = weight_decay

        # Pour validation : accumuler predictions et labels par epoch
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward jusqu'à la sortie de l'avgpool, avant fc.
        On s'arrête juste avant fc → (B, 512).
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward complet → logits (B, n_classes)."""
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(preds)
        self._val_labels.append(y)
        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).cpu().numpy()
        labels = torch.cat(self._val_labels).cpu().numpy()
        f1w = f1_score(labels, preds, average="weighted")
        self.log("val_f1_weighted", f1w, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        """
        Un seul groupe d'optimisation (full FT cohérent).
        AdamW : weight decay découplé (Loshchilov & Hutter 2019).
        """
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


# ====================================================================== #
# Wrapper BaseLearner                                                     #
# ====================================================================== #


class ResNet18FullFT(BaseLearner):
    """
    BaseLearner image ResNet18 full FT (tous les poids défreezés).

    Workflow :
    - fit(X, y) : split 80/20 stratifié, fine-tune avec early stopping patience=3
    - extract_embeddings(X) : forward jusqu'à AvgPool → (n, 512)
    - predict_proba(X) : softmax du forward complet → (n, 27)

    Le DataFrame d'entrée doit contenir au minimum les colonnes 'imageid' et
    'productid' (utilisées pour reconstruire le chemin d'image).

    Augmentation :
    - augmentation_level : "soft" | "medium" (défaut) | "hard"
        Pilote le transform train par défaut (cf. _default_train_transform).
        Ignoré si train_transform est explicitement fourni.
    - train_transform : si fourni, override complet de augmentation_level
        (pour ablation custom, ex: TrivialAugment, AutoAugment, etc.).
        Note : non sauvegardé par save_pretrained (lambda non-JSON-picklable),
        à re-fournir manuellement après from_pretrained si besoin.
    - eval_transform : transforms pour val / extract_embeddings / predict_proba.
        Par défaut (None) : Resize(232) + CenterCrop(224) + normalize ImageNet
        (transforms ImageNet standard de torchvision).
    """

    def __init__(
        self,
        image_folder: str | Path,
        n_classes: int = 27,
        batch_size: int = 32,
        max_epochs: int = 15,
        patience: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_workers: int = 4,
        random_state: int = 42,
        precision: str = "bf16-mixed",
        augmentation_level: AugmentationLevel = "medium",
        train_transform=None,
        eval_transform=None,
    ):
        self._image_folder = Path(image_folder)
        self._n_classes = n_classes
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr = lr
        self._weight_decay = weight_decay
        self._num_workers = num_workers
        self._random_state = random_state
        self._precision = precision
        self._augmentation_level = augmentation_level

        # Validation : si train_transform fourni → priorité ; sinon → preset par level
        if train_transform is not None:
            self._train_transform = train_transform
            self._uses_default_train_transform = False
        else:
            self._train_transform = _default_train_transform(augmentation_level)
            self._uses_default_train_transform = True

        self._eval_transform = (
            eval_transform if eval_transform is not None else _default_eval_transform()
        )
        self._uses_default_eval_transform = eval_transform is None

        self.net: _ResNet18FullFTLightning | None = None

    # ------------------------------------------------------------------ #
    # Propriétés BaseLearner                                              #
    # ------------------------------------------------------------------ #

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "image"

    @property
    def embed_dim(self) -> int:
        """Dimension du vecteur extrait par extract_embeddings (512 pour ResNet18)."""
        return 512

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def name(self) -> str:
        return "resnet18_full_ft"

    # ------------------------------------------------------------------ #
    # Helpers internes                                                    #
    # ------------------------------------------------------------------ #

    def _build_image_paths(self, X: pl.DataFrame) -> list[Path]:
        """
        Construit les chemins d'image depuis productid + imageid.
        Format : image_<imageid>_product_<productid>.jpg
        """
        for col in ("imageid", "productid"):
            if col not in X.columns:
                raise ValueError(
                    f"ResNet18FullFT attend une colonne '{col}' dans X. "
                    f"Colonnes disponibles : {X.columns}"
                )
        productids = X["productid"].to_list()
        imageids = X["imageid"].to_list()
        return [
            self._image_folder / f"image_{iid}_product_{pid}.jpg"
            for iid, pid in zip(imageids, productids)
        ]

    def _make_loader(
        self,
        image_paths: list[Path],
        labels: np.ndarray | None,
        transform,
        shuffle: bool,
    ) -> DataLoader:
        ds = _ImageDataset(image_paths, labels, transform)
        return DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self._num_workers > 0,
        )

    # ------------------------------------------------------------------ #
    # API BaseLearner                                                     #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
         **kwargs,
    ) -> "ResNet18FullFT":
        """
        Fine-tune ResNet18 full FT avec early stopping interne.

        Convention M.0 : le DataModule fournit X_train/X_val pré-splittés.
        Si X_val=None, fallback sur un split interne par compatibilité notebook.

        Étapes :
        1. (fallback) Si X_val=None → split interne 80/20 stratifié
        2. Construit les chemins d'image train + val depuis (productid, imageid)
        3. DataLoaders : train avec augmentation, val sans
        4. Fit Lightning avec early stopping sur val_f1_weighted
        """
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val et y_val doivent être tous deux fournis ou tous deux None."
            )

        if X_val is None:
            print(f"[ResNet18FullFT] WARN: pas de X_val fourni, fallback split interne "
                  f"80/20 stratifié seed={self._random_state} (mode notebook).")
            idx = np.arange(len(y_train))
            idx_tr, idx_v = train_test_split(
                idx,
                test_size=0.2,
                stratify=y_train,
                random_state=self._random_state,
            )
            X_val = X_train[idx_v.tolist()]
            y_val = y_train[idx_v]
            X_train = X_train[idx_tr.tolist()]
            y_train = y_train[idx_tr]

        paths_tr = self._build_image_paths(X_train)
        paths_val = self._build_image_paths(X_val)
        if len(paths_tr) != len(y_train):
            raise ValueError(f"len(X_train)={len(paths_tr)} != len(y_train)={len(y_train)}")
        if len(paths_val) != len(y_val):
            raise ValueError(f"len(X_val)={len(paths_val)} != len(y_val)={len(y_val)}")

        print(
            f"[ResNet18FullFT] fit() | n_train={len(y_train)} n_val={len(y_val)} "
            f"| augmentation_level={self._augmentation_level} "
            f"(uses_default_train_transform={self._uses_default_train_transform})"
        )

        # DataLoaders : train avec augmentation, val sans
        train_loader = self._make_loader(paths_tr, y_train, self._train_transform, shuffle=True)
        val_loader = self._make_loader(paths_val, y_val, self._eval_transform, shuffle=False)

        # Modèle Lightning
        self.net = _ResNet18FullFTLightning(
            n_classes=self._n_classes,
            lr=self._lr,
            weight_decay=self._weight_decay,
        )

        # Trainer
        callbacks = [
            EarlyStopping(
                monitor="val_f1_weighted",
                mode="max",
                patience=self._patience,
                verbose=True,
            ),
        ]
        lightning_logger = kwargs.get("lightning_logger", None)
        trainer = L.Trainer(
            max_epochs=self._max_epochs,
            callbacks=callbacks,
            precision=self._precision,
            enable_checkpointing=False,
            logger=lightning_logger if lightning_logger is not None else False,
            log_every_n_steps=50,
        )
        trainer.fit(self.net, train_loader, val_loader)

        self.net.eval()
        return self

    def _forward_in_batches(
        self,
        X: pl.DataFrame,
        return_features: bool,
    ) -> np.ndarray:
        """Forward generic en eval mode, no grad."""
        if self.net is None:
            raise RuntimeError("ResNet18FullFT.fit() doit être appelé avant.")

        image_paths = self._build_image_paths(X)
        loader = self._make_loader(image_paths, labels=None, transform=self._eval_transform, shuffle=False)

        self.net.eval()
        device = next(self.net.parameters()).device

        outputs = []
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x = x.to(device, non_blocking=True)
                if return_features:
                    feat = self.net._features(x)            # (B, 512)
                    outputs.append(feat.cpu().numpy())
                else:
                    logits = self.net(x)
                    probas = F.softmax(logits, dim=1)
                    outputs.append(probas.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """(n, embed_dim=512) — sortie de l'AvgPool, avant la fc."""
        return self._forward_in_batches(X, return_features=True)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 27) — softmax du forward complet."""
        return self._forward_in_batches(X, return_features=False)

    # ------------------------------------------------------------------ #
    # M.4bis — Persistance PyFunc-compatible                              #
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde réversible du ResNet18FullFT dans `path`.

        Crée 2 fichiers :
        - net_state.pt   : state_dict du nn.Module (_ResNet18FullFTLightning)
        - config.json    : hyperparamètres de construction (incluant
                           augmentation_level pour reconstruction exacte
                           du transform train au reload)

        Notes :
        - augmentation_level EST sauvegardé → from_pretrained reconstruit
          le même transform train via _default_train_transform(level).
        - train_transform / eval_transform custom (non-default) NE SONT PAS
          sauvegardés (lambdas non-JSON-picklables) → à re-fournir
          manuellement après from_pretrained si l'utilisateur en a fourni
          de custom. Les flags _uses_default_*_transform sont écrits à titre
          informatif (warning utilisateur au reload si =False).
        - Le contrat from_pretrained → extract_embeddings est testé par
          sanity check round-trip (diff embeddings < 1e-5) UNIQUEMENT pour
          le cas eval_transform par défaut (pas d'augmentation au extract).

        Pré-condition : fit() doit avoir été appelé (sinon net=None → ValueError).
        """
        if self.net is None:
            raise RuntimeError(
                "ResNet18FullFT.save_pretrained() appelé avant fit(). "
                "net est None — rien à sauvegarder."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. state_dict du nn.Module (tous les poids appris)
        torch.save(self.net.state_dict(), path / "net_state.pt")

        # 2. Hyperparams de construction
        config = {
            "image_folder": str(self._image_folder),
            "n_classes": self._n_classes,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "num_workers": self._num_workers,
            "random_state": self._random_state,
            "precision": self._precision,
            "augmentation_level": self._augmentation_level,
            # Flags transforms custom : info-only, warning au reload si =False
            "_uses_default_train_transform": self._uses_default_train_transform,
            "_uses_default_eval_transform": self._uses_default_eval_transform,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        image_folder: str | Path | None = None,
    ) -> "ResNet18FullFT":
        """
        Reconstruit un ResNet18FullFT depuis un dossier écrit par save_pretrained.

        Args:
            path: dossier écrit par save_pretrained, contient
                net_state.pt + config.json.
            image_folder: override du image_folder sauvegardé. CRITIQUE en prod
                car le chemin local au moment du train (ex: /workspace/data/...)
                diffère du chemin à l'inférence (ex: /mnt/disk/..., ou volume
                cloud monté autrement). Si None, le chemin sauvegardé est utilisé
                tel quel — vraisemblablement invalide hors du contexte de train.

        Steps :
        1. Lit config.json
        2. Override image_folder si fourni
        3. Warning si l'utilisateur avait fourni des transforms custom
           (non re-sérialisables)
        4. Instancie ResNet18FullFT(...) → re-crée transform train via
           _default_train_transform(augmentation_level) + eval par défaut
        5. Instancie self.net = _ResNet18FullFTLightning(...) → re-charge
           backbone ResNet18 IMAGENET1K_V1 + override tête fc → 27 classes
        6. Charge state_dict (qui écrase les poids ImageNet par les poids appris)
        7. self.net.eval()

        Le learner retourné est immédiatement utilisable pour
        extract_embeddings / predict_proba.
        """
        path = Path(path)

        # 1. Lire config
        with open(path / "config.json") as f:
            config = json.load(f)

        # 3. Warning si transforms custom (non reconstructibles)
        if not config.get("_uses_default_train_transform", True):
            print(
                f"[ResNet18FullFT.from_pretrained] WARN: le learner avait été "
                f"entraîné avec un train_transform CUSTOM (non re-sérialisable). "
                f"Re-fournir manuellement après from_pretrained si besoin de "
                f"refit. Pour extract_embeddings/predict_proba c'est OK "
                f"(eval_transform suffit)."
            )
        if not config.get("_uses_default_eval_transform", True):
            print(
                f"[ResNet18FullFT.from_pretrained] WARN: eval_transform CUSTOM "
                f"non reconstruit. Embeddings/probas peuvent différer du train. "
                f"Re-fournir manuellement après from_pretrained."
            )

        # Nettoyer les flags (ne sont pas des params __init__)
        config.pop("_uses_default_train_transform", None)
        config.pop("_uses_default_eval_transform", None)

        # 2. Override image_folder si fourni (cas inference cloud / dashboard)
        if image_folder is not None:
            config["image_folder"] = str(image_folder)

        # 4. Instancier le wrapper (le transform train est reconstruit par
        #    __init__ via _default_train_transform(augmentation_level))
        instance = cls(**config)

        # 5. Reconstruire le _ResNet18FullFTLightning
        instance.net = _ResNet18FullFTLightning(
            n_classes=instance._n_classes,
            lr=instance._lr,
            weight_decay=instance._weight_decay,
        )

        # 6. Charger les poids appris (map_location=cpu pour portabilité GPU↔CPU)
        state = torch.load(path / "net_state.pt", map_location="cpu")
        instance.net.load_state_dict(state)
        instance.net.eval()

        return instance
