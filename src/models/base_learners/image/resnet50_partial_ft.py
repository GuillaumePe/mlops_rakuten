"""
ResNet50PartialFT (Phase 1) : BaseLearner image pour M2.2.

Fine-tune frugal de ResNet50 pré-entraîné ImageNet sur la classification Rakuten.
On défreeze les 27 derniers layers (layer3 + layer4 + nouvelle tête classification),
soit ~14M params trainable sur ~35M total.

Justification du fine-tuning partiel :
- Les premières couches (stem + layer1 + layer2) capturent des features bas-niveau
  (textures, contours, formes simples) qui sont génériques et transférables d'ImageNet
  à Rakuten sans modification.
- Les couches profondes (layer3 + layer4) capturent du sémantique haut-niveau qui doit
  s'adapter au domaine spécifique des produits e-commerce. C'est là que le gradient doit
  remonter pour spécialiser le réseau.
- LR différentiels : lr_head=1e-3 (apprend de zéro), lr_backbone=1e-5 (ajustement fin
  des poids ImageNet pour éviter de "casser" les features pré-entraînées).

Augmentation : injectable via train_transform / eval_transform au constructeur.
Par défaut (None) : transforms "soft" alignés avec le benchmark Rakuten (RandomResizedCrop
léger + HFlip). Pour expérimenter des augmentations plus aggressives (M2.1, ablations),
passer un train_transform custom.

extract_embeddings(X) retourne (n, 2048) : sortie de l'AdaptiveAvgPool2d, avant la
nouvelle tête de classification. Utilisé en aval par K-Fold OOF LogReg du StackingLGBM
pour générer les probas non-fuitées (cohérent avec TextCNN et M2 baseline).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
import json
import lightning as L
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


# ====================================================================== #
# Dataset : charge image depuis disque, applique transforms              #
# ====================================================================== #


class _ImageDataset(Dataset):
    """
    Dataset minimal : chemins d'image + labels → (tensor, label).

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
# Helper : transforms par défaut                                          #
# ====================================================================== #


def _default_eval_transform():
    """Transforms ImageNet standard : resize 232 + center crop 224 + normalize."""
    return models.ResNet50_Weights.IMAGENET1K_V2.transforms()


def _default_train_transform():
    """
    Augmentation 'soft' alignée avec le benchmark Rakuten.

    Resize(232) + RandomResizedCrop(224, scale=0.85-1.0) + HFlip + normalize ImageNet.
    Pas de color jitter ni de rotation, on reste fidèle à la reproduction benchmark.
    """
    return transforms.Compose([
        transforms.Resize(232),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ====================================================================== #
# Lightning module : ResNet50 partial FT                                  #
# ====================================================================== #


class _ResNet50PartialFTLightning(L.LightningModule):
    """
    ResNet50 avec layer3+layer4+head défreezés (~14M trainable / ~35M total).

    Sépare deux groupes d'optimisation :
    - backbone (layer3 + layer4) : LR=1e-5 (ajustement fin)
    - head (nouvelle Linear) : LR=1e-3 (apprend de zéro)
    """

    def __init__(
        self,
        n_classes: int = 27,
        lr_head: float = 1e-3,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
#        self.save_hyperparameters()

        # Backbone pré-entraîné ImageNet
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(weights=weights)

        # Geler stem + layer1 + layer2
        for name, param in backbone.named_parameters():
            if name.startswith(("conv1", "bn1", "layer1", "layer2")):
                param.requires_grad = False
            # layer3, layer4, fc restent trainable

        # Remplacer la tête FC ImageNet (1000 classes) par une nouvelle pour 27 classes
        in_features = backbone.fc.in_features  # 2048 pour ResNet50
        backbone.fc = nn.Linear(in_features, n_classes)
        self.backbone = backbone

        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.embed_dim = in_features  # exposé pour le wrapper

        # Validation accumulators
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward jusqu'à l'AvgPool, AVANT la nouvelle tête fc.

        Reproduit la séquence interne de torchvision.models.resnet : conv1 → bn1 →
        relu → maxpool → layer1..4 → avgpool → flatten. On s'arrête juste avant fc.
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
        return torch.flatten(x, 1)  # (B, 2048)

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
        Deux groupes de params : backbone (LR faible, ajustement fin) et head
        (LR forte, apprend de zéro). Évite de casser les features pré-entraînées.
        """
        head_params = list(self.backbone.fc.parameters())
        head_param_ids = {id(p) for p in head_params}
        backbone_params = [
            p for p in self.backbone.parameters()
            if p.requires_grad and id(p) not in head_param_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay)
        # Scheduler sur lr_head uniquement (param group 0)
        # lr_backbone reste fixe à 1e-5 (déjà dans son basin ImageNet)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,       # divise par 2
            patience=2,       # après 2 epochs sans amélioration val_loss
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",    # métrique surveillée
            "interval": "epoch",
            "frequency": 1,
            },
        }


# ====================================================================== #
# Wrapper BaseLearner                                                     #
# ====================================================================== #


class ResNet50PartialFT(BaseLearner):
    """
    BaseLearner image ResNet50 partial FT (layer3+layer4+head défreezés).

    Workflow :
    - fit(X, y) : split 80/20 stratifié, fine-tune avec early stopping patience=3
    - extract_embeddings(X) : forward jusqu'à AvgPool → (n, 2048)
    - predict_proba(X) : softmax du forward complet → (n, 27)

    Le DataFrame d'entrée doit contenir au minimum les colonnes 'imageid' et
    'productid' (utilisées pour reconstruire le chemin d'image).

    Augmentation :
    - train_transform : transforms appliqués pendant fit() sur le split train.
      Par défaut (None) : Resize(232) + RandomResizedCrop(224, scale=0.85-1) + HFlip
      + normalize ImageNet (alignement benchmark Rakuten, augmentation soft).
    - eval_transform : transforms pour val / extract_embeddings / predict_proba.
      Par défaut (None) : Resize(232) + CenterCrop(224) + normalize ImageNet
      (transforms ImageNet standard de torchvision).
    Pour expérimenter une augmentation plus aggressive (cf. M2.1), passer un
    torchvision.transforms.Compose custom au constructeur.
    """

    def __init__(
        self,
        image_folder: str | Path,
        n_classes: int = 27,
        batch_size: int = 32,
        max_epochs: int = 15,
        patience: int = 3,
        lr_head: float = 1e-3,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        num_workers: int = 4,
        random_state: int = 42,
        precision: str = "bf16-mixed",
        train_transform=None,   # torchvision.transforms.Compose (injectable)
        eval_transform=None,    # torchvision.transforms.Compose (injectable)
    ):
        self._image_folder = Path(image_folder)
        self._n_classes = n_classes
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr_head = lr_head
        self._lr_backbone = lr_backbone
        self._weight_decay = weight_decay
        self._num_workers = num_workers
        self._random_state = random_state
        self._precision = precision

        self._train_transform = train_transform if train_transform is not None else _default_train_transform()
        self._eval_transform = eval_transform if eval_transform is not None else _default_eval_transform()

        self.net: _ResNet50PartialFTLightning | None = None

    # ------------------------------------------------------------------ #
    # Propriétés BaseLearner                                              #
    # ------------------------------------------------------------------ #

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "image"

    @property
    def embed_dim(self) -> int:
        """Dimension du vecteur extrait par extract_embeddings (2048 pour ResNet50)."""
        return 2048

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def name(self) -> str:
        return "resnet50_partial_ft"

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
                    f"ResNet50PartialFT attend une colonne '{col}' dans X. "
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
            ) -> "ResNet50PartialFT":
        """
        Fine-tune ResNet50 partial avec early stopping interne.

        Convention M.0 : le DataModule fournit X_train/X_val pré-splittés.
        Si X_val=None, fallback sur un split interne par compatibilité notebook.

        Étapes :
        1. (fallback) Si X_val=None → split interne 80/20 stratifié
        2. Construit les chemins d'image train + val depuis (productid, imageid)
        3. DataLoaders : train avec augmentation, val sans
        4. Fit Lightning avec early stopping sur val_f1_weighted
		kwargs reconnus (convention M.0a) :
            lightning_logger: pl.loggers.Logger | None
                Si fourni, utilisé par L.Trainer pour logger les métriques par
                epoch (train_loss, val_loss, val_f1_weighted, ...) vers MLflow
                ou autre backend. Sinon, logger=False (pas de logging par epoch).
                Permet à BaseLearnerExperiment de partager son run MLflow actif
                avec le Trainer Lightning interne.        
        """
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val et y_val doivent être tous deux fournis ou tous deux None."
            )

        if X_val is None:
            print(f"[ResNet50PartialFT] WARN: pas de X_val fourni, fallback split interne "
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

        # DataLoaders : train avec augmentation, val sans
        train_loader = self._make_loader(paths_tr, y_train, self._train_transform, shuffle=True)
        val_loader = self._make_loader(paths_val, y_val, self._eval_transform, shuffle=False)

        # Modèle Lightning
        self.net = _ResNet50PartialFTLightning(
            n_classes=self._n_classes,
            lr_head=self._lr_head,
            lr_backbone=self._lr_backbone,
            weight_decay=self._weight_decay,
        )

        # Trainer
        callbacks = [
            EarlyStopping(
#                monitor="val_loss",
#                mode="min",
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
            raise RuntimeError("ResNet50PartialFT.fit() doit être appelé avant.")

        image_paths = self._build_image_paths(X)
        loader = self._make_loader(
            image_paths, labels=None, transform=self._eval_transform, shuffle=False,
        )

        self.net.eval()
        device = next(self.net.parameters()).device

        outputs = []
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x = x.to(device, non_blocking=True)
                if return_features:
                    feat = self.net._features(x)            # (B, 2048)
                    outputs.append(feat.cpu().numpy())
                else:
                    logits = self.net(x)                    # (B, 27)
                    probas = F.softmax(logits, dim=1)
                    outputs.append(probas.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 2048) — sortie de AvgPool, avant la tête fc."""
        return self._forward_in_batches(X, return_features=True)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 27) — softmax du forward complet."""
        return self._forward_in_batches(X, return_features=False)
        # ------------------------------------------------------------------ #
    # M.4bis — Persistance PyFunc-compatible                              #
    # ------------------------------------------------------------------ #
 
    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde réversible du ResNet50PartialFT dans `path`.
 
        Crée 2 fichiers :
        - net_state.pt   : state_dict du nn.Module (_ResNet50PartialFTLightning),
                           contient TOUS les poids (gelés inclus) car nn.Module.state_dict()
                           ne distingue pas requires_grad=True/False — il sauve l'état
                           paramétrique complet.
        - config.json    : hyperparamètres de construction.
 
        Subtilités :
        - image_folder est sauvé mais doit être OVERRIDABLE au reload (le chemin
          train != chemin inference, ex: /workspace/data au pod, /mnt/disk au dashboard).
        - train_transform / eval_transform NON sauvegardés (callables torchvision
          non-JSON-serializables). On assume les défauts au reload. Si l'utilisateur
          a fourni des transforms custom, il devra les re-fournir au constructeur
          après from_pretrained, ou via une surcharge ad hoc.
        - Le contrat from_pretrained → extract_embeddings est testé par sanity
          check round-trip (diff embeddings < 1e-5).
 
        Pré-condition : fit() doit avoir été appelé (sinon net=None → ValueError).
        """
        if self.net is None:
            raise RuntimeError(
                "ResNet50PartialFT.save_pretrained() appelé avant fit(). "
                "net est None — rien à sauvegarder."
            )
 
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
 
        # 1. state_dict du nn.Module (tous les poids, gelés inclus)
        torch.save(self.net.state_dict(), path / "net_state.pt")
 
        # 2. Hyperparams de construction
        #    Note : on flag que les transforms sont les défauts au save.
        #    Si l'utilisateur veut des transforms custom au reload, il doit
        #    passer train_transform/eval_transform après from_pretrained.
        custom_train = self._train_transform is not _default_train_transform()
        custom_eval = self._eval_transform is not _default_eval_transform()
 
        config = {
            "image_folder": str(self._image_folder),
            "n_classes": self._n_classes,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "lr_head": self._lr_head,
            "lr_backbone": self._lr_backbone,
            "weight_decay": self._weight_decay,
            "num_workers": self._num_workers,
            "random_state": self._random_state,
            "precision": self._precision,
            # Flags transforms : info-only, pas exploité au reload
            "_uses_default_train_transform": not custom_train,
            "_uses_default_eval_transform": not custom_eval,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
 
    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        image_folder: str | Path | None = None,
    ) -> "ResNet50PartialFT":
        """
        Reconstruit un ResNet50PartialFT depuis un dossier écrit par save_pretrained.
 
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
        3. Instancie ResNet50PartialFT(...) → re-crée transforms par défaut
        4. Instancie self.net = _ResNet50PartialFTLightning(...) → re-charge
           backbone ResNet50 IMAGENET1K_V2 + override tête fc → 27 classes
        5. Charge state_dict (qui écrase les poids ImageNet par les poids appris)
        6. self.net.eval()
 
        Le learner retourné est immédiatement utilisable pour
        extract_embeddings / predict_proba.
        """
        path = Path(path)
 
        # 1. Lire config
        with open(path / "config.json") as f:
            config = json.load(f)
 
        # Nettoyer les flags purement informatifs (ne sont pas des params __init__)
        config.pop("_uses_default_train_transform", None)
        config.pop("_uses_default_eval_transform", None)
 
        # 2. Override image_folder si fourni (cas inference cloud / dashboard)
        if image_folder is not None:
            config["image_folder"] = str(image_folder)
 
        # 3. Instancier le wrapper (les transforms par défaut sont reconstruits
        #    par le constructeur via _default_*_transform() — alignement
        #    benchmark Rakuten garanti)
        instance = cls(**config)
 
        # 4. Reconstruire le _ResNet50PartialFTLightning :
        #    a. Charge backbone ResNet50 IMAGENET1K_V2 (poids ImageNet)
        #    b. Remplace fc → Linear(2048, n_classes)
        #    Ces poids ImageNet seront immédiatement écrasés par notre state_dict
        #    appris à l'étape 5, mais cette étape est nécessaire pour avoir la
        #    bonne architecture (notamment la tête fc adaptée).
        instance.net = _ResNet50PartialFTLightning(
            n_classes=instance._n_classes,
            lr_head=instance._lr_head,
            lr_backbone=instance._lr_backbone,
            weight_decay=instance._weight_decay,
        )
 
        # 5. Charger les poids appris (map_location=cpu pour portabilité GPU↔CPU)
        state = torch.load(path / "net_state.pt", map_location="cpu")
        instance.net.load_state_dict(state)
        instance.net.eval()
 
        return instance

    
