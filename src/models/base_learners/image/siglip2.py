"""
Siglip2 (Phase 1 — base learner image M4) : encodeur vision SigLIP 2 figé
+ tête de classification linéaire, avec LoRA optionnel sur les derniers blocs.

Calqué sur `text/camembert_lora.py` (LoRA/PEFT, save_pretrained adapter) et
`image/resnet18_full_ft.py` (chargement image depuis imageid, DataLoader,
early stopping sur val_f1_weighted). Même contrat BaseLearner, même cycle de
vie, même intégration BaseLearnerExperiment (cache parquet + cascade
@active_image).

Pourquoi SigLIP 2 et pas MobileCLIP2 :
    Le pipeline extrait et CACHE les embeddings offline (parquet) : la latence
    FLOPs de l'encodeur est amortie. La contrainte dominante est donc la
    QUALITÉ de la représentation figée (elle fixe le plafond du linear probe),
    pas la vitesse on-device → on prend le meilleur transfert extractible
    offline = SigLIP 2 (Tschannen et al. 2025). MobileCLIP2 ne se justifierait
    que pour du scoring temps-réel CPU/edge SANS cache.

Pourquoi image-only (pas la tour texte) :
    SigLIP est un dual-tower. L'encodeur image (ViT) ne voit jamais de texte :
    il mappe pixels -> vecteur. On n'utilise QUE cet encodeur ; la langue de la
    tour texte (penchée anglais) est donc hors-sujet. Le français reste géré par
    CamembertLoRA côté texte.

Discipline frugale (A -> B -> C) :
    - lora_enabled=False (DÉFAUT) : linear probe sur encodeur 100% figé (B1).
      On mesure ÇA d'abord — le domain gap catalogue Rakuten <-> images web est
      faible, le probe suffit souvent.
    - lora_enabled=True (B2) : LoRA (q_proj, v_proj) sur les `lora_last_n_layers`
      derniers blocs du ViT. À activer SEULEMENT si B1 plafonne ET que l'analyse
      de confusion montre des classes fine-grained visuelles qui échouent.

Embedding extrait :
    pooler_output du vision_model (sortie MAP head), AVANT la projection
    contrastive — convention linear probe (la projection est optimisée pour le
    retrieval, pas la classification). Dimension = hidden size du ViT : 768
    (base), 1024 (L), 1152 (So400m). `embed_dim` DOIT matcher le checkpoint
    (fail-fast sinon).

Préprocessing :
    délégué à AutoImageProcessor du checkpoint (resize/normalisation EXACTES
    attendues par le modèle). NE PAS réutiliser la normalisation ImageNet du
    pipeline ResNet.

Dépendances : transformers >= 4.49 (support Siglip2) + peft. Si transformers
    plus ancien : fallback sur le SigLIP original "google/siglip-base-patch16-224"
    via model_name (même API vision_model.pooler_output).
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
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoImageProcessor,
    AutoModel,
    get_cosine_schedule_with_warmup,
)

from src.models.base_learners._base import BaseLearner


# ====================================================================== #
# Dataset : image disque -> pixel_values via AutoImageProcessor           #
# ====================================================================== #


class _Siglip2ImageDataset(Dataset):
    """
    Charge l'image depuis disque (conversion RGB systématique : certaines images
    Rakuten sont RGBA/L), applique le processor HuggingFace DU checkpoint.

    Retourne (pixel_values: (C, H, W), label: long).
    """

    def __init__(self, image_paths, labels, processor):
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.processor = processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        label = int(self.labels[idx]) if self.labels is not None else 0
        return pixel_values, torch.tensor(label, dtype=torch.long)


# ====================================================================== #
# Lightning module : SigLIP2 vision (figé / LoRA) + tête linéaire          #
# ====================================================================== #


class _Siglip2Lightning(L.LightningModule):
    """
    Encodeur vision SigLIP 2 + tête de classification.

    - lora_enabled=False : backbone 100% figé. Les features sont calculées sous
      torch.no_grad() (constantes) -> seule la tête apprend (linear probe pur,
      mémoire minimale). Backbone forcé en eval() à l'entraînement (dropout off
      -> features déterministes).
    - lora_enabled=True : LoRA (q_proj, v_proj) sur les derniers blocs ; le
      gradient remonte dans les adapters + la tête.
    """

    def __init__(
        self,
        model_name: str,
        embed_dim: int,
        n_classes: int,
        lora_enabled: bool,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_last_n_layers: int,
        lr_head: float,
        lr_lora: float,
        weight_decay: float,
        warmup_ratio: float,
        head_dropout: float,
    ):
        super().__init__()
        self.lora_enabled = lora_enabled
        self.lr_head = lr_head
        self.lr_lora = lr_lora
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        # Encodeur vision UNIQUEMENT (la tour texte n'est jamais utilisée).
        # AutoModel(...).vision_model : robuste cross-versions (Siglip / Siglip2).
        # `full` (et donc text_model) est libéré après __init__ (seul vision_model
        # reste référencé par self.backbone).
        full = AutoModel.from_pretrained(model_name)
        self.backbone = full.vision_model

        # Gel total du backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # LoRA optionnel sur les derniers blocs du ViT
        if lora_enabled:
            n_layers = len(self.backbone.encoder.layers)
            start = max(0, n_layers - lora_last_n_layers)
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=list(range(start, n_layers)),
                layers_pattern="layers",
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        # Tête linéaire (linear probe). head_dropout=0 -> probe pur.
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, n_classes),
        )

        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, embed_dim) -- pooler_output du vision_model (MAP head)."""
        if self.lora_enabled:
            out = self.backbone(pixel_values=pixel_values)
        else:
            # Frozen : features constantes -> no_grad (mémoire minimale).
            # La tête reçoit quand même son gradient (entrée traitée en constante).
            with torch.no_grad():
                out = self.backbone(pixel_values=pixel_values)
        pooled = out.pooler_output
        if pooled is None:  # fallback robustesse cross-checkpoints
            pooled = out.last_hidden_state.mean(dim=1)
        return pooled

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward complet -> logits (B, n_classes)."""
        return self.head(self._features(pixel_values))

    def on_train_epoch_start(self):
        # Backbone figé -> eval() (dropout off, features déterministes).
        # En mode LoRA on laisse train() pour garder lora_dropout actif.
        if not self.lora_enabled:
            self.backbone.eval()

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
        head_params = [
            p for n, p in self.named_parameters()
            if n.startswith("head.") and p.requires_grad
        ]
        lora_params = [
            p for n, p in self.named_parameters()
            if not n.startswith("head.") and p.requires_grad
        ]

        groups = [{"params": head_params, "lr": self.lr_head}]
        if lora_params:  # vide en mode frozen
            groups.append({"params": lora_params, "lr": self.lr_lora})

        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ====================================================================== #
# Wrapper BaseLearner                                                     #
# ====================================================================== #


class Siglip2(BaseLearner):
    """
    BaseLearner image SigLIP 2 (encodeur vision figé + tête linéaire, LoRA opt.).

    Workflow :
    - fit(X, y) : split 80/20 (fallback si X_val=None), early stopping patience
      sur val_f1_weighted.
    - extract_embeddings(X) : (n, embed_dim) -- pooler_output du vision_model.
    - predict_proba(X) : (n, 27) -- softmax du forward complet.

    X doit contenir 'imageid' et 'productid' (reconstruction du chemin image,
    format image_<imageid>_product_<productid>.jpg).

    Note M3 β (futur) : une méthode extract_patch_tokens(X) -> (n, n_patches,
    embed_dim) pourra être ajoutée (miroir de ResNet18FullFT.extract_feature_map)
    pour alimenter la fusion par attention. NON implémentée ici : hors scope du
    base learner destiné au stack.
    """

    def __init__(
        self,
        image_folder,
        model_name: str = "google/siglip2-base-patch16-224",
        embed_dim: int = 768,
        n_classes: int = 27,
        lora_enabled: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_last_n_layers: int = 2,
        batch_size: int = 64,
        max_epochs: int = 15,
        patience: int = 3,
        lr_head: float = 1e-3,
        lr_lora: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        head_dropout: float = 0.0,
        num_workers: int = 4,
        random_state: int = 42,
        precision: str = "bf16-mixed",
    ):
        self._image_folder = Path(image_folder)
        self._model_name = model_name
        self._embed_dim = embed_dim
        self._n_classes = n_classes
        self._lora_enabled = lora_enabled
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._lora_last_n_layers = lora_last_n_layers
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr_head = lr_head
        self._lr_lora = lr_lora
        self._weight_decay = weight_decay
        self._warmup_ratio = warmup_ratio
        self._head_dropout = head_dropout
        self._num_workers = num_workers
        self._random_state = random_state
        self._precision = precision

        self._processor = None  # lazy (AutoImageProcessor)
        self.net: _Siglip2Lightning | None = None

    # ------------------------------------------------------------------ #
    # Propriétés BaseLearner                                              #
    # ------------------------------------------------------------------ #

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "image"

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def name(self) -> str:
        return "siglip2"

    # ------------------------------------------------------------------ #
    # Helpers internes                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_processor(self):
        if self._processor is None:
            self._processor = AutoImageProcessor.from_pretrained(self._model_name)
        return self._processor

    def _build_image_paths(self, X: pl.DataFrame) -> list[Path]:
        for col in ("imageid", "productid"):
            if col not in X.columns:
                raise ValueError(
                    f"Siglip2 attend une colonne '{col}' dans X. "
                    f"Colonnes disponibles : {X.columns}"
                )
        productids = X["productid"].to_list()
        imageids = X["imageid"].to_list()
        return [
            self._image_folder / f"image_{iid}_product_{pid}.jpg"
            for iid, pid in zip(imageids, productids)
        ]

    def _make_loader(self, image_paths, labels, shuffle: bool) -> DataLoader:
        ds = _Siglip2ImageDataset(image_paths, labels, self._ensure_processor())
        return DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self._num_workers > 0,
        )

    def _build_net(self) -> _Siglip2Lightning:
        return _Siglip2Lightning(
            model_name=self._model_name,
            embed_dim=self._embed_dim,
            n_classes=self._n_classes,
            lora_enabled=self._lora_enabled,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            lora_last_n_layers=self._lora_last_n_layers,
            lr_head=self._lr_head,
            lr_lora=self._lr_lora,
            weight_decay=self._weight_decay,
            warmup_ratio=self._warmup_ratio,
            head_dropout=self._head_dropout,
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
    ) -> "Siglip2":
        """
        Linear probe (ou LoRA) sur SigLIP 2 avec early stopping interne.

        Convention M.0 : le DataModule fournit X_train/X_val pré-splittés.
        Si X_val=None, fallback split interne 80/20 stratifié (mode notebook).
        """
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val et y_val doivent être tous deux fournis ou tous deux None."
            )

        if X_val is None:
            print(f"[Siglip2] WARN: pas de X_val fourni, fallback split interne "
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
            f"[Siglip2] fit() | n_train={len(y_train)} n_val={len(y_val)} "
            f"| model={self._model_name} | lora_enabled={self._lora_enabled} "
            f"(last_n={self._lora_last_n_layers})"
        )

        train_loader = self._make_loader(paths_tr, y_train, shuffle=True)
        val_loader = self._make_loader(paths_val, y_val, shuffle=False)

        self.net = self._build_net()

        n_trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.net.parameters())
        print(
            f"[Siglip2] Trainable: {n_trainable:,} / {n_total:,} "
            f"({100.0 * n_trainable / n_total:.3f}%)"
        )

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

    def _forward_in_batches(self, X: pl.DataFrame, return_features: bool) -> np.ndarray:
        """Forward générique en eval mode, no grad."""
        if self.net is None:
            raise RuntimeError(
                "Siglip2.fit() ou from_pretrained() doit être appelé avant."
            )
        image_paths = self._build_image_paths(X)
        loader = self._make_loader(image_paths, labels=None, shuffle=False)

        # FIX : Lightning peut avoir remis le modèle sur CPU après fit().
        # On force explicitement le device au lieu de le déduire des params.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.net.eval()
        use_amp = device.type == "cuda"

        outputs = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    out = self.net._features(x) if return_features else F.softmax(self.net(x), dim=1)
                outputs.append(out.float().cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """(n, embed_dim) -- pooler_output du vision_model, avant la tête."""
        feats = self._forward_in_batches(X, return_features=True)
        if feats.shape[1] != self._embed_dim:
            raise ValueError(
                f"[Siglip2] embed_dim={self._embed_dim} déclaré mais le checkpoint "
                f"'{self._model_name}' produit {feats.shape[1]}. Corriger embed_dim "
                f"dans la config (768 base / 1024 L / 1152 So400m)."
            )
        return feats

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 27) -- softmax du forward complet."""
        return self._forward_in_batches(X, return_features=False)

    # ------------------------------------------------------------------ #
    # M.4bis — Persistance PyFunc-compatible                              #
    # ------------------------------------------------------------------ #

    def _config_dict(self) -> dict:
        return {
            "image_folder": str(self._image_folder),
            "model_name": self._model_name,
            "embed_dim": self._embed_dim,
            "n_classes": self._n_classes,
            "lora_enabled": self._lora_enabled,
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "lora_dropout": self._lora_dropout,
            "lora_last_n_layers": self._lora_last_n_layers,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "lr_head": self._lr_head,
            "lr_lora": self._lr_lora,
            "weight_decay": self._weight_decay,
            "warmup_ratio": self._warmup_ratio,
            "head_dropout": self._head_dropout,
            "num_workers": self._num_workers,
            "random_state": self._random_state,
            "precision": self._precision,
        }

    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde réversible dans `path`.

        On NE stocke PAS les poids figés du backbone SigLIP (re-téléchargés
        depuis le Hub via model_name) :
          - mode frozen : seule la tête est apprise -> head_state.
          - mode LoRA   : head_state + adapter_state (poids LoRA uniquement).
        """
        if self.net is None:
            raise RuntimeError("Siglip2.fit() doit être appelé avant save_pretrained.")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self._config_dict(), f, indent=2)

        state = {"head_state": self.net.head.state_dict()}
        if self._lora_enabled:
            state["adapter_state"] = get_peft_model_state_dict(self.net.backbone)
        torch.save(state, path / "net_state.pt")

    @classmethod
    def from_pretrained(cls, path: str | Path, image_folder=None) -> "Siglip2":
        """
        Inverse exact de save_pretrained.

        Args:
            path: dossier écrit par save_pretrained (config.json + net_state.pt).
            image_folder: override du chemin images. CRITIQUE en prod/cloud : le
                chemin du train (ex: /workspace/data/...) diffère de l'inférence.
                Si None, le chemin sauvegardé est utilisé tel quel.

        Steps :
        1. Lit config.json
        2. Override image_folder si fourni
        3. Instancie Siglip2(**config) -> reconstruit le Lightning module
           (re-télécharge le backbone SigLIP + LoRA fraîche si enabled + tête random)
        4. Charge head_state (+ adapter_state si LoRA)
        5. eval()
        """
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        if image_folder is not None:
            config["image_folder"] = str(image_folder)

        instance = cls(**config)
        instance.net = instance._build_net()

        saved = torch.load(path / "net_state.pt", map_location="cpu")
        if instance._lora_enabled:
            set_peft_model_state_dict(instance.net.backbone, saved["adapter_state"])
        instance.net.head.load_state_dict(saved["head_state"])
        instance.net.eval()
        return instance
