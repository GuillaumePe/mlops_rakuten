"""
O.3 — M3AttentionFusion : LightningModule de fusion par attention.

Compose deux base learners frozen (texte + image) avec un
AttentionFusionModule (O.2). Le training_step forward à travers
les encoders gelés puis entraîne uniquement le bloc de fusion.

Séparation des responsabilités :
    - MultimodalDataset (datamodule/datasets.py) : preprocessing, I/O
    - AttentionFusionModule (O.2) : nn.Module pur, reçoit des tensors
    - M3AttentionFusion (ici) : orchestration Lightning (training loop,
      optimizer, scheduler, loss, métriques de monitoring)
    - LightningExperiment (O.4) : orchestration MLflow (run, log, promote)
    - runner.py (O.5) : charge les base learners, instancie M3, lance

Le M3 ne connaît ni MLflow ni le runner ni le Dataset.
Il reçoit des tensors dans training_step, c'est tout.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from src.models.fusion.attention_fusion import AttentionFusionModule


class M3AttentionFusion(L.LightningModule):
    """
    LightningModule de fusion par attention cross-modale.

    Reçoit deux modules PyTorch frozen (les .net des base learners)
    et un AttentionFusionModule entraînable.

    Args:
        text_net: module Lightning interne du text encoder (ex:
            _CamembertLoRALightning). Déjà en .eval(), requires_grad=False.
        image_net: module Lightning interne de l'image encoder (ex:
            _ResNet18FullFTLightning). Déjà en .eval(), requires_grad=False.
        d_text: dimension des token embeddings du text encoder.
        d_image: dimension des feature map patches de l'image encoder.
        config: dict de configuration :
            - d_model (int, default 512)
            - n_heads (int, default 8)
            - n_layers (int, default 2)
            - dim_ff_factor (int, default 4)
            - dropout (float, default 0.3)
            - n_classes (int, default 27)
            - lr (float, default 5e-4)
            - weight_decay (float, default 0.01)
            - warmup_ratio (float, default 0.1)
            - class_weights (list[float] | None, default None)
    """

    def __init__(
        self,
        text_net: L.LightningModule,
        image_net: L.LightningModule,
        d_text: int,
        d_image: int,
        config: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["text_net", "image_net"])

        # --- Encoders frozen ---
        self.text_net = text_net
        self.text_net.eval()
        self.text_net.requires_grad_(False)

        self.image_net = image_net
        self.image_net.eval()
        self.image_net.requires_grad_(False)

        # --- Module de fusion (entraînable) ---
        self.fusion = AttentionFusionModule(
            d_text=d_text,
            d_image=d_image,
            d_model=config.get("d_model", 512),
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 2),
            dim_ff_factor=config.get("dim_ff_factor", 4),
            dropout=config.get("dropout", 0.3),
            n_classes=config.get("n_classes", 27),
        )

        # --- Loss ---
        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

        # --- Optimizer config ---
        self.lr = config.get("lr", 5e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)
        
        # Accumulation validation pour F1 epoch-level
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def _forward_encoders(
        self, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward des encoders frozen → token embeddings + feature map.

        torch.no_grad() économise ~60% de VRAM sur cette partie
        (pas de stockage des activations intermédiaires pour le backward).
        """
        with torch.no_grad():
            text_tokens, text_mask = self.text_net._token_level_features(batch["input_ids"], batch["attention_mask"])
            image_patches = self.image_net._spatial_feature_map(batch["image"])
        return text_tokens, text_mask, image_patches

    def forward(self, batch: dict) -> torch.Tensor:
        """Retourne les logits (B, n_classes)."""
        text_tokens, text_mask, image_patches = self._forward_encoders(batch)
        return self.fusion(text_tokens, text_mask, image_patches)

    # ------------------------------------------------------------------ #
    # Training / validation steps                                          #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.forward(batch)
        loss = F.cross_entropy(
            logits, batch["label"], weight=self.class_weights
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["label"]).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.forward(batch)
        loss = F.cross_entropy(
            logits, batch["label"], weight=self.class_weights
        )
        preds = logits.argmax(dim=-1)
        acc = (preds == batch["label"]).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True, sync_dist=True)
        # Accumulation pour F1 epoch-level (calculé dans on_validation_epoch_end)
        self._val_preds.append(preds.cpu())
        self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self) -> None:
        """
        Calcule le F1 weighted sur l'ensemble de la validation epoch.
 
        Même pattern que les BaseLearners (CamembertLoRA, ResNet18FullFT).
        Loggé dans MLflow via le MLFlowLogger du Trainer → courbe
        val/f1_weighted visible par epoch dans l'UI.
        """
        if not self._val_preds:
            return
        all_preds = torch.cat(self._val_preds).numpy()
        all_labels = torch.cat(self._val_labels).numpy()
        from sklearn.metrics import f1_score
        f1_w = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        self.log("val/f1_weighted", f1_w, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler                                                #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> dict:
        """
        AdamW sur les paramètres du fusion module uniquement.

        Cosine annealing avec warmup linéaire (10% des steps).
        Justification du warmup : stabilise Adam dans les premières
        itérations où le 2e moment (v_t) est biaisé vers 0. Sans warmup,
        les updates initiales sont disproportionnellement grandes
        (lr / (√v_t + ε) diverge quand v_t ≈ 0).
        """
        optimizer = torch.optim.AdamW(
            self.fusion.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------ #
    # Empêche les encoders frozen de passer en train mode                  #
    # ------------------------------------------------------------------ #

    def on_train_epoch_start(self) -> None:
        """
        Garde les encoders en eval mode même quand Lightning appelle
        model.train() au début de chaque epoch.

        Sans ce hook, les BatchNorm du ResNet passeraient en mode train
        (statistiques batch au lieu de running stats) et le dropout du
        CamemBERT serait réactivé — deux comportements non souhaités
        pour des encoders frozen.
        """
        self.text_net.eval()
        self.image_net.eval()
