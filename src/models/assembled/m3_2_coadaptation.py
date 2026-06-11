"""
M3.2 — Fusion par attention avec CO-ADAPTATION DoRA des encodeurs.

Variante de M3AttentionFusion (FICHIER SÉPARÉ, architecture distincte, modèle
enregistré distinct). Trois leviers vers une *vraie* fusion (vs la
late-fusion-équivalente de M3) :

    B.1 — pooling par attention          (via AttentionFusionModuleV2)
    B.2 — injection tabulaire dans la tête (via d_tab > 0)
    B.3 — CO-ADAPTATION : DoRA bas-rang sur les `dora_last_n` dernières couches
          d'attention de CHAQUE encodeur ; le reste reste gelé.

--- Pourquoi co-adaptation (B.3, le vrai levier) ---
Avec des encodeurs 100 % gelés (M3), la cross-attention n'a AUCUN gradient pour
aligner les représentations des deux modalités → l'optimum est proche d'une
combinaison tardive. Donner un gradient bas-rang aux dernières couches élargit
l'espace des solutions vers une fusion early/mid.

--- Pourquoi DoRA et pas LoRA ---
À très bas rang (r=4, régime co-adaptation), DoRA (décomposition
magnitude/direction, Liu et al. 2024) se rapproche davantage du full
fine-tuning que LoRA — exactement le régime ici (capacité minime débloquée).
Drop-in PEFT via use_dora=True, mémoire ~identique, compute légèrement supérieur.

--- Co-adaptation du texte (subtilité) ---
CamemBERT arrive avec sa LoRA STANDALONE (entraînée pour la tâche texte seule,
sur toutes les couches). On la MERGE dans les poids (merge_and_unload) pour
partir de la meilleure représentation texte, PUIS on injecte une DoRA FRAÎCHE
sur les dernières couches pour la tâche de FUSION. Les deux adaptations sont
ainsi séparées : standalone (mergée, figée) + fusion (DoRA, entraînable).
SigLIP arrive figé (lora_enabled=False) → DoRA fraîche directe sur le ViT.

--- Coût (à assumer) ---
On perd la frugalité "encodeurs figés" de M3 : le gradient remonte dans les
dernières couches → VRAM + temps ↑. Mitigations : rank bas (4), dernières
couches seulement, bf16, gradient checkpointing (option Trainer).

Le M3.2 ne connaît ni MLflow ni le runner ni le Dataset : il reçoit un batch
{input_ids, attention_mask, image, [tabular], label} dans training_step.
"""
from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score

from src.models.fusion.attention_fusion_v2 import AttentionFusionModuleV2


class M32CoAdaptationFusion(L.LightningModule):
    """
    LightningModule de fusion par attention + co-adaptation DoRA.

    Args:
        text_net: _CamembertLoRALightning chargé depuis @active_text (sa LoRA
            standalone sera mergée puis re-DoRA'ée sur les dernières couches).
        image_net: _Siglip2Lightning chargé depuis @active_image (figé →
            DoRA fraîche directe).
        d_text: dim tokens texte (768 CamemBERT).
        d_image: dim patches image (768 SigLIP ViT-B).
        d_tab: dim features tabulaires injectées dans la tête (0 = pas
            d'injection ; doit matcher ce que le batch fournit).
        config: dict :
            # --- fusion (AttentionFusionModuleV2) ---
            d_model, n_heads, n_layers, dim_ff_factor, dropout, n_classes,
            pooling ("attention"|"mean")
            # --- co-adaptation DoRA ---
            dora_last_n (int, default 1)   # nb de dernières couches adaptées
            dora_rank (int, default 4)
            dora_alpha (int, default 8)
            dora_dropout (float, default 0.05)
            # --- optimisation ---
            lr (float, default 5e-4)        # fusion
            lr_dora (float, default 1e-4)   # adapters DoRA (bas, près de full-FT)
            weight_decay (float, default 0.01)
            warmup_ratio (float, default 0.1)
            class_weights (list[float] | None)
    """

    def __init__(
        self,
        text_net: L.LightningModule,
        image_net: L.LightningModule,
        d_text: int,
        d_image: int,
        d_tab: int,
        config: dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["text_net", "image_net"])

        self.text_net = text_net
        self.image_net = image_net

        # --- Co-adaptation DoRA (B.3) ---
        self.dora_last_n = config.get("dora_last_n", 1)
        self.dora_rank = config.get("dora_rank", 4)
        self.dora_alpha = config.get("dora_alpha", 8)
        self.dora_dropout = config.get("dora_dropout", 0.05)

        # 1. Geler les têtes standalone (M3 utilise les features encodeur, pas
        #    les têtes de classification des base learners).
        for p in self.text_net.head.parameters():
            p.requires_grad = False
        for p in self.image_net.head.parameters():
            p.requires_grad = False

        # 2. Injecter DoRA sur les dernières couches d'attention de chaque encodeur.
        self._inject_dora_text()
        self._inject_dora_image()

        # 3. Module de fusion (B.1 pooling attention + B.2 tabulaire).
        self.fusion = AttentionFusionModuleV2(
            d_text=d_text,
            d_image=d_image,
            d_tab=d_tab,
            d_model=config.get("d_model", 512),
            n_heads=config.get("n_heads", 8),
            n_layers=config.get("n_layers", 2),
            dim_ff_factor=config.get("dim_ff_factor", 4),
            dropout=config.get("dropout", 0.3),
            n_classes=config.get("n_classes", 27),
            pooling=config.get("pooling", "attention"),
        )

        # --- Loss ---
        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

        # --- Optimisation ---
        self.lr = config.get("lr", 5e-4)
        self.lr_dora = config.get("lr_dora", 1e-4)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)

        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

        # Sanity : log la part entraînable.
        self._log_trainable()

    # ------------------------------------------------------------------ #
    # Injection DoRA                                                       #
    # ------------------------------------------------------------------ #

    def _inject_dora_image(self) -> None:
        """SigLIP figé → DoRA fraîche (q_proj/v_proj) sur les dernières couches ViT."""
        backbone = self.image_net.backbone
        if hasattr(backbone, "merge_and_unload"):
            # Idempotence (reload checkpoint) : si déjà PeftModel, merge avant
            # de réinjecter (le state_dict du checkpoint réécrit ensuite les poids).
            backbone = backbone.merge_and_unload()
        n_layers = len(backbone.encoder.layers)
        start = max(0, n_layers - self.dora_last_n)
        cfg = LoraConfig(
            r=self.dora_rank,
            lora_alpha=self.dora_alpha,
            lora_dropout=self.dora_dropout,
            use_dora=True,
            target_modules=["q_proj", "v_proj"],
            layers_to_transform=list(range(start, n_layers)),
            layers_pattern="layers",
            bias="none",
        )
        self.image_net.backbone = get_peft_model(backbone, cfg)

    def _inject_dora_text(self) -> None:
        """
        CamemBERT : merge la LoRA standalone, puis DoRA fraîche (query/value)
        sur les dernières couches. RoBERTa-style → layers_pattern='layer'.
        """
        backbone = self.text_net.backbone
        if hasattr(backbone, "merge_and_unload"):
            backbone = backbone.merge_and_unload()  # → CamembertModel de base
        n_layers = backbone.config.num_hidden_layers  # 12 (camembert-base)
        start = max(0, n_layers - self.dora_last_n)
        cfg = LoraConfig(
            r=self.dora_rank,
            lora_alpha=self.dora_alpha,
            lora_dropout=self.dora_dropout,
            use_dora=True,
            target_modules=["query", "value"],
            layers_to_transform=list(range(start, n_layers)),
            layers_pattern="layer",
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.text_net.backbone = get_peft_model(backbone, cfg)

    def _trainable_encoder_params(self, net: L.LightningModule) -> list:
        return [p for p in net.parameters() if p.requires_grad]

    def _log_trainable(self) -> None:
        txt = sum(p.numel() for p in self._trainable_encoder_params(self.text_net))
        img = sum(p.numel() for p in self._trainable_encoder_params(self.image_net))
        fus = sum(p.numel() for p in self.fusion.parameters())
        print(
            f"[M3.2] trainable | fusion={fus:,} | dora_text={txt:,} | "
            f"dora_image={img:,} | total={fus + txt + img:,}"
        )
        if txt == 0 or img == 0:
            raise RuntimeError(
                "[M3.2] DoRA n'a produit aucun paramètre entraînable sur un "
                "encodeur (txt={}, img={}). Vérifier target_modules / "
                "layers_pattern / dora_last_n.".format(txt, img)
            )

    # ------------------------------------------------------------------ #
    # Forward (gradient remonte dans les couches DoRA — PAS de no_grad)    #
    # ------------------------------------------------------------------ #

    def _forward_encoders(self, batch: dict):
        text_tokens, text_mask = self.text_net._token_level_features(
            batch["input_ids"], batch["attention_mask"]
        )
        image_patches = self.image_net._spatial_feature_map(batch["image"])
        return text_tokens, text_mask, image_patches

    def forward(self, batch: dict) -> torch.Tensor:
        text_tokens, text_mask, image_patches = self._forward_encoders(batch)
        tabular = batch.get("tabular", None)
        return self.fusion(text_tokens, text_mask, image_patches, tabular)

    # ------------------------------------------------------------------ #
    # Train / val                                                          #
    # ------------------------------------------------------------------ #

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch["label"], weight=self.class_weights)
        acc = (logits.argmax(-1) == batch["label"]).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.forward(batch)
        loss = F.cross_entropy(logits, batch["label"], weight=self.class_weights)
        preds = logits.argmax(-1)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", (preds == batch["label"]).float().mean(),
                 prog_bar=True, sync_dist=True)
        self._val_preds.append(preds.cpu())
        self._val_labels.append(batch["label"].cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds).numpy()
        labels = torch.cat(self._val_labels).numpy()
        f1w = f1_score(labels, preds, average="weighted", zero_division=0)
        self.log("val/f1_weighted", f1w, prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    # ------------------------------------------------------------------ #
    # Optimizer : 3 groupes (fusion / dora_text / dora_image)             #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self) -> dict:
        groups = [{"params": list(self.fusion.parameters()), "lr": self.lr}]
        text_dora = self._trainable_encoder_params(self.text_net)
        image_dora = self._trainable_encoder_params(self.image_net)
        if text_dora:
            groups.append({"params": text_dora, "lr": self.lr_dora})
        if image_dora:
            groups.append({"params": image_dora, "lr": self.lr_dora})

        optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)

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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
