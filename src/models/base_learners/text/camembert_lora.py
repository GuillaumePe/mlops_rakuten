"""
CamembertLoRA : BaseLearner texte pour M2.1 (Bloc N).

Fine-tuning frugal de CamemBERT-base via LoRA (Low-Rank Adaptation).

Justification mathématique LoRA (Hu et al. 2021, "LoRA: Low-Rank Adaptation
of Large Language Models" + Aghajanyan et al. 2020, "Intrinsic Dimensionality") :
- Backbone CamemBERT-base = 110M params, mais hypothèse : l'adaptation à une
  tâche downstream vit dans un sous-espace de rang faible (intrinsic dim).
- Formellement, ΔW ≈ B·A avec A ∈ R^{r×d}, B ∈ R^{d×r}, r << d.
  Avec r=16, d=768 : ΔW se réduit à 2·r·d ≈ 25k params par projection,
  vs 768² ≈ 590k pour full FT → ~25x moins de params.
- Trainable total : ~2.4M (~2% des 110M base) → contrôle variance ↑ overfit ↓.
- Justif stat (biais-variance) : n/d=38k/110M << 1 (mauvais ratio pour full FT).
  LoRA impose structure low-rank → biais ↑ MAIS variance ↓↓ → meilleure
  généralisation sur petit n (Aghajanyan 2020 montre que rank intrinsèque
  des tâches NLP est typiquement < 100, donc r=16 est conservateur-frugal).

Vs alternatives :
- Full FT (110M trainable) : meilleurs résultats sur n grand, mais risque
  overfit sur 38k samples + coût mémoire prohibitif (4x VRAM vs LoRA).
- Frozen + LogReg : 0 params trainables, mais limite biais haut → F1 ≈ 0.78.
  LoRA gagne ~3-5% F1 sur frozen pour ~2x moins de coût que full FT.
- Adapter modules (Houlsby) : ajoute des bottleneck layers — comparable à
  LoRA mais ajoute de la latence inference (les adapters restent inline).
  LoRA peut être *merged* dans les poids du base au déploiement → 0 surcoût.

Architecture :
    CamemBERT-base (gelé, 110M)
        ├─ Self-attention.query → LoRA(r=16) (trainable, ~2.4M)
        ├─ Self-attention.value → LoRA(r=16) (trainable, ~2.4M)
        └─ ...
    → mean pooling (attention-mask-aware) → (B, 768)
    → Linear(768, 27) (trainable, ~20k) → logits

Note : CamemBERT a une archi RoBERTa, donc target_modules = ["query", "value"]
(et pas "q_proj"/"v_proj" qui sont LLaMA-style).

Mean pooling vs [CLS] : RoBERTa (et CamemBERT) n'a pas d'objectif NSP, donc
le [CLS] token n'est pas particulièrement bien entraîné comme représentation
de phrase. Mean pooling masked est plus robuste et standard pour
sentence-transformers / fine-tune downstream.

Workflow :
- fit(X, y) : tokenize, split 80/20 (fallback), train avec early stopping
- extract_embeddings(X) : forward jusqu'au mean pooling → (n, 768)
- predict_proba(X) : softmax du forward complet → (n, 27)

Le DataFrame d'entrée doit contenir la colonne 'text' (concat designation +
description, déjà produite par RakutenLightningDataModule mode raw_for_finetune).

Hyperparamètres de référence (cf. base_learner_camembert_lora.yaml) :
- model_name="camembert-base"     # 110M params, BPE tokenizer
- max_len=128                     # truncate, couvre 95+% des textes Rakuten
- lora_rank=16, lora_alpha=32     # alpha/rank=2 (scaling LoRA standard)
- lora_dropout=0.05               # régul LoRA légère
- lr_lora=5e-4, lr_head=1e-3      # LR différentiels (head=random init)
- weight_decay=0.01               # AdamW standard pour BERT-like
- warmup_ratio=0.1                # warmup 10% des steps (stabilité initiale)
- batch_size=32, max_epochs=10    # GPU manageable
- patience=2                      # early stopping sur val_f1_weighted
- precision="bf16-mixed"          # cohérent benchmark
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
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from src.models.base_learners._base import BaseLearner


# ====================================================================== #
# Dataset : tokenisation à la volée (padding fixe à max_len)              #
# ====================================================================== #


class _CamembertTextDataset(Dataset):
    """
    Dataset texte tokenisé pour CamemBERT.

    Tokenise à la volée dans __getitem__ avec padding fixe à max_len.
    Avantage : pas de pré-tokenisation lourde en mémoire (38k textes × 128
    tokens × 4 bytes ≈ 20 MB, donc on pourrait pré-tokeniser ; on garde
    on-the-fly pour cohérence avec usage en production sur stream de textes).

    Retourne un dict {input_ids, attention_mask, labels} — le default
    torch collate stack chaque clé en batch.
    """

    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        tokenizer,
        max_len: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label = int(self.labels[idx]) if self.labels is not None else 0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ====================================================================== #
# Lightning module : CamemBERT + LoRA + head                              #
# ====================================================================== #


class _CamembertLoRALightning(L.LightningModule):
    """
    CamemBERT (gelé) + LoRA adapters sur (query, value) + classification head.

    Architecture :
    - self.backbone : PeftModel wrappant CamemBERT-base. Tous les poids de
      base sont gelés ; seuls les adapters LoRA (lora_A, lora_B) sont
      trainable.
    - self.head : Linear(768, n_classes), trainable.

    Mean pooling : on récupère outputs.last_hidden_state, on masque les
    tokens de padding via attention_mask, on moyenne sur la dim séquence.

    Optimiseur : AdamW avec deux groupes de LR (LoRA et head séparés).
    Scheduler : cosine decay avec warmup linéaire 10% (transformers util).
    """

    def __init__(
        self,
        model_name: str = "camembert-base",
        n_classes: int = 27,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lr_lora: float = 5e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
    ):
        super().__init__()
        # save_hyperparameters : standard Lightning, OK car tout est JSON-serializable
        #self.save_hyperparameters()

        # Backbone CamemBERT-base (sans head)
        base = AutoModel.from_pretrained(model_name)
        hidden_size = base.config.hidden_size  # 768

        # LoRA config : RoBERTa-style modules nommés "query"/"value"
        # task_type=FEATURE_EXTRACTION car on a notre propre head (pas SEQ_CLS)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(base, lora_cfg)

        # Classification head (trainable, random init)
        self.head = nn.Linear(hidden_size, n_classes)

        # Hparams optimizer
        self.lr_lora = lr_lora
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        # Accumulation validation
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling pondéré par l'attention mask.

        Args:
            last_hidden: (B, L, H) sortie de la dernière couche transformer
            attention_mask: (B, L) 1 pour tokens réels, 0 pour padding

        Returns:
            (B, H) embedding moyen sur les tokens non-padding
        """
        mask = attention_mask.unsqueeze(-1).float()       # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)           # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)           # (B, 1)
        return summed / counts                             # (B, H)

    def _features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward jusqu'au mean pooling, AVANT la head → (B, 768)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    def _token_level_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward jusqu'au last_hidden_state, AVANT mean pooling.
 
        Returns:
            tuple (last_hidden_state, attention_mask) :
              - last_hidden_state : (B, seq_len, 768)
              - attention_mask : (B, seq_len) — nécessaire pour masquer
                le padding côté M3 (tokens pad ne doivent pas participer
                à l'attention inter-modale)
        """
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return outputs.last_hidden_state, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward complet → logits (B, n_classes)."""
        feats = self._features(input_ids, attention_mask)
        return self.head(feats)

    # ------------------------------------------------------------------ #
    # Training / validation steps                                         #
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = F.cross_entropy(logits, batch["labels"])
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_preds.append(preds)
        self._val_labels.append(batch["labels"])
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

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler                                                #
    # ------------------------------------------------------------------ #

    def configure_optimizers(self):
        """
        Deux groupes de params :
        - LoRA adapters (lr_lora=5e-4, plus petit car partent d'init ~0)
        - Head (lr_head=1e-3, plus gros car random init complet)

        Évite que la head apprenne trop vite et casse la consistance des
        features LoRA, ou que LoRA stagne pendant que la head explose.

        Scheduler : warmup linéaire 10% des steps puis cosine decay
        (transformers.get_cosine_schedule_with_warmup, standard pour BERT FT).
        """
        lora_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("head."):
                head_params.append(param)
            else:
                # Tous les autres params trainables sont LoRA (les autres
                # poids backbone sont gelés par peft)
                lora_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": self.lr_lora},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )

        # estimated_stepping_batches est disponible quand le Trainer est attaché
        # (Lightning le résout au moment de configure_optimizers via fit())
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # update à chaque step (pas chaque epoch)
            },
        }


# ====================================================================== #
# Wrapper BaseLearner                                                     #
# ====================================================================== #


class CamembertLoRA(BaseLearner):
    """
    BaseLearner texte CamemBERT + LoRA (Bloc N — M2.1).

    Workflow :
    - fit(X, y) : split 80/20 stratifié, fine-tune avec early stopping patience=2
    - extract_embeddings(X) : forward jusqu'au mean pooling → (n, 768)
    - predict_proba(X) : softmax du forward complet → (n, 27)

    Le DataFrame d'entrée doit contenir la colonne 'text' (concat designation
    + description, déjà produite par RakutenLightningDataModule mode
    raw_for_finetune).
    """

    def __init__(
        self,
        model_name: str = "camembert-base",
        n_classes: int = 27,
        max_len: int = 128,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        batch_size: int = 32,
        max_epochs: int = 10,
        patience: int = 2,
        lr_lora: float = 5e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        text_col: str = "text",
        num_workers: int = 2,
        random_state: int = 42,
        precision: str = "bf16-mixed",
    ):
        self._model_name = model_name
        self._n_classes = n_classes
        self._max_len = max_len
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr_lora = lr_lora
        self._lr_head = lr_head
        self._weight_decay = weight_decay
        self._warmup_ratio = warmup_ratio
        self._text_col = text_col
        self._num_workers = num_workers
        self._random_state = random_state
        self._precision = precision

        # Rempli au fit (ou from_pretrained)
        self.tokenizer = None
        self.net: _CamembertLoRALightning | None = None

    # ------------------------------------------------------------------ #
    # Propriétés BaseLearner                                              #
    # ------------------------------------------------------------------ #

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "text"

    @property
    def embed_dim(self) -> int:
        """Dimension du vecteur extrait par extract_embeddings (768 pour CamemBERT-base)."""
        return 768

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def name(self) -> str:
        return "camembert_lora"

    # ------------------------------------------------------------------ #
    # Helpers internes                                                    #
    # ------------------------------------------------------------------ #

    def _extract_texts(self, X: pl.DataFrame) -> list[str]:
        """Récupère la colonne texte du DataFrame en list[str]."""
        if self._text_col not in X.columns:
            raise ValueError(
                f"CamembertLoRA attend une colonne '{self._text_col}' dans X. "
                f"Colonnes disponibles : {X.columns}"
            )
        # Cast safe : remplace None par "" pour éviter crash tokenizer
        return [t if t is not None else "" for t in X[self._text_col].to_list()]

    def _make_loader(
        self,
        texts: list[str],
        labels: np.ndarray | None,
        shuffle: bool,
    ) -> DataLoader:
        ds = _CamembertTextDataset(texts, labels, self.tokenizer, self._max_len)
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
        **kwargs
    ) -> "CamembertLoRA":
        """
        Fine-tune CamemBERT + LoRA avec early stopping interne.

        Convention M.0 : le DataModule fournit X_train/X_val pré-splittés.
        Si X_val=None, fallback sur un split interne par compatibilité notebook.

        Étapes :
        1. Charge le tokenizer (HuggingFace, partagé entre train et inference)
        2. (fallback) Si X_val=None → split interne 80/20 stratifié
        3. Extrait les textes train + val
        4. DataLoaders (tokenisation à la volée)
        5. Instancie le Lightning module (cela charge le backbone CamemBERT
           depuis HuggingFace hub + applique LoRA + initialise la head)
        6. Fit Lightning avec early stopping sur val_f1_weighted
        """
        if (X_val is None) != (y_val is None):
            raise ValueError(
                "X_val et y_val doivent être tous deux fournis ou tous deux None."
            )

        # 1. Tokenizer (CRITIQUE : le même au train et au reload pour
        #    garantir l'idempotence de la tokenisation)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        # 2. Fallback split si X_val absent
        if X_val is None:
            print(f"[CamembertLoRA] WARN: pas de X_val fourni, fallback split interne "
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

        # 3. Extraire textes
        texts_tr = self._extract_texts(X_train)
        texts_val = self._extract_texts(X_val)

        if len(texts_tr) != len(y_train):
            raise ValueError(f"len(X_train)={len(texts_tr)} != len(y_train)={len(y_train)}")
        if len(texts_val) != len(y_val):
            raise ValueError(f"len(X_val)={len(texts_val)} != len(y_val)={len(y_val)}")

        print(
            f"[CamembertLoRA] fit() | n_train={len(y_train)} n_val={len(y_val)} "
            f"| model={self._model_name} | LoRA r={self._lora_rank} α={self._lora_alpha}"
        )

        # 4. DataLoaders
        train_loader = self._make_loader(texts_tr, y_train, shuffle=True)
        val_loader = self._make_loader(texts_val, y_val, shuffle=False)

        # 5. Lightning module
        self.net = _CamembertLoRALightning(
            model_name=self._model_name,
            n_classes=self._n_classes,
            lora_rank=self._lora_rank,
            lora_alpha=self._lora_alpha,
            lora_dropout=self._lora_dropout,
            lr_lora=self._lr_lora,
            lr_head=self._lr_head,
            weight_decay=self._weight_decay,
            warmup_ratio=self._warmup_ratio,
        )

        # Log trainable params (sanity check sur ~2.4M)
        n_trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.net.parameters())
        print(
            f"[CamembertLoRA] Trainable: {n_trainable:,} / {n_total:,} "
            f"({100.0 * n_trainable / n_total:.2f}%)"
        )

        # 6. Trainer
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
        if self.net is None or self.tokenizer is None:
            raise RuntimeError("CamembertLoRA.fit() doit être appelé avant.")

        texts = self._extract_texts(X)
        loader = self._make_loader(texts, labels=None, shuffle=False)

        self.net.eval()
        device = next(self.net.parameters()).device

        outputs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                if return_features:
                    feat = self.net._features(input_ids, attention_mask)   # (B, 768)
                    outputs.append(feat.cpu().numpy())
                else:
                    logits = self.net(input_ids, attention_mask)
                    probas = F.softmax(logits, dim=1)
                    outputs.append(probas.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """(n, embed_dim=768) — mean pooling du dernier hidden state."""
        return self._forward_in_batches(X, return_features=True)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 27) — softmax de la head sur les features pooled."""
        return self._forward_in_batches(X, return_features=False)

    @property
    def sequence_dim(self) -> int:
        """768 — hidden_size du backbone CamemBERT-base."""
        return 768
    
    def extract_token_embeddings(
        self, X: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (n, max_len, 768) + (n, max_len) attention_mask.
 
        Retourne le last_hidden_state complet avant mean pooling, ainsi que
        l'attention_mask pour que M3 ignore les tokens de padding.
 
        Returns:
            (token_embeddings, attention_mask) :
              - token_embeddings : np.ndarray float32 (n, max_len, 768)
              - attention_mask   : np.ndarray int64   (n, max_len)
                1 = token réel, 0 = padding
        """
        if self.net is None or self.tokenizer is None:
            raise RuntimeError(
                "CamembertLoRA.fit() ou from_pretrained() "
                "doit être appelé avant extract_token_embeddings."
            )
 
        texts = self._extract_texts(X)
        loader = self._make_loader(texts, labels=None, shuffle=False)
        self.net.eval()
        device = next(self.net.parameters()).device
 
        all_hidden, all_masks = [], []
        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                mask = batch["attention_mask"].to(device, non_blocking=True)
                hidden, mask = self.net._token_level_features(ids, mask)
                all_hidden.append(hidden.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
 
        return (
            np.concatenate(all_hidden, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(np.int64),
        )



    # ------------------------------------------------------------------ #
    # M.4bis — Persistance PyFunc-compatible                              #
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde réversible du CamembertLoRA dans `path`.

        Crée :
        - net_state.pt   : dict {adapter_state, head_state}
                           - adapter_state : LoRA weights seuls (~10 MB)
                             via peft.get_peft_model_state_dict
                           - head_state : Linear(768, 27).state_dict (~80 KB)
                           Total ≈ 10 MB (vs 440 MB si on sauvegardait
                           tout le state_dict du Lightning module).
        - tokenizer/     : tokenizer HuggingFace (BPE vocab + config)
                           CRITIQUE pour reproducibilité de la tokenisation
                           au reload (équivalent du vocab.json de TextCNN).
        - config.json    : hyperparamètres de construction

        Justification du choix LoRA-state-only :
        - Les poids du backbone CamemBERT sont identiques à `model_name` sur
          HuggingFace Hub. Pas besoin de les sauvegarder, on les re-télécharge
          au reload. Gain ~430 MB stockage par version.
        - Si HuggingFace Hub change ou supprime le modèle, le reload casse.
          Mitigation : on log `model_name` dans config.json et MLflow,
          docker image cache les poids HF dans /root/.cache.

        Pré-condition : fit() doit avoir été appelé (sinon net=None et
        tokenizer=None → ValueError explicite).
        """
        if self.net is None or self.tokenizer is None:
            raise RuntimeError(
                "CamembertLoRA.save_pretrained() appelé avant fit(). "
                "net/tokenizer sont None — rien à sauvegarder."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 1. LoRA adapter state + head state (combinés en un seul .pt)
        adapter_state = get_peft_model_state_dict(self.net.backbone)
        head_state = self.net.head.state_dict()
        torch.save(
            {
                "adapter_state": adapter_state,
                "head_state": head_state,
            },
            path / "net_state.pt",
        )

        # 2. Tokenizer (CRITIQUE pour idempotence tokenisation)
        self.tokenizer.save_pretrained(path / "tokenizer")

        # 3. Hyperparams de construction
        config = {
            "model_name": self._model_name,
            "n_classes": self._n_classes,
            "max_len": self._max_len,
            "lora_rank": self._lora_rank,
            "lora_alpha": self._lora_alpha,
            "lora_dropout": self._lora_dropout,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "lr_lora": self._lr_lora,
            "lr_head": self._lr_head,
            "weight_decay": self._weight_decay,
            "warmup_ratio": self._warmup_ratio,
            "text_col": self._text_col,
            "num_workers": self._num_workers,
            "random_state": self._random_state,
            "precision": self._precision,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "CamembertLoRA":
        """
        Reconstruit un CamembertLoRA depuis un dossier écrit par save_pretrained.

        Steps :
        1. Lit config.json → instancie CamembertLoRA(**config)
        2. Recharge tokenizer depuis path/tokenizer (identique à celui du train)
        3. Instancie _CamembertLoRALightning(...) → recharge backbone
           CamemBERT depuis HuggingFace Hub + applique LoRA fraîche +
           head random
        4. Charge net_state.pt :
           - adapter_state → set_peft_model_state_dict (écrase LoRA random)
           - head_state → load_state_dict sur la head (écrase head random)
        5. self.net.eval()

        Le learner retourné est immédiatement utilisable pour
        extract_embeddings / predict_proba (pas besoin de re-fit).
        """
        path = Path(path)

        # 1. Lire config
        with open(path / "config.json") as f:
            config = json.load(f)

        # 2. Instancier le wrapper (laisse tokenizer=None, net=None)
        instance = cls(**config)

        # 3. Recharger tokenizer depuis le snapshot du train (cohérence
        #    garantie même si HuggingFace pousse une nouvelle version)
        instance.tokenizer = AutoTokenizer.from_pretrained(str(path / "tokenizer"))

        # 4. Reconstruire le Lightning module
        #    Note : cela télécharge le backbone CamemBERT depuis HuggingFace
        #    Hub (ou cache local si déjà tiré). La LoRA appliquée est
        #    initialement random — on l'écrase juste après.
        instance.net = _CamembertLoRALightning(
            model_name=instance._model_name,
            n_classes=instance._n_classes,
            lora_rank=instance._lora_rank,
            lora_alpha=instance._lora_alpha,
            lora_dropout=instance._lora_dropout,
            lr_lora=instance._lr_lora,
            lr_head=instance._lr_head,
            weight_decay=instance._weight_decay,
            warmup_ratio=instance._warmup_ratio,
        )

        # 5. Charger LoRA adapter state + head state
        saved = torch.load(path / "net_state.pt", map_location="cpu")
        set_peft_model_state_dict(instance.net.backbone, saved["adapter_state"])
        instance.net.head.load_state_dict(saved["head_state"])
        instance.net.eval()

        return instance
