"""
TextCNN (Yoon Kim 2014) : base learner texte from-scratch pour M2.2.

Architecture :
    Input         : sequence d'ids tokenisés (B, max_len)
    Embedding     : (B, max_len, 300) - learnable from scratch
    Conv1d × 6    : kernel sizes 1, 2, 3, 4, 5, 6, out_channels=512 chacun
    MaxPool1d     : global max sur la dimension temporelle → (B, 512) par filtre
    Concat        : (B, 6 × 512) = (B, 3072)
    Dropout 0.5
    Linear(3072, 27)

Justification mathématique :
- Embedding from-scratch : pas de pré-entraînement, le vocab Rakuten (catégories
  produits e-commerce) diffère de Wikipedia (CamemBERT). Apprentissage direct
  de la sémantique pertinente, biais inductif favorable au domaine.
- Convs kernel 1-6 : capture des n-grams de longueur 1 à 6 (mots, bigrams,
  trigrams, ...). Le maxpool global → invariance à la position : "ce mot/n-gram
  apparaît quelque part dans le texte" = signal suffisant pour la classification.
- Dropout 0.5 : approximation de model averaging bayésien (Gal & Ghahramani 2016).
  Régularise un réseau de 20M params sur ~30k samples train.
- Cross-entropy loss : équivalent MLE sous hypothèse multinomial sur les classes.

Workflow d'entraînement :
1. fit(X, y, val_split=0.2) : construit le vocab top-K=50000 sur le train,
   tokenise, split interne 80/20 train/val, fit Lightning avec early stopping
   patience=2 sur val_f1_weighted.
2. extract_embeddings(X) → (n, 3072) : forward jusqu'au concat post-pool,
   avant le classifier final. Utilisé en aval par K-Fold OOF LogReg du
   StackingLGBM (pattern frugal validé en M2 v4).
3. predict_proba(X) → (n, 27) : softmax du classifier final. Exposé pour les
   analyses de complémentarité (Bloc Q) mais NON utilisé directement par le
   StackingLGBM (fuite si appliqué sur le train pool).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal
import math
import json
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.base_learners._base import BaseLearner


# ====================================================================== #
# Tokenizer simple (lowercase + split + vocab top-K)                     #
# ====================================================================== #

_PUNCT_REGEX = r"[\.,;:!?\(\)\[\]\{\}\"'`~/\\\-_<>\|=\+\*&%\$#@]"


def _simple_tokenize(text: str) -> list[str]:
    """
    Tokenisation minimaliste : lowercase, séparateurs whitespace + ponctuation.

    On évite les dépendances lourdes (keras/spacy) pour rester portable.
    L'objectif est de reproduire fidèlement le comportement par défaut de
    keras.preprocessing.text.Tokenizer utilisé dans le benchmark Rakuten.
    """
    import re
    if not isinstance(text, str) or text == "":
        return []
    text = text.lower()
    text = re.sub(_PUNCT_REGEX, " ", text)
    return [tok for tok in text.split() if tok]


def _build_vocab(texts: list[str], top_k: int = 50000) -> dict[str, int]:
    """
    Construit un vocab `{token: id}` avec les top-K tokens les plus fréquents.

    Ids réservés :
    - 0 : <PAD> (padding)
    - 1 : <UNK> (out-of-vocabulary, à utiliser au predict pour tout token
                 absent du vocab construit sur le train)
    """
    counter: Counter[str] = Counter()
    for txt in texts:
        counter.update(_simple_tokenize(txt))

    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for token, _count in counter.most_common(top_k - 2):  # -2 pour PAD et UNK
        vocab[token] = len(vocab)
    return vocab


def _tokenize_and_pad(
    texts: list[str], vocab: dict[str, int], max_len: int
) -> np.ndarray:
    """
    Tokenise + pad/truncate vers `max_len`. Retourne array int64 (n, max_len).

    - Tokens hors vocab → <UNK> (id=1)
    - Truncation post (queue) si > max_len
    - Padding post avec <PAD> (id=0) si < max_len
    """
    unk_id = vocab["<UNK>"]
    pad_id = vocab["<PAD>"]
    n = len(texts)
    out = np.full((n, max_len), pad_id, dtype=np.int64)

    for i, txt in enumerate(texts):
        tokens = _simple_tokenize(txt)
        ids = [vocab.get(tok, unk_id) for tok in tokens[:max_len]]
        out[i, : len(ids)] = ids

    return out


# ====================================================================== #
# Lightning module : le réseau pur                                       #
# ====================================================================== #


class _TextCNNLightning(L.LightningModule):
    """
    Architecture pure TextCNN. Ne sait rien de polars ni de vocab.

    Reçoit en entrée des `torch.Tensor` d'ids (B, max_len), retourne des logits.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        n_filters: int = 512,
        kernel_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        n_classes: int = 27,
        dropout: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
#        self.save_hyperparameters()

        # Embedding : padding_idx=0 → gradient nul sur <PAD>, n'apprend pas
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Une Conv1d par taille de kernel.
        # Conv1d attend (B, C_in, L) → on permute Embedding (B, L, C) → (B, C, L)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=n_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters * len(kernel_sizes), n_classes)

        self.lr = lr
        self.weight_decay = weight_decay

        # Pour validation : on accumule predictions et labels par epoch
        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward jusqu'au concat post max-pool, avant classifier.

        Args:
            x: (B, max_len) ids tokenisés
        Returns:
            (B, n_filters * len(kernel_sizes)) = (B, 3072) features
        """
        emb = self.embedding(x)                # (B, L, embed_dim)
        emb = emb.transpose(1, 2)              # (B, embed_dim, L)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(emb))              # (B, n_filters, L - k + 1)
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)  # (B, n_filters)
            pooled.append(h)

        return torch.cat(pooled, dim=1)        # (B, n_filters * n_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward complet : ids → logits (B, n_classes)."""
        feats = self._features(x)
        feats = self.dropout(feats)
        return self.classifier(feats)

    def _token_level_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward jusqu'aux feature maps des convolutions, AVANT max-pool.
 
        Chaque Conv1d(kernel_size=k) produit (B, n_filters, L-k+1).
        On pad chaque sortie à la longueur max (L, obtenue avec k=1),
        puis on concat sur la dimension features.
 
        Args:
            x: (B, max_len) ids tokenisés
 
        Returns:
            (B, max_len, n_filters * len(kernel_sizes)) — ex: (B, 128, 3072)
        """
        emb = self.embedding(x)          # (B, L, embed_dim)
        emb = emb.transpose(1, 2)        # (B, embed_dim, L)
 
        max_len = x.size(1)
        padded_maps = []
 
        for conv in self.convs:
            h = F.relu(conv(emb))        # (B, n_filters, L - k + 1)
            pad_size = max_len - h.size(2)
            if pad_size > 0:
                h = F.pad(h, (0, pad_size))
            padded_maps.append(h)
 
        concat = torch.cat(padded_maps, dim=1)  # (B, n_filters * n_kernels, L)
        return concat.transpose(1, 2)            # (B, L, n_filters * n_kernels)

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
		# M.1a : AdamW + warmup linéaire + cosine decay sur lr_head.
        #
        # Justification mathématique :
        # - AdamW : découplage weight_decay du gradient (Loshchilov & Hutter 2019).
        #   Adam classique applique le weight decay via le gradient normalisé
        #   → sous-régularise les poids à grand gradient. AdamW le découple.
        # - Warmup : à l'init, Adam sous-estime v_t (second moment) → updates
        #   trop grands. Warmup linéaire sur 10% des steps laisse v_t se
        #   stabiliser avant d'appliquer lr_max. (Goyal et al. 2017)
        # - Cosine decay : lr(t) = 0.5 * lr_max * (1 + cos(π*t/T_max))
        #   Converge vers flat minimum (Hochreiter & Schmidhuber 1997) vs
        #   sharp minimum avec lr fixe → meilleure généralisation.
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # estimated_stepping_batches = steps total sur toute la durée du training
        # (max_epochs * steps_per_epoch). Disponible après que le Trainer a
        # été attaché au module (appelé dans configure_optimizers() = après setup).
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = max(1, int(0.1 * total_steps))

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


# ====================================================================== #
# Wrapper BaseLearner : orchestrateur                                    #
# ====================================================================== #


class TextCNN(BaseLearner):
    """
    BaseLearner texte TextCNN from-scratch.

    Workflow :
    - fit(X, y) : build vocab, tokenize, split 80/20, train avec early stopping
    - extract_embeddings(X) : forward jusqu'au concat post-pool (avant classifier)
    - predict_proba(X) : softmax du classifier final
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        max_len: int = 128,
        embed_dim: int = 300,
        n_filters: int = 512,
        kernel_sizes: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
        n_classes: int = 27,
        dropout: float = 0.5,
        batch_size: int = 64,
        max_epochs: int = 15,
        patience: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        text_col: str = "text",  # nom de la colonne texte dans le DataFrame
        random_state: int = 42,
        precision: str = "bf16-mixed",
    ):
        self._vocab_size_cap = vocab_size
        self._max_len = max_len
        self._embed_dim = embed_dim
        self._n_filters = n_filters
        self._kernel_sizes = tuple(kernel_sizes)
        self._n_classes = n_classes
        self._dropout = dropout
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._lr = lr
        self._weight_decay = weight_decay
        self._text_col = text_col
        self._random_state = random_state
        self._precision = precision

        # Rempli au fit
        self.vocab: dict[str, int] | None = None
        self.net: _TextCNNLightning | None = None

    # ------------------------------------------------------------------ #
    # Propriétés BaseLearner                                              #
    # ------------------------------------------------------------------ #

    @property
    def modality(self) -> Literal["text", "image", "tabular"]:
        return "text"

    @property
    def embed_dim(self) -> int:
        """Dimension du vecteur extrait par extract_embeddings."""
        return self._n_filters * len(self._kernel_sizes)  # 512 × 6 = 3072

    @property
    def n_classes(self) -> int:
        return self._n_classes

    @property
    def name(self) -> str:
        return "textcnn"

    # ------------------------------------------------------------------ #
    # Helpers internes                                                    #
    # ------------------------------------------------------------------ #

    def _extract_texts(self, X: pl.DataFrame) -> list[str]:
        """Récupère la colonne texte du DataFrame en list[str]."""
        if self._text_col not in X.columns:
            raise ValueError(
                f"TextCNN attend une colonne '{self._text_col}' dans X. "
                f"Colonnes disponibles : {X.columns}"
            )
        return X[self._text_col].fill_null("").to_list()

    def _make_loader(
        self,
        ids: np.ndarray,
        labels: np.ndarray | None,
        shuffle: bool,
        num_workers: int = 2,
    ) -> DataLoader:
        ids_t = torch.from_numpy(ids)
        if labels is not None:
            y_t = torch.from_numpy(labels).long()
            ds = TensorDataset(ids_t, y_t)
        else:
            # Pour predict : on a quand même besoin d'un placeholder de labels
            # pour TensorDataset, mais on ignorera l'output en mode eval
            y_t = torch.zeros(len(ids_t), dtype=torch.long)
            ds = TensorDataset(ids_t, y_t)
        return DataLoader(
            ds,
            batch_size=self._batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
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
    ) -> "TextCNN":
        """
        Entraîne le TextCNN avec early stopping interne.

        Convention M.0 : le DataModule fournit X_train/X_val pré-splittés
        (80/20 stratifié seed=42 sur train_pool_effective). Si X_val=None,
        fallback sur un split interne par compatibilité notebook.

        Étapes :
        1. (fallback) Si X_val=None → split interne 80/20 stratifié
        2. Build vocab top-K=50k UNIQUEMENT sur X_train (anti-fuite val→vocab)
        3. Tokenize X_train et X_val
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
            print(f"[TextCNN] WARN: pas de X_val fourni, fallback split interne "
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

        texts_tr = self._extract_texts(X_train)
        texts_val = self._extract_texts(X_val)
        if len(texts_tr) != len(y_train):
            raise ValueError(
                f"len(X_train)={len(texts_tr)} != len(y_train)={len(y_train)}"
            )
        if len(texts_val) != len(y_val):
            raise ValueError(
                f"len(X_val)={len(texts_val)} != len(y_val)={len(y_val)}"
            )

        # Build vocab UNIQUEMENT sur le train (anti-fuite val→vocab).
        # Garantie par construction : X_train fourni par le DataModule
        # n'inclut PAS les samples de X_val (split orthogonal).
        self.vocab = _build_vocab(texts_tr, top_k=self._vocab_size_cap)
        print(f"[TextCNN] Vocab construit : {len(self.vocab)} tokens "
              f"(top-{self._vocab_size_cap} demandés)")

        # Tokenize
        ids_tr = _tokenize_and_pad(texts_tr, self.vocab, self._max_len)
        ids_val = _tokenize_and_pad(texts_val, self.vocab, self._max_len)

        # DataLoaders
        train_loader = self._make_loader(ids_tr, y_train, shuffle=True)
        val_loader = self._make_loader(ids_val, y_val, shuffle=False)

        # Modèle Lightning
        self.net = _TextCNNLightning(
            vocab_size=len(self.vocab),
            embed_dim=self._embed_dim,
            n_filters=self._n_filters,
            kernel_sizes=self._kernel_sizes,
            n_classes=self._n_classes,
            dropout=self._dropout,
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        # T.1 — TextCNN est stateless-only PAR CONCEPTION (warm-start neutralisé).
        # Le vocab est reconstruit à chaque batch via _build_vocab(corpus_courant),
        # donc l'Embedding est NON-STATIONNAIRE : la ligne i de la table ne
        # référence pas le même token d'un batch à l'autre (mapping token→id
        # trié par fréquence sur un corpus qui change). Injecter les poids d'un
        # @active reviendrait à charger un prior mal-aligné (bruit), pas un
        # warm-start — et le filtre par shape ne protège pas du cas V_n == V_{n-1}
        # (tailles égales mais mapping différent). On neutralise donc tout état
        # de warm-start que l'orchestrateur aurait pu poser.
        if getattr(self, "_warm_start_net_state", None) is not None:
            print(
                "[TextCNN] warm-start ignoré : stateless-only "
                "(vocab non-stationnaire, Embedding non-transférable)."
            )
            self._warm_start_net_state = None

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
        # M.1a : gradient clipping pour stabiliser les updates sur embedding
        # (init aléatoire → gradients potentiellement larges au début)

        lightning_logger = kwargs.get("lightning_logger", None)
        trainer = L.Trainer(
            max_epochs=self._max_epochs,
            callbacks=callbacks,
            precision=self._precision,
            enable_checkpointing=False,
            logger=lightning_logger if lightning_logger is not None else False,
            log_every_n_steps=50,
            gradient_clip_val=1.0,

        )
        trainer.fit(self.net, train_loader, val_loader)

        # Switch en mode eval pour les utilisations downstream
        self.net.eval()
        return self

    def _forward_in_batches(
        self,
        X: pl.DataFrame,
        return_features: bool,
    ) -> np.ndarray:
        """
        Forward generic, en eval mode, no grad.

        Args:
            return_features: True → embeddings (n, 3072), False → probas (n, 27)
        """
        if self.net is None or self.vocab is None:
            raise RuntimeError("TextCNN.fit() doit être appelé avant.")

        texts = self._extract_texts(X)
        ids = _tokenize_and_pad(texts, self.vocab, self._max_len)
        loader = self._make_loader(ids, labels=None, shuffle=False)

        self.net.eval()
        device = next(self.net.parameters()).device

        outputs = []
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x = x.to(device, non_blocking=True)
                if return_features:
                    feat = self.net._features(x)           # (B, 3072)
                    outputs.append(feat.cpu().numpy())
                else:
                    logits = self.net(x)
                    probas = F.softmax(logits, dim=1)
                    outputs.append(probas.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)

    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """(n, embed_dim=3072) — concat des 6 max-pools, avant le classifier."""
        return self._forward_in_batches(X, return_features=True)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """(n, 27) — softmax du classifier final."""
        return self._forward_in_batches(X, return_features=False)

    def extract_token_embeddings(
        self, X: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        (n, max_len, 3072) + (n, max_len) attention_mask.
 
        Feature maps des 6 convolutions avant max-pool. Chaque position t
        agrège les vues n-gram (kernel sizes 1..6), zéro-paddé en fin
        pour les convs plus courtes.
 
        Le mask est dérivé des input ids : position != PAD (id=0) → 1.
 
        Returns:
            (token_embeddings, attention_mask) :
              - token_embeddings : np.ndarray float32 (n, max_len, 3072)
              - attention_mask   : np.ndarray int64   (n, max_len)
                1 = token réel, 0 = padding
        """
        if self.net is None or self.vocab is None:
            raise RuntimeError(
                "TextCNN.fit() ou from_pretrained() "
                "doit être appelé avant extract_token_embeddings."
            )
 
        texts = self._extract_texts(X)
        ids = _tokenize_and_pad(texts, self.vocab, self._max_len)
        loader = self._make_loader(ids, labels=None, shuffle=False)
        self.net.eval()
        device = next(self.net.parameters()).device
 
        all_features, all_masks = [], []
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x = x.to(device, non_blocking=True)
                feat = self.net._token_level_features(x)  # (B, L, 3072)
                mask = (x != 0).long()                    # (B, L) — PAD=0
                all_features.append(feat.cpu().numpy())
                all_masks.append(mask.cpu().numpy())
 
        return (
            np.concatenate(all_features, axis=0).astype(np.float32),
            np.concatenate(all_masks, axis=0).astype(np.int64),
        )

    # ------------------------------------------------------------------ #
    # M.4bis — Persistance PyFunc-compatible                              #
    # ------------------------------------------------------------------ #
 
    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde réversible du TextCNN dans `path`.
 
        Crée 3 fichiers :
        - net_state.pt   : state_dict du nn.Module (_TextCNNLightning)
        - vocab.json     : dict {token: id} construit au fit() (CRITIQUE pour
                           refaire la tokenisation à l'inférence)
        - config.json    : hyperparamètres de construction (pour reconstruire
                           la classe via from_pretrained)
 
        Pré-condition : fit() doit avoir été appelé (sinon vocab=None et
        net=None → ValueError explicite).
 
        Justification :
        - Le state_dict suffit pour les poids appris (Embedding, Conv1d, Linear).
        - Le vocab N'EST PAS dans le state_dict — c'est un dict Python construit
          au fit() à partir du corpus train. Sans lui, impossible de tokeniser
          un nouveau texte → reload inutilisable. C'est l'invariant central.
        """
        if self.vocab is None or self.net is None:
            raise RuntimeError(
                "TextCNN.save_pretrained() appelé avant fit(). "
                "vocab et net sont None — rien à sauvegarder."
            )
 
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
 
        # 1. Poids du nn.Module Lightning
        torch.save(self.net.state_dict(), path / "net_state.pt")
 
        # 2. Vocab : critique, ne PAS perdre (tokenize impossible sinon)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
 
        # 3. Hyperparams de construction
        #    On utilise vocab_size_cap (paramètre demandé) plutôt que
        #    len(self.vocab) (taille réelle) — la vraie vocab_size effective
        #    sera len(vocab) au reload.
        config = {
            "vocab_size": self._vocab_size_cap,
            "max_len": self._max_len,
            "embed_dim": self._embed_dim,
            "n_filters": self._n_filters,
            "kernel_sizes": list(self._kernel_sizes),  # tuple → list pour JSON
            "n_classes": self._n_classes,
            "dropout": self._dropout,
            "batch_size": self._batch_size,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
            "lr": self._lr,
            "weight_decay": self._weight_decay,
            "text_col": self._text_col,
            "random_state": self._random_state,
            "precision": self._precision,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
 
    @classmethod
    def from_pretrained(cls, path: str | Path) -> "TextCNN":
        """
        Reconstruit un TextCNN depuis un dossier écrit par save_pretrained.
 
        Steps :
        1. Lit config.json → instancie TextCNN avec hyperparams
        2. Lit vocab.json → restaure self.vocab
        3. Reconstruit self.net = _TextCNNLightning(vocab_size=len(vocab), ...)
           ATTENTION : utiliser len(vocab) (taille effective) et pas
           vocab_size_cap (paramètre demandé, qui peut être > len(vocab)
           si le corpus a moins de tokens uniques).
        4. Charge state_dict dans self.net
        5. Met self.net en mode eval()
 
        Le learner retourné est immédiatement utilisable pour
        extract_embeddings / predict_proba (pas besoin de re-fit).
        """
        path = Path(path)
 
        # 1. Lire config + restaurer types (tuple)
        with open(path / "config.json") as f:
            config = json.load(f)
        config["kernel_sizes"] = tuple(config["kernel_sizes"])
 
        # 2. Instancier TextCNN avec les hyperparams
        instance = cls(**config)
 
        # 3. Restaurer le vocab AVANT de reconstruire le nn.Module
        #    (besoin de len(vocab) pour la taille de l'Embedding)
        with open(path / "vocab.json", encoding="utf-8") as f:
            instance.vocab = json.load(f)
 
        # 4. Reconstruire _TextCNNLightning avec la vraie vocab_size effective
        instance.net = _TextCNNLightning(
            vocab_size=len(instance.vocab),
            embed_dim=instance._embed_dim,
            n_filters=instance._n_filters,
            kernel_sizes=instance._kernel_sizes,
            n_classes=instance._n_classes,
            dropout=instance._dropout,
            lr=instance._lr,
            weight_decay=instance._weight_decay,
        )
 
        # 5. Charger les poids appris (map_location=cpu pour portabilité GPU↔CPU)
        state = torch.load(path / "net_state.pt", map_location="cpu")
        instance.net.load_state_dict(state)
        instance.net.eval()
 
        return instance

   