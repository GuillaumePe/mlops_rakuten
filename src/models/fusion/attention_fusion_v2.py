"""
M3.2 — Module de fusion par attention, variante co-adaptation.

Deux différences par rapport à `AttentionFusionModule` :

    B.1 — POOLING PAR ATTENTION (au lieu du masked mean pooling)
        Un token-query appris agrège la séquence fusionnée via une
        MultiheadAttention. Justification : le mean-pool uniforme donne le
        MÊME poids à chaque token, donc dilue le texte (fort, ~quelques
        tokens informatifs) dans 196 patches image souvent bruités. Une
        query apprise laisse le modèle SÉLECTIONNER les tokens pertinents
        (attention pooling, cf. set-transformer PMA, Lee et al. 2019).
        Coût : +~d_model² params (la MHA de pooling). Un flag `pooling`
        permet de retomber sur le mean-pool pour l'A/B.

    B.2 — INJECTION TABULAIRE DANS LA TÊTE
        Les d_tab features tabulaires (que M2 voit, pas M3) sont
        concaténées au vecteur poolé AVANT la tête MLP. Réduit l'asymétrie
        d'input avec M2. NB : on n'injecte QUE le tabulaire brut, PAS les
        marginales p_text/p_image — sinon M3.2 deviendrait un stacker
        déguisé (late fusion) et perdrait sa valeur de membre décorrélé.

Reste identique à M3 : projections modales → d_model + LayerNorm,
type embeddings additifs, TransformerEncoder Pre-LN, tête GELU.

Ce module est un nn.Module pur : il ne connaît ni les BaseLearners, ni
MLflow, ni la co-adaptation DoRA (gérée par l'assembled M32CoAdaptationFusion).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusionModuleV2(nn.Module):
    """
    Fusion cross-modale + pooling par attention + injection tabulaire.

    Args:
        d_text: dimension des tokens texte (768 CamemBERT).
        d_image: dimension des patches image (768 SigLIP ViT-B, 512 ResNet18).
        d_tab: dimension des features tabulaires injectées dans la tête
            (0 = pas d'injection, comportement compatible M3).
        d_model: dimension commune de projection et du Transformer.
        n_heads: nombre de têtes d'attention (Transformer ET pooling).
        n_layers: nombre de couches TransformerEncoder.
        dim_ff_factor: dim_feedforward = d_model * dim_ff_factor.
        dropout: taux de dropout (attention + FFN + pooling + MLP head).
        n_classes: nombre de classes de sortie.
        pooling: "attention" (B.1, défaut) | "mean" (fallback M3 pour A/B).
    """

    def __init__(
        self,
        d_text: int = 768,
        d_image: int = 768,
        d_tab: int = 0,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff_factor: int = 4,
        dropout: float = 0.3,
        n_classes: int = 27,
        pooling: str = "attention",
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) doit être divisible par n_heads ({n_heads})."
            )
        if pooling not in ("attention", "mean"):
            raise ValueError(f"pooling doit être 'attention' ou 'mean', reçu {pooling!r}.")

        self.d_model = d_model
        self.d_tab = d_tab
        self.pooling = pooling
        dim_ff = d_model * dim_ff_factor

        # --- Projections modales → espace commun d_model ---
        self.proj_text = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_image = nn.Sequential(
            nn.Linear(d_image, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Type embeddings additifs (0 = texte, 1 = image) ---
        self.type_embedding = nn.Embedding(2, d_model)

        # --- Transformer Encoder (Pre-LN) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # --- B.1 : pooling par attention ---
        # Query apprise (1, 1, d_model) répliquée sur le batch ; attend la
        # séquence fusionnée et en extrait un vecteur pondéré. LayerNorm
        # finale pour stabiliser l'entrée de la tête.
        if self.pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.pool_norm = nn.LayerNorm(d_model)

        # --- B.2 : tête, dimensionnée pour le tabulaire concaténé ---
        head_in = d_model + d_tab
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform pour projections/tête ; type embeddings petits (std=0.02)."""
        for module in [self.proj_text, self.proj_image, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.normal_(self.type_embedding.weight, std=0.02)
        if self.pooling == "attention":
            nn.init.normal_(self.pool_query, std=0.02)

    def _pool(self, output: torch.Tensor, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Agrège la séquence (B, S, d_model) en un vecteur (B, d_model).

        full_mask : (B, S) avec 1 = token réel, 0 = padding.
        """
        if self.pooling == "mean":
            mask_expanded = full_mask.unsqueeze(-1)              # (B, S, 1)
            summed = (output * mask_expanded).sum(dim=1)         # (B, d_model)
            counts = mask_expanded.sum(dim=1).clamp(min=1)       # (B, 1)
            return summed / counts                               # (B, d_model)

        # --- attention pooling ---
        B = output.size(0)
        query = self.pool_query.expand(B, -1, -1)                # (B, 1, d_model)
        key_padding_mask = full_mask == 0                        # True = ignoré
        pooled, _ = self.pool_attn(
            query=query,
            key=output,
            value=output,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                        # (B, 1, d_model)
        return self.pool_norm(pooled.squeeze(1))                 # (B, d_model)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        image_patches: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: (B, S_t, d_text).
            text_mask: (B, S_t) — 1 = token réel, 0 = padding.
            image_patches: (B, S_i, d_image) — pas de padding.
            tabular: (B, d_tab) si d_tab > 0, sinon None.

        Returns:
            logits: (B, n_classes).
        """
        B = text_tokens.size(0)
        device = text_tokens.device
        S_i = image_patches.size(1)

        # --- Projections + type embeddings ---
        text_proj = self.proj_text(text_tokens) + self.type_embedding(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        image_proj = self.proj_image(image_patches) + self.type_embedding(
            torch.ones(1, dtype=torch.long, device=device)
        )

        # --- Concat [texte, image] + masque ---
        sequence = torch.cat([text_proj, image_proj], dim=1)     # (B, S_t+S_i, d_model)
        image_mask = torch.ones(B, S_i, device=device)
        full_mask = torch.cat([text_mask.float(), image_mask], dim=1)
        padding_mask = full_mask == 0

        # --- Transformer ---
        output = self.transformer(sequence, src_key_padding_mask=padding_mask)

        # --- Pooling (B.1) ---
        pooled = self._pool(output, full_mask)                   # (B, d_model)

        # --- Injection tabulaire (B.2) ---
        if self.d_tab > 0:
            if tabular is None:
                raise ValueError(
                    f"d_tab={self.d_tab} mais aucun tensor tabular passé au forward."
                )
            pooled = torch.cat([pooled, tabular.float()], dim=-1)  # (B, d_model+d_tab)

        return self.head(pooled)                                 # (B, n_classes)
