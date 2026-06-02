"""
O.2 — Module de fusion par attention cross-modale.

nn.Module pur : reçoit des tensors, ne connaît ni les BaseLearners ni MLflow.
Consommé par M3AttentionFusion (LightningModule, O.3).

Architecture :
    text_tokens  (B, S_t, d_text)  ──► Linear(d_text, d_model) + LayerNorm ──┐
                                                                               │
    image_patches (B, S_i, d_image) ─► Linear(d_image, d_model) + LayerNorm ──┤
                                                                               ▼
                              text_proj + image_proj  (+ type embeddings additifs)
                                                │
                                    TransformerEncoder (n_layers couches)
                                                │
                                    masked mean pooling (ignore le padding)
                                                │
                                    MLP d_model → d_model//2 → n_classes
                                        GELU + dropout

Choix du pooling (mean vs CLS) :
    On utilise le masked mean pooling plutôt qu'un token [CLS] appris.
    Raisons :
    1. Un CLS randomly initialized avec seulement 2 couches et 40k samples
       n'a pas assez de signal pour apprendre à agréger ~178 tokens.
       Dans BERT, le CLS est pré-entraîné sur des milliards de tokens avec
       l'objectif NSP — ici on part de zéro.
    2. Le mean pooling est empiriquement supérieur au CLS pour les
       représentations de phrases (Reimers & Gurevych 2019, Sentence-BERT),
       même avec un CLS pré-entraîné.
    3. Plus robuste : chaque token contribue, pas de single point of failure.
    4. Zéro paramètre supplémentaire.

d_model est un hyperparamètre libre (typiquement 512).
Les deux modalités sont projetées dans cet espace commun via Linear + LayerNorm.
La projection apprend le sous-espace task-relevant de chaque modalité — la
dimensionnalité intrinsèque utile pour 27 classes est très probablement << 768.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusionModule(nn.Module):
    """
    Fusion cross-modale par self-attention bidirectionnelle + mean pooling.

    Args:
        d_text: dimension des tokens texte (768 CamemBERT, 3072 TextCNN).
        d_image: dimension des patches image (512 ResNet18, 2048 ResNet50).
        d_model: dimension commune de projection et du Transformer.
        n_heads: nombre de têtes d'attention.
        n_layers: nombre de couches TransformerEncoder.
        dim_ff_factor: dim_feedforward = d_model * dim_ff_factor.
        dropout: taux de dropout (attention + FFN + MLP head).
        n_classes: nombre de classes de sortie.
    """

    def __init__(
        self,
        d_text: int = 768,
        d_image: int = 512,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_ff_factor: int = 4,
        dropout: float = 0.3,
        n_classes: int = 27,
    ):
        super().__init__()

        dim_ff = d_model * dim_ff_factor

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) doit être divisible par n_heads ({n_heads})."
            )

        self.d_model = d_model

        # --- Projections modales → espace commun d_model ---
        # Chaque Linear apprend le sous-espace de dimension d_model le plus
        # pertinent pour la tâche. LayerNorm stabilise les magnitudes avant
        # le Transformer (les deux branches peuvent avoir des échelles très
        # différentes : CamemBERT normalise en interne, ResNet non).
        self.proj_text = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
        )
        self.proj_image = nn.Sequential(
            nn.Linear(d_image, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Type embeddings additifs ---
        # Même principe que les segment embeddings de BERT (Devlin et al. 2019).
        # Ajoutés à chaque token pour que le Transformer distingue la modalité
        # source sans altérer la structure séquentielle.
        # 0 = texte, 1 = image.
        self.type_embedding = nn.Embedding(2, d_model)

        # --- Transformer Encoder ---
        # Self-attention bidirectionnelle inter-modale : chaque token (texte
        # ou image) attend à tous les autres. Complexité O((S_t + S_i)² · d).
        # norm_first=True (Pre-LN) : plus stable que Post-LN pour les petits
        # modèles (Xiong et al. 2020), évite les gradients explosifs dans les
        # premières itérations.
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

        # --- Tête de classification ---
        # MLP à une couche cachée (d_model → d_model//2 → n_classes).
        # GELU plutôt que ReLU : approximation douce, meilleure performance
        # empirique en NLP (Hendrycks & Gimpel 2016).
        # Dropout avant la dernière couche pour régulariser.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier uniform pour les projections et la tête.

        Justification : les projections mappent des distributions de magnitudes
        différentes dans le même espace. Xavier normalise Var(output) ≈ Var(input),
        ce qui stabilise l'entrée du Transformer (complété par LayerNorm).
        """
        for module in [self.proj_text, self.proj_image, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        # Type embeddings : init petit pour ne pas dominer les token embeddings
        # au début du training. std=0.02 suit la convention BERT.
        nn.init.normal_(self.type_embedding.weight, std=0.02)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        image_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward complet : projection → concat → attention → mean pool → logits.

        Args:
            text_tokens: (B, S_t, d_text) — tokens texte bruts.
            text_mask: (B, S_t) — 1 = token réel, 0 = padding.
            image_patches: (B, S_i, d_image) — patches image, pas de padding.

        Returns:
            logits: (B, n_classes) — scores bruts avant softmax.
        """
        B = text_tokens.size(0)
        device = text_tokens.device
        S_i = image_patches.size(1)

        # --- Projections → d_model ---
        text_proj = self.proj_text(text_tokens)      # (B, S_t, d_model)
        image_proj = self.proj_image(image_patches)  # (B, S_i, d_model)

        # --- Type embeddings additifs ---
        text_proj = text_proj + self.type_embedding(
            torch.zeros(1, dtype=torch.long, device=device)
        )
        image_proj = image_proj + self.type_embedding(
            torch.ones(1, dtype=torch.long, device=device)
        )

        # --- Concat : [text, image] ---
        sequence = torch.cat([text_proj, image_proj], dim=1)
        # (B, S_t + S_i, d_model)

        # --- Padding mask ---
        # Les patches image sont toujours valides (pas de padding spatial dans
        # des images 224×224). Seul le texte a du padding variable.
        image_mask = torch.ones(B, S_i, device=device)
        full_mask = torch.cat([text_mask.float(), image_mask], dim=1)
        # (B, S_t + S_i)

        # PyTorch convention : src_key_padding_mask True = position ignorée
        padding_mask = full_mask == 0

        # --- Transformer Encoder ---
        output = self.transformer(sequence, src_key_padding_mask=padding_mask)
        # (B, S_t + S_i, d_model)

        # --- Masked mean pooling ---
        # On moyenne uniquement sur les positions non-paddées.
        # Chaque token (texte réel ou patch image) contribue également.
        # Plus robuste qu'un CLS appris from scratch : pas de single point
        # of failure, et le gradient de la loss se distribue uniformément
        # sur tous les tokens au lieu de passer par un seul vecteur.
        mask_expanded = full_mask.unsqueeze(-1)           # (B, S, 1)
        summed = (output * mask_expanded).sum(dim=1)      # (B, d_model)
        counts = mask_expanded.sum(dim=1).clamp(min=1)    # (B, 1)
        pooled = summed / counts                          # (B, d_model)

        return self.head(pooled)                          # (B, n_classes)
