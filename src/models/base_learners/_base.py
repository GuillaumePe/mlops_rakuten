"""
Interface BaseLearner : contrat commun pour tous les base learners
(texte, image, tabulaire) consommés par les modèles assemblés Phase 1.

Cycle de vie standard :
1. fit(X, y, val_split) : training avec early stopping interne pour les deep learners,
   fit direct pour les sklearn-style. Idempotent : appeler fit() deux fois écrase
   l'état précédent.
2. extract_embeddings(X) : forward pass en mode eval, retourne (n, embed_dim).
   Reproductible : appel deux fois sur la même entrée → même sortie (model.eval()
   + torch.no_grad() pour les deep).
3. (optionnel) predict_proba(X) : probabilités softmax (n, n_classes), utilisé
   par StackingLGBM pour générer les meta-features OOF.

Stratégie de fit pour les deep base learners :
- Single train/val split (80/20), early stopping patience configurable
- Pas de K-Fold sur le deep (coût prohibitif) : la robustesse vient du K-Fold
  OOF du LogReg en aval (cf. StackingLGBM)
- Léger biais d'embeddings accepté (early stopping limite l'overfit, et les
  modèles ne sont pas assez grands pour mémoriser ~38k samples)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl


Modality = Literal["text", "image", "tabular"]


class BaseLearner(ABC):
    """
    Contrat commun pour les base learners Phase 1.

    Doit être implémenté par chaque combinaison (modalité, architecture) :
      - text/camembert_frozen.py     : CamemBERT frozen + LogReg (M2 v4)
      - text/camembert_lora.py       : CamemBERT + LoRA r=16 (M2.1)
      - text/textcnn.py              : TextCNN Yoon Kim from scratch (M2.2)
      - image/resnet18_frozen.py     : ResNet18 frozen + LogReg (M2 v4)
      - image/resnet18_full_ft.py    : ResNet18 full fine-tune (M2.1)
      - image/resnet50_partial_ft.py : ResNet50 partial FT (M2.2)
      - tabular/rakuten_tab_features.py : features handcraftées Rakuten

    Convention sur les colonnes attendues dans X :
      - Texte  : colonnes 'designation', 'description' (str, peut être null)
      - Image  : colonne 'imageid' (int, sert à construire le chemin d'image)
      - Tabular: colonnes 'designation', 'description' (string features sources)
                 + 'imageid', 'productid' (identifiants)

    Le pré-traitement (tokenisation, transforms image) est interne au base learner.
    Pas de pré-extraction d'embeddings côté DataModule, pour préserver le
    découplage base_learner ↔ DataModule.
    """

    # ------------------------------------------------------------------ #
    # Propriétés statiques (signature du learner)                        #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def modality(self) -> Modality:
        """Modalité d'entrée : 'text' | 'image' | 'tabular'."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """
        Dimension du vecteur d'embedding retourné par extract_embeddings.
        Connu avant fit() (ex: 768 pour CamemBERT base, 512 pour ResNet18).
        """

    @property
    def n_classes(self) -> int:
        """
        Nombre de classes en sortie pour predict_proba.
        27 pour Rakuten (fixe au niveau du projet).
        """
        return 27

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifiant court du learner (ex: 'camembert_frozen', 'textcnn').
        Sert à logger dans MLflow et à différencier dans les caches.
        """

    # ------------------------------------------------------------------ #
    # Cycle de vie                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fit(
        self,
        X: pl.DataFrame,
        y: np.ndarray,
        val_split: float = 0.2,
    ) -> "BaseLearner":
        """
        Entraîne le base learner.

        Args:
            X: DataFrame avec les colonnes brutes nécessaires à la modalité.
            y: labels encodés [0..n_classes-1], shape (n,).
            val_split: fraction du train à réserver pour validation
                (utilisée par les deep learners pour l'early stopping).
                Ignoré par les learners sklearn-style sans early stopping.

        Returns:
            self (chaînable).

        Raises:
            ValueError: si X ne contient pas les colonnes attendues pour la modalité.
        """

    @abstractmethod
    def extract_embeddings(self, X: pl.DataFrame) -> np.ndarray:
        """
        Forward pass en mode eval.

        Args:
            X: DataFrame avec les colonnes attendues par la modalité.

        Returns:
            Embeddings de shape (len(X), embed_dim), dtype float32.

        Notes:
            - Doit être déterministe (model.eval() + torch.no_grad() pour les deep).
            - L'ordre des lignes en sortie correspond exactement à l'ordre des
              lignes en entrée (pas de shuffle interne).
        """

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """
        Probabilités par classe.

        Implémentation par défaut : NotImplementedError.
        À surcharger par les learners qui exposent un classifier (par ex.
        camembert_lora avec sa tête de classification, textcnn, etc.).
        Non requis pour les learners purement encodeurs (frozen + LogReg
        séparé en aval).

        Returns:
            Probas de shape (len(X), n_classes), dtype float32, sommant à 1
            sur axis=1.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} ne fournit pas predict_proba. "
            "Utiliser extract_embeddings + un classifier externe (LogReg)."
        )

    # ------------------------------------------------------------------ #
    # Persistance (à surcharger ou utiliser pickle par défaut)            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Sérialise l'état du learner sur disque.

        Doit sauvegarder :
        - Poids du modèle (deep) ou pipeline sklearn fitté
        - Tokenizers, vocabulaires, transforms si applicable
        - Métadonnées suffisantes pour load() sans paramètres externes
        """

    @abstractmethod
    def load(self, path: str | Path) -> "BaseLearner":
        """
        Charge l'état depuis disque. Retourne self.

        Doit être l'inverse exact de save() : après load(), le learner doit
        être dans le même état qu'après fit() initial (extract_embeddings
        produit les mêmes valeurs).
        """

    # ------------------------------------------------------------------ #
    # Utilitaires                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modality={self.modality}, embed_dim={self.embed_dim})"