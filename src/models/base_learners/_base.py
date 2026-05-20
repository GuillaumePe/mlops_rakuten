"""
Interface BaseLearner : contrat commun pour tous les base learners
(texte, image, tabulaire) consommés par les modèles assemblés Phase 1.

Cycle de vie standard :
1. fit(X_train, y_train, X_val, y_val) : training avec early stopping interne
   pour les deep learners, fit direct pour les sklearn-style. Idempotent :
   appeler fit() deux fois écrase l'état précédent. Les frozen learners
   ignorent X_val/y_val (passe-plats).
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
        X_train: pl.DataFrame,
        y_train: np.ndarray,
        X_val: pl.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "BaseLearner":
        """
        Entraîne le base learner.

        Args:
            X_train: DataFrame d'entraînement avec colonnes brutes selon modalité.
            y_train: labels encodés [0..n_classes-1], shape (len(X_train),).
            X_val: DataFrame de validation (optionnel). Si fourni, utilisé par
                les deep learners pour l'early stopping et le monitoring.
                Les frozen learners ignorent cet argument.
            y_val: labels de validation, shape (len(X_val),). Doit être fourni
                si X_val l'est.

        Returns:
            self (chaînable).

        Raises:
            ValueError: si X_train ne contient pas les colonnes attendues,
                ou si (X_val is None) XOR (y_val is None).

        Convention sur le split (M.0) :
            Le DataModule fournit en standard un split 80/20 stratifié seed=42
            sur train_pool_effective. Le call-site typique :
                X_train, y_train = dm.get_sklearn_data("train")
                X_val,   y_val   = dm.get_sklearn_data("val")
                learner.fit(X_train, y_train, X_val, y_val)
            En notebook exploratoire, fit(X, y) reste valide grâce aux défauts
            None ; les deep learners utilisent alors un fallback interne.
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
    # Utilitaires                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modality={self.modality}, embed_dim={self.embed_dim})"