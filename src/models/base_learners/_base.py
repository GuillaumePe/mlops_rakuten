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

Persistance (M.4bis) :
4. save_pretrained(path) : sérialise tout l'état (state_dict + métadonnées Python)
   pour que from_pretrained(path) puisse reconstruire un learner fonctionnellement
   identique. API conçue pour s'intégrer avec mlflow.pyfunc.log_model via
   src/models/base_learners/_pyfunc_wrapper.py::BaseLearnerPyfunc.
5. from_pretrained(cls, path) : classmethod inverse exacte de save_pretrained.

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
    # Persistance legacy (M.0) — concrète, NotImplementedError par défaut #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """
        [LEGACY M.0] Sérialise l'état du learner sur disque.

        Non recommandé pour les nouveaux développements. Préférer
        save_pretrained() qui est compatible PyFunc / MLflow registry.

        Maintenu pour compat ascendante avec ResNet18Frozen / CamembertFrozen
        qui ont peut-être leur propre format texte (cf. ResNet18Frozen.save).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.save() non implémenté. "
            "Utiliser save_pretrained() pour la nouvelle API PyFunc-compatible."
        )

    def load(self, path: str | Path) -> "BaseLearner":
        """
        [LEGACY M.0] Charge l'état depuis disque. Retourne self.

        Inverse de save(). Non recommandé pour les nouveaux développements,
        préférer from_pretrained() (classmethod).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.load() non implémenté. "
            "Utiliser from_pretrained() (classmethod) pour la nouvelle API."
        )

    # ------------------------------------------------------------------ #
    # Persistance PyFunc-compatible (M.4bis) — interface uniforme         #
    # ------------------------------------------------------------------ #

    def save_pretrained(self, path: str | Path) -> None:
        """
        Sauvegarde le BaseLearner de façon réversible dans `path`.

        Le dossier `path` doit contenir tout ce qui est nécessaire pour
        reconstruire le learner via `from_pretrained(path)` sans accès
        à un état Python pré-existant :
        - state_dict des nn.Module (poids appris)
        - métadonnées de prétraitement (vocab, transforms args, etc.)
        - hyperparamètres de construction (config JSON)

        Convention : `path` peut être un dossier MLflow artifact, un chemin
        local, ou un mount cloud. L'implémentation ne fait AUCUNE hypothèse
        sur la persistence externe (DVC, R2, S3) — c'est le rôle de
        BaseLearnerExperiment.

        Conçu pour être consommé par
        src/models/base_learners/_pyfunc_wrapper.py::BaseLearnerPyfunc
        qui implémente mlflow.pyfunc.PythonModel.

        Raises:
            NotImplementedError: par défaut. À implémenter dans toute classe
                concrète dont le state est non-trivial (TextCNN, ResNet50PartialFT,
                CamembertLoRA, ResNet18FullFT).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.save_pretrained() non implémenté. "
            "Voir docstring BaseLearner.save_pretrained pour le contrat."
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BaseLearner":
        """
        Reconstruit le BaseLearner depuis `path` (écrit par save_pretrained).

        Le learner retourné est en mode eval (model.eval() côté nn.Module)
        et prêt pour extract_embeddings / predict_proba. Pas de fit() nécessaire.

        Inverse exact de save_pretrained : la composition
        `from_pretrained(p) ∘ save_pretrained(p)` doit être l'identité
        fonctionnelle (les embeddings produits sur la même entrée doivent
        être bit-à-bit identiques aux embeddings d'avant save).

        Raises:
            NotImplementedError: par défaut. À implémenter dans toute classe
                concrète dont le state est non-trivial.
        """
        raise NotImplementedError(
            f"{cls.__name__}.from_pretrained() non implémenté. "
            "Voir docstring BaseLearner.from_pretrained pour le contrat."
        )

    # ------------------------------------------------------------------ #
    # Utilitaires                                                         #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(modality={self.modality}, embed_dim={self.embed_dim})"
