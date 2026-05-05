"""
M2Stacking : architecture de stacking multimodal pour la classification Rakuten.

Architecture :
    x_text   ──► f_text (LogReg)   ──► p_text ∈ Δ^27
    x_image  ──► f_image (LogReg)  ──► p_image ∈ Δ^27
    x_text + x_image ──► extracteur tabulaire ──► t ∈ R^(27*2+nb_features_tabulaires)

    [p_text, p_image, t] ──► g (LightGBM) ──► ŷ

Stratégie d'entraînement :
1. K-fold sur le train pour générer OOF predictions de f_text et f_image
   (évite la fuite de données pour l'entraînement de g)
2. HPO Optuna sur les hyperparams de g (LightGBM), évalué via le même K-fold
3. Refit final : f_text, f_image, g entraînés sur l'intégralité du train

Au predict : f_text_final.predict_proba + f_image_final.predict_proba + tabular
→ concat → g.predict
"""
from __future__ import annotations
from typing import Optional

import mlflow
import numpy as np
import optuna
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class M2Stacking:
    """
    Stacking classifier multimodal LogReg → LightGBM.

    Args:
        text_cols: noms des colonnes d'embeddings texte (ex: text_feat_0..767)
        image_cols: noms des colonnes d'embeddings image (ex: image_feat_0..511)
        tabular_cols: noms des colonnes tabulaires (tab_*)
        n_classes: nombre de classes (27 pour Rakuten)
        n_folds: K du StratifiedKFold pour OOF + HPO
        n_trials: nombre de trials Optuna sur le meta LGBM
        random_state: seed (folds + LGBM)
        n_jobs_optuna: parallélisme des trials Optuna
    """

    def __init__(
        self,
        text_cols: list[str],
        image_cols: list[str],
        tabular_cols: list[str],
        n_classes: int = 27,
        n_folds: int = 5,
        n_trials: int = 30,
        random_state: int = 42,
        n_jobs_optuna: int = 3,
    ):
        self.text_cols = text_cols
        self.image_cols = image_cols
        self.tabular_cols = tabular_cols
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs_optuna = n_jobs_optuna
        
        # Templates des base learners (LogReg avec scaler).
        # On les clone à chaque fit pour avoir des estimateurs frais.
        # multi_class='multinomial' + solver='lbfgs' = MAP estimation propre
        # avec prior gaussien (régularisation L2 par défaut, C=1.0).
        self._base_text_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
            )),
        ])
        self._base_image_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
            )),
        ])

        # Attributs remplis par fit()
        self.f_text_: Optional[Pipeline] = None
        self.f_image_: Optional[Pipeline] = None
        self.meta_: Optional[LGBMClassifier] = None
        self.best_params_: Optional[dict] = None
        self.best_score_: Optional[float] = None
        self.oof_p_text_: Optional[np.ndarray] = None
        self.oof_p_image_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None  # gardé pour calcul de métriques OOF
        self.cv_scores_: Optional[list[float]] = None
    # --- Helpers : extraire les sous-matrices ---------------------------

    def _to_numpy(self, X: pl.DataFrame, cols: list[str]) -> np.ndarray:
        """Sélection + conversion numpy float32 (LightGBM préfère float32)."""
        return X.select(cols).to_numpy().astype(np.float32)

    # --- OOF predictions -------------------------------------------------

    def _compute_oof(
        self, X: pl.DataFrame, y: np.ndarray, skf: StratifiedKFold
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Génère les OOF predictions pour f_text et f_image via K-fold.
        
        Returns:
            (oof_p_text, oof_p_image) : deux matrices (n, n_classes)
        """
        X_text = self._to_numpy(X, self.text_cols)
        X_image = self._to_numpy(X, self.image_cols)

        oof_p_text = np.zeros((len(y), self.n_classes), dtype=np.float32)
        oof_p_image = np.zeros((len(y), self.n_classes), dtype=np.float32)

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_text, y), start=1):
            print(f"[M2Stacking] OOF fold {fold_idx}/{self.n_folds}")

            # f_text fold : LogReg fittée sur tr, prédit sur val
            f_text_fold = clone(self._base_text_template)
            f_text_fold.fit(X_text[tr_idx], y[tr_idx])
            classes_text = f_text_fold.named_steps["logreg"].classes_
            proba_text = f_text_fold.predict_proba(X_text[val_idx])
            # Map les classes vues vers les colonnes de la matrice 27
            for i, c in enumerate(classes_text):
                oof_p_text[val_idx, c] = proba_text[:, i]

            # f_image fold : idem
            f_image_fold = clone(self._base_image_template)
            f_image_fold.fit(X_image[tr_idx], y[tr_idx])
            classes_image = f_image_fold.named_steps["logreg"].classes_
            proba_image = f_image_fold.predict_proba(X_image[val_idx])
            for i, c in enumerate(classes_image):
                oof_p_image[val_idx, c] = proba_image[:, i]

        return oof_p_text, oof_p_image

    # --- HPO Optuna ------------------------------------------------------

    def _build_meta_X(
        self, oof_p_text: np.ndarray, oof_p_image: np.ndarray, X_tab: np.ndarray
    ) -> np.ndarray:
        """Concatène les 3 familles de features en l'ordre canonique."""
        return np.hstack([oof_p_text, oof_p_image, X_tab])

    def _optuna_objective(
        self, meta_X: np.ndarray, y: np.ndarray, skf: StratifiedKFold
    ):
        """Returns la fonction objective Optuna (closure)."""
        parent_run = mlflow.active_run()
        parent_run_id = parent_run.info.run_id if parent_run else None
        tracking_uri = mlflow.get_tracking_uri()  # ← capture l'URI
        experiment_id = parent_run.info.experiment_id if parent_run else None
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            scores = []
            for tr_idx, val_idx in skf.split(meta_X, y):
                model = LGBMClassifier(
                    **params,
                    objective="multiclass",
                    num_class=self.n_classes,
                    random_state=self.random_state,
                    n_jobs=1,            # 1 thread par trial (Optuna parallélise)
                    force_col_wise=True, # plus rapide sur features denses
                    verbosity=-1,
                )
                model.fit(meta_X[tr_idx], y[tr_idx])
                preds = model.predict(meta_X[val_idx])
                scores.append(f1_score(y[val_idx], preds, average="weighted"))

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

            # Log directement dans le worker (compatible n_jobs > 1)
            with mlflow.start_run(run_name=f"trial_{trial.number}", 
                                  nested=True, 
                                  parent_run_id=parent_run_id,
                                  experiment_id=experiment_id,
                                  ):
                mlflow.log_params(trial.params)
                mlflow.log_metric("optuna_objective", mean_score - std_score)
                mlflow.log_metric("f1_weighted_mean", mean_score)
                mlflow.log_metric("f1_weighted_std", std_score)

            # On retourne mean - std pour pénaliser les solutions instables
            # entre folds (préférence pour la robustesse).
            return mean_score - std_score

        return objective

    # --- Fit final -------------------------------------------------------

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray) -> "M2Stacking":
        """
        Pipeline complet : OOF → HPO Optuna → refit final.
        """
        # Une seule instance de skf, réutilisée pour OOF et HPO (cohérence des folds)
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        # 1. OOF predictions
        print(f"[M2Stacking] Génération des OOF predictions ({self.n_folds} folds)...")
        self.oof_p_text_, self.oof_p_image_ = self._compute_oof(X_train, y_train, skf)
        self.y_train_ = y_train.copy()
        # 2. Construire meta_X
        X_tab = self._to_numpy(X_train, self.tabular_cols)
        meta_X = self._build_meta_X(self.oof_p_text_, self.oof_p_image_, X_tab)
        print(f"[M2Stacking] meta_X shape : {meta_X.shape}")

        # 3. HPO Optuna sur le meta LGBM
        print(f"[M2Stacking] Optuna HPO ({self.n_trials} trials)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        study.optimize(
            self._optuna_objective(meta_X, y_train, skf),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs_optuna,
            show_progress_bar=False,
        )
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.cv_scores_ = [t.value for t in study.trials if t.value is not None]
        print(f"[M2Stacking] Best Optuna score: {self.best_score_:.4f}")
        print(f"[M2Stacking] Best params: {self.best_params_}")

        # 4. Refit final sur l'intégralité du train
        print("[M2Stacking] Refit final f_text, f_image, meta sur tout le train...")
        X_text = self._to_numpy(X_train, self.text_cols)
        X_image = self._to_numpy(X_train, self.image_cols)

        self.f_text_ = clone(self._base_text_template).fit(X_text, y_train)
        self.f_image_ = clone(self._base_image_template).fit(X_image, y_train)

        self.meta_ = LGBMClassifier(
            **self.best_params_,
            objective="multiclass",
            num_class=self.n_classes,
            random_state=self.random_state,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1,
        )
        self.meta_.fit(meta_X, y_train)

        return self

    # --- Predict ---------------------------------------------------------

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        Prédit les classes pour un DataFrame de samples.
        Le DataFrame doit contenir les mêmes colonnes que le train (text + image + tabular).
        """
        if self.meta_ is None:
            raise RuntimeError("M2Stacking.fit() doit être appelé avant predict.")

        X_text = self._to_numpy(X, self.text_cols)
        X_image = self._to_numpy(X, self.image_cols)
        X_tab = self._to_numpy(X, self.tabular_cols)

        p_text = self.f_text_.predict_proba(X_text)
        p_image = self.f_image_.predict_proba(X_image)
        meta_X = self._build_meta_X(p_text, p_image, X_tab)

        return self.meta_.predict(meta_X)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Prédit les probabilités. Utile pour ECE et calibration."""
        if self.meta_ is None:
            raise RuntimeError("M2Stacking.fit() doit être appelé avant predict_proba.")

        X_text = self._to_numpy(X, self.text_cols)
        X_image = self._to_numpy(X, self.image_cols)
        X_tab = self._to_numpy(X, self.tabular_cols)

        p_text = self.f_text_.predict_proba(X_text)
        p_image = self.f_image_.predict_proba(X_image)
        meta_X = self._build_meta_X(p_text, p_image, X_tab)

        return self.meta_.predict_proba(meta_X)
    
    @property
    def feature_importances_lgbm_(self) -> np.ndarray:
        """Importance des features dans le meta LGBM (longueur 54+k)."""
        if self.meta_ is None:
            raise RuntimeError("Fit d'abord.")
        return self.meta_.feature_importances_

    @property
    def feature_importances_logreg_text_(self) -> np.ndarray:
        """
        Pour LogReg multinomiale, coef_ a shape (n_classes, n_features).
        On résume en moyenne de |coef| sur les classes → vecteur de taille n_features.
        """
        if self.f_text_ is None:
            raise RuntimeError("Fit d'abord.")
        coef = self.f_text_.named_steps["logreg"].coef_  # (27, 768)
        return np.abs(coef).mean(axis=0)

    @property
    def feature_importances_logreg_image_(self) -> np.ndarray:
        if self.f_image_ is None:
            raise RuntimeError("Fit d'abord.")
        coef = self.f_image_.named_steps["logreg"].coef_  # (27, 512)
        return np.abs(coef).mean(axis=0)