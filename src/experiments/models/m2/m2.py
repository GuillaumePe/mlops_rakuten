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
   - Early stopping intra-fold pour décider du nombre d'arbres optimal
   - MedianPruner pour couper les trials sous-performants
   - TPE multivariate pour modéliser les corrélations entre hyperparams
   - Optionnel : warm-start partiel à partir d'un champion existant
3. Refit final : f_text, f_image, g entraînés sur l'intégralité du train
   - n_estimators fixé via max(best_iters_folds) * 1.1 (marge pour le scaling
     avec n_samples : full_train > fold_size, le modèle peut converger un peu
     plus tard). Pas de split val interne → 100% du train utilisé.

Au predict : f_text_final.predict_proba + f_image_final.predict_proba + tabular
→ concat → g.predict
deprecated:: Phase 1
    M2Stacking est remplacé par M2Baseline dans src/models/assembled/m2_baseline.py.
    Cette classe est conservée pour rollback rapide pendant la transition
    Phase 1 et sera supprimée en fin de Phase 1 (Bloc R).
    
    Migration : utiliser `--experiment m2_baseline` au lieu de `--experiment m2`.
"""
import warnings
warnings.warn(
    "M2Stacking is deprecated and will be removed in Phase 1 end. "
    "Use src.models.assembled.m2_baseline.M2Baseline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from __future__ import annotations
from typing import Optional

import mlflow
import numpy as np
import optuna
import polars as pl
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
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
        warm_start_params: optionnel, dict d'hyperparams à enqueue en trial #0.
            Seules les clés présentes dans l'espace de recherche actuel sont
            utilisées ; les autres sont ignorées silencieusement.
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
        warm_start_params: dict | None = None,
    ):
        self.text_cols = text_cols
        self.image_cols = image_cols
        self.tabular_cols = tabular_cols
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs_optuna = n_jobs_optuna
        self.warm_start_params = warm_start_params

        # Templates des base learners (LogReg avec scaler).
        # On les clone à chaque fit pour avoir des estimateurs frais.
        # multi_class='multinomial' + solver='lbfgs' = MAP estimation propre
        # avec prior gaussien (régularisation L2 par défaut, C=1.0).
        # Pas de class_weight : la métrique cible Rakuten est F1 weighted, donc
        # pas de raison de re-pondérer (le déséquilibre est intentionnellement préservé).
        self._base_text_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
            )),
        ])
        self._base_image_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=2000,
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
        """Returns la fonction objective Optuna (closure).

        Espace de recherche révisé :
        - max_depth (4, 10) : zone réaliste pour ~30k samples méta
        - num_leaves (15, 127) : bornes Microsoft recommandées
        - learning_rate (0.02, 0.15) log : zone empirique optimale pour boosting multi-class
        - min_child_samples (10, 80) : multi-class 27 classes → besoin minimum
          de ~10-30 samples par feuille pour estimation P(y|leaf) stable
        - n_estimators FIXÉ à 2000 + early stopping : best_iteration_ décide
        - bagging_freq=1 fixé en constante : sans ça, subsample était silencieusement ignoré

        Pour chaque trial, on collecte best_iteration_ de chaque fold et on
        stocke max et mean dans user_attrs pour usage au refit final.
        """
        parent_run = mlflow.active_run()
        parent_run_id = parent_run.info.run_id if parent_run else None
        experiment_id = parent_run.info.experiment_id if parent_run else None

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }

            scores = []
            best_iters = []
            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(meta_X, y)):
                model = LGBMClassifier(
                    **params,
                    n_estimators=2000,           # upper bound, early stopping décide
                    bagging_freq=1,              # active subsample
                    objective="multiclass",
                    num_class=self.n_classes,
                    random_state=self.random_state,
                    n_jobs=1,                    # 1 thread par trial (Optuna parallélise)
                    force_col_wise=True,
                    verbosity=-1,
                )
                model.fit(
                    meta_X[tr_idx], y[tr_idx],
                    eval_set=[(meta_X[val_idx], y[val_idx])],
                    callbacks=[
                        early_stopping(50, verbose=False),
                        log_evaluation(0),
                    ],
                )
                preds = model.predict(meta_X[val_idx])
                score = f1_score(y[val_idx], preds, average="weighted")
                scores.append(score)
                best_iters.append(int(model.best_iteration_ or 0))

                # Report score moyen courant à Optuna pour décision de pruning
                trial.report(float(np.mean(scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            best_iters_mean = float(np.mean(best_iters))
            best_iters_max = int(np.max(best_iters))

            # Stockage en user_attrs pour récupération après study.optimize()
            trial.set_user_attr("best_iters_mean", best_iters_mean)
            trial.set_user_attr("best_iters_max", best_iters_max)

            # Log dans nested run MLflow
            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
                parent_run_id=parent_run_id,
                experiment_id=experiment_id,
            ):
                mlflow.log_params(trial.params)
                mlflow.log_metric("optuna_objective", mean_score - 0.25 * std_score)
                mlflow.log_metric("f1_weighted_mean", mean_score)
                mlflow.log_metric("f1_weighted_std", std_score)
                mlflow.log_metric("best_iters_mean", best_iters_mean)
                mlflow.log_metric("best_iters_max", float(best_iters_max))

            # Pénalisation std plus douce (0.25 au lieu de 1.0) :
            # on accepte un peu de variance pour de meilleurs scores moyens
            return mean_score - 0.25 * std_score

        return objective

    # --- Fit final -------------------------------------------------------

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray) -> "M2Stacking":
        """
        Pipeline complet : OOF → HPO Optuna → refit final sur 100% du train.
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
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                multivariate=True,           # modélise les corrélations entre hyperparams
                n_startup_trials=10,         # warmup random avant TPE bayésien
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8,          # pas de pruning sur les 8 premiers trials
                n_warmup_steps=2,            # pas de pruning sur les 2 premiers folds
                interval_steps=1,
            ),
        )

        # 3.bis Warm-start partiel : enqueue les params du @champion si fournis.
        # Garde-fou : n_startup_trials=10 garantit que les 9 trials suivants
        # sont random, évitant un piège de local optimum autour du warm-start.
        if self.warm_start_params is not None:
            search_keys = {
                "num_leaves", "max_depth", "learning_rate", "min_child_samples",
                "reg_alpha", "reg_lambda", "subsample", "colsample_bytree",
            }
            filtered = {k: v for k, v in self.warm_start_params.items() if k in search_keys}
            if filtered:
                study.enqueue_trial(filtered)
                print(f"[M2Stacking] Warm-start enqueued : {filtered}")
            else:
                print("[M2Stacking] warm_start_params fourni mais aucune clé valide, skip")

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

        # 4. Récupérer le nombre d'arbres optimal pour le refit final.
        # On part du max sur les 5 folds (conservateur) puis on ajoute 10% de marge
        # pour absorber le scaling : le refit final tourne sur 100% du train
        # (~38k samples), pas sur 80% comme les folds (~30k). Plus de données →
        # convergence un peu plus tardive, on prévoit donc un budget légèrement plus large.
        best_iters_max_best_trial = study.best_trial.user_attrs.get("best_iters_max", 500)
        best_iter_for_refit = int(best_iters_max_best_trial * 1.1)
        print(f"[M2Stacking] best_iters du best trial (max sur 5 folds) : {best_iters_max_best_trial}")
        print(f"[M2Stacking] n_estimators retenu pour le refit (+10% marge) : {best_iter_for_refit}")
        mlflow.log_metric("model/meta_best_iter_for_refit", best_iter_for_refit)

        # 5. Refit final sur l'intégralité du train
        print("[M2Stacking] Refit final f_text, f_image, meta sur 100% du train...")
        X_text = self._to_numpy(X_train, self.text_cols)
        X_image = self._to_numpy(X_train, self.image_cols)

        self.f_text_ = clone(self._base_text_template).fit(X_text, y_train)
        self.f_image_ = clone(self._base_image_template).fit(X_image, y_train)

        # Meta refit : 100% du train, n_estimators fixé via best_iter_for_refit.
        # Plus de split val interne ni d'early stopping : la cross-validation
        # nous a déjà donné le nombre d'arbres optimal.
        self.meta_ = LGBMClassifier(
            **self.best_params_,
            n_estimators=best_iter_for_refit,
            bagging_freq=1,
            objective="multiclass",
            num_class=self.n_classes,
            random_state=self.random_state,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1,
        )
        self.meta_.fit(meta_X, y_train)
        print(f"[M2Stacking] Meta refit terminé sur {len(y_train)} samples avec {best_iter_for_refit} arbres")

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