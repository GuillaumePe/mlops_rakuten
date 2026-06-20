"""
StackingLGBM : meta-classifier multimodal generalisé (Phase 1).

Architecture :
    X_text  ──► LogReg (scaler + L2)  ──► p_text  ∈ Δ^27 (OOF)
    X_image ──► LogReg (scaler + L2)  ──► p_image ∈ Δ^27 (OOF)
    X_tab   ────────────────────────────► t       ∈ R^d_tab

    [p_text, p_image, t]  ──► LightGBM méta  ──► ŷ

Stratégie d'entraînement (identique à M2 v4 validée) :
1. K-Fold OOF sur X_text et X_image pour générer p_text et p_image sans fuite
2. HPO Optuna sur le meta LGBM :
   - TPE multivariate (modélise corrélations entre hyperparams)
   - MedianPruner (coupe les trials sous-performants)
   - Early stopping intra-fold (best_iteration_ par fold)
   - Pénalisation soft variance : objective = mean - 0.25 * std
   - Warm-start partiel optionnel (enqueue params du @champion)
3. Refit final sur 100% du train :
   - n_estimators = max(best_iters_folds) * 1.1 (marge pour scaling)
   - Pas de split val interne, pas d'early stopping

Note Phase 1 : ce module remplace `src/experiments/models/m2/m2.py::M2Stacking`
sans changement de signature. Les `text_cols` / `image_cols` / `tabular_cols`
sont produits par les `BaseLearner.extract_embeddings()` dans les `assembled/*.py`.
"""
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
from scipy.optimize import minimize_scalar
from scipy.special import softmax as scipy_softmax

class StackingLGBM:
    """
    Stacking classifier LogReg(text) + LogReg(image) → LightGBM méta.

    Args:
        text_cols: noms des colonnes d'embeddings texte du DataFrame d'entrée
            (typiquement produites par un BaseLearner.extract_embeddings()
             puis renommées en `<learner_name>_feat_0..d` par l'assembled)
        image_cols: noms des colonnes d'embeddings image
        tabular_cols: noms des colonnes tabulaires (tab_*)
        n_classes: nombre de classes (27 pour Rakuten)
        n_folds: K du StratifiedKFold pour OOF + HPO
        n_trials: nombre de trials Optuna sur le meta LGBM
        random_state: seed (folds + LGBM)
        n_jobs_optuna: parallélisme des trials Optuna
        warm_start_params: dict optionnel d'hyperparams à enqueue en trial #0.
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
        logreg_C_text: float = 0.01,
        logreg_C_image: float = 0.1,

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
        self.logreg_C_text = logreg_C_text
        self.logreg_C_image = logreg_C_image


        # Templates des base learners (LogReg avec scaler).
        # multinomial + lbfgs = MAP avec prior gaussien N(0, C·I).
        # C configurable : défaut C_text=0.01 (3072d, sous-déterminé),
        # C_image=0.1 (2048d). Plus C est petit, plus la régularisation est forte.

        # signal pour le LGBM méta. Le niveau 1 doit projeter, pas classifier.
        self._base_text_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=self.logreg_C_text,
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])
        self._base_image_template = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                C=self.logreg_C_image,
                max_iter=2000,
                solver="lbfgs",
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
        self.y_train_: Optional[np.ndarray] = None
        self.cv_scores_: Optional[list[float]] = None
        # Temperature scaling (fitté sur les OOF, appliqué avant _build_meta_X)
        self.T_text_: Optional[float] = None
        self.T_image_: Optional[float] = None
    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _to_numpy(self, X: pl.DataFrame, cols: list[str]) -> np.ndarray:
        """Sélection + conversion numpy float32 (LightGBM préfère float32)."""
        return X.select(cols).to_numpy().astype(np.float32)

    def _build_meta_X(
        self, oof_p_text: np.ndarray, oof_p_image: np.ndarray, X_tab: np.ndarray, derived: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Concatène les familles de features en l'ordre canonique.
        Si derived est fourni (post-A.2), l'ordre est :
            [p_text_cal (27), p_image_cal (27), tab (12), derived (8)] = 74 dims
        Sinon (backward compat) :
            [p_text (27), p_image (27), tab (12)] = 66 dims
        """
        parts = [oof_p_text, oof_p_image, X_tab]
        if derived is not None:
            parts.append(derived)
        return np.hstack(parts)
    # ------------------------------------------------------------------ #
    # Temperature scaling                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fit_temperature(p_oof: np.ndarray, y: np.ndarray) -> float:
        """
        Fitte un scalaire T par minimisation de la NLL sur les OOF.

        p_oof : (n, n_classes) — probas brutes OOF
        y     : (n,)           — labels entiers

        Retourne T > 0. Si T < 1, le modèle était sous-confiant (rare pour
        LogReg). Garde-fou : on valide que NLL_calibrée < NLL_brute ; sinon T=1.
        """
        eps = 1e-7
        log_p = np.log(np.clip(p_oof, eps, 1.0))  # (n, K)

        def nll(T: float) -> float:
            scaled = scipy_softmax(log_p / T, axis=1)  # (n, K)
            return -np.log(scaled[np.arange(len(y)), y] + eps).mean()

        nll_baseline = nll(1.0)
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T_opt = float(result.x)

        # Garde-fou : T ne doit pas aggraver la NLL
        if nll(T_opt) >= nll_baseline:
            print(f"[StackingLGBM] Temperature scaling : NLL ne s'améliore pas "
                  f"(T={T_opt:.3f}), fallback T=1.0")
            return 1.0

        print(f"[StackingLGBM] Temperature scaling : T={T_opt:.4f} "
              f"(NLL {nll_baseline:.4f} → {nll(T_opt):.4f})")
        return T_opt

    @staticmethod
    def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
        """
        Applique le temperature scaling : p' = softmax(log(p) / T).

        Travaille depuis les log-probas pour éviter les -inf sur p≈0.
        T=1.0 est un no-op exact.
        """
        if T == 1.0:
            return p
        eps = 1e-7
        log_p = np.log(np.clip(p, eps, 1.0))
        return scipy_softmax(log_p / T, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Features dérivées                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _derived_features(
        p_text: np.ndarray, p_image: np.ndarray
    ) -> np.ndarray:
        """
        Calcule 8 features dérivées à partir des probas calibrées.

        Toutes calculées APRÈS temperature scaling pour refléter
        l'incertitude réelle des bases.

        Features (par sample) :
            0 - agreement    : 1 si argmax(text)==argmax(image)
            1 - margin_text  : top1 - top2 de p_text
            2 - margin_image : top1 - top2 de p_image
            3 - max_text     : max(p_text)  — confiance brute texte
            4 - max_image    : max(p_image) — confiance brute image
            5 - entropy_text : -sum(p log p) / log(K) normalisée en [0,1]
            6 - entropy_image: idem
            7 - kl_text_img  : KL(p_text || p_image) — divergence inter-bases

        Returns:
            (n, 8) float32
        """
        eps = 1e-7
        n, K = p_text.shape
        log_K = np.log(K)

        # argmax accord
        agreement = (np.argmax(p_text, axis=1) == np.argmax(p_image, axis=1)
                     ).astype(np.float32)

        # Tri descend pour top1/top2
        sorted_text = np.sort(p_text, axis=1)[:, ::-1]
        sorted_image = np.sort(p_image, axis=1)[:, ::-1]
        margin_text = (sorted_text[:, 0] - sorted_text[:, 1]).astype(np.float32)
        margin_image = (sorted_image[:, 0] - sorted_image[:, 1]).astype(np.float32)

        max_text = sorted_text[:, 0].astype(np.float32)
        max_image = sorted_image[:, 0].astype(np.float32)

        # Entropie normalisée (0 = certitude, 1 = uniforme)
        entropy_text = (-np.sum(p_text * np.log(p_text + eps), axis=1) / log_K
                        ).astype(np.float32)
        entropy_image = (-np.sum(p_image * np.log(p_image + eps), axis=1) / log_K
                         ).astype(np.float32)

        # KL(p_text || p_image) — asymétrique, mesure comment p_image
        # s'écarte de p_text. Non symétrique intentionnellement : le texte
        # est le signal dominant, on mesure l'écart de l'image par rapport à lui.
        kl = np.sum(
            p_text * np.log((p_text + eps) / (p_image + eps)), axis=1
        ).astype(np.float32)

        return np.column_stack([
            agreement, margin_text, margin_image,
            max_text, max_image,
            entropy_text, entropy_image,
            kl,
        ])
    # ------------------------------------------------------------------ #
    # OOF predictions                                                    #
    # ------------------------------------------------------------------ #

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
            print(f"[StackingLGBM] OOF fold {fold_idx}/{self.n_folds}")

            f_text_fold = clone(self._base_text_template)
            f_text_fold.fit(X_text[tr_idx], y[tr_idx])
            classes_text = f_text_fold.named_steps["logreg"].classes_
            proba_text = f_text_fold.predict_proba(X_text[val_idx])
            for i, c in enumerate(classes_text):
                oof_p_text[val_idx, c] = proba_text[:, i]

            f_image_fold = clone(self._base_image_template)
            f_image_fold.fit(X_image[tr_idx], y[tr_idx])
            classes_image = f_image_fold.named_steps["logreg"].classes_
            proba_image = f_image_fold.predict_proba(X_image[val_idx])
            for i, c in enumerate(classes_image):
                oof_p_image[val_idx, c] = proba_image[:, i]

        return oof_p_text, oof_p_image

    # ------------------------------------------------------------------ #
    # HPO Optuna                                                         #
    # ------------------------------------------------------------------ #

    def _optuna_objective(
        self, meta_X: np.ndarray, y: np.ndarray, skf: StratifiedKFold
    ):
        """
        Objective Optuna (closure).

        Espace de recherche révisé (Phase 0 v4) :
        - max_depth (4, 10) : zone réaliste pour ~30k samples méta
        - num_leaves (15, 127) : bornes Microsoft recommandées
        - learning_rate (0.02, 0.15) log : zone empirique optimale multiclass
        - min_child_samples (10, 80) : multi-class 27 classes → besoin minimum
          de ~10-30 samples par feuille pour P(y|leaf) stable
        - n_estimators FIXÉ à 2000 + early stopping : best_iteration_ décide
        - bagging_freq=1 fixé : sans ça, subsample est silencieusement ignoré
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
                    n_jobs=1,                    # 1 thread / trial (Optuna parallélise)
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

                # Report score moyen courant pour décision de pruning
                trial.report(float(np.mean(scores)), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            best_iters_mean = float(np.mean(best_iters))
            best_iters_max = int(np.max(best_iters))

            trial.set_user_attr("best_iters_mean", best_iters_mean)
            trial.set_user_attr("best_iters_max", best_iters_max)

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

            # Pénalisation std douce (0.25 vs 1.0) : on accepte un peu de
            # variance pour de meilleurs scores moyens
            return mean_score - 0.25 * std_score

        return objective

    # ------------------------------------------------------------------ #
    # Fit                                                                #
    # ------------------------------------------------------------------ #

    def fit(self, X_train: pl.DataFrame, y_train: np.ndarray) -> "StackingLGBM":
        """
        Pipeline complet : OOF → HPO Optuna → refit final sur 100% du train.
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        # 1. OOF predictions
        print(f"[StackingLGBM] Génération des OOF predictions ({self.n_folds} folds)...")
        self.oof_p_text_, self.oof_p_image_ = self._compute_oof(X_train, y_train, skf)
        self.y_train_ = y_train.copy()

        # 1.bis — Temperature scaling sur les OOF
        print("[StackingLGBM] Fitting temperature scaling sur les OOF...")
        self.T_text_ = self._fit_temperature(self.oof_p_text_, y_train)
        self.T_image_ = self._fit_temperature(self.oof_p_image_, y_train)
        mlflow.log_param("temperature/T_text", round(self.T_text_, 4))
        mlflow.log_param("temperature/T_image", round(self.T_image_, 4))

        # 1.ter — Appliquer T + features dérivées
        oof_p_text_cal = self._apply_temperature(self.oof_p_text_, self.T_text_)
        oof_p_image_cal = self._apply_temperature(self.oof_p_image_, self.T_image_)
        derived = self._derived_features(oof_p_text_cal, oof_p_image_cal)

        # 2. Construire meta_X
        X_tab = self._to_numpy(X_train, self.tabular_cols)
        meta_X = self._build_meta_X(oof_p_text_cal, oof_p_image_cal, X_tab, derived)
        print(f"[StackingLGBM] meta_X shape : {meta_X.shape}")

        # 3. HPO Optuna sur le meta LGBM
        print(f"[StackingLGBM] Optuna HPO ({self.n_trials} trials)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                multivariate=True,           # modélise corrélations entre hyperparams
                n_startup_trials=10,         # warmup random avant TPE bayésien
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=8,          # pas de pruning sur les 8 premiers trials
                n_warmup_steps=2,            # pas de pruning sur les 2 premiers folds
                interval_steps=1,
            ),
        )

        # 3.bis Warm-start partiel : enqueue params du @champion si fourni.
        # Garde-fou : n_startup_trials=10 garantit que les 9 trials suivants sont
        # random, évitant un piège de local optimum autour du warm-start.
        if self.warm_start_params is not None:
            search_keys = {
                "num_leaves", "max_depth", "learning_rate", "min_child_samples",
                "reg_alpha", "reg_lambda", "subsample", "colsample_bytree",
            }
            filtered = {k: v for k, v in self.warm_start_params.items() if k in search_keys}
            if filtered:
                study.enqueue_trial(filtered)
                print(f"[StackingLGBM] Warm-start enqueued : {filtered}")
            else:
                print("[StackingLGBM] warm_start_params fourni mais aucune clé valide, skip")

        study.optimize(
            self._optuna_objective(meta_X, y_train, skf),
            n_trials=self.n_trials,
            n_jobs=self.n_jobs_optuna,
            show_progress_bar=False,
        )
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.cv_scores_ = [t.value for t in study.trials if t.value is not None]
        print(f"[StackingLGBM] Best Optuna score: {self.best_score_:.4f}")
        print(f"[StackingLGBM] Best params: {self.best_params_}")

        # 4. n_estimators pour le refit final.
        # max sur les 5 folds (conservateur) + 10% de marge pour absorber le
        # scaling : le refit final tourne sur 100% du train (~38k), pas sur
        # 80% comme les folds (~30k). Plus de données → convergence un peu plus
        # tardive, budget légèrement plus large.
        best_iters_max_best_trial = study.best_trial.user_attrs.get("best_iters_max", 500)
        best_iter_for_refit = int(best_iters_max_best_trial * 1.1)
        print(f"[StackingLGBM] best_iters du best trial (max sur folds) : {best_iters_max_best_trial}")
        print(f"[StackingLGBM] n_estimators retenu pour refit (+10%) : {best_iter_for_refit}")
        mlflow.log_metric("model/meta_best_iter_for_refit", best_iter_for_refit)

        # 5. Refit final sur 100% du train
        print("[StackingLGBM] Refit final f_text, f_image, meta sur 100% du train...")
        X_text = self._to_numpy(X_train, self.text_cols)
        X_image = self._to_numpy(X_train, self.image_cols)

        self.f_text_ = clone(self._base_text_template).fit(X_text, y_train)
        self.f_image_ = clone(self._base_image_template).fit(X_image, y_train)

        # Meta refit : 100% train, n_estimators fixé via best_iter_for_refit.
        # Plus de split val interne ni d'early stopping : la CV a déjà donné
        # le nombre d'arbres optimal.
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
        print(f"[StackingLGBM] Meta refit terminé sur {len(y_train)} samples avec {best_iter_for_refit} arbres")

        return self

    # ------------------------------------------------------------------ #
    # Predict                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Classes prédites pour un DataFrame de samples."""
        if self.meta_ is None:
            raise RuntimeError("StackingLGBM.fit() doit être appelé avant predict.")

        X_text = self._to_numpy(X, self.text_cols)
        X_image = self._to_numpy(X, self.image_cols)
        X_tab = self._to_numpy(X, self.tabular_cols)

        p_text = self.f_text_.predict_proba(X_text)
        p_image = self.f_image_.predict_proba(X_image)
        T_text = getattr(self, "T_text_", 1.0) or 1.0
        T_image = getattr(self, "T_image_", 1.0) or 1.0
        p_text_cal = self._apply_temperature(p_text, T_text)
        p_image_cal = self._apply_temperature(p_image, T_image)
        has_calibration = hasattr(self, "T_text_")
        derived = self._derived_features(p_text_cal, p_image_cal) if has_calibration else None
        meta_X = self._build_meta_X(p_text_cal, p_image_cal, X_tab, derived)

        return self.meta_.predict(meta_X)

    def predict_proba(self, X: pl.DataFrame) -> np.ndarray:
        """Probabilités softmax. Utile pour ECE et calibration."""
        if self.meta_ is None:
            raise RuntimeError("StackingLGBM.fit() doit être appelé avant predict_proba.")

        X_text = self._to_numpy(X, self.text_cols)
        X_image = self._to_numpy(X, self.image_cols)
        X_tab = self._to_numpy(X, self.tabular_cols)

        p_text = self.f_text_.predict_proba(X_text)
        p_image = self.f_image_.predict_proba(X_image)
        T_text = getattr(self, "T_text_", 1.0) or 1.0
        T_image = getattr(self, "T_image_", 1.0) or 1.0
        p_text_cal = self._apply_temperature(p_text, T_text)
        p_image_cal = self._apply_temperature(p_image, T_image)
        has_calibration = hasattr(self, "T_text_")
        derived = self._derived_features(p_text_cal, p_image_cal) if has_calibration else None
        meta_X = self._build_meta_X(p_text_cal, p_image_cal, X_tab, derived)

        return self.meta_.predict_proba(meta_X)

    # ------------------------------------------------------------------ #
    # Feature importances (introspection)                                #
    # ------------------------------------------------------------------ #

    @property
    def feature_importances_lgbm_(self) -> np.ndarray:
        """Importances dans le meta LGBM (longueur 2*n_classes + n_tab)."""
        if self.meta_ is None:
            raise RuntimeError("Fit d'abord.")
        return self.meta_.feature_importances_

    @property
    def feature_importances_logreg_text_(self) -> np.ndarray:
        """
        LogReg multinomiale : coef_ shape (n_classes, n_features).
        On résume en moyenne de |coef| sur les classes → vecteur (n_features,).
        """
        if self.f_text_ is None:
            raise RuntimeError("Fit d'abord.")
        coef = self.f_text_.named_steps["logreg"].coef_
        return np.abs(coef).mean(axis=0)

    @property
    def feature_importances_logreg_image_(self) -> np.ndarray:
        if self.f_image_ is None:
            raise RuntimeError("Fit d'abord.")
        coef = self.f_image_.named_steps["logreg"].coef_
        return np.abs(coef).mean(axis=0)