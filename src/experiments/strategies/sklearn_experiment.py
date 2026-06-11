"""
SklearnExperiment : orchestrateur de fit + tracking MLflow pour les modèles
sklearn-style (M2 et variantes).

Sépare strictement :
- le modèle (M2Stacking) : pur, ne sait rien de MLflow
- l'orchestration : MLflow run parent, nested runs Optuna, métriques, artefacts
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # pas de display, on save en PNG uniquement
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
)

from src.experiments.strategies._metrics import (
    brier_score_multiclass,
    count_model_params,
    expected_calibration_error,
    measure_inference_latency,
    per_class_metrics,
    reliability_diagram_data,
    stacking_analysis,
    top_confusions,
)

from src.experiments.visualizations import (
    default_group_of_m2,
    log_calibration_plots_to_mlflow,
    log_feature_importance_to_mlflow,
    one_vs_rest_plot,
    select_hard_classes,
)


class SklearnExperiment:
    """
    Wrap un modèle sklearn-style pour le faire tourner avec tracking MLflow.

    Args:
        model_factory: callable() → modèle non-fitté. Le callable reçoit en
                       paramètre l'optuna_callback à passer au modèle.
        run_name: nom du run MLflow parent
        tags: dict de tags MLflow (model_family, base_text, base_image, etc.)
    """

    def __init__(
        self,
        model_factory,  # Callable[[Callable], M2Stacking]
        run_name: str,
        tags: dict | None = None,
    ):
        self.model_factory = model_factory
        self.run_name = run_name
        self.tags = tags or {}
        self.model = None  # rempli par fit()

    # --- fit -------------------------------------------------------------

    def fit(self, datamodule):
        with mlflow.start_run(run_name=self.run_name) as run:
            print(f"[SklearnExperiment] MLflow run started: {run.info.run_id}")
            if self.tags:
                mlflow.set_tags(self.tags)

            # 1. Récupérer les données
            # include_raw : True si le modèle a besoin des colonnes brutes
            # (text, imageid, productid) pour ses BaseLearners non-frozen (TextCNN,
            # ResNet50PartialFT, etc.). Lu depuis les tags YAML pour découpler
            # le DataModule de la connaissance du modèle.
            include_raw = self.tags.get("requires_raw_data", "false").lower() == "true"
            X_train, y_train = datamodule.get_sklearn_data("train", include_raw=include_raw)
            X_val, y_val = datamodule.get_sklearn_data("val", include_raw=include_raw)

            # 2. Construire le modèle
            self.model = self.model_factory(None)

            # 3. Fit (HPO Optuna interne, log les nested runs)
            self.model.fit(X_train, y_train)

            # 4. Log params best
            mlflow.log_params({f"best/{k}": v for k, v in self.model.best_params_.items()})
            mlflow.log_metric("optuna_best_score", self.model.best_score_)

            # 5. Métriques sur OOF (train) et val (jamais touché par Optuna)
            self._log_oof_metrics()
            self._log_val_metrics(X_val, y_val)
            self._log_stacking_analysis(X_val, y_val)
            self._log_per_class_artifacts(X_val, y_val)
            self._log_optuna_stability()
            self._log_calibration(X_val, y_val)
            self._log_inference_metrics(X_val)
            self._log_artifacts(X_val, y_val)

            # 6. Référence pour le PSI gold vs val
            # On utilise les prédictions du val (jamais vu en training) comme
            # référence représentative du pool d'entraînement, sans biais d'overfit
            # qu'aurait predict(X_train) puisque le modèle a été entraîné dessus.
            p_val_ref = self.model.predict_proba(X_val)

            # 7. Évaluation gold (arbitre transverse @champion)
            y_gold, p_gold = self._log_eval_gold(datamodule, p_pool_ref=p_val_ref)

            # 8. Métadonnées learning curve
            #    Permettra de tracer F1_gold = f(train_pool_size) à travers les runs
            mlflow.set_tag("train_batches", str(list(datamodule.train_batches)))
            mlflow.set_tag("learning_curve_step", len(datamodule.train_batches))
            mlflow.log_metric("train_pool_size", len(datamodule._df_train_pool))

            # 9. Plots de calibration sur gold (reliability, confidence, margin)
            log_calibration_plots_to_mlflow(
                y_true=y_gold,
                p_pred=p_gold,
                prefix="calibration/gold",
            )

            # 10. One-vs-Rest sur les classes les plus difficiles du gold
            hard = select_hard_classes(y_gold, p_gold, top_k=6, min_support=20)
            mlflow.log_param("hard_classes_gold", str(hard))
            fig = one_vs_rest_plot(
                y_gold, p_gold,
                classes_to_plot=hard,
                title=f"OvR classes difficiles — gold (train_batches={list(datamodule.train_batches)})",
            )
            mlflow.log_figure(fig, "class_breakdown/one_vs_rest_hard_classes_gold.png")
            plt.close(fig)

            # 11. Feature importance du méta-learner LightGBM, agrégée par groupe.
            # On reconstruit les noms de features dans l'ordre exact où M2Stacking
            # les concatène dans _build_meta_X : [p_text | p_image | tabular]
            meta_feature_names = (
                [f"p_text_class_{c}" for c in range(self.model.n_classes)]
                + [f"p_image_class_{c}" for c in range(self.model.n_classes)]
                + [f"tab__{name}" for name in self.model.tabular_cols]
                + [
                    "derived__agreement",
                    "derived__margin_text",
                    "derived__margin_image",
                    "derived__max_text",
                    "derived__max_image",
                    "derived__entropy_text",
                    "derived__entropy_image",
                    "derived__kl_text_img",
                ]
            )
            log_feature_importance_to_mlflow(
                model=self.model.meta_,
                feature_names=meta_feature_names,
                group_of=default_group_of_m2,
                prefix="feature_importance",
                top_n_individual=15,
            )

            # 12. Registry + promotion @champion
            model_name = self.tags.get("registry_model_name", "rakuten-m2-stacking")
            print(f"[SklearnExperiment] Enregistrement modèle dans registry '{model_name}'...")
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="model",
                registered_model_name=model_name,
            )
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='{model_name}' AND run_id='{run.info.run_id}'"
            )
            if not versions:
                print(
                    f"[SklearnExperiment] WARN: aucune version trouvée pour run {run.info.run_id}, "
                    "skip promotion"
                )
            else:
                candidate_version = versions[0].version
                from src.models.utils import evaluate_promotion_via_logged_metrics
                epsilon = float(self.tags.get("promotion_epsilon", 0.005))
                result = evaluate_promotion_via_logged_metrics(
                    model_name=model_name,
                    candidate_version=candidate_version,
                    metric_key="eval_gold/f1_weighted",
                    epsilon=epsilon,
                )
                print(result)
                mlflow.set_tag("promotion_promoted", str(result.promoted).lower())
                mlflow.set_tag("promotion_reason", result.reason)
                if result.gain is not None:
                    mlflow.log_metric("promotion_gain_f1", result.gain)

        return self

    # --- Logging helpers -------------------------------------------------

    def _log_oof_metrics(self):
        """Métriques de f_text et f_image en isolé, calculées sur les OOF du train."""
        y = self.model.y_train_
        # Argmax des OOF probas → prédiction de chaque base learner seul
        preds_text = np.argmax(self.model.oof_p_text_, axis=1)
        preds_image = np.argmax(self.model.oof_p_image_, axis=1)

        mlflow.log_metric("base/f1_text_macro_oof", f1_score(y, preds_text, average="macro"))
        mlflow.log_metric("base/f1_text_weighted_oof", f1_score(y, preds_text, average="weighted"))
        mlflow.log_metric("base/accuracy_text_oof", accuracy_score(y, preds_text))

        mlflow.log_metric("base/f1_image_macro_oof", f1_score(y, preds_image, average="macro"))
        mlflow.log_metric("base/f1_image_weighted_oof", f1_score(y, preds_image, average="weighted"))
        mlflow.log_metric("base/accuracy_image_oof", accuracy_score(y, preds_image))

        # F1 par classe pour chaque modalité (artifact JSON pour Streamlit)
        n_classes = self.model.n_classes
        f1_text_per_class = f1_score(y, preds_text, labels=range(n_classes), average=None, zero_division=0)
        f1_image_per_class = f1_score(y, preds_image, labels=range(n_classes), average=None, zero_division=0)
        mlflow.log_dict(
            {
                "f1_text_per_class": f1_text_per_class.tolist(),
                "f1_image_per_class": f1_image_per_class.tolist(),
            },
            "base_learners_per_class.json",
        )

    def _log_val_metrics(self, X_val: pl.DataFrame, y_val: np.ndarray):
        """Métriques globales sur le val set (jamais touché par Optuna)."""
        preds = self.model.predict(X_val)
        probas = self.model.predict_proba(X_val)

        mlflow.log_metric("final/accuracy", accuracy_score(y_val, preds))
        mlflow.log_metric("final/f1_macro", f1_score(y_val, preds, average="macro"))
        mlflow.log_metric("final/f1_weighted", f1_score(y_val, preds, average="weighted"))
        mlflow.log_metric("final/log_loss", log_loss(y_val, probas, labels=range(self.model.n_classes)))
        mlflow.log_metric("final/brier_score", brier_score_multiclass(y_val, probas))

        f1_per_class = f1_score(y_val, preds, average=None)
        for class_idx, score in enumerate(f1_per_class):
            mlflow.log_metric(f"final/f1_class_{class_idx}", score)

    def _log_per_class_artifacts(self, X_val: pl.DataFrame, y_val: np.ndarray):
        preds = self.model.predict(X_val)
        per_class = per_class_metrics(y_val, preds, n_classes=self.model.n_classes)
        confusions = top_confusions(y_val, preds, n_classes=self.model.n_classes, top_k=10)
        mlflow.log_dict(
            {"per_class": per_class, "top_confusions": confusions},
            "per_class_analysis.json",
        )

    def _log_calibration(self, X_val: pl.DataFrame, y_val: np.ndarray):
        """
        Calibration sur le val (legacy : ECE + reliability simple).
        Pour le gold, des plots plus riches sont générés via
        log_calibration_plots_to_mlflow (voir fit()).
        """
        probas = self.model.predict_proba(X_val)
        ece = expected_calibration_error(y_val, probas, n_bins=10)
        mlflow.log_metric("final/ece", ece)

        reliability = reliability_diagram_data(y_val, probas, n_bins=10)
        mlflow.log_dict(reliability, "reliability_diagram.json")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.plot(
            reliability["confidence_per_bin"], reliability["accuracy_per_bin"],
            "o-", label=f"Model (ECE={ece:.3f})",
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability diagram (val)")
        ax.legend()
        ax.set_aspect("equal")
        mlflow.log_figure(fig, "reliability_diagram_val.png")
        plt.close(fig)

    def _log_stacking_analysis(self, X_val: pl.DataFrame, y_val: np.ndarray):
        """Mesure l'apport du méta-learner par rapport aux base learners."""
        n_classes = self.model.n_classes
        X_text = self.model._to_numpy(X_val, self.model.text_cols)
        X_image = self.model._to_numpy(X_val, self.model.image_cols)

        p_text = np.zeros((len(X_val), n_classes), dtype=np.float32)
        classes_text = self.model.f_text_.named_steps["logreg"].classes_
        proba_text = self.model.f_text_.predict_proba(X_text)
        for i, c in enumerate(classes_text):
            p_text[:, c] = proba_text[:, i]

        p_image = np.zeros((len(X_val), n_classes), dtype=np.float32)
        classes_image = self.model.f_image_.named_steps["logreg"].classes_
        proba_image = self.model.f_image_.predict_proba(X_image)
        for i, c in enumerate(classes_image):
            p_image[:, c] = proba_image[:, i]

        preds_meta = self.model.predict(X_val)

        analysis = stacking_analysis(y_val, p_text, p_image, preds_meta)
        for k, v in analysis.items():
            mlflow.log_metric(f"stacking/{k}", v)
        mlflow.log_dict(analysis, "stacking_analysis.json")

    def _log_optuna_stability(self):
        """Stabilité des objectives Optuna à travers les trials."""
        if self.model.cv_scores_:
            scores = np.array(self.model.cv_scores_)
            mlflow.log_metric("stability/cv_score_std", float(scores.std()))
            mlflow.log_metric("stability/cv_score_min", float(scores.min()))
            mlflow.log_metric("stability/cv_score_max", float(scores.max()))
            mlflow.log_metric("stability/cv_score_range", float(scores.max() - scores.min()))

    def _log_inference_metrics(self, X_val: pl.DataFrame):
        """Latence et taille du modèle."""
        latency = measure_inference_latency(self.model, X_val, n_warmup=5, n_repeat=50)
        for k, v in latency.items():
            mlflow.log_metric(f"inference/{k}", v)
        mlflow.log_metric("model/n_params", count_model_params(self.model))

    def _log_artifacts(self, X_val: pl.DataFrame, y_val: np.ndarray):
        """Confusion matrix + classification report + feature importances brutes."""
        preds = self.model.predict(X_val)

        cm = confusion_matrix(y_val, preds)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=range(cm.shape[0]), yticklabels=range(cm.shape[0]))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {self.run_name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report = classification_report(y_val, preds, digits=3)
        mlflow.log_text(report, "classification_report.txt")

        self._log_feature_importance_plot(
            self.model.feature_importances_lgbm_, "feature_importance_lgbm.png", top_n=30
        )
        self._log_feature_importance_plot(
            self.model.feature_importances_logreg_text_, "feature_importance_logreg_text.png", top_n=30
        )
        self._log_feature_importance_plot(
            self.model.feature_importances_logreg_image_, "feature_importance_logreg_image.png", top_n=30
        )

    def _log_feature_importance_plot(self, importances: np.ndarray, name: str, top_n: int = 30):
        """Plot brut top-N features par importance, utilisé pour les bases LogReg."""
        idx = np.argsort(importances)[-top_n:][::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
        ax.barh(range(len(idx)), importances[idx][::-1])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([f"feat_{i}" for i in idx][::-1])
        ax.set_xlabel("Importance")
        ax.set_title(name.replace(".png", "").replace("_", " "))
        plt.tight_layout()
        mlflow.log_figure(fig, name)
        plt.close(fig)

    # --- Métriques transverses (utilisées pour gold) ---------------------

    @staticmethod
    def _compute_perf_metrics(y_true, y_pred, y_proba, n_classes) -> dict:
        """Métriques de performance standard."""
        return {
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_proba, labels=range(n_classes))),
        }

    @staticmethod
    def _compute_drift_metrics(p_target: np.ndarray, p_ref: np.ndarray | None = None) -> dict:
        """
        Métriques de drift sur les sorties du modèle (pas sur les inputs).

        - entropy_mean : H(p̂) moyen sur l'ensemble target. Augmente si modèle moins certain.
        - margin_mean : (p_top1 - p_top2) moyen. Diminue si modèle moins confiant.
        - psi : Population Stability Index sur la distribution argmax target vs ref.
                Seuils usuels : < 0.1 stable, 0.1-0.25 alerte modérée, > 0.25 alerte forte.

        Si p_ref est None, le PSI n'est pas calculé.
        """
        eps = 1e-12
        metrics = {}

        # Entropie prédictive : H(p) = -Σ p_k log(p_k)
        entropy = -np.sum(p_target * np.log(p_target + eps), axis=1)
        metrics["entropy_mean"] = float(entropy.mean())
        metrics["entropy_std"] = float(entropy.std())

        # Margin top1 - top2
        sorted_probs = np.sort(p_target, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2]
        metrics["margin_mean"] = float(margin.mean())
        metrics["margin_std"] = float(margin.std())

        # PSI sur distribution argmax (compare target vs ref) — optionnel
        if p_ref is not None and p_ref.shape[1] == p_target.shape[1]:
            n_classes = p_target.shape[1]
            preds_target = np.argmax(p_target, axis=1)
            preds_ref = np.argmax(p_ref, axis=1)

            eps_psi = 1e-6
            hist_target = (np.bincount(preds_target, minlength=n_classes) + eps_psi).astype(float)
            hist_target /= hist_target.sum()
            hist_ref = (np.bincount(preds_ref, minlength=n_classes) + eps_psi).astype(float)
            hist_ref /= hist_ref.sum()

            psi = float(np.sum((hist_target - hist_ref) * np.log(hist_target / hist_ref)))
            metrics["psi"] = psi

        return metrics

    def _log_eval_gold(self, dm, p_pool_ref: np.ndarray | None = None):
        """
        Évaluation sur le gold test set transverse (arbitre @champion).

        Le gold est le seul ensemble eval — les autres batches alimentent le training
        au fil des runs (learning curve).

        Args:
            dm: DataModule
            p_pool_ref: optionnel, probas de référence pour le PSI (typiquement
                les prédictions sur le val set du run courant)

        Returns:
            (y_gold, probas) pour réutilisation downstream (plots calibration, OvR)
        """
        include_raw = self.tags.get("requires_raw_data", "false").lower() == "true"
        X_gold, y_gold = dm.get_eval_data("gold", include_raw=include_raw)
        preds = self.model.predict(X_gold)
        probas = self.model.predict_proba(X_gold)

        perf = self._compute_perf_metrics(y_gold, preds, probas, self.model.n_classes)
        drift = self._compute_drift_metrics(probas, p_ref=p_pool_ref)
        ece = expected_calibration_error(y_gold, probas, n_bins=10)

        for k, v in perf.items():
            mlflow.log_metric(f"eval_gold/{k}", v)
        for k, v in drift.items():
            mlflow.log_metric(f"eval_gold/{k}", v)
        mlflow.log_metric("eval_gold/ece", ece)

        psi_str = f" psi={drift['psi']:.4f}" if "psi" in drift else ""
        print(f"  eval_gold : f1_weighted={perf['f1_weighted']:.4f} ece={ece:.4f}{psi_str}")
        return y_gold, probas

    # --- predict / evaluate ----------------------------------------------

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() d'abord.")
        return self.model.predict(X)

    def evaluate(self, datamodule) -> dict:
        """
        Évalue le modèle sur le gold test set (arbitre transverse @champion).
        Pour évaluer sur d'autres ensembles, utiliser directement
        model.predict + get_eval_data dans le code appelant.
        """
        if self.model is None:
            raise RuntimeError("fit() d'abord.")
        X_gold, y_gold = datamodule.get_eval_data("gold")
        preds = self.model.predict(X_gold)
        return {
            "accuracy": accuracy_score(y_gold, preds),
            "f1_macro": f1_score(y_gold, preds, average="macro"),
            "f1_weighted": f1_score(y_gold, preds, average="weighted"),
        }
