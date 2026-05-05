"""
SklearnExperiment : orchestrateur de fit + tracking MLflow pour les modèles
sklearn-style (M2 et variantes).

Sépare strictement :
- le modèle (M2Stacking) : pur, n'importe quoi sait MLflow
- l'orchestration : MLflow run parent, nested runs Optuna, métriques, artefacts
"""
from __future__ import annotations
import io
from pathlib import Path

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
            X_train, y_train = datamodule.get_sklearn_data("train")
            X_val, y_val = datamodule.get_sklearn_data("val")

            # Le callback est désormais inutilisé (logging direct dans M2Stacking)
            # mais on garde l'arg pour compat avec d'autres modèles potentiels.
            self.model = self.model_factory(None)

            # 3. Fit (la HPO Optuna se déroule à l'intérieur, log les nested runs)
            self.model.fit(X_train, y_train)

            # 4. Log params best (du parent)
            mlflow.log_params({f"best/{k}": v for k, v in self.model.best_params_.items()})
            mlflow.log_metric("optuna_best_score", self.model.best_score_)

            # 5. Métriques des base learners en OOF (sur le train)
            self._log_oof_metrics()

            # 6. Métriques sur val (sur le DataModule)
            self._log_val_metrics(X_val, y_val)

            # 7. Artefacts
            self._log_artifacts(X_val, y_val)

            # Histoires 1/2/3/4 : sur OOF (train) et val
            self._log_oof_metrics()
            self._log_val_metrics(X_val, y_val)
            self._log_stacking_analysis(X_val, y_val)
            self._log_per_class_artifacts(X_val, y_val)
            self._log_optuna_stability()
            self._log_calibration(X_val, y_val)

            # Histoire 5 : inférence
            self._log_inference_metrics(X_val)

            # Artefacts visuels
            self._log_artifacts(X_val, y_val)

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

    # --- Histoire 1 : calibration ----------------------------------------

    def _log_calibration(self, X_val: pl.DataFrame, y_val: np.ndarray):
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
        ax.set_title("Reliability diagram")
        ax.legend()
        ax.set_aspect("equal")
        mlflow.log_figure(fig, "reliability_diagram.png")
        plt.close(fig)

    # --- Histoire 2 : apport stacking ------------------------------------

    def _log_stacking_analysis(self, X_val: pl.DataFrame, y_val: np.ndarray):
        # On reproduit les probas par modalité avec le mapping classes_ pour
        # gérer le cas où une LogReg n'aurait pas vu toutes les classes.
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

    # --- Histoire 4 : stabilité Optuna -----------------------------------

    def _log_optuna_stability(self):
        if self.model.cv_scores_:
            scores = np.array(self.model.cv_scores_)
            mlflow.log_metric("stability/cv_score_std", float(scores.std()))
            mlflow.log_metric("stability/cv_score_min", float(scores.min()))
            mlflow.log_metric("stability/cv_score_max", float(scores.max()))
            mlflow.log_metric("stability/cv_score_range", float(scores.max() - scores.min()))

    # --- Histoire 5 : inférence ------------------------------------------

    def _log_inference_metrics(self, X_val: pl.DataFrame):
        latency = measure_inference_latency(self.model, X_val, n_warmup=5, n_repeat=50)
        for k, v in latency.items():
            mlflow.log_metric(f"inference/{k}", v)
        mlflow.log_metric("model/n_params", count_model_params(self.model))

    # --- Artefacts visuels (matrices, importances) -----------------------

    def _log_artifacts(self, X_val: pl.DataFrame, y_val: np.ndarray):
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

    # --- predict / evaluate ----------------------------------------------

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit() d'abord.")
        return self.model.predict(X)

    def evaluate(self, datamodule) -> dict:
        if self.model is None:
            raise RuntimeError("fit() d'abord.")
        X_test, y_test = datamodule.get_sklearn_data("test")
        preds = self.model.predict(X_test)
        return {
            "accuracy": accuracy_score(y_test, preds),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        }