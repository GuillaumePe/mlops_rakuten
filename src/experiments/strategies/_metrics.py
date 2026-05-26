"""
Métriques étendues pour analyse fine d'un modèle de stacking multimodal.

Histoire 1 : calibration (ECE, log_loss, Brier)
Histoire 2 : apport fusion (agreement, oracle, gain net du meta)
Histoire 3 : par classe (precision, recall, support, top confusions)
Histoire 4 : stabilité Optuna
Histoire 5 : inférence (latence p50/p95, throughput, n_params)
"""
from __future__ import annotations
import time
from collections import Counter

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score,
)


# === Histoire 1 : calibration =================================================

def expected_calibration_error(
    y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10
) -> float:
    """
    Expected Calibration Error multi-classe.
    
    Définition : on bin les samples par confiance (max proba prédite),
    pour chaque bin on calcule |accuracy_bin - confidence_bin|, on moyenne 
    pondéré par taille de bin.
    
    ECE bas = bien calibré : la confiance du modèle reflète sa précision.
    """
    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        acc_bin = accuracies[in_bin].mean()
        conf_bin = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_diagram_data(
    y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Données pour le reliability diagram (à plotter côté Streamlit).
    Retourne un dict avec les centres de bins, accuracy par bin, 
    confidence par bin, et taille de chaque bin.
    """
    confidences = probas.max(axis=1)
    predictions = probas.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, accs, confs, sizes = [], [], [], []
    for i in range(n_bins):
        in_bin = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        if in_bin.sum() == 0:
            accs.append(0.0)
            confs.append(0.0)
        else:
            accs.append(float(accuracies[in_bin].mean()))
            confs.append(float(confidences[in_bin].mean()))
        sizes.append(int(in_bin.sum()))
    return {
        "bin_centers": bin_centers,
        "accuracy_per_bin": accs,
        "confidence_per_bin": confs,
        "size_per_bin": sizes,
    }


def brier_score_multiclass(y_true: np.ndarray, probas: np.ndarray) -> float:
    """
    Brier score multi-classe : moyenne sur les samples de la somme des écarts 
    quadratiques entre proba prédite et one-hot de la vraie classe.
    """
    n_classes = probas.shape[1]
    one_hot = np.eye(n_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((probas - one_hot) ** 2, axis=1)))


# === Histoire 2 : apport fusion ===============================================

def stacking_analysis(
    y_true: np.ndarray,
    p_text: np.ndarray,
    p_image: np.ndarray,
    preds_meta: np.ndarray,
) -> dict:
    """
    Quantifie l'apport de la fusion par rapport aux base learners.
    
    Retourne un dict (à logger en JSON artifact + métriques scalaires).
    """
    preds_text = p_text.argmax(axis=1)
    preds_image = p_image.argmax(axis=1)

    text_correct = preds_text == y_true
    image_correct = preds_image == y_true
    meta_correct = preds_meta == y_true

    # Agreement entre les deux modalités
    agreement = float((preds_text == preds_image).mean())

    # Cas de complémentarité
    text_only_correct = int(((text_correct) & (~image_correct)).sum())
    image_only_correct = int((~text_correct & image_correct).sum())
    both_correct = int((text_correct & image_correct).sum())
    both_wrong = int((~text_correct & ~image_correct).sum())

    # Gain net de la fusion :
    # samples où meta a raison mais aucune base seule n'avait raison.
    meta_saves = int((meta_correct & ~text_correct & ~image_correct).sum())

    # Cas inverse : meta a tort alors qu'au moins une base avait raison.
    # Si élevé, le meta dégrade les bons signaux.
    meta_loses = int((~meta_correct & (text_correct | image_correct)).sum())

    # Oracle : F1 si on pouvait toujours choisir la meilleure modalité.
    # Argmax de p_text si correct, sinon p_image, sinon une des deux par défaut.
    oracle_preds = np.where(text_correct, preds_text,
                  np.where(image_correct, preds_image, preds_text))
    oracle_f1_macro = float(f1_score(y_true, oracle_preds, average="macro"))
    oracle_f1_weighted = float(f1_score(y_true, oracle_preds, average="weighted"))

    return {
        "agreement_text_image": agreement,
        "text_only_correct": text_only_correct,
        "image_only_correct": image_only_correct,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "meta_saves_when_both_wrong": meta_saves,
        "meta_loses_when_one_correct": meta_loses,
        "oracle_f1_macro": oracle_f1_macro,
        "oracle_f1_weighted": oracle_f1_weighted,
    }


# === Histoire 3 : par classe ==================================================

def per_class_metrics(y_true: np.ndarray, preds: np.ndarray, n_classes: int) -> dict:
    """Precision, recall, F1, support par classe."""
    p = precision_score(y_true, preds, labels=range(n_classes), average=None, zero_division=0)
    r = recall_score(y_true, preds, labels=range(n_classes), average=None, zero_division=0)
    f1 = f1_score(y_true, preds, labels=range(n_classes), average=None, zero_division=0)
    support = np.bincount(y_true, minlength=n_classes)
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
    }


def top_confusions(
    y_true: np.ndarray, preds: np.ndarray, n_classes: int, top_k: int = 10
) -> list[dict]:
    """
    Top paires (vraie classe, prédite) les plus confondues.
    Hors diagonale.
    """
    cm = confusion_matrix(y_true, preds, labels=range(n_classes))
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": int(i), "predicted": int(j), "count": int(cm[i, j])})
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_k]


# === Histoire 5 : inférence ===================================================

def measure_inference_latency(
    model, X: pl.DataFrame, n_warmup: int = 5, n_repeat: int = 50
) -> dict:
    """
    Mesure la latence per-sample du predict du modèle.
    
    Args:
        model: doit avoir une méthode .predict(X)
        X: DataFrame Polars de samples (idéalement le test set)
        n_warmup: nb d'inférences à jeter (cache CPU, JIT, etc.)
        n_repeat: nb d'inférences à mesurer

    Returns:
        Dict avec p50, p95, p99 en ms par sample, throughput en samples/sec.
    """
    if len(X) == 0:
        return {}

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(X.head(1))

    # Mesure : on prédit 1 sample à la fois (pire cas pour latence)
    latencies_ms = []
    for _ in range(n_repeat):
        idx = np.random.randint(0, len(X))
        sample = X[idx:idx + 1]
        start = time.perf_counter()
        _ = model.predict(sample)
        elapsed = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed)

    latencies = np.array(latencies_ms)
    return {
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "latency_ms_p99": float(np.percentile(latencies, 99)),
        "latency_ms_mean": float(latencies.mean()),
        "throughput_samples_per_sec": float(1000.0 / latencies.mean()),
    }


def count_model_params(model) -> int:
    """
    Compte le nombre total de paramètres du M2Stacking.
    LogReg : coef + intercept. LGBM : nombre de feuilles d'arbres × 2 (split + leaf).
    Approximation, mais utile pour comparer M2 / M3 / M4.
    """
    n = 0
    # LogReg text
    if hasattr(model, "f_text_") and model.f_text_ is not None:
        logreg = model.f_text_.named_steps["logreg"]
        n += logreg.coef_.size + logreg.intercept_.size
    # LogReg image
    if hasattr(model, "f_image_") and model.f_image_ is not None:
        logreg = model.f_image_.named_steps["logreg"]
        n += logreg.coef_.size + logreg.intercept_.size
    # LGBM meta : approximation = somme des arbres (nombre de feuilles × 2)
    if hasattr(model, "meta_") and model.meta_ is not None:
        booster = model.meta_.booster_
        n += sum(booster.num_model_per_iteration() for _ in range(booster.num_trees()))
    return n