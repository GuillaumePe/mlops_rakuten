"""
Plots de calibration et de discrimination des prédictions multiclasses.

Trois plots complémentaires :

- `reliability_diagram` : confidence prédite vs accuracy observée, avec histogramme
  de support en bas. Diagonale = calibration parfaite. ECE annoté.
- `confidence_histograms` : distribution de max(p) split par correct/incorrect.
  Analogue multiclasse du plot Dataiku "distribution proba vrais 0 / vrais 1".
- `margin_histograms` : distribution de p[top-1] - p[top-2] split par correct/incorrect.
  Mesure la marge de décision et complète l'histogramme de confidence.

Helper :

- `log_calibration_plots_to_mlflow` : logge les trois plots pour un ensemble donné.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Couleurs cohérentes à travers le module
COLOR_CORRECT = "#2ca02c"      # vert
COLOR_INCORRECT = "#d62728"    # rouge
COLOR_CALIBRATION_BAR = "#1f77b4"  # bleu
COLOR_DIAGONAL = "#444444"


def ece_score(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (Naeini et al. 2015).

    ECE = sum_b (|b| / N) * |acc(b) - conf(b)|

    où b parcourt les bins de confidence (largeur = 1/n_bins).

    Parameters
    ----------
    y_true : array (n,) labels entiers
    p_pred : array (n, n_classes) probas softmax
    n_bins : nombre de bins de confidence

    Returns
    -------
    ECE comme float dans [0, 1]. 0 = calibration parfaite.
    """
    confidences = p_pred.max(axis=1)
    predictions = p_pred.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (
            confidences < bin_edges[i + 1] if i < n_bins - 1 else confidences <= bin_edges[i + 1]
        )
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> "Figure":
    """
    Reliability diagram avec histogramme de support sous le plot principal.

    Justification statistique
    -------------------------
    Un classifieur est calibré si P(y = ŷ | conf = c) ≈ c pour tout c. Le
    reliability diagram trace cette quantité empiriquement, par bin de confidence.

    Si les barres sont SOUS la diagonale → modèle sur-confiant (cas typique en DL).
    Si AU-DESSUS → modèle sous-confiant (rare, signal d'over-régularisation).

    L'histogramme de support en bas répond à "où vit le modèle ?". En multiclasse
    27 classes, la confidence ne descend quasi jamais sous 1/27 ≈ 0.04 — le modèle
    n'est jamais "totalement perdu". La masse se concentre en haut.

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    n_bins : nombre de bins, 10 par défaut (Guo et al. 2017)
    title : titre du plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    confidences = p_pred.max(axis=1)
    predictions = p_pred.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        else:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs[i] = accuracies[mask].mean()
            bin_confs[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    ece = ece_score(y_true, p_pred, n_bins=n_bins)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(7, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.15},
        layout="constrained",
    )

    # --- Top: reliability diagram
    width = 1.0 / n_bins
    # Barres d'accuracy observée
    ax_top.bar(
        bin_centers, bin_accs,
        width=width * 0.95,
        edgecolor="black",
        color=COLOR_CALIBRATION_BAR,
        alpha=0.7,
        label="Accuracy observée",
    )
    # Barres d'écart (gap entre confidence et accuracy)
    gap_heights = bin_confs - bin_accs
    ax_top.bar(
        bin_centers, gap_heights,
        bottom=bin_accs,
        width=width * 0.95,
        edgecolor="black",
        color="red",
        alpha=0.3,
        hatch="//",
        label="Gap (over-confidence)",
    )
    # Diagonale
    ax_top.plot([0, 1], [0, 1], "--", color=COLOR_DIAGONAL, label="Calibration parfaite")

    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)
    ax_top.set_ylabel("Accuracy observée")
    ax_top.set_title(f"{title}\nECE = {ece:.4f}, N = {len(y_true)}")
    ax_top.legend(loc="upper left", framealpha=0.9)
    ax_top.grid(alpha=0.3)

    # --- Bottom: histogramme du support
    ax_bot.bar(
        bin_centers, bin_counts,
        width=width * 0.95,
        edgecolor="black",
        color="#888888",
        alpha=0.7,
    )
    ax_bot.set_xlim(0, 1)
    ax_bot.set_xlabel("Confidence prédite (max p)")
    ax_bot.set_ylabel("# samples")
    ax_bot.grid(alpha=0.3, axis="y")

    return fig


def confidence_histograms(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 30,
    title: str = "Confidence distribution",
) -> "Figure":
    """
    Distribution de max(p) split par correct vs incorrect.

    Justification statistique
    -------------------------
    Adaptation multiclasse du plot binaire Dataiku qui compare P(score | y=0)
    et P(score | y=1). En multiclasse, on regarde la distribution conditionnelle :

        f_correct(c) = densité de max(p) sachant ŷ = y_true
        f_incorrect(c) = densité de max(p) sachant ŷ ≠ y_true

    Bonne séparation entre les deux → le modèle "sait quand il sait", c'est-à-dire
    sa confidence est informative pour distinguer les bonnes et mauvaises décisions.
    Mauvaise séparation → la confidence est peu corrélée à la justesse (sur-confiance
    systématique).

    Le recouvrement des deux distributions dans la zone [0.7, 0.95] est le symptôme
    visuel direct d'un ECE élevé.

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    n_bins : nombre de bins de l'histogramme
    title : titre du plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    confidences = p_pred.max(axis=1)
    predictions = p_pred.argmax(axis=1)
    is_correct = predictions == y_true

    n_correct = int(is_correct.sum())
    n_incorrect = int((~is_correct).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, n_bins + 1)

    if n_correct > 0:
        ax.hist(
            confidences[is_correct],
            bins=bins,
            alpha=0.6,
            label=f"Corrects (n={n_correct})",
            density=True,
            color=COLOR_CORRECT,
            edgecolor="black",
        )
    if n_incorrect > 0:
        ax.hist(
            confidences[~is_correct],
            bins=bins,
            alpha=0.6,
            label=f"Incorrects (n={n_incorrect})",
            density=True,
            color=COLOR_INCORRECT,
            edgecolor="black",
        )

    acc = n_correct / max(len(y_true), 1)
    ax.set_xlabel("Confidence max(p)")
    ax.set_ylabel("Densité")
    ax.set_title(f"{title}\nAccuracy = {acc:.4f}, N = {len(y_true)}")
    ax.set_xlim(0, 1)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def margin_histograms(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 30,
    title: str = "Margin distribution (top-1 minus top-2)",
) -> "Figure":
    """
    Distribution de margin = p[top-1] - p[top-2] split par correct vs incorrect.

    Justification statistique
    -------------------------
    La margin mesure l'écart à la frontière de décision en multiclasse. Deux samples
    peuvent avoir une confidence identique (max p = 0.8) mais des margins très
    différentes :

        - Sample A : p = [0.8, 0.15, 0.05, ...] → margin = 0.65 (modèle sûr)
        - Sample B : p = [0.8, 0.18, 0.02, ...] → margin = 0.62 (modèle sûr aussi)
        - Sample C : p = [0.42, 0.40, 0.18, ...] → margin = 0.02 (modèle hésite !)

    Pour les **erreurs**, on attend une margin faible (le modèle hésitait, sa
    décision est marginale). Si les erreurs ont des margins élevées, cela signifie
    que le modèle se trompe avec assurance — signal de **biais structurel** : il
    confond systématiquement deux classes, pas par hasard.

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    n_bins : nombre de bins
    title : titre du plot

    Returns
    -------
    matplotlib.figure.Figure
    """
    sorted_p = np.sort(p_pred, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]
    predictions = p_pred.argmax(axis=1)
    is_correct = predictions == y_true

    n_correct = int(is_correct.sum())
    n_incorrect = int((~is_correct).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, n_bins + 1)

    if n_correct > 0:
        ax.hist(
            margins[is_correct],
            bins=bins,
            alpha=0.6,
            label=f"Corrects (n={n_correct}, mean={margins[is_correct].mean():.3f})",
            density=True,
            color=COLOR_CORRECT,
            edgecolor="black",
        )
    if n_incorrect > 0:
        ax.hist(
            margins[~is_correct],
            bins=bins,
            alpha=0.6,
            label=f"Incorrects (n={n_incorrect}, mean={margins[~is_correct].mean():.3f})",
            density=True,
            color=COLOR_INCORRECT,
            edgecolor="black",
        )

    ax.set_xlabel("Margin = p[top-1] − p[top-2]")
    ax.set_ylabel("Densité")
    ax.set_title(f"{title}\nN = {len(y_true)}")
    ax.set_xlim(0, 1)
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def log_calibration_plots_to_mlflow(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    prefix: str = "calibration/eval/",
    n_bins_reliability: int = 10,
    n_bins_histograms: int = 30,
    close_figures: bool = True,
) -> dict:
    """
    Helper de logging : génère et logge les trois plots de calibration pour un ensemble.

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    prefix : préfixe du chemin dans MLflow artifacts. Ex: "calibration/gold/"
    n_bins_reliability : bins du reliability diagram
    n_bins_histograms : bins des histogrammes confidence/margin
    close_figures : ferme les figures matplotlib après log (économise la RAM)

    Returns
    -------
    dict : {nom_plot: chemin MLflow} pour traçabilité

    Notes
    -----
    Nécessite un run MLflow actif (`mlflow.start_run` déjà appelé).
    """
    import mlflow

    suffix = prefix.rstrip("/").split("/")[-1] if "/" in prefix else "eval"

    logged = {}

    fig = reliability_diagram(
        y_true, p_pred, n_bins=n_bins_reliability,
        title=f"Reliability — {suffix}",
    )
    path = f"{prefix.rstrip('/')}/reliability.png"
    mlflow.log_figure(fig, path)
    logged["reliability"] = path
    if close_figures:
        plt.close(fig)

    fig = confidence_histograms(
        y_true, p_pred, n_bins=n_bins_histograms,
        title=f"Confidence — {suffix}",
    )
    path = f"{prefix.rstrip('/')}/confidence_histogram.png"
    mlflow.log_figure(fig, path)
    logged["confidence"] = path
    if close_figures:
        plt.close(fig)

    fig = margin_histograms(
        y_true, p_pred, n_bins=n_bins_histograms,
        title=f"Margin — {suffix}",
    )
    path = f"{prefix.rstrip('/')}/margin_histogram.png"
    mlflow.log_figure(fig, path)
    logged["margin"] = path
    if close_figures:
        plt.close(fig)

    return logged
