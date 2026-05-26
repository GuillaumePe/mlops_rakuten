"""
Analyse par classe pour le diagnostic en multiclasse.

- `one_vs_rest_plot` : adaptation directe du plot binaire Dataiku à chaque classe.
  Pour une classe c, on compare la distribution de P(y=c|x) pour les samples
  réellement de classe c vs les autres. Bonne séparation → la classe est bien
  reconnue.
- `select_hard_classes` : helper qui calcule F1 par classe et retourne les K
  plus problématiques. Utile pour cibler le one-vs-rest sur les classes
  intéressantes sans surcharger l'output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure


COLOR_POSITIVE = "#1f77b4"   # bleu : "vrais y = c"
COLOR_NEGATIVE = "#ff7f0e"   # orange : "autres classes"


def select_hard_classes(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    top_k: int = 6,
    min_support: int = 10,
) -> list[int]:
    """
    Retourne les top_k classes les plus difficiles selon F1 par classe.

    Justification
    -------------
    Plotter 27 classes en one-vs-rest est illisible. On cible les classes
    problématiques (F1 faible) pour le diagnostic. Le seuil `min_support`
    évite de retourner des classes avec trop peu de samples (F1 instable
    statistiquement : σ_F1 ≈ √(p(1-p)/n) ≈ 0.1 pour n=20).

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    top_k : nombre de classes à retourner
    min_support : nombre minimum de samples pour qu'une classe soit considérée

    Returns
    -------
    list[int] : indices des classes, ordonnés du F1 le plus bas au plus haut
    """
    from sklearn.metrics import f1_score

    predictions = p_pred.argmax(axis=1)
    n_classes = p_pred.shape[1]

    f1_per_class = f1_score(
        y_true, predictions,
        labels=range(n_classes),
        average=None,
        zero_division=0,
    )

    # Filtrer les classes avec support suffisant
    eligible = []
    for c in range(n_classes):
        support = int((y_true == c).sum())
        if support >= min_support:
            eligible.append((c, f1_per_class[c], support))

    # Tri ascendant sur F1, on garde les top_k pires
    eligible.sort(key=lambda x: x[1])
    return [c for c, _, _ in eligible[:top_k]]


def one_vs_rest_plot(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    classes_to_plot: list[int],
    n_bins: int = 30,
    class_names: dict[int, str] | None = None,
    title: str = "One-vs-Rest probability distribution",
) -> "Figure":
    """
    Plot one-vs-rest par classe : adaptation directe du plot binaire Dataiku.

    Justification statistique
    -------------------------
    Pour une classe c, on trace deux distributions empiriques :

        f_pos(p) = densité de P(y=c|x) pour les samples où y_true = c
        f_neg(p) = densité de P(y=c|x) pour les samples où y_true ≠ c

    Bien séparées → la classe se distingue. Recouvrement important → la classe
    est mal apprise OU confondue avec une voisine (top confusions analysis
    nécessaire pour trancher).

    L'AUC OvR pour la classe c est implicitement visualisé : si on imagine
    déplacer un seuil de décision sur l'axe x, l'AUC mesure la qualité de la
    séparation entre f_pos et f_neg.

    Parameters
    ----------
    y_true : (n,) labels entiers
    p_pred : (n, n_classes) probas softmax
    classes_to_plot : liste des classes (indices) à plotter. Typiquement issu
        de `select_hard_classes(...)`.
    n_bins : bins par histogramme
    class_names : optionnel, mapping {class_id: nom_lisible}
    title : titre de la figure

    Returns
    -------
    matplotlib.figure.Figure (grille 2 colonnes)
    """
    n = len(classes_to_plot)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5,
            "Aucune classe à plotter\n(seuil min_support trop élevé ?)",
            ha="center", va="center",
        )
        ax.axis("off")
        return fig

    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    bins = np.linspace(0, 1, n_bins + 1)

    for i, c in enumerate(classes_to_plot):
        ax = axes[i]
        is_class_c = y_true == c
        n_pos = int(is_class_c.sum())
        n_neg = int((~is_class_c).sum())

        p_c = p_pred[:, c]

        if n_pos > 0:
            ax.hist(
                p_c[is_class_c],
                bins=bins,
                alpha=0.6,
                label=f"y = {c} (n={n_pos})",
                density=True,
                color=COLOR_POSITIVE,
                edgecolor="black",
            )
        if n_neg > 0:
            ax.hist(
                p_c[~is_class_c],
                bins=bins,
                alpha=0.6,
                label=f"y ≠ {c} (n={n_neg})",
                density=True,
                color=COLOR_NEGATIVE,
                edgecolor="black",
            )

        # F1 de la classe pour annotation
        predictions = p_pred.argmax(axis=1)
        tp = int(((predictions == c) & is_class_c).sum())
        fp = int(((predictions == c) & ~is_class_c).sum())
        fn = int(((predictions != c) & is_class_c).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_label = class_names.get(c, str(c)) if class_names else str(c)
        ax.set_title(f"Classe {class_label} — F1 = {f1:.3f}, support = {n_pos}")
        ax.set_xlabel(f"P(y = {c})")
        ax.set_ylabel("Densité")
        ax.set_xlim(0, 1)
        # Log scale en y pour éviter que f_neg écrase f_pos visuellement
        # (les "autres classes" sont ~27x plus nombreuses)
        ax.set_yscale("log")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.grid(alpha=0.3)

    # Masquer les axes en trop si nombre impair de classes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.0)
    plt.tight_layout()
    return fig
