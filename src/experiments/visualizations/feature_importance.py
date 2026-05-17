"""
Feature importance LightGBM avec agrégation par groupe.

Statistiquement, les feature importances LightGBM (gain et split) sont
**additives par construction** :

- `split` = compte des nœuds utilisant la feature (sommable trivialement)
- `gain` = somme des baisses de loss aux splits qui utilisent la feature

Donc sommer par groupe (text / image / tabulaire pour M2) est valide. Mais :

1. **Sommer brut favorise les gros groupes** (M2 a 27 + 27 features dérivées des
   modalités vs ~10 tabulaires). On rapporte donc à la fois `gain_sum` (vue
   "totale") et `gain_mean` (vue "à feature égale, qui pèse le plus").

2. **Le gain est mesuré sur le train**, donc reflète l'apprentissage, pas la
   généralisation. Une feature à gain élevé peut overfitter — à confirmer par
   permutation importance sur le gold pour les analyses approfondies.

3. **Features corrélées partagent leur importance** entre elles. L'agrégation
   par groupe récupère le signal au niveau du groupe, mais l'importance
   individuelle d'une feature peut être sous-estimée.

API :

- `aggregate_importance_by_group` : retourne un DataFrame agrégé
- `feature_importance_plot` : figure 2 panneaux (par groupe + top individuelles)
- `default_group_of_m2` : mapping par défaut pour le naming M2
- `log_feature_importance_to_mlflow` : helper de logging complet
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Couleurs par groupe (cohérence visuelle inter-plots)
GROUP_COLORS = {
    "text": "#1f77b4",       # bleu
    "image": "#ff7f0e",      # orange
    "tabular": "#7f7f7f",    # gris
    # Couleurs de réserve pour groupes ajoutés (M3+, autres modèles)
    "other": "#2ca02c",
    "meta": "#d62728",
}


def default_group_of_m2(feature_name: str) -> str:
    """
    Mapping par défaut pour M2 stacking. Adapter selon la convention de
    nommage utilisée à la création des features.

    Conventions reconnues :
    - `p_text_*`, `text_p_*`, `text_proba_*` → 'text'
    - `p_image_*`, `image_p_*`, `image_proba_*` → 'image'
    - tout le reste → 'tabular'
    """
    name = feature_name.lower()
    if (
        name.startswith("p_text_")
        or name.startswith("text_p_")
        or "text_proba" in name
        or name.startswith("proba_text_")
    ):
        return "text"
    if (
        name.startswith("p_image_")
        or name.startswith("image_p_")
        or "image_proba" in name
        or name.startswith("proba_image_")
    ):
        return "image"
    return "tabular"


def _get_booster(model):
    """Extrait le Booster LightGBM depuis un wrapper sklearn ou Booster direct."""
    if hasattr(model, "booster_"):
        return model.booster_
    if hasattr(model, "_Booster"):
        return model._Booster
    return model  # assume c'est déjà un Booster


def aggregate_importance_by_group(
    model,
    feature_names: list[str] | None = None,
    group_of: Callable[[str], str] = default_group_of_m2,
) -> pd.DataFrame:
    """
    Calcule l'importance LightGBM (gain et split) par feature et l'agrège par groupe.

    Parameters
    ----------
    model : Booster LightGBM ou wrapper sklearn (LGBMClassifier, ...)
    feature_names : optionnel, sinon lu depuis le booster
    group_of : fonction str -> str qui mappe un nom de feature à son groupe

    Returns
    -------
    pd.DataFrame avec deux structures :
        - `per_feature` (attribut) : DataFrame complet feature par feature
        - retour direct : DataFrame agrégé par groupe avec colonnes
          `gain_sum`, `gain_mean`, `gain_share`, `split_sum`, `split_mean`,
          `n_features`
    """
    booster = _get_booster(model)
    if feature_names is None:
        feature_names = booster.feature_name()

    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")

    per_feature = pd.DataFrame({
        "feature": feature_names,
        "gain": gain,
        "split": split,
        "group": [group_of(n) for n in feature_names],
    })

    total_gain = per_feature["gain"].sum()
    total_split = per_feature["split"].sum()

    agg = per_feature.groupby("group").agg(
        gain_sum=("gain", "sum"),
        gain_mean=("gain", "mean"),
        split_sum=("split", "sum"),
        split_mean=("split", "mean"),
        n_features=("feature", "count"),
    ).reset_index()

    agg["gain_share"] = agg["gain_sum"] / max(total_gain, 1e-12)
    agg["split_share"] = agg["split_sum"] / max(total_split, 1e-12)

    # Attacher per_feature comme attribut pour récupération facile
    agg.attrs["per_feature"] = per_feature
    return agg


def feature_importance_plot(
    model,
    feature_names: list[str] | None = None,
    group_of: Callable[[str], str] = default_group_of_m2,
    top_n_individual: int = 15,
    title: str = "Feature importance (LightGBM gain)",
) -> "Figure":
    """
    Plot 2 panneaux : importance par groupe + top features individuelles.

    Panneau gauche
    --------------
    Pour chaque groupe (text / image / tabulaire), deux barres côte à côte :
    - `gain_sum` (axe gauche) : importance totale du groupe
    - `gain_mean` (axe droit) : importance moyenne par feature dans le groupe

    Lecture : si `gain_sum` est élevé pour `text` mais `gain_mean` est faible,
    le signal est diffus à travers les 27 features texte. Si `gain_mean` est
    élevé pour `tabular`, chaque feature tabulaire individuelle est très
    informative malgré leur faible nombre.

    Panneau droit
    -------------
    Top N features individuelles, colorées par groupe. Aide à identifier les
    features qui dominent l'apprentissage du méta-learner.

    Parameters
    ----------
    model : Booster ou wrapper sklearn LightGBM
    feature_names : optionnel
    group_of : mapping str -> str
    top_n_individual : N features à afficher dans le panneau droit
    title : titre global

    Returns
    -------
    matplotlib.figure.Figure
    """
    agg = aggregate_importance_by_group(model, feature_names, group_of)
    per_feature = agg.attrs["per_feature"]

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [1, 1.3]},
    )

    # === Panneau gauche : par groupe ===
    groups = agg["group"].tolist()
    x = np.arange(len(groups))
    width = 0.35

    # Pour comparer gain_sum et gain_mean sur des échelles différentes,
    # on utilise deux axes Y.
    gain_sum = agg["gain_sum"].values
    gain_mean = agg["gain_mean"].values
    gain_share = agg["gain_share"].values

    bars1 = ax_left.bar(
        x - width / 2, gain_sum,
        width=width, color="#4a4a4a", alpha=0.8, edgecolor="black",
        label="gain_sum",
    )

    ax_left2 = ax_left.twinx()
    bars2 = ax_left2.bar(
        x + width / 2, gain_mean,
        width=width, color="#a0a0a0", alpha=0.8, edgecolor="black",
        label="gain_mean",
    )

    ax_left.set_xticks(x)
    ax_left.set_xticklabels([
        f"{g}\n(n={agg.loc[agg['group'] == g, 'n_features'].iat[0]})"
        for g in groups
    ])
    ax_left.set_ylabel("gain_sum (importance totale du groupe)", color="#4a4a4a")
    ax_left2.set_ylabel("gain_mean (par feature dans le groupe)", color="#a0a0a0")

    # Annotation des shares au-dessus de chaque barre gain_sum
    for bar, share in zip(bars1, gain_share):
        ax_left.annotate(
            f"{share:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=10, color="#4a4a4a", fontweight="bold",
        )

    ax_left.set_title("Importance par groupe")
    ax_left.grid(alpha=0.3, axis="y")

    # Légendes combinées
    lines1, labels1 = ax_left.get_legend_handles_labels()
    lines2, labels2 = ax_left2.get_legend_handles_labels()
    ax_left.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # === Panneau droit : top individuelles ===
    top = per_feature.nlargest(top_n_individual, "gain").iloc[::-1]  # tri inversé pour barh
    colors = [GROUP_COLORS.get(g, "#666666") for g in top["group"]]

    ax_right.barh(
        range(len(top)),
        top["gain"].values,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )
    ax_right.set_yticks(range(len(top)))
    ax_right.set_yticklabels(top["feature"].values, fontsize=9)
    ax_right.set_xlabel("gain")
    ax_right.set_title(f"Top {top_n_individual} features individuelles")
    ax_right.grid(alpha=0.3, axis="x")

    # Légende des couleurs par groupe (pour le panneau droit)
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=GROUP_COLORS.get(g, "#666666"), label=g, edgecolor="black")
        for g in groups
    ]
    ax_right.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

    fig.suptitle(title, fontsize=13, y=1.0)
    plt.tight_layout()
    return fig


def log_feature_importance_to_mlflow(
    model,
    feature_names: list[str] | None = None,
    group_of: Callable[[str], str] = default_group_of_m2,
    prefix: str = "feature_importance",
    top_n_individual: int = 15,
    log_top_metrics: bool = True,
    close_figures: bool = True,
) -> dict:
    """
    Logue dans MLflow : plot + métriques par groupe + top features (en metric).

    Parameters
    ----------
    model : Booster ou wrapper sklearn LightGBM
    feature_names : optionnel
    group_of : mapping str -> str
    prefix : préfixe utilisé pour les noms de metric et le chemin du plot
    top_n_individual : nombre de top features à afficher dans le plot
    log_top_metrics : si True, logue aussi les top features comme metrics
        (pratique pour comparer entre runs MLflow)
    close_figures : ferme la figure après log

    Returns
    -------
    dict : informations sur ce qui a été loggué
    """
    import mlflow

    agg = aggregate_importance_by_group(model, feature_names, group_of)
    per_feature = agg.attrs["per_feature"]

    # 1) Métriques agrégées par groupe (faciles à comparer entre runs)
    for _, row in agg.iterrows():
        g = row["group"]
        mlflow.log_metric(f"{prefix}/{g}/gain_sum", float(row["gain_sum"]))
        mlflow.log_metric(f"{prefix}/{g}/gain_mean", float(row["gain_mean"]))
        mlflow.log_metric(f"{prefix}/{g}/gain_share", float(row["gain_share"]))
        mlflow.log_metric(f"{prefix}/{g}/split_sum", float(row["split_sum"]))
        mlflow.log_metric(f"{prefix}/{g}/n_features", int(row["n_features"]))

    # 2) Top features individuelles en metrics (optionnel)
    if log_top_metrics:
        top = per_feature.nlargest(top_n_individual, "gain")
        for _, row in top.iterrows():
            # MLflow restreint les caractères dans les noms de metrics : alphanum, _, /, -, .
            safe_name = "".join(ch if (ch.isalnum() or ch in "_-./") else "_" for ch in row["feature"])
            mlflow.log_metric(f"{prefix}/top/{safe_name}", float(row["gain"]))

    # 3) Plot
    fig = feature_importance_plot(
        model, feature_names, group_of,
        top_n_individual=top_n_individual,
    )
    plot_path = f"{prefix}/feature_importance.png"
    mlflow.log_figure(fig, plot_path)
    if close_figures:
        plt.close(fig)

    # 4) Table complète comme CSV (pour analyse offline)
    csv_path = f"{prefix}/per_feature.csv"
    per_feature_sorted = per_feature.sort_values("gain", ascending=False).reset_index(drop=True)
    mlflow.log_text(per_feature_sorted.to_csv(index=False), csv_path)

    agg_csv_path = f"{prefix}/aggregated_by_group.csv"
    mlflow.log_text(agg.to_csv(index=False), agg_csv_path)

    return {
        "aggregated": agg,
        "per_feature": per_feature_sorted,
        "plot_path": plot_path,
        "csv_paths": [csv_path, agg_csv_path],
    }
