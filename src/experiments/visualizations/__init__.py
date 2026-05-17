"""
Visualizations pour les expériences MLflow du projet Rakuten.

Trois groupes de plots :

1. **Calibration** : reliability diagram, confidence histograms, margin histograms.
   À utiliser sur chaque ensemble eval (gold, shadow batches, val interne) pour
   diagnostiquer la qualité des sorties probabilistes.

2. **Class breakdown** : one-vs-rest par classe difficile, sélection automatique
   des classes les plus problématiques.

3. **Feature importance** : agrégation par groupe (texte / image / tabulaire pour M2,
   adaptable aux autres modèles) + top features individuelles.

Toutes les fonctions retournent des `matplotlib.figure.Figure`, à logger via
`mlflow.log_figure(fig, "path/in/run.png")`.

Helper de logging multi-ensembles :
    log_calibration_plots_to_mlflow(y_true, p_pred, prefix="calibration/gold/")
    log_feature_importance_to_mlflow(model, feature_names, group_of, prefix="feat_imp/")
"""

from .calibration import (
    reliability_diagram,
    confidence_histograms,
    margin_histograms,
    ece_score,
    log_calibration_plots_to_mlflow,
)
from .class_breakdown import (
    one_vs_rest_plot,
    select_hard_classes,
)
from .feature_importance import (
    aggregate_importance_by_group,
    feature_importance_plot,
    default_group_of_m2,
    log_feature_importance_to_mlflow,
)

__all__ = [
    # Calibration
    "reliability_diagram",
    "confidence_histograms",
    "margin_histograms",
    "ece_score",
    "log_calibration_plots_to_mlflow",
    # Class breakdown
    "one_vs_rest_plot",
    "select_hard_classes",
    # Feature importance
    "aggregate_importance_by_group",
    "feature_importance_plot",
    "default_group_of_m2",
    "log_feature_importance_to_mlflow",
]
