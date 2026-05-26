"""
M.4bis — Wrapper PyFunc générique pour tous les BaseLearner deep.

Utilise import dynamique (`importlib`) pour reconstruire la classe à partir
de son module path complet, puis appelle `BaseLearner.from_pretrained()`
(interface uniforme imposée par l'ABC dans `_base.py`).

Conçu pour s'intégrer avec mlflow.pyfunc.log_model :

    learner_class_path = (
        f"{learner.__class__.__module__}.{learner.__class__.__name__}"
    )
    with tempfile.TemporaryDirectory() as tmp:
        learner_dir = Path(tmp) / "learner"
        learner.save_pretrained(learner_dir)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BaseLearnerPyfunc(learner_class_path),
            artifacts={"learner_dir": str(learner_dir)},
            registered_model_name=f"rakuten-base-{learner.name}",
        )

Et l'usage downstream (M2Benchmark, dashboards, drift detection) :

    import mlflow.pyfunc

    model = mlflow.pyfunc.load_model("models:/rakuten-base-textcnn@active")

    # Probas (n, 27) — default
    probas = model.predict(X_df)

    # Embeddings (n, embed_dim)
    embeddings = model.predict(X_df, params={"return_embeddings": True})

Justification du design :
- Un seul wrapper pour tous les BaseLearner deep (TextCNN, ResNet50PartialFT,
  CamembertLoRA, ResNet18FullFT, ...). L'ABC garantit que tous exposent
  `from_pretrained`, `predict_proba`, `extract_embeddings`.
- Pas de spécialisation par learner → ajouter un nouveau base learner ne
  nécessite AUCUNE modification de ce wrapper.
- L'import dynamique évite d'avoir à pickler la classe entière (qui peut
  contenir des references circulaires ou des nn.Module non-picklable directement).
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import mlflow.pyfunc


logger = logging.getLogger(__name__)


class BaseLearnerPyfunc(mlflow.pyfunc.PythonModel):
    """
    Wrapper MLflow PyFunc pour exposer un BaseLearner deep en mode inference.

    Stocke le chemin complet de la classe (ex: 'src.models.base_learners.text.textcnn.TextCNN')
    pour pouvoir importer dynamiquement et reconstruire au `load_context`.

    Attributes:
        learner_class_path: chemin Python complet de la classe BaseLearner
            sous la forme 'module.submodule.ClassName'.

    Le learner est reconstruit au moment du `load_context` (premier appel après
    `mlflow.pyfunc.load_model`) via `BaseLearner.from_pretrained(artifacts_dir)`.
    """

    def __init__(self, learner_class_path: str):
        """
        Args:
            learner_class_path: chemin Python complet de la classe BaseLearner.
                Exemple : "src.models.base_learners.text.textcnn.TextCNN".
                Utiliser :
                    f"{learner.__class__.__module__}.{learner.__class__.__name__}"
                pour récupérer ce chemin sans le hardcoder.
        """
        if not isinstance(learner_class_path, str) or "." not in learner_class_path:
            raise ValueError(
                f"learner_class_path doit être 'module.submodule.ClassName', "
                f"reçu : {learner_class_path!r}"
            )
        self.learner_class_path = learner_class_path
        self.learner = None  # rempli par load_context

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """
        Reconstruit le BaseLearner depuis les artefacts.

        Appelé automatiquement par MLflow lors du premier `predict`.

        Args:
            context: contient `context.artifacts["learner_dir"]` qui pointe
                vers le dossier écrit par `BaseLearner.save_pretrained`.

        Raises:
            ImportError: si learner_class_path ne correspond à aucune classe.
            AttributeError: si la classe n'a pas `from_pretrained`.
            FileNotFoundError: si les artefacts attendus ne sont pas présents.
        """
        if "learner_dir" not in context.artifacts:
            raise KeyError(
                "Artefact 'learner_dir' manquant. "
                "Vérifier que log_model a bien reçu artifacts={'learner_dir': ...}."
            )

        # Import dynamique de la classe BaseLearner
        module_path, class_name = self.learner_class_path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Impossible d'importer le module {module_path!r} "
                f"référencé par learner_class_path={self.learner_class_path!r}. "
                f"Le fichier est-il bien dans le pythonpath ? Cause : {e}"
            ) from e

        try:
            learner_class = getattr(module, class_name)
        except AttributeError as e:
            raise AttributeError(
                f"Module {module_path!r} ne contient pas la classe {class_name!r}. "
                f"Vérifier learner_class_path={self.learner_class_path!r}."
            ) from e

        if not hasattr(learner_class, "from_pretrained"):
            raise AttributeError(
                f"{class_name} doit implémenter from_pretrained (classmethod). "
                f"Voir BaseLearner.from_pretrained pour le contrat."
            )

        # Reconstruction du learner
        logger.info(
            f"[BaseLearnerPyfunc] Reconstruction de {class_name} depuis "
            f"{context.artifacts['learner_dir']}"
        )
        self.learner = learner_class.from_pretrained(context.artifacts["learner_dir"])
        logger.info(f"[BaseLearnerPyfunc] {class_name} prêt (mode eval)")

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: dict | None = None,
    ) -> Any:
        """
        Forward pass en mode eval.

        Args:
            context: passé par MLflow, non utilisé directement (artifacts déjà
                chargés au load_context).
            model_input: pd.DataFrame ou pl.DataFrame avec les colonnes attendues
                par le BaseLearner concret (text, imageid/productid selon modalité).
                MLflow renvoie typiquement pd.DataFrame ; conversion automatique
                vers pl.DataFrame avant délégation au learner.
            params: dict optionnel passé via mlflow >=2.8.
                Clés reconnues :
                    - "return_embeddings" (bool, default False) :
                        si True → extract_embeddings (n, embed_dim)
                        si False → predict_proba (n, n_classes)

        Returns:
            np.ndarray :
                - (n, n_classes) si return_embeddings=False (default)
                - (n, embed_dim) si return_embeddings=True

        Raises:
            RuntimeError: si load_context n'a pas été appelé (cas anormal).
        """
        if self.learner is None:
            raise RuntimeError(
                "BaseLearnerPyfunc.learner non initialisé. "
                "load_context() doit être appelé avant predict()."
            )

        # Conversion défensive pandas → polars (MLflow renvoie souvent pandas)
        import polars as pl
        if not isinstance(model_input, pl.DataFrame):
            try:
                import pandas as pd
                if isinstance(model_input, pd.DataFrame):
                    model_input = pl.from_pandas(model_input)
                else:
                    raise TypeError(
                        f"model_input doit être pd.DataFrame ou pl.DataFrame, "
                        f"reçu : {type(model_input)}"
                    )
            except ImportError:
                raise TypeError(
                    f"model_input doit être pl.DataFrame (pandas indisponible). "
                    f"Reçu : {type(model_input)}"
                )

        # Dispatch selon le mode demandé
        return_embeddings = bool(params and params.get("return_embeddings", False))
        if return_embeddings:
            return self.learner.extract_embeddings(model_input)
        return self.learner.predict_proba(model_input)
