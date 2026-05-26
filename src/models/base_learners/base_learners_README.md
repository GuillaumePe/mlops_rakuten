# Base Learners — Architecture Phase 1

Ce dossier contient les implémentations concrètes du contrat `BaseLearner` défini dans `_base.py`. Chaque base learner est un encodeur (texte, image, tabulaire) qui :

1. S'entraîne via `fit(X_train, y_train, X_val=None, y_val=None)` (signature unifiée M.0c)
2. Extrait des embeddings via `extract_embeddings(X) → (n, embed_dim)`
3. (Optionnel) Expose `predict_proba(X) → (n, n_classes)` pour les analyses de complémentarité

Les BaseLearners sont consommés en aval par les modèles **assemblés** (`src/models/assembled/`) via `StackingLGBM` (K-Fold OOF + meta-features).

## Convention `modality`

Chaque BaseLearner expose une propriété `modality ∈ {"text", "image", "tabular"}`. Cette propriété sert à :

- Tagger les `registered_model` MLflow (`client.set_registered_model_tag(name, "modality", learner.modality)`)
- Permettre la cascade automatique des alias `@active_text` / `@active_image` (cf. `refresh_modality_alias`)

| BaseLearner | Modality | embed_dim | Trainable params |
|-------------|----------|-----------|------------------|
| `CamembertFrozen` | text | 768 | 0 (no-op) |
| `TextCNN` | text | 3072 | ~20M |
| `CamembertLoRA` *(Bloc N)* | text | 768 | ~2.4M |
| `ResNet18Frozen` | image | 512 | 0 (no-op) |
| `ResNet50PartialFT` | image | 2048 | ~14M |
| `ResNet18FullFT` *(Bloc N)* | image | 512 | ~11.7M |

## Système d'aliases MLflow à 3 niveaux

Le projet utilise une mécanique d'aliases MLflow découplée pour gérer indépendamment la sélection des base learners individuels (`@active`), la propagation au niveau modalité (`@active_text`, `@active_image`), et la sélection des modèles assemblés en production (`@champion`).

### Niveau 1 — `@active` (par base learner)

- **Cible** : une version d'un `rakuten-base-{name}` (ex: `rakuten-base-textcnn @active` → v3)
- **Arbitre** : F1 weighted sur `is_val_selection_v{ACTIVE_VAL_SELECTION_VERSION}`
- **Promotion** : automatique à la fin de `BaseLearnerExperiment.fit()`, si delta F1 > 0.005
- **Helper** : `compute_promotion_decision(name, run_id_new, threshold=0.005)`

### Niveau 2 — `@active_text`, `@active_image` (par modalité projet)

- **Cible** : le meilleur `rakuten-base-*` de la modalité, parmi tous ceux qui portent `@active`
- **Arbitre** : F1 weighted sur `is_val_selection_v{ACTIVE_VAL_SELECTION_VERSION}` (lu depuis les runs MLflow des `@active`)
- **Promotion** : cascade automatique après chaque `@active` mis à jour. La fonction `refresh_modality_alias(modality)` :
  1. Scan tous les `rakuten-base-*` taggés `modality={modality}` avec un `@active`
  2. Pose `@active_{modality}` sur celui ayant le meilleur F1 val_selection
  3. Retire l'alias des autres candidats
- **Helper de résolution** : `resolve_active_modality(modality) → (registered_model_name, version)`

### Niveau 3 — `@champion` (par modèle assemblé)

- **Cible** : une version d'un modèle assemblé (ex: `rakuten-m2-benchmark @champion` → v5)
- **Arbitre** : `eval_gold/f1_weighted` sur le **gold test set** (test métier)
- **Promotion** : via la mécanique historique `evaluate_promotion_via_logged_metrics` (cf. checklist Bloc P)
- **Conservé** depuis Phase 0, indépendant de la mécanique val_selection

## Convention `val_selection` versionné

Pour arbitrer les promotions `@active` sans utiliser le gold (réservé au test métier), Phase 1 introduit un **3ème split orthogonal** : `is_val_selection_v{N}`.

| Version | Sur-ensemble | Créé à | Cas d'usage |
|---------|--------------|--------|-------------|
| `v1` | batch_1 non-gold (~30k) | Démarrage Phase 1 (`init_val_selection.py --version 1`) | Tous les fits Phase 1 |
| `v2` | batch_1 ∪ batch_2 non-gold | Ingestion batch_2 (Phase 2) | Refit Phase 2 |
| `v3` | batch_1 ∪ batch_2 ∪ batch_3 non-gold | Ingestion batch_3 (Phase 2) | Refit Phase 2 |

La version active est résolue par la variable d'environnement `ACTIVE_VAL_SELECTION_VERSION` (ou Airflow Variable), via `get_active_val_selection_version()`.

## Garde-fou cache parquet ↔ alias

Chaque cache parquet d'embeddings extra (produit par `BaseLearnerExperiment.fit()`) contient 2 colonnes constantes :
- `source_model_name` (ex: `"rakuten-base-textcnn"`)
- `source_model_version` (ex: `3`)

Au `DataModule.setup()`, pour chaque `extra_embedding_cache` (cf. Bloc M.7), le DataModule vérifie :
1. Les 2 colonnes sont uniques (1 valeur chacune)
2. `source_model_version` correspond bien à `client.get_model_version_by_alias(name, "active").version`

Sinon, erreur explicite : `"Cache désynchronisé, @active de {name} pointe vers v{X}, cache produit par v{Y}. Relance fit_base_learner."`

Ce garde-fou empêche de consommer un cache d'embeddings produit par une ancienne version qui ne serait plus celle de référence.

## Workflow standard d'un base learner deep (TextCNN, ResNet50PartialFT, ...)

```python
# Côté BaseLearnerExperiment (cf. Bloc M.4)
ACTIVE_VAL_SELECTION_VERSION=1  # env var

datamodule.setup()  # lit la version, charge le parquet, calcule les splits

# Splits standards 80/20 sur train_pool_effective
X_train, y_train = datamodule.get_sklearn_data("train", include_raw=True)
X_val, y_val = datamodule.get_sklearn_data("val", include_raw=True)

learner = TextCNN(...)
learner.fit(X_train, y_train, X_val, y_val)  # early stopping sur val_f1_weighted

# Évaluation sur val_selection (arbitre @active)
X_vs, y_vs = datamodule.get_sklearn_data("val_selection", include_raw=True)
y_pred = learner.predict_proba(X_vs).argmax(axis=1)
f1 = f1_score(y_vs, y_pred, average="weighted")
mlflow.log_metric("val_selection_v1/f1_weighted", f1)

# Extract embeddings sur _df_full, write parquet, dvc push
# ... (cf. Bloc M.4 pour les détails)

# Log model + tag modality + promotion conditionnelle
mlflow.sklearn.log_model(learner, registered_model_name=f"rakuten-base-{learner.name}")
client.set_registered_model_tag(name=..., key="modality", value=learner.modality)

if compute_promotion_decision(name, run_id_new, threshold=0.005):
    client.set_registered_model_alias(name, "active", new_version)
    refresh_modality_alias(learner.modality)  # cascade vers @active_text / @active_image
```

## Fichiers de ce dossier

- `_base.py` : ABC `BaseLearner` + `Modality` type alias
- `text/`
  - `camembert_frozen.py` : encoder texte frozen (M2 v4 baseline)
  - `textcnn.py` : Yoon Kim 2014 from-scratch (M2.2)
  - `camembert_lora.py` : *(à venir Bloc N)* CamemBERT + LoRA r=16 (M2.1)
- `image/`
  - `resnet18_frozen.py` : encoder image frozen (M2 v4 baseline)
  - `resnet50_partial_ft.py` : partial FT layer3+layer4 (M2.2)
  - `resnet18_full_ft.py` : *(à venir Bloc N)* full fine-tune ImageNet→Rakuten (M2.1)

## Voir aussi

- `src/models/utils.py` : helpers MLflow alias (`@active`, `@active_text`, `@active_image`, `@champion`)
- `src/data/init_val_selection.py` : initialisation one-shot d'une version de val_selection
- `src/experiments/datamodule/rakuten_datamodule.py` : 3 niveaux de splits exposés via `get_sklearn_data`
- `checklist_phase_1.md` : avancement projet
