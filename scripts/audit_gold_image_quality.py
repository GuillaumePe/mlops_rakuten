"""
Diagnostic autonome — qualité des images du gold set.

Objectif : mesurer le taux réel de corruption d'images AVANT de décider
d'une quelconque stratégie (fallback, quarantaine, stratification). Principe
frugal : ne pas concevoir de gestion de corruption pour un phénomène dont on
n'a pas établi l'existence ni la structure.

Trois mesures, dans l'ordre de décision :
  1. Taux global de corruption τ = P(C=1) sur le gold.
  2. Test d'indépendance χ² entre corruption et label (l'hypothèse i.i.d.-par-classe).
  3. Ventilation des CAUSES : fichier absent (bug infra) vs présent-mais-illisible
     (phénomène distributionnel légitime).

Ne modifie RIEN. Lecture seule (Mongo + disque images).

Usage :
    PYTHONPATH=. python scripts/audit_gold_image_quality.py
    PYTHONPATH=. python scripts/audit_gold_image_quality.py --scope all   # tout X_raw, pas que gold
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

from src.data.mongo_utils import get_db

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
IMAGE_FOLDER_TRAIN = DATA_ROOT / "data/raw_data/images/image_train"


def _image_path(imageid, productid) -> Path:
    return IMAGE_FOLDER_TRAIN / f"image_{imageid}_product_{productid}.jpg"


def _classify(imageid, productid) -> str:
    """
    Retourne l'une des catégories :
      - "ok"        : fichier présent et décodable en RGB
      - "absent"    : fichier introuvable → ANOMALIE INFRA (ne doit pas exister)
      - "illisible" : fichier présent mais Image.open/convert échoue → phénomène prod
    """
    path = _image_path(imageid, productid)
    if not path.exists():
        return "absent"
    try:
        with Image.open(path) as img:
            img.convert("RGB").load()  # force le décodage complet, pas juste l'en-tête
        return "ok"
    except Exception:
        return "illisible"


def _chi2_independence(contingency: np.ndarray) -> tuple[float, float, int]:
    """
    χ² d'indépendance sur une table de contingence (classes × {ok, corrompu}).
    Retourne (chi2, p_value, dof). Implémentation sans scipy si indisponible.
    """
    observed = contingency.astype(float)
    # Retirer les lignes entièrement nulles (classes absentes du scope)
    observed = observed[observed.sum(axis=1) > 0]
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row_sums @ col_sums / total
    # Éviter division par zéro sur colonnes vides
    mask = expected > 0
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum())
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(chi2_dist.sf(chi2, dof))
    except ImportError:
        p_value = float("nan")  # scipy absent → on rend chi2/dof brut
    return chi2, p_value, dof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope", choices=["gold", "all"], default="gold",
        help="'gold' = seulement is_gold=True ; 'all' = tout X_raw_data_batches.",
    )
    args = parser.parse_args()

    db = get_db()

    # 1. Récupérer les docs du scope + leurs labels
    filt = {"is_gold": True} if args.scope == "gold" else {}
    docs = list(db["X_raw_data_batches"].find(
        filt, {"_id": 0, "productid": 1, "imageid": 1}
    ))
    if not docs:
        print(f"[audit] Aucun document pour scope='{args.scope}'. "
              "Gold ingéré ? Lancer ingest_batch d'abord.")
        return

    pids = [d["productid"] for d in docs]
    label_map = {
        d["productid"]: d["prdtypecode"]
        for d in db["Y_raw_data_batches"].find(
            {"productid": {"$in": pids}}, {"_id": 0, "productid": 1, "prdtypecode": 1}
        )
    }

    # 2. Classifier chaque image
    cause_counts = Counter()
    per_class = defaultdict(lambda: {"ok": 0, "corrupted": 0})
    n_unlabeled = 0

    for d in docs:
        cat = _classify(d["imageid"], d["productid"])
        cause_counts[cat] += 1

        label = label_map.get(d["productid"])
        if label is None:
            n_unlabeled += 1
            continue
        if cat == "ok":
            per_class[label]["ok"] += 1
        else:  # absent OU illisible = corrompu au sens "non scorable proprement"
            per_class[label]["corrupted"] += 1

    n_total = len(docs)
    n_ok = cause_counts["ok"]
    n_absent = cause_counts["absent"]
    n_illisible = cause_counts["illisible"]
    n_corrupted = n_absent + n_illisible
    tau = n_corrupted / n_total

    # 3. Rapport
    print("=" * 64)
    print(f"AUDIT QUALITÉ IMAGES — scope='{args.scope}'  (n={n_total})")
    print("=" * 64)
    print(f"  OK (présent + décodable)   : {n_ok:6d}  ({n_ok/n_total:6.2%})")
    print(f"  ABSENT (→ bug infra)       : {n_absent:6d}  ({n_absent/n_total:6.2%})")
    print(f"  ILLISIBLE (→ phénomène prod): {n_illisible:6d}  ({n_illisible/n_total:6.2%})")
    print(f"  ── τ corruption globale    : {tau:6.2%}")
    if n_unlabeled:
        print(f"  (⚠ {n_unlabeled} docs sans label, exclus du χ²)")
    print()

    # 4. Décision guidée
    if n_corrupted == 0:
        print("VERDICT : aucune image corrompue dans le scope.")
        print("  → Le débat 'gold propre vs corrompu' est SANS OBJET sur ce dataset.")
        print("  → Recommandation frugale : PAS de gestion de corruption.")
        print("    Faire crasher bruyamment sur fichier absent (invariant infra),")
        print("    supprimer le fallback tensor-noir silencieux.")
        return

    if n_absent > 0:
        print(f"⚠ {n_absent} fichiers ABSENTS : ce sont des ANOMALIES INFRA")
        print("  (dvc pull incomplet, mauvais chemin, nommage). NE relèvent PAS")
        print("  de la distribution prod. À traiter comme bug, pas comme donnée.")
        print()

    # 5. Test d'indépendance corruption × label (l'hypothèse i.i.d.-par-classe)
    labels_sorted = sorted(per_class.keys())
    contingency = np.array(
        [[per_class[c]["ok"], per_class[c]["corrupted"]] for c in labels_sorted]
    )
    chi2, p_value, dof = _chi2_independence(contingency)

    print("TEST D'INDÉPENDANCE  corruption ⟂ label  (χ²)")
    print(f"  χ² = {chi2:.2f}   dof = {dof}   p-value = {p_value:.4g}")
    print()
    if not np.isnan(p_value):
        if p_value >= 0.05:
            print("  → p ≥ 0.05 : on NE rejette PAS l'indépendance.")
            print("    La corruption est compatible avec un bruit i.i.d. selon les classes.")
            print("    STATISTIQUEMENT SAIN de garder les corrompues dans le gold :")
            print("    biais constant sur toutes les classes, classement des modèles préservé.")
        else:
            print("  → p < 0.05 : on REJETTE l'indépendance.")
            print("    La corruption est CORRÉLÉE au label → biais différentiel par classe.")
            print("    Garder les corrompues telles quelles fausse le f1_weighted de façon")
            print("    dépendante du modèle. Options : stratifier, ou reporter f1 clean vs")
            print("    corrupted séparément.")
    else:
        print("  (scipy absent : p-value non calculée. `pip install scipy` pour le test complet.")
        print("   Interpréter χ² brut vs seuil table à dof donné.)")
    print()

    # 6. Top classes les plus touchées (diagnostic de la corrélation)
    rates = []
    for c in labels_sorted:
        n_c = per_class[c]["ok"] + per_class[c]["corrupted"]
        if n_c > 0:
            rates.append((c, per_class[c]["corrupted"] / n_c, n_c))
    rates.sort(key=lambda t: t[1], reverse=True)
    print("Top 5 classes par taux de corruption (diagnostic corrélation) :")
    for c, r, n_c in rates[:5]:
        print(f"  classe {c:>4} : {r:6.2%}  (n={n_c})")


if __name__ == "__main__":
    main()