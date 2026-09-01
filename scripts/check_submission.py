"""
Audit d'une soumission ENS avant envoi (P4.5e).

Cinq blocs, du moins au plus interpretatif :
    1. SCHEMA      — structure du CSV (contrat ENS, binaire : passe ou non)
    2. VALEURS     — les 27 classes, aucune valeur hors nomenclature
    3. COHERENCE   — le CSV redit-il exactement ce que contient Mongo
    4. DISTRIBUTION— p_pred vs p_train : ecart global + par classe
    5. CONFIANCE   — distribution des max-proba, indice de calibration

Les blocs 1-3 sont des CONTRATS : un echec = soumission invalide, exit 1.
Les blocs 4-5 sont des DIAGNOSTICS : ils ne peuvent pas "echouer", ils
informent. Confondre les deux mene a bloquer une soumission correcte parce
qu'un chi2 est significatif — voir la note sur la puissance ci-dessous.

Usage :
    python -m scripts.check_submission --batch-id 2
    python -m scripts.check_submission --batch-id 2 --submission data/submissions/submission_batch2.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from src.data.mongo_utils import get_db

load_dotenv()

DATA_ROOT = Path(os.getenv("DATA_ROOT", "."))
X_TEST_CSV = DATA_ROOT / "data/raw_data_test/X_test_update.csv"
Y_TRAIN_CSV = DATA_ROOT / "data/raw_data/Y_train_update.csv"

VALID_CLASSES = {
    10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302,
    1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582,
    2583, 2585, 2705, 2905,
}

_OK = "  [OK]  "
_KO = "  [KO]  "
_INFO = "  [--]  "


class Report:
    """Accumule les echecs de CONTRAT uniquement (blocs 1-3)."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        if condition:
            print(f"{_OK}{label}" + (f" — {detail}" if detail else ""))
        else:
            print(f"{_KO}{label}" + (f" — {detail}" if detail else ""))
            self.failures.append(label)
        return condition

    def info(self, label: str) -> None:
        print(f"{_INFO}{label}")


def _section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ------------------------------------------------------------------ #
# 1. SCHEMA                                                          #
# ------------------------------------------------------------------ #
def check_schema(path: Path, rep: Report) -> list[tuple[int, int]]:
    _section("1. SCHEMA — contrat de format ENS")

    raw = path.read_bytes()
    rep.check(b"\r\n" not in raw, "Terminaisons LF (pas de CRLF)")
    rep.check(raw.endswith(b"\n"), "Newline finale presente")

    text = raw.decode("utf-8")
    lines = text.splitlines()
    rep.check(lines[0] == ",prdtypecode",
              "En-tete exact", f"lu : {lines[0]!r}")

    rows: list[tuple[int, int]] = []
    malformed = 0
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            malformed += 1
            continue
        rows.append((int(parts[0]), int(parts[1])))

    rep.check(malformed == 0, "Toutes les lignes en 2 champs entiers",
              f"{malformed} malformees")

    idx = [r[0] for r in rows]
    rep.check(len(set(idx)) == len(idx), "Index unique",
              f"{len(idx) - len(set(idx))} doublons")
    rep.check(idx == sorted(idx), "Index croissant")
    expected = list(range(min(idx), min(idx) + len(idx))) if idx else []
    rep.check(idx == expected, "Index contigu sans trou",
              f"[{min(idx)}..{max(idx)}]" if idx else "vide")

    print(f"{_INFO}{len(rows)} lignes de donnees")
    return rows


# ------------------------------------------------------------------ #
# 2. EXHAUSTIVITE + VALEURS                                          #
# ------------------------------------------------------------------ #
def check_coverage(rows: list[tuple[int, int]], rep: Report) -> None:
    _section("2. EXHAUSTIVITE ET NOMENCLATURE")

    with X_TEST_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_test = sum(1 for r in reader if r)

    rep.check(len(rows) == n_test, "Couverture complete de X_test",
              f"{len(rows)} soumis / {n_test} attendus")

    codes = {c for _, c in rows}
    unknown = codes - VALID_CLASSES
    rep.check(not unknown, "Tous les codes dans la nomenclature",
              f"inconnus : {sorted(unknown)}" if unknown else "27 classes de reference")

    never = VALID_CLASSES - codes
    if never:
        rep.info(f"{len(never)} classe(s) JAMAIS predite(s) : {sorted(never)}")
        rep.info("  -> rappel nul sur ces classes, penalise le F1 pondere")
    else:
        print(f"{_OK}Les 27 classes sont toutes predites au moins une fois")


# ------------------------------------------------------------------ #
# 3. COHERENCE CSV <-> MONGO                                         #
# ------------------------------------------------------------------ #
def check_mongo_consistency(rows, batch_id: int, rep: Report) -> list[dict]:
    _section("3. COHERENCE CSV <-> Prediction_test")

    db = get_db()
    docs = list(db["Prediction_test"].find(
        {"batch_id": batch_id},
        {"_id": 0, "productid": 1, "prediction": 1, "confidence": 1, "model": 1},
    ))
    rep.check(bool(docs), "Prédictions presentes en base",
              f"{len(docs)} documents (batch_id={batch_id})")
    if not docs:
        return []

    models = {d.get("model") for d in docs}
    rep.check(len(models) == 1, "Modele unique sur le batch", str(sorted(models)))

    preds = {int(d["productid"]): int(d["prediction"]) for d in docs}
    rep.check(len(preds) == len(docs), "Aucun productid duplique en base")

    # Rejeu de la jointure, independamment de build_submission :
    # si les deux chemins divergent, l'un des deux a un bug.
    with X_TEST_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        pid_col = header.index("productid")
        expected = [(int(r[0]), preds.get(int(r[pid_col]))) for r in reader if r]

    mismatch = sum(1 for (i1, c1), (i2, c2) in zip(rows, expected)
                   if i1 != i2 or c1 != c2)
    rep.check(mismatch == 0, "CSV == rejeu independant de la jointure",
              f"{mismatch} divergences")

    return docs


# ------------------------------------------------------------------ #
# 4. DISTRIBUTION                                                    #
# ------------------------------------------------------------------ #
def _load_train_counts() -> Counter:
    counts: Counter = Counter()
    with Y_TRAIN_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = header.index("prdtypecode")
        for row in reader:
            if row:
                counts[int(row[col])] += 1
    return counts


def _chi2_sf(x: float, k: int) -> float:
    """
    P(X > x) pour X ~ chi2(k), sans scipy.

    Serie de Wilson-Hilferty : ((x/k)^(1/3) - (1 - 2/(9k))) / sqrt(2/(9k))
    suit approximativement N(0,1). Precision largement suffisante ici, on ne
    fait qu'un ordre de grandeur.
    """
    if x <= 0:
        return 1.0
    t = (2.0 / (9.0 * k))
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - t)) / math.sqrt(t)
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def analyse_distribution(rows: list[tuple[int, int]]) -> None:
    _section("4. DISTRIBUTION p_pred vs p_train")

    pred_counts = Counter(c for _, c in rows)
    train_counts = _load_train_counts()

    n_pred = sum(pred_counts.values())
    n_train = sum(train_counts.values())
    classes = sorted(VALID_CLASSES)

    p_pred = {c: pred_counts.get(c, 0) / n_pred for c in classes}
    p_train = {c: train_counts.get(c, 0) / n_train for c in classes}

    # --- Ecarts globaux ---
    tv = 0.5 * sum(abs(p_pred[c] - p_train[c]) for c in classes)
    kl = sum(p_pred[c] * math.log(p_pred[c] / p_train[c])
             for c in classes if p_pred[c] > 0 and p_train[c] > 0)

    # chi2 d'adequation : H0 = le test suit la loi du train.
    chi2 = sum(
        (pred_counts.get(c, 0) - n_pred * p_train[c]) ** 2 / (n_pred * p_train[c])
        for c in classes if p_train[c] > 0
    )
    dof = len(classes) - 1
    p_value = _chi2_sf(chi2, dof)
    cramers_v = math.sqrt(chi2 / (n_pred * dof))

    print(f"  Distance en variation totale : {tv:.4f}")
    print(f"  KL(p_pred || p_train)        : {kl:.4f} nats")
    print(f"  chi2({dof})                    : {chi2:,.1f}  (p ~ {p_value:.2e})")
    print(f"  V de Cramer                  : {cramers_v:.4f}")
    print()
    print("  Lecture : avec n=13812 la puissance du chi2 est enorme, un ecart")
    print("  de quelques dixiemes de point suffit a rejeter H0. C'est la TAILLE")
    print("  D'EFFET qui informe, pas la p-value. Reperes usuels pour V :")
    print("    V < 0.10 negligeable | 0.10-0.30 modere | > 0.30 fort")
    print("  De meme, TV < 0.05 signifie que les deux lois sont pratiquement")
    print("  superposables du point de vue metier.")

    # --- Par classe ---
    print(f"\n  {'classe':>7} {'n_pred':>7} {'p_pred':>8} {'p_train':>8} "
          f"{'ratio':>7} {'log2':>7}")
    print("  " + "-" * 50)
    deviations: list[tuple[float, int, str]] = []
    for c in classes:
        r = (p_pred[c] / p_train[c]) if p_train[c] > 0 else float("inf")
        lg = math.log2(r) if r > 0 and math.isfinite(r) else float("-inf")
        line = (f"  {c:>7} {pred_counts.get(c, 0):>7} {p_pred[c]:>8.4f} "
                f"{p_train[c]:>8.4f} {r:>7.2f} {lg:>7.2f}")
        print(line)
        if math.isfinite(lg):
            deviations.append((abs(lg), c, line.strip()))

    # --- Signature de sur-prediction des classes majoritaires ---
    majority = sorted(classes, key=lambda c: -p_train[c])[:5]
    minority = sorted(classes, key=lambda c: p_train[c])[:5]
    maj_ratio = (sum(p_pred[c] for c in majority)
                 / max(sum(p_train[c] for c in majority), 1e-12))
    min_ratio = (sum(p_pred[c] for c in minority)
                 / max(sum(p_train[c] for c in minority), 1e-12))

    print(f"\n  Masse des 5 classes les plus frequentes : ratio {maj_ratio:.3f}")
    print(f"  Masse des 5 classes les plus rares      : ratio {min_ratio:.3f}")
    print()
    if maj_ratio > 1.02 and min_ratio < 0.98:
        print("  -> Signature classique du decodage argmax : la regle de Bayes")
        print("     minimise l'erreur 0-1, pas le F1 pondere. Sous incertitude")
        print("     elle penche vers le prior, donc gonfle les classes")
        print("     frequentes au detriment des rares. Correctif possible :")
        print("     ajustement du prior (diviser les proba par p_train^tau)")
        print("     ou seuillage par classe cale sur la validation.")
    else:
        print("  -> Pas de derive marquee vers le prior : le decodage argmax")
        print("     ne semble pas ecraser les classes rares.")

    print("\n  Ecarts les plus forts (|log2 ratio|) :")
    for _, _, line in sorted(deviations, reverse=True)[:5]:
        print(f"    {line}")


# ------------------------------------------------------------------ #
# 5. CONFIANCE                                                       #
# ------------------------------------------------------------------ #
def analyse_confidence(docs: list[dict]) -> None:
    _section("5. CONFIANCE — indice de calibration")

    conf = sorted(float(d["confidence"]) for d in docs if d.get("confidence") is not None)
    if not conf:
        print(f"{_INFO}Aucune confiance enregistree.")
        return

    n = len(conf)
    mean = sum(conf) / n

    def q(p: float) -> float:
        return conf[min(int(p * n), n - 1)]

    print(f"  n = {n}")
    print(f"  moyenne : {mean:.4f}")
    print(f"  p05 {q(0.05):.4f} | q1 {q(0.25):.4f} | median {q(0.50):.4f} "
          f"| q3 {q(0.75):.4f} | p95 {q(0.95):.4f}")

    for t in (0.5, 0.7, 0.9, 0.99):
        share = sum(1 for c in conf if c >= t) / n
        print(f"  part avec confiance >= {t:.2f} : {share:6.2%}")

    print()
    print("  Sans les labels de test, l'ECE est incalculable. Mais la confiance")
    print("  MOYENNE est un estimateur de l'accuracy attendue sous calibration")
    print("  parfaite : E[max_k p_k] = P(argmax correct) si le modele est calibre.")
    print(f"  Confiance moyenne mesuree : {mean:.4f}")
    print("  A comparer au F1 pondere obtenu sur eval_gold. Un ecart de l'ordre")
    print("  de +0.10 reproduirait la sur-confiance de ~10.5 pts deja mesuree")
    print("  sur M2 (ECE 0.105). Le F1 n'en souffre pas — le temperature scaling")
    print("  est invariant par argmax — mais toute regle de decision a seuil")
    print("  (routage vers relecture humaine, abstention) serait mal calee.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit d'une soumission ENS.")
    parser.add_argument("--batch-id", type=int, required=True)
    parser.add_argument("--submission", default="")
    args = parser.parse_args()

    path = Path(args.submission) if args.submission else (
        DATA_ROOT / f"data/submissions/submission_batch{args.batch_id}.csv"
    )
    if not path.is_file():
        print(f"{_KO}Fichier introuvable : {path}")
        return 1

    print(f"Soumission : {path}")
    print(f"Batch      : {args.batch_id}")

    rep = Report()
    rows = check_schema(path, rep)
    check_coverage(rows, rep)
    docs = check_mongo_consistency(rows, args.batch_id, rep)
    analyse_distribution(rows)
    if docs:
        analyse_confidence(docs)

    _section("VERDICT")
    if rep.failures:
        print(f"{_KO}{len(rep.failures)} contrat(s) viole(s) : {rep.failures}")
        print("  NE PAS SOUMETTRE.")
        return 1
    print(f"{_OK}Tous les contrats de format et de coherence sont respectes.")
    print("  Les blocs 4-5 sont des diagnostics : ils n'invalident pas la")
    print("  soumission, ils orientent le prochain cycle d'amelioration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
