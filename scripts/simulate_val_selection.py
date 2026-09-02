"""
Sanity check [ADR-003] — simulation READ-ONLY de la construction monotone
de val_selection pour les batches 2 et 3.

N'ÉCRIT RIEN dans Mongo. Calcule ce que produirait rebase_val_selection
après correctif, et le compare à l'état actuel.

Racine de l'union : `is_val_selection_v1` réel, conservé (le batch 1 est le
bootstrap manuel, la purge s'arrête avant lui).

Vérifie cinq propriétés :
    P1  monotonie      v1 ⊆ v2 ⊆ v3
    P2  orthogonalité  val_selection ∩ gold = ∅
    P3  proportion     ~10 % de chaque batch, ~10 % du total
    P4  déterminisme   deux exécutions donnent le même résultat
    P5  stratification aucune classe absente de la sélection

Et quantifie l'écart avec l'état actuel (le tirage global fautif).

Usage :
    python -m scripts.simulate_val_selection
    python -m scripts.simulate_val_selection --batches 2 3
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.mongo_utils import get_db
# Constantes importées du module réel : si elles changent là-bas, la
# simulation suit. Ne PAS les redéfinir ici.
from src.data.rebase_val_selection import (
    VAL_SELECTION_FRACTION,
    VAL_SELECTION_SEED,
)

_OK = "  [OK]  "
_KO = "  [KO]  "
_INFO = "  [--]  "


def _section(t: str) -> None:
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def _load_batch(db, batch_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge (productids, labels) du batch n, non-gold.
    Reproduit exactement le chemin de rebase_val_selection §1.
    """
    col = db["X_raw_data_batches"]
    docs = list(col.find(
        {"batch_id": batch_id, "is_gold": False},
        {"_id": 0, "productid": 1, "prdtypecode": 1},
    ))
    if not docs:
        raise RuntimeError(f"Batch {batch_id} : sur-ensemble non-gold vide.")

    if "prdtypecode" not in docs[0]:
        pids = [d["productid"] for d in docs]
        y_map = {
            d["productid"]: d["prdtypecode"]
            for d in db["Y_raw_data_batches"].find(
                {"productid": {"$in": pids}},
                {"_id": 0, "productid": 1, "prdtypecode": 1},
            )
        }
        for d in docs:
            d["prdtypecode"] = y_map.get(d["productid"])

    missing = [d["productid"] for d in docs if d.get("prdtypecode") is None]
    if missing:
        raise RuntimeError(
            f"Batch {batch_id} : {len(missing)} docs sans prdtypecode "
            f"(ex. {missing[:5]}). Stratification impossible."
        )

    return (
        np.array([d["productid"] for d in docs]),
        np.array([d["prdtypecode"] for d in docs]),
    )


def _split_batch(pids: np.ndarray, labels: np.ndarray, batch_id: int) -> set[int]:
    """Split stratifié 10 % du batch n — même appel que le module réel."""
    uniq, counts = np.unique(labels, return_counts=True)
    too_rare = [(int(c), int(k)) for c, k in zip(uniq, counts) if k < 2]
    if too_rare:
        raise RuntimeError(
            f"Batch {batch_id} : classes < 2 individus {too_rare[:5]} — "
            f"stratification impossible."
        )
    _, idx = train_test_split(
        np.arange(len(pids)),
        test_size=VAL_SELECTION_FRACTION,
        stratify=labels,
        random_state=VAL_SELECTION_SEED,
    )
    return set(pids[idx].tolist())


def _flag_set(db, field: str) -> set[int]:
    return {
        d["productid"]
        for d in db["X_raw_data_batches"].find({field: True}, {"_id": 0, "productid": 1})
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Simulation read-only de val_selection.")
    p.add_argument("--batches", type=int, nargs="+", default=[2, 3])
    args = p.parse_args()

    db = get_db()
    col = db["X_raw_data_batches"]
    failures: list[str] = []

    # ------------------------------------------------------------------ #
    _section("0. ÉTAT ACTUEL EN BASE")
    all_batches = sorted({d for d in col.distinct("batch_id") if d is not None})
    gold = _flag_set(db, "is_gold")
    print(f"  batches présents : {all_batches}")
    print(f"  gold             : {len(gold)}")
    for b in all_batches:
        n = col.count_documents({"batch_id": b})
        ng = col.count_documents({"batch_id": b, "is_gold": True})
        print(f"    batch {b} : n={n:6d}  gold={ng:5d} ({ng/max(n,1):.1%})")

    current: dict[int, set[int]] = {}
    for v in sorted(set([1] + args.batches)):
        f = f"is_val_selection_v{v}"
        s = _flag_set(db, f)
        if s:
            current[v] = s
            per = {b: len(s & {d["productid"] for d in col.find({"batch_id": b}, {"productid": 1})})
                   for b in all_batches}
            print(f"  {f} : {len(s):5d}  par batch {per}")

    if 1 not in current:
        print(f"{_KO}is_val_selection_v1 absent — racine de l'union manquante.")
        return 1
    print(f"{_OK}Racine v1 présente : {len(current[1])} productids (conservée)")

    # ------------------------------------------------------------------ #
    _section("1. SIMULATION DE LA CONSTRUCTION MONOTONE")
    sim: dict[int, set[int]] = {1: set(current[1])}
    new_only: dict[int, set[int]] = {}

    for b in args.batches:
        pids, labels = _load_batch(db, b)
        fresh = _split_batch(pids, labels, b)
        new_only[b] = fresh
        sim[b] = sim[b - 1] | fresh
        print(f"  v{b} = v{b-1} ∪ split_10%(batch {b})")
        print(f"     sur-ensemble batch {b} (non-gold) : {len(pids):6d}")
        print(f"     nouveaux                          : {len(fresh):6d} "
              f"({len(fresh)/len(pids):.2%})")
        print(f"     hérités de v{b-1}                   : {len(sim[b-1]):6d}")
        print(f"     → v{b}                             : {len(sim[b]):6d}")

    # ------------------------------------------------------------------ #
    _section("2. INVARIANTS")
    versions = sorted(sim)

    # P1 monotonie
    for a, b in zip(versions, versions[1:]):
        ok = sim[a] <= sim[b]
        (print(f"{_OK}P1 monotonie v{a} ⊆ v{b}") if ok
         else (print(f"{_KO}P1 monotonie v{a} ⊆ v{b} — {len(sim[a]-sim[b])} perdus"),
               failures.append(f"monotonie v{a}⊆v{b}")))

    # P2 orthogonalité gold
    for v in versions:
        inter = sim[v] & gold
        ok = not inter
        (print(f"{_OK}P2 v{v} ∩ gold = ∅") if ok
         else (print(f"{_KO}P2 v{v} ∩ gold = {len(inter)}"),
               failures.append(f"gold∩v{v}")))

    # P3 proportion
    for v in versions:
        pop = col.count_documents({"batch_id": {"$lte": v}, "is_gold": False})
        frac = len(sim[v]) / max(pop, 1)
        ok = 0.085 <= frac <= 0.115
        line = f"P3 v{v} = {frac:.2%} du non-gold cumulé ({len(sim[v])}/{pop})"
        (print(_OK + line) if ok
         else (print(_KO + line), failures.append(f"proportion v{v}")))

    # P4 déterminisme
    for b in args.batches:
        pids, labels = _load_batch(db, b)
        again = _split_batch(pids, labels, b)
        ok = again == new_only[b]
        (print(f"{_OK}P4 déterminisme batch {b} (2 tirages identiques)") if ok
         else (print(f"{_KO}P4 déterminisme batch {b} — {len(again ^ new_only[b])} divergences"),
               failures.append(f"déterminisme b{b}")))

    # P5 couverture des classes
    y = db["Y_raw_data_batches"]
    last = versions[-1]
    labs = [d["prdtypecode"] for d in y.find(
        {"productid": {"$in": list(sim[last])}}, {"_id": 0, "prdtypecode": 1})]
    n_cls = len(set(labs))
    ok = n_cls == 27
    line = f"P5 v{last} couvre {n_cls}/27 classes"
    (print(_OK + line) if ok else (print(_KO + line), failures.append("couverture classes")))
    if labs:
        c = Counter(labs)
        rare = c.most_common()[-3:]
        print(f"{_INFO}classes les plus rares dans v{last} : {rare}")

    # ------------------------------------------------------------------ #
    _section("3. COMPOSITION PAR BATCH D'ORIGINE")
    batch_of = {}
    for b in all_batches:
        for d in col.find({"batch_id": b}, {"_id": 0, "productid": 1}):
            batch_of[d["productid"]] = b
    for v in versions:
        comp = Counter(batch_of.get(pid, "?") for pid in sim[v])
        tot = {b: col.count_documents({"batch_id": b, "is_gold": False}) for b in all_batches}
        detail = "  ".join(
            f"b{b}:{comp.get(b,0)}/{tot.get(b,0)} ({comp.get(b,0)/max(tot.get(b,1),1):.1%})"
            for b in all_batches
        )
        print(f"  v{v} : {detail}")

    # ------------------------------------------------------------------ #
    _section("4. ÉCART AVEC L'ÉTAT ACTUEL (tirage global fautif)")
    for v in args.batches:
        if v not in current:
            print(f"  v{v} : absent en base — rien à comparer.")
            continue
        cur, new = current[v], sim[v]
        print(f"  v{v} : actuel={len(cur):5d}  simulé={len(new):5d}  "
              f"communs={len(cur & new):5d}  "
              f"sortiraient={len(cur - new):5d}  entreraient={len(new - cur):5d}")

    # Le chiffre qui a motivé l'ADR : produits du train b_{n-1} devenus val b_n
    if 2 in current and 3 in current:
        leak = len(current[3] - current[2] - set(
            d["productid"] for d in col.find({"batch_id": 3}, {"_id": 0, "productid": 1})
        ))
        print(f"\n{_INFO}Fuite actuelle : {leak} produits des batches 1-2 sont dans "
              f"v3 sans être dans v2")
        print(f"{_INFO}→ ils étaient dans le TRAIN au batch 2, donc mémorisés par "
              f"les titulaires.")
        sim_leak = len(sim[3] - sim[2] - set(
            d["productid"] for d in col.find({"batch_id": 3}, {"_id": 0, "productid": 1})
        )) if 3 in sim and 2 in sim else 0
        print(f"{_INFO}Après correctif : {sim_leak} (doit être 0)")
        if sim_leak != 0:
            failures.append("fuite résiduelle")

    # ------------------------------------------------------------------ #
    _section("VERDICT")
    if failures:
        print(f"{_KO}{len(failures)} invariant(s) violé(s) : {failures}")
        return 1
    print(f"{_OK}Tous les invariants sont respectés.")
    print("  Aucune écriture effectuée — simulation pure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
