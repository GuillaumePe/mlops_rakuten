"""
Diagnostic complet d'un cycle (batch n) — inventaire MLflow exhaustif.

Répond à cinq questions, dans l'ordre où elles doivent être posées :

    1. INVENTAIRE   — quels runs `_b{n}` existent, dans quel experiment,
                      avec quel statut et quelle durée ?
    2. MANQUANTS    — quels runs d'entraînement attendus n'existent PAS ?
                      (un fit vert dans Airflow sans run MLflow = le run a
                      échoué à logger, ou le handler a court-circuité)
    3. ÉCHECS       — quels runs sont en FAILED / KILLED ?
    4. REGISTRY     — quelles versions ont été créées, qui porte quel alias ?
    5. DIVERGENCE   — la F1 lue par le TOURNOI (métrique du run d'origine du
                      champion) vs celle mesurée par eval_gold_champion
                      (run séparé). Ces deux sources ne se rencontrent
                      jamais dans le code : le tournoi fait
                      get_model_version_by_alias -> get_run(mv.run_id), donc
                      il ignore le rebaselining. Si elles divergent, la
                      sélection de @production repose sur une mesure périmée
                      ou sur un modèle rechargé différemment.

Lecture seule : aucun alias déplacé, aucun run créé.

Usage :
    python -m scripts.diagnose_batch --batch-id 3
    python -m scripts.diagnose_batch --batch-id 3 --compare-batch 2
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from datetime import datetime

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

STRATEGIES = ("stateless", "stateful")

# Métriques affichées si présentes, dans cet ordre.
METRIC_KEYS = (
    "eval_gold/f1_weighted",
    "val_selection/f1_weighted",
    "val/f1_weighted",
    "f1_weighted",
)


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _name(run) -> str:
    return run.data.tags.get("mlflow.runName", "?")


def _kind(run_name: str) -> str:
    """Catégorise un run par son nom."""
    if run_name.startswith("eval_gold"):
        return "eval_gold"
    if run_name.startswith("compare_and_promote"):
        return "compare_promote"
    if run_name.startswith("tournament"):
        return "tournament"
    if run_name.startswith("reevaluate") or "reeval" in run_name:
        return "reevaluate"
    return "fit"


def _duration(run) -> str:
    if not run.info.end_time or not run.info.start_time:
        return "-"
    s = (run.info.end_time - run.info.start_time) / 1000.0
    return f"{s/60:.1f}min" if s >= 60 else f"{s:.1f}s"


def _metric(run) -> str:
    for k in METRIC_KEYS:
        if k in run.data.metrics:
            return f"{k.split('/')[-1]}={run.data.metrics[k]:.4f}"
    return "-"


def _fetch_runs(client: MlflowClient, batch_id: int):
    """Tous les runs dont le nom se termine par _b{batch_id}, par experiment."""
    suffix = re.compile(rf"_b{batch_id}$")
    out: dict[str, list] = {}
    for exp in client.search_experiments():
        runs = client.search_runs(
            [exp.experiment_id],
            max_results=1000,
            order_by=["attributes.start_time DESC"],
        )
        matched = [r for r in runs if suffix.search(_name(r))]
        if matched:
            out[exp.name] = matched
    return out


def inventory(by_exp: dict[str, list], batch_id: int) -> dict[str, list]:
    _section(f"1. INVENTAIRE DES RUNS _b{batch_id}")
    if not by_exp:
        print(f"  AUCUN run _b{batch_id} dans tout le tracking.")
        return {}

    for exp_name in sorted(by_exp):
        runs = by_exp[exp_name]
        buckets: dict[str, list] = defaultdict(list)
        for r in runs:
            buckets[_kind(_name(r))].append(r)

        counts = " | ".join(f"{k}:{len(v)}" for k, v in sorted(buckets.items()))
        print(f"\n  ── {exp_name}  ({len(runs)} runs — {counts})")
        for r in sorted(runs, key=lambda x: _name(x)):
            status = r.info.status
            flag = "  " if status == "FINISHED" else "!!"
            ts = datetime.fromtimestamp(r.info.start_time / 1000)
            print(f"   {flag} {_name(r):48s} {status:9s} {_duration(r):>8s} "
                  f"{ts:%m-%d %H:%M}  {_metric(r)}")
    return by_exp


def missing_fits(by_exp: dict[str, list], batch_id: int) -> None:
    """
    Un experiment qui contient des eval_gold_*_b{n} mais AUCUN fit _b{n}
    signale un entraînement qui n'a rien enregistré : la tâche Airflow peut
    être verte alors que le modèle n'a jamais été (re)construit.
    """
    _section(f"2. ENTRAÎNEMENTS MANQUANTS AU BATCH {batch_id}")
    anomalies = []
    for exp_name, runs in sorted(by_exp.items()):
        kinds = {_kind(_name(r)) for r in runs}
        if "eval_gold" in kinds and "fit" not in kinds:
            anomalies.append(exp_name)
            evals = [_name(r) for r in runs if _kind(_name(r)) == "eval_gold"]
            print(f"  !! {exp_name}")
            print(f"     eval_gold présents : {len(evals)}, fits _b{batch_id} : 0")
            print(f"     -> le champion évalué date d'un batch ANTÉRIEUR.")
    if not anomalies:
        print("  Aucun experiment sans fit. (Ne prouve pas que TOUS les fits")
        print("  attendus sont là — comparer avec les tâches du DAG Training.)")


def failures(by_exp: dict[str, list]) -> None:
    _section("3. RUNS EN ÉCHEC")
    found = False
    for exp_name, runs in sorted(by_exp.items()):
        for r in runs:
            if r.info.status == "FINISHED":
                continue
            found = True
            print(f"  !! {exp_name} / {_name(r)}")
            print(f"     statut={r.info.status}  durée={_duration(r)}  "
                  f"run_id={r.info.run_id}")
            print(f"     métriques loggées : {sorted(r.data.metrics) or 'AUCUNE'}")
    if not found:
        print("  Aucun run en échec.")


def registry(client: MlflowClient, batch_id: int, day: str | None) -> None:
    _section("4. REGISTRY — versions et alias")
    for rm in sorted(client.search_registered_models(), key=lambda x: x.name):
        if not rm.name.startswith("rakuten-"):
            continue
        versions = client.search_model_versions(f"name='{rm.name}'")
        versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        recent = []
        for v in versions[:6]:
            d = datetime.fromtimestamp(v.creation_timestamp / 1000)
            mark = "*" if (day and d.date().isoformat() == day) else " "
            recent.append(f"{mark}v{v.version}({d:%m-%d %H:%M})"
                          f"{'[' + ','.join(v.aliases) + ']' if v.aliases else ''}")
        print(f"\n  ── {rm.name}  ({len(versions)} versions)")
        print("     " + "  ".join(recent))
    if day:
        print(f"\n  (* = version créée le {day})")


def divergence(client: MlflowClient, by_exp: dict[str, list], batch_id: int) -> None:
    """
    Compare la F1 vue par le TOURNOI et celle mesurée par eval_gold_champion.

    Le tournoi lit la métrique du run d'ORIGINE du champion ; le rebaselining
    écrit dans un run SÉPARÉ. Les deux ne se croisent nulle part dans le code.
    """
    _section(f"5. DIVERGENCE — tournoi vs eval_gold_champion (_b{batch_id})")

    # Côté rebaseline : dernier eval_gold_*_{strategy}_b{n} par (registry, strat).
    rebase: dict[tuple[str, str], tuple[float, str, int]] = {}
    for runs in by_exp.values():
        for r in runs:
            if _kind(_name(r)) != "eval_gold":
                continue
            model = r.data.tags.get("rescored_model", "")
            f1 = r.data.metrics.get("eval_gold/f1_weighted")
            if not model or f1 is None:
                continue
            reg, _, alias = model.partition("@")
            strat = alias.replace("champion_", "")
            n_gold = int(r.data.params.get("n_gold", 0) or 0)
            key = (reg, strat)
            if key not in rebase or r.info.start_time > rebase[key][2]:
                rebase[key] = (float(f1), f"n_gold={n_gold}", r.info.start_time)

    header = (f"  {'registry':32s} {'strat':10s} {'tournoi':>9s} "
              f"{'rebaseline':>11s} {'écart':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for rm in sorted(client.search_registered_models(), key=lambda x: x.name):
        if not rm.name.startswith("rakuten-"):
            continue
        for strat in STRATEGIES:
            try:
                mv = client.get_model_version_by_alias(rm.name, f"champion_{strat}")
                run = client.get_run(mv.run_id)
            except MlflowException:
                continue
            f1_tournoi = run.data.metrics.get("eval_gold/f1_weighted")
            entry = rebase.get((rm.name, strat))
            f1_rebase = entry[0] if entry else None

            t = f"{f1_tournoi:.4f}" if f1_tournoi is not None else "absent"
            b = f"{f1_rebase:.4f}" if f1_rebase is not None else "ABSENT"
            if f1_tournoi is not None and f1_rebase is not None:
                delta = f1_rebase - f1_tournoi
                d = f"{delta:+.4f}"
                rows.append((rm.name, strat, delta))
            else:
                d = "-"
            print(f"  {rm.name:32s} {strat:10s} {t:>9s} {b:>11s} {d:>9s}")

    if not rows:
        return

    print()
    big = [r for r in rows if abs(r[2]) > 0.005]
    if not big:
        print("  Aucune divergence au-delà de 0.005 : le tournoi et le")
        print("  rebaselining mesurent la même chose.")
        return

    print(f"  {len(big)} divergence(s) au-delà de 0.005 :")
    by_strat = defaultdict(list)
    for reg, strat, delta in big:
        by_strat[strat].append(delta)
        print(f"    {reg} [{strat}] : {delta:+.4f}")

    print()
    for strat, deltas in sorted(by_strat.items()):
        signs = {d > 0 for d in deltas}
        direction = "systématiquement " + ("SUR" if all(signs) else "SOUS") \
            if len(signs) == 1 else "de signe variable"
        print(f"    lignée {strat} : {len(deltas)} écart(s), {direction}"
              f"-évaluée par le tournoi (moy {sum(deltas)/len(deltas):+.4f})")
    print()
    print("  Un biais qui frappe UNE SEULE lignée n'est pas du bruit : il")
    print("  désigne le chemin de rechargement du modèle. Un biais des deux")
    print("  côtés désigne plutôt un changement du jeu gold entre les mesures")
    print("  (vérifier n_gold dans le bloc 1).")


def compare_batches(client: MlflowClient, a: int, b: int) -> None:
    _section(f"6. COMPARAISON DES TOURNOIS b{a} vs b{b}")
    exp = client.get_experiment_by_name("training_compare")
    if exp is None:
        print("  Experiment 'training_compare' absent — aucun tournoi loggé.")
        return
    runs = client.search_runs(
        [exp.experiment_id], max_results=200,
        order_by=["attributes.start_time DESC"],
    )
    found = {}
    for r in runs:
        n = _name(r)
        for bid in (a, b):
            if n == f"tournament_b{bid}" and bid not in found:
                found[bid] = r
    for bid in (a, b):
        if bid not in found:
            print(f"  tournament_b{bid} : ABSENT")
    if len(found) < 2:
        return

    ra, rb = found[a], found[b]
    keys = sorted(set(ra.data.metrics) | set(rb.data.metrics))
    print(f"  {'métrique':50s} {'b'+str(a):>9s} {'b'+str(b):>9s} {'Δ':>9s}")
    print("  " + "-" * 80)
    for k in keys:
        va, vb = ra.data.metrics.get(k), rb.data.metrics.get(k)
        sa = f"{va:.4f}" if va is not None else "-"
        sb = f"{vb:.4f}" if vb is not None else "-"
        sd = f"{vb-va:+.4f}" if (va is not None and vb is not None) else "-"
        mark = "  " if sd in ("-", "+0.0000") else "!!"
        print(f"  {mark}{k:48s} {sa:>9s} {sb:>9s} {sd:>9s}")
    print()
    print("  Une métrique ranking/ INCHANGÉE entre deux batches signifie que")
    print("  le champion de cette lignée n'a pas été remplacé ET que sa")
    print("  métrique d'origine n'a pas été réécrite.")


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnostic complet d'un batch.")
    p.add_argument("--batch-id", type=int, required=True)
    p.add_argument("--compare-batch", type=int, default=None)
    p.add_argument("--day", default=None,
                   help="Marque les versions créées ce jour (YYYY-MM-DD).")
    p.add_argument("--tracking-uri", default=TRACKING_URI)
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(args.tracking_uri)

    print(f"Tracking : {args.tracking_uri}")
    print(f"Batch    : {args.batch_id}")

    by_exp = _fetch_runs(client, args.batch_id)
    inventory(by_exp, args.batch_id)
    missing_fits(by_exp, args.batch_id)
    failures(by_exp)
    registry(client, args.batch_id, args.day)
    divergence(client, by_exp, args.batch_id)
    if args.compare_batch is not None:
        compare_batches(client, args.compare_batch, args.batch_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
