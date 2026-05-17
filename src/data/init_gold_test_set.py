# scripts/init_gold_test_set.py
"""
Initialise le gold test set : marque chaque document de X_raw_data_batches
avec un champ is_gold (boolean) selon la fonction de hash déterministe.

À exécuter UNE FOIS au démarrage du projet, après la migration Mongo Atlas.
Idempotent : peut être ré-exécuté sans effet de bord, vérifie la cohérence.
"""
from __future__ import annotations
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

from src.data.gold_test import is_gold, GOLD_HASH_MODULO

load_dotenv()

MONGO_URI = os.environ["MONGO_URI"]
DB_NAME = os.environ.get("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN")
COLLECTION = "X_raw_data_batches"
BATCH_SIZE = 5000


def main() -> int:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[COLLECTION]

    n_total = coll.count_documents({})
    n_already_marked = coll.count_documents({"is_gold": {"$exists": True}})
    print(f"[init_gold] Collection {COLLECTION} : {n_total} docs total")
    print(f"[init_gold] Déjà marqués (is_gold exists) : {n_already_marked}")

    if n_already_marked == n_total:
        print("[init_gold] Tous les docs sont déjà marqués. Vérification cohérence...")
        # Validation : la cardinalité gold doit être ~10%
        n_gold = coll.count_documents({"is_gold": True})
        pct = 100 * n_gold / n_total
        expected_pct = 100 / GOLD_HASH_MODULO
        print(f"[init_gold] Gold count : {n_gold}/{n_total} ({pct:.2f}%, attendu ~{expected_pct:.0f}%)")
        if abs(pct - expected_pct) > 1.0:
            print("[init_gold] WARN: cardinalité gold anormale, vérifier la fonction is_gold")
            return 1
        print("[init_gold] ✓ Cohérence OK, rien à faire")
        return 0

    # Bulk update par batches pour ne pas saturer Mongo
    print(f"[init_gold] Marquage en cours par batches de {BATCH_SIZE}...")
    cursor = coll.find({}, {"productid": 1, "_id": 1})
    ops = []
    n_processed = 0
    n_gold_marked = 0

    for doc in cursor:
        gold_flag = is_gold(doc["productid"])
        if gold_flag:
            n_gold_marked += 1
        ops.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": {"is_gold": gold_flag}})
        )
        if len(ops) >= BATCH_SIZE:
            coll.bulk_write(ops, ordered=False)
            n_processed += len(ops)
            print(f"[init_gold]   ... {n_processed}/{n_total}")
            ops = []

    if ops:
        coll.bulk_write(ops, ordered=False)
        n_processed += len(ops)

    print(f"[init_gold] ✓ Terminé : {n_processed} docs marqués, dont {n_gold_marked} gold")
    print(f"[init_gold] Ratio gold : {100*n_gold_marked/n_processed:.2f}%")

    # Index pour accélérer les requêtes futures
    coll.create_index([("is_gold", 1), ("batch_id", 1)])
    print("[init_gold] Index (is_gold, batch_id) créé")

    return 0


if __name__ == "__main__":
    sys.exit(main())