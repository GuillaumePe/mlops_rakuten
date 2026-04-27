"""Helper du DAG SimulateDataArrival. À supprimer en prod."""
from pymongo import MongoClient
from datetime import datetime
import random


def inject_random_batch(min_size: int = 500, max_size: int = 5000) -> dict:
    client = MongoClient("mongodb://mongodb:27017")
    db = client["MAR25_CMLOPS_RAKUTEN"]

    pool_ids = [d["_id"] for d in db["X_to_predict_pool"].find(
        {"injected": {"$ne": True}}, {"_id": 1}
    )]
    if not pool_ids:
        return {"injected": 0, "remaining": 0}

    batch_size = min(random.randint(min_size, max_size), len(pool_ids))
    selected = random.sample(pool_ids, batch_size)

    docs = list(db["X_to_predict_pool"].find({"_id": {"$in": selected}}))
    now = datetime.utcnow()
    for d in docs:
        d.pop("_id", None)
        d["arrived_at"] = now
    db["X_to_predict"].insert_many(docs)

    db["X_to_predict_pool"].update_many(
        {"_id": {"$in": selected}}, {"$set": {"injected": True}}
    )

    return {"injected": batch_size, "remaining": len(pool_ids) - batch_size}