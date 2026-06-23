"""
Helpers MongoDB — point d'entrée unique pour les connexions Mongo.

Sur RunPod, le tunnel SOCKS5 (mongo_tunnel.py) réécrit MONGO_URI vers
localhost:27018. Côté Python, aucune configuration proxy nécessaire —
la connexion est transparente.

Usage :
    from src.data.mongo_utils import get_mongo_client, get_db

    client = get_mongo_client()
    db = get_db()
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "MAR25_CMLOPS_RAKUTEN")


def get_mongo_client(uri: str = "", **kwargs) -> MongoClient:
    """Crée un MongoClient. L'URI est lue de l'env (réécrite par le tunnel sur RunPod)."""
    uri = uri or MONGO_URI
    return MongoClient(uri, **kwargs)


def get_db(uri: str = "", db_name: str = ""):
    """Retourne la database par défaut."""
    client = get_mongo_client(uri)
    return client[db_name or DB_NAME]
