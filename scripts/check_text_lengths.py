"""
Quick check : distribution des longueurs en tokens CamemBERT
de la colonne 'text' (designation + description).

Usage : python scripts/check_text_lengths.py
"""
import os
import numpy as np
from pymongo import MongoClient
from transformers import AutoTokenizer
from src.features.utils import clean_description
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongodb:27017")
DB_NAME = "MAR25_CMLOPS_RAKUTEN"

# 1. Charger les textes depuis Mongo (même logique que _setup_raw_for_finetune)
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
docs = list(db["X_train"].find({}, {"designation": 1, "description": 1, "_id": 0}))

texts = []
for d in docs:
    desig = str(d.get("designation", "") or "")
    desc = str(d.get("description", "") or "")
    desc = clean_description(desc)
    texts.append(f"{desig} {desc}".strip())

print(f"Nombre de textes : {len(texts)}")

# 2. Tokeniser (sans padding, sans troncation → longueurs réelles)
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
lengths = [len(enc) for enc in tokenizer(texts, add_special_tokens=True)["input_ids"]]
lengths = np.array(lengths)

# 3. Stats
print(f"\n{'='*50}")
print(f"Distribution des longueurs en tokens CamemBERT")
print(f"{'='*50}")
print(f"  min    : {lengths.min()}")
print(f"  Q25    : {np.percentile(lengths, 25):.0f}")
print(f"  median : {np.percentile(lengths, 50):.0f}")
print(f"  Q75    : {np.percentile(lengths, 75):.0f}")
print(f"  P90    : {np.percentile(lengths, 90):.0f}")
print(f"  P95    : {np.percentile(lengths, 95):.0f}")
print(f"  P99    : {np.percentile(lengths, 99):.0f}")
print(f"  max    : {lengths.max()}")
print(f"  mean   : {lengths.mean():.1f}")
print(f"  std    : {lengths.std():.1f}")
print(f"\n  % tronqué à max_len=128 : {(lengths > 128).mean()*100:.1f}%")
print(f"  % tronqué à max_len=96  : {(lengths > 96).mean()*100:.1f}%")
print(f"  % tronqué à max_len=160 : {(lengths > 160).mean()*100:.1f}%")