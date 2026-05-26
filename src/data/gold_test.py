"""
Gold test set : sous-ensemble fixe de productids exclu de tout entraînement,
sert d'arbitre constant pour comparer les modèles à travers les batches et
les expériences.

Méthode : hash MD5 sur le productid, modulo 10 == 0 → ~10% du dataset.
Déterministe (pas de seed à transmettre), reproductible, résistant à un
re-shuffling accidentel des données.
"""
from __future__ import annotations
import hashlib
from typing import Iterable


GOLD_HASH_MODULO = 10  # 1/10 = ~8500 samples sur 85k


def is_gold(productid: int) -> bool:
    """True si le productid appartient au gold test set."""
    h = hashlib.md5(str(productid).encode("utf-8")).digest()
    # On prend les 8 premiers bytes en int et on module
    bucket = int.from_bytes(h[:8], "big") % GOLD_HASH_MODULO
    return bucket == 0


def get_gold_productids(all_productids: Iterable[int]) -> set[int]:
    """Filtre l'itérable pour ne garder que les productids du gold set."""
    return {pid for pid in all_productids if is_gold(pid)}


def filter_out_gold(productids: Iterable[int]) -> list[int]:
    """Retourne la liste des productids hors gold set (pour train/val)."""
    return [pid for pid in productids if not is_gold(pid)]