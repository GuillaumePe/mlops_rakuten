"""
Encodage centralisé des labels Rakuten : prdtypecode ↔ index 0-26.

Mapping déterministe : sorted(27 prdtypecodes uniques) → indices consécutifs.
Réplique exactement la logique du DataModule (self._code_to_idx).

Usage :
    from src.data.label_encoding import CLASS_TO_IDX, IDX_TO_CLASS, encode_labels, decode_labels
"""

CLASSES_SORTED = sorted([
    10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302,
    1320, 1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582,
    2583, 2585, 2705, 2905,
])

CLASS_TO_IDX = {code: idx for idx, code in enumerate(CLASSES_SORTED)}
IDX_TO_CLASS = {idx: code for idx, code in enumerate(CLASSES_SORTED)}
N_CLASSES = len(CLASSES_SORTED)


def encode_labels(prdtypecodes) -> list[int]:
    """prdtypecodes → indices 0-26."""
    return [CLASS_TO_IDX[c] for c in prdtypecodes]


def decode_labels(indices) -> list[int]:
    """indices 0-26 → prdtypecodes."""
    return [IDX_TO_CLASS[i] for i in indices]
