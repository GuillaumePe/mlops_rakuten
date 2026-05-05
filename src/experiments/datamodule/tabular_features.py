"""
Extraction des features tabulaires manuelles utilisées par M2 (et potentiellement
M3/M4 en concat avec les embeddings).

Ces features sont des statistiques simples sur le texte brut et l'image brute,
calculées une fois lors de l'extraction et stockées dans le cache parquet.
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
from PIL import Image


_PUNCT_CHARS = set(".,;:!?-")


def extract_text_tabular(designation: str, description: str | None) -> dict:
    """
    Calcule 7 features tabulaires sur le texte brut.
    
    Args:
        designation: titre du produit (toujours présent)
        description: description longue (peut être None ou vide)
    
    Returns:
        Dict des 7 features.
    """
    designation = designation or ""
    description = description or ""
    full_text = f"{designation}. {description}" if description else designation

    text_length = len(full_text)
    word_count = len(full_text.split())
    designation_length = len(designation)
    description_present = int(bool(description.strip()))

    # Ratios — protéger contre division par 0 sur texte vide
    if text_length > 0:
        uppercase_count = sum(1 for c in full_text if c.isupper())
        digit_count = sum(1 for c in full_text if c.isdigit())
        punct_count = sum(1 for c in full_text if c in _PUNCT_CHARS)
        uppercase_ratio = uppercase_count / text_length
        digit_ratio = digit_count / text_length
        punct_density = punct_count / text_length
    else:
        uppercase_ratio = 0.0
        digit_ratio = 0.0
        punct_density = 0.0

    return {
        "tab_text_length": text_length,
        "tab_word_count": word_count,
        "tab_designation_length": designation_length,
        "tab_description_present": description_present,
        "tab_uppercase_ratio": uppercase_ratio,
        "tab_digit_ratio": digit_ratio,
        "tab_punct_density": punct_density,
    }


def extract_image_tabular(image_path: Path) -> dict:
    """
    Calcule 5 features tabulaires sur une image.
    
    Charge l'image avec PIL, calcule width/height directement, puis convertit
    en grayscale numpy pour les stats de brightness.
    
    Args:
        image_path: chemin vers l'image (Rakuten train ou test)
    
    Returns:
        Dict des 5 features. Si l'image est introuvable ou corrompue, 
        retourne des valeurs par défaut (zéros) plutôt que de planter — c'est
        défensif pour ne pas casser tout le batch d'extraction sur 1 image moisie.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        # Conversion grayscale + numpy. On float32 pour éviter les overflows
        # sur la moyenne (uint8 saturerait pour des images très claires).
        gray_np = np.asarray(img.convert("L"), dtype=np.float32)
        brightness_mean = float(gray_np.mean())
        brightness_std = float(gray_np.std())
    except (FileNotFoundError, OSError) as e:
        # Image manquante ou corrompue. On loggue et on retourne des défauts.
        # Un gros volume de tels cas signalerait un bug d'ingestion à investiguer.
        print(f"[WARN] Image illisible {image_path}: {e}")
        return {
            "tab_image_width": 0,
            "tab_image_height": 0,
            "tab_image_aspect_ratio": 1.0,  # carré par défaut, neutre
            "tab_image_brightness_mean": 0.0,
            "tab_image_brightness_std": 0.0,
        }

    aspect_ratio = width / height if height > 0 else 1.0

    return {
        "tab_image_width": width,
        "tab_image_height": height,
        "tab_image_aspect_ratio": aspect_ratio,
        "tab_image_brightness_mean": brightness_mean,
        "tab_image_brightness_std": brightness_std,
    }


# Liste des colonnes tabulaires, exportée pour usage dans le DataModule et les Pipelines
TEXT_TABULAR_COLS = [
    "tab_text_length",
    "tab_word_count",
    "tab_designation_length",
    "tab_description_present",
    "tab_uppercase_ratio",
    "tab_digit_ratio",
    "tab_punct_density",
]

IMAGE_TABULAR_COLS = [
    "tab_image_width",
    "tab_image_height",
    "tab_image_aspect_ratio",
    "tab_image_brightness_mean",
    "tab_image_brightness_std",
]

ALL_TABULAR_COLS = TEXT_TABULAR_COLS + IMAGE_TABULAR_COLS