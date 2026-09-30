"""Pretraitements d'image pour l'inference (jamais pour le training).

Ces fonctions produisent uniquement des copies temporaires en memoire :
l'image originale (en session ou sur disque) n'est jamais modifiee.
"""

from __future__ import annotations

import numpy as np

# Parametres CLAHE, facilement modifiables.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)


def apply_clahe(
    image_rgb: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
) -> np.ndarray:
    """CLAHE applique uniquement sur la luminance (canal L de LAB).

    RGB -> LAB -> CLAHE sur L -> fusion LAB -> RGB.
    Retourne une nouvelle image RGB uint8 de meme resolution.
    """
    import cv2

    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Image RGB attendue (H, W, 3), recu shape {image_rgb.shape}.")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_channel)
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
