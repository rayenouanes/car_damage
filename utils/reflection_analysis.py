"""Analyse des reflets speculaires par bounding box.

Les metriques sont calculees box par box (jamais frame par frame) a partir
du masque SpecSeg de la frame complete. Elles n'alterent jamais la confiance
YOLO : ``original_yolo_confidence`` reste intacte, les metriques sont
purement additives.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Seuils EXPERIMENTAUX de l'indicateur reflection_risk.
# Ils ne suppriment jamais une detection automatiquement.
REFLECTION_RISK_THRESHOLDS = {
    "medium": 0.25,  # reflection_overlap >= medium -> MEDIUM
    "high": 0.60,    # reflection_overlap >= high   -> HIGH
}

DEFAULT_MASK_THRESHOLD = 0.5

EMPTY_METRICS = {
    "reflection_overlap": 0.0,
    "reflection_mean": 0.0,
    "reflection_max": 0.0,
}


def compute_reflection_metrics(
    reflection_mask: Optional[np.ndarray],
    bbox: Sequence[float],
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
) -> dict:
    """Metriques de reflet pour une bounding box ``[x1, y1, x2, y2]``.

    Retourne un dictionnaire avec, toutes dans ``[0.0, 1.0]`` :

    - ``reflection_overlap`` : proportion des pixels de la box dont la
      probabilite SpecSeg depasse ``mask_threshold`` ;
    - ``reflection_mean`` : valeur moyenne du masque dans la box ;
    - ``reflection_max`` : valeur maximale du masque dans la box.

    Gere masque absent, coordonnees negatives, box hors image et box vide
    en retournant des metriques nulles.
    """
    if reflection_mask is None:
        return dict(EMPTY_METRICS)

    mask = np.asarray(reflection_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.size == 0:
        return dict(EMPTY_METRICS)

    height, width = mask.shape
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return dict(EMPTY_METRICS)

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = int(np.clip(np.floor(x1), 0, width))
    x2 = int(np.clip(np.ceil(x2), 0, width))
    y1 = int(np.clip(np.floor(y1), 0, height))
    y2 = int(np.clip(np.ceil(y2), 0, height))

    if x2 <= x1 or y2 <= y1:
        return dict(EMPTY_METRICS)

    region = np.clip(mask[y1:y2, x1:x2], 0.0, 1.0)
    if region.size == 0:
        return dict(EMPTY_METRICS)

    overlap = float((region > float(mask_threshold)).mean())
    return {
        "reflection_overlap": float(np.clip(overlap, 0.0, 1.0)),
        "reflection_mean": float(np.clip(region.mean(), 0.0, 1.0)),
        "reflection_max": float(np.clip(region.max(), 0.0, 1.0)),
    }


def compute_reflection_risk(
    reflection_overlap: float,
    thresholds: Optional[dict] = None,
) -> str:
    """Indicateur experimental LOW / MEDIUM / HIGH base sur l'overlap."""
    limits = thresholds or REFLECTION_RISK_THRESHOLDS
    overlap = float(reflection_overlap)
    if overlap >= limits["high"]:
        return "HIGH"
    if overlap >= limits["medium"]:
        return "MEDIUM"
    return "LOW"
