"""Stockage des validations par bounding box (TP / FP / IGNORE).

Chaque validation est une ligne JSONL dans ``reflection_validation_data.jsonl``
(une ligne = une bounding box, jamais une decision globale par frame).
``export_validation_dataset_csv`` prepare ensuite le dataset d'entrainement
d'un futur modele de calibration des faux positifs lies aux reflets
(LogisticRegression, petit XGBoost...).
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Optional, Sequence

VALIDATION_JSONL_FILENAME = "reflection_validation_data.jsonl"
VALIDATION_CSV_FILENAME = "reflection_validation_dataset.csv"

CSV_COLUMNS = [
    "image_id",
    "class_name",
    "yolo_confidence",
    "reflection_overlap",
    "reflection_mean",
    "reflection_max",
    "bbox_width",
    "bbox_height",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "target",
]

VALID_LABELS = {"TP", "FP", "IGNORE"}


def bbox_features(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> dict:
    """Features geometriques d'une box pour le futur modele de calibration."""
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    image_area = float(max(1, image_width) * max(1, image_height))
    return {
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area_ratio": (width * height) / image_area,
        "bbox_aspect_ratio": width / height if height > 0 else 0.0,
    }


def save_bbox_validation(
    jsonl_path: str,
    *,
    image_id: str,
    class_name: str,
    yolo_confidence: float,
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    validation: str,
    method: str = "Original",
    reflection_overlap: Optional[float] = None,
    reflection_mean: Optional[float] = None,
    reflection_max: Optional[float] = None,
    reflection_risk: Optional[str] = None,
) -> dict:
    """Ajoute une validation de bounding box au fichier JSONL."""
    label = str(validation).upper()
    if label not in VALID_LABELS:
        raise ValueError(f"Validation invalide : {validation!r} (TP, FP ou IGNORE).")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_id": image_id,
        "method": method,
        "class_name": class_name,
        "yolo_confidence": float(yolo_confidence),
        "reflection_overlap": None if reflection_overlap is None else float(reflection_overlap),
        "reflection_mean": None if reflection_mean is None else float(reflection_mean),
        "reflection_max": None if reflection_max is None else float(reflection_max),
        "reflection_risk": reflection_risk,
        "validation": label,
        "bbox": [float(value) for value in bbox],
        "image_width": int(image_width),
        "image_height": int(image_height),
    }
    record.update(bbox_features(bbox, image_width, image_height))

    directory = os.path.dirname(jsonl_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_validations(jsonl_path: str) -> list:
    """Charge toutes les validations enregistrees (liste de dicts)."""
    if not os.path.exists(jsonl_path):
        return []
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def export_validation_dataset_csv(jsonl_path: str, csv_path: str) -> int:
    """Exporte les validations TP/FP au format CSV d'entrainement.

    Target : 0 = True Positive, 1 = False Positive. Les lignes IGNORE et
    celles sans metriques SpecSeg sont exclues. Retourne le nombre de
    lignes exportees.
    """
    records = load_validations(jsonl_path)
    rows = []
    for record in records:
        validation = record.get("validation")
        if validation not in {"TP", "FP"}:
            continue
        if record.get("reflection_overlap") is None:
            continue
        rows.append({
            "image_id": record.get("image_id", ""),
            "class_name": record.get("class_name", ""),
            "yolo_confidence": record.get("yolo_confidence", 0.0),
            "reflection_overlap": record.get("reflection_overlap", 0.0),
            "reflection_mean": record.get("reflection_mean", 0.0),
            "reflection_max": record.get("reflection_max", 0.0),
            "bbox_width": record.get("bbox_width", 0.0),
            "bbox_height": record.get("bbox_height", 0.0),
            "bbox_area_ratio": record.get("bbox_area_ratio", 0.0),
            "bbox_aspect_ratio": record.get("bbox_aspect_ratio", 0.0),
            "target": 0 if validation == "TP" else 1,
        })

    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
