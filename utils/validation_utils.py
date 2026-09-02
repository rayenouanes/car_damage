"""Stockage des validations humaines par bounding box (TP / FP / UNSURE).

Chaque validation est une ligne JSONL (une ligne = une bounding box, jamais
une décision globale par frame). Ces validations constituent la vérité
terrain de l'Active Learning : elles servent aux statistiques TP/FP, au
hard negative mining et au prochain entraînement.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Sequence

VALIDATION_JSONL_FILENAME = "validation_data.jsonl"
LEGACY_VALIDATION_JSONL_FILENAMES = ["reflection_validation_data.jsonl"]
VALIDATION_CSV_FILENAME = "validation_dataset.csv"

CSV_COLUMNS = [
    "image_id",
    "detection_id",
    "class_name",
    "yolo_confidence",
    "bbox_width",
    "bbox_height",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "target",
]

VALID_LABELS = {"TP", "FP", "UNSURE"}
LEGACY_LABEL_MAP = {"IGNORE": "UNSURE"}


def normalize_label(label: str) -> str:
    value = str(label).upper()
    return LEGACY_LABEL_MAP.get(value, value)


def bbox_features(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> dict:
    """Features géométriques d'une box (utiles pour filtres et futurs modèles)."""
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
    detection_id: str,
    class_name: str,
    yolo_confidence: float,
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
    validation: str,
) -> dict:
    """Ajoute une validation de bounding box au fichier JSONL."""
    label = normalize_label(validation)
    if label not in VALID_LABELS:
        raise ValueError(f"Validation invalide : {validation!r} (TP, FP ou UNSURE).")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_id": image_id,
        "detection_id": str(detection_id),
        "class_name": class_name,
        "yolo_confidence": float(yolo_confidence),
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


def _load_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_validations(jsonl_path: str) -> list:
    """Charge toutes les validations (nouveau fichier + anciens fichiers legacy)."""
    records = []
    directory = os.path.dirname(jsonl_path)
    for legacy_name in LEGACY_VALIDATION_JSONL_FILENAMES:
        legacy_path = os.path.join(directory, legacy_name) if directory else legacy_name
        records.extend(_load_jsonl(legacy_path))
    records.extend(_load_jsonl(jsonl_path))
    for record in records:
        record["validation"] = normalize_label(record.get("validation", ""))
    return records


def export_validation_dataset_csv(jsonl_path: str, csv_path: str) -> int:
    """Exporte les validations TP/FP au format CSV.

    Target : 0 = True Positive, 1 = False Positive. Les lignes UNSURE sont
    exclues. Retourne le nombre de lignes exportées.
    """
    records = load_validations(jsonl_path)
    rows = []
    for record in records:
        validation = record.get("validation")
        if validation not in {"TP", "FP"}:
            continue
        rows.append({
            "image_id": record.get("image_id", ""),
            "detection_id": record.get("detection_id", ""),
            "class_name": record.get("class_name", ""),
            "yolo_confidence": record.get("yolo_confidence", 0.0),
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


def validation_summary(records: list) -> dict:
    """Compte TP / FP / UNSURE et le nombre d'images distinctes validées."""
    counts = {"TP": 0, "FP": 0, "UNSURE": 0}
    images = set()
    for record in records:
        label = record.get("validation")
        if label in counts:
            counts[label] += 1
        if record.get("image_id"):
            images.add(record["image_id"])
    counts["images"] = len(images)
    return counts
