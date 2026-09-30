"""Comparaison des methodes d'analyse (Original / CLAHE / SpecSeg).

Les trois methodes partagent exactement la meme image originale :

- Original : RGB original -> YOLO ;
- CLAHE    : RGB original -> CLAHE -> YOLO ;
- SpecSeg  : RGB original -> YOLO (detections identiques a Original)
             + masque SpecSeg analyse par bounding box.

Ce module fournit l'association des boxes entre methodes (IoU + classe),
la construction de la table de comparaison et l'export des resultats.
"""

from __future__ import annotations

import csv
import json
import os
import time
from typing import Optional, Sequence

METHOD_RESULTS_JSONL_FILENAME = "method_comparison_results.jsonl"
METHOD_RESULTS_CSV_FILENAME = "method_comparison_results.csv"

RESULTS_CSV_COLUMNS = [
    "image_id",
    "method",
    "class_name",
    "bbox",
    "yolo_confidence",
    "reflection_overlap",
    "reflection_mean",
    "reflection_max",
    "reflection_risk",
]

# Association box-a-box entre methodes : meme classe ET IoU minimal.
BOX_MATCH_IOU_THRESHOLD = 0.5


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    boxes_a: Sequence[dict],
    boxes_b: Sequence[dict],
    iou_threshold: float = BOX_MATCH_IOU_THRESHOLD,
) -> list:
    """Associe les boxes de deux methodes par classe + IoU (glouton).

    Ne suppose jamais que les index correspondent. Retourne une liste de
    tuples ``(index_a, index_b, iou)`` ; les boxes sans correspondance
    n'apparaissent pas.
    """
    candidates = []
    for ia, item_a in enumerate(boxes_a):
        for ib, item_b in enumerate(boxes_b):
            if item_a.get("class_id") != item_b.get("class_id"):
                continue
            iou = box_iou(item_a["box"], item_b["box"])
            if iou > iou_threshold:
                candidates.append((iou, ia, ib))
    candidates.sort(reverse=True)
    matched, used_a, used_b = [], set(), set()
    for iou, ia, ib in candidates:
        if ia in used_a or ib in used_b:
            continue
        used_a.add(ia)
        used_b.add(ib)
        matched.append((ia, ib, iou))
    return matched


def build_comparison_rows(
    method_boxes: dict,
    class_name_getter,
) -> list:
    """Table de comparaison : une ligne par (methode, detection).

    ``method_boxes`` : {"Original": [boxes], "CLAHE": [boxes], "SpecSeg": [boxes]}.
    Chaque box est un dict avec class_id, box, conf et, pour SpecSeg,
    reflection_overlap / reflection_risk.
    """
    rows = []
    for method, boxes in method_boxes.items():
        for item in boxes:
            overlap = item.get("reflection_overlap")
            risk = item.get("reflection_risk")
            rows.append({
                "Method": method,
                "Class": class_name_getter(item.get("class_id")),
                "YOLO confidence": round(float(item.get("conf", 0.0)), 3),
                "Reflection overlap": (
                    f"{overlap:.0%}" if overlap is not None else "-"
                ),
                "Risk": risk if risk is not None else "-",
            })
    return rows


def save_method_results(
    jsonl_path: str,
    *,
    image_id: str,
    method: str,
    boxes: Sequence[dict],
    class_name_getter,
) -> int:
    """Ajoute une ligne JSONL par detection pour une methode donnee.

    Pour Original et CLAHE, les champs reflection_* restent null sauf si
    SpecSeg a ete volontairement execute sur ces resultats.
    """
    directory = os.path.dirname(jsonl_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    count = 0
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        for item in boxes:
            record = {
                "timestamp": timestamp,
                "image_id": image_id,
                "method": method,
                "class_name": class_name_getter(item.get("class_id")),
                "bbox": [float(v) for v in item["box"]],
                "yolo_confidence": float(item.get("conf", 0.0)),
                "reflection_overlap": item.get("reflection_overlap"),
                "reflection_mean": item.get("reflection_mean"),
                "reflection_max": item.get("reflection_max"),
                "reflection_risk": item.get("reflection_risk"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_method_results_csv(jsonl_path: str, csv_path: str) -> int:
    """Exporte toutes les lignes JSONL vers un CSV de comparaison."""
    rows = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append({
                    "image_id": record.get("image_id", ""),
                    "method": record.get("method", ""),
                    "class_name": record.get("class_name", ""),
                    "bbox": json.dumps(record.get("bbox", [])),
                    "yolo_confidence": record.get("yolo_confidence", 0.0),
                    "reflection_overlap": record.get("reflection_overlap"),
                    "reflection_mean": record.get("reflection_mean"),
                    "reflection_max": record.get("reflection_max"),
                    "reflection_risk": record.get("reflection_risk"),
                })
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
