from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from backend.app.adapters.yolo_ultralytics_adapter import YoloUltralyticsAdapter
from backend.app.config import CANONICAL_CLASSES, Settings
from backend.app.database import Database


class MockYoloAdapter:
    provider = "yolo_mock"

    def predict(self, image_path: Path) -> list[dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image illisible: {image_path}")
        height, width = image.shape[:2]
        seed = int(hashlib.sha256(image_path.name.encode()).hexdigest()[:8], 16)
        class_name = CANONICAL_CLASSES[seed % len(CANONICAL_CLASSES)]
        confidence = 0.35 + (seed % 51) / 100.0
        box_width = max(24, int(width * (0.18 + (seed % 18) / 100)))
        box_height = max(20, int(height * (0.14 + (seed % 16) / 100)))
        x1 = float((seed // 11) % max(1, width - box_width))
        y1 = float((seed // 23) % max(1, height - box_height))
        return [
            {
                "class_name": class_name,
                "raw_class_name": f"mock_{class_name}",
                "confidence": min(confidence, 0.95),
                "bbox": [x1, y1, x1 + box_width, y1 + box_height],
                "status": "pending_review",
            }
        ]


class YoloService:
    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings
        real_adapter = YoloUltralyticsAdapter(
            settings.yolo_model_path, settings.yolo_confidence_floor
        )
        self.adapter = real_adapter if real_adapter.available else MockYoloAdapter()
        self.provider = "ultralytics" if real_adapter.available else "yolo_mock"

    def predict(self, image: dict) -> list[dict]:
        image_path = Path(image["path"])
        detections = self.adapter.predict(image_path)
        pixels = cv2.imread(str(image_path))
        if pixels is None:
            raise ValueError(f"Image illisible: {image_path}")
        for detection in detections:
            detection["active_learning_reasons"] = self._priority_reasons(
                pixels, detection, image
            )
        return detections

    def _priority_reasons(self, image_bgr: np.ndarray, detection: dict, image: dict) -> list[str]:
        reasons: list[str] = []
        confidence = float(detection["confidence"])
        if confidence < self.settings.active_learning_threshold:
            reasons.append("confidence_inferieure_0_70")
        if 0.40 <= confidence <= 0.65 and detection["class_name"] in {
            "impact", "defaut_peinture", "rayure"
        }:
            reasons.append("classes_confondues")
        x1, y1, x2, y2 = detection["bbox"]
        area_ratio = ((x2 - x1) * (y2 - y1)) / max(1, image["width"] * image["height"])
        if area_ratio > 0.35:
            reasons.append("bbox_trop_grande")
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(image["width"], int(x2)), min(image["height"], int(y2))
        crop = image_bgr[iy1:iy2, ix1:ix2]
        if crop.size and (float(crop.mean()) > 175 or float((crop > 225).mean()) > 0.18):
            reasons.append("proche_zone_brillante")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bright_mask = (gray > 225).astype(np.uint8) * 255
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        elongated = 0
        for contour in contours:
            _, _, width, height = cv2.boundingRect(contour)
            if max(width, height) > 20 and max(width, height) / max(1, min(width, height)) > 3:
                elongated += 1
        if elongated >= 2:
            reasons.append("plusieurs_reflets")
        if self.database.similar_error_count(detection["class_name"]) >= 2:
            reasons.append("faux_positifs_frequents_error_bank")
        return reasons

