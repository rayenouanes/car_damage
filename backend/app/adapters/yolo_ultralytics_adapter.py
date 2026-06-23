from __future__ import annotations

from pathlib import Path


def canonicalize_class(raw_name: str) -> str | None:
    name = " ".join(raw_name.lower().replace("_", " ").replace("-", " ").split())
    if any(word in name for word in ("glass", "vitre", "lamp", "phare", "tire", "tyre", "pneu")):
        return None
    if "scratch" in name or "rayure" in name:
        return "rayure"
    if "dent" in name or "bosse" in name:
        return "bosse"
    if any(word in name for word in ("impact", "crack", "fissure", "hole", "trou")):
        return "impact"
    if any(word in name for word in ("paint", "peinture", "varnish", "vernis", "clear coat")):
        return "defaut_peinture"
    return None


class YoloUltralyticsAdapter:
    def __init__(self, model_path: Path, confidence_floor: float = 0.10):
        self.model_path = Path(model_path)
        self.confidence_floor = confidence_floor
        self._model = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    def _load(self):
        if not self.available:
            raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, image_path: Path) -> list[dict]:
        result = self._load().predict(
            str(image_path), conf=self.confidence_floor, verbose=False
        )[0]
        if result.boxes is None:
            return []
        names = result.names
        detections: list[dict] = []
        for box in result.boxes:
            raw_id = int(box.cls[0].item())
            raw_name = str(names[raw_id] if isinstance(names, dict) else names[raw_id])
            class_name = canonicalize_class(raw_name)
            if class_name is None:
                continue
            detections.append(
                {
                    "class_name": class_name,
                    "raw_class_name": raw_name,
                    "confidence": float(box.conf[0].item()),
                    "bbox": [float(value) for value in box.xyxy[0].tolist()],
                    "status": "pending_review",
                }
            )
        return detections

