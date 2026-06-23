from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.app.adapters.sam2_adapter import SAM2Adapter
from backend.app.adapters.sam2_mock import SAM2Mock
from backend.app.config import Settings
from backend.app.models.schemas import SAM2Mask


class SAM2Service:
    def __init__(self, settings: Settings):
        self.settings = settings
        real = SAM2Adapter(settings.sam2_checkpoint, settings.sam2_model_config)
        if settings.sam2_provider == "sam2" and real.available:
            self.adapter = real
        else:
            self.adapter = SAM2Mock()

    @property
    def provider(self) -> str:
        return "disabled" if not self.settings.sam2_enabled else self.adapter.provider

    def propose(self, image: dict, prediction_id: str, bbox: list[float]) -> SAM2Mask | None:
        image_path = Path(image["path"])
        pixels = cv2.imread(str(image_path))
        if pixels is None:
            raise ValueError(f"Image illisible: {image_path}")
        x1, y1, x2, y2 = [int(value) for value in bbox]
        crop = pixels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        crop_dir = self.settings.path("crops") / image["job_id"]
        crop_dir.mkdir(parents=True, exist_ok=True)
        if crop.size:
            cv2.imwrite(str(crop_dir / f"{prediction_id}.jpg"), crop)
        if not self.settings.sam2_enabled:
            return None
        try:
            mask = self.adapter.predict(image_path, bbox)
        except Exception:
            mask = SAM2Mock().predict(image_path, bbox)
        mask_dir = self.settings.path("masks") / image["job_id"]
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_path = mask_dir / f"{prediction_id}.png"
        binary = np.zeros(pixels.shape[:2], dtype=np.uint8)
        points = np.asarray(mask.polygon, dtype=np.int32)
        cv2.fillPoly(binary, [points], 255)
        cv2.imwrite(str(mask_path), binary)
        mask.mask_path = str(mask_path.resolve())
        return mask

