from __future__ import annotations

import math
from pathlib import Path

from backend.app.models.schemas import SAM2Mask


class SAM2Mock:
    provider = "sam2_mock"

    def predict(self, image_path: Path, bbox: list[float]) -> SAM2Mask:
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius_x, radius_y = (x2 - x1) * 0.47, (y2 - y1) * 0.44
        polygon = [
            [
                center_x + radius_x * math.cos(2 * math.pi * index / 20),
                center_y + radius_y * math.sin(2 * math.pi * index / 20),
            ]
            for index in range(20)
        ]
        return SAM2Mask(polygon=polygon, confidence=0.72, provider=self.provider)

