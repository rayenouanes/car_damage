from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend.app.models.schemas import SAM2Mask


class SAM2Adapter:
    provider = "sam2"

    def __init__(self, checkpoint: Path | None, model_config: str):
        self.checkpoint = checkpoint
        self.model_config = model_config
        self._predictor = None

    @property
    def available(self) -> bool:
        if self.checkpoint is None or not self.checkpoint.is_file():
            return False
        try:
            import sam2  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):
        if not self.available:
            raise RuntimeError("SAM2 package or checkpoint is unavailable")
        if self._predictor is None:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(self.model_config, str(self.checkpoint), device="cpu")
            self._predictor = SAM2ImagePredictor(model)
        return self._predictor

    def predict(self, image_path: Path, bbox: list[float]) -> SAM2Mask:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"Image illisible: {image_path}")
        predictor = self._load()
        predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            box=np.asarray(bbox, dtype=np.float32), multimask_output=True
        )
        best_index = int(np.argmax(scores))
        binary = masks[best_index].astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("SAM2 returned an empty mask")
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
        return SAM2Mask(
            polygon=polygon, confidence=float(scores[best_index]), provider=self.provider
        )

