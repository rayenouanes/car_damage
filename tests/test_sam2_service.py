from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from backend.app.config import Settings
from backend.app.services.sam2_service import SAM2Service


class SAM2ServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_mock_produces_polygon_crop_and_mask(self):
        settings = Settings(
            data_dir=self.root,
            yolo_model_path=self.root / "missing.pt",
            sam2_enabled=True,
            sam2_provider="mock",
        )
        settings.ensure_directories()
        image_path = self.root / "car.jpg"
        Image.new("RGB", (160, 100), (60, 60, 60)).save(image_path)
        mask = SAM2Service(settings).propose(
            {"path": str(image_path), "job_id": "job"}, "prediction", [20, 15, 120, 75]
        )
        self.assertIsNotNone(mask)
        self.assertGreaterEqual(len(mask.polygon), 3)
        self.assertTrue(Path(mask.mask_path).is_file())
        self.assertTrue((settings.path("crops") / "job" / "prediction.jpg").is_file())

    def test_disabled_sam2_allows_bbox_only_pipeline(self):
        settings = Settings(
            data_dir=self.root / "disabled",
            yolo_model_path=self.root / "missing.pt",
            sam2_enabled=False,
        )
        settings.ensure_directories()
        image_path = self.root / "disabled.jpg"
        Image.new("RGB", (80, 60), (50, 50, 50)).save(image_path)
        result = SAM2Service(settings).propose(
            {"path": str(image_path), "job_id": "job"}, "prediction", [5, 5, 50, 40]
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

