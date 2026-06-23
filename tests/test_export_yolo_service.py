from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.models.schemas import CorrectionRequest, ExportSplitRequest
from backend.app.services.correction_service import CorrectionService
from backend.app.services.export_yolo_service import ExportYoloService


class ExportYoloServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.settings = Settings(data_dir=self.root, yolo_model_path=self.root / "missing.pt")
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_export_has_yolo_tree_and_keeps_video_frames_together(self):
        job_id = self.database.create_job("video.mp4 + images", "mixed")
        correction_service = CorrectionService(self.database, self.settings)
        groups = ["video:A", "video:A", "video:A", "image:B", "image:C", "image:D"]
        for index, source_group in enumerate(groups):
            path = self.root / f"frame_{index}.jpg"
            Image.new("RGB", (100, 80), (index * 20, 20, 20)).save(path)
            image_id = self.database.add_image(
                job_id, path, path.name, source_group, 100, 80, frame_index=index
            )
            prediction_id = self.database.add_prediction(
                image_id,
                {
                    "class_name": "bosse", "raw_class_name": "dent", "confidence": 0.8,
                    "bbox": [10, 10, 70, 60], "status": "pending_review",
                    "active_learning_reasons": [],
                },
            )
            correction_service.correct(prediction_id, CorrectionRequest(action="accept"))

        summary = ExportYoloService(self.database, self.settings).export(
            job_id, ExportSplitRequest(train=0.7, val=0.2, test=0.1)
        )
        export_dir = self.settings.path("exports") / summary["export_id"]
        for kind in ("images", "labels"):
            for split in ("train", "val", "test"):
                self.assertTrue((export_dir / kind / split).is_dir())
        self.assertTrue((export_dir / "data.yaml").is_file())
        self.assertFalse(summary["group_leakage"])
        video_splits = {
            item["split"] for item in summary["manifest"] if item["source_group"] == "video:A"
        }
        self.assertEqual(len(video_splits), 1)
        label_files = list((export_dir / "labels").rglob("*.txt"))
        self.assertTrue(label_files)
        first_line = label_files[0].read_text(encoding="utf-8").strip().split()
        self.assertGreaterEqual(len(first_line), 7)


if __name__ == "__main__":
    unittest.main()
