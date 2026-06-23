from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.services.frame_selection_service import FrameSelectionService


class FakeYolo:
    def predict(self, image: dict) -> list[dict]:
        return [
            {
                "class_name": "rayure", "raw_class_name": "scratch", "confidence": 0.48,
                "bbox": [10, 10, 60, 45], "status": "pending_review",
                "active_learning_reasons": ["confidence_inferieure_0_70"],
            }
        ]


class FrameSelectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.settings = Settings(
            data_dir=self.root,
            yolo_model_path=self.root / "missing.pt",
            blur_threshold=20.0,
            duplicate_hamming_threshold=0,
            angle_hamming_threshold=8,
            reflection_ratio_threshold=0.01,
        )
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_filters_blur_and_duplicates_then_keeps_informative_frames(self):
        job_id = self.database.create_job("phone.avi", "video")
        checker = np.indices((80, 120)).sum(axis=0) % 2 * 255
        checker = np.repeat(checker[:, :, None], 3, axis=2).astype(np.uint8)
        variants = [
            checker,
            checker.copy(),
            np.full((80, 120, 3), 90, np.uint8),
            np.roll(checker, 7, axis=1),
        ]
        for index, pixels in enumerate(variants):
            path = self.root / f"frame_{index}.png"
            cv2.imwrite(str(path), pixels)
            self.database.add_image(
                job_id, path, path.name, "video:phone", 120, 80, frame_index=index
            )
        prepass, summary = FrameSelectionService(self.database, self.settings).select(
            job_id, FakeYolo()
        )
        job = self.database.get_job(job_id)
        statuses = [image["selection_status"] for image in job["images"]]
        self.assertIn("rejected_duplicate", statuses)
        self.assertIn("rejected_blur", statuses)
        self.assertGreaterEqual(summary["keyframes"], 1)
        self.assertEqual(len(prepass), summary["keyframes"])
        self.assertEqual(summary["tracks"], 1)
        selected = next(iter(prepass.values()))[0]
        self.assertTrue(selected["is_track_keyframe"])
        self.assertTrue(selected["track_id"].startswith("track_001_"))


if __name__ == "__main__":
    unittest.main()
