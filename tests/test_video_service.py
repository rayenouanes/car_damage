from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.services.video_service import VideoService


class VideoServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.settings = Settings(data_dir=self.root, yolo_model_path=self.root / "missing.pt")
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_extracts_every_two_frames_into_job_directory(self):
        source = self.root / "source.avi"
        writer = cv2.VideoWriter(
            str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
        )
        self.assertTrue(writer.isOpened())
        for index in range(6):
            frame = np.full((64, 96, 3), index * 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        job_id = self.database.create_job("source.avi", "video")
        count = VideoService(self.database, self.settings).ingest(
            job_id, "source.avi", source.read_bytes(), 0, every_n_frames=2
        )
        job = self.database.get_job(job_id)
        self.assertEqual(count, 3)
        self.assertEqual([item["frame_index"] for item in job["images"]], [0, 2, 4])
        self.assertTrue(all(Path(item["path"]).parent.name == job_id for item in job["images"]))


if __name__ == "__main__":
    unittest.main()

