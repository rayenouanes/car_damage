from __future__ import annotations

import io
import shutil
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app


class APISmokeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        settings = Settings(
            data_dir=self.root, yolo_model_path=self.root / "missing.pt", blur_threshold=0.0
        )
        self.client = TestClient(create_app(settings))

    def tearDown(self):
        self.client.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_upload_mock_inference_correction_and_export(self):
        image_buffer = io.BytesIO()
        Image.new("RGB", (160, 100), (45, 45, 45)).save(image_buffer, format="JPEG")
        image_bytes = image_buffer.getvalue()
        production = self.client.post(
            "/api/production/infer",
            files={"file": ("car.jpg", image_bytes, "image/jpeg")},
        )
        self.assertEqual(production.status_code, 200, production.text)
        self.assertEqual(set(production.json()[0]), {"class_name", "bbox", "score"})
        audit = self.client.post(
            "/api/audit/infer",
            files={"file": ("car.jpg", image_bytes, "image/jpeg")},
            data={"confidence_threshold": "1.0"},
        )
        self.assertEqual(audit.status_code, 200, audit.text)
        self.assertTrue(audit.json()["detections"][0]["audited"])
        self.assertIsNotNone(audit.json()["detections"][0]["vlm"])
        self.assertIsNotNone(audit.json()["detections"][0]["llm"])
        upload = self.client.post(
            "/api/jobs/upload",
            files=[("files", ("car.jpg", image_bytes, "image/jpeg"))],
            data={"every_n_frames": "30"},
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        job_id = upload.json()["job_id"]

        inference = self.client.post(
            f"/api/jobs/{job_id}/infer", json={"force": False, "mode": "training"}
        )
        self.assertEqual(inference.status_code, 200, inference.text)
        prediction = inference.json()["images"][0]["predictions"][0]
        self.assertIn("active_learning_reasons", prediction)
        self.assertIsNotNone(prediction["track_id"])
        self.assertTrue(prediction["is_track_keyframe"])
        self.assertIsNotNone(prediction["sam2"])
        self.assertIsNotNone(prediction["vlm"])
        self.assertIsNotNone(prediction["llm"])

        correction = self.client.post(
            f"/api/predictions/{prediction['id']}/corrections", json={"action": "accept"}
        )
        self.assertEqual(correction.status_code, 200, correction.text)
        self.assertIn("prediction_yolo", correction.json()["error_bank_record"])

        export = self.client.post(
            f"/api/jobs/{job_id}/export-yolo",
            json={"train": 0.7, "val": 0.2, "test": 0.1},
        )
        self.assertEqual(export.status_code, 200, export.text)
        self.assertFalse(export.json()["group_leakage"])
        self.assertEqual(export.json()["images"], 1)
        self.assertEqual(export.json()["annotation_format"], "segmentation")


if __name__ == "__main__":
    unittest.main()
