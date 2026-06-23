from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.client = TestClient(
            create_app(
                Settings(
                    data_dir=self.root,
                    yolo_model_path=self.root / "missing.pt",
                    api_key="secret-api-key",
                )
            )
        )

    def tearDown(self):
        self.client.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_api_key_protects_private_routes_but_not_health(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200, health.text)
        self.assertTrue(health.json()["auth_enabled"])

        rejected = self.client.get("/api/jobs")
        self.assertEqual(rejected.status_code, 401, rejected.text)

        accepted = self.client.get("/api/jobs", headers={"X-API-Key": "secret-api-key"})
        self.assertEqual(accepted.status_code, 200, accepted.text)


if __name__ == "__main__":
    unittest.main()
