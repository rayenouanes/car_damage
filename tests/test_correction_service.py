from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from PIL import Image

from backend.app.adapters.vector_store_chroma import ChromaVectorStore
from backend.app.config import Settings
from backend.app.database import Database
from backend.app.models.schemas import CorrectionRequest
from backend.app.services.correction_service import CorrectionService
from backend.app.services.rag_service import RAGService


class CorrectionServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.settings = Settings(data_dir=self.root, yolo_model_path=self.root / "missing.pt")
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()
        self.rag_service = RAGService(
            ChromaVectorStore(self.settings.path("rag"), prefer_chroma=False)
        )
        image_path = self.root / "car.jpg"
        Image.new("RGB", (120, 80), (30, 30, 30)).save(image_path)
        job_id = self.database.create_job("car.jpg", "image")
        image_id = self.database.add_image(
            job_id, image_path, "car.jpg", "image:one", 120, 80
        )
        self.prediction_id = self.database.add_prediction(
            image_id,
            {
                "class_name": "rayure", "raw_class_name": "scratch",
                "confidence": 0.51, "bbox": [10, 10, 70, 40],
                "status": "pending_review", "active_learning_reasons": ["proche_zone_brillante"],
            },
        )
        self.database.save_analysis(
            "vlm_analyses", self.prediction_id, "mock",
            {"defaut_reel": False, "type_erreur_possible": "faux_positif_reflet"},
        )
        self.database.save_analysis(
            "llm_decisions", self.prediction_id, "mock",
            {"decision": "rejeter", "ajouter_rag": True},
        )

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_correction_creates_complete_error_bank_record(self):
        result = CorrectionService(self.database, self.settings, self.rag_service).correct(
            self.prediction_id, CorrectionRequest(action="reflection")
        )
        record = result["error_bank_record"]
        self.assertEqual(
            set(record),
            {
                "image_path", "prediction_yolo", "prediction_vlm", "prediction_sam2",
                "decision_llm", "masque_final",
                "correction_humaine", "type_erreur", "classe_finale", "bbox_finale",
                "ajouter_finetuning",
            },
        )
        self.assertEqual(record["type_erreur"], "faux_positif_reflet")
        self.assertIsNone(record["classe_finale"])
        self.assertEqual(len(self.database.list_error_bank()), 1)
        self.assertIsNotNone(result["rag_rule_id"])
        self.assertEqual(len(self.rag_service.list_rules()), 1)
        self.assertTrue((self.settings.path("error_bank") / f"{result['correction_id']}.json").exists())


if __name__ == "__main__":
    unittest.main()
