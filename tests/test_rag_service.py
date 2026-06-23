from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from backend.app.adapters.vector_store_chroma import ChromaVectorStore
from backend.app.services.rag_service import INITIAL_RULES, RAGService


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path("backend/app/data/test_runs") / uuid.uuid4().hex
        self.root.mkdir(parents=True, exist_ok=False)
        self.service = RAGService(ChromaVectorStore(self.root, prefer_chroma=False))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_seeds_adds_searches_and_formats_rules(self):
        self.service.seed_initial_rules()
        self.assertEqual(len(self.service.list_rules()), len(INITIAL_RULES))
        self.service.add_rule(
            "Vernis mat", "Une perte de vernis cree une zone mate stable.", ["defaut_peinture"]
        )
        results = self.service.search("vernis zone mate", limit=3)
        self.assertEqual(results[0]["title"], "Vernis mat")
        text = self.service.relevant_text(
            {"class_name": "defaut_peinture", "active_learning_reasons": ["classes_confondues"]}
        )
        self.assertIn("- ", text)


if __name__ == "__main__":
    unittest.main()

