from __future__ import annotations

from pathlib import Path

import cv2

from backend.app.adapters.vlm_teacher_mock import VLMTeacherMock
from backend.app.database import Database
from backend.app.models.schemas import SAM2Mask, VLMTeacherAnalysis
from backend.app.services.rag_service import RAGService


class VLMTeacherService:
    def __init__(self, database: Database, rag_service: RAGService, adapter=None):
        self.database = database
        self.rag_service = rag_service
        self.adapter = adapter or VLMTeacherMock()

    @property
    def provider(self) -> str:
        return self.adapter.provider

    def analyze(
        self, image: dict, prediction: dict, sam2_mask: SAM2Mask | None = None
    ) -> tuple[VLMTeacherAnalysis, str]:
        pixels = cv2.imread(image["path"])
        if pixels is None:
            raise ValueError(f"Image illisible: {image['path']}")
        x1, y1, x2, y2 = [int(value) for value in prediction["bbox"]]
        crop = pixels[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        neighbors = [
            Path(item["path"]) for item in self.database.neighboring_images(image["id"], radius=2)
        ]
        rules_text = self.rag_service.relevant_text(prediction)
        analysis = self.adapter.analyze(
            Path(image["path"]), crop, neighbors, prediction, sam2_mask, rules_text
        )
        return analysis, rules_text
