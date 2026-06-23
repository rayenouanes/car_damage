from __future__ import annotations

from backend.app.adapters.llm_critic_mock import LLMCriticMock
from backend.app.database import Database
from backend.app.models.schemas import LLMCriticDecision, SAM2Mask, VLMTeacherAnalysis


class LLMCriticService:
    def __init__(self, database: Database, adapter=None):
        self.database = database
        self.adapter = adapter or LLMCriticMock()

    @property
    def provider(self) -> str:
        return self.adapter.provider

    def critique(
        self, prediction: dict, sam2_mask: SAM2Mask | None,
        vlm_analysis: VLMTeacherAnalysis, rules_text: str
    ) -> LLMCriticDecision:
        similar = [
            item for item in self.database.list_error_bank(100)
            if item.get("classe_finale") == prediction["class_name"]
            or item.get("record", {}).get("prediction_yolo", {}).get("class_name")
            == prediction["class_name"]
        ][:5]
        return self.adapter.critique(prediction, sam2_mask, vlm_analysis, rules_text, similar)
