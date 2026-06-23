from __future__ import annotations

import json
from pathlib import Path

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.models.schemas import CorrectionRequest, ErrorBankRecord, HumanAction
from backend.app.services.rag_service import RAGService


ACTION_ERROR_TYPES = {
    HumanAction.reject: "faux_positif",
    HumanAction.change_class: "mauvaise_classe",
    HumanAction.reflection: "faux_positif_reflet",
    HumanAction.dirt: "faux_positif_salete",
    HumanAction.shadow: "faux_positif_ombre",
    HumanAction.bbox_later: "bbox_imprecise",
}


class CorrectionService:
    def __init__(
        self, database: Database, settings: Settings, rag_service: RAGService | None = None
    ):
        self.database = database
        self.settings = settings
        self.rag_service = rag_service

    def correct(self, prediction_id: str, correction: CorrectionRequest) -> dict:
        context = self.database.get_prediction_context(prediction_id)
        if not context:
            raise KeyError(prediction_id)
        action = correction.action
        rejected_actions = {
            HumanAction.reject, HumanAction.reflection, HumanAction.dirt, HumanAction.shadow
        }
        if action in rejected_actions:
            final_class = None
            final_bbox = None
            final_mask = None
        else:
            final_class = correction.classe_finale or context["class_name"]
            final_bbox = correction.bbox_finale or context["bbox"]
            final_mask = correction.masque_final
            if final_mask is None and correction.masque_valide is not False:
                final_mask = (context.get("sam2") or {}).get("polygon")
        error_type = correction.type_erreur or ACTION_ERROR_TYPES.get(action)
        yolo_payload = {
            "class_name": context["class_name"],
            "confidence": context["confidence"],
            "bbox": context["bbox"],
            "status": context["status"],
        }
        record = ErrorBankRecord(
            image_path=context["image_path"],
            prediction_yolo=yolo_payload,
            prediction_vlm=context.get("vlm"),
            prediction_sam2=context.get("sam2"),
            decision_llm=context.get("llm"),
            correction_humaine=correction.model_dump(mode="json"),
            type_erreur=error_type,
            classe_finale=final_class,
            bbox_finale=final_bbox,
            masque_final=final_mask,
            ajouter_finetuning=correction.ajouter_finetuning,
        )
        correction_id = self.database.save_correction_and_error(
            prediction_id, correction, record
        )
        error_dir = self.settings.path("error_bank")
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / f"{correction_id}.json").write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rag_rule_id = None
        llm_requests_rule = bool((context.get("llm") or {}).get("ajouter_rag"))
        recurrent_error = self.database.similar_error_count(context["class_name"]) >= 3
        if self.rag_service and error_type and (llm_requests_rule or recurrent_error):
            rag_rule_id = self.rag_service.add_rule(
                f"Erreur recurrente: {error_type}",
                (
                    f"Pour une prediction {context['class_name']}, verifier le cas {error_type}; "
                    f"la correction humaine retenue est {action.value}."
                ),
                [context["class_name"], error_type, "error_bank"],
            )
        return {
            "correction_id": correction_id,
            "error_bank_record": record.model_dump(mode="json"),
            "rag_rule_id": rag_rule_id,
        }
