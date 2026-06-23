from __future__ import annotations

import json

from backend.app.database import Database
from backend.app.services.frame_selection_service import FrameSelectionService
from backend.app.services.llm_critic_service import LLMCriticService
from backend.app.services.sam2_service import SAM2Service
from backend.app.services.vlm_teacher_service import VLMTeacherService
from backend.app.services.yolo_service import YoloService


class TrainingPipelineService:
    def __init__(
        self,
        database: Database,
        frame_selection: FrameSelectionService,
        yolo: YoloService,
        sam2: SAM2Service,
        vlm: VLMTeacherService,
        llm: LLMCriticService,
    ):
        self.database = database
        self.frame_selection = frame_selection
        self.yolo = yolo
        self.sam2 = sam2
        self.vlm = vlm
        self.llm = llm

    def run(self, job_id: str, force: bool = False) -> dict:
        job = self.database.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if any(image["predictions"] for image in job["images"]) and not force:
            return job
        self.database.update_job(job_id, "processing")
        if force:
            self.database.clear_predictions(job_id)
        try:
            prepass, selection_summary = self.frame_selection.select(job_id, self.yolo)
            refreshed = self.database.get_job(job_id) or {}
            for image in refreshed.get("images", []):
                if image["id"] not in prepass:
                    continue
                persisted: list[dict] = []
                for detection in prepass[image["id"]]:
                    prediction_id = self.database.add_prediction(image["id"], detection)
                    item = {"id": prediction_id, **detection}
                    sam2_mask = self.sam2.propose(image, prediction_id, detection["bbox"])
                    if sam2_mask is not None:
                        self.database.save_sam2_mask(
                            prediction_id, sam2_mask.model_dump(mode="json")
                        )
                    vlm_analysis, rules_text = self.vlm.analyze(image, item, sam2_mask)
                    self.database.save_analysis(
                        "vlm_analyses", prediction_id, self.vlm.provider,
                        vlm_analysis.model_dump(mode="json"),
                    )
                    llm_decision = self.llm.critique(
                        item, sam2_mask, vlm_analysis, rules_text
                    )
                    self.database.save_analysis(
                        "llm_decisions", prediction_id, self.llm.provider,
                        llm_decision.model_dump(mode="json"),
                    )
                    persisted.append(
                        {
                            **item,
                            "sam2": sam2_mask.model_dump(mode="json") if sam2_mask else None,
                            "vlm": vlm_analysis.model_dump(mode="json"),
                            "llm": llm_decision.model_dump(mode="json"),
                        }
                    )
                prediction_dir = self.frame_selection.settings.path("predictions") / job_id
                prediction_dir.mkdir(parents=True, exist_ok=True)
                (prediction_dir / f"{image['id']}.json").write_text(
                    json.dumps({"image_id": image["id"], "detections": persisted}, indent=2),
                    encoding="utf-8",
                )
            self.database.update_job(job_id, "ready_for_review")
            result = self.database.get_job(job_id) or {}
            result["frame_selection"] = selection_summary
            return result
        except Exception as exc:
            self.database.update_job(job_id, "failed", str(exc))
            raise

