from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from backend.app.config import CANONICAL_CLASSES


class JobStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready_for_review = "ready_for_review"
    failed = "failed"


class DetectionStatus(str, Enum):
    pending_review = "pending_review"
    accepted = "accepted"
    rejected = "rejected"
    corrected = "corrected"


class PipelineMode(str, Enum):
    training = "training"
    audit = "audit"
    production = "production"


class FrameSelectionStatus(str, Enum):
    pending = "pending"
    rejected_blur = "rejected_blur"
    rejected_duplicate = "rejected_duplicate"
    keyframe = "keyframe"
    not_selected = "not_selected"


class HumanAction(str, Enum):
    accept = "accept"
    reject = "reject"
    change_class = "change_class"
    reflection = "reflection"
    dirt = "dirt"
    shadow = "shadow"
    error_bank = "error_bank"
    bbox_later = "bbox_later"


class BBoxModel(BaseModel):
    values: list[float]

    @field_validator("values")
    @classmethod
    def validate_bbox(cls, values: list[float]) -> list[float]:
        if len(values) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        x1, y1, x2, y2 = values
        if min(values) < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox coordinates are invalid")
        return [float(value) for value in values]


class Detection(BaseModel):
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float]
    status: DetectionStatus = DetectionStatus.pending_review
    raw_class_name: str | None = None
    active_learning_reasons: list[str] = Field(default_factory=list)
    track_id: str | None = None
    track_score: float | None = Field(default=None, ge=0, le=1)
    track_length: int | None = Field(default=None, ge=1)
    is_track_keyframe: bool = False

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        return BBoxModel(values=value).values


class ImagePredictions(BaseModel):
    image_id: str
    detections: list[Detection]


class SAM2Mask(BaseModel):
    polygon: list[list[float]]
    confidence: float = Field(ge=0, le=1)
    provider: str
    mask_path: str | None = None
    is_bbox_fallback: bool = False

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, polygon: list[list[float]]) -> list[list[float]]:
        if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
            raise ValueError("mask polygon requires at least three [x, y] points")
        return [[float(x), float(y)] for x, y in polygon]


class VLMTeacherAnalysis(BaseModel):
    defaut_reel: bool
    classe_probable: str | None
    bbox_correcte: bool
    niveau_doute: str
    type_erreur_possible: str | None
    a_envoyer_humain: bool
    reason_short: str


class LLMCriticDecision(BaseModel):
    decision: str
    classe_finale: str | None
    correction_bbox: str
    type_erreur: str | None
    ajouter_rag: bool
    ajouter_finetuning: bool


class CorrectionRequest(BaseModel):
    action: HumanAction
    classe_finale: str | None = None
    bbox_finale: list[float] | None = None
    masque_final: list[list[float]] | None = None
    masque_valide: bool | None = None
    type_erreur: str | None = None
    note: str | None = Field(default=None, max_length=1500)
    ajouter_finetuning: bool = True

    @model_validator(mode="after")
    def validate_correction(self) -> "CorrectionRequest":
        if self.action == HumanAction.change_class and self.classe_finale not in CANONICAL_CLASSES:
            raise ValueError("change_class requires a canonical classe_finale")
        if self.bbox_finale is not None:
            self.bbox_finale = BBoxModel(values=self.bbox_finale).values
        if self.masque_final is not None:
            self.masque_final = SAM2Mask(
                polygon=self.masque_final, confidence=1.0, provider="human"
            ).polygon
        return self


class ErrorBankRecord(BaseModel):
    image_path: str
    prediction_yolo: dict[str, Any]
    prediction_vlm: dict[str, Any] | None
    prediction_sam2: dict[str, Any] | None = None
    decision_llm: dict[str, Any] | None
    correction_humaine: dict[str, Any]
    type_erreur: str | None
    classe_finale: str | None
    bbox_finale: list[float] | None
    masque_final: list[list[float]] | None = None
    ajouter_finetuning: bool


class RAGRuleCreate(BaseModel):
    title: str = Field(min_length=3, max_length=180)
    text: str = Field(min_length=5, max_length=3000)
    tags: list[str] = Field(default_factory=list)


class RAGSearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)


class ExportSplitRequest(BaseModel):
    train: float = Field(default=0.70, ge=0, le=1)
    val: float = Field(default=0.20, ge=0, le=1)
    test: float = Field(default=0.10, ge=0, le=1)
    annotation_format: Literal["segmentation", "detection"] = "segmentation"

    @model_validator(mode="after")
    def total_is_one(self) -> "ExportSplitRequest":
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("train + val + test must equal 1.0")
        return self


class InferenceRequest(BaseModel):
    force: bool = False
    mode: PipelineMode = PipelineMode.training
    enable_sam2: bool = True


class AuditRequest(BaseModel):
    confidence_threshold: float = Field(default=0.70, ge=0, le=1)


class YoloTrainingRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=1000)
    image_size: int = Field(default=640, ge=128, le=2048)
    batch: int = Field(default=8, ge=1, le=256)
    device: str = "cpu"


class UploadSummary(BaseModel):
    job_id: str
    images_created: int
    warnings: list[str] = Field(default_factory=list)
