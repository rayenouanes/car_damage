from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AL_DATA_DIR", APP_DIR / "data")).resolve()
    )
    yolo_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("YOLO_MODEL_PATH", PROJECT_ROOT / "best.pt")
        ).resolve()
    )
    yolo_confidence_floor: float = float(os.getenv("YOLO_CONFIDENCE_FLOOR", "0.10"))
    active_learning_threshold: float = float(os.getenv("ACTIVE_LEARNING_THRESHOLD", "0.70"))
    max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "500"))
    random_seed: int = int(os.getenv("DATASET_SPLIT_SEED", "42"))
    use_chroma: bool = os.getenv("USE_CHROMA", "false").lower() in {"1", "true", "yes"}
    sam2_enabled: bool = os.getenv("SAM2_ENABLED", "true").lower() in {"1", "true", "yes"}
    sam2_provider: str = os.getenv("SAM2_PROVIDER", "mock").lower()
    sam2_checkpoint: Path | None = field(
        default_factory=lambda: Path(os.environ["SAM2_CHECKPOINT"]).resolve()
        if os.getenv("SAM2_CHECKPOINT") else None
    )
    sam2_model_config: str = os.getenv("SAM2_MODEL_CONFIG", "sam2_hiera_l.yaml")
    blur_threshold: float = float(os.getenv("FRAME_BLUR_THRESHOLD", "45.0"))
    duplicate_hamming_threshold: int = int(os.getenv("FRAME_DUPLICATE_HAMMING", "6"))
    angle_hamming_threshold: int = int(os.getenv("FRAME_ANGLE_HAMMING", "18"))
    reflection_ratio_threshold: float = float(os.getenv("FRAME_REFLECTION_RATIO", "0.025"))
    tracking_max_gap: int = int(os.getenv("TRACKING_MAX_GAP", "2"))
    tracking_iou_threshold: float = float(os.getenv("TRACKING_IOU_THRESHOLD", "0.12"))
    tracking_center_distance_threshold: float = float(
        os.getenv("TRACKING_CENTER_DISTANCE", "0.22")
    )
    yolo_training_enabled: bool = os.getenv("YOLO_TRAINING_ENABLED", "false").lower() in {
        "1", "true", "yes"
    }

    @property
    def database_path(self) -> Path:
        return self.data_dir / "active_learning.sqlite3"

    def path(self, name: str) -> Path:
        return self.data_dir / name

    def ensure_directories(self) -> None:
        for name in (
            "uploads", "frames", "keyframes", "crops", "masks", "predictions",
            "exports", "rag", "error_bank", "models", "training_runs"
        ):
            self.path(name).mkdir(parents=True, exist_ok=True)


CANONICAL_CLASSES = ("rayure", "bosse", "impact", "defaut_peinture")
CLASS_TO_ID = {name: index for index, name in enumerate(CANONICAL_CLASSES)}
