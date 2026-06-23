from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_list(name: str, default: str = "") -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(slots=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "local")
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")
    api_key: str = os.getenv("AL_API_KEY", "")
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: env_list(
            "AL_CORS_ORIGINS",
            "http://127.0.0.1:8501,http://localhost:8501",
        )
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("AL_DATA_DIR", APP_DIR / "data")).resolve()
    )
    yolo_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("YOLO_MODEL_PATH", PROJECT_ROOT / "best.pt")
        ).resolve()
    )
    yolo_confidence_floor: float = float(os.getenv("YOLO_CONFIDENCE_FLOOR", "0.10"))
    yolo_model_s3_uri: str = os.getenv("YOLO_MODEL_S3_URI", "")
    active_learning_threshold: float = float(os.getenv("ACTIVE_LEARNING_THRESHOLD", "0.70"))
    max_video_frames: int = int(os.getenv("MAX_VIDEO_FRAMES", "500"))
    random_seed: int = int(os.getenv("DATASET_SPLIT_SEED", "42"))
    use_chroma: bool = env_bool("USE_CHROMA", False)
    sam2_enabled: bool = env_bool("SAM2_ENABLED", True)
    sam2_provider: str = os.getenv("SAM2_PROVIDER", "mock").lower()
    sam2_checkpoint: Path | None = field(
        default_factory=lambda: Path(os.environ["SAM2_CHECKPOINT"]).resolve()
        if os.getenv("SAM2_CHECKPOINT") else None
    )
    sam2_checkpoint_s3_uri: str = os.getenv("SAM2_CHECKPOINT_S3_URI", "")
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
    yolo_training_enabled: bool = env_bool("YOLO_TRAINING_ENABLED", False)
    aws_region: str = os.getenv("AWS_REGION", "eu-west-3")
    s3_bucket: str = os.getenv("AWS_S3_BUCKET", "")
    s3_prefix: str = os.getenv("AWS_S3_PREFIX", "car-damage")

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
