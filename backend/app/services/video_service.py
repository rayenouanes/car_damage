from __future__ import annotations

import re
from pathlib import Path

import cv2

from backend.app.config import Settings
from backend.app.database import Database


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def safe_filename(filename: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return result or "upload.bin"


class VideoService:
    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings

    def ingest(
        self,
        job_id: str,
        filename: str,
        content: bytes,
        source_index: int,
        every_n_frames: int = 30,
        every_seconds: float | None = None,
    ) -> int:
        name = safe_filename(filename)
        upload_dir = self.settings.path("uploads") / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / f"{source_index:03d}_{name}"
        upload_path.write_bytes(content)
        suffix = upload_path.suffix.lower()
        source_group = f"{job_id}:{source_index:03d}:{name}"
        if suffix in IMAGE_EXTENSIONS:
            image = cv2.imread(str(upload_path))
            if image is None:
                raise ValueError(f"Image illisible: {filename}")
            height, width = image.shape[:2]
            self.database.add_image(
                job_id, upload_path, filename, source_group, width, height
            )
            return 1
        if suffix in VIDEO_EXTENSIONS:
            return self.extract_frames(
                job_id, upload_path, filename, source_group, every_n_frames, every_seconds
            )
        raise ValueError(f"Format non supporte: {suffix or 'sans extension'}")

    def extract_frames(
        self,
        job_id: str,
        video_path: Path,
        original_name: str,
        source_group: str,
        every_n_frames: int = 30,
        every_seconds: float | None = None,
    ) -> int:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Video illisible: {original_name}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if every_seconds is not None and every_seconds > 0:
            step = max(1, int(round(fps * every_seconds)))
        else:
            step = max(1, int(every_n_frames))
        frames_dir = self.settings.path("frames") / job_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        frame_index = 0
        stem = safe_filename(Path(original_name).stem)
        try:
            while saved < self.settings.max_video_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    frame_path = frames_dir / f"{stem}_{frame_index:07d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise OSError(f"Impossible d'ecrire {frame_path}")
                    height, width = frame.shape[:2]
                    self.database.add_image(
                        job_id, frame_path, original_name, source_group, width, height,
                        frame_index=frame_index, timestamp_seconds=frame_index / fps,
                    )
                    saved += 1
                frame_index += 1
        finally:
            capture.release()
        if saved == 0:
            raise ValueError(f"Aucune frame extraite: {original_name}")
        return saved

