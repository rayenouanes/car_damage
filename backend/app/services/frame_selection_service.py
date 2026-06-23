from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import Settings
from backend.app.database import Database
from backend.app.services.tracking_service import TemporalTrackingService
from backend.app.services.yolo_service import YoloService


class FrameSelectionService:
    """Selects keyframes before expensive SAM2/VLM/LLM annotation calls."""

    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings
        self.tracker = TemporalTrackingService(settings)

    def select(self, job_id: str, yolo_service: YoloService) -> tuple[dict[str, list[dict]], dict]:
        job = self.database.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for image in job["images"]:
            grouped[image["source_group"]].append(image)

        prepass: dict[str, list[dict]] = {}
        counters = defaultdict(int)
        all_track_summaries: list[dict] = []
        keyframe_dir = self.settings.path("keyframes") / job_id
        keyframe_dir.mkdir(parents=True, exist_ok=True)

        for group_index, group_images in enumerate(grouped.values(), start=1):
            ordered = sorted(
                group_images,
                key=lambda item: (
                    item["frame_index"] is None,
                    item["frame_index"] if item["frame_index"] is not None else 0,
                    item["created_at"],
                ),
            )
            accepted_hashes: list[str] = []
            quality_candidates: list[dict] = []
            for image in ordered:
                metrics = self._metrics(Path(image["path"]))
                base = {**image, **metrics}
                if metrics["blur_score"] < self.settings.blur_threshold:
                    self._persist(base, "rejected_blur", ["frame_floue"])
                    counters["rejected_blur"] += 1
                    continue
                if any(
                    self._hamming(metrics["perceptual_hash"], previous)
                    <= self.settings.duplicate_hamming_threshold
                    for previous in accepted_hashes
                ):
                    self._persist(base, "rejected_duplicate", ["doublon_visuel"])
                    counters["rejected_duplicate"] += 1
                    continue
                accepted_hashes.append(metrics["perceptual_hash"])
                quality_candidates.append(base)

            tracked_frames = [
                {"image": image, "detections": yolo_service.predict(image)}
                for image in quality_candidates
            ]
            tracked_frames, track_summaries = self.tracker.track(
                tracked_frames, prefix=f"track_{group_index:03d}"
            )
            all_track_summaries.extend(track_summaries)

            for index, frame in enumerate(tracked_frames):
                image = frame["image"]
                selected_detections = [
                    detection for detection in frame["detections"]
                    if detection["is_track_keyframe"]
                ]
                reasons: list[str] = []
                if selected_detections:
                    reasons.extend(["meilleure_frame_piste", "yolo_detection"])
                    for detection in selected_detections:
                        detection.setdefault("active_learning_reasons", []).append(
                            "meilleure_frame_piste"
                        )
                    if any(
                        detection["confidence"] < self.settings.active_learning_threshold
                        for detection in selected_detections
                    ):
                        reasons.append("yolo_peu_confiant")
                    if image["reflection_ratio"] >= self.settings.reflection_ratio_threshold:
                        reasons.append("reflets_forts")
                elif not track_summaries and index == 0:
                    reasons.append("couverture_sans_detection")

                if reasons:
                    self._persist(image, "keyframe", reasons)
                    prepass[image["id"]] = selected_detections
                    target = keyframe_dir / f"{image['id']}_{Path(image['path']).name}"
                    if not target.exists():
                        shutil.copy2(image["path"], target)
                    counters["keyframes"] += 1
                else:
                    self._persist(image, "not_selected", ["hors_meilleure_frame_piste"])
                    counters["not_selected"] += 1

        summary = {
            "job_id": job_id,
            "total_frames": len(job["images"]),
            "keyframes": counters["keyframes"],
            "rejected_blur": counters["rejected_blur"],
            "rejected_duplicate": counters["rejected_duplicate"],
            "not_selected": counters["not_selected"],
            "expensive_model_frames": len(prepass),
            "tracks": len(all_track_summaries),
            "tracked_detections": sum(
                track["frames_seen"] for track in all_track_summaries
            ),
            "best_frames": len(
                {track["best_image_id"] for track in all_track_summaries}
            ),
            "tracking_method": "class_iou_center_quality",
            "track_summaries": all_track_summaries,
        }
        prediction_dir = self.settings.path("predictions") / job_id
        prediction_dir.mkdir(parents=True, exist_ok=True)
        (prediction_dir / "frame_selection.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return prepass, summary

    def _persist(self, image: dict, status: str, reasons: list[str]) -> None:
        self.database.update_image_selection(
            image["id"], status, reasons, image["blur_score"], image["brightness"],
            image["reflection_ratio"], image["perceptual_hash"],
        )

    @staticmethod
    def _metrics(path: Path) -> dict:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Image illisible: {path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        reflection_ratio = float((gray >= 235).mean())
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = (resized[:, 1:] > resized[:, :-1]).flatten()
        value = sum(int(bit) << index for index, bit in enumerate(bits))
        return {
            "blur_score": blur_score,
            "brightness": brightness,
            "reflection_ratio": reflection_ratio,
            "perceptual_hash": f"{value:016x}",
        }

    @staticmethod
    def _hamming(first: str, second: str) -> int:
        return (int(first, 16) ^ int(second, 16)).bit_count()

    @staticmethod
    def _predictions_changed(previous: list[dict], current: list[dict]) -> bool:
        if len(previous) != len(current):
            return True
        if {item["class_name"] for item in previous} != {item["class_name"] for item in current}:
            return True
        if not previous:
            return False
        previous_confidence = sum(item["confidence"] for item in previous) / len(previous)
        current_confidence = sum(item["confidence"] for item in current) / len(current)
        if abs(previous_confidence - current_confidence) >= 0.15:
            return True
        return not any(
            FrameSelectionService._iou(first["bbox"], second["bbox"]) >= 0.35
            for first in previous for second in current
            if first["class_name"] == second["class_name"]
        )

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / max(1e-9, first_area + second_area - intersection)
