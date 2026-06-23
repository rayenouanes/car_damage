from __future__ import annotations

from math import hypot

from backend.app.config import Settings


class TemporalTrackingService:
    """Groups sampled-frame detections into short class-aware temporal tracks."""

    def __init__(self, settings: Settings):
        self.max_gap = max(1, settings.tracking_max_gap)
        self.iou_threshold = settings.tracking_iou_threshold
        self.center_distance_threshold = settings.tracking_center_distance_threshold

    def track(
        self, frames: list[dict], prefix: str = "track"
    ) -> tuple[list[dict], list[dict]]:
        tracks: list[dict] = []
        next_id = 1

        for position, frame in enumerate(frames):
            image = frame["image"]
            detections = sorted(
                (dict(item) for item in frame.get("detections", [])),
                key=lambda item: float(item["confidence"]),
                reverse=True,
            )
            used_tracks: set[str] = set()
            for detection in detections:
                best_track: dict | None = None
                best_match = -1.0
                for track in tracks:
                    if track["id"] in used_tracks:
                        continue
                    if track["class_name"] != detection["class_name"]:
                        continue
                    if position - track["last_position"] > self.max_gap:
                        continue
                    match_score = self._match_score(
                        track["last_bbox"], detection["bbox"], image["width"], image["height"]
                    )
                    if match_score is not None and match_score > best_match:
                        best_track = track
                        best_match = match_score

                if best_track is None:
                    best_track = {
                        "id": f"{prefix}_{next_id:04d}",
                        "class_name": detection["class_name"],
                        "last_position": position,
                        "last_bbox": detection["bbox"],
                        "length": 0,
                        "best_score": -1.0,
                        "best_image_id": None,
                        "best_frame_index": None,
                        "best_timestamp_seconds": None,
                        "best_confidence": 0.0,
                    }
                    tracks.append(best_track)
                    next_id += 1

                track_score = self._frame_score(image, detection)
                detection["track_id"] = best_track["id"]
                detection["track_score"] = track_score
                detection["is_track_keyframe"] = False
                best_track["last_position"] = position
                best_track["last_bbox"] = detection["bbox"]
                best_track["length"] += 1
                used_tracks.add(best_track["id"])
                if track_score > best_track["best_score"]:
                    best_track["best_score"] = track_score
                    best_track["best_image_id"] = image["id"]
                    best_track["best_frame_index"] = image.get("frame_index")
                    best_track["best_timestamp_seconds"] = image.get("timestamp_seconds")
                    best_track["best_confidence"] = float(detection["confidence"])
            frame["detections"] = detections

        track_by_id = {track["id"]: track for track in tracks}
        for frame in frames:
            for detection in frame.get("detections", []):
                track = track_by_id[detection["track_id"]]
                detection["track_length"] = track["length"]
                detection["is_track_keyframe"] = (
                    frame["image"]["id"] == track["best_image_id"]
                )

        summaries = [
            {
                "track_id": track["id"],
                "class_name": track["class_name"],
                "frames_seen": track["length"],
                "best_image_id": track["best_image_id"],
                "best_frame_index": track["best_frame_index"],
                "best_timestamp_seconds": track["best_timestamp_seconds"],
                "best_confidence": track["best_confidence"],
                "best_score": track["best_score"],
            }
            for track in tracks
        ]
        return frames, summaries

    def _match_score(
        self, previous: list[float], current: list[float], width: int, height: int
    ) -> float | None:
        iou = self._iou(previous, current)
        previous_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
        current_center = ((current[0] + current[2]) / 2, (current[1] + current[3]) / 2)
        center_distance = hypot(
            previous_center[0] - current_center[0], previous_center[1] - current_center[1]
        ) / max(1.0, hypot(width, height))
        previous_area = max(1.0, (previous[2] - previous[0]) * (previous[3] - previous[1]))
        current_area = max(1.0, (current[2] - current[0]) * (current[3] - current[1]))
        area_similarity = min(previous_area, current_area) / max(previous_area, current_area)
        if iou < self.iou_threshold and (
            center_distance > self.center_distance_threshold or area_similarity < 0.35
        ):
            return None
        return 0.65 * iou + 0.25 * max(0.0, 1.0 - center_distance) + 0.10 * area_similarity

    @staticmethod
    def _frame_score(image: dict, detection: dict) -> float:
        confidence = float(detection["confidence"])
        sharpness = min(1.0, float(image.get("blur_score") or 0.0) / 250.0)
        brightness = float(image.get("brightness") or 128.0)
        brightness_quality = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        reflection = float(image.get("reflection_ratio") or 0.0)
        reflection_quality = max(0.0, 1.0 - reflection / 0.12)
        x1, y1, x2, y2 = detection["bbox"]
        edge_margin = min(x1, y1, image["width"] - x2, image["height"] - y2)
        edge_quality = min(1.0, max(0.0, edge_margin) / max(1.0, min(image["width"], image["height"]) * 0.04))
        return round(
            0.60 * confidence
            + 0.18 * sharpness
            + 0.08 * brightness_quality
            + 0.08 * reflection_quality
            + 0.06 * edge_quality,
            6,
        )

    @staticmethod
    def _iou(first: list[float], second: list[float]) -> float:
        x1, y1 = max(first[0], second[0]), max(first[1], second[1])
        x2, y2 = min(first[2], second[2]), min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return intersection / max(1e-9, first_area + second_area - intersection)
