from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.app.models.db_models import SCHEMA_SQL
from backend.app.models.schemas import CorrectionRequest, ErrorBankRecord


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA_SQL)
            self._ensure_column(connection, "jobs", "pipeline_mode", "TEXT NOT NULL DEFAULT 'training'")
            self._ensure_column(connection, "images", "blur_score", "REAL")
            self._ensure_column(connection, "images", "brightness", "REAL")
            self._ensure_column(connection, "images", "reflection_ratio", "REAL")
            self._ensure_column(connection, "images", "perceptual_hash", "TEXT")
            self._ensure_column(
                connection, "images", "selection_status", "TEXT NOT NULL DEFAULT 'pending'"
            )
            self._ensure_column(
                connection, "images", "selection_reasons_json", "TEXT NOT NULL DEFAULT '[]'"
            )
            self._ensure_column(connection, "predictions", "track_id", "TEXT")
            self._ensure_column(connection, "predictions", "track_score", "REAL")
            self._ensure_column(connection, "predictions", "track_length", "INTEGER")
            self._ensure_column(
                connection, "predictions", "is_track_keyframe", "INTEGER NOT NULL DEFAULT 0"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_track ON predictions(track_id)"
            )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection, table: str, column: str, definition: str
    ) -> None:
        existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_job(self, source_name: str, source_type: str, pipeline_mode: str = "training") -> str:
        job_id = uuid.uuid4().hex
        now = utc_now()
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO jobs
                   (id, source_name, source_type, status, pipeline_mode, error, created_at, updated_at)
                   VALUES (?, ?, ?, 'uploaded', ?, NULL, ?, ?)""",
                (job_id, source_name, source_type, pipeline_mode, now, now),
            )
        return job_id

    def update_job(self, job_id: str, status: str, error: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error, utc_now(), job_id),
            )

    def add_image(
        self, job_id: str, path: Path, original_name: str, source_group: str,
        width: int, height: int, frame_index: int | None = None,
        timestamp_seconds: float | None = None,
    ) -> str:
        image_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO images
                   (id, job_id, path, original_name, source_group, frame_index,
                    timestamp_seconds, width, height, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (image_id, job_id, str(path.resolve()), original_name, source_group,
                 frame_index, timestamp_seconds, width, height, utc_now()),
            )
        return image_id

    def add_prediction(self, image_id: str, detection: dict[str, Any]) -> str:
        prediction_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute(
                """INSERT INTO predictions
                   (id, image_id, class_name, raw_class_name, confidence, bbox_json,
                    status, active_learning_reasons_json, track_id, track_score,
                    track_length, is_track_keyframe, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id, image_id, detection["class_name"],
                    detection.get("raw_class_name"), detection["confidence"],
                    json.dumps(detection["bbox"]), detection.get("status", "pending_review"),
                    json.dumps(detection.get("active_learning_reasons", [])),
                    detection.get("track_id"), detection.get("track_score"),
                    detection.get("track_length"),
                    int(detection.get("is_track_keyframe", False)), utc_now(),
                ),
            )
        return prediction_id

    def save_analysis(self, table: str, prediction_id: str, provider: str, payload: dict) -> None:
        if table not in {"vlm_analyses", "llm_decisions"}:
            raise ValueError("Invalid analysis table")
        with self.connect() as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?)",
                (prediction_id, provider, json.dumps(payload, ensure_ascii=False), utc_now()),
            )

    def save_sam2_mask(self, prediction_id: str, mask: dict[str, Any]) -> None:
        with self.connect() as connection:
            connection.execute(
                """INSERT OR REPLACE INTO sam2_masks
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id, mask["provider"], json.dumps(mask["polygon"]),
                    mask["confidence"], mask.get("mask_path"),
                    int(mask.get("is_bbox_fallback", False)), utc_now(),
                ),
            )

    def update_image_selection(
        self,
        image_id: str,
        status: str,
        reasons: list[str],
        blur_score: float,
        brightness: float,
        reflection_ratio: float,
        perceptual_hash: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """UPDATE images SET selection_status = ?, selection_reasons_json = ?,
                   blur_score = ?, brightness = ?, reflection_ratio = ?, perceptual_hash = ?
                   WHERE id = ?""",
                (
                    status, json.dumps(reasons), blur_score, brightness,
                    reflection_ratio, perceptual_hash, image_id,
                ),
            )

    def clear_predictions(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM predictions WHERE image_id IN (SELECT id FROM images WHERE job_id = ?)",
                (job_id,),
            )

    def get_job(self, job_id: str, include_details: bool = True) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            job = dict(row)
            if include_details:
                images = connection.execute(
                    "SELECT * FROM images WHERE job_id = ? ORDER BY source_group, frame_index, created_at",
                    (job_id,),
                ).fetchall()
                job["images"] = []
                for image in images:
                    image_item = self._decode_row(image)
                    image_item["predictions"] = self._predictions_for_image(connection, image["id"])
                    job["images"].append(image_item)
            return job

    def list_jobs(self) -> list[dict]:
        query = """
        SELECT j.*, COUNT(DISTINCT i.id) AS image_count,
               COUNT(DISTINCT CASE WHEN i.selection_status = 'keyframe' THEN i.id END) AS keyframe_count,
               COUNT(DISTINCT p.id) AS prediction_count,
               COUNT(DISTINCT c.prediction_id) AS reviewed_count
        FROM jobs j LEFT JOIN images i ON i.job_id = j.id
        LEFT JOIN predictions p ON p.image_id = i.id
        LEFT JOIN corrections c ON c.prediction_id = p.id
        GROUP BY j.id ORDER BY j.created_at DESC
        """
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query).fetchall()]

    def get_image(self, image_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()
        return dict(row) if row else None

    def neighboring_images(self, image_id: str, radius: int = 2) -> list[dict]:
        image = self.get_image(image_id)
        if not image or image["frame_index"] is None:
            return []
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT * FROM images WHERE source_group = ? AND id != ?
                   ORDER BY ABS(frame_index - ?) LIMIT ?""",
                (image["source_group"], image_id, image["frame_index"], radius * 2),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_prediction_context(self, prediction_id: str) -> dict | None:
        query = """
        SELECT p.*, i.path AS image_path, i.width, i.height, i.job_id, i.source_group,
               v.payload_json AS vlm_json, l.payload_json AS llm_json,
               s.provider AS sam2_provider, s.polygon_json AS sam2_polygon_json,
               s.confidence AS sam2_confidence, s.mask_path AS sam2_mask_path,
               s.is_bbox_fallback AS sam2_is_bbox_fallback
        FROM predictions p JOIN images i ON i.id = p.image_id
        LEFT JOIN vlm_analyses v ON v.prediction_id = p.id
        LEFT JOIN llm_decisions l ON l.prediction_id = p.id
        LEFT JOIN sam2_masks s ON s.prediction_id = p.id
        WHERE p.id = ?
        """
        with self.connect() as connection:
            row = connection.execute(query, (prediction_id,)).fetchone()
        return self._decode_row(row) if row else None

    def review_queue(self, job_id: str | None = None, limit: int = 200) -> list[dict]:
        where = "AND i.job_id = ?" if job_id else ""
        params: tuple[Any, ...] = (job_id, limit) if job_id else (limit,)
        query = f"""
        SELECT p.*, i.path AS image_path, i.original_name, i.job_id, i.width, i.height,
               v.payload_json AS vlm_json, l.payload_json AS llm_json,
               s.provider AS sam2_provider, s.polygon_json AS sam2_polygon_json,
               s.confidence AS sam2_confidence, s.mask_path AS sam2_mask_path,
               s.is_bbox_fallback AS sam2_is_bbox_fallback
        FROM predictions p JOIN images i ON i.id = p.image_id
        LEFT JOIN vlm_analyses v ON v.prediction_id = p.id
        LEFT JOIN llm_decisions l ON l.prediction_id = p.id
        LEFT JOIN sam2_masks s ON s.prediction_id = p.id
        WHERE NOT EXISTS (SELECT 1 FROM corrections c WHERE c.prediction_id = p.id)
        {where}
        ORDER BY CASE WHEN l.payload_json IS NOT NULL THEN 0 ELSE 1 END, p.confidence ASC LIMIT ?
        """
        with self.connect() as connection:
            return [self._decode_row(row) for row in connection.execute(query, params).fetchall()]

    def save_correction_and_error(
        self, prediction_id: str, correction: CorrectionRequest, record: ErrorBankRecord
    ) -> str:
        correction_id = uuid.uuid4().hex
        now = utc_now()
        payload = correction.model_dump(mode="json")
        with self.connect() as connection:
            if not connection.execute(
                "SELECT 1 FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone():
                raise KeyError(prediction_id)
            connection.execute(
                "INSERT INTO corrections VALUES (?, ?, ?, ?)",
                (correction_id, prediction_id, json.dumps(payload, ensure_ascii=False), now),
            )
            connection.execute(
                """INSERT INTO error_bank VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex, correction_id, prediction_id,
                    record.model_dump_json(), record.type_erreur, record.classe_finale,
                    int(record.ajouter_finetuning), now,
                ),
            )
            status = "accepted" if correction.action.value == "accept" else (
                "rejected" if correction.action.value in {"reject", "reflection", "dirt", "shadow"}
                else "corrected"
            )
            connection.execute("UPDATE predictions SET status = ? WHERE id = ?", (status, prediction_id))
        return correction_id

    def list_error_bank(self, limit: int = 500) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM error_bank ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_row(row) for row in rows]

    def similar_error_count(self, class_name: str) -> int:
        query = """
        SELECT COUNT(*) FROM error_bank e JOIN predictions p ON p.id = e.prediction_id
        WHERE p.class_name = ? AND e.type_erreur IS NOT NULL
        """
        with self.connect() as connection:
            return int(connection.execute(query, (class_name,)).fetchone()[0])

    def reviewed_export_rows(self, job_id: str) -> list[dict]:
        job = self.get_job(job_id, include_details=True)
        if not job:
            raise KeyError(job_id)
        rows: list[dict] = []
        for image in job["images"]:
            predictions = image["predictions"]
            if not predictions or any(prediction.get("correction") is None for prediction in predictions):
                continue
            rows.append({**image, "predictions": predictions})
        return rows

    def _predictions_for_image(self, connection: sqlite3.Connection, image_id: str) -> list[dict]:
        query = """
        SELECT p.*, v.payload_json AS vlm_json, l.payload_json AS llm_json,
               s.provider AS sam2_provider, s.polygon_json AS sam2_polygon_json,
               s.confidence AS sam2_confidence, s.mask_path AS sam2_mask_path,
               s.is_bbox_fallback AS sam2_is_bbox_fallback,
               c.payload_json AS correction_json
        FROM predictions p
        LEFT JOIN vlm_analyses v ON v.prediction_id = p.id
        LEFT JOIN llm_decisions l ON l.prediction_id = p.id
        LEFT JOIN sam2_masks s ON s.prediction_id = p.id
        LEFT JOIN corrections c ON c.id = (
            SELECT c2.id FROM corrections c2 WHERE c2.prediction_id = p.id
            ORDER BY c2.created_at DESC, c2.rowid DESC LIMIT 1
        )
        WHERE p.image_id = ? ORDER BY p.confidence ASC
        """
        return [self._decode_row(row) for row in connection.execute(query, (image_id,)).fetchall()]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                target = key[:-5]
                try:
                    item[target] = json.loads(item[key]) if item[key] is not None else None
                except (TypeError, json.JSONDecodeError):
                    item[target] = item[key]
                del item[key]
        if "sam2_provider" in item:
            provider = item.pop("sam2_provider")
            polygon = item.pop("sam2_polygon", None)
            confidence = item.pop("sam2_confidence", None)
            mask_path = item.pop("sam2_mask_path", None)
            fallback = item.pop("sam2_is_bbox_fallback", None)
            item["sam2"] = (
                {
                    "provider": provider,
                    "polygon": polygon,
                    "confidence": confidence,
                    "mask_path": mask_path,
                    "is_bbox_fallback": bool(fallback),
                }
                if provider else None
            )
        if "is_track_keyframe" in item:
            item["is_track_keyframe"] = bool(item["is_track_keyframe"])
        return item
