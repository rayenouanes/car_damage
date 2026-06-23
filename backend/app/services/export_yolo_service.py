from __future__ import annotations

import json
import random
import shutil
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path

from backend.app.config import CANONICAL_CLASSES, CLASS_TO_ID, Settings
from backend.app.database import Database
from backend.app.models.schemas import ExportSplitRequest


class ExportYoloService:
    def __init__(self, database: Database, settings: Settings):
        self.database = database
        self.settings = settings

    def export(self, job_id: str, split: ExportSplitRequest) -> dict:
        images = self.database.reviewed_export_rows(job_id)
        eligible = [image for image in images if self._image_is_exportable(image)]
        assignments = self._assign_groups(eligible, split)
        export_id = f"dataset_{job_id[:8]}_{uuid.uuid4().hex[:8]}"
        export_dir = self.settings.path("exports") / export_id
        for subset in ("train", "val", "test"):
            (export_dir / "images" / subset).mkdir(parents=True, exist_ok=False)
            (export_dir / "labels" / subset).mkdir(parents=True, exist_ok=False)

        manifest: list[dict] = []
        box_count = 0
        for image in eligible:
            subset = assignments[image["source_group"]]
            source = Path(image["path"])
            target_name = f"{image['id'][:10]}_{source.name}"
            shutil.copy2(source, export_dir / "images" / subset / target_name)
            lines = self._label_lines(image, split.annotation_format)
            label_path = export_dir / "labels" / subset / f"{Path(target_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            box_count += len(lines)
            manifest.append(
                {
                    "image_id": image["id"],
                    "source_group": image["source_group"],
                    "split": subset,
                    "labels": len(lines),
                }
            )

        yaml_lines = [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CANONICAL_CLASSES)],
        ]
        (export_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
        counts = {
            subset: sum(1 for item in manifest if item["split"] == subset)
            for subset in ("train", "val", "test")
        }
        summary = {
            "export_id": export_id,
            "job_id": job_id,
            "images": len(eligible),
            "boxes": box_count,
            "annotation_format": split.annotation_format,
            "skipped_images": len(images) - len(eligible),
            "split_counts": counts,
            "group_leakage": self._has_group_leakage(manifest),
            "manifest": manifest,
        }
        (export_dir / "manifest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        archive_path = self.settings.path("exports") / f"{export_id}.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in export_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(export_dir))
        summary["archive_name"] = archive_path.name
        return summary

    @staticmethod
    def _image_is_exportable(image: dict) -> bool:
        blocked = {"bbox_later", "error_bank"}
        return not any(
            prediction["correction"]["action"] in blocked for prediction in image["predictions"]
        )

    def _label_lines(self, image: dict, annotation_format: str) -> list[str]:
        lines: list[str] = []
        excluded = {"reject", "reflection", "dirt", "shadow"}
        for prediction in image["predictions"]:
            correction = prediction["correction"]
            if correction["action"] in excluded:
                continue
            class_name = correction.get("classe_finale") or prediction["class_name"]
            bbox = correction.get("bbox_finale") or prediction["bbox"]
            if class_name not in CLASS_TO_ID:
                continue
            if annotation_format == "segmentation":
                polygon = correction.get("masque_final")
                if polygon is None and prediction.get("sam2"):
                    polygon = prediction["sam2"].get("polygon")
                if polygon is None:
                    x1, y1, x2, y2 = bbox
                    polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                lines.append(
                    self._to_yolo_segmentation(
                        CLASS_TO_ID[class_name], polygon,
                        int(image["width"]), int(image["height"]),
                    )
                )
            else:
                lines.append(
                    self._to_yolo(
                        CLASS_TO_ID[class_name], bbox,
                        int(image["width"]), int(image["height"]),
                    )
                )
        return lines

    def _assign_groups(self, images: list[dict], split: ExportSplitRequest) -> dict[str, str]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for image in images:
            grouped[image["source_group"]].append(image)
        groups = list(grouped.items())
        random.Random(self.settings.random_seed).shuffle(groups)
        groups.sort(key=lambda item: len(item[1]), reverse=True)
        ratios = {"train": split.train, "val": split.val, "test": split.test}
        targets = {name: ratio * len(images) for name, ratio in ratios.items()}
        current = {name: 0 for name in ratios}
        assignments: dict[str, str] = {}
        for group_name, members in groups:
            subset = max(
                ratios,
                key=lambda name: (targets[name] - current[name], ratios[name]),
            )
            assignments[group_name] = subset
            current[subset] += len(members)
        return assignments

    @staticmethod
    def _has_group_leakage(manifest: list[dict]) -> bool:
        groups: dict[str, set[str]] = defaultdict(set)
        for item in manifest:
            groups[item["source_group"]].add(item["split"])
        return any(len(splits) > 1 for splits in groups.values())

    @staticmethod
    def _to_yolo(class_id: int, bbox: list[float], width: int, height: int) -> str:
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    @staticmethod
    def _to_yolo_segmentation(
        class_id: int, polygon: list[list[float]], width: int, height: int
    ) -> str:
        normalized: list[str] = []
        for x, y in polygon:
            normalized.append(f"{max(0.0, min(float(width), x)) / width:.6f}")
            normalized.append(f"{max(0.0, min(float(height), y)) / height:.6f}")
        return f"{class_id} {' '.join(normalized)}"

