import csv
import json
import os
import tempfile
import unittest

from utils.validation_utils import (
    bbox_features,
    export_validation_dataset_csv,
    load_validations,
    normalize_label,
    save_bbox_validation,
    validation_summary,
)


class TestNormalizeLabel(unittest.TestCase):
    def test_valid_labels(self):
        self.assertEqual(normalize_label("tp"), "TP")
        self.assertEqual(normalize_label("FP"), "FP")
        self.assertEqual(normalize_label("unsure"), "UNSURE")

    def test_legacy_ignore_maps_to_unsure(self):
        self.assertEqual(normalize_label("IGNORE"), "UNSURE")


class TestBboxFeatures(unittest.TestCase):
    def test_basic_features(self):
        features = bbox_features([10, 20, 60, 120], 200, 200)
        self.assertEqual(features["bbox_width"], 50.0)
        self.assertEqual(features["bbox_height"], 100.0)
        self.assertAlmostEqual(features["bbox_area_ratio"], 5000.0 / 40000.0)
        self.assertAlmostEqual(features["bbox_aspect_ratio"], 0.5)

    def test_empty_box(self):
        features = bbox_features([50, 50, 50, 50], 100, 100)
        self.assertEqual(features["bbox_width"], 0.0)
        self.assertEqual(features["bbox_aspect_ratio"], 0.0)


class TestSaveAndLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.jsonl = os.path.join(self.tmp.name, "validation_data.jsonl")
        self.csv = os.path.join(self.tmp.name, "validation_dataset.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, validation, **overrides):
        params = dict(
            image_id="img1.jpg",
            detection_id="img1#0",
            class_name="scratch",
            yolo_confidence=0.87,
            bbox=[10, 10, 50, 50],
            image_width=100,
            image_height=100,
            validation=validation,
        )
        params.update(overrides)
        return save_bbox_validation(self.jsonl, **params)

    def test_save_creates_one_line_per_bbox(self):
        self._save("TP")
        self._save("FP", detection_id="img1#1")
        with open(self.jsonl, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["validation"], "TP")
        self.assertEqual(lines[1]["detection_id"], "img1#1")
        self.assertIn("timestamp", lines[0])

    def test_invalid_label_raises(self):
        with self.assertRaises(ValueError):
            self._save("MAYBE")

    def test_load_normalizes_legacy_ignore(self):
        self._save("IGNORE")
        records = load_validations(self.jsonl)
        self.assertEqual(records[0]["validation"], "UNSURE")

    def test_load_reads_legacy_file(self):
        legacy = os.path.join(self.tmp.name, "reflection_validation_data.jsonl")
        with open(legacy, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"image_id": "old.jpg", "validation": "TP"}) + "\n")
        self._save("FP")
        records = load_validations(self.jsonl)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["image_id"], "old.jpg")

    def test_export_csv_excludes_unsure(self):
        self._save("TP")
        self._save("FP", detection_id="img1#1")
        self._save("UNSURE", detection_id="img1#2")
        exported = export_validation_dataset_csv(self.jsonl, self.csv)
        self.assertEqual(exported, 2)
        with open(self.csv, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["target"], "0")
        self.assertEqual(rows[1]["target"], "1")

    def test_summary(self):
        self._save("TP")
        self._save("FP", image_id="img2.jpg")
        summary = validation_summary(load_validations(self.jsonl))
        self.assertEqual(summary["TP"], 1)
        self.assertEqual(summary["FP"], 1)
        self.assertEqual(summary["UNSURE"], 0)
        self.assertEqual(summary["images"], 2)


if __name__ == "__main__":
    unittest.main()
