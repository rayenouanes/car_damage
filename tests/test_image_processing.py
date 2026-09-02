import unittest

import numpy as np

from utils.image_processing import apply_clahe
from utils.method_comparison import (
    box_iou,
    build_comparison_rows,
    export_method_results_csv,
    match_boxes,
    save_method_results,
)


class TestApplyClahe(unittest.TestCase):
    def test_returns_new_array_same_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
        original = image.copy()
        result = apply_clahe(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertIsNot(result, image)
        np.testing.assert_array_equal(image, original)  # input untouched

    def test_uniform_image_stays_valid(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = apply_clahe(image)
        self.assertEqual(result.shape, (32, 32, 3))
        self.assertTrue(result.min() >= 0 and result.max() <= 255)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            apply_clahe(np.zeros((10, 10), dtype=np.uint8))

    def test_float_input_converted(self):
        image = np.random.rand(16, 16, 3) * 255
        result = apply_clahe(image)
        self.assertEqual(result.dtype, np.uint8)


class TestBoxMatching(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_iou_disjoint(self):
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_match_requires_same_class_and_iou(self):
        boxes_a = [
            {"class_id": 0, "box": [0, 0, 10, 10], "conf": 0.9},
            {"class_id": 1, "box": [50, 50, 60, 60], "conf": 0.8},
        ]
        boxes_b = [
            {"class_id": 0, "box": [1, 1, 10, 10], "conf": 0.85},
            {"class_id": 0, "box": [50, 50, 60, 60], "conf": 0.7},
        ]
        matched = match_boxes(boxes_a, boxes_b)
        self.assertEqual(len(matched), 1)
        ia, ib, iou = matched[0]
        self.assertEqual((ia, ib), (0, 0))
        self.assertGreater(iou, 0.5)

    def test_no_index_assumption(self):
        boxes_a = [{"class_id": 0, "box": [100, 100, 120, 120], "conf": 0.9}]
        boxes_b = [
            {"class_id": 0, "box": [0, 0, 10, 10], "conf": 0.5},
            {"class_id": 0, "box": [101, 101, 121, 121], "conf": 0.6},
        ]
        matched = match_boxes(boxes_a, boxes_b)
        self.assertEqual(matched[0][:2], (0, 1))


class TestComparisonRowsAndExport(unittest.TestCase):
    def test_rows_reflection_dash_when_absent(self):
        rows = build_comparison_rows(
            {
                "Original": [{"class_id": 0, "box": [0, 0, 5, 5], "conf": 0.87}],
                "SpecSeg": [{
                    "class_id": 0,
                    "box": [0, 0, 5, 5],
                    "conf": 0.87,
                    "reflection_overlap": 0.82,
                    "reflection_risk": "HIGH",
                }],
            },
            lambda cid: f"class_{cid}",
        )
        self.assertEqual(rows[0]["Reflection overlap"], "-")
        self.assertEqual(rows[0]["Risk"], "-")
        self.assertEqual(rows[1]["Reflection overlap"], "82%")
        self.assertEqual(rows[1]["Risk"], "HIGH")

    def test_save_and_export_with_method(self):
        import csv
        import tempfile, os

        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = os.path.join(tmp, "results.jsonl")
            csv_path = os.path.join(tmp, "results.csv")
            save_method_results(
                jsonl_path,
                image_id="img.jpg",
                method="Original",
                boxes=[{"class_id": 0, "box": [0, 0, 5, 5], "conf": 0.9}],
                class_name_getter=lambda cid: "scratch",
            )
            save_method_results(
                jsonl_path,
                image_id="img.jpg",
                method="SpecSeg",
                boxes=[{
                    "class_id": 0,
                    "box": [0, 0, 5, 5],
                    "conf": 0.9,
                    "reflection_overlap": 0.8,
                    "reflection_mean": 0.7,
                    "reflection_max": 0.95,
                    "reflection_risk": "HIGH",
                }],
                class_name_getter=lambda cid: "scratch",
            )
            count = export_method_results_csv(jsonl_path, csv_path)
            self.assertEqual(count, 2)
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["method"], "Original")
            self.assertEqual(rows[0]["reflection_overlap"], "")
            self.assertEqual(rows[1]["method"], "SpecSeg")
            self.assertEqual(rows[1]["reflection_risk"], "HIGH")


if __name__ == "__main__":
    unittest.main()
