import unittest

import numpy as np

from utils.reflection_analysis import (
    REFLECTION_RISK_THRESHOLDS,
    compute_reflection_metrics,
    compute_reflection_risk,
)


class ComputeReflectionMetricsTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((100, 200), dtype=np.float32)

    def assert_in_range(self, metrics):
        for key, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, key)
            self.assertLessEqual(value, 1.0, key)

    def test_black_mask_returns_zero_metrics(self):
        metrics = compute_reflection_metrics(self.mask, [10, 10, 50, 50])
        self.assertEqual(metrics["reflection_overlap"], 0.0)
        self.assertEqual(metrics["reflection_mean"], 0.0)
        self.assertEqual(metrics["reflection_max"], 0.0)

    def test_white_mask_returns_full_metrics(self):
        mask = np.ones((100, 200), dtype=np.float32)
        metrics = compute_reflection_metrics(mask, [10, 10, 50, 50])
        self.assertEqual(metrics["reflection_overlap"], 1.0)
        self.assertEqual(metrics["reflection_mean"], 1.0)
        self.assertEqual(metrics["reflection_max"], 1.0)

    def test_partial_overlap(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[0:50, :] = 1.0  # moitie superieure reflechissante
        metrics = compute_reflection_metrics(mask, [0, 0, 100, 100])
        self.assertAlmostEqual(metrics["reflection_overlap"], 0.5, places=3)
        self.assertAlmostEqual(metrics["reflection_mean"], 0.5, places=3)
        self.assertEqual(metrics["reflection_max"], 1.0)
        self.assert_in_range(metrics)

    def test_mask_threshold_changes_overlap(self):
        mask = np.full((10, 10), 0.4, dtype=np.float32)
        low = compute_reflection_metrics(mask, [0, 0, 10, 10], mask_threshold=0.3)
        high = compute_reflection_metrics(mask, [0, 0, 10, 10], mask_threshold=0.5)
        self.assertEqual(low["reflection_overlap"], 1.0)
        self.assertEqual(high["reflection_overlap"], 0.0)

    def test_negative_coordinates_are_clipped(self):
        mask = np.ones((100, 200), dtype=np.float32)
        metrics = compute_reflection_metrics(mask, [-50, -50, 20, 20])
        self.assertEqual(metrics["reflection_overlap"], 1.0)
        self.assert_in_range(metrics)

    def test_box_fully_outside_image(self):
        metrics = compute_reflection_metrics(self.mask, [500, 500, 600, 600])
        self.assertEqual(metrics, {
            "reflection_overlap": 0.0,
            "reflection_mean": 0.0,
            "reflection_max": 0.0,
        })

    def test_empty_box(self):
        metrics = compute_reflection_metrics(self.mask, [30, 30, 30, 30])
        self.assertEqual(metrics["reflection_overlap"], 0.0)

    def test_inverted_box_coordinates(self):
        mask = np.ones((100, 100), dtype=np.float32)
        metrics = compute_reflection_metrics(mask, [50, 50, 10, 10])
        self.assertEqual(metrics["reflection_overlap"], 1.0)

    def test_missing_mask(self):
        metrics = compute_reflection_metrics(None, [10, 10, 50, 50])
        self.assertEqual(metrics, {
            "reflection_overlap": 0.0,
            "reflection_mean": 0.0,
            "reflection_max": 0.0,
        })

    def test_invalid_bbox(self):
        metrics = compute_reflection_metrics(self.mask, ["a", None, 3])
        self.assertEqual(metrics["reflection_overlap"], 0.0)

    def test_values_clipped_to_unit_range(self):
        mask = np.full((10, 10), 3.0, dtype=np.float32)
        metrics = compute_reflection_metrics(mask, [0, 0, 10, 10])
        self.assert_in_range(metrics)


class ComputeReflectionRiskTest(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(compute_reflection_risk(0.0), "LOW")
        self.assertEqual(compute_reflection_risk(0.24), "LOW")
        self.assertEqual(compute_reflection_risk(0.25), "MEDIUM")
        self.assertEqual(compute_reflection_risk(0.59), "MEDIUM")
        self.assertEqual(compute_reflection_risk(0.60), "HIGH")
        self.assertEqual(compute_reflection_risk(1.0), "HIGH")

    def test_custom_thresholds(self):
        custom = {"medium": 0.1, "high": 0.2}
        self.assertEqual(compute_reflection_risk(0.15, custom), "MEDIUM")
        self.assertEqual(compute_reflection_risk(0.25, custom), "HIGH")

    def test_default_thresholds_constant(self):
        self.assertIn("medium", REFLECTION_RISK_THRESHOLDS)
        self.assertIn("high", REFLECTION_RISK_THRESHOLDS)


if __name__ == "__main__":
    unittest.main()
