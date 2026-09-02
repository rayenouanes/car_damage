import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from utils.specseg_adapter import SpecSegAdapter, SpecSegError, image_sha256
from utils.reflection_validation import (
    export_validation_dataset_csv,
    load_validations,
    save_bbox_validation,
)


class SpecSegAdapterTest(unittest.TestCase):
    def setUp(self):
        self.image = (np.random.rand(60, 80, 3) * 255).astype(np.uint8)

    def test_invalid_image_none(self):
        adapter = SpecSegAdapter()
        with self.assertRaises(SpecSegError):
            adapter.predict_mask(None)

    def test_invalid_image_shape(self):
        adapter = SpecSegAdapter()
        with self.assertRaises(SpecSegError):
            adapter.predict_mask(np.zeros((10, 10), dtype=np.uint8))

    def test_unknown_backend(self):
        adapter = SpecSegAdapter(backend="autre")
        with self.assertRaises(SpecSegError):
            adapter.predict_mask(self.image)

    def test_local_backend_not_available_yet(self):
        adapter = SpecSegAdapter(backend="local")
        with self.assertRaises(SpecSegError):
            adapter.predict_mask(self.image)

    def test_network_error_raises_specseg_error(self):
        adapter = SpecSegAdapter()
        with mock.patch.object(
            adapter, "_get_client", side_effect=SpecSegError("pas de reseau")
        ):
            with self.assertRaises(SpecSegError):
                adapter.predict_mask(self.image)

    def test_mask_normalized_resized_and_cached(self):
        adapter = SpecSegAdapter()
        raw_mask = (np.random.rand(30, 40) * 255).astype(np.uint8)
        calls = []

        def fake_predict(image_rgb):
            calls.append(1)
            return raw_mask

        with mock.patch.object(adapter, "_predict_huggingface", side_effect=fake_predict):
            mask1 = adapter.predict_mask(self.image)
            mask2 = adapter.predict_mask(self.image)

        self.assertEqual(mask1.shape, (60, 80))
        self.assertGreaterEqual(float(mask1.min()), 0.0)
        self.assertLessEqual(float(mask1.max()), 1.0)
        # Cache : un seul appel backend pour deux predictions identiques.
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(mask1, mask2)

    def test_image_sha256_deterministic(self):
        self.assertEqual(image_sha256(self.image), image_sha256(self.image.copy()))
        other = self.image.copy()
        other[0, 0, 0] = (int(other[0, 0, 0]) + 1) % 256
        self.assertNotEqual(image_sha256(self.image), image_sha256(other))


class ReflectionValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self.tmpdir, "reflection_validation_data.jsonl")
        self.csv_path = os.path.join(self.tmpdir, "reflection_validation_dataset.csv")

    def _save(self, validation="TP", overlap=0.8):
        return save_bbox_validation(
            self.jsonl_path,
            image_id="frame_001.jpg",
            class_name="scratch",
            yolo_confidence=0.87,
            bbox=[10, 20, 110, 70],
            image_width=640,
            image_height=480,
            validation=validation,
            reflection_overlap=overlap,
            reflection_mean=0.71,
            reflection_max=0.99,
            reflection_risk="HIGH",
        )

    def test_one_line_per_bbox(self):
        self._save("TP")
        self._save("FP")
        self._save("IGNORE")
        records = load_validations(self.jsonl_path)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            [record["validation"] for record in records], ["TP", "FP", "IGNORE"]
        )
        with open(self.jsonl_path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(lines), 3)

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):
            self._save("MAYBE")

    def test_bbox_features(self):
        record = self._save("TP")
        self.assertEqual(record["bbox_width"], 100.0)
        self.assertEqual(record["bbox_height"], 50.0)
        self.assertAlmostEqual(record["bbox_area_ratio"], 5000 / (640 * 480))
        self.assertAlmostEqual(record["bbox_aspect_ratio"], 2.0)

    def test_csv_export_targets_and_filters(self):
        self._save("TP")
        self._save("FP")
        self._save("IGNORE")
        self._save("TP", overlap=None)  # sans metriques SpecSeg -> exclue
        count = export_validation_dataset_csv(self.jsonl_path, self.csv_path)
        self.assertEqual(count, 2)
        with open(self.csv_path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        self.assertEqual(
            lines[0],
            "image_id,class_name,yolo_confidence,reflection_overlap,reflection_mean,"
            "reflection_max,bbox_width,bbox_height,bbox_area_ratio,bbox_aspect_ratio,target",
        )
        targets = [line.split(",")[-1] for line in lines[1:]]
        self.assertEqual(targets, ["0", "1"])


if __name__ == "__main__":
    unittest.main()
