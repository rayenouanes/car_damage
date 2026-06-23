from __future__ import annotations

import unittest

from backend.app.config import Settings
from backend.app.services.tracking_service import TemporalTrackingService


class TemporalTrackingServiceTests(unittest.TestCase):
    def test_groups_same_class_and_selects_highest_quality_frame(self):
        settings = Settings(
            tracking_max_gap=2,
            tracking_iou_threshold=0.10,
            tracking_center_distance_threshold=0.25,
        )
        frames = []
        confidences = [0.45, 0.91, 0.70]
        for index, confidence in enumerate(confidences):
            frames.append(
                {
                    "image": {
                        "id": f"image-{index}",
                        "width": 200,
                        "height": 120,
                        "frame_index": index * 10,
                        "timestamp_seconds": float(index),
                        "blur_score": 180.0 + index * 10,
                        "brightness": 125.0,
                        "reflection_ratio": 0.01,
                    },
                    "detections": [
                        {
                            "class_name": "rayure",
                            "confidence": confidence,
                            "bbox": [30 + index * 2, 25, 90 + index * 2, 65],
                        }
                    ],
                }
            )

        tracked, summaries = TemporalTrackingService(settings).track(frames, "video")

        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["frames_seen"], 3)
        self.assertEqual(summaries[0]["best_image_id"], "image-1")
        detections = [frame["detections"][0] for frame in tracked]
        self.assertEqual({item["track_id"] for item in detections}, {"video_0001"})
        self.assertEqual([item["is_track_keyframe"] for item in detections], [False, True, False])
        self.assertTrue(all(item["track_length"] == 3 for item in detections))

    def test_separates_classes_even_when_boxes_overlap(self):
        settings = Settings()
        frames = [
            {
                "image": {
                    "id": "image-0", "width": 200, "height": 120,
                    "frame_index": 0, "blur_score": 200.0,
                    "brightness": 128.0, "reflection_ratio": 0.0,
                },
                "detections": [
                    {"class_name": "rayure", "confidence": 0.8, "bbox": [20, 20, 80, 60]},
                    {"class_name": "bosse", "confidence": 0.7, "bbox": [20, 20, 80, 60]},
                ],
            }
        ]

        _, summaries = TemporalTrackingService(settings).track(frames, "video")

        self.assertEqual(len(summaries), 2)
        self.assertEqual({item["class_name"] for item in summaries}, {"rayure", "bosse"})


if __name__ == "__main__":
    unittest.main()
