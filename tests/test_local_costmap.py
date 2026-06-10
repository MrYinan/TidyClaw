from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local_costmap import LocalCostmap


class LocalCostmapTests(unittest.TestCase):
    def vision_with_front_obstacle(self, distance_m: float = 0.35) -> dict:
        depth = np.full((60, 60), np.nan, dtype=np.float32)
        # Central obstacle ahead at camera height. With horizon=0 it falls
        # inside the configured obstacle-height band.
        depth[28:33, 28:33] = float(distance_m)
        return {
            "depth_frame": depth,
            "camera": {
                "fx": 30.0,
                "fy": 30.0,
                "cx": 30.0,
                "cy": 30.0,
                "camera_height_m": 0.90,
                "camera_horizon_deg": 0.0,
            },
        }

    def test_front_obstacle_vetoes_moveahead_but_keeps_lateral_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            result = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={},
                holding_object=False,
                step=1,
            )
            self.assertEqual(result["status"], "success")
            self.assertFalse(result["moveahead_safe"])
            self.assertIn("MoveAhead", result["blocked_actions"])
            moveahead = result["action_safety"]["MoveAhead"]
            self.assertEqual(moveahead["reason"], "inflated_obstacle_in_swept_volume")
            self.assertGreater(moveahead["blocked_cell_count"], 0)
            self.assertTrue(moveahead["blocked_sources"])
            source_records = moveahead["blocked_sources"][0]["source_records"]
            self.assertTrue(source_records)
            self.assertIn("pixel", source_records[0])
            self.assertIn("point_m", source_records[0])
            self.assertEqual(result["moveahead_blockers"], moveahead["blocked_sources"])
            self.assertEqual(result["front_corridor"]["status"], "success")
            self.assertIn("dominant_blocker_side", result["front_corridor"])
            self.assertIn("lanes", result["front_corridor"])
            self.assertTrue((Path(tmp) / "navigation-costmap.json").exists())

    def test_moveahead_blocker_reports_overlapping_visual_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            result = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={
                    "best_obstacle_candidate": {
                        "label": "CounterTop",
                        "task_semantic_class": "obstacle",
                        "confidence": 0.98,
                        "bbox": {"x": 26, "y": 26, "w": 12, "h": 12},
                    }
                },
                holding_object=False,
                step=1,
                persist=False,
            )
            blocker_records = [
                record
                for blocker in result["moveahead_blockers"]
                for record in blocker.get("source_records", [])
            ]
            overlaps = [
                overlap
                for record in blocker_records
                for overlap in record.get("candidate_overlaps", [])
            ]
            self.assertTrue(overlaps)
            self.assertIn("CounterTop", {str(item.get("label")) for item in overlaps})

    def test_held_object_expands_collision_footprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            empty_hand = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={},
                holding_object=False,
                persist=False,
            )
            held = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={},
                holding_object=True,
                held_object_labels=["apple"],
                persist=False,
            )
            self.assertGreater(held["inflated_robot_radius_m"], empty_hand["inflated_robot_radius_m"])
            self.assertGreater(held["inflated_cell_count"], empty_hand["inflated_cell_count"])
            self.assertEqual(held["held_object_labels"], ["apple"])

    def test_held_overlay_depth_is_masked_as_self_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            result = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={
                    "held_object_family": "food",
                    "held_object_overlay_candidates": [
                        {"label": "apple", "bbox": {"x": 24, "y": 24, "w": 16, "h": 16}}
                    ],
                },
                holding_object=True,
                held_object_labels=["apple"],
                persist=False,
            )
            self.assertEqual(result["held_overlay_bbox_count"], 1)
            self.assertGreater(result["skipped_held_overlay_depth_point_count"], 0)
            self.assertEqual(result["occupied_cell_count"], 0)
            moveahead = result["action_safety"]["MoveAhead"]
            self.assertEqual(moveahead["blocked_cell_count"], 0)
            self.assertNotEqual(moveahead["reason"], "inflated_obstacle_in_swept_volume")

    def test_held_footprint_can_veto_a_move_that_is_safe_empty_hand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            vision = self.vision_with_front_obstacle(distance_m=0.50)
            empty_hand = manager.update(vision=vision, analysis={}, holding_object=False, persist=False)
            held_pan = manager.update(
                vision=vision,
                analysis={"held_object_family": "cookware"},
                holding_object=True,
                held_object_labels=["pan"],
                persist=False,
            )
            self.assertTrue(empty_hand["moveahead_safe"])
            self.assertFalse(held_pan["moveahead_safe"])

    def test_large_held_object_uses_larger_profile_than_small_food(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            apple = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={"held_object_family": "food"},
                holding_object=True,
                held_object_labels=["apple"],
                persist=False,
            )
            pan = manager.update(
                vision=self.vision_with_front_obstacle(),
                analysis={"held_object_family": "cookware"},
                holding_object=True,
                held_object_labels=["pan"],
                persist=False,
            )
            self.assertEqual(apple["held_footprint_profile"], "default_carried_object")
            self.assertEqual(pan["held_footprint_profile"], "large_carried_object")
            self.assertGreater(pan["inflated_robot_radius_m"], apple["inflated_robot_radius_m"])

    def test_missing_depth_is_explicitly_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = LocalCostmap(Path(tmp))
            result = manager.update(vision={}, analysis={}, step=1)
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["result_type"], "local_costmap_no_depth")


if __name__ == "__main__":
    unittest.main()
