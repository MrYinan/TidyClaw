import unittest

from scripts.exploration_context import build_exploration_context
from scripts.explore_planner import build_explore_plan


class ExplorePlannerTests(unittest.TestCase):
    def test_rotation_loop_generates_safe_translation_waypoint(self) -> None:
        position_map = {
            "pose": {"cell": "0,3", "heading": "south"},
            "cells": {
                "0,3": {"state": "free", "visited": True},
                "0,2": {"state": "free", "visited": True},
                "-1,3": {"state": "unknown", "visited": False},
                "-1,2": {"state": "unknown", "visited": False},
            },
            "frontiers": ["-1,3", "-1,2"],
            "recent_actions": [
                "LookDown",
                "RotateLeft",
                "RotateRight",
                "RotateLeft",
                "RotateRight",
                "RotateLeft",
            ],
            "stats": {"visited_cell_count": 4, "collision_count": 0},
        }
        costmap = {
            "action_safety": {
                "MoveAhead": {
                    "safe": True,
                    "reason": "clear_swept_volume",
                    "observed_ratio": 0.8,
                    "min_observed_ratio": 0.3,
                },
                "MoveLeft": {"safe": False, "reason": "unknown_swept_volume"},
                "MoveRight": {"safe": False, "reason": "unknown_swept_volume"},
                "MoveBack": {"safe": False, "reason": "unknown_swept_volume"},
                "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
            }
        }
        exploration = build_exploration_context(
            position_map=position_map,
            navigation_costmap=costmap,
            global_plan={},
        )

        plan = build_explore_plan(
            position_map=position_map,
            navigation_costmap=costmap,
            exploration=exploration,
        )

        self.assertEqual(plan["mode"], "break_rotation_loop")
        self.assertTrue(plan["waypoint_candidates"])
        self.assertEqual(plan["waypoint_candidates"][0]["action"], "MoveAhead")
        self.assertEqual(plan["waypoint_candidates"][0]["cell"], "0,2")
        self.assertIn("rotation_loop_active", plan["waypoint_candidates"][0]["reasons"])
        self.assertTrue(plan["suppressed_frontiers"])
        self.assertTrue(
            any(
                "safe_translation_waypoint_available" in item["reason"]
                for item in plan["suppressed_frontiers"]
            )
        )

    def test_rotation_loop_without_safe_translation_does_not_force_waypoint(self) -> None:
        position_map = {
            "pose": {"cell": "0,3", "heading": "west"},
            "cells": {
                "0,3": {"state": "free", "visited": True},
                "-1,3": {"state": "unknown", "visited": False},
                "0,2": {"state": "free", "visited": True},
            },
            "frontiers": ["-1,3", "0,2"],
            "recent_actions": ["RotateRight", "RotateLeft", "RotateRight", "RotateLeft", "RotateRight"],
            "stats": {"visited_cell_count": 4, "collision_count": 0},
        }
        costmap = {
            "action_safety": {
                "MoveAhead": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "MoveLeft": {"safe": False, "reason": "unknown_swept_volume"},
                "MoveRight": {"safe": False, "reason": "unknown_swept_volume"},
                "MoveBack": {"safe": False, "reason": "unknown_swept_volume"},
                "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
            }
        }
        exploration = build_exploration_context(
            position_map=position_map,
            navigation_costmap=costmap,
            global_plan={},
        )

        plan = build_explore_plan(
            position_map=position_map,
            navigation_costmap=costmap,
            exploration=exploration,
        )

        self.assertEqual(plan["mode"], "rotation_loop_scan_limited")
        self.assertNotIn("waypoint_candidates", plan)
        self.assertEqual(plan["recovery_actions"][0]["action"], "LookUp")
        self.assertTrue(plan["suppressed_frontiers"])
        self.assertTrue(
            any("immediate_rotation_undo" in item["reason"] for item in plan["suppressed_frontiers"])
        )


if __name__ == "__main__":
    unittest.main()
