import json
import unittest

from scripts.exploration_context import build_exploration_context


def collect_keys(value):
    keys = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(collect_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(collect_keys(item))
    return keys


class ExplorationContextTests(unittest.TestCase):
    def test_detects_loop_and_exposes_frontier_hints_without_full_map(self) -> None:
        position_map = {
            "pose": {
                "cell": "0,0",
                "heading": "north",
                "pose_confidence": 0.91,
                "position_uncertainty_cells": 0.2,
            },
            "cells": {
                "0,0": {"state": "free", "visited": True, "seen_count": 7},
                "0,1": {"state": "free", "visited": True, "seen_count": 4},
                "1,0": {"state": "unknown", "visited": False},
                "-1,0": {"state": "unknown", "visited": False},
            },
            "frontiers": ["1,0", "-1,0"],
            "recent_actions": [
                "RotateLeft",
                "RotateRight",
                "RotateLeft",
                "RotateRight",
                "RotateLeft",
                "RotateRight",
            ],
            "stats": {"visited_cell_count": 2, "collision_count": 0},
        }
        costmap = {
            "status": "success",
            "action_safety": {
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                "MoveBack": {"safe": False, "reason": "unknown_swept_volume"},
            },
        }

        context = build_exploration_context(
            position_map=position_map,
            navigation_costmap=costmap,
            global_plan={"status": "success", "selected_goal_cell": "1,0", "next_action": "RotateRight"},
        )

        self.assertTrue(context["loop_warning"]["active"])
        self.assertEqual(context["loop_warning"]["severity"], "strong")
        self.assertGreaterEqual(context["revisit_counts"]["current_cell_recent_count"], 3)
        self.assertTrue(context["frontier_candidates"])
        self.assertEqual(context["frontier_candidates"][0]["cell"], "1,0")
        self.assertEqual(context["frontier_candidates"][0]["first_action_hint"], "RotateRight")
        self.assertEqual(context["action_effects"]["MoveAhead"]["effect"], "enters_visited_cell")
        self.assertTrue(context["action_effects"]["MoveAhead"]["enters_visited_cell"])
        self.assertTrue(context["action_effects"]["RotateRight"]["faces_unvisited_area"])
        self.assertTrue(context["action_effects"]["RotateRight"]["facing_frontier"])
        avoid_ids = {item["action"] for item in context["avoid_actions"]}
        self.assertIn("MoveAhead", avoid_ids)
        self.assertIn("MoveBack", avoid_ids)
        self.assertNotIn("cells", collect_keys(context))
        self.assertLess(len(json.dumps(context, ensure_ascii=False)), 12000)

    def test_frontier_hint_normalizes_lateral_planner_step_to_rotation(self) -> None:
        position_map = {
            "pose": {"cell": "0,0", "heading": "north"},
            "cells": {
                "0,0": {"state": "free", "visited": True},
                "-1,0": {"state": "unknown", "visited": False},
            },
            "frontiers": ["-1,0"],
            "recent_actions": [],
            "stats": {"visited_cell_count": 1, "collision_count": 0},
        }

        context = build_exploration_context(
            position_map=position_map,
            navigation_costmap={},
            global_plan={"selected_goal_cell": "-1,0", "next_action": "MoveLeft"},
        )

        self.assertEqual(context["frontier_candidates"][0]["cell"], "-1,0")
        self.assertEqual(context["frontier_candidates"][0]["first_action_hint"], "RotateLeft")
        self.assertEqual(context["frontier_candidates"][0]["first_action_policy"], "rotate_or_forward_only")

    def test_dead_end_summary_marks_all_body_actions_blocked(self) -> None:
        position_map = {
            "pose": {"cell": "1,4", "heading": "east"},
            "cells": {"1,4": {"state": "free", "visited": True}},
            "frontiers": ["1,5"],
            "recent_actions": ["MoveAhead"],
            "stats": {"visited_cell_count": 6, "collision_count": 0},
        }
        costmap = {
            "status": "success",
            "action_safety": {
                "MoveAhead": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "MoveBack": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "MoveLeft": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "MoveRight": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "RotateLeft": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "RotateRight": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                "LookDown": {"safe": True, "reason": "camera_pitch_action"},
            },
        }

        context = build_exploration_context(
            position_map=position_map,
            navigation_costmap=costmap,
            global_plan={},
        )

        self.assertTrue(context["dead_end"]["active"])
        self.assertTrue(context["dead_end"]["all_body_actions_blocked"])
        self.assertEqual(context["dead_end"]["suggested_backtrack_action"], "MoveBack")
        self.assertEqual(context["dead_end"]["suggested_backtrack_cell"], "0,4")
        self.assertEqual(context["dead_end"]["avoid_frontiers"][0]["cell"], "1,4")
        self.assertIn("LookDown", context["dead_end"]["safe_camera_actions"])

    def test_camera_posture_requires_lookdown_after_lookup(self) -> None:
        position_map = {
            "pose": {"cell": "0,0", "heading": "north"},
            "cells": {"0,0": {"state": "free", "visited": True}},
            "frontiers": ["0,1"],
            "recent_actions": ["MoveAhead", "LookUp", "RotateRight"],
            "stats": {"visited_cell_count": 1, "collision_count": 0},
        }

        context = build_exploration_context(
            position_map=position_map,
            navigation_costmap={},
            global_plan={},
        )

        self.assertTrue(context["camera_posture"]["needs_normalization"])
        self.assertEqual(context["camera_posture"]["normalize_action"], "LookDown")
        self.assertEqual(context["camera_posture"]["last_camera_action"], "LookUp")

    def test_frontier_candidate_penalizes_first_step_into_visited_cell(self) -> None:
        position_map = {
            "pose": {"cell": "1,3", "heading": "west"},
            "cells": {
                "1,3": {"state": "free", "visited": True},
                "0,3": {"state": "free", "visited": True},
                "-1,3": {"state": "unknown", "visited": False},
                "-1,4": {"state": "unknown", "visited": False},
            },
            "frontiers": ["-1,3", "-1,4"],
            "recent_actions": ["MoveBack"],
            "stats": {"visited_cell_count": 2, "collision_count": 0},
        }

        context = build_exploration_context(
            position_map=position_map,
            navigation_costmap={},
            global_plan={},
        )
        by_cell = {item["cell"]: item for item in context["frontier_candidates"]}

        self.assertEqual(by_cell["-1,3"]["first_step_action"], "MoveAhead")
        self.assertEqual(by_cell["-1,3"]["first_step_target_cell"], "0,3")
        self.assertTrue(by_cell["-1,3"]["first_step_enters_visited_cell"])
        self.assertIn(by_cell["-1,3"]["route_risk"], {"visited_first_step", "backtrack_first_step"})
        self.assertIn("first_step", " ".join(by_cell["-1,3"]["reasons"]))


if __name__ == "__main__":
    unittest.main()
