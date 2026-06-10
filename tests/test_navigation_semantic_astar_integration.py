from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.navigation_memory_core import NavigationMemory


class NavigationSemanticAStarIntegrationTests(unittest.TestCase):
    def test_object_target_navigation_uses_astar_not_legacy_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            data = navigation.position_map.load()
            for cell in ["0,0", "0,1", "1,1", "2,1", "2,0"]:
                navigation.position_map.mark_free(data, cell, evidence="test", visited=True)
            navigation.position_map.mark_occupied(data, "1,0", evidence="wall")
            navigation.position_map.save(data)
            navigation.semantic_map.sync_position_layer()
            result = navigation.recommend(
                vision={},
                analysis={"open_directions": ["forward", "left", "right"], "obstacle_ahead": False},
                target_cell="2,0",
                target_reason="object_memory_target",
                target_track_id="objtrk:apple:0001",
                goal_type="pickup_target",
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["planner"], "astar")
            self.assertEqual(recommendation["plan_status"], "success")
            self.assertNotIn("heuristic", recommendation["reason"])
            self.assertNotIn("1,0", recommendation["path"])

    def test_astar_translation_is_vetoed_by_local_costmap_and_turns_before_side_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            data = navigation.position_map.load()
            navigation.position_map.mark_free(data, "0,0", evidence="test", visited=True)
            navigation.position_map.mark_free(data, "0,1", evidence="test", visited=True)
            navigation.position_map._set_pose(data, cell="0,0", heading="north", source="test", step=1)
            navigation.position_map.save(data)
            navigation.semantic_map.sync_position_layer()
            result = navigation.recommend(
                vision={},
                analysis={
                    "open_directions": ["forward", "left", "right"],
                    "obstacle_ahead": False,
                    "local_costmap": {
                        "status": "success",
                        "action_safety": {
                            "MoveAhead": {"safe": False, "confidence": 1.0},
                            "MoveLeft": {"safe": True, "confidence": 0.8},
                            "MoveRight": {"safe": True, "confidence": 0.8},
                            "RotateLeft": {"safe": True, "confidence": 0.8},
                            "RotateRight": {"safe": True, "confidence": 0.8},
                            "MoveBack": {"safe": True, "confidence": 0.2},
                        },
                    },
                },
                target_cell="0,1",
                target_reason="object_memory_target",
                target_track_id="objtrk:apple:0001",
                goal_type="pickup_target",
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["planner"], "astar")
            self.assertEqual(recommendation["action"], "RotateLeft")
            self.assertIn("local_costmap_rejected_moveahead", recommendation["reason"])

    def test_semantic_observation_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            result = navigation.observe_semantics(
                analysis={
                    "best_pickup_candidate": {
                        "label": "apple",
                        "raw_label": "Apple",
                        "task_semantic_class": "pickup_target",
                        "confidence": 0.9,
                        "bearing_deg": 0.0,
                        "distance_m": 0.5,
                        "ground_distance_m": 0.5,
                        "position_hint": "front-center",
                    }
                },
                step=1,
            )
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["stats"]["semantic_cell_count"], 0)
            self.assertTrue((memory_dir / "semantic-map.json").exists())

    def test_depth_costmap_safe_forward_overrides_yolo_obstacle_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            result = navigation.recommend(
                vision={},
                analysis={
                    "open_directions": ["left", "right"],
                    "obstacle_ahead": True,
                    "local_costmap": {
                        "status": "success",
                        "action_safety": {
                            "MoveAhead": {
                                "safe": True,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "min_observed_ratio": 0.15,
                                "reason": "clear_swept_volume",
                            },
                            "MoveLeft": {"safe": True, "confidence": 0.8, "observed_ratio": 0.8},
                            "MoveRight": {"safe": True, "confidence": 0.8, "observed_ratio": 0.8},
                        },
                    },
                },
                persist=True,
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["action"], "MoveAhead")
            self.assertIn("local_forward_progress", recommendation["reason"])
            self.assertIn("0,0->0,1", result["known_open_edges"])
            self.assertNotIn("0,0->0,1", result["blocked_edges"])

    def test_one_sided_front_blocker_uses_visible_corner_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            result = navigation.recommend(
                vision={},
                analysis={
                    "open_directions": ["left", "right"],
                    "obstacle_ahead": True,
                    "local_costmap": {
                        "status": "success",
                        "action_safety": {
                            "MoveAhead": {
                                "safe": False,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "reason": "inflated_obstacle_in_swept_volume",
                            },
                            "MoveLeft": {
                                "safe": True,
                                "confidence": 0.8,
                                "observed_ratio": 1.0,
                                "min_observed_ratio": 0.15,
                                "reason": "clear_swept_volume",
                            },
                            "MoveRight": {
                                "safe": False,
                                "confidence": 0.8,
                                "observed_ratio": 1.0,
                                "reason": "inflated_obstacle_in_swept_volume",
                            },
                            "RotateLeft": {"safe": True, "confidence": 0.8},
                            "RotateRight": {"safe": True, "confidence": 0.8},
                        },
                        "front_corridor": {
                            "status": "success",
                            "dominant_blocker_side": "right",
                            "preferred_bypass_side": "left",
                            "asymmetric_bypass_available": True,
                            "lanes": {
                                "left": {
                                    "safe": True,
                                    "reason": "clear_lane",
                                    "observed_ratio": 1.0,
                                },
                                "center": {
                                    "safe": False,
                                    "reason": "inflated_obstacle_in_lane",
                                    "observed_ratio": 1.0,
                                },
                                "right": {
                                    "safe": False,
                                    "reason": "inflated_obstacle_in_lane",
                                    "observed_ratio": 1.0,
                                },
                            },
                        },
                    },
                },
                persist=True,
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["source"], "local_bootstrap")
            self.assertEqual(recommendation["action"], "MoveLeft")
            self.assertIn("local_front_corner_bypass", recommendation["reason"])
            self.assertNotEqual(recommendation["planner"], "astar")

    def test_active_frontier_goal_prevents_lateral_goal_flapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            data = navigation.position_map.load()
            navigation.position_map._set_pose(data, cell="0,-2", heading="west", source="test", step=1)
            for cell in ["0,-2", "0,-3"]:
                navigation.position_map.mark_free(data, cell, evidence="test", visited=True)
            navigation.position_map.save(data)
            room = navigation.load_room()
            room["last_cell"] = "0,-2"
            room["last_heading"] = "west"
            room["frontier_cells"] = ["1,-2", "1,-3"]
            room["active_frontier_goal"] = {
                "cell": "1,-2",
                "started_step": 1,
                "last_distance": 1,
                "stale_count": 0,
            }
            navigation.save_room(room)
            result = navigation.recommend(
                vision={},
                analysis={"open_directions": ["left", "right"], "obstacle_ahead": False},
                persist=True,
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["planner"], "astar")
            self.assertEqual(recommendation["selected_goal_cell"], "1,-2")
            self.assertEqual(recommendation["action"], "RotateRight")
            self.assertEqual(result["active_frontier_goal"]["cell"], "1,-2")

    def test_frontier_candidates_are_filtered_to_current_open_component(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            room = navigation.load_room()
            room["last_cell"] = "0,0"
            room["last_heading"] = "north"
            room["visited_cells"] = ["0,0", "0,1", "10,10"]
            room["known_open_edges"] = [
                "0,0->0,1",
                "0,1->0,0",
                "0,1->0,2",
                "0,2->0,1",
                "10,10->10,11",
                "10,11->10,10",
            ]
            room["blocked_edges"] = []

            frontiers = navigation._global_frontier_cells(room)
            self.assertIn("0,2", frontiers)
            self.assertNotIn("10,11", frontiers)

    def test_lateral_oscillation_cools_down_frontier_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            room = navigation.load_room()
            room["recent_navigation_actions"] = ["MoveLeft", "MoveRight", "MoveLeft"]
            room["last_frontier_target"] = "1,-2"
            room["active_frontier_goal"] = {
                "cell": "1,-2",
                "started_step": 1,
                "last_distance": 1,
                "stale_count": 0,
            }
            navigation.save_room(room)
            result = navigation.record_step(
                action="MoveRight",
                success=True,
                vision={},
                analysis={},
                recommendation={"selected_goal_cell": "1,-2", "requested_target_cell": "1,-2"},
            )
            self.assertIn("1,-2", result["frontier_cooldowns"])
            self.assertIsNone(result["active_frontier_goal"])
            self.assertGreaterEqual(result["frontier_oscillation_count"], 1)

    def test_forward_block_after_progress_uses_local_side_probe_before_rear_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            data = navigation.position_map.load()
            navigation.position_map._set_pose(data, cell="0,3", heading="north", source="test", step=3)
            for cell in ["0,0", "0,1", "0,2", "0,3"]:
                navigation.position_map.mark_free(data, cell, evidence="test", visited=True)
            navigation.position_map.save(data)
            room = navigation.load_room()
            room["last_cell"] = "0,3"
            room["last_heading"] = "north"
            room["recent_navigation_actions"] = ["MoveAhead", "MoveAhead", "MoveAhead"]
            room["frontier_cells"] = ["0,-1"]
            navigation.save_room(room)
            result = navigation.recommend(
                vision={},
                analysis={
                    "open_directions": ["left", "right"],
                    "obstacle_ahead": True,
                    "local_costmap": {
                        "status": "success",
                        "action_safety": {
                            "MoveAhead": {
                                "safe": False,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "reason": "inflated_obstacle_in_swept_volume",
                            },
                            "MoveLeft": {
                                "safe": True,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "reason": "clear_swept_volume",
                            },
                            "MoveRight": {
                                "safe": True,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "reason": "clear_swept_volume",
                            },
                        },
                    },
                },
                persist=True,
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["action"], "RotateLeft")
            self.assertIn("local_side_probe_after_forward_blocked", recommendation["reason"])
            self.assertNotEqual(recommendation.get("selected_goal_cell"), "0,-1")

    def test_local_side_explore_precedes_old_rear_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            navigation = NavigationMemory(memory_dir)
            navigation.reset()
            data = navigation.position_map.load()
            navigation.position_map._set_pose(data, cell="0,0", heading="north", source="test", step=3)
            for cell in ["0,0", "0,1", "0,2"]:
                navigation.position_map.mark_free(data, cell, evidence="test", visited=True)
            navigation.position_map.save(data)
            room = navigation.load_room()
            room["last_cell"] = "0,0"
            room["last_heading"] = "north"
            room["frontier_cells"] = ["0,-2"]
            room["recent_navigation_actions"] = ["MoveAhead", "MoveAhead", "MoveAhead"]
            navigation.save_room(room)
            result = navigation.recommend(
                vision={},
                analysis={
                    "open_directions": ["left", "forward", "right"],
                    "obstacle_ahead": False,
                    "local_costmap": {
                        "status": "success",
                        "action_safety": {
                            "MoveAhead": {
                                "safe": True,
                                "confidence": 1.0,
                                "observed_ratio": 1.0,
                                "reason": "clear_swept_volume",
                            },
                            "MoveLeft": {
                                "safe": True,
                                "confidence": 0.8,
                                "observed_ratio": 0.8,
                                "reason": "clear_swept_volume",
                            },
                            "MoveRight": {
                                "safe": True,
                                "confidence": 0.8,
                                "observed_ratio": 0.8,
                                "reason": "clear_swept_volume",
                            },
                            "RotateLeft": {
                                "safe": True,
                                "confidence": 0.8,
                                "observed_ratio": 0.8,
                                "reason": "clear_swept_volume",
                            },
                            "RotateRight": {
                                "safe": True,
                                "confidence": 0.8,
                                "observed_ratio": 0.8,
                                "reason": "clear_swept_volume",
                            },
                        },
                    },
                },
                persist=True,
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["source"], "local_bootstrap")
            self.assertEqual(recommendation["action"], "RotateLeft")
            self.assertIn("local_side_explore_before_global_frontier", recommendation["reason"])
            self.assertNotEqual(recommendation.get("selected_goal_cell"), "0,-2")


if __name__ == "__main__":
    unittest.main()
