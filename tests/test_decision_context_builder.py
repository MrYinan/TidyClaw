import argparse
import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.decision_context_builder import (
    DECISION_CONTEXT_SCHEMA,
    LoadedJson,
    build_explore_frontier_options,
    build_explore_route_step_options,
    build_explore_waypoint_options,
    build_consistency_warnings,
    build_context,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def loaded_source(name: str, last_modified: str, data: dict | None = None) -> LoadedJson:
    return LoadedJson(
        path=Path(f"memory/{name}.json"),
        data=data or {},
        info={"loaded": True, "last_modified": last_modified},
    )


class DecisionContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_env = patch.dict(os.environ, {"ROBOT_MAP_BACKEND": "action_odometry"}, clear=False)
        self._backend_env.start()

    def tearDown(self) -> None:
        self._backend_env.stop()

    def build_args(self, memory_dir: Path, perception_json: Path) -> argparse.Namespace:
        return argparse.Namespace(
            memory_dir=str(memory_dir),
            perception_json=str(perception_json),
            output="",
            max_candidates=3,
            max_options=8,
            task_mode="tidy",
            format="compact",
        )

    def test_builds_bounded_public_context(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-test-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 3})
            write_json(
                memory / "service-task-state.json",
                {
                    "phase": "SEARCH_PICKUP_TARGET",
                    "holding_object": False,
                    "pickup_surface_policy": "floor-only",
                },
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "north"},
                    "frontiers": ["0,1"],
                    "cells": {"0,0": {"state": "free"}},
                },
            )
            write_json(
                memory / "object-memory.json",
                {
                    "tracks": {
                        "objtrk:apple:0001": {
                            "track_id": "objtrk:apple:0001",
                            "label": "apple",
                            "task_class": "pickup_target",
                            "status": "unpicked",
                            "confidence": 0.9,
                            "interaction": {"pickup_goal_eligible": True},
                        }
                    },
                    "history": [{"large": "not agent facing"}],
                },
            )
            write_json(memory / "global-plan.json", {"status": "success", "next_action": "MoveAhead"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": True,
                    "frontier_exists": True,
                    "recommended_action": "MoveAhead",
                    "best_pickup_candidate": {
                        "label": "apple",
                        "task_semantic_class": "pickup_target",
                        "confidence": 0.9,
                        "reachable": True,
                        "pickup_now": True,
                    },
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(context["schema"], DECISION_CONTEXT_SCHEMA)
        self.assertTrue(context["forbidden_private_fields_absent"])
        self.assertTrue(context["context_policy"]["full_maps_not_included"])
        self.assertEqual(context["navigation"]["map_backend"]["backend"], "action_odometry")
        self.assertEqual(
            context["navigation"]["map_backend"]["schema"],
            "robot_cleaner_map_snapshot_v1",
        )
        self.assertIn("navigation_core", context)
        self.assertEqual(context["navigation_core"]["schema"], "robot_cleaner_navigation_core_v1")
        self.assertNotIn('"cells"', json.dumps(context["navigation"]))
        self.assertIn("exploration", context)
        self.assertIn("recent_path", context["exploration"])
        self.assertIn("frontier_candidates", context["exploration"])
        self.assertNotIn('"cells"', json.dumps(context["exploration"]))
        self.assertNotIn("history", json.dumps(context["worklist"]))
        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertIn("pick:best_pickup_candidate_0", option_ids)
        self.assertIn("explore:frontier_cluster:x0_z1", option_ids)
        self.assertIn("explore:frontier:x0_z1", option_ids)
        self.assertIn("move:moveahead", option_ids)
        self.assertIn("primary_options", context["option_set"])
        self.assertIn("fallback_options", context["option_set"])
        self.assertIn("goal_options", context["option_set"])
        self.assertIn("explore:frontier_cluster:x0_z1", context["option_set"]["primary_options"])
        self.assertIn("explore:frontier:x0_z1", context["option_set"]["fallback_options"])
        self.assertIn("move:moveahead", context["option_set"]["fallback_options"])
        cluster = next(
            item for item in context["option_set"]["options"] if item["option_id"] == "explore:frontier_cluster:x0_z1"
        )
        self.assertEqual(cluster["kind"], "explore_frontier_cluster")
        self.assertEqual(cluster["decision_level"], "goal")
        self.assertEqual(cluster["llm_priority"], "primary")
        self.assertEqual(cluster["frontier_cluster_target"]["cell"], "0,1")
        explore = next(item for item in context["option_set"]["options"] if item["option_id"] == "explore:frontier:x0_z1")
        self.assertEqual(explore["kind"], "explore_frontier")
        self.assertEqual(explore["decision_level"], "goal")
        self.assertEqual(explore["llm_priority"], "fallback")
        self.assertTrue(explore["fallback_only"])
        self.assertTrue(explore["one_step_only"])
        self.assertEqual(explore["action"], "MoveAhead")
        self.assertEqual(explore["resolved_step_option_id"], "move:moveahead")
        self.assertEqual(explore["frontier_target"]["cell"], "0,1")
        moveahead = next(item for item in context["option_set"]["options"] if item["option_id"] == "move:moveahead")
        self.assertEqual(moveahead["decision_level"], "motor")
        self.assertEqual(moveahead["llm_priority"], "fallback")
        self.assertTrue(moveahead["fallback_only"])
        self.assertEqual(moveahead["exploration_effect"]["target_cell"], "0,1")
        self.assertTrue(moveahead["exploration_effect"]["enters_frontier"])
        self.assertEqual(moveahead["exploration_effect"]["effect"], "enters_frontier")
        rotateleft = next(item for item in context["option_set"]["options"] if item["option_id"] == "move:rotateleft")
        self.assertTrue(rotateleft["exploration_effect"]["faces_unvisited_area"])
        self.assertTrue(context["option_set"]["selection_contract"]["model_must_choose_from_options"])
        self.assertTrue(context["option_set"]["selection_contract"]["rule_baseline_is_not_instruction"])
        self.assertEqual(context["option_set"]["selection_contract"]["llm_should_choose_from"], "primary_options_first")
        self.assertIn("rule_baseline_option_id", context["option_set"])
        self.assertNotIn("recommended_option_id", context["option_set"])

    def test_low_confidence_translation_move_is_not_exposed(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-low-confidence-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 4})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveLeft": {
                            "safe": True,
                            "reason": "clear_swept_volume",
                            "observed_ratio": 0.2,
                            "min_observed_ratio": 0.5,
                        },
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "east"},
                    "frontiers": ["0,1"],
                    "cells": {"0,0": {"state": "free"}, "0,1": {"state": "unknown"}},
                },
            )
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertNotIn("move:moveleft", option_ids)
        self.assertIn(
            {
                "action": "MoveLeft",
                "observed_ratio": 0.2,
                "min_observed_ratio": 0.5,
                "reason": "low_observed_swept_volume",
                "costmap_reason": "clear_swept_volume",
            },
            context["option_set"]["low_confidence_moves"],
        )

    def test_inspection_waypoint_options_are_primary_and_legacy_frontier_is_fallback(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-inspection-waypoint-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(
                memory / "room-state.json",
                {
                    "room_name": "current_room",
                    "room_complete": False,
                    "inspection_waypoints": [
                        {"waypoint_id": "wp_front", "cell": "0,1", "label": "front scan"},
                        {"waypoint_id": "wp_side", "cell": "1,0", "label": "side scan"},
                    ],
                },
            )
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 5})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "north"},
                    "frontiers": ["0,1"],
                    "cells": {
                        "0,0": {"state": "free", "visited": True},
                        "0,1": {"state": "free"},
                        "1,0": {"state": "free"},
                    },
                },
            )
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": ["forward"],
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_set = context["option_set"]
        option_ids = [item["option_id"] for item in option_set["options"]]
        self.assertIn("explore:inspection_waypoint:wp_front", option_ids)
        self.assertIn("explore:inspection_waypoint:wp_side", option_ids)
        self.assertIn("explore:inspection_waypoint:wp_front", option_set["primary_options"])
        self.assertIn("explore:inspection_waypoint:wp_front", option_set["explore_inspection_waypoint_options"])
        cluster = next(
            item for item in option_set["options"] if item["option_id"] == "explore:frontier_cluster:x0_z1"
        )
        self.assertEqual(cluster["llm_priority"], "fallback")
        self.assertTrue(cluster["fallback_only"])
        self.assertIn("explore:frontier_cluster:x0_z1", option_set["fallback_options"])
        self.assertEqual(context["coverage_waypoints"]["required_waypoint_count"], 2)
        self.assertEqual(context["coverage_waypoints"]["pending_waypoint_count"], 2)

    def test_active_inspection_waypoint_exposes_continue_option(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-active-waypoint-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(
                memory / "room-state.json",
                {
                    "room_name": "current_room",
                    "room_complete": False,
                    "inspection_waypoints": [
                        {"waypoint_id": "wp_front", "cell": "0,1", "label": "front scan"},
                        {"waypoint_id": "wp_side", "cell": "1,0", "label": "side scan"},
                    ],
                    "coverage_waypoints": {
                        "active_waypoint_goal": {
                            "waypoint_id": "wp_front",
                            "cell": "0,1",
                            "status": "active",
                        }
                    },
                },
            )
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 5})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "north"},
                    "frontiers": ["0,1"],
                    "cells": {
                        "0,0": {"state": "free", "visited": True},
                        "0,1": {"state": "free"},
                        "1,0": {"state": "free"},
                    },
                },
            )
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": ["forward"],
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_set = context["option_set"]
        option = next(item for item in option_set["options"] if item["option_id"] == "continue:active_waypoint_goal")
        self.assertEqual(option["kind"], "continue_active_waypoint_goal")
        self.assertEqual(option["active_waypoint_goal"]["waypoint_id"], "wp_front")
        self.assertIn("continue:active_waypoint_goal", option_set["primary_options"])
        self.assertIn("continue:active_waypoint_goal", option_set["continue_active_waypoint_options"])

    def test_explore_frontier_can_be_synthesized_from_safe_move_effect(self) -> None:
        options = build_explore_frontier_options(
            exploration={"frontier_candidates": [], "action_effects": {}},
            move_options=[
                {
                    "option_id": "move:moveright",
                    "kind": "move_action",
                    "action": "MoveRight",
                    "executable_now": True,
                    "physical_action": True,
                    "safety_source": "navigation-costmap",
                    "exploration_effect": {
                        "action": "MoveRight",
                        "effect": "enters_frontier",
                        "enters_frontier": True,
                        "target_cell": "1,0",
                        "target_state": "unknown",
                    },
                }
            ],
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["option_id"], "explore:frontier:x1_z0")
        self.assertEqual(options[0]["resolved_step_option_id"], "move:moveright")
        self.assertEqual(options[0]["frontier_target"]["cell"], "1,0")
        self.assertEqual(options[0]["frontier_target"]["source"], "move_exploration_effect")
        self.assertEqual(options[0]["llm_priority"], "primary")

    def test_explore_waypoint_options_use_planner_candidates(self) -> None:
        options = build_explore_waypoint_options(
            explore_plan={
                "mode": "break_rotation_loop",
                "waypoint_candidates": [
                    {
                        "cell": "0,2",
                        "action": "MoveAhead",
                        "purpose": "break_rotation_loop_via_safe_translation",
                        "score": 8.2,
                        "target_state": "free",
                        "target_visited": True,
                        "target_recent": True,
                        "reasons": ["break_rotation_loop_via_safe_translation", "rotation_loop_active"],
                    }
                ],
            },
            move_options=[
                {
                    "option_id": "move:moveahead",
                    "kind": "move_action",
                    "action": "MoveAhead",
                    "executable_now": True,
                    "physical_action": True,
                    "safety_source": "navigation-costmap",
                    "exploration_effect": {"target_cell": "0,2", "effect": "enters_visited_cell"},
                }
            ],
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["option_id"], "explore:waypoint:x0_z2")
        self.assertEqual(options[0]["kind"], "explore_waypoint")
        self.assertEqual(options[0]["resolved_step_option_id"], "move:moveahead")
        self.assertEqual(options[0]["waypoint_target"]["cell"], "0,2")
        self.assertEqual(options[0]["decision_level"], "goal")
        self.assertEqual(options[0]["llm_priority"], "primary")

    def test_explore_route_step_options_use_active_route(self) -> None:
        options = build_explore_route_step_options(
            explore_plan={
                "active_route": {
                    "status": "active",
                    "route_id": "route-frontier-xm1-z3-abcd1234",
                    "goal_cell": "-1,3",
                    "goal_type": "frontier_cluster",
                    "path_length": 2,
                    "distance_to_goal": 2,
                    "route_step": {
                        "status": "active",
                        "route_id": "route-frontier-xm1-z3-abcd1234",
                        "step_index": 0,
                        "action": "RotateLeft",
                        "current_cell": "0,4",
                        "current_heading": "north",
                        "target_cell": "0,4",
                        "next_cell": "0,3",
                        "goal_cell": "-1,3",
                        "desired_heading": "south",
                        "progress_effect": "turnaround_toward_next_cell",
                    },
                }
            },
            move_options=[
                {
                    "option_id": "move:rotateleft",
                    "kind": "move_action",
                    "action": "RotateLeft",
                    "executable_now": True,
                    "physical_action": True,
                    "safety_source": "navigation-costmap",
                }
            ],
        )

        self.assertEqual(len(options), 1)
        self.assertTrue(options[0]["option_id"].startswith("explore:route_step:"))
        self.assertEqual(options[0]["kind"], "explore_route_step")
        self.assertEqual(options[0]["action"], "RotateLeft")
        self.assertEqual(options[0]["resolved_step_option_id"], "move:rotateleft")
        self.assertEqual(options[0]["route_step"]["next_cell"], "0,3")
        self.assertEqual(options[0]["route_step"]["goal_cell"], "-1,3")
        self.assertEqual(options[0]["llm_priority"], "primary")

    def test_build_context_exposes_waypoint_before_rotation_frontier_in_loop(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-waypoint-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 12})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {
                            "safe": True,
                            "reason": "clear_swept_volume",
                            "observed_ratio": 0.85,
                            "min_observed_ratio": 0.3,
                        },
                        "MoveLeft": {"safe": False, "reason": "unknown_swept_volume"},
                        "MoveRight": {"safe": False, "reason": "unknown_swept_volume"},
                        "MoveBack": {"safe": False, "reason": "unknown_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,3", "heading": "south"},
                    "frontiers": ["-1,3", "-1,2"],
                    "cells": {
                        "0,3": {"state": "free", "visited": True},
                        "0,2": {"state": "free", "visited": True},
                        "-1,3": {"state": "unknown", "visited": False},
                        "-1,2": {"state": "unknown", "visited": False},
                    },
                    "recent_actions": [
                        "MoveAhead",
                        "RotateLeft",
                        "RotateRight",
                        "RotateLeft",
                        "RotateRight",
                        "RotateLeft",
                    ],
                    "stats": {"visited_cell_count": 4, "collision_count": 0},
                },
            )
            write_json(memory / "global-plan.json", {"status": "success"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": ["forward"],
                    "obstacle_ahead": False,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertIn("explore:waypoint:x0_z2", option_ids)
        self.assertIn("explore_plan", context)
        self.assertEqual(context["explore_plan"]["mode"], "break_rotation_loop")
        self.assertIn("explore:frontier_cluster:xm1_z3", context["option_set"]["primary_options"])
        self.assertIn("explore:waypoint:x0_z2", context["option_set"]["fallback_options"])
        self.assertIn("explore:waypoint:x0_z2", context["option_set"]["explore_waypoint_options"])
        waypoint_index = option_ids.index("explore:waypoint:x0_z2")
        move_index = option_ids.index("move:moveahead")
        self.assertLess(waypoint_index, move_index)

    def test_build_context_exposes_frontier_cluster_before_committed_route_step(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-route-step-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(
                memory / "room-state.json",
                {
                    "room_name": "current_room",
                    "room_complete": False,
                    "active_frontier_goal": {
                        "schema": "robot_cleaner_active_frontier_goal_v1",
                        "status": "active",
                        "mode": "frontier_cluster",
                        "cell": "-1,3",
                        "path": ["0,4", "0,3", "-1,3"],
                    },
                    "active_route": {
                        "schema": "robot_cleaner_active_route_v1",
                        "status": "active",
                        "route_id": "route-frontier-xm1-z3-abcd1234",
                        "goal_cell": "-1,3",
                        "goal_type": "frontier_cluster",
                        "current_cell": "0,4",
                        "current_heading": "north",
                        "next_cell": "0,3",
                        "next_action": "RotateLeft",
                        "path": ["0,4", "0,3", "-1,3"],
                        "route_step": {
                            "schema": "robot_cleaner_route_step_v1",
                            "status": "active",
                            "route_id": "route-frontier-xm1-z3-abcd1234",
                            "step_index": 0,
                            "action": "RotateLeft",
                            "current_cell": "0,4",
                            "current_heading": "north",
                            "target_cell": "0,4",
                            "next_cell": "0,3",
                            "goal_cell": "-1,3",
                            "desired_heading": "south",
                            "progress_effect": "turnaround_toward_next_cell",
                        },
                    },
                },
            )
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 18})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,4", "heading": "north"},
                    "frontiers": ["-1,3"],
                    "cells": {
                        "0,4": {"state": "free", "visited": True},
                        "0,3": {"state": "free", "visited": True},
                        "-1,3": {"state": "unknown", "visited": False},
                    },
                    "stats": {"visited_cell_count": 2, "collision_count": 0},
                },
            )
            write_json(memory / "global-plan.json", {"status": "success"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": ["left", "right"],
                    "obstacle_ahead": False,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        cluster_id = "explore:frontier_cluster:xm1_z3"
        self.assertIn(cluster_id, option_ids)
        route_ids = [item for item in option_ids if item.startswith("explore:route_step:")]
        self.assertEqual(len(route_ids), 1)
        route_id = route_ids[0]
        self.assertEqual(context["navigation"]["active_route"]["goal_cell"], "-1,3")
        self.assertEqual(context["explore_plan"]["mode"], "committed_route")
        self.assertIn(cluster_id, context["option_set"]["primary_options"])
        self.assertIn(cluster_id, context["option_set"]["explore_frontier_cluster_options"])
        self.assertIn(route_id, context["option_set"]["fallback_options"])
        self.assertIn(route_id, context["option_set"]["explore_route_options"])
        cluster_option = next(item for item in context["option_set"]["options"] if item["option_id"] == cluster_id)
        self.assertEqual(cluster_option["kind"], "explore_frontier_cluster")
        self.assertEqual(cluster_option["action"], "RotateLeft")
        self.assertEqual(cluster_option["frontier_cluster_target"]["cell"], "-1,3")
        self.assertEqual(cluster_option["route_step"]["next_cell"], "0,3")
        route_option = next(item for item in context["option_set"]["options"] if item["option_id"] == route_id)
        self.assertEqual(route_option["kind"], "explore_route_step")
        self.assertTrue(route_option["fallback_only"])
        self.assertEqual(route_option["action"], "RotateLeft")
        self.assertEqual(route_option["route_step"]["next_cell"], "0,3")

    def test_build_context_prioritizes_recovery_when_route_blocked_and_camera_low(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-route-blocked-recovery-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(
                memory / "room-state.json",
                {
                    "room_name": "current_room",
                    "room_complete": False,
                    "active_frontier_goal": {
                        "schema": "robot_cleaner_active_frontier_goal_v1",
                        "status": "active",
                        "mode": "frontier_cluster",
                        "cell": "2,-4",
                        "path": ["1,-4", "2,-4"],
                    },
                    "active_route": {
                        "schema": "robot_cleaner_active_route_v1",
                        "status": "blocked",
                        "route_id": "route-frontier-x2-zm4-abcd1234",
                        "goal_cell": "2,-4",
                        "next_action": "MoveAhead",
                        "blocked_reason": "inflated_obstacle_in_swept_volume",
                    },
                },
            )
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 22})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                        "MoveRight": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 1.0},
                        "MoveLeft": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 1.0},
                        "MoveBack": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 1.0},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                        "LookUp": {"safe": True, "reason": "camera_pitch_action"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "1,-4", "heading": "east"},
                    "frontiers": ["1,-5", "2,-4"],
                    "cells": {
                        "1,-4": {"state": "free", "visited": True},
                        "1,-5": {"state": "unknown", "visited": False},
                        "2,-4": {"state": "unknown", "visited": False},
                    },
                    "recent_actions": ["MoveRight", "MoveRight", "LookDown"],
                    "stats": {"visited_cell_count": 8, "collision_count": 0},
                },
            )
            write_json(memory / "global-plan.json", {"status": "success"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": ["left", "right"],
                    "obstacle_ahead": False,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(context["explore_plan"]["mode"], "route_blocked_recovery")
        self.assertEqual(context["exploration"]["camera_posture"]["normalize_action"], "LookUp")
        self.assertIn("recover:rotateleft", context["option_set"]["recovery_options"])
        self.assertIn("recover:rotateright", context["option_set"]["recovery_options"])
        self.assertNotIn("recover:lookup", context["option_set"]["recovery_options"])
        self.assertEqual(context["option_set"]["primary_options"][0], "recover:rotateleft")
        waypoint_actions = [
            item.get("action")
            for item in context["option_set"]["options"]
            if item.get("kind") == "explore_waypoint"
        ]
        self.assertNotIn("MoveRight", waypoint_actions)

    def test_rotation_loop_without_translation_exposes_planner_recovery(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-planner-recovery-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 15})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": False, "reason": "inflated_obstacle_in_swept_volume"},
                        "MoveLeft": {"safe": False, "reason": "unknown_swept_volume"},
                        "MoveRight": {"safe": False, "reason": "unknown_swept_volume"},
                        "MoveBack": {"safe": False, "reason": "unknown_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,3", "heading": "west"},
                    "frontiers": ["-1,3", "0,2"],
                    "cells": {
                        "0,3": {"state": "free", "visited": True},
                        "-1,3": {"state": "unknown", "visited": False},
                        "0,2": {"state": "free", "visited": True},
                    },
                    "recent_actions": ["RotateRight", "RotateLeft", "RotateRight", "RotateLeft", "RotateRight"],
                    "stats": {"visited_cell_count": 4, "collision_count": 0},
                },
            )
            write_json(memory / "global-plan.json", {"status": "success"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "open_directions": [],
                    "obstacle_ahead": True,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertEqual(context["explore_plan"]["mode"], "rotation_loop_scan_limited")
        self.assertIn("recover:rotateleft", option_ids)
        self.assertIn("recover:rotateright", option_ids)
        self.assertNotIn("recover:lookup", option_ids)
        self.assertIn("recover:rotateleft", context["option_set"]["recovery_options"])
        self.assertEqual(context["option_set"]["primary_options"][0], "recover:rotateleft")

    def test_dead_end_context_exposes_recovery_options(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-dead-end-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 11})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            blocked = {"safe": False, "reason": "inflated_obstacle_in_swept_volume", "observed_ratio": 1.0}
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": dict(blocked),
                        "MoveBack": dict(blocked),
                        "MoveLeft": dict(blocked),
                        "MoveRight": dict(blocked),
                        "RotateLeft": dict(blocked),
                        "RotateRight": dict(blocked),
                        "LookDown": {"safe": True, "reason": "camera_pitch_action"},
                        "LookUp": {"safe": True, "reason": "camera_pitch_action"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "1,4", "heading": "east"},
                    "frontiers": ["1,5"],
                    "cells": {"1,4": {"state": "free", "visited": True}, "1,5": {"state": "unknown"}},
                    "recent_actions": ["MoveAhead"],
                    "stats": {"visited_cell_count": 6, "collision_count": 0},
                },
            )
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "obstacle_ahead": True,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertTrue(context["exploration"]["dead_end"]["active"])
        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertIn("recover:lookdown", option_ids)
        self.assertIn("recover:lookup", option_ids)
        self.assertIn("recover:lookdown", context["option_set"]["primary_options"])
        self.assertIn("recover:lookdown", context["option_set"]["recovery_options"])
        self.assertEqual(context["option_set"]["rule_baseline_option_id"], "recover:lookdown")
        recover = next(item for item in context["option_set"]["options"] if item["option_id"] == "recover:lookdown")
        self.assertEqual(recover["kind"], "recovery_action")
        self.assertEqual(recover["action"], "LookDown")
        self.assertEqual(recover["dead_end_context"]["current_cell"], "1,4")

    def test_camera_posture_normalization_exposes_recover_lookdown(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-camera-posture-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 12})
            write_json(
                memory / "service-task-state.json",
                {"phase": "SEARCH_PICKUP_TARGET", "holding_object": False, "pickup_surface_policy": "floor-only"},
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 1.0},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                        "LookDown": {"safe": True, "reason": "camera_pitch_action"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "1,4", "heading": "east"},
                    "frontiers": ["1,5"],
                    "cells": {"1,4": {"state": "free", "visited": True}, "1,5": {"state": "unknown"}},
                    "recent_actions": ["LookUp", "RotateRight"],
                    "stats": {"visited_cell_count": 6, "collision_count": 0},
                },
            )
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "pickup_target_detected": False,
                    "frontier_exists": True,
                    "obstacle_ahead": False,
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertTrue(context["exploration"]["camera_posture"]["needs_normalization"])
        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertNotIn("recover:lookdown", option_ids)
        self.assertNotIn("recover:lookdown", context["option_set"].get("recovery_options", []))
        self.assertTrue(
            any(item.get("kind") in {"explore_frontier_cluster", "move_action"} for item in context["option_set"]["options"])
        )

    def test_fresh_perception_with_unchanged_mission_state_skew_is_advisory(self) -> None:
        warnings = build_consistency_warnings(
            service_state={"holding_object": False},
            perception={"holding_object_context": False},
            sources={
                "mission_state": loaded_source("mission-state", "2026-06-10T19:40:24+08:00"),
                "patrol_state": loaded_source("patrol-state", "2026-06-10T19:40:24+08:00"),
                "perception": loaded_source("yolo-current-rgbd", "2026-06-10T19:55:06+08:00"),
            },
        )

        skew = [item for item in warnings if item["type"] == "source_time_skew"]
        self.assertEqual(len(skew), 1)
        self.assertFalse(skew[0]["blocking"])
        self.assertEqual(skew[0]["reason"], "fresh_observation_with_unchanged_durable_state")

    def test_stale_perception_skew_blocks_physical_actions(self) -> None:
        warnings = build_consistency_warnings(
            service_state={"holding_object": False},
            perception={"holding_object_context": False},
            sources={
                "perception": loaded_source("yolo-current-rgbd", "2026-06-10T19:40:24+08:00"),
                "mission_state": loaded_source("mission-state", "2026-06-10T19:55:06+08:00"),
            },
        )

        skew = [item for item in warnings if item["type"] == "source_time_skew"]
        self.assertEqual(len(skew), 1)
        self.assertTrue(skew[0]["blocking"])
        self.assertEqual(skew[0]["reason"], "stale_observation_source")

    def test_costmap_reset_allows_perception_bootstrap_moveahead(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-bootstrap-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 1})
            write_json(
                memory / "service-task-state.json",
                {
                    "phase": "SEARCH_PICKUP_TARGET",
                    "holding_object": False,
                    "pickup_surface_policy": "floor-only",
                },
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "reset",
                    "result_type": "local_costmap_reset",
                    "action_safety": {},
                },
            )
            write_json(memory / "position-map.json", {"pose": {"cell": "0,0", "heading": "north"}})
            write_json(memory / "object-memory.json", {"tracks": {}})
            write_json(memory / "global-plan.json", {"status": "reset"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "frontier_exists": True,
                    "obstacle_ahead": False,
                    "open_directions": ["left", "forward", "right"],
                    "occupancy": {"forward": 0.0, "left": 0.0, "right": 0.0},
                    "recommended_action": "MoveAhead",
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        moveahead = next(
            item
            for item in context["option_set"]["options"]
            if item["option_id"] == "move:moveahead"
        )
        self.assertEqual(moveahead["safety_source"], "perception-navigation-bootstrap")
        self.assertEqual(moveahead["reason"], "current_rgbd_perception_marks_forward_open")
        self.assertEqual(context["option_set"]["rule_baseline_option_id"], "move:moveahead")

    def test_place_option_requires_executor_ready_candidate(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-place-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 7})
            write_json(
                memory / "service-task-state.json",
                {
                    "phase": "SEARCH_RECEPTACLE",
                    "holding_object": True,
                    "held_object_labels": ["apple"],
                },
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(memory / "position-map.json", {"pose": {"cell": "0,0", "heading": "north"}})
            write_json(memory / "object-memory.json", {"tracks": {}})
            write_json(memory / "global-plan.json", {"status": "success", "next_action": "RotateLeft"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "holding_object_context": True,
                    "recommended_action": "RotateLeft",
                    "best_surface_candidate": {
                        "candidate_id": "visual_only_surface",
                        "label": "pc_surface",
                        "task_semantic_class": "place_receptacle",
                        "reachable": True,
                        "place_now": False,
                        "visual_place_ready": True,
                        "final_place_ready": False,
                        "needs_alignment": True,
                        "executor_checks": {"precheck_ok": False, "precheck_supported": True},
                    },
                    "place_affordance_candidates": [
                        {
                            "candidate_id": "executor_ready_surface",
                            "label": "pc_surface",
                            "task_semantic_class": "place_receptacle",
                            "surface_candidate_id": "pc_grid:counter_top:ready",
                            "surface_candidate_source": "pointcloud_plane_grid_completion",
                            "reachable": True,
                            "place_now": True,
                            "visual_place_ready": True,
                            "final_place_ready": True,
                            "affordance_ready": True,
                            "needs_alignment": False,
                            "interaction_point": {"x": 104.0, "y": 328.0},
                            "placement_points": [
                                {
                                    "x": 104.0,
                                    "y": 328.0,
                                    "rank": 1,
                                    "clearance_m": 0.04,
                                    "center_3d": {
                                        "x": -0.5326,
                                        "y": 0.4252,
                                        "z": 0.6706,
                                        "ground_distance_m": 0.8563,
                                        "ground_forward_m": 0.6706,
                                    },
                                }
                            ],
                            "placement_safety_contract": {
                                "version": "plane_local_grid_v1",
                                "clearance_owner": "pointcloud_plane_local_grid",
                                "target_coordinate_frame": "camera_relative_x_right_y_height_z_forward",
                                "requires_exact_target_execution": True,
                                "grid_occupancy_clear": True,
                                "grid_edge_eroded": True,
                                "camera_height_m": 0.901,
                            },
                            "free_space_completion": {
                                "mode": "plane_local_2d_grid",
                                "placement_point_count": 1,
                                "selected_center_3d": {
                                    "x": -0.5326,
                                    "y": 0.4252,
                                    "z": 0.6706,
                                    "ground_forward_m": 0.6706,
                                },
                            },
                            "geometry_checks": {
                                "distance_ok": True,
                                "grid_occupancy_clear": True,
                                "grid_edge_eroded": True,
                            },
                            "occupancy_checks": {
                                "blocked": False,
                                "free_space_grid_completion": True,
                                "blocked_by": [],
                            },
                            "visible_occupants": [
                                {
                                    "objectId": "private-object-id",
                                    "metadata": {"private": True},
                                    "label": "mug",
                                    "raw_label": "Mug",
                                    "task_semantic_class": "pickup_target",
                                    "bbox": {"x": 10, "y": 20, "w": 30, "h": 40},
                                }
                            ],
                            "executor_checks": {"precheck_ok": True, "precheck_supported": True},
                        }
                    ],
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertNotIn("place:visual_only_surface", option_ids)
        self.assertIn("place:executor_ready_surface", option_ids)
        self.assertEqual(context["option_set"]["rule_baseline_option_id"], "place:executor_ready_surface")
        ready = next(
            item
            for item in context["worklist"]["current_view"]["surface_candidates"]
            if item["candidate_id"] == "executor_ready_surface"
        )
        self.assertEqual(ready["surface_candidate_source"], "pointcloud_plane_grid_completion")
        self.assertEqual(ready["interaction_point"], {"x": 104.0, "y": 328.0})
        self.assertEqual(ready["placement_points"][0]["center_3d"]["ground_forward_m"], 0.6706)
        self.assertTrue(ready["placement_safety_contract"]["requires_exact_target_execution"])
        self.assertTrue(ready["geometry_checks"]["grid_occupancy_clear"])
        self.assertTrue(ready["occupancy_checks"]["free_space_grid_completion"])
        serialized = json.dumps(ready)
        self.assertNotIn("objectId", serialized)
        self.assertNotIn("metadata", serialized)

    def test_service_state_false_overrides_stale_perception_held_context(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-held-source-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 9})
            write_json(
                memory / "service-task-state.json",
                {
                    "phase": "SEARCH_PICKUP_TARGET",
                    "holding_object": False,
                    "held_object_labels": [],
                    "held_object_track_id": None,
                },
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(memory / "position-map.json", {"pose": {"cell": "0,0", "heading": "north"}})
            write_json(memory / "object-memory.json", {"tracks": {}})
            write_json(memory / "global-plan.json", {"status": "success", "next_action": "RotateLeft"})
            perception = memory / "yolo-current-rgbd.json"
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "holding_object_context": True,
                    "held_object_labels": ["apple"],
                    "held_object_family": "food",
                    "recommended_action": "RotateLeft",
                    "place_affordance_candidates": [
                        {
                            "candidate_id": "surface-ready",
                            "label": "pc_surface",
                            "task_semantic_class": "place_receptacle",
                            "surface_candidate_source": "pointcloud_plane_grid_completion",
                            "reachable": True,
                            "place_now": True,
                            "visual_place_ready": True,
                            "final_place_ready": True,
                            "affordance_ready": True,
                            "executor_checks": {"precheck_ok": True, "precheck_supported": True},
                        }
                    ],
                },
            )

            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertFalse(context["worklist"]["held_object"]["holding_object"])
        self.assertEqual(context["worklist"]["held_object"]["source"], "service_task_state")
        warning = next(
            item for item in context["consistency_warnings"] if item["type"] == "holding_state_mismatch"
        )
        self.assertFalse(warning["blocking"])
        self.assertEqual(warning["severity"], "advisory")
        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertNotIn("place:surface-ready", option_ids)
        self.assertEqual(context["option_set"]["rule_baseline_option_id"], "move:rotateleft")

    def test_offcenter_pointcloud_surface_uses_precheck_before_place(self) -> None:
        memory = WORKSPACE_ROOT / "memory" / "decision-context-precheck-fixture"
        if memory.exists():
            shutil.rmtree(memory)
        try:
            write_json(memory / "mission-state.json", {"enabled": True, "mode": "SERVICE", "max_steps": 80})
            write_json(memory / "room-state.json", {"room_name": "current_room", "room_complete": False})
            write_json(memory / "patrol-state.json", {"enabled": True, "mode": "SERVICE", "step_count": 8})
            write_json(
                memory / "service-task-state.json",
                {
                    "phase": "SEARCH_RECEPTACLE",
                    "holding_object": True,
                    "held_object_labels": ["apple"],
                },
            )
            write_json(
                memory / "navigation-costmap.json",
                {
                    "status": "success",
                    "action_safety": {
                        "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                    },
                },
            )
            write_json(memory / "position-map.json", {"pose": {"cell": "0,0", "heading": "north"}})
            write_json(memory / "object-memory.json", {"tracks": {}})
            write_json(memory / "global-plan.json", {"status": "success", "next_action": "RotateLeft"})
            perception = memory / "yolo-current-rgbd.json"
            offcenter_surface = {
                "candidate_id": "offcenter_surface",
                "label": "pc_surface",
                "task_semantic_class": "place_receptacle",
                "surface_candidate_id": "pc_grid:counter_top:offcenter",
                "surface_candidate_source": "pointcloud_plane_grid_completion",
                "reachable": True,
                "blocked": False,
                "place_now": False,
                "visual_place_ready": True,
                "final_place_ready": False,
                "affordance_ready": True,
                "needs_alignment": True,
                "geometry": {"bearing_deg": -30.0, "ground_distance_m": 0.72},
                "interaction_point": {"x": 156.0, "y": 340.0},
                "placement_points": [
                    {
                        "x": 156.0,
                        "y": 340.0,
                        "rank": 1,
                        "center_3d": {"x": -0.35, "y": 0.42, "z": 0.62},
                    }
                ],
                "placement_safety_contract": {
                    "version": "plane_local_grid_v1",
                    "requires_exact_target_execution": True,
                    "grid_occupancy_clear": True,
                    "grid_edge_eroded": True,
                },
                "executor_checks": {"precheck_ok": False, "precheck_supported": True},
            }
            write_json(
                perception,
                {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "online_safe": True,
                    "holding_object_context": True,
                    "recommended_action": "RotateLeft",
                    "place_affordance_candidates": [offcenter_surface],
                },
            )

            context = build_context(self.build_args(memory, perception))
            option_ids = [item["option_id"] for item in context["option_set"]["options"]]
            self.assertIn("place_precheck:offcenter_surface", option_ids)
            self.assertNotIn("place:offcenter_surface", option_ids)
            self.assertEqual(
                context["option_set"]["rule_baseline_option_id"],
                "place_precheck:offcenter_surface",
            )

            write_json(
                memory / "place-precheck-cache.json",
                {
                    "schema": "robot_cleaner_place_precheck_cache_v1",
                    "cached_at": "2026-06-09T19:00:00+08:00",
                    "perception_source_last_modified": "2026-06-09T18:59:00+08:00",
                    "candidate_ref": {"candidate_id": "offcenter_surface"},
                    "result": {
                        "status": "success",
                        "result_type": "place_precheck_ok",
                        "precheck_ok": True,
                        "precheck_reason": "place_precheck_ok",
                        "placement_execution_mode": "world_point_exact",
                    },
                },
            )
            context = build_context(self.build_args(memory, perception))
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertIn("place:offcenter_surface", option_ids)
        self.assertNotIn("place_precheck:offcenter_surface", option_ids)
        self.assertEqual(context["option_set"]["rule_baseline_option_id"], "place:offcenter_surface")
        candidate = context["worklist"]["current_view"]["surface_candidates"][0]
        self.assertTrue(candidate["executor_checks"]["precheck_ok"])
        self.assertTrue(candidate["executor_checks"]["cached"])
        self.assertFalse(candidate["executor_checks"]["cached_source_time_matches"])
        self.assertTrue(candidate["actionability"]["place_now"])

    def test_route_step_option_does_not_depend_on_raw_move_option(self) -> None:
        options = build_explore_route_step_options(
            explore_plan={
                "active_route": {
                    "status": "active",
                    "route_id": "route-frontier-x1-z0-abcd1234",
                    "goal_cell": "1,0",
                    "next_action": "MoveAhead",
                    "route_step": {
                        "status": "active",
                        "route_id": "route-frontier-x1-z0-abcd1234",
                        "step_index": 2,
                        "action": "MoveAhead",
                        "current_cell": "0,0",
                        "current_heading": "east",
                        "target_cell": "1,0",
                        "next_cell": "1,0",
                        "goal_cell": "1,0",
                        "desired_heading": "east",
                        "progress_effect": "advance_to_next_cell",
                    },
                }
            },
            move_options=[],
        )

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0]["kind"], "explore_route_step")
        self.assertEqual(options[0]["action"], "MoveAhead")
        self.assertEqual(options[0]["safety_source"], "active-route-costmap")
        self.assertNotIn("resolved_step_option_id", options[0])
        self.assertEqual(options[0]["route_step"]["next_cell"], "1,0")


if __name__ == "__main__":
    unittest.main()
