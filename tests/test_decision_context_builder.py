import argparse
import json
import shutil
import unittest
from pathlib import Path

from scripts.decision_context_builder import (
    DECISION_CONTEXT_SCHEMA,
    LoadedJson,
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
        self.assertNotIn("cells", json.dumps(context["navigation"]))
        self.assertIn("exploration", context)
        self.assertIn("recent_path", context["exploration"])
        self.assertIn("frontier_candidates", context["exploration"])
        self.assertNotIn('"cells"', json.dumps(context["exploration"]))
        self.assertNotIn("history", json.dumps(context["worklist"]))
        option_ids = [item["option_id"] for item in context["option_set"]["options"]]
        self.assertIn("pick:best_pickup_candidate_0", option_ids)
        self.assertIn("explore:frontier:x0_z1", option_ids)
        self.assertIn("move:moveahead", option_ids)
        self.assertIn("primary_options", context["option_set"])
        self.assertIn("fallback_options", context["option_set"])
        self.assertIn("goal_options", context["option_set"])
        self.assertIn("explore:frontier:x0_z1", context["option_set"]["primary_options"])
        self.assertIn("move:moveahead", context["option_set"]["fallback_options"])
        explore = next(item for item in context["option_set"]["options"] if item["option_id"] == "explore:frontier:x0_z1")
        self.assertEqual(explore["kind"], "explore_frontier")
        self.assertEqual(explore["decision_level"], "goal")
        self.assertEqual(explore["llm_priority"], "primary")
        self.assertFalse(explore["fallback_only"])
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


if __name__ == "__main__":
    unittest.main()
