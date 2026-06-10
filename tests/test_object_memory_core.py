from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.navigation_memory_core import NavigationMemory
from scripts.object_memory_core import (
    ObjectMemory,
    compute_recommended_view_cell,
    load_memory,
    project_observation_to_cell,
)


def nav_status() -> dict:
    return {
        "cell_size": 0.25,
        "last_cell": "1,-3",
        "last_heading": "south",
        "visited_cells": ["1,-3", "1,-4", "2,-4", "0,-4"],
        "frontier_cells": ["1,-5"],
        "blocked_edges": [],
    }


def apple_candidate() -> dict:
    return {
        "label": "apple",
        "raw_label": "Apple",
        "task_semantic_class": "pickup_target",
        "confidence": 0.91,
        "bearing_deg": 25.0,
        "ground_distance_m": 1.2,
        "distance_m": 1.2,
        "position_hint": "front-right",
        "bbox": {"x": 287, "y": 525, "w": 26, "h": 32},
        "center_3d": {"x": 0.4, "y": -0.6, "z": 1.1},
        "reachable": True,
        "pickup_now": False,
    }


def apple_low_info_duplicate() -> dict:
    candidate = apple_candidate()
    candidate.pop("position_hint", None)
    candidate["source"] = "placement_avoidance_candidates"
    return candidate


def lettuce_candidate() -> dict:
    candidate = apple_candidate()
    candidate["label"] = "lettuce"
    candidate["raw_label"] = "Lettuce"
    candidate["bbox"] = {"x": 330, "y": 505, "w": 28, "h": 34}
    candidate["bearing_deg"] = 18.0
    candidate["ground_distance_m"] = 1.15
    candidate["distance_m"] = 1.15
    return candidate


def elevated_mug_candidate() -> dict:
    return {
        "label": "mug",
        "raw_label": "Mug",
        "task_semantic_class": "pickup_target",
        "confidence": 0.96,
        "bearing_deg": 49.0,
        "ground_distance_m": 1.16,
        "distance_m": 1.16,
        "position_hint": "front-right",
        "bbox": {"x": 547, "y": 325, "w": 52, "h": 47},
        "center_3d": {"x": 0.88, "y": 0.28, "z": 0.97},
        "surface_hint": "surface_or_elevated",
        "is_floor_level": False,
        "pickup_now": False,
        "support_context_blocked": True,
        "geometry": {"bottom_y_ratio": 0.62, "height_m": 0.28},
    }


def elevated_low_height_cup_candidate() -> dict:
    return {
        "label": "cup",
        "raw_label": "Cup",
        "task_semantic_class": "pickup_target",
        "confidence": 0.85,
        "bearing_deg": -22.0,
        "ground_distance_m": 1.36,
        "distance_m": 1.87,
        "position_hint": "front-left",
        "bbox": {"x": 202, "y": 394, "w": 29, "h": 48},
        "center_3d": {"x": -0.52, "y": -0.67, "z": 1.87},
        "surface_hint": "surface_or_elevated",
        "is_floor_level": False,
        "pickup_now": False,
        "support_context_blocked": False,
        "geometry": {"bottom_y_ratio": 0.736, "height_m": -0.67},
    }


def far_ambiguous_floor_apple_candidate() -> dict:
    return {
        "label": "apple",
        "raw_label": "Apple",
        "task_semantic_class": "pickup_target",
        "confidence": 0.95,
        "bearing_deg": 0.0,
        "ground_distance_m": 1.068,
        "distance_m": 1.068,
        "position_hint": "front-center",
        "bbox": {"x": 290, "y": 430, "w": 20, "h": 21},
        "center_3d": {"x": 0.0, "y": -0.6252, "z": 1.068},
        "surface_hint": "surface_or_elevated",
        "is_floor_level": False,
        "pickup_now": False,
        "support_context_blocked": False,
        "geometry": {"bottom_y_ratio": 0.751, "height_m": -0.6252},
    }


def ignored_pan_candidate() -> dict:
    return {
        "label": "pan",
        "raw_label": "Pan",
        "task_semantic_class": "ignored_object",
        "confidence": 0.93,
        "bearing_deg": -20.0,
        "ground_distance_m": 1.0,
        "bbox": {"x": 70, "y": 210, "w": 38, "h": 16},
    }


def countertop_candidate() -> dict:
    return {
        "id": "pc_grid:counter_top:1",
        "surface_candidate_id": "pc_grid:counter_top:1",
        "label": "pc_surface",
        "raw_label": "CounterTop",
        "parent_object": "counter_top",
        "task_semantic_class": "place_receptacle",
        "surface_candidate_source": "pointcloud_plane_grid_completion",
        "source": "pointcloud_plane_grid_completion",
        "confidence": 0.88,
        "bearing_deg": -35.0,
        "ground_distance_m": 1.1,
        "position_hint": "front-left",
        "visual_place_ready": True,
        "bbox": {"x": 40, "y": 360, "w": 120, "h": 80},
        "center_3d": {"x": -0.5, "y": 0.5, "z": 0.95},
    }


class ObjectMemoryCoreTests(unittest.TestCase):
    def test_project_observation_uses_observed_from_and_estimated_cell(self) -> None:
        front = project_observation_to_cell(
            "0,0",
            "north",
            bearing_deg=0.0,
            distance_m=0.5,
            cell_size_m=0.25,
        )
        self.assertEqual(front["estimated_object_cell"], "0,2")

        east = project_observation_to_cell(
            "0,0",
            "east",
            bearing_deg=0.0,
            distance_m=0.5,
            cell_size_m=0.25,
        )
        self.assertEqual(east["estimated_object_cell"], "2,0")

        result = project_observation_to_cell(
            "1,-3",
            "south",
            bearing_deg=25.0,
            distance_m=1.2,
            cell_size_m=0.25,
            position_hint="front-right",
        )
        self.assertEqual(result["method"], "bearing_distance_projection")
        self.assertNotEqual(result["estimated_object_cell"], "1,-3")
        self.assertTrue(result["candidate_cells"])

        hinted = project_observation_to_cell(
            "0,0",
            "north",
            bearing_deg=None,
            distance_m=None,
            cell_size_m=0.25,
            position_hint="front-right",
        )
        self.assertEqual(hinted["method"], "position_hint_prior")
        self.assertNotEqual(hinted["estimated_object_cell"], "0,0")

    def test_recommended_viewpoint_is_not_object_cell(self) -> None:
        result = compute_recommended_view_cell(
            "2,-5",
            task_class="pickup_target",
            current_cell="1,-3",
            current_heading="south",
            navigation_status=nav_status(),
            cell_size_m=0.25,
        )
        self.assertNotEqual(result["recommended_view_cell"], "2,-5")
        self.assertIn(result["recommended_heading"], {"north", "east", "south", "west"})

    def test_update_merges_and_selects_pickup_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            analysis = {
                "best_pickup_candidate": apple_candidate(),
                "service_candidates": [apple_candidate()],
            }
            first = memory.update_from_observation(
                analysis,
                current_cell="1,-3",
                heading="south",
                step=12,
                navigation_status=nav_status(),
            )
            second = memory.update_from_observation(
                analysis,
                current_cell="1,-3",
                heading="south",
                step=13,
                navigation_status=nav_status(),
            )
            data = memory.load_memory()
            self.assertEqual(first["created_count"], 1)
            self.assertEqual(second["created_count"], 0)
            self.assertEqual(len(data["tracks"]), 1)
            track = next(iter(data["tracks"].values()))
            self.assertEqual(track["seen_count"], 2)
            self.assertEqual(track["observed_from"]["cell"], "1,-3")
            self.assertIn("bearing_deg", track["last_observation"])
            self.assertIn("distance_m", track["last_observation"])
            self.assertIn("position_hint", track["last_observation"])
            self.assertIn("estimated_object_cell", track["estimated_location"])
            self.assertIn("candidate_cells", track["estimated_location"])
            self.assertIn("recommended_view_cell", track["viewpoint"])
            self.assertTrue(analysis["best_pickup_candidate"]["object_memory_track_id"])

            target = memory.select_pickup_target(
                current_cell="1,-3",
                heading="south",
                step=14,
                navigation_status=nav_status(),
            )
            self.assertIsNotNone(target)
            self.assertEqual(target["goal_type"], "pickup_target")
            self.assertTrue(target["recommended_view_cell"])
            active_goal = memory.load_goals()["active_goal"]
            self.assertIsNotNone(active_goal)
            self.assertEqual(active_goal["planner_inputs"]["goal_type"], "pickup_target")
            self.assertEqual(active_goal["planner_inputs"]["target_track_id"], target["track_id"])

    def test_floor_only_rejects_elevated_pickup_as_navigation_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            analysis = {
                "best_pickup_candidate": elevated_mug_candidate(),
                "service_candidates": [elevated_mug_candidate()],
            }
            memory.update_from_observation(
                analysis,
                current_cell="-6,0",
                heading="north",
                step=25,
                navigation_status=nav_status(),
            )

            floor_only = memory.select_pickup_target(
                current_cell="-6,0",
                heading="north",
                step=26,
                navigation_status=nav_status(),
                pickup_surface_policy="floor-only",
            )
            self.assertIsNone(floor_only)
            selected = memory.load_memory()["selected_targets"]["pickup"]
            self.assertIsNone(selected["track_id"])
            reasons = selected["rejected_tracks"][0]["goal_rejection_reasons"]
            self.assertIn("floor_only_goal_surface_hint_surface_or_elevated", reasons)
            self.assertIn("floor_only_goal_support_context_blocked", reasons)

            any_surface = memory.select_pickup_target(
                current_cell="-6,0",
                heading="north",
                step=27,
                navigation_status=nav_status(),
                pickup_surface_policy="any-surface",
            )
            self.assertIsNotNone(any_surface)
            self.assertEqual(any_surface["label"], "mug")

    def test_floor_only_rejects_surface_hint_even_when_height_is_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            analysis = {
                "best_pickup_candidate": elevated_low_height_cup_candidate(),
                "service_candidates": [elevated_low_height_cup_candidate()],
            }
            memory.update_from_observation(
                analysis,
                current_cell="-7,-8",
                heading="north",
                step=28,
                navigation_status=nav_status(),
            )

            floor_only = memory.select_pickup_target(
                current_cell="-7,-8",
                heading="north",
                step=29,
                navigation_status=nav_status(),
                pickup_surface_policy="floor-only",
            )
            self.assertIsNone(floor_only)
            selected = memory.load_memory()["selected_targets"]["pickup"]
            self.assertIsNone(selected["track_id"])
            reasons = selected["rejected_tracks"][0]["goal_rejection_reasons"]
            self.assertIn("floor_only_goal_surface_hint_surface_or_elevated", reasons)
            self.assertIn("floor_only_goal_insufficient_floor_evidence", reasons)

    def test_floor_only_allows_ambiguous_food_as_approach_verify_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            analysis = {
                "best_pickup_candidate": far_ambiguous_floor_apple_candidate(),
                "service_candidates": [far_ambiguous_floor_apple_candidate()],
            }
            memory.update_from_observation(
                analysis,
                current_cell="-9,-8",
                heading="east",
                step=100,
                navigation_status=nav_status(),
            )

            target = memory.select_pickup_target(
                current_cell="-9,-8",
                heading="east",
                step=101,
                navigation_status=nav_status(),
                pickup_surface_policy="floor-only",
            )
            self.assertIsNotNone(target)
            self.assertEqual(target["label"], "apple")
            self.assertTrue(target["pickup_goal_eligible"])
            self.assertTrue(target["pickup_approach_verify_goal"])
            self.assertFalse(target["pickup_action_ready"])
            self.assertEqual(target["pickup_goal_note"], "eligible_for_approach_verify_navigation")
            self.assertIn("surface_hint_surface_or_elevated", target["pickup_action_rejection_reasons"])
            self.assertIn("insufficient_floor_evidence", target["pickup_action_rejection_reasons"])

    def test_low_info_duplicates_and_ignored_objects_do_not_pollute_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            analysis = {
                "best_pickup_candidate": apple_candidate(),
                "service_candidates": [lettuce_candidate()],
                "placement_avoidance_candidates": [
                    apple_low_info_duplicate(),
                    ignored_pan_candidate(),
                ],
            }
            result = memory.update_from_observation(
                analysis,
                current_cell="1,-3",
                heading="south",
                step=12,
                navigation_status=nav_status(),
            )
            data = memory.load_memory()
            labels = sorted(track["label"] for track in data["tracks"].values())
            self.assertEqual(labels, ["apple", "lettuce"])
            self.assertEqual(result["created_count"], 2)
            apple = next(track for track in data["tracks"].values() if track["label"] == "apple")
            self.assertEqual(apple["raw_labels"], ["Apple"])
            self.assertEqual(apple["seen_count"], 1)
            self.assertEqual(apple["last_observation"]["position_hint"], "front-right")

    def test_receptacle_selection_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            pickup_analysis = {"best_pickup_candidate": apple_candidate()}
            memory.update_from_observation(
                pickup_analysis,
                current_cell="1,-3",
                heading="south",
                step=18,
                navigation_status=nav_status(),
            )
            pickup_track_id = next(iter(memory.load_memory()["tracks"].keys()))
            picked = memory.mark_picked(track_id=pickup_track_id, step=19)
            self.assertEqual(picked["status"], "picked")

            receptacle_analysis = {
                "best_surface_candidate": countertop_candidate(),
                "visual_ready_surface_regions": [countertop_candidate()],
            }
            memory.update_from_observation(
                receptacle_analysis,
                current_cell="0,0",
                heading="north",
                step=20,
                navigation_status=nav_status(),
            )
            target = memory.select_receptacle_target(
                holding_object=True,
                current_cell="0,0",
                heading="north",
                step=21,
                navigation_status=nav_status(),
            )
            self.assertIsNotNone(target)
            self.assertEqual(target["goal_type"], "place_receptacle")

            data = memory.load_memory()
            receptacle_track_id = next(
                track_id
                for track_id, track in data["tracks"].items()
                if track.get("task_class") in {"place_receptacle", "surface_target"}
            )
            result = memory.mark_placed(
                held_track_id=pickup_track_id,
                receptacle_track_id=receptacle_track_id,
                step=22,
            )
            self.assertEqual(result["held_track_id"], pickup_track_id)
            self.assertEqual(result["receptacle_track_id"], receptacle_track_id)
            updated = memory.load_memory()["tracks"]
            self.assertEqual(updated[pickup_track_id]["status"], "placed")
            self.assertEqual(updated[receptacle_track_id]["status"], "used_for_place")

    def test_unreachable_track_clears_active_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            memory.update_from_observation(
                {"best_pickup_candidate": apple_candidate()},
                current_cell="1,-3",
                heading="south",
                step=30,
                navigation_status=nav_status(),
            )
            target = memory.select_pickup_target(
                current_cell="1,-3",
                heading="south",
                step=31,
                navigation_status=nav_status(),
            )
            self.assertIsNotNone(target)
            self.assertIsNotNone(memory.load_goals()["active_goal"])
            for step in (32, 33, 34):
                memory.mark_unreachable(
                    track_id=target["track_id"],
                    step=step,
                    reason="navigation_no_path:test",
                )
            self.assertEqual(memory.load_memory()["tracks"][target["track_id"]]["status"], "unreachable")
            self.assertIsNone(memory.load_goals()["active_goal"])

    def test_stale_pickup_track_clears_goal_but_can_reactivate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            memory.update_from_observation(
                {"best_pickup_candidate": apple_candidate()},
                current_cell="1,-3",
                heading="south",
                step=30,
                navigation_status=nav_status(),
            )
            target = memory.select_pickup_target(
                current_cell="1,-3",
                heading="south",
                step=31,
                navigation_status=nav_status(),
            )
            self.assertIsNotNone(target)
            self.assertIsNotNone(memory.load_goals()["active_goal"])

            stale = memory.mark_stale(
                track_id=target["track_id"],
                step=32,
                reason="pickup_lock_released:locked_pickup_target_not_actionable",
            )
            self.assertEqual(stale["status"], "stale")
            self.assertTrue(stale["active_goal_cleared"])
            self.assertEqual(memory.load_memory()["tracks"][target["track_id"]]["status"], "stale")
            self.assertIsNone(memory.load_goals()["active_goal"])
            self.assertGreater(
                memory.pickup_cooldown_remaining(track_id=target["track_id"], step=32),
                0,
            )

            memory.update_from_observation(
                {"best_pickup_candidate": apple_candidate()},
                current_cell="1,-3",
                heading="south",
                step=33,
                navigation_status=nav_status(),
            )
            self.assertEqual(memory.load_memory()["tracks"][target["track_id"]]["status"], "unpicked")
            self.assertIsNone(
                memory.select_pickup_target(
                    current_cell="1,-3",
                    heading="south",
                    step=33,
                    navigation_status=nav_status(),
                )
            )
            self.assertIsNotNone(
                memory.select_pickup_target(
                    current_cell="1,-3",
                    heading="south",
                    step=37,
                    navigation_status=nav_status(),
                )
            )

    def test_rejected_pickup_track_clears_goal_and_does_not_reactivate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ObjectMemory(Path(tmp))
            memory.update_from_observation(
                {"best_pickup_candidate": apple_candidate()},
                current_cell="1,-3",
                heading="south",
                step=30,
                navigation_status=nav_status(),
            )
            target = memory.select_pickup_target(
                current_cell="1,-3",
                heading="south",
                step=31,
                navigation_status=nav_status(),
            )
            self.assertIsNotNone(target)
            self.assertIsNotNone(memory.load_goals()["active_goal"])

            rejected = memory.mark_rejected_false_positive(
                track_id=target["track_id"],
                step=32,
                reason="pickup_target_rejected_after_release:test",
            )
            self.assertEqual(rejected["status"], "rejected_false_positive")
            self.assertTrue(rejected["active_goal_cleared"])
            self.assertEqual(
                memory.load_memory()["tracks"][target["track_id"]]["status"],
                "rejected_false_positive",
            )
            self.assertIsNone(memory.load_goals()["active_goal"])
            self.assertIsNone(
                memory.select_pickup_target(
                    current_cell="1,-3",
                    heading="south",
                    step=33,
                    navigation_status=nav_status(),
                )
            )

            candidate = apple_candidate()
            memory.update_from_observation(
                {"best_pickup_candidate": candidate},
                current_cell="1,-3",
                heading="south",
                step=34,
                navigation_status=nav_status(),
            )
            self.assertEqual(candidate["object_memory_status"], "rejected_false_positive")
            self.assertEqual(
                memory.load_memory()["tracks"][target["track_id"]]["status"],
                "rejected_false_positive",
            )
            self.assertIsNone(
                memory.select_pickup_target(
                    current_cell="1,-3",
                    heading="south",
                    step=35,
                    navigation_status=nav_status(),
                )
            )

    def test_navigation_recommend_accepts_object_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nav = NavigationMemory(Path(tmp))
            nav.observe(
                vision={},
                analysis={"open_directions": ["forward", "left", "right"]},
            )
            result = nav.recommend(
                vision={},
                analysis={"open_directions": ["forward", "left", "right"], "obstacle_ahead": False},
                target_cell="0,2",
                target_reason="object_memory_target",
                target_track_id="objtrk:apple:0001",
                goal_type="pickup_target",
            )
            recommendation = result["recommendation"]
            self.assertEqual(recommendation["source"], "object_memory")
            self.assertEqual(recommendation["target_track_id"], "objtrk:apple:0001")
            self.assertEqual(recommendation["goal_type"], "pickup_target")

    def test_empty_memory_file_loads_default_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "object-memory.json"
            path.write_text("", encoding="utf-8")
            data = load_memory(path)
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["tracks"], {})


if __name__ == "__main__":
    unittest.main()
