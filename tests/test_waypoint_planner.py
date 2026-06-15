import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.coverage_waypoint_state import normalize_coverage_waypoint_state
from scripts.inspection_waypoints import INSPECTION_WAYPOINTS_SCHEMA
from scripts.waypoint_planner import (
    continue_active_waypoint_goal,
    plan_to_inspection_waypoint,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = WORKSPACE_ROOT / ".test-tmp"


def make_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_memory_fixture(
    memory: Path,
    *,
    cell: str = "0,0",
    heading: str = "north",
    action_safety: dict | None = None,
    room_extra: dict | None = None,
) -> None:
    cells = {
        "0,0": {"cell": "0,0", "state": "free", "visited": True},
        "0,1": {"cell": "0,1", "state": "free", "visited": False},
        "0,2": {"cell": "0,2", "state": "free", "visited": False},
        "1,2": {"cell": "1,2", "state": "free", "visited": False},
    }
    write_json(
        memory / "position-map.json",
        {
            "pose": {"cell": cell, "heading": heading},
            "cells": cells,
            "frontiers": ["0,1", "0,2", "1,2"],
            "stats": {"free_cell_count": len(cells), "visited_cell_count": 1},
        },
    )
    room = {"last_cell": cell, "last_heading": heading, "room_complete": False}
    if room_extra:
        room.update(room_extra)
    write_json(memory / "room-state.json", room)
    write_json(
        memory / "navigation-costmap.json",
        {
            "status": "success",
            "action_safety": action_safety
            or {
                "MoveAhead": {
                    "safe": True,
                    "reason": "clear_swept_volume",
                    "observed_ratio": 0.9,
                    "min_observed_ratio": 0.3,
                },
                "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
            },
        },
    )


def authored_waypoint_state(observed: list[str] | None = None) -> dict:
    waypoint_set = {
        "schema": INSPECTION_WAYPOINTS_SCHEMA,
        "source": "map_snapshot_inspection_waypoint",
        "map_backend": "action_odometry",
        "coverage_radius_cells": 2,
        "required_waypoint_ids": ["wp_front"],
        "waypoints": [
            {
                "waypoint_id": "wp_front",
                "cell": "0,1",
                "purpose": "coverage_scan",
                "waypoint_source": "test_authored",
            }
        ],
    }
    return normalize_coverage_waypoint_state(
        waypoint_set,
        {"observed_waypoint_ids": observed or []},
        current_cell="0,0",
    )


class WaypointPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_env = patch.dict(os.environ, {"ROBOT_MAP_BACKEND": "action_odometry"}, clear=False)
        self._backend_env.start()

    def tearDown(self) -> None:
        self._backend_env.stop()

    def test_plan_to_inspection_waypoint_creates_active_route_step(self) -> None:
        memory = make_tmp_dir("waypoint-plan-active")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_memory_fixture(
            memory,
            room_extra={
                "inspection_waypoints": [
                    {
                        "waypoint_id": "wp_front",
                        "cell": "0,1",
                        "purpose": "coverage_scan",
                        "waypoint_source": "test_authored",
                    }
                ],
                "coverage_waypoints": authored_waypoint_state(),
            },
        )

        result = plan_to_inspection_waypoint("wp_front", memory_dir=memory, persist=True)

        self.assertEqual(result["status"], "active")
        self.assertEqual(result["result_type"], "waypoint_plan_ready")
        self.assertEqual(result["route"]["waypoint_id"], "wp_front")
        self.assertEqual(result["route"]["next_action"], "MoveAhead")
        self.assertEqual(result["route"]["route_step"]["next_cell"], "0,1")
        self.assertEqual(result["coverage_waypoints"]["active_waypoint_goal"]["waypoint_id"], "wp_front")
        self.assertTrue((memory / "waypoint-plan.json").exists())

    def test_continue_active_waypoint_goal_reports_reached_and_requires_observe(self) -> None:
        memory = make_tmp_dir("waypoint-plan-reached")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        state = authored_waypoint_state()
        state["active_waypoint_goal"] = {
            "waypoint_id": "wp_front",
            "cell": "0,1",
            "status": "active",
        }
        write_memory_fixture(
            memory,
            cell="0,1",
            heading="north",
            room_extra={
                "inspection_waypoints": [
                    {
                        "waypoint_id": "wp_front",
                        "cell": "0,1",
                        "purpose": "coverage_scan",
                        "waypoint_source": "test_authored",
                    }
                ],
                "coverage_waypoints": state,
            },
        )

        result = continue_active_waypoint_goal(memory_dir=memory, persist=True)

        self.assertEqual(result["status"], "reached")
        self.assertEqual(result["required_next"], "observe_at_waypoint")
        self.assertTrue(result["route"]["requires_observe"])
        self.assertEqual(result["coverage_waypoints"]["active_waypoint_goal"]["status"], "reached")

    def test_plan_blocks_when_next_action_is_not_costmap_safe(self) -> None:
        memory = make_tmp_dir("waypoint-plan-blocked")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_memory_fixture(
            memory,
            action_safety={
                "MoveAhead": {
                    "safe": False,
                    "reason": "inflated_obstacle_in_swept_volume",
                    "observed_ratio": 0.9,
                    "min_observed_ratio": 0.3,
                }
            },
            room_extra={
                "inspection_waypoints": [
                    {
                        "waypoint_id": "wp_front",
                        "cell": "0,1",
                        "purpose": "coverage_scan",
                        "waypoint_source": "test_authored",
                    }
                ],
                "coverage_waypoints": authored_waypoint_state(),
            },
        )

        result = plan_to_inspection_waypoint("wp_front", memory_dir=memory, persist=True)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["required_next"], "recover_or_choose_new_waypoint")
        self.assertEqual(result["route"]["blocked_reason"], "inflated_obstacle_in_swept_volume")
        self.assertEqual(result["coverage_waypoints"]["active_waypoint_goal"]["status"], "blocked")

    def test_plan_avoids_hard_blocked_edge_when_rerouting_waypoint(self) -> None:
        memory = make_tmp_dir("waypoint-plan-hard-blocked-edge")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        state = authored_waypoint_state()
        state["active_waypoint_goal"] = {
            "waypoint_id": "wp_front",
            "cell": "0,1",
            "status": "active",
        }
        write_memory_fixture(
            memory,
            room_extra={
                "blocked_edges": ["0,0->0,1", "0,1->0,0"],
                "hard_blocked_edges": ["0,0->0,1", "0,1->0,0"],
                "inspection_waypoints": [
                    {
                        "waypoint_id": "wp_front",
                        "cell": "0,1",
                        "purpose": "coverage_scan",
                        "waypoint_source": "test_authored",
                    }
                ],
                "coverage_waypoints": state,
            },
        )

        result = continue_active_waypoint_goal(memory_dir=memory, persist=True)

        self.assertEqual(result["status"], "active")
        path_edges = [
            f"{source}->{target}"
            for source, target in zip(result["route"]["path"], result["route"]["path"][1:])
        ]
        self.assertNotIn("0,0->0,1", path_edges)
        self.assertNotIn("0,1->0,0", path_edges)
        self.assertNotEqual(result["route"]["route_step"]["next_cell"], "0,1")
        self.assertTrue(result["route"]["action_safety"]["safe"])

    def test_plan_rejects_unknown_or_observed_waypoint(self) -> None:
        memory = make_tmp_dir("waypoint-plan-invalid")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_memory_fixture(
            memory,
            room_extra={
                "inspection_waypoints": [
                    {
                        "waypoint_id": "wp_front",
                        "cell": "0,1",
                        "purpose": "coverage_scan",
                        "waypoint_source": "test_authored",
                    }
                ],
                "coverage_waypoints": authored_waypoint_state(observed=["wp_front"]),
            },
        )

        unknown = plan_to_inspection_waypoint("missing_wp", memory_dir=memory, persist=False)
        observed = plan_to_inspection_waypoint("wp_front", memory_dir=memory, persist=False)

        self.assertEqual(unknown["status"], "invalid_waypoint")
        self.assertEqual(unknown["reason"], "unknown_waypoint_id")
        self.assertEqual(observed["status"], "invalid_waypoint")
        self.assertIn("already observed", observed["reason"])


if __name__ == "__main__":
    unittest.main()
