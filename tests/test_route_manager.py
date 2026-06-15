import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.route_manager import update_active_route


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = WORKSPACE_ROOT / ".test-tmp"


def make_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_route_fixture(
    memory: Path,
    *,
    cell: str,
    heading: str,
    path: list[str],
    goal: str,
    action_safety: dict,
) -> None:
    write_json(
        memory / "position-map.json",
        {
            "pose": {"cell": cell, "heading": heading},
            "cells": {item: {"state": "free", "visited": True} for item in path},
            "frontiers": [goal],
            "stats": {"visited_cell_count": len(path), "unknown_frontier_count": 1},
        },
    )
    write_json(
        memory / "room-state.json",
        {
            "last_cell": cell,
            "last_heading": heading,
            "frontier_cells": [goal],
            "active_frontier_goal": {
                "schema": "robot_cleaner_active_frontier_goal_v1",
                "status": "active",
                "mode": "frontier_cluster",
                "cell": goal,
                "cluster_cells": [goal],
                "path": path,
                "next_cell": path[1] if len(path) > 1 else goal,
            },
        },
    )
    write_json(memory / "navigation-costmap.json", {"status": "success", "action_safety": action_safety})


class RouteManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._backend_env = patch.dict(os.environ, {"ROBOT_MAP_BACKEND": "action_odometry"}, clear=False)
        self._backend_env.start()

    def tearDown(self) -> None:
        self._backend_env.stop()

    def test_committed_route_turnaround_uses_deterministic_left_turn_not_goal_drift(self) -> None:
        memory = make_tmp_dir("route-turnaround")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_route_fixture(
            memory,
            cell="0,4",
            heading="north",
            path=["0,4", "0,3", "-1,3"],
            goal="-1,3",
            action_safety={
                "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
            },
        )

        result = update_active_route(memory_dir=memory, persist=True)
        route = result["active_route"]
        step = route["route_step"]

        self.assertEqual(route["status"], "active")
        self.assertEqual(route["goal_cell"], "-1,3")
        self.assertEqual(step["next_cell"], "0,3")
        self.assertEqual(step["desired_heading"], "south")
        self.assertEqual(step["action"], "RotateLeft")
        self.assertEqual(step["progress_effect"], "turnaround_toward_next_cell")
        self.assertEqual(step["heading_after_action"], "west")

    def test_committed_route_continues_turn_toward_same_next_cell(self) -> None:
        memory = make_tmp_dir("route-continue-turn")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_route_fixture(
            memory,
            cell="0,4",
            heading="west",
            path=["0,4", "0,3", "-1,3"],
            goal="-1,3",
            action_safety={
                "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
            },
        )

        route = update_active_route(memory_dir=memory, persist=True)["active_route"]
        step = route["route_step"]

        self.assertEqual(step["next_cell"], "0,3")
        self.assertEqual(step["desired_heading"], "south")
        self.assertEqual(step["action"], "RotateLeft")
        self.assertEqual(step["progress_effect"], "turn_toward_next_cell")

    def test_committed_route_advances_after_current_cell_is_on_path(self) -> None:
        memory = make_tmp_dir("route-advance")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_route_fixture(
            memory,
            cell="0,3",
            heading="south",
            path=["0,4", "0,3", "-1,3"],
            goal="-1,3",
            action_safety={
                "RotateRight": {"safe": True, "reason": "clear_swept_volume"},
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
            },
        )

        route = update_active_route(memory_dir=memory, persist=True)["active_route"]
        step = route["route_step"]

        self.assertEqual(route["path"], ["0,3", "-1,3"])
        self.assertEqual(step["next_cell"], "-1,3")
        self.assertEqual(step["desired_heading"], "west")
        self.assertEqual(step["action"], "RotateRight")

    def test_committed_route_blocks_when_next_action_is_not_costmap_safe(self) -> None:
        memory = make_tmp_dir("route-blocked")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_route_fixture(
            memory,
            cell="0,4",
            heading="north",
            path=["0,4", "0,3", "-1,3"],
            goal="-1,3",
            action_safety={
                "RotateLeft": {"safe": False, "reason": "rotation_blocked"},
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
            },
        )

        route = update_active_route(memory_dir=memory, persist=True)["active_route"]

        self.assertEqual(route["status"], "blocked")
        self.assertEqual(route["next_action"], "RotateLeft")
        self.assertEqual(route["blocked_reason"], "rotation_blocked")
        self.assertEqual(route["route_step"]["action"], "RotateLeft")


if __name__ == "__main__":
    unittest.main()
