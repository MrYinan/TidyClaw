import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.global_planner import AStarGlobalPlanner
from scripts.map_backend import (
    MAP_SNAPSHOT_SCHEMA,
    ActionOdometryMapBackend,
    BackendUnavailableError,
    load_map_backend,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def fresh_memory(name: str) -> Path:
    memory = WORKSPACE_ROOT / "memory" / name
    if memory.exists():
        shutil.rmtree(memory)
    memory.mkdir(parents=True, exist_ok=True)
    return memory


class MapBackendTests(unittest.TestCase):
    def test_action_odometry_backend_normalizes_position_and_room_state(self) -> None:
        memory = fresh_memory("map-backend-test-normalize")
        try:
            write_json(
                memory / "position-map.json",
                {
                    "room_name": "current_room",
                    "map_frame": {
                        "coordinate_mode": "action_odometry_grid",
                        "cell_size_m": 0.25,
                    },
                    "pose": {
                        "cell": "0,3",
                        "x_cell": 0,
                        "z_cell": 3,
                        "heading": "west",
                        "pose_confidence": 0.91,
                        "position_uncertainty_cells": 0.2,
                        "heading_confidence": 0.97,
                    },
                    "cells": {
                        "0,3": {"cell": "0,3", "state": "free", "visited": True, "seen_count": 2},
                        "-1,3": {"cell": "-1,3", "state": "unknown", "visited": False},
                    },
                    "edges": {"known_open_edges": ["0,3->0,2"], "blocked_edges": ["0,3->1,3"]},
                    "frontiers": ["-1,3"],
                    "stats": {
                        "visited_cell_count": 1,
                        "free_cell_count": 1,
                        "occupied_cell_count": 0,
                        "inflated_cell_count": 0,
                        "unknown_frontier_count": 1,
                        "collision_count": 0,
                    },
                    "recent_actions": ["MoveAhead", "RotateLeft"],
                },
            )
            write_json(
                memory / "room-state.json",
                {
                    "room_name": "current_room",
                    "coverage_estimate": 0.25,
                    "frontier_cells": ["-1,3", "0,4"],
                    "visited_cells": ["0,3"],
                    "active_frontier_goal": {"cell": "-1,3", "selected_at_step": 4},
                    "frontier_history": [{"event": "frontier_selected", "cell": "-1,3"}],
                    "frontier_cooldowns": {"0,4": {"until_step": 9}},
                },
            )

            snapshot = ActionOdometryMapBackend(memory).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(snapshot.schema, MAP_SNAPSHOT_SCHEMA)
        self.assertEqual(snapshot.backend, "action_odometry")
        self.assertEqual(snapshot.pose["cell"], "0,3")
        self.assertEqual(snapshot.pose["heading"], "west")
        self.assertEqual(snapshot.coverage["coverage_estimate"], 0.25)
        self.assertEqual(snapshot.coverage["visited_cell_count"], 1)
        self.assertIn("-1,3", snapshot.frontiers)
        self.assertIn("0,4", snapshot.frontiers)
        self.assertEqual(snapshot.edges["blocked_edges"], ["0,3->1,3"])
        self.assertEqual(snapshot.active_frontier_goal["cell"], "-1,3")
        self.assertEqual(snapshot.frontier_cooldowns["0,4"]["until_step"], 9)
        public = snapshot.public_summary()
        self.assertEqual(public["schema"], MAP_SNAPSHOT_SCHEMA)
        self.assertEqual(public["backend"], "action_odometry")
        self.assertNotIn('"cells"', json.dumps(public))

    def test_missing_files_return_valid_empty_snapshot(self) -> None:
        memory = fresh_memory("map-backend-test-missing")
        try:
            snapshot = ActionOdometryMapBackend(memory).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(snapshot.schema, MAP_SNAPSHOT_SCHEMA)
        self.assertEqual(snapshot.backend, "action_odometry")
        self.assertEqual(snapshot.pose["cell"], "0,0")
        self.assertEqual(snapshot.pose["heading"], "north")
        self.assertIsInstance(snapshot.to_position_status(), dict)
        self.assertIsInstance(snapshot.to_navigation_room_state(), dict)

    def test_snapshot_position_status_works_with_global_planner(self) -> None:
        memory = fresh_memory("map-backend-test-global-planner")
        try:
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "east"},
                    "cells": {
                        "0,0": {"cell": "0,0", "state": "free", "visited": True},
                        "1,0": {"cell": "1,0", "state": "unknown", "visited": False},
                    },
                    "edges": {"known_open_edges": [], "blocked_edges": []},
                    "frontiers": ["1,0"],
                    "stats": {"visited_cell_count": 1, "unknown_frontier_count": 1},
                },
            )
            write_json(memory / "room-state.json", {"coverage_estimate": 0.01})
            snapshot = ActionOdometryMapBackend(memory).load_snapshot()
            planner = AStarGlobalPlanner(memory)

            plan = planner.plan_to_best_frontier(
                current_cell="0,0",
                current_heading="east",
                position_status=snapshot.to_position_status(),
                semantic_status={"cells": {}},
            )
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(plan["status"], "success")
        self.assertEqual(plan["selected_goal_cell"], "1,0")
        self.assertEqual(plan["next_action"], "MoveAhead")

    def test_unknown_backend_is_explicit_error(self) -> None:
        memory = fresh_memory("map-backend-test-unknown")
        try:
            with patch.dict("os.environ", {"ROBOT_MAP_BACKEND": "missing_backend"}):
                with self.assertRaises(BackendUnavailableError):
                    load_map_backend(memory)
        finally:
            if memory.exists():
                shutil.rmtree(memory)


if __name__ == "__main__":
    unittest.main()
