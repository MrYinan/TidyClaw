import json
import shutil
import unittest
from pathlib import Path

from scripts.coverage_waypoint_state import (
    COVERAGE_WAYPOINT_STATE_SCHEMA,
    clear_active_waypoint_goal,
    load_room_state,
    mark_waypoint_blocked,
    mark_waypoint_observed,
    normalize_coverage_waypoint_state,
    set_active_waypoint_goal,
    sync_room_coverage_waypoints,
)
from scripts.inspection_waypoints import INSPECTION_WAYPOINTS_SCHEMA


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def fresh_memory(name: str) -> Path:
    memory = WORKSPACE_ROOT / "memory" / name
    if memory.exists():
        shutil.rmtree(memory)
    memory.mkdir(parents=True, exist_ok=True)
    return memory


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def waypoint_set() -> dict:
    return {
        "schema": INSPECTION_WAYPOINTS_SCHEMA,
        "source": "map_backend_reachable_coverage",
        "map_backend": "map_bundle",
        "coverage_radius_cells": 3,
        "required_waypoint_ids": ["wp_x0_z0", "wp_x3_z0", "wp_x3_z3"],
        "waypoints": [
            {
                "waypoint_id": "wp_x0_z0",
                "cell": "0,0",
                "x_cell": 0,
                "z_cell": 0,
                "purpose": "coverage_scan",
                "waypoint_source": "map_backend_reachable_coverage",
                "coverage_estimate": 0.33,
            },
            {
                "waypoint_id": "wp_x3_z0",
                "cell": "3,0",
                "x_cell": 3,
                "z_cell": 0,
                "purpose": "coverage_scan",
                "waypoint_source": "map_backend_reachable_coverage",
                "coverage_estimate": 0.33,
            },
            {
                "waypoint_id": "wp_x3_z3",
                "cell": "3,3",
                "x_cell": 3,
                "z_cell": 3,
                "purpose": "coverage_scan",
                "waypoint_source": "map_backend_reachable_coverage",
                "coverage_estimate": 0.34,
            },
        ],
    }


class CoverageWaypointStateTests(unittest.TestCase):
    def test_normalize_preserves_observed_and_blocked_ids(self) -> None:
        state = normalize_coverage_waypoint_state(
            waypoint_set(),
            {
                "observed_waypoint_ids": ["wp_x0_z0", "stale_wp"],
                "blocked_waypoint_ids": ["wp_x3_z0"],
                "active_waypoint_goal": {"waypoint_id": "wp_x3_z3", "cell": "3,3"},
            },
            current_cell="0,0",
        )

        self.assertEqual(state["schema"], COVERAGE_WAYPOINT_STATE_SCHEMA)
        self.assertEqual(state["required_waypoint_count"], 3)
        self.assertEqual(state["observed_waypoint_ids"], ["wp_x0_z0"])
        self.assertEqual(state["blocked_waypoint_ids"], ["wp_x3_z0"])
        self.assertEqual(state["active_waypoint_goal"]["waypoint_id"], "wp_x3_z3")
        self.assertEqual(state["pending_waypoint_ids"], ["wp_x3_z3"])
        self.assertAlmostEqual(state["sweep_coverage_rate"], 2 / 3, places=5)
        self.assertEqual(state["waypoint_status"]["wp_x0_z0"]["status"], "observed")
        self.assertEqual(state["waypoint_status"]["wp_x3_z0"]["status"], "blocked")

    def test_active_goal_is_dropped_when_already_observed(self) -> None:
        state = normalize_coverage_waypoint_state(
            waypoint_set(),
            {
                "observed_waypoint_ids": ["wp_x3_z3"],
                "active_waypoint_goal": {"waypoint_id": "wp_x3_z3", "cell": "3,3"},
            },
        )

        self.assertIsNone(state["active_waypoint_goal"])

    def test_set_clear_observe_and_block_waypoints(self) -> None:
        state = normalize_coverage_waypoint_state(waypoint_set())
        state = set_active_waypoint_goal(state, "wp_x3_z0", selected_at="2026-06-13T00:00:00+08:00")

        self.assertEqual(state["active_waypoint_goal"]["waypoint_id"], "wp_x3_z0")
        self.assertEqual(state["waypoint_status"]["wp_x3_z0"]["attempt_count"], 1)

        state = clear_active_waypoint_goal(state, reason="operator_abort")
        self.assertIsNone(state["active_waypoint_goal"])
        self.assertEqual(state["waypoint_status"]["wp_x3_z0"]["status"], "pending")

        state = mark_waypoint_observed(state, "wp_x3_z0", observation_id="obs-001")
        self.assertIn("wp_x3_z0", state["observed_waypoint_ids"])
        self.assertEqual(state["waypoint_status"]["wp_x3_z0"]["last_observation_id"], "obs-001")

        state = mark_waypoint_blocked(state, "wp_x3_z3", reason="no_static_path")
        self.assertIn("wp_x3_z3", state["blocked_waypoint_ids"])
        self.assertEqual(state["waypoint_status"]["wp_x3_z3"]["blocked_reason"], "no_static_path")
        self.assertAlmostEqual(state["sweep_coverage_rate"], 2 / 3, places=5)

    def test_sync_room_coverage_waypoints_persists_state(self) -> None:
        memory = fresh_memory("coverage-waypoint-sync")
        try:
            write_json(memory / "room-state.json", {"room_name": "current_room"})
            state = sync_room_coverage_waypoints(waypoint_set(), memory_dir=memory, current_cell="0,0")
            room = load_room_state(memory)
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(state["required_waypoint_count"], 3)
        self.assertEqual(room["coverage_waypoints"]["schema"], COVERAGE_WAYPOINT_STATE_SCHEMA)
        self.assertEqual(room["coverage_waypoints"]["pending_waypoint_count"], 3)


if __name__ == "__main__":
    unittest.main()
