import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.exploration_goal_manager import update_exploration_goals
from scripts.explore_planner import build_explore_plan


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = WORKSPACE_ROOT / ".test-tmp"


def make_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ExplorationGoalManagerTests(unittest.TestCase):
    def test_selects_and_persists_active_frontier_goal(self) -> None:
        memory = make_tmp_dir("active-frontier-goal")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_json(
            memory / "position-map.json",
            {
                "pose": {"cell": "0,0", "heading": "east"},
                "cells": {
                    "0,0": {"cell": "0,0", "state": "free", "visited": True, "seen_count": 3},
                    "1,0": {"cell": "1,0", "state": "unknown", "visited": False},
                    "1,1": {"cell": "1,1", "state": "unknown", "visited": False},
                },
                "edges": {"known_open_edges": [], "blocked_edges": []},
                "frontiers": ["1,0", "1,1"],
                "stats": {"visited_cell_count": 1, "unknown_frontier_count": 2},
            },
        )
        write_json(
            memory / "room-state.json",
            {
                "last_cell": "0,0",
                "last_heading": "east",
                "visited_cells": ["0,0"],
                "frontier_cells": ["1,0", "1,1"],
                "coverage_estimate": 0.01,
                "explored_steps": 2,
            },
        )

        result = update_exploration_goals(memory_dir=memory, persist=True)
        room = json.loads((memory / "room-state.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(room["active_frontier_goal"]["schema"], "robot_cleaner_active_frontier_goal_v1")
        self.assertEqual(room["active_frontier_goal"]["cell"], "1,0")
        self.assertEqual(room["active_frontier_goal"]["next_action"], "MoveAhead")
        self.assertGreaterEqual(room["active_frontier_goal"]["cluster_size"], 2)
        self.assertFalse(room["coverage_patrol"]["active"])

    def test_coverage_patrol_activates_when_no_frontiers_remain(self) -> None:
        memory = make_tmp_dir("coverage-patrol")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_json(
            memory / "position-map.json",
            {
                "pose": {"cell": "0,0", "heading": "north"},
                "cells": {
                    "0,0": {"cell": "0,0", "state": "free", "visited": True, "seen_count": 8},
                    "0,1": {"cell": "0,1", "state": "free", "visited": True, "seen_count": 6},
                    "1,0": {"cell": "1,0", "state": "free", "visited": True, "seen_count": 1},
                    "2,0": {"cell": "2,0", "state": "free", "visited": True, "seen_count": 1},
                },
                "edges": {"known_open_edges": [], "blocked_edges": []},
                "frontiers": [],
                "stats": {"visited_cell_count": 4, "unknown_frontier_count": 0},
            },
        )
        write_json(
            memory / "room-state.json",
            {
                "last_cell": "0,0",
                "last_heading": "north",
                "visited_cells": ["0,0", "0,1", "1,0", "2,0"],
                "coverage_estimate": 0.7,
                "frontier_cells": [],
                "explored_steps": 12,
            },
        )

        result = update_exploration_goals(memory_dir=memory, persist=True)
        coverage = result["coverage_patrol"]

        self.assertTrue(coverage["active"])
        self.assertEqual(coverage["schema"], "robot_cleaner_coverage_patrol_v1")
        self.assertIn(coverage["target_cell"], {"1,0", "2,0", "0,1"})
        self.assertIsNone(result["active_frontier_goal"])

    def test_explore_plan_prefers_step_that_advances_active_frontier(self) -> None:
        position_map = {
            "pose": {"cell": "0,0", "heading": "east"},
            "cells": {
                "0,0": {"cell": "0,0", "state": "free", "visited": True},
                "1,0": {"cell": "1,0", "state": "unknown", "visited": False},
                "0,1": {"cell": "0,1", "state": "free", "visited": True},
            },
            "frontiers": ["1,0"],
            "recent_actions": [],
            "active_frontier_goal": {
                "schema": "robot_cleaner_active_frontier_goal_v1",
                "cell": "1,0",
                "next_action": "MoveAhead",
                "next_cell": "1,0",
                "cluster_size": 1,
            },
        }
        costmap = {
            "action_safety": {
                "MoveAhead": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
                "MoveLeft": {"safe": True, "reason": "clear_swept_volume", "observed_ratio": 0.9},
            }
        }

        plan = build_explore_plan(position_map=position_map, navigation_costmap=costmap, exploration={})

        self.assertEqual(plan["active_frontier_goal"]["cell"], "1,0")
        self.assertTrue(plan["waypoint_candidates"])
        self.assertEqual(plan["waypoint_candidates"][0]["action"], "MoveAhead")
        self.assertEqual(plan["waypoint_candidates"][0]["purpose"], "advance_active_frontier_goal")
        self.assertIn("advances_active_frontier_goal", plan["waypoint_candidates"][0]["reasons"])

    def test_blocked_active_route_invalidates_existing_frontier_goal(self) -> None:
        memory = make_tmp_dir("blocked-active-route")
        self.addCleanup(shutil.rmtree, memory, ignore_errors=True)
        write_json(
            memory / "position-map.json",
            {
                "pose": {"cell": "1,-4", "heading": "east"},
                "cells": {
                    "1,-4": {"cell": "1,-4", "state": "free", "visited": True},
                    "2,-4": {"cell": "2,-4", "state": "unknown", "visited": False},
                    "1,-5": {"cell": "1,-5", "state": "unknown", "visited": False},
                },
                "edges": {"known_open_edges": [], "blocked_edges": []},
                "frontiers": ["2,-4", "1,-5"],
                "stats": {"visited_cell_count": 8, "unknown_frontier_count": 2},
            },
        )
        write_json(
            memory / "room-state.json",
            {
                "last_cell": "1,-4",
                "last_heading": "east",
                "frontier_cells": ["2,-4", "1,-5"],
                "explored_steps": 20,
                "active_frontier_goal": {
                    "schema": "robot_cleaner_active_frontier_goal_v1",
                    "status": "active",
                    "cell": "2,-4",
                    "path": ["1,-4", "2,-4"],
                },
                "active_route": {
                    "schema": "robot_cleaner_active_route_v1",
                    "status": "blocked",
                    "goal_cell": "2,-4",
                    "blocked_reason": "inflated_obstacle_in_swept_volume",
                },
            },
        )

        result = update_exploration_goals(memory_dir=memory, persist=True)
        room = json.loads((memory / "room-state.json").read_text(encoding="utf-8"))

        self.assertEqual(result["active_goal_validity"], "active_route_blocked_by_costmap")
        self.assertIn("2,-4", room["frontier_cooldowns"])
        self.assertNotEqual(room["active_frontier_goal"]["cell"], "2,-4")


if __name__ == "__main__":
    unittest.main()
