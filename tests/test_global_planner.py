from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.global_planner import AStarGlobalPlanner
from scripts.position_map_core import PositionMap
from scripts.semantic_mapping_core import SemanticMap


class GlobalPlannerTests(unittest.TestCase):
    def build_map(self, memory_dir: Path) -> PositionMap:
        position = PositionMap(memory_dir)
        position.reset()
        data = position.load()
        for cell in ["0,0", "0,1", "1,1", "2,1", "2,0"]:
            position.mark_free(data, cell, evidence="test", visited=True)
        position.mark_occupied(data, "1,0", evidence="wall")
        position.save(data)
        SemanticMap(memory_dir).sync_position_layer()
        return position

    def test_astar_routes_around_occupied_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            self.build_map(memory_dir)
            planner = AStarGlobalPlanner(memory_dir)
            result = planner.plan_to_goal(
                current_cell="0,0",
                current_heading="east",
                target_cell="2,0",
                target_reason="known_apple_recommended_view_cell",
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["planner"], "astar")
            self.assertEqual(result["selected_goal_cell"], "2,0")
            self.assertNotIn("1,0", result["path"])
            self.assertEqual(result["path"][0], "0,0")
            self.assertEqual(result["path"][-1], "2,0")
            self.assertEqual(result["next_action"], "RotateLeft")
            self.assertEqual(result["next_cell"], "0,0")

    def test_blocked_edge_is_respected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = self.build_map(memory_dir)
            data = position.load()
            position.add_blocked_edge(data, "0,0", "0,1")
            position.save(data)
            planner = AStarGlobalPlanner(memory_dir)
            result = planner.plan_to_goal(
                current_cell="0,0",
                current_heading="north",
                target_cell="2,0",
            )
            self.assertEqual(result["status"], "success")
            self.assertNotEqual(result["path"][:2], ["0,0", "0,1"])

    def test_frontier_planner_returns_astar_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            data = position.load()
            position.mark_free(data, "0,0", evidence="test", visited=True)
            position.mark_free(data, "0,1", evidence="test", visited=True)
            position.save(data)
            semantic = SemanticMap(memory_dir)
            semantic.sync_position_layer()
            planner = AStarGlobalPlanner(memory_dir)
            status = position.status()
            result = planner.plan_to_best_frontier(
                current_cell="0,0",
                current_heading="north",
                frontier_cells=status["frontier_cells"],
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["planner"], "astar")
            self.assertTrue(result["path"])
            self.assertEqual(result["selected_goal_kind"], "semantic_frontier")

    def test_frontier_planner_keeps_sticky_goal_despite_semantic_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            data = position.load()
            for cell in ["0,-2", "0,-3"]:
                position.mark_free(data, cell, evidence="test", visited=True)
            position.save(data)
            planner = AStarGlobalPlanner(memory_dir)
            result = planner.plan_to_best_frontier(
                current_cell="0,-2",
                current_heading="west",
                frontier_cells=["1,-2", "1,-3"],
                preferred_frontier="1,-2",
                position_status=position.status(),
                semantic_status={
                    "cells": {},
                    "frontier_scores": {
                        "1,-2": {"exploration_score": 0.1, "unknown_neighbor_count": 1},
                        "1,-3": {"exploration_score": 0.9, "unknown_neighbor_count": 2},
                    },
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["selected_goal_cell"], "1,-2")
            self.assertEqual(result["selected_goal_kind"], "sticky_semantic_frontier")
            self.assertEqual(result["next_action"], "RotateRight")

    def test_frontier_planner_skips_cooldown_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            data = position.load()
            for cell in ["0,0", "0,1"]:
                position.mark_free(data, cell, evidence="test", visited=True)
            position.save(data)
            planner = AStarGlobalPlanner(memory_dir)
            result = planner.plan_to_best_frontier(
                current_cell="0,0",
                current_heading="north",
                frontier_cells=["0,2", "1,0"],
                frontier_cooldowns={"0,2": {"until_step": 10, "reason": "test"}},
                position_status=position.status(),
                semantic_status={
                    "cells": {},
                    "frontier_scores": {
                        "0,2": {"exploration_score": 1.0, "unknown_neighbor_count": 3},
                        "1,0": {"exploration_score": 0.1, "unknown_neighbor_count": 1},
                    },
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["selected_goal_cell"], "1,0")

    def test_frontier_backtrack_guard_prefers_nearby_side_information_gain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            data = position.load()
            for cell in ["0,0", "0,1", "0,2", "0,3"]:
                position.mark_free(data, cell, evidence="test", visited=True)
            position.save(data)
            planner = AStarGlobalPlanner(memory_dir)
            result = planner.plan_to_best_frontier(
                current_cell="0,3",
                current_heading="north",
                frontier_cells=["0,-1", "1,3"],
                allow_backtrack=False,
                position_status=position.status(),
                semantic_status={
                    "cells": {},
                    "frontier_scores": {
                        "0,-1": {"exploration_score": 1.0, "unknown_neighbor_count": 4},
                        "1,3": {"exploration_score": 0.1, "unknown_neighbor_count": 1},
                    },
                },
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["selected_goal_cell"], "1,3")
            self.assertFalse(result["frontier_backtrack_allowed"])


if __name__ == "__main__":
    unittest.main()
