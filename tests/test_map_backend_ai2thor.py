from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.authoritative_map_sync import sync_authoritative_room_state
from scripts.map_backend import AI2ThorGroundTruthMapBackend, BackendUnavailableError, load_map_backend


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


class AI2ThorGroundTruthMapBackendTests(unittest.TestCase):
    def test_cache_payload_builds_complete_reachable_snapshot(self) -> None:
        memory = fresh_memory("map-backend-ai2thor")
        try:
            write_json(
                memory / "ai2thor-groundtruth-map.json",
                {
                    "status": "success",
                    "result_type": "ai2thor_groundtruth_map",
                    "scene": "FloorPlan1",
                    "grid_size_m": 0.25,
                    "robot": {
                        "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                        "rotation": {"y": 90.0},
                        "cameraHorizon": 30.0,
                    },
                    "reachable_positions": [
                        {"x": 0.0, "y": 0.9, "z": 0.0},
                        {"x": 0.25, "y": 0.9, "z": 0.0},
                        {"x": 0.0, "y": 0.9, "z": 0.25},
                    ],
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {"cell": "0,0", "heading": "north"},
                    "cells": {"0,0": {"cell": "0,0", "state": "free", "visited": True}},
                    "edges": {"blocked_edges": []},
                    "recent_actions": ["MoveAhead"],
                },
            )
            write_json(memory / "room-state.json", {"step_count": 4})

            snapshot = AI2ThorGroundTruthMapBackend(memory, source_mode="cache").load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(snapshot.backend, "ai2thor_groundtruth")
        self.assertEqual(snapshot.pose["cell"], "0,0")
        self.assertEqual(snapshot.pose["heading"], "east")
        self.assertEqual(snapshot.coverage["free_cell_count"], 3)
        self.assertEqual(snapshot.coverage["visited_cell_count"], 1)
        self.assertIn("1,0", snapshot.frontiers)
        self.assertIn("0,1", snapshot.frontiers)
        status = snapshot.to_position_status()
        self.assertEqual(status["cells"]["1,0"]["state"], "free")
        self.assertIn("0,0->1,0", status["edges"]["known_open_edges"])
        self.assertEqual(status["map_contract"]["usage_scope"], "offline_mapping_or_simulator_backend_only")

    def test_loader_supports_ai2thor_alias(self) -> None:
        memory = fresh_memory("map-backend-ai2thor-loader")
        try:
            with patch.dict("os.environ", {"ROBOT_MAP_BACKEND": "ai2thor"}):
                backend = load_map_backend(memory)
            self.assertIsInstance(backend, AI2ThorGroundTruthMapBackend)
        finally:
            if memory.exists():
                shutil.rmtree(memory)

    def test_missing_cache_in_cache_mode_is_explicit_error(self) -> None:
        memory = fresh_memory("map-backend-ai2thor-missing")
        try:
            backend = AI2ThorGroundTruthMapBackend(memory, source_mode="cache")
            with self.assertRaises(BackendUnavailableError):
                backend.load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

    def test_last_blocked_edge_is_preserved_as_hard_blocked_groundtruth_edge(self) -> None:
        memory = fresh_memory("map-backend-ai2thor-last-blocked")
        try:
            write_json(
                memory / "ai2thor-groundtruth-map.json",
                {
                    "status": "success",
                    "result_type": "ai2thor_groundtruth_map",
                    "scene": "FloorPlan1",
                    "grid_size_m": 0.25,
                    "robot": {
                        "position": {"x": 1.0, "y": 0.9, "z": -1.0},
                        "rotation": {"y": 0.0},
                    },
                    "reachable_positions": [
                        {"x": 1.0, "y": 0.9, "z": -1.0},
                        {"x": 1.0, "y": 0.9, "z": -0.75},
                        {"x": 1.25, "y": 0.9, "z": -1.0},
                    ],
                },
            )
            write_json(
                memory / "room-state.json",
                {
                    "map_backend": "ai2thor_groundtruth",
                    "blocked_edges": [],
                    "hard_blocked_edges": [],
                    "last_blocked_edge": {
                        "edge": "4,-4->4,-3",
                        "reverse_edge": "4,-3->4,-4",
                        "source": "ai2thor_groundtruth_collision_feedback",
                    },
                },
            )

            snapshot = AI2ThorGroundTruthMapBackend(memory, source_mode="cache").load_snapshot()
            edges = snapshot.to_position_status()["edges"]

            self.assertIn("4,-4->4,-3", edges["blocked_edges"])
            self.assertIn("4,-3->4,-4", edges["blocked_edges"])
            self.assertIn("4,-4->4,-3", edges["hard_blocked_edges"])
        finally:
            if memory.exists():
                shutil.rmtree(memory)

    def test_authoritative_sync_merges_existing_hard_blocked_edges(self) -> None:
        memory = fresh_memory("map-backend-ai2thor-sync-blocked")
        try:
            write_json(
                memory / "ai2thor-groundtruth-map.json",
                {
                    "status": "success",
                    "result_type": "ai2thor_groundtruth_map",
                    "scene": "FloorPlan1",
                    "grid_size_m": 0.25,
                    "robot": {
                        "position": {"x": 1.0, "y": 0.9, "z": -1.0},
                        "rotation": {"y": 0.0},
                    },
                    "reachable_positions": [
                        {"x": 1.0, "y": 0.9, "z": -1.0},
                        {"x": 1.0, "y": 0.9, "z": -0.75},
                        {"x": 1.25, "y": 0.9, "z": -1.0},
                    ],
                },
            )
            write_json(
                memory / "room-state.json",
                {
                    "map_backend": "ai2thor_groundtruth",
                    "blocked_edges": ["0,0->1,0"],
                    "hard_blocked_edges": ["0,0->1,0"],
                    "last_blocked_edge": {
                        "edge": "4,-4->4,-3",
                        "reverse_edge": "4,-3->4,-4",
                    },
                },
            )

            with patch.dict("os.environ", {"ROBOT_MAP_BACKEND": "ai2thor", "ROBOT_AI2THOR_MAP_SOURCE": "cache"}):
                result = sync_authoritative_room_state(memory)
            room = json.loads((memory / "room-state.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "success")
            self.assertIn("0,0->1,0", room["hard_blocked_edges"])
            self.assertIn("4,-4->4,-3", room["hard_blocked_edges"])
            self.assertIn("4,-3->4,-4", room["blocked_edges"])
        finally:
            if memory.exists():
                shutil.rmtree(memory)


if __name__ == "__main__":
    unittest.main()
