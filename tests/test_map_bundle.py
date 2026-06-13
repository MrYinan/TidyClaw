from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.map_backend import MapBundleBackend
from scripts.map_bundle.export_map_bundle import export_map_bundle


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


class MapBundleTests(unittest.TestCase):
    def test_export_bundle_and_load_with_map_bundle_backend(self) -> None:
        memory = fresh_memory("map-bundle-export")
        output = memory / "maps" / "current"
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
                        "rotation": {"y": 0.0},
                    },
                    "reachable_positions": [
                        {"x": 0.0, "y": 0.9, "z": 0.0},
                        {"x": 0.25, "y": 0.9, "z": 0.0},
                    ],
                },
            )
            write_json(
                memory / "position-map.json",
                {"pose": {"cell": "0,0", "heading": "north"}, "cells": {"0,0": {"visited": True}}},
            )

            manifest = export_map_bundle(
                memory_dir=memory,
                output_dir=output,
                backend="ai2thor_groundtruth",
            )
            snapshot = MapBundleBackend(memory, bundle_path=output).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["backend"], "ai2thor_groundtruth")
        self.assertEqual(snapshot.backend, "map_bundle")
        self.assertEqual(snapshot.pose["cell"], "0,0")
        self.assertEqual(snapshot.coverage["free_cell_count"], 2)
        self.assertIn("1,0", snapshot.to_position_status()["cells"])
        self.assertEqual(snapshot.source_info["map_bundle"]["loaded"], True)

    def test_map_bundle_overlays_live_pose_instead_of_export_pose(self) -> None:
        memory = fresh_memory("map-bundle-live-pose")
        output = memory / "maps" / "current"
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_json(
                output / "map-snapshot.json",
                {
                    "schema": "robot_cleaner_map_snapshot_v1",
                    "backend": "ai2thor_groundtruth",
                    "generated_at": "2026-06-12T00:00:00+08:00",
                    "map_frame": {"coordinate_mode": "ai2thor_groundtruth_grid", "cell_size_m": 0.25},
                    "pose": {"cell": "-4,0", "heading": "south", "pose_confidence": 1.0},
                    "coverage": {"free_cell_count": 3, "visited_cell_count": 1},
                    "frontiers": ["-3,0"],
                    "edges": {"known_open_edges": [], "blocked_edges": []},
                    "position_status": {
                        "cells": {
                            "-4,0": {"cell": "-4,0", "state": "free", "visited": True},
                            "1,-2": {"cell": "1,-2", "state": "free", "visited": False},
                            "2,-2": {"cell": "2,-2", "state": "free", "visited": False},
                        }
                    },
                    "room_state": {},
                },
            )
            write_json(
                memory / "position-map.json",
                {
                    "pose": {
                        "cell": "1,-2",
                        "heading": "east",
                        "pose_confidence": 0.74,
                        "position_uncertainty_cells": 1.2,
                    },
                    "cells": {
                        "1,-2": {"cell": "1,-2", "state": "free", "visited": True, "seen_count": 4}
                    },
                },
            )
            write_json(memory / "room-state.json", {"last_cell": "1,-2", "last_heading": "east"})

            snapshot = MapBundleBackend(memory, bundle_path=output).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertEqual(snapshot.pose["cell"], "1,-2")
        self.assertEqual(snapshot.pose["heading"], "east")
        self.assertEqual(snapshot.coverage["free_cell_count"], 3)
        self.assertGreaterEqual(snapshot.coverage["visited_cell_count"], 1)
        status = snapshot.to_position_status()
        self.assertEqual(status["pose"]["cell"], "1,-2")
        self.assertEqual(status["cells"]["1,-2"]["visited"], True)
        self.assertIn("map_bundle_overlay", status)

    def test_map_bundle_ignores_exported_active_route_when_live_room_has_none(self) -> None:
        memory = fresh_memory("map-bundle-ignore-exported-route")
        output = memory / "maps" / "current"
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_json(
                output / "map-snapshot.json",
                {
                    "schema": "robot_cleaner_map_snapshot_v1",
                    "backend": "ai2thor_groundtruth",
                    "generated_at": "2026-06-12T00:00:00+08:00",
                    "map_frame": {"coordinate_mode": "ai2thor_groundtruth_grid", "cell_size_m": 0.25},
                    "pose": {"cell": "0,0", "heading": "north"},
                    "coverage": {"free_cell_count": 2, "visited_cell_count": 1},
                    "frontiers": ["1,0"],
                    "edges": {},
                    "active_frontier_goal": {"cell": "9,9", "status": "active"},
                    "active_route": {"status": "active", "goal_cell": "9,9"},
                    "position_status": {
                        "cells": {
                            "0,0": {"cell": "0,0", "state": "free", "visited": True},
                            "1,0": {"cell": "1,0", "state": "free", "visited": False},
                        },
                        "active_frontier_goal": {"cell": "9,9", "status": "active"},
                        "active_route": {"status": "active", "goal_cell": "9,9"},
                    },
                    "room_state": {
                        "active_frontier_goal": {"cell": "9,9", "status": "active"},
                        "active_route": {"status": "active", "goal_cell": "9,9"},
                    },
                },
            )
            write_json(
                memory / "position-map.json",
                {"pose": {"cell": "0,0", "heading": "north"}, "cells": {"0,0": {"visited": True}}},
            )
            write_json(memory / "room-state.json", {"last_cell": "0,0", "last_heading": "north"})

            snapshot = MapBundleBackend(memory, bundle_path=output).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertIsNone(snapshot.active_frontier_goal)
        self.assertIsNone(snapshot.active_route)
        status = snapshot.to_position_status()
        self.assertIsNone(status["active_frontier_goal"])
        self.assertIsNone(status["active_route"])

    def test_map_bundle_drops_live_active_route_without_live_active_goal(self) -> None:
        memory = fresh_memory("map-bundle-drop-orphan-live-route")
        output = memory / "maps" / "current"
        try:
            output.mkdir(parents=True, exist_ok=True)
            write_json(
                output / "map-snapshot.json",
                {
                    "schema": "robot_cleaner_map_snapshot_v1",
                    "backend": "ai2thor_groundtruth",
                    "generated_at": "2026-06-12T00:00:00+08:00",
                    "map_frame": {"coordinate_mode": "ai2thor_groundtruth_grid", "cell_size_m": 0.25},
                    "pose": {"cell": "0,0", "heading": "north"},
                    "coverage": {"free_cell_count": 2, "visited_cell_count": 1},
                    "frontiers": ["1,0"],
                    "edges": {},
                    "position_status": {
                        "cells": {
                            "0,0": {"cell": "0,0", "state": "free", "visited": True},
                            "1,0": {"cell": "1,0", "state": "free", "visited": False},
                        }
                    },
                    "room_state": {},
                },
            )
            write_json(
                memory / "position-map.json",
                {"pose": {"cell": "0,0", "heading": "north"}, "cells": {"0,0": {"visited": True}}},
            )
            write_json(
                memory / "room-state.json",
                {
                    "last_cell": "0,0",
                    "last_heading": "north",
                    "active_route": {"status": "blocked", "goal_cell": "2,-2"},
                },
            )

            snapshot = MapBundleBackend(memory, bundle_path=output).load_snapshot()
        finally:
            if memory.exists():
                shutil.rmtree(memory)

        self.assertIsNone(snapshot.active_frontier_goal)
        self.assertIsNone(snapshot.active_route)
        self.assertIsNone(snapshot.to_position_status()["active_route"])


if __name__ == "__main__":
    unittest.main()
