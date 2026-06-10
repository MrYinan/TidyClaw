from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.global_planner import AStarGlobalPlanner
from scripts.position_map_core import PositionMap


class ActionExpansionTests(unittest.TestCase):
    def test_global_planner_uses_relative_cardinal_translations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = AStarGlobalPlanner(Path(tmp))
            left, _ = planner._action_for_path(current_cell="0,0", current_heading="north", path=["0,0", "-1,0"], target_heading=None)
            right, _ = planner._action_for_path(current_cell="0,0", current_heading="north", path=["0,0", "1,0"], target_heading=None)
            back, _ = planner._action_for_path(current_cell="0,0", current_heading="north", path=["0,0", "0,-1"], target_heading=None)
            ahead, _ = planner._action_for_path(current_cell="0,0", current_heading="north", path=["0,0", "0,1"], target_heading=None)
            self.assertEqual(left, "MoveLeft")
            self.assertEqual(right, "MoveRight")
            self.assertEqual(back, "MoveBack")
            self.assertEqual(ahead, "MoveAhead")

    def test_position_map_tracks_lateral_odometry_without_changing_heading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = PositionMap(Path(tmp))
            manager.reset()
            right = manager.record_action(action="MoveRight", success=True, vision={}, analysis={}, persist=True)
            self.assertEqual(right["last_cell"], "1,0")
            self.assertEqual(right["last_heading"], "north")
            left = manager.record_action(action="MoveLeft", success=True, vision={}, analysis={}, persist=True)
            self.assertEqual(left["last_cell"], "0,0")
            self.assertEqual(left["last_heading"], "north")

    def test_position_map_loads_npy_when_depth_field_is_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            depth_path = memory_dir / "depth.npy"
            depth = np.full((60, 60), np.nan, dtype=np.float32)
            depth[24, 24] = 0.50
            np.save(depth_path, depth)
            manager = PositionMap(memory_dir)
            data = manager.load()
            result = manager.update_from_depth(
                data,
                vision={
                    "depth": {"encoding": "base64_npy_float32", "height": 60, "width": 60},
                    "depth_path": str(depth_path),
                    "camera": {
                        "fx": 30.0,
                        "fy": 30.0,
                        "cx": 30.0,
                        "cy": 30.0,
                        "camera_height_m": 0.90,
                        "camera_horizon_deg": 30.0,
                    },
                },
                analysis={},
                step=1,
            )
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["marked_occupied"], 0)

    def test_camera_pitch_action_does_not_move_pose(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = PositionMap(Path(tmp))
            manager.reset()
            status = manager.record_action(action="LookDown", success=True, vision={}, analysis={}, persist=True)
            self.assertEqual(status["last_cell"], "0,0")
            self.assertEqual(status["last_heading"], "north")


if __name__ == "__main__":
    unittest.main()
