from __future__ import annotations

import importlib.util
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MOVE_ROBOT_SCRIPT = WORKSPACE_ROOT / "skills" / "move-robot" / "scripts" / "move_robot.py"


def fresh_memory(name: str) -> Path:
    memory = WORKSPACE_ROOT / "memory" / name
    if memory.exists():
        shutil.rmtree(memory)
    memory.mkdir(parents=True, exist_ok=True)
    return memory


def load_move_robot_module():
    spec = importlib.util.spec_from_file_location("move_robot_skill_for_test", MOVE_ROBOT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MoveRobotPublicBlockedEdgeTests(unittest.TestCase):
    def test_failed_translation_records_public_groundtruth_blocked_edge(self) -> None:
        module = load_move_robot_module()
        memory = fresh_memory("move-robot-public-blocked-edge")
        try:
            (memory / "room-state.json").write_text(
                json.dumps(
                    {
                        "map_backend": "ai2thor_groundtruth",
                        "blocked_edges": ["0,0->1,0"],
                        "hard_blocked_edges": [],
                    }
                ),
                encoding="utf-8",
            )
            pose = {
                "last_cell": "4,-4",
                "last_heading": "north",
                "map_backend": "ai2thor_groundtruth",
                "coordinate_mode": "ai2thor_groundtruth_grid",
            }

            with patch.object(module, "MEMORY_DIR", memory), patch.object(
                module, "_authoritative_pose_status", return_value=pose
            ):
                result = module._record_public_blocked_edge_from_authoritative_pose(
                    action="MoveAhead",
                    success=False,
                    failure_reason="Stool blocked movement",
                )

            room = json.loads((memory / "room-state.json").read_text(encoding="utf-8"))

            self.assertEqual(result["status"], "success")
            self.assertIn("4,-4->4,-3", room["blocked_edges"])
            self.assertIn("4,-3->4,-4", room["blocked_edges"])
            self.assertIn("4,-4->4,-3", room["hard_blocked_edges"])
            self.assertEqual(room["last_blocked_edge"]["source"], "ai2thor_groundtruth_collision_feedback")
        finally:
            if memory.exists():
                shutil.rmtree(memory)


if __name__ == "__main__":
    unittest.main()
