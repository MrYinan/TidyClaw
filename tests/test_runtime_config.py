import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.runtime_config import apply_runtime_environment, configured_map_backend, restore_runtime_environment


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def fresh_dir(name: str) -> Path:
    path = WORKSPACE_ROOT / "memory" / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class RuntimeConfigTests(unittest.TestCase):
    def test_empty_config_defaults_to_ai2thor_groundtruth(self) -> None:
        selection = configured_map_backend({"map": {}})
        self.assertEqual(selection["backend"], "ai2thor_groundtruth")
        self.assertEqual(selection["configured_backend"], "ai2thor_groundtruth")

    def test_map_bundle_config_is_used_when_snapshot_exists(self) -> None:
        bundle = fresh_dir("runtime-config-bundle")
        try:
            (bundle / "map-snapshot.json").write_text(
                json.dumps({"schema": "robot_cleaner_map_snapshot_v1"}),
                encoding="utf-8",
            )
            selection = configured_map_backend(
                {
                    "map": {
                        "backend": "map_bundle",
                        "bundle_path": "memory/runtime-config-bundle",
                        "fallback_backend": "action_odometry",
                    }
                }
            )
        finally:
            if bundle.exists():
                shutil.rmtree(bundle)

        self.assertEqual(selection["backend"], "map_bundle")
        self.assertIsNone(selection["fallback_reason"])

    def test_missing_bundle_falls_back_to_action_odometry(self) -> None:
        selection = configured_map_backend(
            {
                "map": {
                    "backend": "map_bundle",
                    "bundle_path": "memory/runtime-config-missing",
                    "fallback_backend": "action_odometry",
                    "fallback_if_bundle_missing": True,
                }
            }
        )
        self.assertEqual(selection["backend"], "action_odometry")
        self.assertEqual(selection["fallback_reason"], "map_bundle_missing")

    def test_apply_runtime_environment_does_not_override_operator_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ROBOT_MAP_BACKEND": "ai2thor_groundtruth",
                "ROBOT_MAP_BUNDLE_PATH": "custom/path",
            },
            clear=False,
        ):
            result = apply_runtime_environment({"map": {"backend": "map_bundle"}})
            try:
                self.assertEqual(os.environ["ROBOT_MAP_BACKEND"], "ai2thor_groundtruth")
                self.assertEqual(os.environ["ROBOT_MAP_BUNDLE_PATH"], "custom/path")
                self.assertEqual(result["env_applied"], {})
            finally:
                restore_runtime_environment(result["previous_env"])

    def test_apply_runtime_environment_can_force_tool_backend(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ROBOT_MAP_BACKEND": "action_odometry",
                "ROBOT_MAP_BUNDLE_PATH": "old/path",
                "ROBOT_AI2THOR_MAP_SOURCE": "cache",
            },
            clear=False,
        ):
            result = apply_runtime_environment(
                {"map": {"backend": "ai2thor_groundtruth", "source_mode": "live"}},
                override_existing=True,
            )
            try:
                self.assertEqual(os.environ["ROBOT_MAP_BACKEND"], "ai2thor_groundtruth")
                self.assertEqual(os.environ["ROBOT_AI2THOR_MAP_SOURCE"], "live")
                self.assertEqual(result["env_applied"]["ROBOT_MAP_BACKEND"], "ai2thor_groundtruth")
            finally:
                restore_runtime_environment(result["previous_env"])


if __name__ == "__main__":
    unittest.main()
