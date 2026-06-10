import argparse
import unittest
from unittest.mock import patch

from scripts.execute_option import ScriptResult
from scripts.prepare_decision_turn import prepare_decision_turn


def args() -> argparse.Namespace:
    return argparse.Namespace(
        task_mode="tidy",
        output="memory/decision-context.json",
        timeout=30,
        observe_retries=1,
        max_candidates=6,
        max_options=12,
        format="compact",
    )


class PrepareDecisionTurnTests(unittest.TestCase):
    @patch("scripts.prepare_decision_turn.run_script")
    @patch("scripts.prepare_decision_turn.update_local_costmap_from_refresh")
    @patch("scripts.prepare_decision_turn.run_observe_refresh")
    @patch("scripts.prepare_decision_turn.append_trace")
    def test_prepare_success_reuses_observe_and_context_builder(
        self,
        _trace,
        observe_refresh,
        update_costmap,
        run_script,
    ) -> None:
        observe_refresh.return_value = {
            "status": "success",
            "result_type": "observe_refresh_executed",
            "vision": {"data": {"status": "success", "image_path": "rgb.jpg", "depth_path": "depth.npy"}},
            "perception": {
                "data": {
                    "status": "success",
                    "result_type": "scene_analyzed_yolo",
                    "perception_backend": "yolo",
                    "candidate_count": 1,
                    "recommended_action": "MoveAhead",
                }
            },
            "perception_written": "memory/yolo-current-rgbd.json",
        }
        update_costmap.return_value = {
            "status": "success",
            "result_type": "local_costmap_updated",
            "moveahead_safe": True,
        }
        context = {
            "status": "success",
            "schema": "robot_cleaner_decision_context_v1",
            "option_set": {
                "rule_baseline_option_id": "move:moveahead",
                "options": [{"option_id": "move:moveahead"}],
            },
            "consistency_warnings": [],
        }
        run_script.return_value = ScriptResult(
            command=["python", "decision_context_builder.py"],
            returncode=0,
            stdout="{}",
            stderr="",
            data=context,
        )

        result = prepare_decision_turn(args())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "decision_turn_prepared")
        self.assertTrue(result["model_decision_required"])
        self.assertEqual(result["rule_baseline_option_id"], "move:moveahead")
        self.assertEqual(result["option_count"], 1)
        self.assertEqual(result["local_costmap"]["result_type"], "local_costmap_updated")
        observe_refresh.assert_called_once()
        update_costmap.assert_called_once()
        self.assertIn("--task-mode", run_script.call_args.args[1])

    @patch("scripts.prepare_decision_turn.run_script")
    @patch("scripts.prepare_decision_turn.run_observe_refresh")
    @patch("scripts.prepare_decision_turn.append_trace")
    def test_observe_failure_does_not_build_context(self, _trace, observe_refresh, run_script) -> None:
        observe_refresh.return_value = {"status": "error", "result_type": "observe_refresh_vision_failed"}

        result = prepare_decision_turn(args())

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "observe_refresh")
        self.assertEqual(result["required_next"], "retry_prepare_decision_turn_or_stop")
        run_script.assert_not_called()


if __name__ == "__main__":
    unittest.main()
