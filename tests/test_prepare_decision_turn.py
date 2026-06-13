import argparse
import os
import unittest
from unittest.mock import patch

from scripts.execute_option import ScriptResult
from scripts.prepare_decision_turn import prepare_decision_turn


def args() -> argparse.Namespace:
    return argparse.Namespace(
        task_mode="tidy",
        output="memory/decision-context.json",
        timeout=30,
        vision_timeout=10,
        yolo_timeout=20,
        context_timeout=15,
        observe_retries=0,
        max_candidates=6,
        max_options=12,
        format="compact",
    )


class PrepareDecisionTurnTests(unittest.TestCase):
    @patch("scripts.prepare_decision_turn.run_script")
    @patch("scripts.prepare_decision_turn.update_exploration_goals")
    @patch("scripts.prepare_decision_turn.update_local_costmap_from_refresh")
    @patch("scripts.prepare_decision_turn.run_observe_refresh")
    @patch("scripts.prepare_decision_turn.ensure_tool_mission_active")
    @patch("scripts.prepare_decision_turn.append_trace")
    def test_prepare_success_reuses_observe_and_context_builder(
        self,
        _trace,
        ensure_active,
        observe_refresh,
        update_costmap,
        update_goals,
        run_script,
    ) -> None:
        ensure_active.return_value = {
            "status": "success",
            "result_type": "tool_mission_already_active",
            "activated": False,
        }
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
        update_goals.return_value = {
            "status": "success",
            "result_type": "exploration_goals_updated",
            "active_frontier_goal": {"cell": "1,0"},
        }
        context = {
            "status": "success",
            "schema": "robot_cleaner_decision_context_v1",
            "navigation": {
                "map_backend": {"backend": "map_bundle"},
                "current_pose": {"cell": "0,0", "heading": "north"},
            },
            "worklist": {
                "current_view": {
                    "pickup_candidates": [{"candidate_id": "apple_1", "label": "Apple"}],
                    "place_candidates": [
                        {
                            "candidate_id": "counter_1",
                            "surface_label": "CounterTop",
                            "placement_points": [{"x": index} for index in range(20)],
                        }
                    ],
                    "large_internal_field": ["x"] * 100,
                }
            },
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

        previous_backend = os.environ.get("ROBOT_MAP_BACKEND")
        previous_bundle = os.environ.get("ROBOT_MAP_BUNDLE_PATH")

        result = prepare_decision_turn(args())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "decision_turn_prepared")
        self.assertTrue(result["model_decision_required"])
        self.assertEqual(result["rule_baseline_option_id"], "move:moveahead")
        self.assertEqual(result["option_count"], 1)
        self.assertIn("elapsed_ms", result)
        self.assertIn("context_builder", result)
        self.assertEqual(result["context_builder"]["elapsed_ms"], 0.0)
        self.assertEqual(result["local_costmap"]["result_type"], "local_costmap_updated")
        self.assertEqual(result["exploration_goals"]["result_type"], "exploration_goals_updated")
        self.assertEqual(result["mission_activation"]["status"], "success")
        self.assertIn("runtime_environment", result)
        self.assertIn("map_backend", result["runtime_environment"])
        self.assertTrue(result["full_decision_context_written"])
        self.assertEqual(result["full_decision_context_path"], "memory\\decision-context.json")
        public_context = result["decision_context"]
        self.assertEqual(public_context["full_context_path"], "memory\\decision-context.json")
        self.assertEqual(public_context["navigation"]["map_backend"]["backend"], "map_bundle")
        self.assertNotIn("large_internal_field", public_context["worklist"])
        place_candidate = public_context["worklist"]["place_candidates"][0]
        self.assertEqual(place_candidate["placement_points_count"], 20)
        self.assertEqual(len(place_candidate["placement_points_sample"]), 8)
        self.assertNotIn("placement_points", place_candidate)
        ensure_active.assert_called_once()
        observe_refresh.assert_called_once()
        self.assertEqual(observe_refresh.call_args.kwargs["vision_timeout_seconds"], 10)
        self.assertEqual(observe_refresh.call_args.kwargs["yolo_timeout_seconds"], 20)
        update_costmap.assert_called_once()
        update_goals.assert_called_once()
        self.assertIn("--task-mode", run_script.call_args.args[1])
        self.assertEqual(os.environ.get("ROBOT_MAP_BACKEND"), previous_backend)
        self.assertEqual(os.environ.get("ROBOT_MAP_BUNDLE_PATH"), previous_bundle)

    @patch("scripts.prepare_decision_turn.run_script")
    @patch("scripts.prepare_decision_turn.run_observe_refresh")
    @patch("scripts.prepare_decision_turn.ensure_tool_mission_active")
    @patch("scripts.prepare_decision_turn.append_trace")
    def test_observe_failure_does_not_build_context(self, _trace, ensure_active, observe_refresh, run_script) -> None:
        ensure_active.return_value = {
            "status": "success",
            "result_type": "tool_mission_activated",
            "activated": True,
        }
        observe_refresh.return_value = {"status": "error", "result_type": "observe_refresh_vision_failed"}

        result = prepare_decision_turn(args())

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "observe_refresh")
        self.assertEqual(result["mission_activation"]["result_type"], "tool_mission_activated")
        self.assertEqual(result["required_next"], "retry_prepare_decision_turn_or_stop")
        run_script.assert_not_called()


if __name__ == "__main__":
    unittest.main()
