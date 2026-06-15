import unittest
import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.execute_option import (
    MOVE_SCRIPT,
    ScriptResult,
    candidate_executor_payload,
    find_option,
    mark_context_stale_after_execution,
    run_selected_option,
    validate_context,
    validate_context_runtime_backend,
    validate_option,
)


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


def base_context() -> dict:
    return {
        "schema": "robot_cleaner_decision_context_v1",
        "forbidden_private_fields_absent": True,
        "perception": {
            "structured_perception_available": True,
            "obstacle_ahead": False,
        },
        "navigation": {
            "local_costmap": {
                "action_safety": {
                    "MoveAhead": {"safe": True, "reason": "clear_swept_volume"},
                    "RotateLeft": {"safe": True, "reason": "clear_swept_volume"},
                }
            }
        },
        "worklist": {"held_object": {"holding_object": False}},
        "option_set": {
            "options": [
                {
                    "option_id": "move:moveahead",
                    "kind": "move_action",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "MoveAhead",
                    "executable_now": True,
                },
                {
                    "option_id": "done:probe",
                    "kind": "completion_probe",
                    "physical_action": False,
                    "tool": "state-manager/report",
                    "executable_now": True,
                },
                {
                    "option_id": "explore:frontier_cluster:x0_z1",
                    "kind": "explore_frontier_cluster",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "MoveAhead",
                    "executable_now": True,
                    "one_step_only": True,
                    "resolved_step_option_id": "move:moveahead",
                    "frontier_cluster_target": {"cell": "0,1", "cluster_size": 2},
                    "route_step": {
                        "route_id": "route-frontier-x0-z1-abcd1234",
                        "step_index": 0,
                        "action": "MoveAhead",
                        "current_cell": "0,0",
                        "current_heading": "north",
                        "target_cell": "0,1",
                        "next_cell": "0,1",
                        "goal_cell": "0,1",
                        "desired_heading": "north",
                    },
                },
                {
                    "option_id": "explore:frontier:x0_z1",
                    "kind": "explore_frontier",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "MoveAhead",
                    "executable_now": True,
                    "one_step_only": True,
                    "resolved_step_option_id": "move:moveahead",
                    "frontier_target": {"cell": "0,1", "distance_steps": 1},
                },
                {
                    "option_id": "explore:route_step:route_frontier_x0_z1_abcd1234_0",
                    "kind": "explore_route_step",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "MoveAhead",
                    "executable_now": True,
                    "one_step_only": True,
                    "resolved_step_option_id": "move:moveahead",
                    "route_step": {
                        "route_id": "route-frontier-x0-z1-abcd1234",
                        "step_index": 0,
                        "action": "MoveAhead",
                        "current_cell": "0,0",
                        "current_heading": "north",
                        "target_cell": "0,1",
                        "next_cell": "0,1",
                        "goal_cell": "0,1",
                        "desired_heading": "north",
                    },
                },
                {
                    "option_id": "explore:waypoint:x0_z1",
                    "kind": "explore_waypoint",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "MoveAhead",
                    "executable_now": True,
                    "one_step_only": True,
                    "resolved_step_option_id": "move:moveahead",
                    "waypoint_target": {"cell": "0,1", "purpose": "break_rotation_loop_via_safe_translation"},
                },
                {
                    "option_id": "explore:inspection_waypoint:wp_front",
                    "kind": "explore_inspection_waypoint",
                    "physical_action": True,
                    "tool": "waypoint-planner + move-robot",
                    "action": "plan-to-inspection-waypoint",
                    "executable_now": True,
                    "one_step_only": True,
                    "waypoint_target": {"waypoint_id": "wp_front", "cell": "0,1", "purpose": "coverage_scan"},
                },
                {
                    "option_id": "continue:active_waypoint_goal",
                    "kind": "continue_active_waypoint_goal",
                    "physical_action": True,
                    "tool": "waypoint-planner + move-robot",
                    "action": "continue-active-waypoint-goal",
                    "executable_now": True,
                    "one_step_only": True,
                    "active_waypoint_goal": {"waypoint_id": "wp_front", "cell": "0,1", "status": "active"},
                },
                {
                    "option_id": "recover:lookdown",
                    "kind": "recovery_action",
                    "physical_action": True,
                    "tool": "move-robot",
                    "action": "LookDown",
                    "executable_now": True,
                    "one_step_only": True,
                    "dead_end_context": {"current_cell": "1,4"},
                },
            ]
        },
    }


def make_tmp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class ExecuteOptionTests(unittest.TestCase):
    def test_find_option_requires_existing_id(self) -> None:
        context = base_context()
        self.assertIsNotNone(find_option(context, "move:moveahead"))
        self.assertIsNone(find_option(context, "move:does_not_exist"))

    def test_valid_move_option_passes_without_warnings(self) -> None:
        context = base_context()
        option = find_option(context, "move:moveahead")
        self.assertEqual(validate_context(context), [])
        self.assertEqual(validate_option(context, option or {}), [])

    def test_context_marked_stale_blocks_option_reuse(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = make_tmp_dir("stale-context")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        context_path = tmp / "decision-context.json"
        context = base_context()
        context["context_lifecycle"] = {
            "valid_for_execution": True,
            "stale_after_option_execution": False,
        }
        context_path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")

        mark = mark_context_stale_after_execution(
            context_path,
            option_id="move:moveahead",
            execution={"status": "success", "result_type": "option_move_executed"},
        )
        reloaded = json.loads(context_path.read_text(encoding="utf-8"))
        errors = validate_context(reloaded)

        self.assertEqual(mark["status"], "success")
        self.assertFalse(reloaded["context_lifecycle"]["valid_for_execution"])
        self.assertTrue(reloaded["context_lifecycle"]["stale_after_option_execution"])
        self.assertTrue(
            any(item["type"] == "decision_context_stale_after_option_execution" for item in errors)
        )
        self.assertTrue(any(item.get("required_next") == "robot_cleaner_prepare_decision_turn" for item in errors))

    def test_physical_option_blocks_on_consistency_warning(self) -> None:
        context = base_context()
        context["consistency_warnings"] = [{"type": "source_time_skew"}]
        option = find_option(context, "move:moveahead")
        errors = validate_option(context, option or {})
        self.assertEqual(errors[0]["type"], "context_consistency_warning")

    def test_advisory_consistency_warning_does_not_block_physical_option(self) -> None:
        context = base_context()
        context["consistency_warnings"] = [{"type": "source_time_skew", "blocking": False}]
        option = find_option(context, "move:moveahead")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_context_runtime_backend_mismatch_requires_fresh_prepare(self) -> None:
        context = base_context()
        context["navigation"]["map_backend"] = {"backend": "map_bundle"}

        error = validate_context_runtime_backend(
            context,
            {"map_backend": {"backend": "ai2thor_groundtruth"}},
        )

        self.assertIsNotNone(error)
        self.assertEqual(error["type"], "decision_context_map_backend_mismatch")
        self.assertEqual(error["required_next"], "robot_cleaner_prepare_decision_turn")

    def test_done_probe_not_blocked_by_consistency_warning(self) -> None:
        context = base_context()
        context["consistency_warnings"] = [{"type": "source_time_skew"}]
        option = find_option(context, "done:probe")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_moveahead_blocks_when_perception_obstacle_ahead(self) -> None:
        context = base_context()
        context["perception"]["obstacle_ahead"] = True
        option = find_option(context, "move:moveahead")
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "moveahead_blocked_by_perception" for item in errors))

    def test_committed_route_step_uses_costmap_when_perception_obstacle_ahead_conflicts(self) -> None:
        context = base_context()
        context["perception"]["obstacle_ahead"] = True
        option = dict(find_option(context, "explore:route_step:route_frontier_x0_z1_abcd1234_0") or {})
        option["safety_source"] = "active-route-costmap"

        errors = validate_option(context, option)

        self.assertFalse(any(item["type"] == "moveahead_blocked_by_perception" for item in errors))
        self.assertEqual(errors, [])

    def test_move_action_blocks_on_low_observed_ratio(self) -> None:
        context = base_context()
        context["navigation"]["local_costmap"]["action_safety"]["MoveAhead"] = {
            "safe": True,
            "reason": "clear_swept_volume",
            "observed_ratio": 0.2,
            "min_observed_ratio": 0.3,
        }
        option = find_option(context, "move:moveahead")
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "move_action_low_observed_ratio" for item in errors))
        self.assertTrue(any(item.get("required_next") == "observe_or_rotate_before_translation" for item in errors))

    def test_explore_frontier_option_reuses_move_validation(self) -> None:
        context = base_context()
        option = find_option(context, "explore:frontier:x0_z1")
        self.assertEqual(validate_option(context, option or {}), [])
        context["perception"]["obstacle_ahead"] = True
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "moveahead_blocked_by_perception" for item in errors))

    def test_explore_frontier_blocks_on_low_resolved_action_observed_ratio(self) -> None:
        context = base_context()
        context["navigation"]["local_costmap"]["action_safety"]["MoveAhead"] = {
            "safe": True,
            "reason": "clear_swept_volume",
            "observed_ratio": 0.2,
            "min_observed_ratio": 0.3,
        }
        option = find_option(context, "explore:frontier:x0_z1")
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "move_action_low_observed_ratio" for item in errors))

    def test_explore_route_step_option_reuses_move_validation(self) -> None:
        context = base_context()
        option = find_option(context, "explore:route_step:route_frontier_x0_z1_abcd1234_0")
        self.assertEqual(validate_option(context, option or {}), [])
        context["navigation"]["local_costmap"]["action_safety"]["MoveAhead"] = {
            "safe": True,
            "reason": "clear_swept_volume",
            "observed_ratio": 0.2,
            "min_observed_ratio": 0.3,
        }
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "move_action_low_observed_ratio" for item in errors))

    def test_explore_waypoint_option_reuses_move_validation(self) -> None:
        context = base_context()
        option = find_option(context, "explore:waypoint:x0_z1")
        self.assertEqual(validate_option(context, option or {}), [])
        context["navigation"]["local_costmap"]["action_safety"]["MoveAhead"] = {
            "safe": True,
            "reason": "clear_swept_volume",
            "observed_ratio": 0.2,
            "min_observed_ratio": 0.3,
        }
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "move_action_low_observed_ratio" for item in errors))

    def test_inspection_waypoint_option_validates_goal_id(self) -> None:
        context = base_context()
        option = find_option(context, "explore:inspection_waypoint:wp_front")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_continue_active_waypoint_option_validates_active_goal(self) -> None:
        context = base_context()
        option = find_option(context, "continue:active_waypoint_goal")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_continue_active_waypoint_allows_navigation_only_perception(self) -> None:
        context = base_context()
        context["perception"] = {
            "status": "success",
            "result_type": "navigation_only_observed",
            "perception_mode": "navigation_only",
            "structured_perception_available": False,
            "online_safe": True,
        }
        option = find_option(context, "continue:active_waypoint_goal")

        self.assertEqual(validate_option(context, option or {}), [])

    def test_raw_move_still_blocks_on_navigation_only_perception(self) -> None:
        context = base_context()
        context["perception"] = {
            "status": "success",
            "result_type": "navigation_only_observed",
            "perception_mode": "navigation_only",
            "structured_perception_available": False,
            "online_safe": True,
        }
        option = find_option(context, "move:moveahead")
        errors = validate_option(context, option or {})

        self.assertTrue(any(item["type"] == "structured_perception_unavailable" for item in errors))

    def test_explore_frontier_executes_one_resolved_move_step(self) -> None:
        context = base_context()
        option = find_option(context, "explore:frontier:x0_z1")
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.run_script", return_value=script_result) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_explore_frontier_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        self.assertEqual(result["frontier_target"]["cell"], "0,1")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "MoveAhead")

    def test_explore_inspection_waypoint_executes_planned_one_step(self) -> None:
        context = base_context()
        option = find_option(context, "explore:inspection_waypoint:wp_front")
        plan = {
            "status": "active",
            "result_type": "waypoint_plan_ready",
            "waypoint": {"waypoint_id": "wp_front", "cell": "0,1"},
            "next_action": "MoveAhead",
            "next_cell": "0,1",
            "route": {
                "status": "active",
                "waypoint_id": "wp_front",
                "goal_cell": "0,1",
                "next_action": "MoveAhead",
                "route_step": {"route_id": "route-wp-front", "goal_cell": "0,1", "action": "MoveAhead"},
            },
        }
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.plan_to_inspection_waypoint", return_value=plan) as planner, patch(
            "scripts.execute_option.run_script", return_value=script_result
        ) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        planner.assert_called_once()
        self.assertEqual(planner.call_args.args[0], "wp_front")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_explore_inspection_waypoint_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        self.assertEqual(result["waypoint_plan"]["waypoint"]["waypoint_id"], "wp_front")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "MoveAhead")

    def test_continue_active_waypoint_executes_planned_one_step(self) -> None:
        context = base_context()
        option = find_option(context, "continue:active_waypoint_goal")
        plan = {
            "status": "active",
            "result_type": "waypoint_plan_ready",
            "waypoint": {"waypoint_id": "wp_front", "cell": "0,1"},
            "next_action": "MoveAhead",
            "next_cell": "0,1",
            "route": {
                "status": "active",
                "waypoint_id": "wp_front",
                "goal_cell": "0,1",
                "next_action": "MoveAhead",
                "route_step": {"route_id": "route-wp-front", "goal_cell": "0,1", "action": "MoveAhead"},
            },
        }
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.continue_active_waypoint_goal", return_value=plan) as planner, patch(
            "scripts.execute_option.run_script", return_value=script_result
        ) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ):
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        planner.assert_called_once()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_continue_active_waypoint_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)

    def test_explore_frontier_cluster_executes_one_resolved_move_step(self) -> None:
        context = base_context()
        option = find_option(context, "explore:frontier_cluster:x0_z1")
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.run_script", return_value=script_result) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_explore_frontier_cluster_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        self.assertEqual(result["frontier_cluster_target"]["cell"], "0,1")
        self.assertEqual(result["route_step"]["goal_cell"], "0,1")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "MoveAhead")

    def test_explore_route_step_executes_one_resolved_move_step(self) -> None:
        context = base_context()
        option = find_option(context, "explore:route_step:route_frontier_x0_z1_abcd1234_0")
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.run_script", return_value=script_result) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_explore_route_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        self.assertEqual(result["route_step"]["goal_cell"], "0,1")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "MoveAhead")

    def test_explore_waypoint_executes_one_resolved_move_step(self) -> None:
        context = base_context()
        option = find_option(context, "explore:waypoint:x0_z1")
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "MoveAhead"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.run_script", return_value=script_result) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_explore_waypoint_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "MoveAhead")
        self.assertEqual(result["waypoint_target"]["cell"], "0,1")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "MoveAhead"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "MoveAhead")

    def test_recovery_action_executes_one_validated_move_step(self) -> None:
        context = base_context()
        option = find_option(context, "recover:lookdown")
        script_result = ScriptResult(
            command=["python", str(MOVE_SCRIPT), "--action", "LookDown"],
            returncode=0,
            stdout='{"status":"success","lastActionSuccess":true}',
            stderr="",
            data={"status": "success", "lastActionSuccess": True},
        )
        with patch("scripts.execute_option.run_script", return_value=script_result) as run_script, patch(
            "scripts.execute_option.sync_option_result",
            return_value={"status": "success", "result_type": "move_state_synchronized"},
        ) as sync:
            result = run_selected_option(
                context,
                option or {},
                timeout_seconds=5,
                dry_run=False,
                strict_visual_grounding=True,
            )

        self.assertEqual(validate_option(context, option or {}), [])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result_type"], "option_recovery_step_executed")
        self.assertTrue(result["one_step_only"])
        self.assertEqual(result["resolved_action"], "LookDown")
        self.assertEqual(result["dead_end_context"]["current_cell"], "1,4")
        run_script.assert_called_once_with(MOVE_SCRIPT, ["--action", "LookDown"], timeout_seconds=5)
        synced_option = sync.call_args.kwargs["option"]
        self.assertEqual(synced_option["kind"], "move_action")
        self.assertEqual(synced_option["action"], "LookDown")

    def test_place_option_blocks_when_candidate_needs_alignment(self) -> None:
        context = base_context()
        context["worklist"] = {
            "held_object": {"holding_object": True},
            "current_view": {
                "surface_candidates": [
                    {
                        "candidate_id": "surface-left",
                        "label": "pc_surface",
                        "actionability": {
                            "reachable": True,
                            "place_now": False,
                            "visual_place_ready": True,
                            "final_place_ready": False,
                            "needs_alignment": True,
                        },
                        "executor_checks": {"precheck_ok": False, "precheck_supported": True},
                    }
                ]
            },
        }
        context["option_set"]["options"].append(
            {
                "option_id": "place:surface_left",
                "kind": "service_action",
                "physical_action": True,
                "tool": "place-object",
                "action": "place-object",
                "candidate_ref": {"candidate_id": "surface-left"},
                "executable_now": True,
            }
        )

        option = find_option(context, "place:surface_left")
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "candidate_not_place_executor_ready" for item in errors))
        self.assertTrue(any(item.get("required_next") == "align_receptacle_then_observe" for item in errors))

    def test_place_precheck_allows_offcenter_pointcloud_candidate(self) -> None:
        context = base_context()
        context["worklist"] = {
            "held_object": {"holding_object": True},
            "current_view": {
                "surface_candidates": [
                    {
                        "candidate_id": "surface-left",
                        "surface_candidate_source": "pointcloud_plane_grid_completion",
                        "label": "pc_surface",
                        "actionability": {
                            "reachable": True,
                            "blocked": False,
                            "place_now": False,
                            "visual_place_ready": True,
                            "final_place_ready": False,
                            "affordance_ready": True,
                            "needs_alignment": True,
                        },
                        "interaction_point": {"x": 156.0, "y": 340.0},
                        "placement_points": [
                            {"x": 156.0, "y": 340.0, "center_3d": {"x": -0.35, "y": 0.42, "z": 0.62}}
                        ],
                        "placement_safety_contract": {
                            "version": "plane_local_grid_v1",
                            "requires_exact_target_execution": True,
                        },
                        "executor_checks": {"precheck_ok": False, "precheck_supported": True},
                    }
                ]
            },
        }
        context["consistency_warnings"] = [{"type": "source_time_skew"}]
        context["option_set"]["options"].append(
            {
                "option_id": "place_precheck:surface_left",
                "kind": "place_precheck",
                "physical_action": False,
                "tool": "place-object",
                "action": "place-precheck",
                "candidate_ref": {"candidate_id": "surface-left"},
                "executable_now": True,
            }
        )

        option = find_option(context, "place_precheck:surface_left")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_place_option_passes_for_executor_ready_candidate(self) -> None:
        context = base_context()
        context["worklist"] = {
            "held_object": {"holding_object": True},
            "current_view": {
                "surface_candidates": [
                    {
                        "candidate_id": "surface-ready",
                        "label": "pc_surface",
                        "actionability": {
                            "reachable": True,
                            "place_now": True,
                            "visual_place_ready": True,
                            "final_place_ready": True,
                            "needs_alignment": False,
                        },
                        "executor_checks": {"precheck_ok": True, "precheck_supported": True},
                    }
                ]
            },
        }
        context["option_set"]["options"].append(
            {
                "option_id": "place:surface_ready",
                "kind": "service_action",
                "physical_action": True,
                "tool": "place-object",
                "action": "place-object",
                "candidate_ref": {"candidate_id": "surface-ready"},
                "executable_now": True,
            }
        )

        option = find_option(context, "place:surface_ready")
        self.assertEqual(validate_option(context, option or {}), [])

    def test_place_payload_preserves_pointcloud_execution_contract(self) -> None:
        candidate = {
            "candidate_id": "surface-ready",
            "surface_candidate_id": "pc_grid:surface-ready",
            "surface_candidate_source": "pointcloud_plane_grid_completion",
            "label": "pc_surface",
            "task_class": "place_receptacle",
            "bbox": {"x": 10, "y": 20, "w": 100, "h": 80},
            "geometry": {"bbox": {"x": 10, "y": 20, "w": 100, "h": 80}, "distance_m": 0.8},
            "actionability": {
                "reachable": True,
                "place_now": True,
                "visual_place_ready": True,
                "final_place_ready": True,
                "affordance_ready": True,
            },
            "interaction_point": {"x": 104.0, "y": 328.0},
            "placement_points": [
                {
                    "x": 104.0,
                    "y": 328.0,
                    "rank": 1,
                    "center_3d": {
                        "x": -0.5326,
                        "y": 0.4252,
                        "z": 0.6706,
                        "ground_forward_m": 0.6706,
                    },
                }
            ],
            "placement_safety_contract": {
                "version": "plane_local_grid_v1",
                "clearance_owner": "pointcloud_plane_local_grid",
                "target_coordinate_frame": "camera_relative_x_right_y_height_z_forward",
                "requires_exact_target_execution": True,
                "grid_occupancy_clear": True,
                "grid_edge_eroded": True,
            },
            "free_space_completion": {"mode": "plane_local_2d_grid"},
            "geometry_checks": {
                "distance_ok": True,
                "grid_occupancy_clear": True,
                "grid_edge_eroded": True,
            },
            "occupancy_checks": {
                "blocked": False,
                "free_space_grid_completion": True,
            },
        }

        payload = candidate_executor_payload(candidate, role="place")

        self.assertEqual(payload["interaction_point"], {"x": 104.0, "y": 328.0})
        self.assertEqual(payload["surface_candidate_source"], "pointcloud_plane_grid_completion")
        self.assertEqual(payload["placement_points"][0]["center_3d"]["ground_forward_m"], 0.6706)
        self.assertTrue(payload["placement_safety_contract"]["requires_exact_target_execution"])
        self.assertTrue(payload["geometry_checks"]["grid_occupancy_clear"])
        self.assertTrue(payload["occupancy_checks"]["free_space_grid_completion"])


if __name__ == "__main__":
    unittest.main()
