import unittest
from unittest.mock import patch

from scripts.execute_option import (
    MOVE_SCRIPT,
    ScriptResult,
    candidate_executor_payload,
    find_option,
    run_selected_option,
    validate_context,
    validate_option,
)


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
            ]
        },
    }


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

    def test_explore_frontier_option_reuses_move_validation(self) -> None:
        context = base_context()
        option = find_option(context, "explore:frontier:x0_z1")
        self.assertEqual(validate_option(context, option or {}), [])
        context["perception"]["obstacle_ahead"] = True
        errors = validate_option(context, option or {})
        self.assertTrue(any(item["type"] == "moveahead_blocked_by_perception" for item in errors))

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
