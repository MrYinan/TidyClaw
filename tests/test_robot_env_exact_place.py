from __future__ import annotations

import sys
import types
import unittest


ai2thor_module = types.ModuleType("ai2thor")
ai2thor_controller_module = types.ModuleType("ai2thor.controller")
ai2thor_controller_module.Controller = object
ai2thor_module.controller = ai2thor_controller_module
sys.modules.setdefault("ai2thor", ai2thor_module)
sys.modules.setdefault("ai2thor.controller", ai2thor_controller_module)

from back.robot_env import RobotEnvironment  # noqa: E402


class FakeEvent:
    def __init__(self, metadata: dict) -> None:
        self.metadata = metadata


def grid_candidate() -> dict:
    return {
        "id": "pc_grid:test",
        "surface_candidate_id": "pc_grid:test",
        "label": "pc_surface",
        "raw_label": "CounterTop",
        "task_semantic_class": "place_receptacle",
        "surface_candidate_source": "pointcloud_plane_grid_completion",
        "visual_place_ready": True,
        "affordance_ready": True,
        "reachable": True,
        "blocked": False,
        "rejection_reasons": [],
        "interaction_point": {"x": 300.0, "y": 300.0},
        "placement_points": [
            {
                "rank": 1,
                "x": 300.0,
                "y": 300.0,
                "center_3d": {
                    "x": 0.0,
                    "y": 0.5,
                    "z": 0.5,
                    "ground_forward_m": 0.5,
                },
            }
        ],
        "free_space_completion": {"mode": "plane_local_2d_grid"},
        "placement_safety_contract": {
            "version": "plane_local_grid_v1",
            "clearance_owner": "pointcloud_plane_local_grid",
            "grid_occupancy_clear": True,
            "grid_edge_eroded": True,
            "requires_exact_target_execution": True,
            "target_coordinate_frame": "camera_relative_x_right_y_height_z_forward",
            "camera_height_m": 0.9,
        },
        "geometry_checks": {
            "grid_occupancy_clear": True,
            "grid_edge_eroded": True,
            "distance_ok": True,
        },
        "occupancy_checks": {
            "free_space_grid_completion": True,
            "blocked": False,
        },
    }


class RecordingController:
    def __init__(self, *, apple_x: float = 0.01, spawn_x: float = 0.01) -> None:
        self.calls: list[dict] = []
        self.apple_x = apple_x
        self.spawn_x = spawn_x

    def step(self, **kwargs: object) -> FakeEvent:
        self.calls.append(dict(kwargs))
        action = kwargs.get("action")
        agent = {
            "position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "rotation": {"y": 0.0},
            "cameraHorizon": 30.0,
            "isStanding": True,
        }
        if action == "GetSpawnCoordinatesAboveReceptacle":
            return FakeEvent(
                {
                    "lastActionSuccess": True,
                    "agent": agent,
                    "inventoryObjects": [{"objectId": "Apple|1"}],
                    "actionReturn": [{"x": self.spawn_x, "y": 0.57, "z": 0.5}],
                }
            )
        if action == "PlaceObjectAtPoint":
            return FakeEvent({"lastActionSuccess": True, "agent": agent, "inventoryObjects": []})
        return FakeEvent(
            {
                "lastActionSuccess": True,
                "agent": agent,
                "inventoryObjects": [],
                "objects": [
                    {
                        "objectId": "Apple|1",
                        "objectType": "Apple",
                        "position": {"x": self.apple_x, "y": 0.57, "z": 0.5},
                        "parentReceptacles": ["CounterTop|1"],
                    },
                    {
                        "objectId": "CounterTop|1",
                        "objectType": "CounterTop",
                        "receptacleObjectIds": ["Apple|1"],
                    },
                ],
            }
        )


def make_environment(*, apple_x: float = 0.01, spawn_x: float = 0.01) -> RobotEnvironment:
    env = RobotEnvironment.__new__(RobotEnvironment)
    env.width = 600
    env.height = 600
    env.pickup_target_types = {"Apple"}
    env.place_receptacle_types = {"CounterTop"}
    env.non_floor_proxy_types = {"Apple"}
    env.obstacle_object_types = {"CounterTop"}
    env.controller = RecordingController(apple_x=apple_x, spawn_x=spawn_x)
    env.last_event = FakeEvent(
        {
            "agent": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation": {"y": 0.0},
                "cameraHorizon": 30.0,
                "isStanding": True,
            },
            "inventoryObjects": [{"objectId": "Apple|1"}],
            "objects": [],
        }
    )
    return env


def configure_grounded_receptacle(env: RobotEnvironment) -> None:
    env._get_visible_receptacle_candidates = lambda: [
        {"objectId": "CounterTop|1", "allowed_place_receptacle": True, "receptacle": True}
    ]
    env._ground_visual_service_candidate = lambda *args, **kwargs: {
        "objectId": "CounterTop|1",
        "allowed_place_receptacle": True,
        "receptacle": True,
    }
    env._validate_visual_receptacle_grounding = lambda *args, **kwargs: {
        "passed": True,
        "result_type": "visual_receptacle_instance_grounded",
    }
    env._object_interactable_navigation_guidance = lambda *args, **kwargs: {
        "available": True,
        "current_pose_interactable": True,
        "interactable_pose_count": 1,
    }


class ExactGridPlaceTests(unittest.TestCase):
    def test_camera_relative_target_transforms_using_agent_heading(self) -> None:
        env = make_environment()
        env.last_event.metadata["agent"]["position"] = {"x": -1.0, "y": 0.901, "z": 0.0}
        env.last_event.metadata["agent"]["rotation"] = {"y": 180.0}
        candidate = grid_candidate()
        candidate["placement_safety_contract"]["camera_height_m"] = 0.901
        target = env._camera_relative_surface_target_to_world(
            {"x": -0.5327, "y": 0.4272, "ground_forward_m": 0.6707},
            candidate,
        )
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target["x"], -0.4673, places=4)
        self.assertAlmostEqual(target["y"], 0.4272, places=4)
        self.assertAlmostEqual(target["z"], -0.6707, places=4)

    def test_grid_place_dispatches_exact_world_point_action(self) -> None:
        env = make_environment()
        inventory_states = iter(({"holding_object": True}, {"holding_object": False}))
        env.get_inventory_state = lambda: next(inventory_states)
        configure_grounded_receptacle(env)
        env._object_interactable_from_current_pose = lambda *args, **kwargs: {
            "interactable_from_current_pose": True
        }

        result = env.place_held_object(grid_candidate(), strict_visual_grounding=True)

        self.assertEqual(result["result_type"], "place_executed")
        self.assertTrue(result["placement_target_verified"])
        self.assertEqual(result["placement_execution_mode"], "world_point_exact")
        actions = [call.get("action") for call in env.controller.calls]
        self.assertIn("PlaceObjectAtPoint", actions)
        self.assertNotIn("PutObject", actions)

    def test_grid_contract_cannot_disable_exact_execution(self) -> None:
        env = make_environment()
        candidate = grid_candidate()
        candidate["placement_safety_contract"]["requires_exact_target_execution"] = False

        self.assertTrue(env._grid_exact_target_required(candidate))

    def test_precheck_rejects_exact_target_without_near_legal_coordinate(self) -> None:
        env = make_environment(spawn_x=0.25)
        env.get_inventory_state = lambda: {"holding_object": True}
        configure_grounded_receptacle(env)

        result = env.precheck_place_candidate(grid_candidate(), strict_visual_grounding=True)

        self.assertEqual(result["result_type"], "error_place_exact_target_unavailable")
        self.assertTrue(result["placement_target_required"])
        self.assertEqual(result["placement_execution_mode"], "world_point_exact")
        actions = [call.get("action") for call in env.controller.calls]
        self.assertNotIn("PutObject", actions)
        self.assertNotIn("PlaceObjectAtPoint", actions)

    def test_execute_action_accepts_lateral_and_camera_pitch_actions(self) -> None:
        class MovementController:
            def __init__(self) -> None:
                self.position = {"x": 0.0, "y": 0.9, "z": 0.0}
                self.rotation = {"y": 0.0}
                self.horizon = 0.0

            def event(self) -> FakeEvent:
                return FakeEvent({
                    "lastActionSuccess": True,
                    "errorMessage": "",
                    "agent": {
                        "position": dict(self.position),
                        "rotation": dict(self.rotation),
                        "cameraHorizon": self.horizon,
                    },
                })

            def step(self, **kwargs: object) -> FakeEvent:
                action = kwargs.get("action")
                if action == "MoveRight":
                    self.position["x"] = float(self.position["x"]) + 0.25
                elif action == "LookDown":
                    self.horizon += 30.0
                return self.event()

        env = RobotEnvironment.__new__(RobotEnvironment)
        env.controller = MovementController()
        env.last_event = env.controller.event()
        lateral = env.execute_action("MoveRight")
        self.assertEqual(lateral["result_type"], "move_executed")
        self.assertTrue(lateral["state_changed"])
        pitch = env.execute_action("LookDown")
        self.assertEqual(pitch["result_type"], "move_executed")
        self.assertTrue(pitch["state_changed"])

    def test_camera_pitch_change_counts_as_robot_state_change(self) -> None:
        env = make_environment()
        before = {
            "position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "rotation": {"y": 0.0},
            "cameraHorizon": 0.0,
        }
        after = {
            "position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "rotation": {"y": 0.0},
            "cameraHorizon": 30.0,
        }
        self.assertTrue(env._robot_state_changed(before, after))

    def test_exact_placement_rejects_position_deviation(self) -> None:
        env = make_environment(apple_x=0.25)
        env.last_event = env.controller.step(action="Pass")
        env._object_interactable_from_current_pose = lambda *args, **kwargs: {
            "interactable_from_current_pose": True
        }

        validation = env._validate_place_result(
            held_object_id="Apple|1",
            receptacle={"objectId": "CounterTop|1"},
            action_success=True,
            inventory_empty=True,
            expected_surface_world_target={"x": 0.0, "y": 0.5, "z": 0.5},
            exact_target_required=True,
        )

        self.assertEqual(validation["result_type"], "error_place_target_deviation")
        self.assertFalse(validation["placement_verified"])
        self.assertFalse(validation["placement_target_verified"])


if __name__ == "__main__":
    unittest.main()
