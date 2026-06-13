import unittest

from scripts.navigation_core import build_navigation_core_state


class NavigationCoreTests(unittest.TestCase):
    def test_following_active_route_locks_waypoint_fallback(self) -> None:
        state = build_navigation_core_state(
            navigation={},
            exploration={"camera_posture": {"needs_normalization": False}},
            explore_plan={
                "active_route": {
                    "status": "active",
                    "route_id": "route-frontier-x1-z0-abcd1234",
                    "goal_cell": "1,0",
                    "next_action": "MoveAhead",
                    "route_step": {
                        "status": "active",
                        "route_id": "route-frontier-x1-z0-abcd1234",
                        "step_index": 0,
                        "action": "MoveAhead",
                        "current_cell": "0,0",
                        "current_heading": "east",
                        "next_cell": "1,0",
                        "goal_cell": "1,0",
                        "desired_heading": "east",
                    },
                },
                "waypoint_candidates": [{"cell": "0,1", "action": "MoveLeft"}],
            },
        )

        self.assertEqual(state["state"], "following_active_route")
        self.assertEqual(state["primary_source"], "route_step")
        self.assertTrue(state["route_locked"])
        self.assertFalse(state["allow_waypoint_fallback"])
        self.assertFalse(state["allow_raw_move_fallback"])

    def test_camera_recovery_takes_priority_over_route(self) -> None:
        state = build_navigation_core_state(
            navigation={},
            exploration={
                "camera_posture": {
                    "needs_normalization": True,
                    "normalize_action": "LookDown",
                }
            },
            explore_plan={
                "active_route": {
                    "status": "active",
                    "route_step": {"status": "active", "action": "MoveAhead"},
                }
            },
        )

        self.assertEqual(state["state"], "camera_recovery_required")
        self.assertEqual(state["primary_source"], "recovery")
        self.assertEqual(state["required_next"], "recover_camera_posture")


if __name__ == "__main__":
    unittest.main()
