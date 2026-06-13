#!/usr/bin/env python3
"""Public navigation state machine summary.

HomeRobot keeps mapping, goal selection, and low-level planning separated from
the high-level agent policy.  This module provides the same boundary for this
project: it does not execute actions and it does not write maps.  It normalizes
the current route/recovery state into one compact contract that
decision_context_builder can expose to the LLM.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


JsonDict = dict[str, Any]

NAVIGATION_CORE_SCHEMA = "robot_cleaner_navigation_core_v1"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def clean_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := clean_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := clean_empty(item)) not in (None, "", [], {})]
    return value


def build_navigation_core_state(
    *,
    navigation: JsonDict | None = None,
    exploration: JsonDict | None = None,
    explore_plan: JsonDict | None = None,
) -> JsonDict:
    navigation = as_dict(navigation)
    exploration = as_dict(exploration)
    explore_plan = as_dict(explore_plan)
    active_route = as_dict(explore_plan.get("active_route")) or as_dict(navigation.get("active_route"))
    route_step = as_dict(active_route.get("route_step"))
    camera_posture = as_dict(exploration.get("camera_posture"))
    recovery_actions = as_list(explore_plan.get("recovery_actions"))
    route_status = str(active_route.get("status") or "inactive")

    if camera_posture.get("needs_normalization") is True:
        state = "camera_recovery_required"
        primary_source = "recovery"
        required_next = "recover_camera_posture"
    elif route_status == "blocked":
        state = "route_blocked_recovery"
        primary_source = "recovery"
        required_next = "recover_or_replan_active_route"
    elif route_status == "active" and route_step.get("status") == "active":
        state = "following_active_route"
        primary_source = "route_step"
        required_next = "execute_current_route_step"
    elif recovery_actions:
        state = "planner_recovery_required"
        primary_source = "recovery"
        required_next = "execute_planner_recovery"
    elif as_dict(explore_plan.get("active_frontier_goal")):
        state = "frontier_goal_without_route"
        primary_source = "frontier_goal"
        required_next = "plan_or_replan_route"
    elif as_list(explore_plan.get("waypoint_candidates")):
        state = "local_waypoint_available"
        primary_source = "waypoint"
        required_next = "choose_safe_exploration_waypoint"
    else:
        state = "no_navigation_goal"
        primary_source = "observe_or_recover"
        required_next = "observe_refresh_or_goal_selection"

    route_locked = state in {
        "following_active_route",
        "route_blocked_recovery",
        "camera_recovery_required",
        "planner_recovery_required",
    }
    return clean_empty(
        {
            "schema": NAVIGATION_CORE_SCHEMA,
            "generated_at": now_iso(),
            "state": state,
            "primary_source": primary_source,
            "required_next": required_next,
            "route_locked": route_locked,
            "allow_raw_move_fallback": not route_locked,
            "allow_waypoint_fallback": not route_locked,
            "active_route": {
                "status": active_route.get("status"),
                "route_id": active_route.get("route_id"),
                "goal_cell": active_route.get("goal_cell"),
                "next_action": active_route.get("next_action"),
                "next_cell": active_route.get("next_cell"),
                "blocked_reason": active_route.get("blocked_reason"),
                "fail_count": active_route.get("fail_count"),
            },
            "route_step": {
                "status": route_step.get("status"),
                "route_id": route_step.get("route_id"),
                "step_index": route_step.get("step_index"),
                "action": route_step.get("action"),
                "current_cell": route_step.get("current_cell"),
                "current_heading": route_step.get("current_heading"),
                "next_cell": route_step.get("next_cell"),
                "goal_cell": route_step.get("goal_cell"),
                "desired_heading": route_step.get("desired_heading"),
                "progress_effect": route_step.get("progress_effect"),
            },
            "recovery": {
                "camera_needs_normalization": camera_posture.get("needs_normalization"),
                "camera_normalize_action": camera_posture.get("normalize_action"),
                "planner_recovery_actions": [
                    {
                        "action": as_dict(item).get("action"),
                        "reason": as_dict(item).get("reason"),
                        "trigger": as_dict(item).get("trigger"),
                    }
                    for item in recovery_actions
                    if as_dict(item).get("action")
                ],
            },
            "policy": {
                "single_navigation_exit": (
                    "when following_active_route, expose route_step as the only primary "
                    "navigation goal; waypoint/frontier alternatives stay hidden until "
                    "the route is blocked, reached, or abandoned"
                ),
                "llm_role": "choose task-level option; navigation_core owns route step generation",
            },
        }
    )
