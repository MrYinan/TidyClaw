#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests"]
# ///

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


# 当前文件位置：
# workspace-robot-cleaner/skills/move-robot/scripts/move_robot.py
# parents[3] = workspace-robot-cleaner
REPO_ROOT = Path(__file__).resolve().parents[3]
MEMORY_DIR = REPO_ROOT / "memory"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


JsonDict = Dict[str, Any]


def _is_success(data: JsonDict) -> bool:
    if "lastActionSuccess" in data:
        return bool(data.get("lastActionSuccess"))
    return bool(data.get("status") == "success")


def _safe_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _update_object_goal_pose(nav_status: JsonDict) -> Optional[JsonDict]:
    """同步 object-goals.json 里的 pose_pred，避免 active_goal 继续显示旧 cell。

    这不是重新选择目标，只是把当前导航位姿写进去。
    下一次 patrol_runner 重新 select object-memory target 时，仍会重新评估 active_goal。
    """
    goals_path = MEMORY_DIR / "object-goals.json"
    if not goals_path.exists():
        return None

    try:
        raw = goals_path.read_text(encoding="utf-8-sig")
        goals = json.loads(raw) if raw.strip() else {}
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": "object_goals_invalid_json",
            "message": str(exc),
        }

    if not isinstance(goals, dict):
        return {
            "status": "skipped",
            "reason": "object_goals_not_object",
        }

    active = goals.get("active_goal")
    if not isinstance(active, dict):
        return {
            "status": "skipped",
            "reason": "no_active_goal",
        }

    planner_inputs = active.get("planner_inputs")
    if not isinstance(planner_inputs, dict):
        planner_inputs = {}
        active["planner_inputs"] = planner_inputs

    heading = str(nav_status.get("last_heading") or "north")
    theta_by_heading = {
        "north": 0,
        "east": 90,
        "south": 180,
        "west": 270,
    }

    planner_inputs["pose_pred"] = {
        "cell": str(nav_status.get("last_cell") or "0,0"),
        "heading": heading,
        "theta_deg": theta_by_heading.get(heading, 0),
        "coordinate_mode": "action_odometry_grid",
        "pose_source": "move_robot_skill_navigation_memory",
    }
    planner_inputs["pose_synced_by"] = "move_robot.py"
    active["planner_inputs"] = planner_inputs
    active["pose_dirty"] = True
    active["pose_sync_reason"] = "manual_or_skill_move_robot_action"

    goals["active_goal"] = active

    # 简单原子写入：先写 tmp，再替换
    tmp_path = goals_path.with_suffix(goals_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(goals, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(goals_path)

    return {
        "status": "success",
        "result_type": "object_goal_pose_synced",
        "active_goal_track_id": active.get("track_id"),
        "pose_pred": planner_inputs.get("pose_pred"),
    }


def _update_navigation_memory(
    *,
    action: str,
    success: bool,
    action_result: JsonDict,
) -> JsonDict:
    """把单独执行的 move action 写入 room-state.json。

    注意：
    - 这里不做任务决策；
    - 不调用后端；
    - 只根据 action + success 更新 action_odometry_grid。
    """
    from scripts.navigation_memory_core import NavigationMemory

    manager = NavigationMemory(MEMORY_DIR)

    failure_reason = None
    if not success:
        failure_reason = (
            _safe_text(action_result.get("error_message"))
            or _safe_text(action_result.get("message"))
            or _safe_text(action_result.get("result_type"))
            or "move_failed"
        )

    nav_status = manager.record_step(
        action=action,
        success=success,
        vision={},
        analysis={},
        action_result=action_result,
        failure_reason=failure_reason,
        recommendation={
            "action": action,
            "reason": "move_robot_skill_direct_call",
            "source": "move_robot_skill",
        },
    )

    goal_pose_sync = _update_object_goal_pose(nav_status)

    return {
        "status": "success",
        "result_type": "navigation_memory_updated_by_move_robot",
        "action": action,
        "success": success,
        "last_cell": nav_status.get("last_cell"),
        "last_heading": nav_status.get("last_heading"),
        "visited_cell_count": nav_status.get("visited_cell_count"),
        "coverage_estimate": nav_status.get("coverage_estimate"),
        "collision_count": nav_status.get("collision_count"),
        "oscillation_count": nav_status.get("oscillation_count"),
        "blocked_edges": nav_status.get("blocked_edges", [])[-4:],
        "position_map_status": nav_status.get("position_map_status"),
        "pose_confidence": nav_status.get("pose_confidence"),
        "position_uncertainty_cells": nav_status.get("position_uncertainty_cells"),
        "heading_confidence": nav_status.get("heading_confidence"),
        "occupancy_summary": nav_status.get("occupancy_summary", {}),
        
        "object_goal_pose_sync": goal_pose_sync,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="控制 AI2-THOR 扫地机器人移动")
    parser.add_argument(
        "--action",
        required=True,
        choices=["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown"],
        help="执行的动作",
    )
    parser.add_argument(
        "--memory-mode",
        default="auto",
        choices=["auto", "internal", "external", "none"],
        help=(
            "memory 更新模式："
            "auto/internal=单独运行时更新 navigation memory；"
            "external=由外层 patrol_runner 更新，当前脚本不写 memory；"
            "none=纯后端调试，不写 memory。"
        ),
    )
    args = parser.parse_args()

    try:
        response = requests.post(
            "http://127.0.0.1:5000/move",
            json={"action": args.action},
            timeout=10,
        )

        try:
            data = response.json()
        except Exception:
            data = {
                "status": "error",
                "result_type": "error_move_invalid_response",
                "message": response.text,
                "http_status": response.status_code,
            }

        if response.status_code >= 500:
            data.setdefault("status", "error")
            data.setdefault("result_type", "error_move_service_unavailable")
            data.setdefault("http_status", response.status_code)
            print(json.dumps(data, ensure_ascii=False))
            sys.exit(1)

        data.setdefault("http_status", response.status_code)

        memory_update: Optional[JsonDict] = None
        memory_mode = str(args.memory_mode or "auto")

        if memory_mode in {"auto", "internal"}:
            try:
                success = _is_success(data)
                memory_update = _update_navigation_memory(
                    action=args.action,
                    success=success,
                    action_result=data,
                )
            except Exception as exc:
                memory_update = {
                    "status": "error",
                    "result_type": "navigation_memory_update_failed",
                    "message": str(exc),
                }
        elif memory_mode == "external":
            memory_update = {
                "status": "skipped",
                "result_type": "navigation_memory_external_owner",
                "message": "navigation memory will be updated by patrol_runner or another caller",
            }
        else:
            memory_update = {
                "status": "skipped",
                "result_type": "navigation_memory_disabled",
            }

        data["frontend_memory_update"] = memory_update
        print(json.dumps(data, ensure_ascii=False))

        if data.get("status") == "error":
            sys.exit(1)

    except requests.exceptions.ConnectionError as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_move_service_unavailable",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)
    except requests.exceptions.Timeout as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_move_timeout",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)
    except Exception as e:
        print(
            json.dumps(
                {
                    "status": "error",
                    "result_type": "error_move_unknown",
                    "message": str(e),
                },
                ensure_ascii=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()