#!/usr/bin/env python3
"""
Heartbeat watchdog for the robot-cleaner workspace.

This script is intentionally conservative. It does not move or clean directly.
It only:

1. checks whether patrol-runner is alive;
2. optionally performs a single idle perception pass;
3. starts patrol-runner in detached mode when idle perception finds a clean/task target;
4. returns a ready patrol-runner report on a later heartbeat when the background
   runner has finished;
5. records a heartbeat log for research/debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
MEMORY_DIR = REPO_ROOT / "memory"
PATROL_RUNNER_SKILL = REPO_ROOT / "skills" / "patrol-runner" / "scripts" / "patrol_runner_skill.py"
GET_VISION_SCRIPT = REPO_ROOT / "skills" / "get-vision" / "scripts" / "get_vision.py"
OPENCV_ANALYZE_SCRIPT = REPO_ROOT / "skills" / "analyze-scene-opencv" / "scripts" / "analyze_scene_opencv.py"
YOLO_ANALYZE_SCRIPT = REPO_ROOT / "skills" / "perceive-scene-yolo" / "scripts" / "perceive_scene_yolo.py"
HEARTBEAT_STATE = MEMORY_DIR / "heartbeat-state.json"

PERCEPTION_BACKENDS = {
    "opencv": OPENCV_ANALYZE_SCRIPT,
    "yolo": YOLO_ANALYZE_SCRIPT,
}

sys.path.insert(0, str(REPO_ROOT))
from scripts.state_manager_core import DEFAULT_ROOM, StateManager  # noqa: E402


JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def print_json(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False))


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def hidden_startupinfo() -> Optional[subprocess.STARTUPINFO]:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def parse_json_output(text: str) -> JsonDict:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass
    for line in reversed([line.strip() for line in stripped.splitlines() if line.strip()]):
        try:
            data = json.loads(line)
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            continue
    return {}


def run_command(command: Sequence[str], timeout: int) -> JsonDict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
        check=False,
        startupinfo=hidden_startupinfo(),
        creationflags=hidden_creationflags(),
    )
    data = parse_json_output(completed.stdout)
    if not data and completed.stderr:
        data = parse_json_output(completed.stderr)
    data.setdefault("_returncode", completed.returncode)
    if completed.stderr.strip():
        data.setdefault("_stderr_tail", completed.stderr.splitlines()[-3:])
    return data


def analyze_script_for_backend(backend: str) -> Path:
    key = str(backend or "yolo").strip().lower()
    script = PERCEPTION_BACKENDS.get(key)
    if script is None:
        raise ValueError(f"Unsupported perception backend: {backend}")
    return script


def append_log(event: str, payload: JsonDict) -> None:
    log_path = MEMORY_DIR / f"{datetime.now().date().isoformat()}-heartbeat.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "event": event,
        "time": now_iso(),
        **payload,
    }
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"- {datetime.now().strftime('%H:%M:%S')} `{event}` {compact_json(entry)}\n")


def load_heartbeat_state() -> JsonDict:
    if not HEARTBEAT_STATE.exists():
        return {}
    try:
        data = json.loads(HEARTBEAT_STATE.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_heartbeat_state(payload: JsonDict) -> None:
    HEARTBEAT_STATE.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runner_status(timeout: int) -> JsonDict:
    return run_command(
        [sys.executable, str(PATROL_RUNNER_SKILL), "--command", "status"],
        timeout=timeout,
    )


def runner_report(timeout: int) -> JsonDict:
    return run_command(
        [sys.executable, str(PATROL_RUNNER_SKILL), "--command", "report"],
        timeout=max(timeout, 30),
    )


def start_runner(args: argparse.Namespace) -> JsonDict:
    command = [
        sys.executable,
        str(PATROL_RUNNER_SKILL),
        "--command",
        "start",
        "--room",
        args.room,
        "--max-steps",
        str(args.max_steps),
        "--perception-backend",
        str(args.perception_backend),
        "--task-mode",
        str(args.task_mode),
    ]
    try:
        return run_command(command, timeout=max(args.timeout, 30))
    except subprocess.TimeoutExpired as exc:
        status = runner_status(timeout=args.timeout)
        process = status.get("process", {}) if isinstance(status.get("process"), dict) else {}
        if bool(process.get("alive", False)):
            return {
                "status": "success",
                "result_type": "patrol_runner_start_timeout_but_alive",
                "message": "patrol-runner start command timed out, but the background runner is alive",
                "process": process,
                "timeout": str(exc),
            }
        return {
            "status": "error",
            "result_type": "patrol_runner_start_timeout",
            "message": str(exc),
        }


def status_report_ready(status: JsonDict) -> bool:
    report = status.get("report", {})
    if not isinstance(report, dict):
        report = {}
    return bool(status.get("report_ready", False) or report.get("report_ready", False))


def report_user_message(report_result: JsonDict, status: JsonDict) -> str:
    if report_result.get("user_message"):
        return str(report_result.get("user_message"))
    report = status.get("report", {})
    if isinstance(report, dict) and report.get("user_message"):
        return str(report.get("user_message"))
    return "巡视任务已有结果，请查看 patrol-runner report 获取详情。"


def start_user_message(start_result: JsonDict) -> str:
    if start_result.get("status") == "success" and start_result.get("user_message"):
        return (
            "心跳主动感知到当前房间存在疑似可清扫地面目标；"
            f"{start_result.get('user_message')}"
        )
    if start_result.get("status") == "success":
        return (
            "心跳主动感知到当前房间存在疑似可清扫地面目标，"
            "已启动当前房间自动巡视，后台 runner 正在执行。"
        )
    return (
        "心跳主动感知到当前房间存在疑似可清扫地面目标，"
        "但启动自动巡视失败，请检查 patrol-runner 状态。"
    )


def service_start_user_message(start_result: JsonDict) -> str:
    if start_result.get("status") == "success" and start_result.get("user_message"):
        return f"Heartbeat detected a possible household service target; {start_result.get('user_message')}"
    if start_result.get("status") == "success":
        return "Heartbeat detected a possible household service target and started the background tidy runner."
    return "Heartbeat detected a possible household service target, but the tidy runner did not start."


def analysis_has_trigger_target(analysis: JsonDict, *, task_mode: str) -> bool:
    if bool(analysis.get("floor_trash_detected", False)):
        return True
    if str(task_mode or "clean").lower() != "tidy":
        return False
    if bool(analysis.get("pickup_target_detected", False)):
        return True
    candidates = analysis.get("service_candidates", []) or []
    return isinstance(candidates, list) and any(isinstance(item, dict) for item in candidates)


def idle_scan(timeout: int, perception_backend: str) -> JsonDict:
    vision = run_command([sys.executable, str(GET_VISION_SCRIPT)], timeout=timeout)
    if vision.get("status") != "success" or not vision.get("image_path"):
        return {
            "status": "error",
            "result_type": "heartbeat_vision_failed",
            "vision": {
                "status": vision.get("status"),
                "result_type": vision.get("result_type"),
                "message": vision.get("message"),
            },
        }

    backend = str(perception_backend or "yolo").strip().lower()
    analyze_script = analyze_script_for_backend(backend)
    analysis = run_command(
        [sys.executable, str(analyze_script), "--image", str(vision["image_path"])],
        timeout=timeout,
    )
    if analysis.get("status") != "success":
        return {
            "status": "error",
            "result_type": "heartbeat_analysis_failed",
            "analysis": {
                "status": analysis.get("status"),
                "result_type": analysis.get("result_type"),
                "message": analysis.get("message"),
                "perception_backend": backend,
            },
        }

    return {
        "status": "success",
        "result_type": "heartbeat_idle_scan",
        "vision": {
            "image_path": vision.get("image_path"),
            "observation_contract": vision.get("observation_contract"),
            "online_safe": vision.get("online_safe", True),
            "last_action_feedback": vision.get("last_action_feedback"),
        },
        "analysis": {
            "perception_backend": analysis.get("perception_backend", backend),
            "floor_trash_detected": bool(analysis.get("floor_trash_detected", False)),
            "direct_cleanable_detected": bool(analysis.get("direct_cleanable_detected", False)),
            "pickup_target_detected": bool(analysis.get("pickup_target_detected", False)),
            "place_receptacle_detected": bool(analysis.get("place_receptacle_detected", False)),
            "direct_pickup_detected": bool(analysis.get("direct_pickup_detected", False)),
            "service_candidates": analysis.get("service_candidates", []),
            "receptacle_candidates": analysis.get("receptacle_candidates", []),
            "alignment_needed": bool(analysis.get("alignment_needed", False)),
            "trash_candidates": analysis.get("trash_candidates", []),
            "ignored_candidates_count": len(analysis.get("ignored_candidates", []) or []),
            "obstacle_ahead": bool(analysis.get("obstacle_ahead", False)),
            "open_directions": analysis.get("open_directions", []),
            "frontier_exists": bool(analysis.get("frontier_exists", False)),
            "floor_clean": bool(analysis.get("floor_clean", False)),
            "analysis_confidence": analysis.get("analysis_confidence"),
            "occupancy": analysis.get("occupancy", {}),
            "recommended_action": analysis.get("recommended_action"),
        },
    }


def state_is_idle(state: JsonDict) -> bool:
    patrol = state.get("patrol", {})
    mission = state.get("mission", {})
    patrol_mode = str(patrol.get("mode") or "IDLE").upper()
    mission_mode = str(mission.get("mode") or "IDLE").upper()
    return (
        not bool(patrol.get("enabled"))
        and not bool(mission.get("enabled"))
        and patrol_mode == "IDLE"
        and mission_mode == "IDLE"
    )


def check_once(args: argparse.Namespace) -> JsonDict:
    manager = StateManager(MEMORY_DIR)
    status = runner_status(timeout=args.timeout)
    validation = manager.validate_state()
    state = validation.get("state", {})

    process = status.get("process", {})
    runner_alive = bool(process.get("alive", False))

    result: JsonDict = {
        "status": "success",
        "result_type": "heartbeat_ok",
        "time": now_iso(),
        "runner_alive": runner_alive,
        "action_taken": "none",
        "message": "heartbeat check completed",
        "state_mode": {
            "patrol": state.get("patrol", {}).get("mode"),
            "mission": state.get("mission", {}).get("mode"),
        },
        "should_notify_user": False,
    }

    if runner_alive:
        result.update(
            {
                "result_type": "heartbeat_runner_alive",
                "message": "patrol runner is already active",
                "process": process,
            }
        )
        return result

    if status_report_ready(status):
        report_result = runner_report(timeout=args.timeout)
        report_ok = report_result.get("status") == "success"
        result.update(
            {
                "result_type": "heartbeat_room_report_ready" if report_ok else "heartbeat_room_report_failed",
                "action_taken": "report_patrol_runner" if report_ok else "report_patrol_runner_failed",
                "message": (
                    "patrol runner has a ready report; returned it to the user"
                    if report_ok
                    else "patrol runner has a ready report, but report command failed"
                ),
                "should_notify_user": True,
                "notify_user": True,
                "user_message": report_user_message(report_result, status),
                "runner_report": report_result,
            }
        )
        return result

    if args.scan_idle and state_is_idle(state):
        scan = idle_scan(timeout=args.timeout, perception_backend=args.perception_backend)
        result["idle_scan"] = scan
        analysis = scan.get("analysis", {})
        if scan.get("status") == "success" and analysis_has_trigger_target(
            analysis,
            task_mode=str(args.task_mode),
        ):
            start_result = start_runner(args)
            start_ok = start_result.get("status") == "success"
            task_noun = "service target" if str(args.task_mode).lower() == "tidy" else "floor target"
            result.update(
                {
                    "result_type": (
                        "heartbeat_target_found_started"
                        if start_ok
                        else "heartbeat_target_found_start_failed"
                    ),
                    "action_taken": "start_patrol_runner" if start_ok else "start_patrol_runner_failed",
                    "message": (
                        f"idle scan found a {task_noun}; started patrol-runner in detached mode"
                        if start_ok
                        else f"idle scan found a {task_noun}, but patrol-runner did not start"
                    ),
                    "should_notify_user": True,
                    "notify_user": True,
                    "user_message": (
                        service_start_user_message(start_result)
                        if str(args.task_mode).lower() == "tidy"
                        else start_user_message(start_result)
                    ),
                    "runner_start": start_result,
                }
            )
        else:
            result.update(
                {
                    "result_type": "heartbeat_idle_no_task",
                    "message": "idle scan found no cleanable floor target",
                }
            )
        return result

    if state_is_idle(state):
        result.update(
            {
                "result_type": "heartbeat_idle_no_scan",
                "message": "robot is idle; idle scanning is disabled for this heartbeat",
            }
        )
        return result

    result.update(
        {
            "result_type": "heartbeat_state_observed_no_action",
            "message": "non-idle patrol state observed; patrol-runner status/report owns user-facing summaries",
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heartbeat watchdog for robot-cleaner active task triggering.")
    parser.add_argument("--room", default=DEFAULT_ROOM)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument(
        "--perception-backend",
        choices=sorted(PERCEPTION_BACKENDS.keys()),
        default=os.getenv("ROBOT_PERCEPTION_BACKEND", "yolo"),
        help="Scene analysis backend for idle scan and any started patrol-runner.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["clean", "tidy"],
        default=os.getenv("ROBOT_TASK_MODE", "clean"),
        help="Runner task mode. tidy starts on service-object detections, clean starts on floor-clean targets.",
    )
    parser.add_argument(
        "--scan-idle",
        action="store_true",
        help="When IDLE, do one get-vision + analyze pass and start patrol-runner if floor trash is detected.",
    )
    parser.add_argument(
        "--no-scan-idle",
        action="store_false",
        dest="scan_idle",
        help="Disable idle perception scan.",
    )
    parser.set_defaults(scan_idle=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = check_once(args)
    except subprocess.TimeoutExpired as exc:
        result = {
            "status": "error",
            "result_type": "heartbeat_timeout",
            "message": str(exc),
            "time": now_iso(),
        }
    except Exception as exc:
        result = {
            "status": "error",
            "result_type": "heartbeat_error",
            "message": str(exc),
            "time": now_iso(),
        }

    write_heartbeat_state(result)
    append_log(result.get("result_type", "heartbeat_unknown"), result)
    print_json(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
