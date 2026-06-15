#!/usr/bin/env python3
"""
Heartbeat watchdog for the robot-cleaner workspace.

This script is intentionally conservative. It does not move or clean directly.
It only:

1. checks whether patrol-runner is alive;
2. optionally performs a single idle perception pass;
3. requests an OpenClaw tidy-room-agent run when idle perception finds a task target;
4. optionally starts patrol-runner in detached mode when explicitly configured;
5. returns a ready patrol-runner report on a later heartbeat when the background
   runner has finished;
6. records a heartbeat log for research/debugging.
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


def invocation_metadata(argv: Optional[Sequence[str]], args: argparse.Namespace) -> JsonDict:
    effective_argv = list(sys.argv) if argv is None else [str(Path(__file__)), *[str(item) for item in argv]]
    ppid = os.getppid() if hasattr(os, "getppid") else None
    metadata: JsonDict = {
        "pid": os.getpid(),
        "ppid": ppid,
        "parent_process": process_snapshot(ppid),
        "sys_executable": sys.executable,
        "argv": effective_argv,
        "cwd": os.getcwd(),
        "script": str(Path(__file__).resolve()),
        "scan_idle": bool(getattr(args, "scan_idle", False)),
        "launch_mode": str(getattr(args, "launch_mode", "agent-request")),
    }
    return metadata


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


def process_snapshot(pid: Optional[int]) -> Optional[JsonDict]:
    if not pid or pid <= 0:
        return None
    psutil_error: Optional[str] = None
    try:
        import psutil  # type: ignore

        process = psutil.Process(int(pid))
        return {
            "pid": process.pid,
            "ppid": process.ppid(),
            "process_name": process.name(),
            "path": process.exe(),
            "cmdline": process.cmdline(),
            "create_time": process.create_time(),
            "lookup_status": "found_psutil",
        }
    except Exception as exc:
        psutil_error = str(exc)[:300]
    if os.name != "nt":
        return {"pid": pid, "lookup_status": "error", "psutil_error": psutil_error}
    command = (
        "$p=Get-Process -Id "
        + str(int(pid))
        + " -ErrorAction SilentlyContinue; "
        + "if ($p) { $p | Select-Object Id,ProcessName,Path,StartTime | ConvertTo-Json -Compress }"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
            startupinfo=hidden_startupinfo(),
            creationflags=hidden_creationflags(),
        )
    except Exception as exc:
        return {"pid": pid, "lookup_status": "error", "psutil_error": psutil_error, "error": str(exc)[:300]}
    text = completed.stdout.strip()
    if not text:
        return {
            "pid": pid,
            "lookup_status": "not_found",
            "psutil_error": psutil_error,
            "returncode": completed.returncode,
            "stderr": completed.stderr.strip()[-300:],
        }
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"pid": pid, "lookup_status": "unparsed", "psutil_error": psutil_error, "stdout": text[:300]}
    if not isinstance(data, dict):
        return {"pid": pid, "lookup_status": "unexpected", "psutil_error": psutil_error, "stdout": text[:300]}
    return {
        "pid": data.get("Id", pid),
        "process_name": data.get("ProcessName"),
        "path": data.get("Path"),
        "start_time": data.get("StartTime"),
        "lookup_status": "found",
    }


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


def run_command(command: Sequence[str], timeout: int, *, extra_env: Optional[Dict[str, str]] = None) -> JsonDict:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
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


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


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


def build_tidy_agent_run_request(args: argparse.Namespace, scan: JsonDict) -> JsonDict:
    analysis = scan.get("analysis", {}) if isinstance(scan.get("analysis"), dict) else {}
    return {
        "type": "openclaw_agent_run_request",
        "agent": "robot-cleaner",
        "skill": "tidy-room-agent",
        "task_mode": str(args.task_mode),
        "max_steps": int(args.max_steps),
        "reason": "heartbeat_idle_scan_found_service_target",
        "trigger_candidates": analysis.get("heartbeat_trigger_candidates", []),
        "required_tools": [
            "robot_cleaner_prepare_decision_turn",
            "robot_cleaner_execute_option",
            "robot_cleaner_status",
            "robot_cleaner_report",
            "robot_cleaner_stop",
        ],
        "loop_contract": [
            "Call robot_cleaner_prepare_decision_turn first.",
            "Choose exactly one current executable option_id from option_set.options.",
            "Call robot_cleaner_execute_option with that option_id.",
            "After every execution, prepare again before choosing another option.",
            "Stop only on room_complete, recover_failed, explicit stop, or max_steps.",
        ],
        "prompt": (
            "启动一个非 heartbeat 的 tidy-room-agent 任务来整理当前房间。"
            "heartbeat 回合本身不得执行机器人动作，也不得直接调用 "
            "robot_cleaner_prepare_decision_turn 或 robot_cleaner_execute_option。"
            "新的 tidy-room-agent 任务启动后，每轮先调用 robot_cleaner_prepare_decision_turn，"
            "再从当前 option_set.options 中选择一个 executable_now=true 的 option_id，"
            "并调用 robot_cleaner_execute_option 执行。优先级遵守 tidy-room-agent/SKILL.md："
            "pick/pursue/place/recover/orient/continue/explore:inspection_waypoint，然后才允许 fallback。"
            "每次执行后必须重新 prepare。持续运行直到 room_complete、recover failed、用户停止或达到 "
            "max_steps，然后调用 robot_cleaner_report 汇报。"
        ),
    }


def tidy_agent_request_user_message(request: JsonDict) -> str:
    candidates = request.get("trigger_candidates", [])
    candidate_count = len(candidates) if isinstance(candidates, list) else 0
    return (
        "心跳主动感知到当前房间存在疑似地面/近地面整理目标，"
        "已请求 OpenClaw 进入 tidy-room-agent 自主整理循环。"
        f"触发候选数：{candidate_count}。"
    )


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
            "心跳主动感知到当前房间存在疑似可处理目标；"
            f"{start_result.get('user_message')}"
        )
    if start_result.get("status") == "success":
        return (
            "心跳主动感知到当前房间存在疑似可处理目标，"
            "已启动当前房间自动巡视，后台 runner 正在执行。"
        )
    return (
        "心跳主动感知到当前房间存在疑似可处理目标，"
        "但启动自动巡视失败，请检查 patrol-runner 状态。"
    )


def service_start_user_message(start_result: JsonDict) -> str:
    if start_result.get("status") == "success" and start_result.get("user_message"):
        return f"心跳主动感知到当前房间存在疑似服务整理目标；{start_result.get('user_message')}"
    if start_result.get("status") == "success":
        return "心跳主动感知到当前房间存在疑似服务整理目标，已启动后台 tidy 巡视。"
    return "心跳主动感知到当前房间存在疑似服务整理目标，但后台 tidy 巡视启动失败。"


def analysis_has_trigger_target(analysis: JsonDict, *, task_mode: str) -> bool:
    if str(task_mode or "tidy").lower() != "tidy":
        return bool(analysis.get("floor_trash_detected", False))
    if bool(analysis.get("pickup_target_detected", False)):
        return True
    if bool(analysis.get("direct_pickup_detected", False)):
        return True
    candidates = analysis.get("service_candidates", []) or []
    if isinstance(candidates, list) and any(isinstance(item, dict) for item in candidates):
        return True
    return bool(analysis.get("floor_trash_detected", False))


def service_start_user_message(start_result: JsonDict) -> str:
    if start_result.get("status") == "success" and start_result.get("user_message"):
        return (
            "心跳主动感知到当前房间存在疑似服务整理候选，"
            "已启动后台 tidy 巡视做 RGB-D 复核；"
            f"{start_result.get('user_message')}"
        )
    if start_result.get("status") == "success":
        return "心跳主动感知到当前房间存在疑似服务整理候选，已启动后台 tidy 巡视做 RGB-D 复核。"
    return "心跳主动感知到当前房间存在疑似服务整理候选，但后台 tidy 巡视启动失败。"


def candidate_number(candidate: JsonDict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value: Any = candidate
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def short_candidate(candidate: JsonDict, *, reason: str) -> JsonDict:
    return {
        "label": candidate.get("label"),
        "raw_label": candidate.get("raw_label") or candidate.get("label"),
        "task_semantic_class": candidate.get("task_semantic_class"),
        "confidence": candidate.get("confidence"),
        "surface_hint": candidate.get("surface_hint"),
        "is_floor_level": candidate.get("is_floor_level"),
        "pickup_now": candidate.get("pickup_now"),
        "cleanable_now": candidate.get("cleanable_now"),
        "reachable": candidate.get("reachable"),
        "position_hint": candidate.get("position_hint"),
        "bottom_y_ratio": candidate_number(candidate, "geometry.bottom_y_ratio", "bottom_y_ratio"),
        "ground_distance": candidate.get("ground_distance") or candidate_number(
            candidate,
            "geometry.ground_distance_m",
            "depth.ground_distance_m",
            default=0.0,
        ),
        "reason": reason,
    }


def floor_contact_like(candidate: JsonDict) -> bool:
    detail = candidate.get("floor_contact_geometry") if isinstance(candidate.get("floor_contact_geometry"), dict) else {}
    return bool(detail.get("available") and detail.get("contact_floor_like"))


def heartbeat_floor_pickup_candidate(candidate: Any) -> Optional[JsonDict]:
    if not isinstance(candidate, dict):
        return None
    if str(candidate.get("task_semantic_class") or "") != "pickup_target":
        return None
    if bool(candidate.get("support_context_blocked")) or bool(candidate.get("is_support_surface")):
        return None

    confidence = candidate_number(candidate, "confidence", default=0.0)
    if confidence < env_float("ROBOT_HEARTBEAT_PICKUP_MIN_CONF", 0.65):
        return None

    surface_hint = str(candidate.get("surface_hint") or "")
    bottom_y = candidate_number(candidate, "geometry.bottom_y_ratio", "bottom_y_ratio", default=0.0)
    contact_like = floor_contact_like(candidate)
    if (
        surface_hint in {"surface_or_elevated", "support_surface", "table", "counter_top", "countertop"}
        and candidate.get("is_floor_level") is False
        and not contact_like
    ):
        return None
    floor_like = bool(
        contact_like
        or (surface_hint == "floor" and bool(candidate.get("is_floor_level")))
        or bool(candidate.get("pickup_now"))
    )
    if not floor_like:
        return None
    has_depth_evidence = bool(
        isinstance(candidate.get("depth"), dict)
        or isinstance(candidate.get("center_3d"), dict)
        or isinstance(candidate.get("floor_contact_geometry"), dict)
        or str(candidate.get("actionability_source") or "") == "depth_geometry"
    )
    if (
        not bool(candidate.get("pickup_now"))
        and not has_depth_evidence
        and bottom_y < env_float("ROBOT_HEARTBEAT_PICKUP_MIN_BOTTOM_RATIO", 0.78)
    ):
        return None

    reason = "direct_floor_pickup" if bool(candidate.get("pickup_now")) else "floor_pickup_candidate"
    return short_candidate(candidate, reason=reason)


def heartbeat_floor_clean_candidate(candidate: Any) -> Optional[JsonDict]:
    if not isinstance(candidate, dict):
        return None
    if str(candidate.get("task_semantic_class") or "") != "cleanable_object":
        return None
    if not bool(candidate.get("cleanable_now")):
        return None
    if bool(candidate.get("support_context_blocked")):
        return None
    if str(candidate.get("surface_hint") or "") != "floor" and not bool(candidate.get("is_floor_level")):
        return None
    if candidate_number(candidate, "confidence", default=0.0) < env_float("ROBOT_HEARTBEAT_CLEAN_MIN_CONF", 0.60):
        return None
    return short_candidate(candidate, reason="floor_clean_candidate")


def heartbeat_trigger_candidates(analysis: JsonDict, *, task_mode: str) -> List[JsonDict]:
    triggers: List[JsonDict] = []
    seen = set()

    def add(candidate: Optional[JsonDict]) -> None:
        if not isinstance(candidate, dict):
            return
        key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        triggers.append(candidate)

    if str(task_mode or "tidy").lower() == "tidy":
        pickup_pools = [analysis.get("best_pickup_candidate")]
        pickup_pools.extend(analysis.get("service_candidates", []) or [])
        for candidate in pickup_pools:
            add(heartbeat_floor_pickup_candidate(candidate))

    clean_pools = [analysis.get("best_clean_candidate")]
    clean_pools.extend(analysis.get("trash_candidates", []) or [])
    for candidate in clean_pools:
        add(heartbeat_floor_clean_candidate(candidate))

    return triggers


def analysis_has_trigger_target(analysis: JsonDict, *, task_mode: str) -> bool:
    candidates = analysis.get("heartbeat_trigger_candidates")
    if isinstance(candidates, list):
        return any(isinstance(item, dict) for item in candidates)
    return bool(heartbeat_trigger_candidates(analysis, task_mode=task_mode))


def idle_scan(timeout: int, perception_backend: str, *, task_mode: str) -> JsonDict:
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
    analyze_command = [sys.executable, str(analyze_script), "--image", str(vision["image_path"])]
    analyze_env: Dict[str, str] = {}
    if backend == "yolo":
        depth_path = str(vision.get("depth_path") or "").strip()
        camera = vision.get("camera") if isinstance(vision.get("camera"), dict) else {}
        if depth_path:
            analyze_command.extend(["--depth", depth_path])
        if camera:
            analyze_command.extend(["--camera-json", compact_json(camera)])
        analyze_command.extend(
            [
                "--iou",
                str(env_float("ROBOT_HEARTBEAT_YOLO_IOU", 0.45)),
                "--max-candidates",
                str(env_int("ROBOT_HEARTBEAT_YOLO_MAX_CANDIDATES", 8)),
                "--disable-stale-service-fallback",
            ]
        )
        save_vis = str(
            os.getenv(
                "ROBOT_HEARTBEAT_YOLO_SAVE_VIS",
                "",
            )
            or ""
        ).strip()
        if save_vis:
            analyze_command.extend(["--save-vis", save_vis])
        analyze_env = {
            "ROBOT_YOLO_FORCE_LOCAL_CORE": "0",
            "ROBOT_YOLO_DISABLE_STALE_SERVICE_FALLBACK": "1",
        }

    analysis = run_command(
        analyze_command,
        timeout=timeout,
        extra_env=analyze_env,
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

    trigger_candidates = heartbeat_trigger_candidates(analysis, task_mode=task_mode)
    return {
        "status": "success",
        "result_type": "heartbeat_idle_scan",
        "vision": {
            "image_path": vision.get("image_path"),
            "depth_path": vision.get("depth_path"),
            "camera": vision.get("camera") if isinstance(vision.get("camera"), dict) else None,
            "observation_contract": vision.get("observation_contract"),
            "online_safe": vision.get("online_safe", True),
            "last_action_feedback": vision.get("last_action_feedback"),
        },
        "analysis": {
            "perception_backend": analysis.get("perception_backend", backend),
            "perception_command": " ".join(analyze_command),
            "rgbd_used": bool(analysis.get("depth_path") or vision.get("depth_path")),
            "candidate_interpretation": "suspected_candidates_only;runner_revalidates_before_action",
            "heartbeat_trigger_candidates": trigger_candidates,
            "heartbeat_trigger_candidate_count": len(trigger_candidates),
            "floor_trash_detected": bool(analysis.get("floor_trash_detected", False)),
            "direct_cleanable_detected": bool(analysis.get("direct_cleanable_detected", False)),
            "pickup_target_detected": bool(analysis.get("pickup_target_detected", False)),
            "place_receptacle_detected": bool(analysis.get("place_receptacle_detected", False)),
            "direct_pickup_detected": bool(analysis.get("direct_pickup_detected", False)),
            "direct_place_detected": bool(analysis.get("direct_place_detected", False)),
            "best_pickup_candidate": short_candidate(
                analysis.get("best_pickup_candidate"),
                reason="best_pickup_candidate",
            )
            if isinstance(analysis.get("best_pickup_candidate"), dict)
            else None,
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
        scan = idle_scan(
            timeout=args.timeout,
            perception_backend=args.perception_backend,
            task_mode=str(args.task_mode),
        )
        result["idle_scan"] = scan
        analysis = scan.get("analysis", {})
        if scan.get("status") == "success" and analysis_has_trigger_target(
            analysis,
            task_mode=str(args.task_mode),
        ):
            task_noun = "suspected service candidate" if str(args.task_mode).lower() == "tidy" else "floor target"
            if str(args.launch_mode) == "patrol-runner":
                start_result = start_runner(args)
                start_ok = start_result.get("status") == "success"
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
                agent_request = build_tidy_agent_run_request(args, scan)
                result.update(
                    {
                        "result_type": "heartbeat_tidy_agent_run_requested",
                        "action_taken": "request_openclaw_tidy_agent_run",
                        "message": f"idle scan found a {task_noun}; requested an OpenClaw tidy-room-agent run",
                        "should_notify_user": True,
                        "notify_user": True,
                        "user_message": tidy_agent_request_user_message(agent_request),
                        "agent_run_request": agent_request,
                    }
                )
        else:
            task_noun = (
                "household service target"
                if str(args.task_mode).lower() == "tidy"
                else "cleanable floor target"
            )
            result.update(
                {
                    "result_type": "heartbeat_idle_no_task",
                    "message": f"idle scan found no {task_noun}",
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
        help="Scene analysis backend for idle scan.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["clean", "tidy"],
        default=os.getenv("ROBOT_TASK_MODE", "tidy"),
        help="Task mode. tidy starts on service-object detections, clean starts on floor-clean targets.",
    )
    parser.add_argument(
        "--launch-mode",
        choices=["agent-request", "patrol-runner"],
        default=os.getenv("ROBOT_HEARTBEAT_LAUNCH_MODE", "agent-request"),
        help=(
            "What to do when idle scan finds a task. agent-request asks OpenClaw to start "
            "tidy-room-agent; patrol-runner keeps the legacy detached script runner."
        ),
    )
    parser.add_argument(
        "--scan-idle",
        action="store_true",
        help="When IDLE, do one get-vision + analyze pass and trigger launch-mode if a task target is detected.",
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
    invocation = invocation_metadata(argv, args)
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

    result["invocation"] = invocation
    write_heartbeat_state(result)
    append_log(result.get("result_type", "heartbeat_unknown"), result)
    print_json(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
