#!/usr/bin/env python3
"""
OpenClaw skill wrapper for scripts/patrol_runner.py.

The real patrol logic lives in scripts/patrol_runner.py. This wrapper gives
OpenClaw a stable skill entry point that can start the runner in the
background, run one foreground segment, query status, or request a graceful
stop through the state-manager core.
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
RUNNER_SCRIPT = REPO_ROOT / "scripts" / "patrol_runner.py"
PID_FILE = MEMORY_DIR / "patrol-runner.pid.json"
PROCESS_LOG = MEMORY_DIR / "patrol-runner-process.log"

sys.path.insert(0, str(REPO_ROOT))
from scripts.state_manager_core import DEFAULT_ROOM, StateManager  # noqa: E402


JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def print_json(data: JsonDict) -> None:
    print(json.dumps(data, ensure_ascii=False))


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def hidden_startupinfo() -> Optional[subprocess.STARTUPINFO]:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0
    return startupinfo


def hidden_creationflags(*, background: bool = False) -> int:
    if os.name != "nt":
        return 0
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if background:
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        flags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
    return flags


def load_pid_info() -> Optional[JsonDict]:
    if not PID_FILE.exists():
        return None
    try:
        data = json.loads(PID_FILE.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_pid_info(data: JsonDict) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_pid_info_if_current_process() -> None:
    pid_info = load_pid_info()
    if not pid_info:
        return
    try:
        pid = int(pid_info.get("pid"))
    except (TypeError, ValueError):
        return
    if pid != os.getpid():
        return
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def remove_pid_info_if_stale(pid: Optional[int]) -> None:
    if pid is None or not process_alive(pid):
        try:
            PID_FILE.unlink()
        except FileNotFoundError:
            pass


def process_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False

    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
                startupinfo=hidden_startupinfo(),
                creationflags=hidden_creationflags(),
            )
        except Exception:
            return False
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def parse_json_lines(text: str) -> List[JsonDict]:
    events: List[JsonDict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


def format_counted_labels(labels: List[Any], noun: str) -> str:
    values = [str(item).strip() for item in labels if str(item).strip()]
    if not values:
        return f"0 个{noun}"
    return f"{len(values)} 个{noun}（{', '.join(values)}）"


def format_float(value: Any, default: str = "未知") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def build_patrol_report(state: JsonDict, *, runner_alive: bool) -> JsonDict:
    patrol = state.get("patrol", {}) if isinstance(state.get("patrol"), dict) else {}
    mission = state.get("mission", {}) if isinstance(state.get("mission"), dict) else {}
    room = state.get("room", {}) if isinstance(state.get("room"), dict) else {}

    steps = int(patrol.get("step_count", mission.get("total_steps_completed", room.get("explored_steps", 0))) or 0)
    max_steps = int(patrol.get("max_steps", mission.get("max_steps", 0)) or 0)
    cleaned = list(room.get("targets_cleaned", []) or mission.get("garbage_cleaned", []) or [])
    placed = list(room.get("objects_placed", []) or mission.get("objects_placed", []) or [])
    service_completed = list(
        room.get("service_tasks_completed", []) or mission.get("service_tasks_completed", []) or []
    )
    visual_candidates = list(
        room.get("targets_found", [])
        or mission.get("objects_detected", [])
        or mission.get("garbage_detected", [])
        or []
    )
    coverage = room.get("coverage_estimate")
    frontier_count = len(room.get("frontier_cells", []) or [])
    collisions = int(room.get("collision_count", 0) or 0)
    final_summary = str(mission.get("final_summary") or "")
    last_result = str(mission.get("last_result") or "")
    patrol_mode = str(patrol.get("mode") or "")
    mission_mode = str(mission.get("mode") or "")
    room_complete = bool(room.get("room_complete", False))

    cleaned_text = format_counted_labels(cleaned, "已清理目标")
    visual_count = len(visual_candidates)
    cleaned_count = len(cleaned)
    unconfirmed_count = max(0, visual_count - cleaned_count)

    if unconfirmed_count > 0:
        visual_text = f"视觉疑似目标 {visual_count} 个，其中 {unconfirmed_count} 个未通过执行侧清扫确认或无需清扫"
    elif visual_count > 0:
        visual_text = f"视觉疑似目标 {visual_count} 个，已清理目标均完成验证"
    else:
        visual_text = "未发现需要清理的地面目标"

    base_metrics = {
        "step_count": steps,
        "max_steps": max_steps,
        "targets_cleaned": cleaned,
        "objects_placed": placed,
        "service_tasks_completed": service_completed,
        "visual_candidates": visual_candidates,
        "coverage_estimate": coverage,
        "frontier_count": frontier_count,
        "collision_count": collisions,
        "final_summary": final_summary,
        "last_result": last_result,
        "room_complete": room_complete,
        "patrol_mode": patrol_mode,
        "mission_mode": mission_mode,
        "runner_alive": runner_alive,
    }

    if (room_complete or mission_mode == "MISSION_REPORT" or patrol_mode == "MISSION_REPORT") and (
        placed or service_completed
    ):
        message = (
            f"当前房间服务巡视已结束。共执行 {steps} 步，已放置 {len(placed)} 个物体，"
            f"已清理 {len(cleaned)} 个地面目标；视觉候选 {visual_count} 个。"
            f"覆盖率估计 {format_float(coverage)}，剩余 frontier {frontier_count}，"
            f"碰撞/受阻 {collisions} 次。"
        )
        return {
            "report_ready": True,
            "report_type": "room_complete",
            "user_message": message,
            **base_metrics,
        }

    if room_complete or mission_mode == "MISSION_REPORT" or patrol_mode == "MISSION_REPORT":
        message = (
            f"当前房间已巡视完成。共执行 {steps} 步，{cleaned_text}；"
            f"{visual_text}。覆盖率估计 {format_float(coverage)}，剩余 frontier {frontier_count}，"
            f"碰撞/受阻 {collisions} 次。"
        )
        return {
            "report_ready": True,
            "report_type": "room_complete",
            "user_message": message,
            **base_metrics,
        }

    if mission_mode == "RECOVER" or patrol_mode == "RECOVER":
        reason = final_summary or last_result or "unknown"
        message = (
            f"当前巡视任务进入 RECOVER，已暂停自动推进。当前步数 {steps}/{max_steps}，"
            f"原因：{reason}。建议先查看日志或复位后再继续。"
        )
        return {
            "report_ready": True,
            "report_type": "recover",
            "user_message": message,
            **base_metrics,
        }

    if mission_mode == "DONE" or patrol_mode == "DONE":
        reason = final_summary or last_result or "done"
        message = (
            f"当前巡视任务已结束。共执行 {steps}/{max_steps} 步，{cleaned_text}；"
            f"{visual_text}。结束状态：{reason}。"
        )
        return {
            "report_ready": True,
            "report_type": "done",
            "user_message": message,
            **base_metrics,
        }

    if runner_alive and (placed or service_completed):
        message = (
            f"当前房间服务巡视正在运行。已执行 {steps}/{max_steps} 步，"
            f"已放置 {len(placed)} 个物体，已清理 {len(cleaned)} 个地面目标，"
            f"视觉候选 {visual_count} 个，覆盖率估计 {format_float(coverage)}。"
        )
        return {
            "report_ready": False,
            "report_type": "running",
            "user_message": message,
            **base_metrics,
        }

    if runner_alive:
        message = (
            f"当前房间自动巡视正在后台运行。已执行 {steps}/{max_steps} 步，"
            f"覆盖率估计 {format_float(coverage)}，已发现视觉疑似目标 {visual_count} 个，"
            f"已清理 {cleaned_count} 个。"
        )
        return {
            "report_ready": False,
            "report_type": "running",
            "user_message": message,
            **base_metrics,
        }

    message = (
        f"当前没有运行中的 patrol-runner。最近状态为 patrol={patrol_mode or 'unknown'}，"
        f"mission={mission_mode or 'unknown'}，已记录 {steps}/{max_steps} 步。"
    )
    return {
        "report_ready": False,
        "report_type": "not_running",
        "user_message": message,
        **base_metrics,
    }


def build_runner_args(args: argparse.Namespace, *, continuous: bool) -> List[str]:
    command = [
        sys.executable,
        str(RUNNER_SCRIPT),
        "--segment-steps",
        str(args.segment_steps),
        "--timeout",
        str(args.timeout),
        "--sleep",
        str(args.sleep),
    ]

    if args.start:
        command.append("--start")
    if args.no_reset:
        command.append("--no-reset")
    if args.room:
        command.extend(["--room", args.room])
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if continuous:
        command.append("--continuous")
    if args.max_segments is not None:
        command.extend(["--max-segments", str(args.max_segments)])
    if args.clean_validation:
        command.extend(["--clean-validation", str(args.clean_validation)])
    if args.perception_backend:
        command.extend(["--perception-backend", str(args.perception_backend)])
    if args.task_mode:
        command.extend(["--task-mode", str(args.task_mode)])
    if args.interaction_grounding:
        command.extend(["--interaction-grounding", str(args.interaction_grounding)])
    if args.pickup_surface_policy:
        command.extend(["--pickup-surface-policy", str(args.pickup_surface_policy)])
    if args.pickup_target_labels:
        command.extend(["--pickup-target-labels", str(args.pickup_target_labels)])
    if args.performance_profile:
        command.extend(["--performance-profile", str(args.performance_profile)])
    if args.object_memory_update_interval is not None:
        command.extend(["--object-memory-update-interval", str(args.object_memory_update_interval)])
    if args.semantic_map_update_interval is not None:
        command.extend(["--semantic-map-update-interval", str(args.semantic_map_update_interval)])
    if args.log_detail:
        command.extend(["--log-detail", str(args.log_detail)])
    if args.dry_run:
        command.append("--dry-run")
    if args.verbose:
        command.append("--verbose")
    if args.quiet:
        command.append("--quiet")

    return command


def command_status() -> JsonDict:
    manager = StateManager(MEMORY_DIR)
    pid_info = load_pid_info()
    pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    alive = process_alive(pid)
    if pid_info and not alive:
        remove_pid_info_if_stale(pid)
    validation = manager.validate_state()
    state = validation.get("state", {})
    report = build_patrol_report(state, runner_alive=alive)

    return {
        "status": "success",
        "result_type": "patrol_runner_status",
        "process": {
            "pid": pid,
            "alive": alive,
            "pid_file": str(PID_FILE),
            "log_path": str(PROCESS_LOG),
            "started_at": pid_info.get("started_at") if pid_info else None,
            "mode": pid_info.get("mode") if pid_info else None,
        },
        "validation": validation,
        "should_continue": manager.should_continue(),
        "report": report,
        "report_ready": bool(report.get("report_ready")),
        "notify_user": bool(report.get("report_ready")),
        "user_message": report.get("user_message"),
    }


def command_start(args: argparse.Namespace) -> JsonDict:
    pid_info = load_pid_info()
    old_pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    if process_alive(old_pid) and not args.force:
        return {
            "status": "success",
            "result_type": "patrol_runner_already_running",
            "message": "patrol runner is already running",
            "user_message": (
                "当前房间自动巡视已经在后台运行中。我会继续让 runner 执行，"
                "你可以稍后使用 status 查看进度，完成后使用 report 获取总结。"
            ),
            "pid": old_pid,
            "pid_file": str(PID_FILE),
            "log_path": str(PROCESS_LOG),
            "notify_user": True,
            "report_policy": "startup_notice_only",
        }

    remove_pid_info_if_stale(old_pid)

    args.start = True
    args.quiet = True if args.quiet is None else args.quiet
    command = build_runner_args(args, continuous=True)

    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    log_handle = PROCESS_LOG.open("a", encoding="utf-8", newline="\n")
    log_handle.write(f"\n[{now_iso()}] start command: {' '.join(command)}\n")
    log_handle.flush()

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=hidden_creationflags(background=True),
        startupinfo=hidden_startupinfo(),
        close_fds=True,
    )
    log_handle.close()

    pid_payload = {
        "pid": process.pid,
        "started_at": now_iso(),
        "command": command,
        "log_path": str(PROCESS_LOG),
    }
    write_pid_info(pid_payload)

    return {
        "status": "success",
        "result_type": "patrol_runner_started",
        "message": "patrol runner started in background",
        "user_message": (
            "已启动当前房间自动巡视，后台 runner 正在执行。"
            "你可以稍后使用 status 查看进度，完成后使用 report 获取总结。"
        ),
        "pid": process.pid,
        "pid_file": str(PID_FILE),
        "log_path": str(PROCESS_LOG),
        "state_hint": "Use --command status to check progress, or --command stop to request a graceful stop.",
        "notify_user": True,
        "report_policy": "startup_notice_only",
    }


def command_report(*, finalize: bool = False) -> JsonDict:
    status = command_status()
    report = status.get("report", {})
    process = status.get("process", {}) if isinstance(status.get("process"), dict) else {}
    finalization: Optional[JsonDict] = None
    if (
        finalize
        and bool(report.get("report_ready"))
        and not bool(process.get("alive", False))
    ):
        manager = StateManager(MEMORY_DIR)
        finalization = manager.finalize_report(
            reason="report_delivered",
            report_message=str(report.get("user_message") or ""),
        )
    return {
        "status": "success",
        "result_type": "patrol_runner_report",
        "process": process,
        "validation": status.get("validation", {}),
        "should_continue": status.get("should_continue", {}),
        "report": report,
        "report_ready": bool(report.get("report_ready")),
        "notify_user": True,
        "user_message": report.get("user_message"),
        "finalized_after_report": finalization is not None,
        "finalization": finalization,
    }


def command_run(args: argparse.Namespace) -> JsonDict:
    pid_info = load_pid_info()
    old_pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    if process_alive(old_pid):
        status = command_status()
        forced_suffix = (
            "即使传入 --force，direct run 也不会和一个仍存活的 runner 并发控制机器人。"
            if args.force
            else "为了避免两个 runner 同时控制机器人，本次没有启动新的直接运行任务。"
        )
        return {
            "status": "success",
            "result_type": "patrol_runner_already_running",
            "message": "patrol runner is already running",
            "process": status.get("process", {}),
            "report": status.get("report", {}),
            "notify_user": True,
            "user_message": (
                f"当前已有巡视任务在运行中。{forced_suffix}"
                "你可以使用 status 查看进度，或先 stop 后再 run。"
            ),
        }

    remove_pid_info_if_stale(old_pid)

    args.start = True
    command = build_runner_args(args, continuous=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    pid_payload = {
        "pid": os.getpid(),
        "started_at": now_iso(),
        "command": command,
        "log_path": str(PROCESS_LOG),
        "mode": "direct_run",
    }
    write_pid_info(pid_payload)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")

    completed: Optional[subprocess.CompletedProcess[str]] = None
    try:
        with PROCESS_LOG.open("a", encoding="utf-8", newline="\n") as log_handle:
            log_handle.write(f"\n[{now_iso()}] direct run command: {' '.join(command)}\n")
            log_handle.flush()

            completed = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.run_timeout if args.run_timeout and args.run_timeout > 0 else None,
                env=env,
                check=False,
                startupinfo=hidden_startupinfo(),
                creationflags=hidden_creationflags(),
            )

            if completed.stdout:
                log_handle.write(completed.stdout)
                if not completed.stdout.endswith("\n"):
                    log_handle.write("\n")
            if completed.stderr:
                log_handle.write("[stderr]\n")
                log_handle.write(completed.stderr)
                if not completed.stderr.endswith("\n"):
                    log_handle.write("\n")
            log_handle.write(f"[{now_iso()}] direct run finished: returncode={completed.returncode}\n")
    finally:
        remove_pid_info_if_current_process()

    stdout = completed.stdout if completed else ""
    stderr = completed.stderr if completed else ""
    events = parse_json_lines(stdout)
    report_result = command_report(finalize=not args.no_finalize)
    report = report_result.get("report", {})
    runner_returncode = completed.returncode if completed else 1
    success = runner_returncode == 0 and report_result.get("status") == "success"

    return {
        "status": "success" if success else "error",
        "result_type": "patrol_runner_run_finished",
        "runner_returncode": runner_returncode,
        "events_tail": events[-8:],
        "stdout_tail": stdout.splitlines()[-8:],
        "stderr_tail": stderr.splitlines()[-8:],
        "log_path": str(PROCESS_LOG),
        "report": report,
        "report_ready": bool(report.get("report_ready")),
        "notify_user": True,
        "user_message": report.get("user_message"),
        "finalized_after_report": bool(report_result.get("finalized_after_report")),
        "finalization": report_result.get("finalization"),
    }


def command_segment(args: argparse.Namespace) -> JsonDict:
    command = build_runner_args(args, continuous=False)
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=max(args.timeout * (args.segment_steps + 2), args.timeout + 30),
        check=False,
        startupinfo=hidden_startupinfo(),
        creationflags=hidden_creationflags(),
    )
    events = parse_json_lines(completed.stdout)
    return {
        "status": "success" if completed.returncode == 0 else "error",
        "result_type": "patrol_runner_segment_finished",
        "runner_returncode": completed.returncode,
        "events_tail": events[-8:],
        "stdout_tail": completed.stdout.splitlines()[-8:],
        "stderr_tail": completed.stderr.splitlines()[-8:],
    }


def command_stop(args: argparse.Namespace) -> JsonDict:
    manager = StateManager(MEMORY_DIR)
    state = manager.stop_mission(reason=args.reason)
    pid_info = load_pid_info()
    pid = int(pid_info.get("pid")) if pid_info and pid_info.get("pid") else None
    alive = process_alive(pid)
    user_message = (
        "已请求停止当前巡视任务，后台 runner 仍在当前安全检查点内；"
        "它会在本步结束后退出，不应再开启新的巡视步骤。"
        if alive
        else "当前巡视任务已停止，后台 runner 未在运行。"
    )

    return {
        "status": "success",
        "result_type": "patrol_runner_stop_requested",
        "message": "mission disabled; background runner will stop after its current safe check",
        "user_message": user_message,
        "reason": args.reason,
        "notify_user": True,
        "stop_state": "stop_pending" if alive else "stopped",
        "process": {
            "pid": pid,
            "alive": alive,
            "log_path": str(PROCESS_LOG),
        },
        "state": {
            "patrol_enabled": state.patrol.get("enabled"),
            "patrol_mode": state.patrol.get("mode"),
            "mission_enabled": state.mission.get("enabled"),
            "mission_mode": state.mission.get("mode"),
            "step_count": state.patrol.get("step_count"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenClaw skill wrapper for the patrol runner.")
    parser.add_argument(
        "--command",
        choices=["start", "run", "segment", "status", "stop", "report"],
        default="start",
        help="Skill command to execute.",
    )
    parser.add_argument("--room", default=DEFAULT_ROOM)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--segment-steps", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument(
        "--clean-validation",
        choices=["visual", "metadata", "off"],
        default=os.getenv("ROBOT_CLEAN_VALIDATION", "visual"),
        help="V2 default is visual; metadata is only for V1 baseline/offline debug.",
    )
    parser.add_argument(
        "--perception-backend",
        choices=["opencv", "yolo"],
        default=os.getenv("ROBOT_PERCEPTION_BACKEND", "yolo"),
        help="Scene analysis backend passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--task-mode",
        choices=["clean", "tidy"],
        default=os.getenv("ROBOT_TASK_MODE", "clean"),
        help="Task policy passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--interaction-grounding",
        choices=["metadata-hidden", "legacy-metadata"],
        default=os.getenv("ROBOT_INTERACTION_GROUNDING", "metadata-hidden"),
        help="Pickup/place grounding policy passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--pickup-surface-policy",
        choices=["floor-only", "any-surface"],
        default=os.getenv("ROBOT_PICKUP_SURFACE_POLICY", "floor-only"),
        help="Pickup target surface policy passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--pickup-target-labels",
        default=os.getenv("ROBOT_PICKUP_TARGET_LABELS", ""),
        help="Optional comma-separated pickup label allowlist passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--performance-profile",
        choices=["balanced", "full"],
        default=os.getenv("ROBOT_PERFORMANCE_PROFILE", "balanced"),
        help="Runtime cost profile passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--object-memory-update-interval",
        type=int,
        default=env_int("ROBOT_OBJECT_MEMORY_UPDATE_INTERVAL", 3),
        help="Idle exploration object-memory update interval passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--semantic-map-update-interval",
        type=int,
        default=env_int("ROBOT_SEMANTIC_MAP_UPDATE_INTERVAL", 3),
        help="Idle exploration semantic-map update interval passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--log-detail",
        choices=["summary", "full"],
        default=os.getenv("ROBOT_PATROL_LOG_DETAIL", "summary"),
        help="Verbose script-result detail passed to scripts/patrol_runner.py.",
    )
    parser.add_argument(
        "--run-timeout",
        type=int,
        default=0,
        help="Optional wall-clock timeout for --command run. 0 means no wrapper timeout.",
    )
    parser.add_argument("--start", action="store_true", help="For --command segment, start mission first.")
    parser.add_argument("--no-reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Do not reset memory to IDLE after a ready report is returned.",
    )
    parser.add_argument("--force", action="store_true", help="Start a new background runner even if a PID exists.")
    parser.add_argument("--reason", default="user_stop", help="Stop reason for --command stop.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "status":
            result = command_status()
        elif args.command == "report":
            result = command_report(finalize=not args.no_finalize)
        elif args.command == "run":
            result = command_run(args)
        elif args.command == "start":
            result = command_start(args)
        elif args.command == "segment":
            result = command_segment(args)
        elif args.command == "stop":
            result = command_stop(args)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except subprocess.TimeoutExpired as exc:
        result = {
            "status": "error",
            "result_type": "patrol_runner_skill_timeout",
            "message": str(exc),
        }
    except Exception as exc:
        result = {
            "status": "error",
            "result_type": "patrol_runner_skill_error",
            "message": str(exc),
        }

    print_json(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
