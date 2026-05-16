#!/usr/bin/env python3
"""
Generate a V1 research/evaluation report from robot-cleaner memory files.

The script is read-only. It does not call the backend, move the robot, or use
OpenClaw tokens. It summarizes:

- memory/patrol-state.json
- memory/mission-state.json
- memory/room-state.json
- memory/YYYY-MM-DD-patrol-runner.md
- latest memory/alignment-samples/*/summary.json, when available
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_REPORT_DIR = MEMORY_DIR / "reports"

JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_time() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc), "_path": str(path)}
    return data if isinstance(data, dict) else {"_value": data}


def parse_daily_events(path: Path) -> List[JsonDict]:
    events: List[JsonDict] = []
    if not path.exists():
        return events

    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), start=1):
        start = line.find("{")
        if start < 0:
            continue
        raw_json = line[start:].strip()
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data.setdefault("_line_no", line_no)
            events.append(data)
    return events


def latest_run_events(events: Sequence[JsonDict]) -> List[JsonDict]:
    if not events:
        return []
    start_index = 0
    for index, event in enumerate(events):
        if event.get("event") == "mission_started":
            start_index = index
    return list(events[start_index:])


def latest_alignment_summary(memory_dir: Path) -> Optional[JsonDict]:
    root = memory_dir / "alignment-samples"
    if not root.exists():
        return None

    candidates: List[Tuple[float, Path]] = []
    for path in root.glob("*/summary.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if not candidates:
        return None

    _, path = max(candidates, key=lambda item: item[0])
    data = load_json(path, {})
    data["_path"] = str(path)
    return data


def truthy(value: Any) -> bool:
    return bool(value)


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def value_or_na(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def compact_list(values: Iterable[Any], *, limit: int = 8) -> str:
    items = [str(item) for item in values if str(item).strip()]
    if not items:
        return "none"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit})"


def event_counter(events: Sequence[JsonDict], event_name: str) -> List[JsonDict]:
    return [event for event in events if event.get("event") == event_name]


def last_event(events: Sequence[JsonDict], event_name: str) -> Optional[JsonDict]:
    for event in reversed(events):
        if event.get("event") == event_name:
            return event
    return None


def script_success_metrics(script_results: Sequence[JsonDict], script_name: str) -> JsonDict:
    rows = [event for event in script_results if event.get("script") == script_name]
    success = [event for event in rows if event.get("returncode") == 0 and event.get("status") == "success"]
    return {
        "total": len(rows),
        "success": len(success),
        "success_rate": ratio(len(success), len(rows)),
    }


def build_metrics(
    *,
    patrol: JsonDict,
    mission: JsonDict,
    room: JsonDict,
    events: Sequence[JsonDict],
    alignment: Optional[JsonDict],
) -> JsonDict:
    script_results = event_counter(events, "script_result")
    decisions = event_counter(events, "decision")
    steps = event_counter(events, "step_finished")
    target_validations = event_counter(events, "target_validation")
    suppressions = event_counter(events, "candidate_suppressed")
    nav_observed = event_counter(events, "navigation_observed")
    nav_recommendations = event_counter(events, "navigation_recommendation")
    nav_updates = event_counter(events, "navigation_updated")

    step_success = [event for event in steps if truthy(event.get("success"))]
    recorded_steps = [event for event in steps if truthy(event.get("step_recorded"))]
    action_counts = Counter(str(event.get("action")) for event in steps if event.get("action"))
    decision_action_counts = Counter(str(event.get("action")) for event in decisions if event.get("action"))
    decision_kind_counts = Counter(str(event.get("kind")) for event in decisions if event.get("kind"))
    validation_type_counts = Counter(str(event.get("result_type")) for event in target_validations)
    backend_allowed = [event for event in target_validations if truthy(event.get("clean_allowed"))]
    backend_rejected = [event for event in target_validations if not truthy(event.get("clean_allowed"))]
    clean_results = [event for event in script_results if event.get("script") == "clean_garbage"]
    clean_success = [
        event
        for event in clean_results
        if event.get("returncode") == 0
        and event.get("status") == "success"
        and event.get("result_type") == "clean_executed"
    ]

    room_complete_event = last_event(events, "room_marked_complete")
    recover_event = last_event(events, "patrol_recover_failed")
    completion_reason = None
    if room_complete_event:
        completion_reason = room_complete_event.get("reason")
    elif recover_event:
        completion_reason = recover_event.get("reason")
    elif mission.get("final_summary"):
        completion_reason = mission.get("final_summary")

    latest_nav_event = nav_updates[-1] if nav_updates else {}

    metrics: JsonDict = {
        "generated_at": now_iso(),
        "task": {
            "mission": mission.get("mission"),
            "mode": mission.get("mode"),
            "enabled": mission.get("enabled"),
            "room": mission.get("current_room") or room.get("room_name"),
            "step_count": patrol.get("step_count"),
            "max_steps": patrol.get("max_steps"),
            "room_complete": room.get("room_complete"),
            "completion_reason": completion_reason,
            "final_summary": mission.get("final_summary"),
        },
        "memory": {
            "targets_found": room.get("targets_found", []),
            "targets_cleaned": room.get("targets_cleaned", []),
            "failed_attempts": patrol.get("failed_attempts"),
            "last_action": patrol.get("last_action"),
            "last_update": mission.get("last_update") or patrol.get("last_update"),
        },
        "navigation": {
            "visited_cell_count": len(room.get("visited_cells", []) or []),
            "coverage_estimate": room.get("coverage_estimate", 0.0),
            "frontier_count": len(room.get("frontier_cells", []) or []),
            "known_open_edge_count": len(room.get("known_open_edges", []) or []),
            "blocked_edge_count": len(room.get("blocked_edges", []) or []),
            "collision_count": room.get("collision_count", 0),
            "oscillation_count": room.get("oscillation_count", 0),
            "stagnation_count": room.get("stagnation_count", 0),
            "turn_streak_count": room.get("turn_streak_count", 0),
            "last_cell": room.get("last_cell"),
            "last_heading": room.get("last_heading"),
            "last_navigation_decision": room.get("last_navigation_decision", {}),
            "navigation_update_count": len(nav_updates),
            "navigation_observe_count": len(nav_observed),
            "navigation_recommendation_count": len(nav_recommendations),
            "latest_navigation_event": latest_nav_event,
        },
        "perception": {
            "get_vision": script_success_metrics(script_results, "get_vision"),
            "analyze_scene": script_success_metrics(script_results, "analyze_scene"),
        },
        "actions": {
            "recorded_step_count": len(recorded_steps),
            "step_finished_count": len(steps),
            "successful_step_count": len(step_success),
            "action_success_rate": ratio(len(step_success), len(steps)),
            "action_counts": dict(action_counts),
            "decision_action_counts": dict(decision_action_counts),
            "decision_kind_counts": dict(decision_kind_counts),
            "clean_attempt_count": len(clean_results),
            "clean_success_count": len(clean_success),
            "clean_success_rate": ratio(len(clean_success), len(clean_results)),
        },
        "perception_action_consistency": {
            "target_validation_count": len(target_validations),
            "backend_allowed_clean_count": len(backend_allowed),
            "backend_rejected_clean_count": len(backend_rejected),
            "validation_result_type_counts": dict(validation_type_counts),
            "candidate_suppression_count": len(suppressions),
        },
        "events": {
            "total_events_in_selected_run": len(events),
            "event_counts": dict(Counter(str(event.get("event")) for event in events)),
        },
        "latest_alignment_sample_summary": alignment,
    }

    metrics["v1_pass"] = evaluate_v1_pass(metrics)
    return metrics


def evaluate_v1_pass(metrics: JsonDict) -> JsonDict:
    task = metrics.get("task", {})
    navigation = metrics.get("navigation", {})
    actions = metrics.get("actions", {})
    completion_reason = str(task.get("completion_reason") or "")

    checks = {
        "room_complete": bool(task.get("room_complete")),
        "coverage_ge_0_95": float(navigation.get("coverage_estimate", 0.0) or 0.0) >= 0.95,
        "visited_cells_ge_80": int(navigation.get("visited_cell_count", 0) or 0) >= 80,
        "frontier_count_zero": int(navigation.get("frontier_count", 0) or 0) == 0,
        "action_success_rate_ge_0_90": (actions.get("action_success_rate") or 0.0) >= 0.90,
        "not_recover_failed": "recover_failed" not in completion_reason,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def markdown_table(rows: Sequence[Tuple[str, Any]]) -> List[str]:
    lines = ["| Item | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | {value_or_na(value)} |")
    return lines


def write_markdown_report(path: Path, metrics: JsonDict, *, source_log: Path) -> None:
    task = metrics["task"]
    memory = metrics["memory"]
    nav = metrics["navigation"]
    perception = metrics["perception"]
    actions = metrics["actions"]
    consistency = metrics["perception_action_consistency"]
    alignment = metrics.get("latest_alignment_sample_summary") or {}
    v1_pass = metrics["v1_pass"]

    lines: List[str] = [
        "# V1 Evaluation Report",
        "",
        f"- generated_at: {metrics['generated_at']}",
        f"- source_log: `{source_log}`",
        "",
        "## Verdict",
        "",
        f"- V1 pass: **{v1_pass['passed']}**",
        f"- completion_reason: `{task.get('completion_reason')}`",
        f"- final_summary: `{task.get('final_summary')}`",
        "",
        "## Task State",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                ("mission", task.get("mission")),
                ("room", task.get("room")),
                ("mode", task.get("mode")),
                ("enabled", task.get("enabled")),
                ("step_count", task.get("step_count")),
                ("max_steps", task.get("max_steps")),
                ("room_complete", task.get("room_complete")),
                ("targets_found", compact_list(memory.get("targets_found", []))),
                ("targets_cleaned", compact_list(memory.get("targets_cleaned", []))),
                ("failed_attempts", memory.get("failed_attempts")),
                ("last_action", memory.get("last_action")),
            ]
        )
    )
    lines.extend(
        [
            "",
            "> Note: in V1, `targets_found` is a visual-candidate memory field. "
            "`targets_cleaned`, backend validation counts, and clean success counts "
            "are the stronger evidence for confirmed cleanable targets.",
        ]
    )

    lines.extend(["", "## Navigation Metrics", ""])
    lines.extend(
        markdown_table(
            [
                ("visited_cell_count", nav.get("visited_cell_count")),
                ("coverage_estimate", nav.get("coverage_estimate")),
                ("frontier_count", nav.get("frontier_count")),
                ("known_open_edge_count", nav.get("known_open_edge_count")),
                ("blocked_edge_count", nav.get("blocked_edge_count")),
                ("collision_count", nav.get("collision_count")),
                ("oscillation_count", nav.get("oscillation_count")),
                ("stagnation_count", nav.get("stagnation_count")),
                ("turn_streak_count", nav.get("turn_streak_count")),
                ("last_cell", nav.get("last_cell")),
                ("last_heading", nav.get("last_heading")),
            ]
        )
    )

    lines.extend(["", "## Perception Metrics", ""])
    lines.extend(
        markdown_table(
            [
                ("get_vision_total", perception["get_vision"]["total"]),
                ("get_vision_success_rate", percent(perception["get_vision"]["success_rate"])),
                ("analyze_scene_total", perception["analyze_scene"]["total"]),
                ("analyze_scene_success_rate", percent(perception["analyze_scene"]["success_rate"])),
            ]
        )
    )

    lines.extend(["", "## Action Metrics", ""])
    lines.extend(
        markdown_table(
            [
                ("recorded_step_count", actions.get("recorded_step_count")),
                ("step_finished_count", actions.get("step_finished_count")),
                ("action_success_rate", percent(actions.get("action_success_rate"))),
                ("clean_attempt_count", actions.get("clean_attempt_count")),
                ("clean_success_count", actions.get("clean_success_count")),
                ("clean_success_rate", percent(actions.get("clean_success_rate"))),
                ("action_counts", json.dumps(actions.get("action_counts", {}), ensure_ascii=False)),
            ]
        )
    )

    lines.extend(["", "## Perception-Action Consistency", ""])
    lines.extend(
        markdown_table(
            [
                ("target_validation_count", consistency.get("target_validation_count")),
                ("backend_allowed_clean_count", consistency.get("backend_allowed_clean_count")),
                ("backend_rejected_clean_count", consistency.get("backend_rejected_clean_count")),
                ("candidate_suppression_count", consistency.get("candidate_suppression_count")),
                (
                    "validation_result_type_counts",
                    json.dumps(consistency.get("validation_result_type_counts", {}), ensure_ascii=False),
                ),
            ]
        )
    )

    if alignment:
        lines.extend(["", "## Latest Alignment Sampling", ""])
        lines.extend(
            markdown_table(
                [
                    ("summary_path", alignment.get("_path")),
                    ("sample_count", alignment.get("sample_count")),
                    ("counts", json.dumps(alignment.get("counts", {}), ensure_ascii=False)),
                    ("precision", alignment.get("metrics", {}).get("precision")),
                    ("recall", alignment.get("metrics", {}).get("recall")),
                    ("accuracy", alignment.get("metrics", {}).get("accuracy")),
                ]
            )
        )

    lines.extend(["", "## V1 Pass Checklist", ""])
    lines.extend(["| Check | Pass |", "| --- | --- |"])
    for key, value in v1_pass.get("checks", {}).items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            "- The current V1 demonstrates an active task-planning loop rather than one-shot command execution.",
            "- Visual detections are treated as hypotheses; backend validation prevents many false visual candidates from becoming clean actions.",
            "- Navigation memory provides a measurable frontier-like coverage signal instead of purely reactive wandering.",
            "- The remaining V2 priority is to separate visual candidates from confirmed targets in memory and to replace/augment the OpenCV baseline with a stronger perception module.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_report(path: Path, metrics: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a V1 research report from memory and logs.")
    parser.add_argument("--memory-dir", type=Path, default=MEMORY_DIR)
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--all-runs", action="store_true", help="Use the whole daily log instead of only the latest run.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    memory_dir = args.memory_dir
    log_path = args.log_path or memory_dir / f"{args.date}-patrol-runner.md"

    patrol = load_json(memory_dir / "patrol-state.json", {})
    mission = load_json(memory_dir / "mission-state.json", {})
    room = load_json(memory_dir / "room-state.json", {})
    all_events = parse_daily_events(log_path)
    selected_events = all_events if args.all_runs else latest_run_events(all_events)
    alignment = latest_alignment_summary(memory_dir)

    metrics = build_metrics(
        patrol=patrol,
        mission=mission,
        room=room,
        events=selected_events,
        alignment=alignment,
    )

    stem = f"{safe_time()}-v1-report"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path = args.output_dir / f"{stem}.json"
    write_markdown_report(markdown_path, metrics, source_log=log_path)
    write_json_report(json_path, metrics)

    result = {
        "status": "success",
        "result_type": "v1_report_generated",
        "markdown_report": str(markdown_path),
        "json_report": str(json_path),
        "v1_pass": metrics["v1_pass"]["passed"],
        "completion_reason": metrics["task"].get("completion_reason"),
    }
    if not args.quiet:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
