#!/usr/bin/env python3
"""Persistent exploration-goal management for tidy-room patrol.

This module owns the long-lived exploration objective that the LLM-facing
decision context can reference.  It is deliberately separate from
decision_context_builder: the builder presents options, while this module keeps
the navigation target stable across turns.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from scripts.global_planner import AStarGlobalPlanner
    from scripts.map_backend import load_map_backend
    from scripts.position_map_core import CELL_FREE, CELL_INFLATED, CELL_OCCUPIED, four_neighbors, manhattan_distance
except ImportError:  # pragma: no cover - direct script execution
    from global_planner import AStarGlobalPlanner
    from map_backend import load_map_backend
    from position_map_core import CELL_FREE, CELL_INFLATED, CELL_OCCUPIED, four_neighbors, manhattan_distance


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"

ACTIVE_FRONTIER_SCHEMA = "robot_cleaner_active_frontier_goal_v1"
COVERAGE_PATROL_SCHEMA = "robot_cleaner_coverage_patrol_v1"
EXPLORATION_GOAL_UPDATE_SCHEMA = "robot_cleaner_exploration_goal_update_v1"

FRONTIER_REACHED_RADIUS = 0
FRONTIER_STICKY_MAX_STEPS = 24
FRONTIER_STALE_PATIENCE = 5
FRONTIER_COOLDOWN_STEPS = 8
COVERAGE_PATROL_FRONTIER_THRESHOLD = 0
COVERAGE_PATROL_MIN_VISITED_CELLS = 4

JsonDict = dict[str, Any]


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


def load_json(path: Path, default: JsonDict | None = None) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return dict(default or {})
    return data if isinstance(data, dict) else dict(default or {})


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_cell(value: Any) -> tuple[int, int] | None:
    try:
        left, right = str(value or "").split(",", 1)
        return int(left), int(right)
    except (TypeError, ValueError):
        return None


def unique_strings(*values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in as_list(value):
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
    return result


def current_step(room: JsonDict) -> int:
    for key in ("step_count", "explored_steps"):
        try:
            return max(0, int(room.get(key, 0) or 0))
        except (TypeError, ValueError):
            continue
    return 0


def cell_record(position_status: JsonDict, cell: str) -> JsonDict:
    return as_dict(as_dict(position_status.get("cells")).get(cell))


def cell_state(position_status: JsonDict, cell: str) -> str:
    return str(cell_record(position_status, cell).get("state") or "unknown")


def traversable_cell(position_status: JsonDict, cell: str) -> bool:
    state = cell_state(position_status, cell)
    return state not in {CELL_OCCUPIED, CELL_INFLATED}


def frontier_cells(position_status: JsonDict, room: JsonDict) -> list[str]:
    return [
        cell
        for cell in unique_strings(position_status.get("frontiers"), position_status.get("frontier_cells"), room.get("frontier_cells"))
        if parse_cell(cell) is not None
    ]


def frontier_cluster_map(frontiers: Sequence[str]) -> dict[str, list[str]]:
    remaining = {str(cell) for cell in frontiers if parse_cell(cell) is not None}
    clusters: dict[str, list[str]] = {}
    while remaining:
        root = remaining.pop()
        stack = [root]
        cluster = [root]
        while stack:
            current = stack.pop()
            for neighbor in four_neighbors(current):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)
                cluster.append(neighbor)
        cluster.sort()
        for cell in cluster:
            clusters[cell] = cluster
    return clusters


def active_goal_valid(
    *,
    active: JsonDict,
    active_route: JsonDict,
    current_cell: str,
    step: int,
    frontiers: Sequence[str],
    cooldowns: JsonDict,
) -> tuple[bool, str]:
    goal = str(active.get("cell") or "").strip()
    if not goal:
        return False, "missing_goal_cell"
    if current_cell == goal:
        return False, "frontier_reached"
    if goal not in set(frontiers):
        return False, "goal_no_longer_frontier"
    if goal in cooldowns:
        return False, "goal_in_cooldown"
    if (
        active_route.get("status") == "blocked"
        and str(active_route.get("goal_cell") or "") == goal
    ):
        return False, "active_route_blocked_by_costmap"
    try:
        started_step = int(active.get("started_step", step) or step)
    except (TypeError, ValueError):
        started_step = step
    if step - started_step > FRONTIER_STICKY_MAX_STEPS:
        return False, "goal_sticky_timeout"
    try:
        stale_count = int(active.get("stale_count", 0) or 0)
    except (TypeError, ValueError):
        stale_count = 0
    if stale_count > FRONTIER_STALE_PATIENCE:
        return False, "goal_no_progress"
    return True, "active_goal_valid"


def cooldown_goal(room: JsonDict, cell: str, *, reason: str, step: int) -> None:
    if not cell:
        return
    cooldowns = room.setdefault("frontier_cooldowns", {})
    if not isinstance(cooldowns, dict):
        cooldowns = {}
        room["frontier_cooldowns"] = cooldowns
    cooldowns[cell] = {
        "cell": cell,
        "reason": reason,
        "until_step": int(step) + FRONTIER_COOLDOWN_STEPS,
        "updated_at": now_iso(),
    }


def active_cooldowns(room: JsonDict, step: int) -> JsonDict:
    raw = room.get("frontier_cooldowns") if isinstance(room.get("frontier_cooldowns"), dict) else {}
    active: JsonDict = {}
    for cell, entry in raw.items():
        item = as_dict(entry)
        try:
            until_step = int(item.get("until_step", 0) or 0)
        except (TypeError, ValueError):
            until_step = 0
        if until_step > step:
            active[str(cell)] = item
    room["frontier_cooldowns"] = active
    return active


def append_frontier_history(room: JsonDict, event: str, **payload: Any) -> None:
    history = room.setdefault("frontier_history", [])
    if not isinstance(history, list):
        history = []
        room["frontier_history"] = history
    history.append({"time": now_iso(), "event": event, **payload})
    room["frontier_history"] = history[-60:]


def make_frontier_goal(
    *,
    plan: JsonDict,
    current_cell: str,
    step: int,
    cluster: Sequence[str],
    previous: JsonDict | None = None,
) -> JsonDict:
    previous = as_dict(previous)
    selected = str(plan.get("selected_goal_cell") or plan.get("requested_target_cell") or "").strip()
    same = bool(previous and previous.get("cell") == selected)
    try:
        last_distance = manhattan_distance(current_cell, selected)
    except Exception:
        last_distance = previous.get("last_distance")
    return {
        "schema": ACTIVE_FRONTIER_SCHEMA,
        "mode": "frontier_cluster",
        "status": "active",
        "cell": selected,
        "cluster_cells": list(cluster),
        "cluster_size": len(cluster),
        "started_step": int(previous.get("started_step", step) if same else step),
        "last_selected_step": step,
        "started_from_cell": previous.get("started_from_cell", current_cell) if same else current_cell,
        "last_distance": last_distance,
        "stale_count": int(previous.get("stale_count", 0) or 0) if same else 0,
        "path": list(plan.get("path") or []),
        "next_action": plan.get("next_action"),
        "next_cell": plan.get("next_cell"),
        "planner": "astar",
        "plan_status": plan.get("status"),
        "route_cost": plan.get("route_cost"),
        "objective_score": plan.get("objective_score"),
        "reason": plan.get("replan_reason") or plan.get("target_reason") or "frontier_cluster_patrol",
        "updated_at": now_iso(),
    }


def update_active_goal_progress(room: JsonDict, *, current_cell: str, step: int) -> None:
    active = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
    if not active:
        return
    goal = str(active.get("cell") or "").strip()
    if not goal:
        room["active_frontier_goal"] = None
        return
    if current_cell == goal:
        append_frontier_history(room, "frontier_reached", cell=goal, step=step)
        cooldown_goal(room, goal, reason="frontier_reached", step=step)
        room["active_frontier_goal"] = None
        return
    try:
        distance = manhattan_distance(current_cell, goal)
    except Exception:
        return
    try:
        previous_distance = int(active.get("last_distance", distance) or distance)
    except (TypeError, ValueError):
        previous_distance = distance
    stale = int(active.get("stale_count", 0) or 0)
    if distance < previous_distance:
        stale = 0
    elif distance >= previous_distance:
        stale += 1
    active["last_distance"] = distance
    active["stale_count"] = stale
    active["last_progress_step"] = step
    active["updated_at"] = now_iso()
    room["active_frontier_goal"] = active


def plan_frontier_goal(
    *,
    memory_dir: Path,
    position_status: JsonDict,
    current_cell: str,
    current_heading: str,
    frontiers: Sequence[str],
    preferred: str | None,
    cooldowns: JsonDict,
) -> JsonDict:
    planner = AStarGlobalPlanner(memory_dir)
    return planner.plan_to_best_frontier(
        current_cell=current_cell,
        current_heading=current_heading,
        frontier_cells=list(frontiers),
        position_status=position_status,
        semantic_status={"cells": {}},
        preferred_frontier=preferred,
        frontier_cooldowns=cooldowns,
        target_reason="active_frontier_goal",
        allow_backtrack=True,
    )


def coverage_candidates(position_status: JsonDict, current_cell: str, *, limit: int = 8) -> list[JsonDict]:
    cells = as_dict(position_status.get("cells"))
    candidates: list[JsonDict] = []
    for cell, record in cells.items():
        cell = str(cell)
        if parse_cell(cell) is None or cell == current_cell:
            continue
        rec = as_dict(record)
        state = str(rec.get("state") or "")
        if state not in {"", CELL_FREE} and rec.get("visited") is not True:
            continue
        if not traversable_cell(position_status, cell):
            continue
        try:
            distance = manhattan_distance(current_cell, cell)
        except Exception:
            continue
        if distance <= 1:
            continue
        seen_count = int(rec.get("seen_count", rec.get("visit_count", 0)) or 0)
        collision_count = int(rec.get("collision_count", 0) or 0)
        if collision_count > 0:
            continue
        unknown_neighbors = sum(
            1
            for neighbor in four_neighbors(cell)
            if not as_dict(cells.get(neighbor)) or str(as_dict(cells.get(neighbor)).get("state") or "unknown") == "unknown"
        )
        score = (2.0 / float(max(1, seen_count + 1))) + min(8.0, float(distance)) * 0.15 + unknown_neighbors * 0.35
        candidates.append(
            {
                "cell": cell,
                "score": round(score, 4),
                "distance_steps": distance,
                "seen_count": seen_count,
                "unknown_neighbor_count": unknown_neighbors,
                "reason": "low_visit_known_free_cell",
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["seen_count"]), int(item["distance_steps"]), str(item["cell"])))
    return candidates[: max(1, int(limit))]


def build_coverage_patrol(
    *,
    position_status: JsonDict,
    room: JsonDict,
    current_cell: str,
    frontiers: Sequence[str],
) -> JsonDict:
    coverage = float(room.get("coverage_estimate", position_status.get("coverage_estimate", 0.0)) or 0.0)
    visited_count = len(as_list(room.get("visited_cells"))) or int(as_dict(position_status.get("stats")).get("visited_cell_count", 0) or 0)
    active = bool(len(frontiers) <= COVERAGE_PATROL_FRONTIER_THRESHOLD and visited_count >= COVERAGE_PATROL_MIN_VISITED_CELLS)
    candidates = coverage_candidates(position_status, current_cell, limit=8) if active else []
    target = candidates[0] if candidates else {}
    return {
        "schema": COVERAGE_PATROL_SCHEMA,
        "active": bool(active and target),
        "mode": "coverage_patrol" if active and target else "frontier_exploration",
        "coverage_estimate": round(coverage, 4),
        "visited_cell_count": visited_count,
        "frontier_count": len(frontiers),
        "target_cell": target.get("cell"),
        "target_score": target.get("score"),
        "candidate_cells": candidates,
        "reason": "no_frontiers_remaining_cover_undervisited_known_free_cells" if active and target else "frontiers_available_or_insufficient_map",
        "updated_at": now_iso(),
    }


def update_exploration_goals(*, memory_dir: Path | str = MEMORY_DIR, persist: bool = True) -> JsonDict:
    memory_dir = Path(memory_dir)
    snapshot = load_map_backend(memory_dir).load_snapshot()
    position_status = snapshot.to_position_status()
    room = snapshot.to_navigation_room_state()
    pose = as_dict(position_status.get("pose"))
    current_cell = str(pose.get("cell") or position_status.get("last_cell") or room.get("last_cell") or "0,0")
    current_heading = str(pose.get("heading") or position_status.get("last_heading") or room.get("last_heading") or "north")
    step = current_step(room)
    update_active_goal_progress(room, current_cell=current_cell, step=step)

    frontiers = frontier_cells(position_status, room)
    clusters = frontier_cluster_map(frontiers)
    cooldowns = active_cooldowns(room, step)
    active = room.get("active_frontier_goal") if isinstance(room.get("active_frontier_goal"), dict) else None
    active_route = room.get("active_route") if isinstance(room.get("active_route"), dict) else {}
    preferred = str(as_dict(active).get("cell") or "").strip() or None
    valid = False
    validity_reason = "no_active_goal"
    if active:
        valid, validity_reason = active_goal_valid(
            active=active,
            active_route=as_dict(active_route),
            current_cell=current_cell,
            step=step,
            frontiers=frontiers,
            cooldowns=cooldowns,
        )
        if not valid:
            old_goal = str(active.get("cell") or "").strip()
            if old_goal:
                cooldown_goal(room, old_goal, reason=validity_reason, step=step)
                append_frontier_history(room, "frontier_goal_dropped", cell=old_goal, reason=validity_reason, step=step)
                cooldowns = active_cooldowns(room, step)
            room["active_frontier_goal"] = None
            preferred = None

    selected_plan: JsonDict = {}
    if frontiers:
        plan = plan_frontier_goal(
            memory_dir=memory_dir,
            position_status=position_status,
            current_cell=current_cell,
            current_heading=current_heading,
            frontiers=frontiers,
            preferred=preferred,
            cooldowns=cooldowns,
        )
        if str(plan.get("status")) == "success":
            selected = str(plan.get("selected_goal_cell") or plan.get("requested_target_cell") or "").strip()
            goal = make_frontier_goal(
                plan=plan,
                current_cell=current_cell,
                step=step,
                cluster=clusters.get(selected, [selected] if selected else []),
                previous=active if valid else None,
            )
            previous_cell = str(as_dict(active).get("cell") or "") if active else ""
            room["active_frontier_goal"] = goal
            room["last_frontier_target"] = selected
            selected_plan = plan
            append_frontier_history(
                room,
                "frontier_goal_selected" if selected != previous_cell else "frontier_goal_refreshed",
                cell=selected,
                step=step,
                next_action=plan.get("next_action"),
                route_cost=plan.get("route_cost"),
            )
        elif preferred:
            cooldown_goal(room, preferred, reason=str(plan.get("replan_reason") or "frontier_plan_failed"), step=step)
            append_frontier_history(room, "frontier_goal_plan_failed", cell=preferred, step=step, status=plan.get("status"))
            room["active_frontier_goal"] = None
            selected_plan = plan
    else:
        room["active_frontier_goal"] = None

    coverage = build_coverage_patrol(
        position_status=position_status,
        room=room,
        current_cell=current_cell,
        frontiers=frontiers,
    )
    room["coverage_patrol"] = coverage
    room["exploration_goal_last_update"] = now_iso()
    if persist:
        atomic_write_json(memory_dir / "room-state.json", room)

    return {
        "status": "success",
        "result_type": "exploration_goals_updated",
        "schema": EXPLORATION_GOAL_UPDATE_SCHEMA,
        "updated_at": room["exploration_goal_last_update"],
        "current_cell": current_cell,
        "current_heading": current_heading,
        "frontier_count": len(frontiers),
        "active_frontier_goal": room.get("active_frontier_goal"),
        "active_goal_validity": validity_reason,
        "coverage_patrol": coverage,
        "selected_plan": {
            key: selected_plan.get(key)
            for key in (
                "status",
                "selected_goal_cell",
                "next_action",
                "next_cell",
                "route_cost",
                "objective_score",
                "replan_reason",
            )
            if selected_plan.get(key) not in (None, "", [], {})
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Update persistent exploration goals.")
    parser.add_argument("--memory-dir", default=str(MEMORY_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    args = parser.parse_args()
    result = update_exploration_goals(memory_dir=Path(args.memory_dir), persist=not args.dry_run)
    if args.format == "compact":
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
