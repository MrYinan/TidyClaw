#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A* global planner for the robot-cleaner semantic position map.

The previous navigation layer used frontier-specific BFS and direct heuristic
turning toward remembered objects.  This module replaces that path logic with a
single bounded A* planner inspired by PythonRobotics' grid A* implementation,
adapted to the project's online-safe Action-Odometry Occupancy Grid.

Design rules:
- four-connected grid, matching the robot's cardinal translation interface;
  by default the action controller uses a non-holonomic front-camera policy:
  turn toward a lateral grid edge first, then validate it with the next RGB-D
  frame before moving ahead.  A holonomic mode is kept for simulator ablations;
- occupied and inflated_occupied cells are not traversable;
- unknown cells are allowed with configurable cost, so the agent can reach a
  frontier or a coarse remembered-object viewpoint;
- blocked edges are first-class constraints;
- semantic-map frontier scores bias exploration toward pickup targets or
  receptacles without replacing collision-safe geometric planning;
- planning is bounded to prevent unbounded search in unknown space.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from scripts.position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        CELL_UNKNOWN,
        PositionMap,
        edge_key,
        format_cell,
        four_neighbors,
        heading_between,
        left_heading,
        manhattan_distance,
        parse_cell,
        right_heading,
    )
    from scripts.semantic_mapping_core import SemanticMap
except ImportError:  # pragma: no cover - direct execution
    from position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        CELL_UNKNOWN,
        PositionMap,
        edge_key,
        format_cell,
        four_neighbors,
        heading_between,
        left_heading,
        manhattan_distance,
        parse_cell,
        right_heading,
    )
    from semantic_mapping_core import SemanticMap


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
GLOBAL_PLAN_PATH = MEMORY_DIR / "global-plan.json"
GLOBAL_PLAN_SCHEMA_VERSION = 1
TASK_PICKUP = "pickup_target"
TASK_RECEPTACLE = "place_receptacle"
DRIVE_MODEL_HOLONOMIC = "holonomic"
DRIVE_MODEL_NONHOLONOMIC = "nonholonomic"

JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def normalize_drive_model(value: Optional[Any]) -> str:
    text = str(value or os.getenv("ROBOT_NAV_DRIVE_MODEL", DRIVE_MODEL_NONHOLONOMIC)).strip().lower()
    if text in {"holonomic", "omni", "omnidirectional", "ai2thor"}:
        return DRIVE_MODEL_HOLONOMIC
    return DRIVE_MODEL_NONHOLONOMIC


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        for attempt in range(20):
            try:
                os.replace(temp_name, path)
                temp_name = ""
                break
            except PermissionError:
                if attempt >= 19:
                    raise
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _blocked_edge_set(position_status: JsonDict, blocked_edges: Optional[Iterable[Any]] = None) -> Set[str]:
    values: List[str] = []
    edges = position_status.get("edges") if isinstance(position_status.get("edges"), dict) else {}
    values.extend(str(item) for item in edges.get("blocked_edges", []) or [])
    values.extend(str(item) for item in position_status.get("blocked_edges", []) or [])
    values.extend(str(item) for item in blocked_edges or [])
    return {value for value in values if value.strip()}


def _edge_is_blocked(blocked: Set[str], source: str, target: str) -> bool:
    return edge_key(source, target) in blocked or edge_key(target, source) in blocked


@dataclass(order=True)
class _QueueNode:
    priority: float
    cell: str = field(compare=False)


@dataclass
class GlobalPlan:
    status: str
    planner: str = "astar"
    target_reason: str = "unknown"
    requested_target_cell: Optional[str] = None
    selected_goal_cell: Optional[str] = None
    selected_goal_kind: str = "goal"
    path: List[str] = field(default_factory=list)
    route_cost: Optional[float] = None
    expanded_node_count: int = 0
    unknown_cell_count: int = 0
    next_cell: Optional[str] = None
    next_action: Optional[str] = None
    target_heading: Optional[str] = None
    semantic_score: float = 0.0
    replan_reason: Optional[str] = None
    candidate_goal_count: int = 0
    frontier_cluster_size: int = 0
    frontier_information_gain: float = 0.0
    frontier_backtrack_penalty: float = 0.0
    frontier_revisit_penalty: float = 0.0
    objective_score: Optional[float] = None
    first_step_kind: Optional[str] = None

    def as_dict(self) -> JsonDict:
        result = asdict(self)
        if result.get("route_cost") is not None:
            result["route_cost"] = round(float(result["route_cost"]), 6)
        result["semantic_score"] = round(float(result.get("semantic_score", 0.0) or 0.0), 6)
        result["frontier_information_gain"] = round(float(result.get("frontier_information_gain", 0.0) or 0.0), 6)
        result["frontier_backtrack_penalty"] = round(float(result.get("frontier_backtrack_penalty", 0.0) or 0.0), 6)
        result["frontier_revisit_penalty"] = round(float(result.get("frontier_revisit_penalty", 0.0) or 0.0), 6)
        if result.get("objective_score") is not None:
            result["objective_score"] = round(float(result["objective_score"]), 6)
        return result


class AStarGlobalPlanner:
    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.path = self.memory_dir / "global-plan.json"
        self.position_map = PositionMap(self.memory_dir)
        self.semantic_map = SemanticMap(self.memory_dir)
        self.unknown_cost = max(1.0, env_float("ROBOT_GLOBAL_PLANNER_UNKNOWN_COST", 2.0))
        self.free_cost = max(0.1, env_float("ROBOT_GLOBAL_PLANNER_FREE_COST", 1.0))
        self.visited_free_discount = clamp(env_float("ROBOT_GLOBAL_PLANNER_VISITED_FREE_DISCOUNT", 0.92), 0.5, 1.0)
        self.goal_substitute_radius = max(0, env_int("ROBOT_GLOBAL_PLANNER_GOAL_SUBSTITUTE_RADIUS", 3))
        self.search_padding_cells = max(3, env_int("ROBOT_GLOBAL_PLANNER_SEARCH_PADDING", 8))
        self.max_expansions = max(64, env_int("ROBOT_GLOBAL_PLANNER_MAX_EXPANSIONS", 5000))
        self.max_frontier_candidates = max(4, env_int("ROBOT_GLOBAL_PLANNER_MAX_FRONTIERS", 48))
        self.reverse_first_step_penalty = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_REVERSE_FIRST_STEP_PENALTY", 3.0))
        self.lateral_first_step_penalty = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_LATERAL_FIRST_STEP_PENALTY", 1.25))
        self.turn_first_step_penalty = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_TURN_FIRST_STEP_PENALTY", 1.10))
        self.forward_first_step_bonus = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FORWARD_FIRST_STEP_BONUS", 1.20))
        self.frontier_sticky_bonus = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_STICKY_BONUS", 4.0))
        self.frontier_cooldown_penalty = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_COOLDOWN_PENALTY", 25.0))
        self.frontier_semantic_weight = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_SEMANTIC_WEIGHT", 1.25))
        self.frontier_unknown_neighbor_weight = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_UNKNOWN_NEIGHBOR_WEIGHT", 0.08))
        self.frontier_cluster_weight = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_CLUSTER_WEIGHT", 0.06))
        self.frontier_backtrack_penalty = max(0.0, env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_BACKTRACK_PENALTY", 5.5))
        self.frontier_disallowed_backtrack_penalty = max(
            0.0,
            env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_DISALLOWED_BACKTRACK_PENALTY", 80.0),
        )
        self.frontier_revisit_path_penalty = max(
            0.0,
            env_float("ROBOT_GLOBAL_PLANNER_FRONTIER_REVISIT_PATH_PENALTY", 0.65),
        )

    def reset(self) -> JsonDict:
        result = {
            "schema_version": GLOBAL_PLAN_SCHEMA_VERSION,
            "status": "reset",
            "planner": "astar",
            "last_updated_at": now_iso(),
        }
        atomic_write_json(self.path, result)
        return result

    def last_plan(self) -> JsonDict:
        if not self.path.exists():
            return {"status": "not_planned", "planner": "astar"}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return {"status": "invalid_plan_state", "planner": "astar"}
        return data if isinstance(data, dict) else {"status": "invalid_plan_state", "planner": "astar"}

    def _position_status(self, supplied: Optional[JsonDict] = None) -> JsonDict:
        return supplied if isinstance(supplied, dict) and supplied.get("cells") is not None else self.position_map.status()

    def _semantic_status(self, supplied: Optional[JsonDict] = None) -> JsonDict:
        return supplied if isinstance(supplied, dict) and supplied.get("cells") is not None else self.semantic_map.status()

    def _bounds(self, position_status: JsonDict, *, start: str, goals: Sequence[str]) -> Tuple[int, int, int, int]:
        known_cells = list((position_status.get("cells") or {}).keys()) + [start] + list(goals)
        coords = [parse_cell(cell) for cell in known_cells]
        min_x = min(x for x, _ in coords) - self.search_padding_cells
        max_x = max(x for x, _ in coords) + self.search_padding_cells
        min_z = min(z for _, z in coords) - self.search_padding_cells
        max_z = max(z for _, z in coords) + self.search_padding_cells
        return min_x, max_x, min_z, max_z

    def _in_bounds(self, cell: str, bounds: Tuple[int, int, int, int]) -> bool:
        x, z = parse_cell(cell)
        min_x, max_x, min_z, max_z = bounds
        return min_x <= x <= max_x and min_z <= z <= max_z

    def _position_cell(self, position_status: JsonDict, cell: str) -> JsonDict:
        rec = (position_status.get("cells") or {}).get(cell)
        return rec if isinstance(rec, dict) else {"cell": cell, "state": CELL_UNKNOWN, "visited": False, "occupancy_confidence": 0.0}

    def _traversal_cost(
        self,
        position_status: JsonDict,
        cell: str,
        *,
        goal_cells: Set[str],
        start: str,
        allow_unknown_intermediate: bool,
    ) -> Optional[float]:
        if cell == start:
            return 0.0
        rec = self._position_cell(position_status, cell)
        state = str(rec.get("state") or CELL_UNKNOWN)
        if state in {CELL_OCCUPIED, CELL_INFLATED} and cell not in goal_cells:
            return None
        if state in {CELL_OCCUPIED, CELL_INFLATED} and cell in goal_cells:
            return None
        if state == CELL_FREE:
            cost = self.free_cost
            if bool(rec.get("visited", False)):
                cost *= self.visited_free_discount
            return cost
        if cell not in goal_cells and not allow_unknown_intermediate:
            return None
        return self.unknown_cost

    def _heuristic(self, cell: str, goals: Sequence[str]) -> float:
        return float(min(manhattan_distance(cell, goal) for goal in goals)) if goals else 0.0

    def _reconstruct(self, came_from: Dict[str, Optional[str]], goal: str) -> List[str]:
        path: List[str] = [goal]
        current = goal
        while came_from.get(current) is not None:
            current = str(came_from[current])
            path.append(current)
        path.reverse()
        return path

    def _astar(
        self,
        *,
        position_status: JsonDict,
        start: str,
        goals: Sequence[str],
        blocked_edges: Optional[Iterable[Any]] = None,
        allow_unknown_intermediate: bool = True,
    ) -> Tuple[Optional[List[str]], Optional[float], int]:
        unique_goals = list(dict.fromkeys(str(goal) for goal in goals if str(goal).strip()))
        if not unique_goals:
            return None, None, 0
        goal_set = set(unique_goals)
        if start in goal_set:
            return [start], 0.0, 0
        bounds = self._bounds(position_status, start=start, goals=unique_goals)
        blocked = _blocked_edge_set(position_status, blocked_edges)
        frontier: List[_QueueNode] = [_QueueNode(self._heuristic(start, unique_goals), start)]
        came_from: Dict[str, Optional[str]] = {start: None}
        cost_so_far: Dict[str, float] = {start: 0.0}
        expanded = 0
        while frontier and expanded < self.max_expansions:
            current = heapq.heappop(frontier).cell
            expanded += 1
            if current in goal_set:
                return self._reconstruct(came_from, current), cost_so_far[current], expanded
            for neighbor in four_neighbors(current):
                if not self._in_bounds(neighbor, bounds) or _edge_is_blocked(blocked, current, neighbor):
                    continue
                step_cost = self._traversal_cost(
                    position_status,
                    neighbor,
                    goal_cells=goal_set,
                    start=start,
                    allow_unknown_intermediate=allow_unknown_intermediate,
                )
                if step_cost is None:
                    continue
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(frontier, _QueueNode(new_cost + self._heuristic(neighbor, unique_goals), neighbor))
        return None, None, expanded

    def _goal_alternatives(self, position_status: JsonDict, target_cell: str) -> List[str]:
        tx, tz = parse_cell(target_cell)
        alternatives: List[Tuple[int, str]] = []
        for radius in range(0, self.goal_substitute_radius + 1):
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if radius > 0 and abs(dx) + abs(dz) != radius:
                        continue
                    cell = format_cell(tx + dx, tz + dz)
                    rec = self._position_cell(position_status, cell)
                    state = str(rec.get("state") or CELL_UNKNOWN)
                    if state in {CELL_OCCUPIED, CELL_INFLATED}:
                        continue
                    alternatives.append((radius, cell))
        alternatives.sort(key=lambda item: (item[0], item[1]))
        return [cell for _, cell in alternatives]

    def _action_for_path(
        self,
        *,
        current_cell: str,
        current_heading: str,
        path: Sequence[str],
        target_heading: Optional[str],
        drive_model: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Convert the first A* grid edge into one discrete actuator command.

        The global planner owns cells, not blind reverse motion.  In the
        default non-holonomic mode, a lateral path edge becomes a turn-in-place
        so the next front-camera RGB-D frame validates that side before moving.
        Holonomic mode preserves the older AI2-THOR lateral translation
        behavior for controlled simulator experiments.
        """
        model = normalize_drive_model(drive_model)
        if len(path) >= 2:
            next_cell = str(path[1])
            desired_heading = heading_between(current_cell, next_cell)
            if desired_heading == current_heading:
                return "MoveAhead", next_cell
            if desired_heading == left_heading(current_heading):
                if model == DRIVE_MODEL_NONHOLONOMIC:
                    return "RotateLeft", current_cell
                return "MoveLeft", next_cell
            if desired_heading == right_heading(current_heading):
                if model == DRIVE_MODEL_NONHOLONOMIC:
                    return "RotateRight", current_cell
                return "MoveRight", next_cell
            if desired_heading is not None:
                return "RotateRight", current_cell
            return "RotateRight", next_cell
        desired_heading = str(target_heading or current_heading)
        if desired_heading == current_heading:
            return "RotateLeft", current_cell
        if desired_heading == left_heading(current_heading):
            return "RotateLeft", current_cell
        if desired_heading == right_heading(current_heading):
            return "RotateRight", current_cell
        return "RotateRight", current_cell

    def _first_step_reverse_penalty(self, *, current_cell: str, current_heading: str, path: Sequence[str]) -> float:
        if len(path) < 2:
            return 0.0
        desired_heading = heading_between(str(current_cell), str(path[1]))
        if desired_heading is None:
            return 0.0
        if desired_heading not in {current_heading, left_heading(current_heading), right_heading(current_heading)}:
            return self.reverse_first_step_penalty
        return 0.0

    def _first_step_lateral_penalty(self, *, current_cell: str, current_heading: str, path: Sequence[str]) -> float:
        if len(path) < 2:
            return 0.0
        desired_heading = heading_between(str(current_cell), str(path[1]))
        if desired_heading in {left_heading(current_heading), right_heading(current_heading)}:
            return self.lateral_first_step_penalty
        return 0.0

    def _first_step_kind(self, *, current_cell: str, current_heading: str, path: Sequence[str]) -> str:
        if len(path) < 2:
            return "scan"
        desired_heading = heading_between(str(current_cell), str(path[1]))
        if desired_heading == current_heading:
            return "forward"
        if desired_heading in {left_heading(current_heading), right_heading(current_heading)}:
            return "lateral"
        if desired_heading is not None:
            return "turn"
        return "unknown"

    def _frontier_cluster_sizes(self, frontiers: Sequence[str]) -> Dict[str, int]:
        frontier_set = {str(cell) for cell in frontiers if str(cell).strip()}
        sizes: Dict[str, int] = {}
        while frontier_set:
            root = frontier_set.pop()
            stack = [root]
            cluster = [root]
            while stack:
                current = stack.pop()
                for neighbor in four_neighbors(current):
                    if neighbor in frontier_set:
                        frontier_set.remove(neighbor)
                        stack.append(neighbor)
                        cluster.append(neighbor)
            size = len(cluster)
            for cell in cluster:
                sizes[cell] = size
        return sizes

    def _frontier_backtrack_objective_penalty(
        self,
        *,
        current_cell: str,
        current_heading: str,
        frontier: str,
        path: Sequence[str],
        allow_backtrack: bool,
    ) -> float:
        """Penalize free-exploration goals that pull the robot behind progress.

        Object/receptacle goals may legitimately require returning to a known
        view cell; frontier exploration should not jump back to an old rear
        boundary while the current local area still has safe side probes.  The
        caller controls that policy with allow_backtrack.
        """
        try:
            cx, cz = parse_cell(str(current_cell))
            fx, fz = parse_cell(str(frontier))
        except Exception:
            return 0.0
        heading_vectors = {
            "north": (0, 1),
            "east": (1, 0),
            "south": (0, -1),
            "west": (-1, 0),
        }
        fdx, fdz = heading_vectors.get(str(current_heading), (0, 1))
        projection = (fx - cx) * fdx + (fz - cz) * fdz
        penalty = 0.0
        if projection < 0:
            if not allow_backtrack:
                penalty += abs(float(projection)) * self.frontier_backtrack_penalty
                penalty += self.frontier_disallowed_backtrack_penalty
            else:
                # Sticky frontier goals may be one cell behind after a lateral
                # viewpoint adjustment.  Penalize real retreats, not that small
                # heading-relative jitter.
                penalty += max(0.0, abs(float(projection)) - 1.0) * self.frontier_backtrack_penalty

        first_step_kind = self._first_step_kind(
            current_cell=str(current_cell),
            current_heading=str(current_heading),
            path=path,
        )
        if first_step_kind == "turn" and not allow_backtrack:
            penalty += self.frontier_disallowed_backtrack_penalty * 0.5
        return penalty

    def _frontier_revisit_objective_penalty(self, position_status: JsonDict, path: Sequence[str]) -> float:
        if not path:
            return 0.0
        penalty = 0.0
        # Skip the current cell. Penalize long routes through already-covered
        # space so nearby information gain beats old rear frontiers.
        for index, cell in enumerate(path[1:], start=1):
            rec = self._position_cell(position_status, str(cell))
            if bool(rec.get("visited", False)):
                penalty += self.frontier_revisit_path_penalty * max(1.0, float(index) * 0.25)
        return penalty

    def _unknown_count(self, position_status: JsonDict, path: Sequence[str]) -> int:
        count = 0
        for cell in path:
            if str(self._position_cell(position_status, str(cell)).get("state") or CELL_UNKNOWN) == CELL_UNKNOWN:
                count += 1
        return count

    def _save_plan(self, plan: GlobalPlan) -> JsonDict:
        payload = {
            "schema_version": GLOBAL_PLAN_SCHEMA_VERSION,
            "last_updated_at": now_iso(),
            **plan.as_dict(),
        }
        atomic_write_json(self.path, payload)
        return payload

    def plan_to_goal(
        self,
        *,
        current_cell: str,
        current_heading: str,
        target_cell: str,
        target_heading: Optional[str] = None,
        target_reason: str = "object_memory_target",
        target_track_id: Optional[str] = None,
        goal_type: Optional[str] = None,
        position_status: Optional[JsonDict] = None,
        semantic_status: Optional[JsonDict] = None,
        blocked_edges: Optional[Iterable[Any]] = None,
        drive_model: Optional[str] = None,
    ) -> JsonDict:
        position = self._position_status(position_status)
        semantic = self._semantic_status(semantic_status)
        alternatives = self._goal_alternatives(position, str(target_cell))
        # Prefer the requested viewpoint exactly. Nearby substitutes are only a
        # recovery mechanism for an occupied / unreachable remembered cell.
        path, cost, expanded = self._astar(position_status=position, start=str(current_cell), goals=[str(target_cell)], blocked_edges=blocked_edges)
        if not path:
            substitute_goals = [cell for cell in alternatives if cell not in {str(target_cell), str(current_cell)}]
            path, cost, extra_expanded = self._astar(position_status=position, start=str(current_cell), goals=substitute_goals, blocked_edges=blocked_edges)
            expanded += extra_expanded
        selected = path[-1] if path else None
        action, next_cell = self._action_for_path(
            current_cell=str(current_cell),
            current_heading=str(current_heading),
            path=path or [],
            target_heading=target_heading,
            drive_model=drive_model,
        ) if path else (None, None)
        semantic_cell = (semantic.get("cells") or {}).get(str(target_cell), {})
        scores = semantic_cell.get("task_scores") if isinstance(semantic_cell, dict) and isinstance(semantic_cell.get("task_scores"), dict) else {}
        semantic_score = max(float(scores.get("pickup_target_score", 0.0) or 0.0), float(scores.get("receptacle_score", 0.0) or 0.0), float(scores.get("surface_score", 0.0) or 0.0))
        plan = GlobalPlan(
            status="success" if path else "no_path",
            target_reason=str(target_reason or "object_memory_target"),
            requested_target_cell=str(target_cell),
            selected_goal_cell=selected,
            selected_goal_kind="requested_goal" if selected == str(target_cell) else "nearby_reachable_goal",
            path=list(path or []),
            route_cost=cost,
            expanded_node_count=expanded,
            unknown_cell_count=self._unknown_count(position, path or []),
            next_cell=next_cell,
            next_action=action,
            target_heading=target_heading,
            semantic_score=semantic_score,
            replan_reason=None if path else "astar_no_path_to_object_viewpoint",
            candidate_goal_count=len(alternatives),
        )
        payload = self._save_plan(plan)
        payload.update({"target_track_id": target_track_id, "goal_type": goal_type})
        payload["drive_model"] = normalize_drive_model(drive_model)
        atomic_write_json(self.path, payload)
        return payload

    def _frontier_semantic_score(self, semantic_status: JsonDict, frontier: str, goal_type: Optional[str]) -> float:
        rec = (semantic_status.get("frontier_scores") or {}).get(frontier)
        if not isinstance(rec, dict):
            return 0.0
        if goal_type in {"pickup", "pickup_target", TASK_PICKUP}:
            return float(rec.get("pickup_target_score", 0.0) or 0.0)
        if goal_type in {"place", "place_receptacle", "surface_target", TASK_RECEPTACLE}:
            return max(float(rec.get("receptacle_score", 0.0) or 0.0), float(rec.get("surface_score", 0.0) or 0.0))
        return max(float(rec.get("exploration_score", 0.0) or 0.0), float(rec.get("pickup_target_score", 0.0) or 0.0), float(rec.get("receptacle_score", 0.0) or 0.0))

    def plan_to_best_frontier(
        self,
        *,
        current_cell: str,
        current_heading: str,
        frontier_cells: Optional[Sequence[str]] = None,
        goal_type: Optional[str] = None,
        target_reason: str = "semantic_frontier",
        position_status: Optional[JsonDict] = None,
        semantic_status: Optional[JsonDict] = None,
        blocked_edges: Optional[Iterable[Any]] = None,
        preferred_frontier: Optional[str] = None,
        frontier_cooldowns: Optional[JsonDict] = None,
        allow_backtrack: bool = True,
        drive_model: Optional[str] = None,
    ) -> JsonDict:
        position = self._position_status(position_status)
        semantic = self._semantic_status(semantic_status)
        frontiers = list(dict.fromkeys(str(cell) for cell in (frontier_cells or position.get("frontier_cells") or position.get("frontiers") or []) if str(cell).strip()))
        preferred = str(preferred_frontier or "").strip()
        cooldown_cells = {
            str(cell)
            for cell, entry in (frontier_cooldowns or {}).items()
            if str(cell).strip() and entry
        }
        ranked = sorted(
            frontiers,
            key=lambda cell: (
                0 if cell == preferred else 1,
                -self._frontier_semantic_score(semantic, cell, goal_type),
                manhattan_distance(str(current_cell), cell),
                cell,
            ),
        )[: self.max_frontier_candidates]
        active_ranked = [cell for cell in ranked if cell not in cooldown_cells]
        if not active_ranked:
            active_ranked = list(ranked)
        cluster_sizes = self._frontier_cluster_sizes(active_ranked)
        best: Optional[Tuple[float, float, str, List[str], int, float, int, str, float, float]] = None
        for frontier in active_ranked:
            path, cost, expanded = self._astar(
                position_status=position,
                start=str(current_cell),
                goals=[frontier],
                blocked_edges=blocked_edges,
                allow_unknown_intermediate=False,
            )
            if not path or cost is None:
                continue
            semantic_score = self._frontier_semantic_score(semantic, frontier, goal_type)
            unknown_neighbors = int(((semantic.get("frontier_scores") or {}).get(frontier) or {}).get("unknown_neighbor_count", 0) or 0)
            cluster_size = int(cluster_sizes.get(frontier, 1) or 1)
            information_gain = float(unknown_neighbors) + min(12.0, float(cluster_size)) * 0.5
            reverse_penalty = self._first_step_reverse_penalty(
                current_cell=str(current_cell),
                current_heading=str(current_heading),
                path=path,
            )
            lateral_penalty = self._first_step_lateral_penalty(
                current_cell=str(current_cell),
                current_heading=str(current_heading),
                path=path,
            )
            first_step_kind = self._first_step_kind(
                current_cell=str(current_cell),
                current_heading=str(current_heading),
                path=path,
            )
            turn_penalty = self.turn_first_step_penalty if first_step_kind == "turn" else 0.0
            forward_bonus = self.forward_first_step_bonus if first_step_kind == "forward" else 0.0
            sticky_bonus = self.frontier_sticky_bonus if frontier == preferred else 0.0
            cooldown_penalty = self.frontier_cooldown_penalty if frontier in cooldown_cells else 0.0
            backtrack_penalty = self._frontier_backtrack_objective_penalty(
                current_cell=str(current_cell),
                current_heading=str(current_heading),
                frontier=frontier,
                path=path,
                allow_backtrack=bool(allow_backtrack),
            )
            revisit_penalty = self._frontier_revisit_objective_penalty(position, path)
            if not allow_backtrack and backtrack_penalty > 0:
                sticky_bonus = 0.0
            objective = (
                float(cost)
                + reverse_penalty
                + lateral_penalty
                + turn_penalty
                + cooldown_penalty
                + backtrack_penalty
                + revisit_penalty
                - sticky_bonus
                - forward_bonus
                - self.frontier_semantic_weight * semantic_score
                - self.frontier_unknown_neighbor_weight * unknown_neighbors
                - self.frontier_cluster_weight * cluster_size
            )
            candidate = (
                objective,
                float(cost),
                frontier,
                path,
                expanded,
                information_gain,
                cluster_size,
                first_step_kind,
                backtrack_penalty,
                revisit_penalty,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            plan = GlobalPlan(
                status="no_path",
                target_reason=str(target_reason),
                selected_goal_kind="frontier",
                expanded_node_count=0,
                replan_reason="astar_no_reachable_frontier",
                candidate_goal_count=len(ranked),
            )
            return self._save_plan(plan)
        (
            objective,
            cost,
            frontier,
            path,
            expanded,
            information_gain,
            cluster_size,
            first_step_kind,
            backtrack_penalty,
            revisit_penalty,
        ) = best
        action, next_cell = self._action_for_path(
            current_cell=str(current_cell),
            current_heading=str(current_heading),
            path=path,
            target_heading=None,
            drive_model=drive_model,
        )
        plan = GlobalPlan(
            status="success",
            target_reason=str(target_reason),
            requested_target_cell=frontier,
            selected_goal_cell=frontier,
            selected_goal_kind="sticky_semantic_frontier" if frontier == preferred else "semantic_frontier",
            path=path,
            route_cost=cost,
            expanded_node_count=expanded,
            unknown_cell_count=self._unknown_count(position, path),
            next_cell=next_cell,
            next_action=action,
            semantic_score=self._frontier_semantic_score(semantic, frontier, goal_type),
            replan_reason="sticky_frontier_target" if frontier == preferred else None,
            candidate_goal_count=len(ranked),
            frontier_cluster_size=cluster_size,
            frontier_information_gain=information_gain,
            frontier_backtrack_penalty=backtrack_penalty,
            frontier_revisit_penalty=revisit_penalty,
            objective_score=objective,
            first_step_kind=first_step_kind,
        )
        payload = self._save_plan(plan)
        payload["preferred_frontier"] = preferred or None
        payload["frontier_cooldown_count"] = len(cooldown_cells)
        payload["frontier_backtrack_allowed"] = bool(allow_backtrack)
        payload["drive_model"] = normalize_drive_model(drive_model)
        atomic_write_json(self.path, payload)
        return payload


def parse_json_arg(value: Optional[str]) -> JsonDict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A* global planner over position-map.json and semantic-map.json.")
    parser.add_argument("command", choices=["reset", "goal", "frontier"])
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--current-cell", default="0,0")
    parser.add_argument("--current-heading", default="north")
    parser.add_argument("--target-cell", default=None)
    parser.add_argument("--target-heading", default=None)
    parser.add_argument("--goal-type", default=None)
    parser.add_argument("--target-reason", default="cli")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    planner = AStarGlobalPlanner(Path(args.memory_dir) if args.memory_dir else None)
    if args.command == "reset":
        result = planner.reset()
    elif args.command == "goal":
        if not args.target_cell:
            raise SystemExit("--target-cell is required for goal")
        result = planner.plan_to_goal(
            current_cell=args.current_cell,
            current_heading=args.current_heading,
            target_cell=args.target_cell,
            target_heading=args.target_heading,
            goal_type=args.goal_type,
            target_reason=args.target_reason,
        )
    else:
        result = planner.plan_to_best_frontier(
            current_cell=args.current_cell,
            current_heading=args.current_heading,
            goal_type=args.goal_type,
            target_reason=args.target_reason,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
