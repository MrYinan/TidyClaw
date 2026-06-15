#!/usr/bin/env python3
"""Inspection waypoint generation for coverage-style room patrol.

This module is the stable target layer between MapBackend and navigation
planning.  It intentionally does not execute movement and does not mutate
memory.  It turns a map snapshot into Roboclaws-style inspection waypoints that
the decision layer can expose as durable coverage targets.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.map_backend.base import JsonDict, MapSnapshot, as_dict, as_list
    from scripts.position_map_core import CELL_FREE, CELL_INFLATED, CELL_OCCUPIED, CELL_UNKNOWN
    from scripts.position_map_core import edge_key, four_neighbors, parse_cell
except ImportError:  # pragma: no cover - direct script execution
    from map_backend.base import JsonDict, MapSnapshot, as_dict, as_list
    from position_map_core import CELL_FREE, CELL_INFLATED, CELL_OCCUPIED, CELL_UNKNOWN
    from position_map_core import edge_key, four_neighbors, parse_cell


INSPECTION_WAYPOINTS_SCHEMA = "robot_cleaner_inspection_waypoints_v1"
WAYPOINT_SOURCE_AUTHORED = "map_snapshot_inspection_waypoint"
WAYPOINT_SOURCE_GENERATED = "map_backend_reachable_coverage"
DEFAULT_COVERAGE_RADIUS_CELLS = 4
DEFAULT_MAX_WAYPOINTS = 32

BLOCKING_CELL_STATES = {CELL_OCCUPIED, CELL_INFLATED}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_cell_safe(value: Any) -> tuple[int, int] | None:
    try:
        return parse_cell(value)
    except Exception:
        return None


def format_cell(x: int, z: int) -> str:
    return f"{int(x)},{int(z)}"


def encode_cell_part(value: int) -> str:
    return f"m{abs(int(value))}" if int(value) < 0 else str(int(value))


def waypoint_id_for_cell(cell: str, *, prefix: str = "wp") -> str:
    parsed = parse_cell_safe(cell)
    if parsed is None:
        safe = str(cell or "unknown").strip().replace(",", "_").replace(" ", "_")
        return f"{prefix}_{safe or 'unknown'}"
    x, z = parsed
    return f"{prefix}_x{encode_cell_part(x)}_z{encode_cell_part(z)}"


def sorted_cells(cells: Iterable[str]) -> list[str]:
    def key(cell: str) -> tuple[int, int, str]:
        parsed = parse_cell_safe(cell)
        if parsed is None:
            return (0, 0, str(cell))
        return (parsed[0], parsed[1], str(cell))

    return sorted({str(cell) for cell in cells if str(cell or "").strip()}, key=key)


def manhattan(left: str, right: str) -> int:
    left_cell = parse_cell_safe(left)
    right_cell = parse_cell_safe(right)
    if left_cell is None or right_cell is None:
        return 10**9
    return abs(left_cell[0] - right_cell[0]) + abs(left_cell[1] - right_cell[1])


@dataclass(frozen=True)
class WaypointGraph:
    cells: dict[str, JsonDict]
    adjacency: dict[str, set[str]]
    free_cells: set[str]

    @property
    def free_cell_count(self) -> int:
        return len(self.free_cells)


def _cell_record_state(record: Mapping[str, Any]) -> str:
    return str(record.get("state") or CELL_FREE)


def _is_traversable_cell(record: Mapping[str, Any]) -> bool:
    state = _cell_record_state(record)
    return state not in BLOCKING_CELL_STATES


def graph_from_snapshot(snapshot: MapSnapshot) -> WaypointGraph:
    """Build a four-connected graph from a MapSnapshot.

    Known-open edges are honored when present.  If the map has no explicit
    edges, free four-neighbor adjacency is used as a compatibility fallback.
    """

    position_status = snapshot.to_position_status()
    raw_cells = as_dict(position_status.get("cells"))
    cells: dict[str, JsonDict] = {}
    for cell, record in raw_cells.items():
        text = str(cell or "").strip()
        if not text:
            continue
        item = dict(as_dict(record))
        item.setdefault("cell", text)
        cells[text] = item

    pose_cell = str(as_dict(snapshot.pose).get("cell") or "").strip()
    if pose_cell and pose_cell not in cells:
        cells[pose_cell] = {"cell": pose_cell, "state": CELL_FREE, "visited": True, "source": "pose"}

    free_cells = {cell for cell, record in cells.items() if _is_traversable_cell(record)}
    adjacency: dict[str, set[str]] = {cell: set() for cell in free_cells}
    edges = as_dict(snapshot.edges)
    explicit_edges = [
        str(edge or "").strip()
        for edge in as_list(edges.get("known_open_edges"))
        if str(edge or "").strip()
    ]
    blocked = {
        str(edge or "").strip()
        for edge in as_list(edges.get("blocked_edges")) + as_list(edges.get("hard_blocked_edges"))
        if str(edge or "").strip()
    }
    if explicit_edges:
        for item in explicit_edges:
            if "->" not in item:
                continue
            source, target = [part.strip() for part in item.split("->", 1)]
            if source not in free_cells or target not in free_cells:
                continue
            if edge_key(source, target) in blocked or edge_key(target, source) in blocked:
                continue
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
    else:
        for cell in free_cells:
            for neighbor in four_neighbors(cell):
                if neighbor in free_cells:
                    adjacency.setdefault(cell, set()).add(neighbor)
                    adjacency.setdefault(neighbor, set()).add(cell)

    return WaypointGraph(cells=cells, adjacency=adjacency, free_cells=free_cells)


def connected_components(graph: WaypointGraph) -> list[list[str]]:
    remaining = set(graph.free_cells)
    components: list[list[str]] = []
    while remaining:
        start = sorted_cells(remaining)[0]
        queue: deque[str] = deque([start])
        remaining.remove(start)
        component: list[str] = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for neighbor in sorted_cells(graph.adjacency.get(cell, set())):
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
        components.append(sorted_cells(component))
    components.sort(key=lambda item: (-len(item), item[0] if item else ""))
    return components


def bfs_distances(graph: WaypointGraph, start: str, allowed: set[str] | None = None) -> dict[str, int]:
    allowed_cells = allowed or graph.free_cells
    if start not in allowed_cells:
        return {}
    distances = {start: 0}
    queue: deque[str] = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in sorted_cells(graph.adjacency.get(cell, set())):
            if neighbor not in allowed_cells or neighbor in distances:
                continue
            distances[neighbor] = distances[cell] + 1
            queue.append(neighbor)
    return distances


def _component_seed(component: Sequence[str]) -> str:
    coords = [(cell, parse_cell_safe(cell)) for cell in component]
    numeric = [(cell, coord) for cell, coord in coords if coord is not None]
    if not numeric:
        return sorted(component)[0]
    centroid_x = sum(coord[0] for _, coord in numeric) / float(len(numeric))
    centroid_z = sum(coord[1] for _, coord in numeric) / float(len(numeric))

    def score(item: tuple[str, tuple[int, int]]) -> tuple[float, int, int, str]:
        cell, (x, z) = item
        return ((x - centroid_x) ** 2 + (z - centroid_z) ** 2, x, z, cell)

    return min(numeric, key=score)[0]


def _farthest_uncovered(
    component: Sequence[str],
    distance_maps: dict[str, dict[str, int]],
    selected: Sequence[str],
    radius_cells: int,
) -> str | None:
    best_cell: str | None = None
    best_score = -1
    for cell in sorted_cells(component):
        min_distance = min(
            (distance_maps.get(seed, {}).get(cell, 10**9) for seed in selected),
            default=10**9,
        )
        if min_distance <= radius_cells:
            continue
        tie = parse_cell_safe(cell) or (0, 0)
        score = int(min_distance)
        if score > best_score:
            best_cell = cell
            best_score = score
        elif score == best_score and best_cell is not None:
            best_tie = parse_cell_safe(best_cell) or (0, 0)
            if (tie[0], tie[1], cell) < (best_tie[0], best_tie[1], best_cell):
                best_cell = cell
    return best_cell


def select_component_waypoint_cells(
    graph: WaypointGraph,
    component: Sequence[str],
    *,
    radius_cells: int,
    max_waypoints: int,
) -> list[str]:
    if not component or max_waypoints <= 0:
        return []
    radius = max(0, int(radius_cells))
    selected = [_component_seed(component)]
    allowed = set(component)
    distance_maps = {selected[0]: bfs_distances(graph, selected[0], allowed)}
    while len(selected) < max_waypoints:
        cell = _farthest_uncovered(component, distance_maps, selected, radius)
        if cell is None:
            break
        selected.append(cell)
        distance_maps[cell] = bfs_distances(graph, cell, allowed)
    return selected


def _world_pose_from_record(record: JsonDict) -> JsonDict:
    world = as_dict(record.get("world_position"))
    if not world:
        positions = as_list(record.get("world_positions"))
        world = as_dict(positions[0]) if positions else {}
    if not world:
        return {}
    return {
        "x": safe_float(world.get("x"), 0.0),
        "y": safe_float(world.get("y"), 0.0),
        "z": safe_float(world.get("z"), 0.0),
    }


def _covered_cell_count(graph: WaypointGraph, cell: str, component: Sequence[str], radius_cells: int) -> int:
    distances = bfs_distances(graph, cell, set(component))
    return sum(1 for value in distances.values() if value <= radius_cells)


def waypoint_from_cell(
    graph: WaypointGraph,
    cell: str,
    *,
    component_index: int,
    component_size: int,
    component_cells: Sequence[str],
    radius_cells: int,
    total_free_cells: int,
    source: str = WAYPOINT_SOURCE_GENERATED,
) -> JsonDict:
    x, z = parse_cell_safe(cell) or (0, 0)
    record = as_dict(graph.cells.get(cell))
    covered = _covered_cell_count(graph, cell, component_cells, radius_cells)
    waypoint: JsonDict = {
        "waypoint_id": waypoint_id_for_cell(cell),
        "cell": cell,
        "x_cell": x,
        "z_cell": z,
        "frame_id": "map",
        "label": f"coverage waypoint {cell}",
        "purpose": "coverage_scan",
        "waypoint_source": source,
        "coverage_radius_cells": int(radius_cells),
        "covered_cell_count": int(covered),
        "coverage_estimate": round(covered / float(max(1, total_free_cells)), 6),
        "component_id": f"component_{component_index}",
        "component_index": int(component_index),
        "component_size": int(component_size),
        "map_visited": bool(record.get("visited") is True),
        "visited": False,
    }
    world = _world_pose_from_record(record)
    if world:
        waypoint["world_position"] = world
        waypoint["x"] = world["x"]
        waypoint["y"] = world["z"]
        waypoint["height_y"] = world["y"]
    return waypoint


def _candidate_authored_waypoints(snapshot: MapSnapshot) -> list[Any]:
    status = snapshot.to_position_status()
    room = snapshot.to_navigation_room_state()
    candidates: list[Any] = []
    for source in (
        room.get("inspection_waypoints"),
        room.get("coverage_waypoints", {}).get("required_waypoints") if isinstance(room.get("coverage_waypoints"), dict) else None,
        status.get("inspection_waypoints"),
        status.get("generated_exploration_candidates"),
    ):
        candidates.extend(as_list(source))
    return candidates


def normalize_authored_waypoint(raw: Mapping[str, Any], graph: WaypointGraph, index: int) -> JsonDict | None:
    waypoint_id = str(raw.get("waypoint_id") or "").strip()
    cell = str(raw.get("cell") or raw.get("target_cell") or "").strip()
    if not cell:
        x_cell = raw.get("x_cell")
        z_cell = raw.get("z_cell")
        if x_cell is not None and z_cell is not None:
            cell = format_cell(safe_int(x_cell), safe_int(z_cell))
    if not cell and waypoint_id.startswith("wp_x"):
        # Keep authored waypoint, but it cannot be routed without a cell.
        cell = ""
    if cell and parse_cell_safe(cell) is None:
        return None
    if cell and not waypoint_id:
        waypoint_id = waypoint_id_for_cell(cell)
    if not waypoint_id:
        waypoint_id = f"wp_authored_{index:03d}"
    parsed = parse_cell_safe(cell) if cell else None
    item = dict(raw)
    item.update(
        {
            "waypoint_id": waypoint_id,
            "cell": cell,
            "x_cell": parsed[0] if parsed else safe_int(raw.get("x_cell"), 0),
            "z_cell": parsed[1] if parsed else safe_int(raw.get("z_cell"), 0),
            "frame_id": str(raw.get("frame_id") or "map"),
            "label": str(raw.get("label") or raw.get("name") or waypoint_id),
            "purpose": str(raw.get("purpose") or "coverage_scan"),
            "waypoint_source": str(raw.get("waypoint_source") or WAYPOINT_SOURCE_AUTHORED),
            "coverage_radius_cells": safe_int(raw.get("coverage_radius_cells"), DEFAULT_COVERAGE_RADIUS_CELLS),
            "coverage_estimate": safe_float(raw.get("coverage_estimate"), 0.0),
            "map_visited": bool(raw.get("map_visited") or raw.get("visited") is True),
            "visited": bool(raw.get("visited") is True),
        }
    )
    if cell and cell in graph.cells and "world_position" not in item:
        world = _world_pose_from_record(as_dict(graph.cells.get(cell)))
        if world:
            item["world_position"] = world
    return item


def authored_waypoints_from_snapshot(snapshot: MapSnapshot, graph: WaypointGraph) -> list[JsonDict]:
    result: list[JsonDict] = []
    seen: set[str] = set()
    for index, raw in enumerate(_candidate_authored_waypoints(snapshot)):
        if not isinstance(raw, Mapping):
            continue
        waypoint = normalize_authored_waypoint(raw, graph, index)
        if waypoint is None:
            continue
        waypoint_id = str(waypoint.get("waypoint_id") or "")
        if not waypoint_id or waypoint_id in seen:
            continue
        seen.add(waypoint_id)
        result.append(waypoint)
    return result


def generated_waypoints_from_snapshot(
    snapshot: MapSnapshot,
    *,
    radius_cells: int | None = None,
    max_waypoints: int | None = None,
) -> list[JsonDict]:
    graph = graph_from_snapshot(snapshot)
    radius = max(1, int(radius_cells if radius_cells is not None else env_int("ROBOT_INSPECTION_WAYPOINT_RADIUS_CELLS", DEFAULT_COVERAGE_RADIUS_CELLS)))
    limit = max(1, int(max_waypoints if max_waypoints is not None else env_int("ROBOT_INSPECTION_WAYPOINT_MAX_COUNT", DEFAULT_MAX_WAYPOINTS)))
    components = connected_components(graph)
    waypoints: list[JsonDict] = []
    for component_index, component in enumerate(components):
        remaining = limit - len(waypoints)
        if remaining <= 0:
            break
        component_limit = max(1, min(remaining, len(component)))
        selected = select_component_waypoint_cells(
            graph,
            component,
            radius_cells=radius,
            max_waypoints=component_limit,
        )
        for cell in selected:
            waypoints.append(
                waypoint_from_cell(
                    graph,
                    cell,
                    component_index=component_index,
                    component_size=len(component),
                    component_cells=component,
                    radius_cells=radius,
                    total_free_cells=max(1, graph.free_cell_count),
                )
            )
            if len(waypoints) >= limit:
                break
    return sorted(waypoints, key=lambda item: str(item.get("waypoint_id") or ""))


def build_inspection_waypoints(
    snapshot: MapSnapshot,
    *,
    radius_cells: int | None = None,
    max_waypoints: int | None = None,
) -> JsonDict:
    graph = graph_from_snapshot(snapshot)
    authored = authored_waypoints_from_snapshot(snapshot, graph)
    if authored:
        source = WAYPOINT_SOURCE_AUTHORED
        waypoints = sorted(authored, key=lambda item: str(item.get("waypoint_id") or ""))
    else:
        source = WAYPOINT_SOURCE_GENERATED
        waypoints = generated_waypoints_from_snapshot(
            snapshot,
            radius_cells=radius_cells,
            max_waypoints=max_waypoints,
        )
    waypoint_ids = [str(item.get("waypoint_id") or "") for item in waypoints if str(item.get("waypoint_id") or "")]
    components = connected_components(graph)
    return {
        "schema": INSPECTION_WAYPOINTS_SCHEMA,
        "generated_at": now_iso(),
        "source": source,
        "map_backend": snapshot.backend,
        "map_snapshot_schema": snapshot.schema,
        "map_frame": dict(as_dict(snapshot.map_frame)),
        "pose": dict(as_dict(snapshot.pose)),
        "free_cell_count": graph.free_cell_count,
        "connected_component_count": len(components),
        "coverage_radius_cells": max(
            1,
            int(radius_cells if radius_cells is not None else env_int("ROBOT_INSPECTION_WAYPOINT_RADIUS_CELLS", DEFAULT_COVERAGE_RADIUS_CELLS)),
        ),
        "required_waypoint_ids": waypoint_ids,
        "required_waypoint_count": len(waypoint_ids),
        "waypoints": waypoints,
    }


def waypoint_by_id(waypoint_set: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {
        str(item.get("waypoint_id") or ""): dict(item)
        for item in as_list(waypoint_set.get("waypoints"))
        if isinstance(item, Mapping) and str(item.get("waypoint_id") or "")
    }

