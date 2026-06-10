#!/usr/bin/env python3
"""Long-term object and object-goal memory for tidy patrol.

This module is deliberately online-safe: it consumes only structured
perception, navigation-memory cells/headings, and action feedback surfaced by
the runner. It keeps a SemExp-like long-horizon goal interface without bringing
in training code, policy networks, or simulator metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
OBJECT_MEMORY_PATH = MEMORY_DIR / "object-memory.json"
OBJECT_GOALS_PATH = MEMORY_DIR / "object-goals.json"
POSITION_MAP_PATH = MEMORY_DIR / "position-map.json"

DEFAULT_CELL_SIZE = 0.25
OBJECT_MEMORY_SCHEMA_VERSION = 1
OBJECT_GOALS_SCHEMA_VERSION = 1

HEADING_ORDER = ["north", "east", "south", "west"]
HEADING_THETA_DEG = {"north": 0, "east": 90, "south": 180, "west": 270}
HEADING_VECTORS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}

PICKUP_LABELS = {
    "apple",
    "banana",
    "book",
    "bottle",
    "bowl",
    "butter_knife",
    "cup",
    "kettle",
    "lettuce",
    "mug",
    "orange",
    "pan",
    "plate",
    "pot",
    "potato",
    "remote",
    "remote_control",
    "soap_bottle",
    "tomato",
    "vase",
}
FOOD_LABELS = {"apple", "banana", "lettuce", "orange", "potato", "tomato"}
REJECTED_PICKUP_STATUSES = {"rejected", "rejected_false_positive"}
LABEL_ALIAS_GROUPS = (
    {"counter", "counter_top", "countertop"},
    {"coffee_table", "coffeetable"},
    {"dining_table", "diningtable"},
    {"side_table", "sidetable"},
    {"remote", "remote_control"},
)
RECEPTACLE_LABELS = {
    "counter",
    "counter_top",
    "countertop",
    "coffee_table",
    "coffeetable",
    "dining_table",
    "diningtable",
    "side_table",
    "sidetable",
    "table",
}
SURFACE_SOURCES = {
    "depth_region_geometry",
    "pointcloud_plane",
    "pointcloud_plane_completion",
    "pointcloud_plane_grid_completion",
    "placement_value_map",
}
OBSERVATION_KEYS = (
    "best_pickup_candidate",
    "best_receptacle_candidate",
    "best_surface_candidate",
)
OBSERVATION_LIST_KEYS = (
    "visual_ready_surface_regions",
    "service_candidates",
    "receptacle_candidates",
    "placement_avoidance_candidates",
    "pointcloud_surface_regions",
    "place_affordance_candidates",
    "surface_regions",
)

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


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return dict(default)
    if not text.strip():
        return dict(default)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(default)
    if not isinstance(data, dict):
        return dict(default)
    return data

def load_position_map_frame(memory_dir: Optional[Path] = None) -> JsonDict:
    base_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
    path = base_dir / "position-map.json"
    data = load_json(path, {})
    if not isinstance(data, dict):
        data = {}
    frame = data.get("map_frame") if isinstance(data.get("map_frame"), dict) else {}
    pose = data.get("pose") if isinstance(data.get("pose"), dict) else {}
    return {
        "cell_size_m": float(frame.get("cell_size_m", DEFAULT_CELL_SIZE) or DEFAULT_CELL_SIZE),
        "coordinate_mode": frame.get("coordinate_mode", "action_odometry_grid"),
        "origin_cell": frame.get("origin_cell", "0,0"),
        "pose_confidence": float(pose.get("pose_confidence", 1.0) or 1.0),
        "position_uncertainty_cells": float(pose.get("position_uncertainty_cells", 0.0) or 0.0),
        "heading_confidence": float(pose.get("heading_confidence", 1.0) or 1.0),
    }


def pose_trust_from_navigation(navigation_status: Optional[JsonDict] = None) -> JsonDict:
    """Describe whether remembered grid cells are exact viewpoints or coarse regions.

    The project intentionally uses online-safe action odometry. Without a visual
    relocalization signal, old cells must gradually lose precision. This helper
    prevents ObjectMemory from presenting stale grid cells as exact coordinates.
    """
    nav = navigation_status if isinstance(navigation_status, dict) else {}
    nested = nav.get("pose_trust") if isinstance(nav.get("pose_trust"), dict) else {}

    def _number(key: str, default: float) -> float:
        value = nav.get(key)
        if value is None:
            value = nested.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    confidence = _number("pose_confidence", 1.0)
    uncertainty = _number("position_uncertainty_cells", 0.0)
    heading_confidence = _number("heading_confidence", 1.0)
    low_conf = env_float("ROBOT_OBJECT_MEMORY_LOW_POSE_CONFIDENCE", 0.45)
    high_uncertainty = env_float("ROBOT_OBJECT_MEMORY_HIGH_UNCERTAINTY_CELLS", 3.0)
    medium_conf = env_float("ROBOT_OBJECT_MEMORY_MEDIUM_POSE_CONFIDENCE", 0.68)
    medium_uncertainty = env_float("ROBOT_OBJECT_MEMORY_MEDIUM_UNCERTAINTY_CELLS", 1.75)
    if confidence < low_conf or uncertainty >= high_uncertainty:
        level = "low"
    elif confidence < medium_conf or uncertainty >= medium_uncertainty:
        level = "medium"
    else:
        level = "high"
    return {
        "level": level,
        "pose_confidence": round(max(0.0, min(1.0, confidence)), 4),
        "position_uncertainty_cells": round(max(0.0, uncertainty), 4),
        "heading_confidence": round(max(0.0, min(1.0, heading_confidence)), 4),
        "precise_viewpoint_allowed": level == "high",
        "region_only": level == "low",
    }

def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        replaced = False
        last_error: Optional[PermissionError] = None
        for attempt in range(20):
            try:
                os.replace(tmp_name, path)
                replaced = True
                break
            except PermissionError as exc:
                last_error = exc
                if attempt >= 19:
                    break
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
        if not replaced:
            try:
                with open(path, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
            except PermissionError:
                if last_error is not None:
                    raise last_error
                raise
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except PermissionError:
                pass


def normalize_label(label: Any) -> str:
    value = str(label or "").strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def compact_label(label: Any) -> str:
    return normalize_label(label).replace("_", "")


def label_family(label: Any, task_class: str = "") -> str:
    token = normalize_label(label)
    if token in FOOD_LABELS:
        return "food"
    if token in PICKUP_LABELS or task_class == "pickup_target":
        return "pickup_target"
    if token in RECEPTACLE_LABELS or task_class in {"place_receptacle", "surface_target"}:
        return "receptacle"
    return "unknown"


def env_label_set(name: str, default: Iterable[str]) -> set[str]:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return {normalize_label(item) for item in default if normalize_label(item)}
    if raw in {"*", "all", "ALL"}:
        return {"*"}
    return {normalize_label(item) for item in raw.split(",") if normalize_label(item)}


def canonical_label_key(label: Any) -> str:
    token = normalize_label(label)
    compact = token.replace("_", "")
    for group in LABEL_ALIAS_GROUPS:
        normalized = {normalize_label(item) for item in group}
        compacted = {item.replace("_", "") for item in normalized}
        if token in normalized or compact in compacted:
            return sorted(normalized)[0]
    return token


def labels_compatible(left: Any, right: Any) -> bool:
    return canonical_label_key(left) == canonical_label_key(right)


def parse_cell(cell: Any) -> Tuple[int, int]:
    left, right = str(cell or "0,0").split(",", 1)
    return int(left), int(right)


def format_cell(x: int, z: int) -> str:
    return f"{int(x)},{int(z)}"


def cell_distance(left: Any, right: Any) -> float:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return math.hypot(float(lx - rx), float(lz - rz))


def manhattan_distance(left: Any, right: Any) -> int:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return abs(lx - rx) + abs(lz - rz)


def heading_to_theta(heading: Any) -> int:
    return int(HEADING_THETA_DEG.get(str(heading or "north"), 0))


def heading_from_delta(dx: int, dz: int, fallback: str = "north") -> str:
    if abs(dx) >= abs(dz) and dx != 0:
        return "east" if dx > 0 else "west"
    if dz != 0:
        return "north" if dz > 0 else "south"
    return fallback if fallback in HEADING_VECTORS else "north"


def candidate_signature(candidate: JsonDict) -> str:
    if not isinstance(candidate, dict):
        return "candidate:unknown"
    stable = (
        candidate.get("surface_candidate_id")
        or candidate.get("id")
        or candidate.get("objectId")
    )
    if stable:
        return str(stable)
    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    label = normalize_label(candidate.get("raw_label") or candidate.get("label") or candidate.get("parent_object"))
    try:
        x = int(round(float(bbox.get("x", 0) or 0) / 24.0))
        y = int(round(float(bbox.get("y", 0) or 0) / 24.0))
        w = int(round(float(bbox.get("w", 0) or 0) / 12.0))
        h = int(round(float(bbox.get("h", 0) or 0) / 12.0))
    except (TypeError, ValueError):
        x = y = w = h = 0
    return f"{label}:{x}:{y}:{w}:{h}"


def candidate_geometry_key(candidate: JsonDict, label: str) -> str:
    bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
    try:
        x = int(round(float(bbox.get("x", 0) or 0) / 24.0))
        y = int(round(float(bbox.get("y", 0) or 0) / 24.0))
        w = int(round(float(bbox.get("w", 0) or 0) / 12.0))
        h = int(round(float(bbox.get("h", 0) or 0) / 12.0))
    except (TypeError, ValueError):
        x = y = w = h = 0
    return f"{canonical_label_key(label)}:{x}:{y}:{w}:{h}"


def candidate_position_hint(candidate: JsonDict, bearing_deg: Optional[float]) -> str:
    explicit = str(candidate.get("position_hint") or "").strip()
    if explicit:
        return explicit
    cx_ratio = nested_number(candidate, "geometry", "cx_ratio")
    if cx_ratio is None:
        cx_ratio = nested_number(candidate, "center", "x")
        image_w = nested_number(candidate, "image_size", "w")
        if cx_ratio is not None and image_w:
            cx_ratio = float(cx_ratio) / float(image_w)
    if cx_ratio is not None:
        if cx_ratio < 0.4:
            return "front-left"
        if cx_ratio > 0.6:
            return "front-right"
        return "front-center"
    if bearing_deg is not None:
        if bearing_deg < -12.0:
            return "front-left"
        if bearing_deg > 12.0:
            return "front-right"
        return "front-center"
    return ""


def depth_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def nested_number(candidate: JsonDict, *keys: str, default: Optional[float] = None) -> Optional[float]:
    node: Any = candidate
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return depth_number(node, default)


def candidate_distance_m(candidate: JsonDict) -> Optional[float]:
    for key in ("ground_distance_m", "ground_distance", "distance_m", "distance"):
        value = depth_number(candidate.get(key))
        if value is not None and value > 0:
            return value
    for root in ("geometry", "center_3d", "depth"):
        node = candidate.get(root) if isinstance(candidate.get(root), dict) else {}
        for key in ("ground_distance_m", "distance_m", "z", "median_m"):
            value = depth_number(node.get(key))
            if value is not None and value > 0:
                return value
    return None


def candidate_bearing_deg(candidate: JsonDict) -> Optional[float]:
    for key in ("bearing_deg", "angle_deg"):
        value = depth_number(candidate.get(key))
        if value is not None:
            return value
    geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}
    value = depth_number(geometry.get("bearing_deg"))
    if value is not None:
        return value
    center_3d = candidate.get("center_3d") if isinstance(candidate.get("center_3d"), dict) else {}
    x = depth_number(center_3d.get("x"))
    z = depth_number(center_3d.get("ground_forward_m"), depth_number(center_3d.get("z")))
    if x is not None and z is not None and z > 1e-6:
        return math.degrees(math.atan2(float(x), float(z)))
    return None


def position_hint_prior(position_hint: Any) -> Tuple[float, float]:
    hint = str(position_hint or "").strip().lower()
    if hint == "front-left":
        return -0.45, 0.50
    if hint == "front-right":
        return 0.45, 0.50
    if hint == "front-center":
        return 0.0, 0.50
    if hint == "left":
        return -0.50, 0.25
    if hint == "right":
        return 0.50, 0.25
    return 0.0, 0.50


def _egocentric_to_global(relative_x: float, relative_z: float, heading: str) -> Tuple[float, float]:
    if heading == "east":
        return relative_z, -relative_x
    if heading == "south":
        return -relative_x, -relative_z
    if heading == "west":
        return -relative_z, relative_x
    return relative_x, relative_z


def project_observation_to_cell(
    current_cell: Any,
    heading: Any,
    bearing_deg: Optional[float] = None,
    distance_m: Optional[float] = None,
    cell_size_m: float = DEFAULT_CELL_SIZE,
    position_hint: Optional[str] = None,
) -> JsonDict:
    """Project an egocentric detection into the navigation-memory grid."""
    cell_size = max(0.05, float(cell_size_m or DEFAULT_CELL_SIZE))
    cx, cz = parse_cell(current_cell)
    heading_name = str(heading or "north")

    reliable_distance = distance_m is not None and math.isfinite(float(distance_m)) and float(distance_m) > 0.05
    if reliable_distance:
        bearing = math.radians(float(bearing_deg or 0.0))
        relative_x = math.sin(bearing) * float(distance_m)
        relative_z = math.cos(bearing) * float(distance_m)
        method = "bearing_distance_projection"
        uncertainty = 1
    else:
        relative_x, relative_z = position_hint_prior(position_hint)
        method = "position_hint_prior"
        uncertainty = 2

    global_x_m, global_z_m = _egocentric_to_global(relative_x, relative_z, heading_name)
    ox = cx + int(round(global_x_m / cell_size))
    oz = cz + int(round(global_z_m / cell_size))
    estimated = format_cell(ox, oz)
    candidates = [
        {"cell": estimated, "prob": 0.50 if reliable_distance else 0.38},
    ]
    candidate_offsets = [(0, 1), (1, 0), (-1, 0), (0, -1)]
    probs = [0.20, 0.15, 0.10, 0.05] if reliable_distance else [0.22, 0.18, 0.12, 0.10]
    for (dx, dz), prob in zip(candidate_offsets, probs):
        candidates.append({"cell": format_cell(ox + dx, oz + dz), "prob": round(prob, 2)})
    return {
        "estimated_object_cell": estimated,
        "candidate_cells": candidates,
        "uncertainty_cells": uncertainty,
        "method": method,
        "relative_x_m": round(float(relative_x), 4),
        "relative_z_m": round(float(relative_z), 4),
        "global_offset_cells": [int(ox - cx), int(oz - cz)],
    }


def _edge_blocked(blocked_edges: Iterable[Any], source: str, target: str) -> bool:
    direct = f"{source}->{target}"
    reverse = f"{target}->{source}"
    blocked = {str(item) for item in blocked_edges or []}
    return direct in blocked or reverse in blocked


def compute_recommended_view_cell(
    estimated_object_cell: Any,
    *,
    task_class: str,
    current_cell: Any = "0,0",
    current_heading: Any = "north",
    navigation_status: Optional[JsonDict] = None,
    cell_size_m: float = DEFAULT_CELL_SIZE,
    last_surface_id: Optional[str] = None,
) -> JsonDict:
    """Choose a standoff viewpoint instead of navigating into the object cell."""
    nav = navigation_status if isinstance(navigation_status, dict) else {}
    visited = {str(item) for item in nav.get("visited_cells", []) or []}
    frontier = {str(item) for item in nav.get("frontier_cells", []) or []}
    blocked_edges = nav.get("blocked_edges", []) or []
    current = str(current_cell or nav.get("last_cell") or "0,0")
    heading = str(current_heading or nav.get("last_heading") or "north")
    cell_size = max(0.05, float(cell_size_m or DEFAULT_CELL_SIZE))
    ox, oz = parse_cell(estimated_object_cell)

    if task_class in {"place_receptacle", "surface_target"}:
        radius_range = range(2, 5)
        ideal_m = 1.0
        reason = "receptacle_target_viewpoint"
    else:
        radius_range = range(1, 4)
        ideal_m = 0.70
        reason = "pickup_target_viewpoint"

    best: Optional[Tuple[float, str, str, float]] = None
    for radius in radius_range:
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                if max(abs(dx), abs(dz)) != radius:
                    continue
                candidate = format_cell(ox + dx, oz + dz)
                if candidate == str(estimated_object_cell):
                    continue
                if _edge_blocked(blocked_edges, candidate, str(estimated_object_cell)):
                    continue
                standoff_m = cell_distance(candidate, estimated_object_cell) * cell_size
                desired_heading = heading_from_delta(ox - (ox + dx), oz - (oz + dz), fallback=heading)
                score = 0.0
                score -= abs(standoff_m - ideal_m) * 2.0
                score -= manhattan_distance(current, candidate) * 0.08
                if candidate in visited:
                    score += 0.65
                if candidate in frontier:
                    score += 0.25
                if candidate == current:
                    score += 0.15
                    if desired_heading == heading:
                        score += 0.30
                if last_surface_id:
                    score += 0.05
                if best is None or score > best[0]:
                    best = (score, candidate, desired_heading, standoff_m)

    if best is None:
        current_x, current_z = parse_cell(current)
        desired_heading = heading_from_delta(ox - current_x, oz - current_z, fallback=heading)
        return {
            "recommended_view_cell": current,
            "recommended_heading": desired_heading,
            "standoff_distance_m": round(cell_distance(current, estimated_object_cell) * cell_size, 4),
            "reason": f"{reason}_fallback_current_cell",
        }
    _, view_cell, view_heading, standoff = best
    return {
        "recommended_view_cell": view_cell,
        "recommended_heading": view_heading,
        "standoff_distance_m": round(float(standoff), 4),
        "reason": reason,
    }


@dataclass
class ObjectObservation:
    label: str
    raw_label: str
    label_family: str
    task_class: str
    confidence: float
    source: str
    observed_from: JsonDict
    last_observation: JsonDict
    estimated_location: JsonDict
    viewpoint: JsonDict
    candidate_signature: str
    candidate: Optional[JsonDict] = field(default=None, repr=False, compare=False)

    def to_record(self) -> JsonDict:
        return {
            "label": self.label,
            "raw_label": self.raw_label,
            "label_family": self.label_family,
            "task_class": self.task_class,
            "confidence": round(float(self.confidence), 4),
            "source": self.source,
            "observed_from": dict(self.observed_from),
            "last_observation": dict(self.last_observation),
            "estimated_location": dict(self.estimated_location),
            "viewpoint": dict(self.viewpoint),
            "candidate_signature": self.candidate_signature,
        }


@dataclass
class ObjectTrack:
    track_id: str
    label: str
    raw_labels: List[str]
    label_family: str
    task_class: str
    status: str
    confidence: float
    track_score: float
    staleness: int
    first_seen_step: int
    last_seen_step: int
    seen_count: int
    observed_from: JsonDict
    last_observation: JsonDict
    estimated_location: JsonDict
    viewpoint: JsonDict
    interaction: JsonDict


def default_object_memory() -> JsonDict:
    return {
        "schema_version": OBJECT_MEMORY_SCHEMA_VERSION,
        "map_frame": {
            "cell_size_m": DEFAULT_CELL_SIZE,
            "coordinate_mode": "action_odometry_grid",
            "origin_cell": "0,0",
        },
        "tracks": {},
        "recent_observations": [],
        "selected_targets": {
            "pickup": None,
            "receptacle": None,
        },
        "stats": {
            "track_count": 0,
            "active_pickup_count": 0,
            "active_receptacle_count": 0,
            "created_count": 0,
            "updated_count": 0,
            "merged_count": 0,
            "stale_count": 0,
            "last_update_step": None,
            "last_update": None,
        },
    }


def default_object_goals() -> JsonDict:
    return {
        "schema_version": OBJECT_GOALS_SCHEMA_VERSION,
        "map_frame": {
            "cell_size_m": DEFAULT_CELL_SIZE,
            "coordinate_mode": "action_odometry_grid",
            "origin_cell": "0,0",
        },
        "active_goal": None,
        "history": [],
    }


def normalize_memory(data: JsonDict) -> JsonDict:
    memory = default_object_memory()
    if isinstance(data, dict):
        memory.update(data)
    if not isinstance(memory.get("map_frame"), dict):
        memory["map_frame"] = default_object_memory()["map_frame"]
    memory["map_frame"].setdefault("cell_size_m", DEFAULT_CELL_SIZE)
    memory["map_frame"].setdefault("coordinate_mode", "action_odometry_grid")
    memory["map_frame"].setdefault("origin_cell", "0,0")
    if not isinstance(memory.get("tracks"), dict):
        memory["tracks"] = {}
    else:
        memory["tracks"] = {
            str(track_id): normalize_track_record(track)
            for track_id, track in memory["tracks"].items()
            if isinstance(track, dict)
        }
    if not isinstance(memory.get("recent_observations"), list):
        memory["recent_observations"] = []
    if not isinstance(memory.get("selected_targets"), dict):
        memory["selected_targets"] = {"pickup": None, "receptacle": None}
    if not isinstance(memory.get("stats"), dict):
        memory["stats"] = {}
    memory["schema_version"] = OBJECT_MEMORY_SCHEMA_VERSION
    return memory


def normalize_goals(data: JsonDict) -> JsonDict:
    goals = default_object_goals()
    if isinstance(data, dict):
        goals.update(data)
    if not isinstance(goals.get("map_frame"), dict):
        goals["map_frame"] = default_object_goals()["map_frame"]
    goals["map_frame"].setdefault("cell_size_m", DEFAULT_CELL_SIZE)
    goals["map_frame"].setdefault("coordinate_mode", "action_odometry_grid")
    goals["map_frame"].setdefault("origin_cell", "0,0")
    if not isinstance(goals.get("history"), list):
        goals["history"] = []
    goals["schema_version"] = OBJECT_GOALS_SCHEMA_VERSION
    return goals


def normalize_track_record(track: JsonDict) -> JsonDict:
    if not isinstance(track, dict):
        return track
    label = normalize_label(track.get("label"))
    if label:
        track["label"] = label
    raw_labels = [str(item) for item in (track.get("raw_labels") or []) if str(item or "").strip()]
    compatible_raw_labels = [
        item for item in raw_labels
        if not label or labels_compatible(item, label)
    ]
    track["raw_labels"] = compatible_raw_labels or ([label] if label else [])
    last_observation = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
    if last_observation and not str(last_observation.get("position_hint") or "").strip():
        last_observation["position_hint"] = candidate_position_hint(
            {
                "bbox": last_observation.get("bbox"),
                "center": last_observation.get("center"),
                "geometry": last_observation.get("geometry"),
                "image_size": last_observation.get("image_size"),
            },
            depth_number(last_observation.get("bearing_deg")),
        )
        track["last_observation"] = last_observation

    interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
    interaction.setdefault("pickup_ready", False)
    interaction.setdefault("place_ready", False)
    interaction.setdefault("last_surface_id", None)
    interaction.setdefault("last_failed_step", None)
    interaction.setdefault("failure_count", 0)
    interaction.setdefault("cooldown_until_step", None)

    # Long-horizon object-goal navigation and immediate interaction readiness are
    # separate contracts. Keep additive defaults so old object-memory.json files
    # remain readable without a migration script.
    interaction.setdefault("pickup_goal_eligible", False)
    interaction.setdefault("pickup_action_ready", False)
    interaction.setdefault("pickup_goal_rejection_reasons", [])
    interaction.setdefault("pickup_action_rejection_reasons", [])
    interaction.setdefault("pickup_goal_note", None)
    interaction.setdefault("pickup_action_note", None)
    interaction.setdefault("pickup_goal_contract", "pickup_goal_action_split_v1")
    track["interaction"] = interaction
    return track


def load_memory(path: Optional[Path] = None) -> JsonDict:
    return normalize_memory(load_json(path or OBJECT_MEMORY_PATH, default_object_memory()))


def save_memory(memory: JsonDict, path: Optional[Path] = None) -> None:
    atomic_write_json(path or OBJECT_MEMORY_PATH, normalize_memory(memory))


def load_goals(path: Optional[Path] = None) -> JsonDict:
    return normalize_goals(load_json(path or OBJECT_GOALS_PATH, default_object_goals()))


def save_goals(goals: JsonDict, path: Optional[Path] = None) -> None:
    atomic_write_json(path or OBJECT_GOALS_PATH, normalize_goals(goals))


class ObjectMemory:
    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.memory_path = self.memory_dir / "object-memory.json"
        self.goals_path = self.memory_dir / "object-goals.json"
        self.position_map_frame = load_position_map_frame(self.memory_dir)

    def _position_frame_from_navigation(self, navigation_status: Optional[JsonDict] = None) -> JsonDict:
        """Read the latest position-map frame and override it with live navigation status.

        ObjectMemory may live across multiple runner steps, while position-map.json
        is updated every step. So do not rely only on self.position_map_frame
        initialized in __init__.
        """
        self.position_map_frame = load_position_map_frame(self.memory_dir)

        nav = navigation_status if isinstance(navigation_status, dict) else {}
        frame = dict(self.position_map_frame)

        def _safe_float(value: Any, default: float) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return float(default)
            if not math.isfinite(number):
                return float(default)
            return number

        frame["cell_size_m"] = _safe_float(
            nav.get("cell_size")
            or nav.get("cell_size_m")
            or frame.get("cell_size_m")
            or DEFAULT_CELL_SIZE,
            DEFAULT_CELL_SIZE,
        )
        frame["coordinate_mode"] = (
            nav.get("coordinate_mode")
            or frame.get("coordinate_mode")
            or "action_odometry_grid"
        )
        frame["origin_cell"] = (
            nav.get("origin_cell")
            or frame.get("origin_cell")
            or "0,0"
        )
        frame["pose_confidence"] = _safe_float(
            nav.get("pose_confidence")
            or frame.get("pose_confidence")
            or 1.0,
            1.0,
        )
        frame["position_uncertainty_cells"] = _safe_float(
            nav.get("position_uncertainty_cells")
            or frame.get("position_uncertainty_cells")
            or 0.0,
            0.0,
        )
        frame["heading_confidence"] = _safe_float(
            nav.get("heading_confidence")
            or frame.get("heading_confidence")
            or 1.0,
            1.0,
        )
        return frame

    def load_memory(self) -> JsonDict:
        return load_memory(self.memory_path)

    def save_memory(self, memory: JsonDict) -> None:
        save_memory(memory, self.memory_path)

    def load_goals(self) -> JsonDict:
        return load_goals(self.goals_path)

    def save_goals(self, goals: JsonDict) -> None:
        save_goals(goals, self.goals_path)

    def ensure_files(self) -> None:
        self.save_memory(self.load_memory())
        self.save_goals(self.load_goals())

    def reset(self) -> JsonDict:
        memory = default_object_memory()
        goals = default_object_goals()
        memory["stats"]["last_update"] = now_iso()
        goals["last_update"] = now_iso()
        self.save_memory(memory)
        self.save_goals(goals)
        return self.status()

    def status(self) -> JsonDict:
        memory = self.load_memory()
        goals = self.load_goals()
        return {
            "status": "success",
            "result_type": "object_memory_status",
            "track_count": len(memory.get("tracks", {}) or {}),
            "selected_targets": memory.get("selected_targets", {}),
            "active_goal": goals.get("active_goal"),
            "stats": memory.get("stats", {}),
        }

    def _next_track_id(self, memory: JsonDict, label: str) -> str:
        token = normalize_label(label) or "object"
        prefix = f"objtrk:{token}:"
        max_id = 0
        for track_id in (memory.get("tracks") or {}).keys():
            if not str(track_id).startswith(prefix):
                continue
            try:
                max_id = max(max_id, int(str(track_id).rsplit(":", 1)[-1]))
            except ValueError:
                continue
        return f"{prefix}{max_id + 1:04d}"

    def _extract_observations(
        self,
        analysis: JsonDict,
        *,
        current_cell: str,
        heading: str,
        step: int,
        navigation_status: JsonDict,
    ) -> List[ObjectObservation]:
        position_frame = self._position_frame_from_navigation(navigation_status)
        cell_size = float(position_frame.get("cell_size_m") or DEFAULT_CELL_SIZE)
        pose_confidence = float(position_frame.get("pose_confidence") or 1.0)
        position_uncertainty_cells = float(position_frame.get("position_uncertainty_cells") or 0.0)
        heading_confidence = float(position_frame.get("heading_confidence") or 1.0)

        raw_candidates: List[Tuple[str, JsonDict]] = []
        for key in OBSERVATION_KEYS:
            value = analysis.get(key)
            if isinstance(value, dict):
                raw_candidates.append((key, value))
        for key in OBSERVATION_LIST_KEYS:
            values = analysis.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, dict):
                    raw_candidates.append((key, item))

        observations: List[ObjectObservation] = []
        seen_signatures = set()
        seen_geometry_keys = set()
        for source, candidate in raw_candidates:
            task_class = self._candidate_task_class(candidate)
            if task_class not in {"pickup_target", "place_receptacle", "surface_target"}:
                continue
            label = self._candidate_label(candidate, task_class)
            if not label:
                continue
            geometry_key = candidate_geometry_key(candidate, label)
            if geometry_key in seen_geometry_keys:
                continue
            seen_geometry_keys.add(geometry_key)
            signature = candidate_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            confidence = depth_number(candidate.get("confidence"), 0.0) or 0.0
            bearing = candidate_bearing_deg(candidate)
            distance = candidate_distance_m(candidate)
            position_hint = candidate_position_hint(candidate, bearing)
            estimated = project_observation_to_cell(
                current_cell,
                heading,
                bearing_deg=bearing,
                distance_m=distance,
                cell_size_m=cell_size,
                position_hint=position_hint,
            )

            base_uncertainty = int(estimated.get("uncertainty_cells", 1) or 1)
            if pose_confidence < 0.6:
                base_uncertainty += 1
            if position_uncertainty_cells >= 1:
                base_uncertainty += int(round(position_uncertainty_cells))

            estimated["uncertainty_cells"] = max(1, min(5, base_uncertainty))
            estimated["pose_confidence"] = round(float(pose_confidence), 4)
            estimated["position_uncertainty_cells"] = round(float(position_uncertainty_cells), 4)
            estimated["heading_confidence"] = round(float(heading_confidence), 4)
            estimated["map_source"] = "position_map"

            viewpoint = compute_recommended_view_cell(
                estimated.get("estimated_object_cell"),
                task_class=task_class,
                current_cell=current_cell,
                current_heading=heading,
                navigation_status=navigation_status,
                cell_size_m=cell_size,
                last_surface_id=str(candidate.get("surface_candidate_id") or candidate.get("id") or "") or None,
            )
            bbox = candidate.get("bbox") if isinstance(candidate.get("bbox"), dict) else {}
            center_3d = candidate.get("center_3d") if isinstance(candidate.get("center_3d"), dict) else {}
            observed_from = {
                "cell": str(current_cell),
                "heading": str(heading),
                "theta_deg": heading_to_theta(heading),
                "pose_confidence": round(float(pose_confidence), 4),
                "position_uncertainty_cells": round(float(position_uncertainty_cells), 4),
                "heading_confidence": round(float(heading_confidence), 4),
                "map_source": "position_map",
            }
            geometry = candidate.get("geometry") if isinstance(candidate.get("geometry"), dict) else {}

            last_observation = {
                "bearing_deg": round(float(bearing), 4) if bearing is not None else None,
                "distance_m": round(float(distance), 4) if distance is not None else None,
                "ground_distance_m": round(float(distance), 4) if distance is not None else None,
                "position_hint": position_hint,
                "bbox": dict(bbox),
                "center_3d": dict(center_3d),
                "source": source,
                "candidate_signature": candidate_signature(candidate),
                "surface_candidate_id": candidate.get("surface_candidate_id") or candidate.get("id"),

                # 新增：给 object-memory 后续选择 pickup goal 用
                "surface_hint": candidate.get("surface_hint"),
                "is_floor_level": bool(candidate.get("is_floor_level")),
                "pickup_now": bool(candidate.get("pickup_now")),
                "support_context_blocked": bool(candidate.get("support_context_blocked")),
                "bottom_y_ratio": geometry.get("bottom_y_ratio", candidate.get("bottom_y_ratio")),
                "height_m": geometry.get("height_m", candidate.get("height_m")),
                "task_semantic_class": candidate.get("task_semantic_class"),
            }
            family = label_family(label, task_class)
            observations.append(
                ObjectObservation(
                    label=normalize_label(label),
                    raw_label=str(candidate.get("raw_label") or candidate.get("parent_object") or label),
                    label_family=family,
                    task_class=task_class,
                    confidence=float(confidence),
                    source=source,
                    observed_from=observed_from,
                    last_observation=last_observation,
                    estimated_location=estimated,
                    viewpoint=viewpoint,
                    candidate_signature=candidate_signature(candidate),
                    candidate=candidate,
                )
            )
        return observations

    def _candidate_task_class(self, candidate: JsonDict) -> str:
        task_class = str(candidate.get("task_semantic_class") or "")
        source = str(candidate.get("surface_candidate_source") or candidate.get("source") or "")
        if source in SURFACE_SOURCES:
            return "surface_target"
        if task_class in {"pickup_target", "place_receptacle", "surface_target"}:
            return task_class
        if task_class and task_class not in {"unknown", "candidate"}:
            return task_class
        label = normalize_label(candidate.get("label") or candidate.get("raw_label") or candidate.get("parent_object"))
        if label in PICKUP_LABELS:
            return "pickup_target"
        if label in RECEPTACLE_LABELS:
            return "place_receptacle"
        return task_class

    def _candidate_label(self, candidate: JsonDict, task_class: str) -> str:
        if task_class == "surface_target":
            return normalize_label(
                candidate.get("parent_object")
                or candidate.get("parent_label")
                or candidate.get("raw_label")
                or candidate.get("label")
            )
        return normalize_label(candidate.get("raw_label") or candidate.get("label") or candidate.get("parent_object"))

    def _association_score(self, track: JsonDict, observation: ObjectObservation, step: int) -> float:
        track_label = normalize_label(track.get("label"))
        track_task_class = str(track.get("task_class") or "")
        observation_task_class = str(observation.task_class or "")
        task_compatible = (
            track_task_class == observation_task_class
            or {track_task_class, observation_task_class} <= {"place_receptacle", "surface_target"}
        )
        if not task_compatible:
            return -1.0
        same_label = labels_compatible(track_label, observation.label)
        if not same_label:
            return -1.0
        try:
            last_seen = int(track.get("last_seen_step", 0) or 0)
        except (TypeError, ValueError):
            last_seen = 0
        stale_threshold = env_int("ROBOT_OBJECT_MEMORY_STALE_STEPS", 35)
        if step - last_seen > stale_threshold:
            return -1.0
        track_cell = (
            (track.get("estimated_location") or {}).get("estimated_object_cell")
            if isinstance(track.get("estimated_location"), dict)
            else None
        )
        obs_cell = observation.estimated_location.get("estimated_object_cell")
        if not track_cell or not obs_cell:
            return -1.0
        distance_cells = cell_distance(track_cell, obs_cell)
        if distance_cells > 2.0:
            return -1.0
        track_bearing = depth_number((track.get("last_observation") or {}).get("bearing_deg"))
        obs_bearing = depth_number(observation.last_observation.get("bearing_deg"))
        bearing_delta = 0.0
        if track_bearing is not None and obs_bearing is not None:
            bearing_delta = abs(float(track_bearing) - float(obs_bearing))
            if bearing_delta > 180.0:
                bearing_delta = 360.0 - bearing_delta
        track_hint = str((track.get("last_observation") or {}).get("position_hint") or "")
        observation_hint = str(observation.last_observation.get("position_hint") or "")
        position_match = 0.2 if (track_hint and track_hint == observation_hint) else 0.0
        return (
            1.4
            + max(0.0, 1.0 - distance_cells / 2.0)
            + max(0.0, 0.4 - bearing_delta / 90.0)
            + position_match
        )

    def _find_track(self, memory: JsonDict, observation: ObjectObservation, step: int) -> Optional[str]:
        best: Optional[Tuple[float, str]] = None
        for track_id, track in (memory.get("tracks") or {}).items():
            if not isinstance(track, dict):
                continue
            status = str(track.get("status") or "")
            if status in {"picked", "unreachable"}:
                continue
            score = self._association_score(track, observation, step)
            if score < 0:
                continue
            if status == "placed" and score < env_float("ROBOT_OBJECT_MEMORY_PLACED_ASSOC_MIN_SCORE", 2.0):
                continue
            if best is None or score > best[0]:
                best = (score, str(track_id))
        return best[1] if best else None

    def _new_track(self, memory: JsonDict, observation: ObjectObservation, step: int) -> JsonDict:
        track_id = self._next_track_id(memory, observation.label)
        candidate = observation.candidate if isinstance(observation.candidate, dict) else {}
        if observation.task_class == "pickup_target":
            status = "unpicked"
        elif bool(candidate.get("blocked")):
            status = "blocked"
        elif bool(candidate.get("visual_place_ready")):
            status = "surface_ready"
        else:
            status = "seen"
        interaction = {
            "pickup_ready": bool(candidate.get("pickup_now")),
            "place_ready": bool(candidate.get("visual_place_ready")),
            "last_surface_id": observation.last_observation.get("surface_candidate_id"),
            "last_failed_step": None,
            "failure_count": 0,
            "cooldown_until_step": None,
            "pickup_goal_eligible": False,
            "pickup_action_ready": False,
            "pickup_goal_rejection_reasons": [],
            "pickup_action_rejection_reasons": [],
            "pickup_goal_note": None,
            "pickup_action_note": None,
            "pickup_goal_contract": "pickup_goal_action_split_v1",
        }
        return {
            "track_id": track_id,
            "label": observation.label,
            "raw_labels": [observation.raw_label],
            "label_family": observation.label_family,
            "task_class": observation.task_class,
            "status": status,
            "confidence": round(float(observation.confidence), 4),
            "track_score": round(float(observation.confidence), 4),
            "staleness": 0,
            "first_seen_step": int(step),
            "last_seen_step": int(step),
            "seen_count": 1,
            "observed_from": dict(observation.observed_from),
            "last_observation": dict(observation.last_observation),
            "estimated_location": dict(observation.estimated_location),
            "viewpoint": dict(observation.viewpoint),
            "interaction": interaction,
        }

    def _merge_track(self, track: JsonDict, observation: ObjectObservation, step: int) -> None:
        seen_count = max(1, int(track.get("seen_count", 1) or 1))
        old_conf = depth_number(track.get("confidence"), 0.0) or 0.0
        fused_conf = min(1.0, old_conf * 0.65 + float(observation.confidence) * 0.35 + 0.02)
        raw_labels = list(track.get("raw_labels") or [])
        if observation.raw_label not in raw_labels:
            raw_labels.append(observation.raw_label)
        previous_cell = (track.get("estimated_location") or {}).get("estimated_object_cell")
        observed_cell = observation.estimated_location.get("estimated_object_cell")
        if previous_cell and observed_cell:
            px, pz = parse_cell(previous_cell)
            ox, oz = parse_cell(observed_cell)
            blended_cell = format_cell(
                int(round((px * seen_count + ox) / float(seen_count + 1))),
                int(round((pz * seen_count + oz) / float(seen_count + 1))),
            )
            observation.estimated_location["estimated_object_cell"] = blended_cell
            observation.estimated_location["candidate_cells"] = [
                {"cell": blended_cell, "prob": 0.55},
                *[
                    item for item in observation.estimated_location.get("candidate_cells", [])
                    if isinstance(item, dict) and item.get("cell") != blended_cell
                ][:4],
            ]
        track["raw_labels"] = raw_labels[-8:]
        track["confidence"] = round(float(fused_conf), 4)
        track["track_score"] = round(float(min(1.0, fused_conf + min(0.18, seen_count * 0.02))), 4)
        track["staleness"] = 0
        track["last_seen_step"] = int(step)
        track["seen_count"] = int(seen_count + 1)
        track["observed_from"] = dict(observation.observed_from)
        track["last_observation"] = dict(observation.last_observation)
        track["estimated_location"] = dict(observation.estimated_location)
        track["viewpoint"] = dict(observation.viewpoint)
        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        interaction["pickup_ready"] = bool(interaction.get("pickup_ready") or (observation.candidate and observation.candidate.get("pickup_now")))
        interaction["place_ready"] = bool(interaction.get("place_ready") or (observation.candidate and observation.candidate.get("visual_place_ready")))
        interaction["last_surface_id"] = observation.last_observation.get("surface_candidate_id") or interaction.get("last_surface_id")
        track["interaction"] = interaction
        if track.get("task_class") in {"place_receptacle", "surface_target"} and interaction.get("place_ready"):
            track["status"] = "surface_ready"
        elif track.get("status") == "stale":
            track["status"] = "seen" if track.get("task_class") != "pickup_target" else "unpicked"

    def update_from_observation(
        self,
        analysis: JsonDict,
        *,
        current_cell: str,
        heading: str,
        step: int,
        navigation_status: Optional[JsonDict] = None,
        persist: bool = True,
    ) -> JsonDict:
        memory = self.load_memory()
        nav_status = navigation_status if isinstance(navigation_status, dict) else {}

        position_frame = self._position_frame_from_navigation(nav_status)
        cell_size = float(position_frame.get("cell_size_m") or DEFAULT_CELL_SIZE)

        memory["map_frame"]["cell_size_m"] = cell_size
        memory["map_frame"]["coordinate_mode"] = position_frame.get("coordinate_mode", "action_odometry_grid")
        memory["map_frame"]["origin_cell"] = position_frame.get("origin_cell", "0,0")
        memory["map_frame"]["pose_confidence"] = round(float(position_frame.get("pose_confidence", 1.0) or 1.0), 4)
        memory["map_frame"]["position_uncertainty_cells"] = round(
            float(position_frame.get("position_uncertainty_cells", 0.0) or 0.0),
            4,
        )
        memory["map_frame"]["heading_confidence"] = round(float(position_frame.get("heading_confidence", 1.0) or 1.0), 4)

        observations = self._extract_observations(
            analysis,
            current_cell=current_cell,
            heading=heading,
            step=step,
            navigation_status=nav_status,
        )
        seen_track_ids: List[str] = []
        events: List[JsonDict] = []
        created = updated = merged = 0
        tracks = memory.setdefault("tracks", {})
        for observation in observations:
            track_id = self._find_track(memory, observation, step)
            if track_id is None:
                track = self._new_track(memory, observation, step)
                track_id = track["track_id"]
                tracks[track_id] = track
                created += 1
                events.append({"event": "object_track_created", "track_id": track_id, "label": observation.label})
            else:
                track = tracks[track_id]
                self._merge_track(track, observation, step)
                updated += 1
                merged += 1
                events.append({"event": "object_track_updated", "track_id": track_id, "label": observation.label})
                events.append({"event": "object_track_merged", "track_id": track_id, "label": observation.label})
            seen_track_ids.append(track_id)
            if isinstance(observation.candidate, dict):
                observation.candidate["object_memory_track_id"] = track_id
                observation.candidate["object_memory_estimated_cell"] = tracks[track_id].get("estimated_location", {}).get("estimated_object_cell")
                observation.candidate["object_memory_view_cell"] = tracks[track_id].get("viewpoint", {}).get("recommended_view_cell")
                observation.candidate["object_memory_status"] = tracks[track_id].get("status")
                interaction = tracks[track_id].get("interaction") if isinstance(tracks[track_id].get("interaction"), dict) else {}
                if interaction.get("last_rejected_reason"):
                    observation.candidate["object_memory_rejected_reason"] = interaction.get("last_rejected_reason")

        stale_threshold = env_int("ROBOT_OBJECT_MEMORY_STALE_STEPS", 35)
        for track_id, track in list(tracks.items()):
            if not isinstance(track, dict):
                continue
            if track_id in seen_track_ids:
                continue
            if str(track.get("status") or "") in {"picked", "placed", "unreachable", *REJECTED_PICKUP_STATUSES}:
                continue
            try:
                staleness = max(0, int(step) - int(track.get("last_seen_step", step) or step))
            except (TypeError, ValueError):
                staleness = int(track.get("staleness", 0) or 0) + 1
            track["staleness"] = staleness
            if staleness >= stale_threshold and str(track.get("status") or "") != "stale":
                track["status"] = "stale"
                events.append({"event": "object_status_changed", "track_id": track_id, "status": "stale"})

        recent = list(memory.get("recent_observations") or [])
        recent.extend(
            {
                "step": int(step),
                "track_id": track_id,
                **observation.to_record(),
            }
            for track_id, observation in zip(seen_track_ids, observations)
        )
        memory["recent_observations"] = recent[-80:]
        self._update_stats(memory, step, created=created, updated=updated, merged=merged)
        if persist:
            self.save_memory(memory)
        summary = {
            "status": "success",
            "result_type": "object_memory_updated",
            "step": int(step),
            "observed_from": {"cell": current_cell, "heading": heading},
            "observation_count": len(observations),
            "created_count": created,
            "updated_count": updated,
            "merged_count": merged,
            "track_count": len(tracks),
            "events": events,
        }
        return summary

    def _update_stats(self, memory: JsonDict, step: int, *, created: int = 0, updated: int = 0, merged: int = 0) -> None:
        tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
        stats = memory.get("stats") if isinstance(memory.get("stats"), dict) else {}
        stats["track_count"] = len(tracks)
        stats["active_pickup_count"] = sum(
            1 for track in tracks.values()
            if isinstance(track, dict)
            and track.get("task_class") == "pickup_target"
            and track.get("status") in {"unpicked", "targeted"}
        )
        stats["active_receptacle_count"] = sum(
            1 for track in tracks.values()
            if isinstance(track, dict)
            and track.get("task_class") in {"place_receptacle", "surface_target"}
            and track.get("status") not in {"stale", "unreachable"}
        )
        stats["created_count"] = int(stats.get("created_count", 0) or 0) + int(created)
        stats["updated_count"] = int(stats.get("updated_count", 0) or 0) + int(updated)
        stats["merged_count"] = int(stats.get("merged_count", 0) or 0) + int(merged)
        stats["stale_count"] = sum(1 for track in tracks.values() if isinstance(track, dict) and track.get("status") == "stale")
        stats["rejected_pickup_count"] = sum(
            1
            for track in tracks.values()
            if isinstance(track, dict)
            and track.get("task_class") == "pickup_target"
            and str(track.get("status") or "") in REJECTED_PICKUP_STATUSES
        )
        stats["last_update_step"] = int(step)
        stats["last_update"] = now_iso()
        memory["stats"] = stats

    def _pickup_floor_observation(
        self,
        track: JsonDict,
    ) -> JsonDict:
        """Summarize whether the latest pickup observation supports floor pickup."""
        obs = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
        surface_hint = str(obs.get("surface_hint") or track.get("surface_hint") or "").strip().lower()
        is_floor_level_raw = obs.get("is_floor_level")
        is_floor_level = bool(is_floor_level_raw)
        support_context_blocked = bool(obs.get("support_context_blocked"))

        bottom_y_ratio = depth_number(obs.get("bottom_y_ratio"))
        if bottom_y_ratio is None:
            geometry = obs.get("geometry") if isinstance(obs.get("geometry"), dict) else {}
            bottom_y_ratio = depth_number(geometry.get("bottom_y_ratio"))

        center_3d = obs.get("center_3d") if isinstance(obs.get("center_3d"), dict) else {}
        height_m = depth_number(obs.get("height_m"), depth_number(center_3d.get("y")))

        floor_min_bottom = env_float("ROBOT_PICKUP_FLOOR_MIN_BOTTOM_RATIO", 0.78)
        # In the camera-relative frame used by the project, floor objects usually
        # have negative y, while countertop objects often appear around positive y.
        floor_max_center_y = env_float("ROBOT_PICKUP_FLOOR_MAX_CENTER_Y", -0.10)
        elevated_hints = {"surface_or_elevated", "support_surface", "table", "counter_top", "countertop"}
        explicit_elevated = bool(surface_hint in elevated_hints and is_floor_level_raw is False)
        floor_contact = obs.get("floor_contact_geometry") if isinstance(obs.get("floor_contact_geometry"), dict) else {}

        floor_evidence = False
        if surface_hint == "floor" and is_floor_level:
            floor_evidence = True
        if bottom_y_ratio is not None and bottom_y_ratio >= floor_min_bottom and is_floor_level:
            floor_evidence = True
        if bool(floor_contact.get("contact_floor_like")):
            floor_evidence = True
        if (
            height_m is not None
            and height_m <= floor_max_center_y
            and (
                not explicit_elevated
                or (
                    bottom_y_ratio is not None
                    and bottom_y_ratio >= floor_min_bottom
                )
            )
        ):
            floor_evidence = True

        return {
            "surface_hint": surface_hint,
            "is_floor_level_raw": is_floor_level_raw,
            "is_floor_level": is_floor_level,
            "support_context_blocked": support_context_blocked,
            "bottom_y_ratio": bottom_y_ratio,
            "height_m": height_m,
            "floor_evidence": floor_evidence,
        }

    def _pickup_goal_approach_verify_floor_like(
        self,
        track: JsonDict,
        floor: Optional[JsonDict] = None,
    ) -> bool:
        """Allow conservative navigation toward likely floor objects.

        This is intentionally weaker than immediate pickup readiness.  It only
        keeps a remembered object as a navigation/re-observation goal; the
        action contract still requires strict floor evidence before PickObject.
        """
        floor_data = floor if isinstance(floor, dict) else self._pickup_floor_observation(track)
        if bool(floor_data.get("floor_evidence")):
            return True
        if bool(floor_data.get("support_context_blocked")):
            return False

        label = normalize_label(track.get("label") or "")
        raw_labels = {
            normalize_label(item)
            for item in (track.get("raw_labels") or [])
            if normalize_label(item)
        }
        allowed_labels = env_label_set("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_LABELS", FOOD_LABELS)
        if "*" not in allowed_labels and label not in allowed_labels and not (raw_labels & allowed_labels):
            return False

        confidence = depth_number(track.get("confidence"), 0.0) or 0.0
        if confidence < env_float("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_MIN_CONFIDENCE", 0.70):
            return False

        obs = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
        position_hint = str(obs.get("position_hint") or "").strip()
        if position_hint and position_hint not in {"front-left", "front-center", "front-right", "memory"}:
            return False

        ground_distance = depth_number(
            obs.get("ground_distance_m"),
            depth_number(obs.get("ground_distance"), depth_number(obs.get("distance_m"))),
        )
        if ground_distance is not None:
            if ground_distance < env_float("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_MIN_GROUND_DISTANCE", 0.25):
                return False
            if ground_distance > env_float("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_MAX_GROUND_DISTANCE", 2.40):
                return False

        bottom_y_ratio = depth_number(floor_data.get("bottom_y_ratio"))
        height_m = depth_number(floor_data.get("height_m"))
        near_bottom = bool(
            bottom_y_ratio is not None
            and bottom_y_ratio >= env_float("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_MIN_BOTTOM_RATIO", 0.72)
        )
        low_center = bool(
            height_m is not None
            and height_m <= env_float("ROBOT_OBJECT_MEMORY_APPROACH_VERIFY_MAX_CENTER_Y", -0.10)
        )
        return bool(near_bottom and low_center)

    def _pickup_goal_rejection_reasons(
        self,
        track: JsonDict,
        *,
        step: int = 0,
        pickup_surface_policy: str = "floor-only",
    ) -> List[str]:
        """Return hard reasons why a track cannot become a navigation goal.

        Long-horizon navigation eligibility is broader than immediate pickup
        executability, but it still has to honor the active task contract.  In
        ``floor-only`` mode, a track that was just confirmed on a support surface
        should not keep stealing the pickup goal and forcing no-path scans.
        """
        reasons: List[str] = []

        if track.get("task_class") != "pickup_target":
            reasons.append("not_pickup_target")
            return reasons

        status = str(track.get("status") or "")
        if status in {"picked", "placed", "unreachable", "stale", *REJECTED_PICKUP_STATUSES}:
            reasons.append(f"status_{status}")
            return reasons

        confidence = depth_number(track.get("confidence"), 0.0) or 0.0
        min_confidence = env_float("ROBOT_OBJECT_MEMORY_PICKUP_GOAL_MIN_CONFIDENCE", 0.30)
        if confidence < min_confidence:
            reasons.append("confidence_below_goal_threshold")

        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        cooldown_until = interaction.get("cooldown_until_step")
        if cooldown_until is not None:
            try:
                if int(step) < int(cooldown_until):
                    reasons.append("cooldown_active")
            except (TypeError, ValueError):
                reasons.append("invalid_cooldown_until_step")

        estimated = track.get("estimated_location") if isinstance(track.get("estimated_location"), dict) else {}
        viewpoint = track.get("viewpoint") if isinstance(track.get("viewpoint"), dict) else {}
        estimated_cell = str(estimated.get("estimated_object_cell") or "").strip()
        view_cell = str(viewpoint.get("recommended_view_cell") or "").strip()
        if not estimated_cell:
            reasons.append("missing_estimated_object_cell")
        if not view_cell and not estimated_cell:
            reasons.append("missing_navigation_goal_cell")

        policy = str(pickup_surface_policy or "floor-only").strip().lower()
        if policy != "any-surface":
            floor = self._pickup_floor_observation(track)
            surface_hint = str(floor.get("surface_hint") or "")
            floor_evidence = bool(floor.get("floor_evidence"))
            approach_verify_floor_like = self._pickup_goal_approach_verify_floor_like(track, floor)
            elevated_hints = {"surface_or_elevated", "support_surface", "table", "counter_top", "countertop"}
            if surface_hint in elevated_hints and not (floor_evidence or approach_verify_floor_like):
                reasons.append(f"floor_only_goal_surface_hint_{surface_hint}")
            if bool(floor.get("support_context_blocked")) and not (floor_evidence or approach_verify_floor_like):
                reasons.append("floor_only_goal_support_context_blocked")
            has_floor_contract = any(
                value is not None and value != ""
                for value in (
                    surface_hint,
                    floor.get("bottom_y_ratio"),
                    floor.get("height_m"),
                )
            )
            if has_floor_contract and not (floor_evidence or approach_verify_floor_like):
                reasons.append("floor_only_goal_insufficient_floor_evidence")

        return reasons

    def _pickup_action_rejection_reasons(
        self,
        track: JsonDict,
        *,
        pickup_surface_policy: str = "floor-only",
    ) -> List[str]:
        """Return strict reasons why PickObject is not executable *right now*.

        These reasons are intentionally kept separate from navigation-goal
        eligibility. They may request approach, alignment, viewpoint reacquisition,
        or a later re-observation without deleting the long-horizon object goal.
        """
        reasons: List[str] = []

        if track.get("task_class") != "pickup_target":
            reasons.append("not_pickup_target")
            return reasons

        status = str(track.get("status") or "")
        if status in {"picked", "placed", "unreachable", "stale", *REJECTED_PICKUP_STATUSES}:
            reasons.append(f"status_{status}")
            return reasons

        obs = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
        pickup_now = bool(obs.get("pickup_now"))
        if not pickup_now:
            reasons.append("pickup_now_false")

        policy = str(pickup_surface_policy or "floor-only").strip().lower()
        if policy == "any-surface":
            return reasons

        floor = self._pickup_floor_observation(track)
        surface_hint = str(floor.get("surface_hint") or "")
        support_context_blocked = bool(floor.get("support_context_blocked"))

        if surface_hint in {"surface_or_elevated", "support_surface", "table", "counter_top", "countertop"}:
            reasons.append(f"surface_hint_{surface_hint}")

        if support_context_blocked:
            reasons.append("support_context_blocked")

        if not bool(floor.get("floor_evidence")):
            reasons.append("insufficient_floor_evidence")

        return reasons

    def _track_score(
        self,
        track: JsonDict,
        *,
        goal_type: str,
        current_cell: str,
        step: int,
        navigation_status: Optional[JsonDict] = None,
    ) -> float:
        confidence = depth_number(track.get("confidence"), 0.0) or 0.0
        track_score = depth_number(track.get("track_score"), confidence) or confidence
        try:
            staleness = max(0, int(track.get("staleness", 0) or 0))
        except (TypeError, ValueError):
            staleness = 0
        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        try:
            failures = max(0, int(interaction.get("failure_count", 0) or 0))
        except (TypeError, ValueError):
            failures = 0
        estimated = track.get("estimated_location") if isinstance(track.get("estimated_location"), dict) else {}
        viewpoint = track.get("viewpoint") if isinstance(track.get("viewpoint"), dict) else {}
        target_cell = str(viewpoint.get("recommended_view_cell") or estimated.get("estimated_object_cell") or current_cell)
        uncertainty = depth_number(estimated.get("uncertainty_cells"), 1.0) or 1.0
        track_pose_confidence = depth_number(estimated.get("pose_confidence"), 1.0) or 1.0
        track_position_uncertainty = depth_number(estimated.get("position_uncertainty_cells"), uncertainty) or uncertainty
        current_pose_trust = pose_trust_from_navigation(navigation_status)
        distance_cost = manhattan_distance(current_cell, target_cell) * 0.04
        recency_bonus = max(0.0, 0.30 - staleness * 0.015)
        task_relevance = 1.0
        known_surface_bonus = 0.0
        if goal_type == "receptacle":
            known_surface_bonus = 0.35 if interaction.get("place_ready") or track.get("status") == "surface_ready" else 0.10
            if track.get("status") == "used_for_place":
                known_surface_bonus += 0.10
        trust_penalty = max(0.0, 0.65 - float(track_pose_confidence)) * 0.75
        trust_penalty += max(0.0, float(track_position_uncertainty) - 1.0) * 0.10
        if current_pose_trust.get("level") == "low":
            # The remembered object remains useful as a coarse region, but its
            # old exact viewpoint is no longer a strong navigation commitment.
            trust_penalty += 0.25 + min(0.30, staleness * 0.01)
        elif current_pose_trust.get("level") == "medium":
            trust_penalty += 0.10
        return (
            float(track_score) * task_relevance
            + recency_bonus
            + known_surface_bonus
            - staleness * 0.025
            - failures * 0.18
            - distance_cost
            - float(uncertainty) * 0.08
            - trust_penalty
        )

    def _target_payload(
        self,
        track: JsonDict,
        *,
        goal_type: str,
        score: float,
        navigation_status: Optional[JsonDict] = None,
    ) -> JsonDict:
        viewpoint = track.get("viewpoint") if isinstance(track.get("viewpoint"), dict) else {}
        estimated = track.get("estimated_location") if isinstance(track.get("estimated_location"), dict) else {}
        pose_trust = pose_trust_from_navigation(navigation_status)
        track_pose_confidence = depth_number(estimated.get("pose_confidence"), 1.0) or 1.0
        track_uncertainty = depth_number(estimated.get("position_uncertainty_cells"), estimated.get("uncertainty_cells", 1.0)) or 1.0
        precise_viewpoint_allowed = bool(
            pose_trust.get("precise_viewpoint_allowed")
            and float(track_pose_confidence) >= env_float("ROBOT_OBJECT_MEMORY_TRACK_PRECISE_CONFIDENCE", 0.60)
            and float(track_uncertainty) <= env_float("ROBOT_OBJECT_MEMORY_TRACK_PRECISE_UNCERTAINTY", 2.0)
        )
        goal_resolution = "exact_view_cell" if precise_viewpoint_allowed else "coarse_region"
        target = {
            "track_id": track.get("track_id"),
            "label": track.get("label"),
            "raw_label": (track.get("raw_labels") or [track.get("label")])[0],
            "label_family": track.get("label_family"),
            "task_class": track.get("task_class"),
            "goal_type": "pickup_target" if goal_type == "pickup" else "place_receptacle",
            "score": round(float(score), 4),
            "status": track.get("status"),
            "goal_cell": viewpoint.get("recommended_view_cell") or estimated.get("estimated_object_cell"),
            "recommended_view_cell": viewpoint.get("recommended_view_cell"),
            "nominal_recommended_view_cell": viewpoint.get("recommended_view_cell"),
            "recommended_heading": viewpoint.get("recommended_heading"),
            "goal_resolution": goal_resolution,
            "precise_viewpoint_allowed": precise_viewpoint_allowed,
            "pose_trust": pose_trust,
            "estimated_object_cell": estimated.get("estimated_object_cell"),
            "candidate_cells": estimated.get("candidate_cells", []),
            "viewpoint": dict(viewpoint),
            "estimated_location": dict(estimated),
            "last_observation": dict(track.get("last_observation") or {}),
            "observed_from": dict(track.get("observed_from") or {}),
            "first_seen_step": track.get("first_seen_step"),
            "last_seen_step": track.get("last_seen_step"),
            "staleness": track.get("staleness"),
            "object_memory_target": True,
        }
        return target

    def _write_goal(
        self,
        target: JsonDict,
        *,
        goal_type: str,
        step: int,
        current_cell: str,
        heading: str,
    ) -> JsonDict:
        goals = self.load_goals()
        active = goals.get("active_goal")
        track_id = str(target.get("track_id") or "unknown")
        goal_kind = "pickup_target" if goal_type == "pickup" else "place_receptacle"
        same_goal = bool(
            isinstance(active, dict)
            and active.get("track_id") == track_id
            and active.get("goal_type") == goal_kind
        )
        if isinstance(active, dict) and not same_goal:
            history = list(goals.get("history") or [])
            active["closed_step"] = int(step)
            active["goal_status"] = "superseded"
            history.append(active)
            goals["history"] = history[-80:]
        label = str(target.get("label") or "object")
        goal_cell = target.get("recommended_view_cell") or target.get("goal_cell")
        try:
            last_seen_step = int(target.get("last_seen_step", -1000000) or -1000000)
        except (TypeError, ValueError):
            last_seen_step = -1000000
        planner_inputs = {
            "interface": "object_goal_memory_v1",
            "map_pred": None,
            "goal": {
                "target_cell": goal_cell,
                "recommended_view_cell": target.get("recommended_view_cell"),
                "recommended_heading": target.get("recommended_heading"),
                "estimated_object_cell": target.get("estimated_object_cell"),
                "candidate_cells": target.get("candidate_cells", []),
                "goal_resolution": target.get("goal_resolution"),
                "precise_viewpoint_allowed": target.get("precise_viewpoint_allowed"),
            },
            "pose_trust": target.get("pose_trust", {}),
            "pose_pred": {
                "cell": str(current_cell),
                "heading": str(heading),
                "theta_deg": heading_to_theta(heading),
                "coordinate_mode": "action_odometry_grid",
            },
            "found_goal": bool(last_seen_step == int(step)),
            "new_goal": bool(not same_goal),
            "target_track_id": track_id,
            "goal_type": goal_kind,
        }
        active_goal = {
            "goal_id": (
                active.get("goal_id")
                if same_goal and isinstance(active, dict) and active.get("goal_id")
                else f"goal:{goal_type}:{label}:{track_id.rsplit(':', 1)[-1]}"
            ),
            "goal_type": goal_kind,
            "track_id": track_id,
            "goal_cell": goal_cell,
            "goal_status": "navigating_to_viewpoint",
            "created_step": int(active.get("created_step", step)) if same_goal and isinstance(active, dict) else int(step),
            "last_updated_step": int(step),
            "recommended_heading": target.get("recommended_heading"),
            "estimated_object_cell": target.get("estimated_object_cell"),
            "goal_resolution": target.get("goal_resolution"),
            "precise_viewpoint_allowed": target.get("precise_viewpoint_allowed"),
            "pose_trust": target.get("pose_trust", {}),
            "planner_inputs": planner_inputs,
        }
        goals["active_goal"] = active_goal
        self.save_goals(goals)
        return active_goal

    def _clear_active_goal(
        self,
        *,
        track_id: Optional[str] = None,
        goal_type: Optional[str] = None,
        step: int = 0,
        status: str = "failed",
        reason: str = "goal_cleared",
    ) -> Optional[JsonDict]:
        goals = self.load_goals()
        active = goals.get("active_goal")
        if not isinstance(active, dict):
            return None
        if track_id and str(active.get("track_id") or "") != str(track_id):
            return None
        if goal_type and str(active.get("goal_type") or "") != str(goal_type):
            return None
        active["goal_status"] = str(status or "failed")
        active["closed_step"] = int(step)
        active["closed_reason"] = str(reason or "goal_cleared")
        history = list(goals.get("history") or [])
        history.append(active)
        goals["history"] = history[-80:]
        goals["active_goal"] = None
        goals["last_update"] = now_iso()
        self.save_goals(goals)
        return active

    def update_active_goal_viewpoint(
        self,
        *,
        track_id: str,
        viewpoint: JsonDict,
        step: int,
        planner_status: str,
    ) -> Optional[JsonDict]:
        """Attach PlacementViewpointPlanner output to the persistent active goal."""
        goals = self.load_goals()
        active = goals.get("active_goal")
        if not isinstance(active, dict) or str(active.get("track_id") or "") != str(track_id or ""):
            return None
        target_cell = viewpoint.get("target_cell") or viewpoint.get("cell")
        target_heading = viewpoint.get("target_heading") or viewpoint.get("heading")
        if target_cell:
            active["goal_cell"] = str(target_cell)
        if target_heading:
            active["recommended_heading"] = str(target_heading)
        active["goal_status"] = str(planner_status or "placement_viewpoint_planning")
        active["last_updated_step"] = int(step)
        active["placement_viewpoint"] = dict(viewpoint)
        planner_inputs = active.get("planner_inputs") if isinstance(active.get("planner_inputs"), dict) else {}
        planner_inputs["placement_viewpoint"] = dict(viewpoint)
        goal = planner_inputs.get("goal") if isinstance(planner_inputs.get("goal"), dict) else {}
        if target_cell:
            goal["target_cell"] = str(target_cell)
            goal["recommended_view_cell"] = str(target_cell)
        if target_heading:
            goal["recommended_heading"] = str(target_heading)
        planner_inputs["goal"] = goal
        active["planner_inputs"] = planner_inputs
        goals["active_goal"] = active
        self.save_goals(goals)
        return active

    def select_pickup_target(
        self,
        *,
        current_cell: str,
        heading: str,
        step: int,
        navigation_status: Optional[JsonDict] = None,
        analysis: Optional[JsonDict] = None,
        pickup_surface_policy: str = "floor-only",
        blocked_track_ids: Optional[Iterable[Any]] = None,
        persist: bool = True,
    ) -> Optional[JsonDict]:
        """Select a long-horizon pickup navigation goal.

        A pickup track is allowed to remain a navigation/reacquisition goal even
        when it is not immediately executable. Immediate action readiness is
        reported separately through ``pickup_action_ready``.
        """
        memory = self.load_memory()
        tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
        goals = self.load_goals()
        active_goal = goals.get("active_goal") if isinstance(goals.get("active_goal"), dict) else {}
        sticky_track_id = ""
        if (
            isinstance(active_goal, dict)
            and active_goal.get("goal_type") == "pickup_target"
            and str(active_goal.get("goal_status") or "") not in {"completed", "unreachable", "failed", "superseded"}
        ):
            sticky_track_id = str(active_goal.get("track_id") or "")
        sticky_bonus = env_float("ROBOT_OBJECT_MEMORY_STICKY_PICKUP_GOAL_BONUS", 0.40)

        candidates: List[Tuple[float, JsonDict, bool]] = []
        rejected_tracks: List[JsonDict] = []
        blocked_track_id_set = {str(item) for item in (blocked_track_ids or []) if str(item or "").strip()}

        for track in tracks.values():
            if not isinstance(track, dict):
                continue
            if track.get("task_class") != "pickup_target":
                continue
            if track.get("status") not in {"unpicked", "targeted"}:
                continue

            goal_rejection_reasons = self._pickup_goal_rejection_reasons(
                track,
                step=step,
                pickup_surface_policy=pickup_surface_policy,
            )
            track_id = str(track.get("track_id") or "")
            if track_id in blocked_track_id_set and "recently_placed_track" not in goal_rejection_reasons:
                goal_rejection_reasons.append("recently_placed_track")
            action_rejection_reasons = self._pickup_action_rejection_reasons(
                track,
                pickup_surface_policy=pickup_surface_policy,
            )
            interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
            pickup_action_ready = not action_rejection_reasons
            floor = self._pickup_floor_observation(track)
            approach_verify_goal = bool(
                not goal_rejection_reasons
                and not pickup_action_ready
                and self._pickup_goal_approach_verify_floor_like(track, floor)
                and not bool(floor.get("floor_evidence"))
            )

            interaction["pickup_goal_contract"] = "pickup_goal_action_split_v1"
            interaction["pickup_goal_eligible"] = not goal_rejection_reasons
            interaction["pickup_action_ready"] = pickup_action_ready
            interaction["pickup_approach_verify_goal"] = approach_verify_goal
            # Keep the legacy field aligned with the strict immediate-action contract.
            interaction["pickup_ready"] = pickup_action_ready
            interaction["pickup_goal_rejection_reasons"] = list(goal_rejection_reasons)
            interaction["pickup_action_rejection_reasons"] = list(action_rejection_reasons)
            interaction["pickup_goal_note"] = (
                "eligible_for_approach_verify_navigation"
                if approach_verify_goal
                else "eligible_for_reacquire_navigation"
                if not goal_rejection_reasons
                else "rejected_as_navigation_goal"
            )
            interaction["pickup_action_note"] = (
                "immediate_pickup_ready"
                if pickup_action_ready
                else "needs_reobservation_alignment_or_approach"
            )
            track["interaction"] = interaction

            if goal_rejection_reasons:
                rejected_tracks.append(
                    {
                        "track_id": track.get("track_id"),
                        "label": track.get("label"),
                        "status": track.get("status"),
                        "goal_rejection_reasons": list(goal_rejection_reasons),
                        "action_rejection_reasons": list(action_rejection_reasons),
                        "surface_hint": (track.get("last_observation") or {}).get("surface_hint"),
                        "is_floor_level": (track.get("last_observation") or {}).get("is_floor_level"),
                        "pickup_now": (track.get("last_observation") or {}).get("pickup_now"),
                    }
                )
                continue

            score = self._track_score(
                track,
                goal_type="pickup",
                current_cell=current_cell,
                step=step,
                navigation_status=navigation_status,
            )
            if pickup_action_ready:
                score += 0.08

            sticky_reused = bool(sticky_track_id and str(track.get("track_id") or "") == sticky_track_id)
            if sticky_reused:
                # Do not let a slightly different frontier or a noisy second track
                # steal the active long-horizon reacquisition goal every frame.
                score += sticky_bonus

            candidates.append((score, track, sticky_reused))

        if not candidates:
            if persist:
                memory.setdefault("selected_targets", {})["pickup"] = {
                    "track_id": None,
                    "selected_step": int(step),
                    "reason": "no_pickup_navigation_goal_eligible",
                    "pickup_surface_policy": str(pickup_surface_policy or "floor-only"),
                    "blocked_track_ids": sorted(blocked_track_id_set),
                    "rejected_tracks": rejected_tracks[-12:],
                }
                self.save_memory(memory)
                rejected_ids = {
                    str(item.get("track_id") or "")
                    for item in rejected_tracks
                    if isinstance(item, dict)
                }
                if sticky_track_id and sticky_track_id in rejected_ids:
                    self._clear_active_goal(
                        track_id=sticky_track_id,
                        goal_type="pickup_target",
                        step=step,
                        status="failed",
                        reason="no_pickup_navigation_goal_eligible",
                    )
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, track, sticky_reused = candidates[0]
        track["status"] = "targeted"
        target = self._target_payload(
            track,
            goal_type="pickup",
            score=score,
            navigation_status=navigation_status,
        )
        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        target["current_cell"] = str(current_cell)
        target["current_heading"] = str(heading)
        target["pickup_surface_policy"] = str(pickup_surface_policy or "floor-only")
        target["pickup_goal_eligible"] = bool(interaction.get("pickup_goal_eligible"))
        target["pickup_action_ready"] = bool(interaction.get("pickup_action_ready"))
        target["pickup_approach_verify_goal"] = bool(interaction.get("pickup_approach_verify_goal"))
        target["pickup_goal_rejection_reasons"] = list(interaction.get("pickup_goal_rejection_reasons") or [])
        target["pickup_action_rejection_reasons"] = list(interaction.get("pickup_action_rejection_reasons") or [])
        target["pickup_goal_note"] = interaction.get("pickup_goal_note")
        target["pickup_action_note"] = interaction.get("pickup_action_note")
        target["sticky_goal_reused"] = bool(sticky_reused)

        memory.setdefault("selected_targets", {})["pickup"] = {
            "track_id": track.get("track_id"),
            "selected_step": int(step),
            "score": round(float(score), 4),
            "recommended_view_cell": target.get("recommended_view_cell"),
            "pickup_surface_policy": str(pickup_surface_policy or "floor-only"),
            "blocked_track_ids": sorted(blocked_track_id_set),
            "pickup_goal_eligible": target.get("pickup_goal_eligible"),
            "pickup_action_ready": target.get("pickup_action_ready"),
            "pickup_approach_verify_goal": bool(interaction.get("pickup_approach_verify_goal")),
            "pickup_goal_rejection_reasons": target.get("pickup_goal_rejection_reasons"),
            "pickup_action_rejection_reasons": target.get("pickup_action_rejection_reasons"),
            "sticky_goal_reused": target.get("sticky_goal_reused"),
            "goal_resolution": target.get("goal_resolution"),
            "pose_trust": target.get("pose_trust", {}),
        }
        if persist:
            self.save_memory(memory)
            target["active_goal"] = self._write_goal(
                target,
                goal_type="pickup",
                step=step,
                current_cell=current_cell,
                heading=heading,
            )
        return target


    def select_receptacle_target(
        self,
        *,
        holding_object: bool,
        current_cell: str,
        heading: str,
        step: int,
        navigation_status: Optional[JsonDict] = None,
        held_object_family: Optional[str] = None,
        analysis: Optional[JsonDict] = None,
        persist: bool = True,
    ) -> Optional[JsonDict]:
        if not holding_object:
            return None
        memory = self.load_memory()
        tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
        candidates: List[Tuple[float, JsonDict]] = []
        for track in tracks.values():
            if not isinstance(track, dict):
                continue
            if track.get("task_class") not in {"place_receptacle", "surface_target"}:
                continue
            if track.get("status") in {"stale", "unreachable", "blocked"}:
                continue
            score = self._track_score(
                track,
                goal_type="receptacle",
                current_cell=current_cell,
                step=step,
                navigation_status=navigation_status,
            )
            candidates.append((score, track))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, track = candidates[0]
        if track.get("status") != "surface_ready":
            track["status"] = "target_receptacle"
        target = self._target_payload(
            track,
            goal_type="receptacle",
            score=score,
            navigation_status=navigation_status,
        )
        target["current_cell"] = str(current_cell)
        target["current_heading"] = str(heading)
        memory.setdefault("selected_targets", {})["receptacle"] = {
            "track_id": track.get("track_id"),
            "selected_step": int(step),
            "score": round(float(score), 4),
            "recommended_view_cell": target.get("recommended_view_cell"),
            "goal_resolution": target.get("goal_resolution"),
            "pose_trust": target.get("pose_trust", {}),
        }
        if persist:
            self.save_memory(memory)
            target["active_goal"] = self._write_goal(
                target,
                goal_type="receptacle",
                step=step,
                current_cell=current_cell,
                heading=heading,
            )
        return target

    def _resolve_track_id(self, memory: JsonDict, *, track_id: Optional[str] = None, candidate: Optional[JsonDict] = None, task_class: Optional[str] = None) -> Optional[str]:
        tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
        if track_id and track_id in tracks:
            return str(track_id)
        if isinstance(candidate, dict):
            direct = candidate.get("object_memory_track_id")
            if direct and str(direct) in tracks:
                return str(direct)
            signature = candidate_signature(candidate)
            label = normalize_label(candidate.get("raw_label") or candidate.get("label") or candidate.get("parent_object"))
            best: Optional[Tuple[int, str]] = None
            for candidate_track_id, track in tracks.items():
                if not isinstance(track, dict):
                    continue
                if task_class and str(track.get("task_class") or "") not in {task_class, "surface_target" if task_class == "place_receptacle" else task_class}:
                    continue
                obs = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
                if obs.get("candidate_signature") == signature:
                    return str(candidate_track_id)
                score = 0
                if labels_compatible(track.get("label"), label):
                    score += 2
                elif label:
                    continue
                candidate_hint = candidate_position_hint(candidate, candidate_bearing_deg(candidate))
                if candidate_hint and str(obs.get("position_hint") or "") == candidate_hint:
                    score += 1
                if best is None or score > best[0]:
                    best = (score, str(candidate_track_id))
            if best and best[0] > 0:
                return best[1]
        return None

    def pickup_cooldown_remaining(
        self,
        *,
        track_id: Optional[str] = None,
        candidate: Optional[JsonDict] = None,
        step: int = 0,
    ) -> int:
        memory = self.load_memory()
        resolved = self._resolve_track_id(memory, track_id=track_id, candidate=candidate, task_class="pickup_target")
        if not resolved:
            return 0
        track = memory.get("tracks", {}).get(resolved)
        if not isinstance(track, dict):
            return 0
        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        try:
            until_step = int(interaction.get("cooldown_until_step", 0) or 0)
        except (TypeError, ValueError):
            return 0
        return max(0, until_step - int(step))

    def mark_picked(
        self,
        *,
        track_id: Optional[str] = None,
        candidate: Optional[JsonDict] = None,
        step: int = 0,
    ) -> Optional[JsonDict]:
        memory = self.load_memory()
        resolved = self._resolve_track_id(memory, track_id=track_id, candidate=candidate, task_class="pickup_target")
        if not resolved:
            return None
        track = memory["tracks"][resolved]
        old_status = track.get("status")
        track["status"] = "picked"
        track.setdefault("interaction", {})["picked_step"] = int(step)
        memory.setdefault("selected_targets", {})["pickup"] = {"track_id": resolved, "selected_step": int(step), "status": "picked"}
        self.save_memory(memory)
        return {"track_id": resolved, "old_status": old_status, "status": "picked"}

    def mark_placed(
        self,
        *,
        held_track_id: Optional[str] = None,
        receptacle_track_id: Optional[str] = None,
        held_candidate: Optional[JsonDict] = None,
        receptacle_candidate: Optional[JsonDict] = None,
        step: int = 0,
    ) -> JsonDict:
        memory = self.load_memory()
        events: List[JsonDict] = []
        held_id = self._resolve_track_id(memory, track_id=held_track_id, candidate=held_candidate, task_class="pickup_target")
        receptacle_id = self._resolve_track_id(memory, track_id=receptacle_track_id, candidate=receptacle_candidate, task_class="place_receptacle")
        if held_id:
            track = memory["tracks"][held_id]
            old = track.get("status")
            track["status"] = "placed"
            track.setdefault("interaction", {})["placed_step"] = int(step)
            events.append({"track_id": held_id, "old_status": old, "status": "placed"})
            held_label = track.get("label")
            held_obs = track.get("last_observation") if isinstance(track.get("last_observation"), dict) else {}
            held_signature = str(held_obs.get("candidate_signature") or "")
            held_estimated = track.get("estimated_location") if isinstance(track.get("estimated_location"), dict) else {}
            held_cell = str(held_estimated.get("estimated_object_cell") or "")
            duplicate_radius = env_float("ROBOT_OBJECT_MEMORY_PLACED_DUPLICATE_CELL_RADIUS", 1.5)
            for other_id, other_track in (memory.get("tracks") or {}).items():
                if str(other_id) == str(held_id) or not isinstance(other_track, dict):
                    continue
                if other_track.get("task_class") != "pickup_target":
                    continue
                if str(other_track.get("status") or "") not in {"unpicked", "targeted"}:
                    continue
                if not labels_compatible(other_track.get("label"), held_label):
                    continue
                other_obs = other_track.get("last_observation") if isinstance(other_track.get("last_observation"), dict) else {}
                other_signature = str(other_obs.get("candidate_signature") or "")
                same_signature = bool(held_signature and other_signature and held_signature == other_signature)
                other_estimated = (
                    other_track.get("estimated_location")
                    if isinstance(other_track.get("estimated_location"), dict)
                    else {}
                )
                other_cell = str(other_estimated.get("estimated_object_cell") or "")
                same_region = False
                if held_cell and other_cell:
                    try:
                        same_region = cell_distance(held_cell, other_cell) <= duplicate_radius
                    except (TypeError, ValueError):
                        same_region = False
                if not (same_signature or same_region):
                    continue
                duplicate_old = other_track.get("status")
                other_track["status"] = "placed"
                interaction = other_track.setdefault("interaction", {})
                interaction["placed_step"] = int(step)
                interaction["duplicate_of_placed_track_id"] = held_id
                events.append(
                    {
                        "track_id": str(other_id),
                        "old_status": duplicate_old,
                        "status": "placed",
                        "duplicate_of": held_id,
                    }
                )
        if receptacle_id:
            track = memory["tracks"][receptacle_id]
            old = track.get("status")
            track["status"] = "used_for_place"
            track.setdefault("interaction", {})["used_for_place_step"] = int(step)
            events.append({"track_id": receptacle_id, "old_status": old, "status": "used_for_place"})
        goals = self.load_goals()
        active = goals.get("active_goal")
        if isinstance(active, dict):
            active["goal_status"] = "completed"
            active["closed_step"] = int(step)
            history = list(goals.get("history") or [])
            history.append(active)
            goals["history"] = history[-80:]
            goals["active_goal"] = None
            self.save_goals(goals)
        self.save_memory(memory)
        return {"events": events, "held_track_id": held_id, "receptacle_track_id": receptacle_id}

    def mark_unreachable(
        self,
        *,
        track_id: Optional[str] = None,
        candidate: Optional[JsonDict] = None,
        step: int = 0,
        reason: str = "unreachable",
    ) -> Optional[JsonDict]:
        memory = self.load_memory()
        resolved = self._resolve_track_id(memory, track_id=track_id, candidate=candidate)
        if not resolved:
            return None
        track = memory["tracks"][resolved]
        interaction = track.get("interaction") if isinstance(track.get("interaction"), dict) else {}
        interaction["failure_count"] = int(interaction.get("failure_count", 0) or 0) + 1
        interaction["last_failed_step"] = int(step)
        interaction["last_failure_reason"] = reason
        if interaction["failure_count"] >= env_int("ROBOT_OBJECT_MEMORY_UNREACHABLE_FAILURES", 3):
            track["status"] = "unreachable"
        track["interaction"] = interaction
        self.save_memory(memory)
        if track.get("status") == "unreachable":
            self._clear_active_goal(
                track_id=resolved,
                step=step,
                status="unreachable",
                reason=reason,
            )
        return {"track_id": resolved, "status": track.get("status"), "failure_count": interaction["failure_count"]}

    def mark_stale(
        self,
        *,
        track_id: Optional[str] = None,
        candidate: Optional[JsonDict] = None,
        step: int = 0,
        reason: str = "stale",
    ) -> Optional[JsonDict]:
        memory = self.load_memory()
        resolved = self._resolve_track_id(memory, track_id=track_id, candidate=candidate)
        if not resolved:
            return None
        track = memory["tracks"][resolved]
        old = track.get("status")
        track["status"] = "stale"
        track["staleness"] = max(int(track.get("staleness", 0) or 0), 1)
        interaction = track.setdefault("interaction", {})
        interaction["last_stale_step"] = int(step)
        interaction["last_stale_reason"] = reason
        if str(reason or "").startswith("pickup_lock_released:"):
            cooldown_steps = max(0, env_int("ROBOT_OBJECT_MEMORY_PICKUP_STALE_COOLDOWN_STEPS", 4))
            if cooldown_steps > 0:
                try:
                    existing_until = int(interaction.get("cooldown_until_step", 0) or 0)
                except (TypeError, ValueError):
                    existing_until = 0
                interaction["cooldown_until_step"] = max(existing_until, int(step) + cooldown_steps)
                interaction["cooldown_reason"] = reason
        selected = memory.setdefault("selected_targets", {}).get("pickup")
        if isinstance(selected, dict) and str(selected.get("track_id") or "") == str(resolved):
            memory["selected_targets"]["pickup"] = {
                "track_id": None,
                "selected_step": int(step),
                "reason": "track_marked_stale",
                "stale_track_id": resolved,
                "stale_reason": reason,
            }
        self.save_memory(memory)
        cleared_goal = self._clear_active_goal(
            track_id=resolved,
            goal_type="pickup_target",
            step=step,
            status="stale",
            reason=reason,
        )
        return {
            "track_id": resolved,
            "old_status": old,
            "status": "stale",
            "active_goal_cleared": bool(cleared_goal),
        }

    def mark_rejected_false_positive(
        self,
        *,
        track_id: Optional[str] = None,
        candidate: Optional[JsonDict] = None,
        step: int = 0,
        reason: str = "rejected_false_positive",
    ) -> Optional[JsonDict]:
        """Permanently remove a failed pickup hypothesis from goal selection.

        ``stale`` is recoverable: a later observation may revive the track.
        This state is stricter and is used after the runner has already chased
        a pickup hypothesis but still cannot reacquire or execute it.
        """
        memory = self.load_memory()
        resolved = self._resolve_track_id(memory, track_id=track_id, candidate=candidate, task_class="pickup_target")
        if not resolved:
            return None
        track = memory["tracks"][resolved]
        old = track.get("status")
        track["status"] = "rejected_false_positive"
        track["staleness"] = max(int(track.get("staleness", 0) or 0), 1)
        interaction = track.setdefault("interaction", {})
        interaction["last_rejected_step"] = int(step)
        interaction["last_rejected_reason"] = str(reason or "rejected_false_positive")
        interaction["rejected_count"] = int(interaction.get("rejected_count", 0) or 0) + 1
        interaction["pickup_goal_eligible"] = False
        interaction["pickup_action_ready"] = False
        interaction["pickup_ready"] = False
        interaction["pickup_goal_note"] = "rejected_after_failed_reacquisition"
        interaction["pickup_action_note"] = "rejected_after_failed_reacquisition"
        interaction["pickup_goal_rejection_reasons"] = ["status_rejected_false_positive"]
        interaction["pickup_action_rejection_reasons"] = ["status_rejected_false_positive"]

        selected = memory.setdefault("selected_targets", {}).get("pickup")
        if isinstance(selected, dict) and str(selected.get("track_id") or "") == str(resolved):
            memory["selected_targets"]["pickup"] = {
                "track_id": None,
                "selected_step": int(step),
                "reason": "track_rejected_false_positive",
                "rejected_track_id": resolved,
                "rejected_reason": str(reason or "rejected_false_positive"),
            }
        self.save_memory(memory)
        cleared_goal = self._clear_active_goal(
            track_id=resolved,
            goal_type="pickup_target",
            step=step,
            status="rejected",
            reason=reason,
        )
        return {
            "track_id": resolved,
            "old_status": old,
            "status": "rejected_false_positive",
            "active_goal_cleared": bool(cleared_goal),
            "reason": str(reason or "rejected_false_positive"),
        }


def update_from_observation(
    analysis: JsonDict,
    *,
    current_cell: str,
    heading: str,
    step: int,
    navigation_status: Optional[JsonDict] = None,
    memory_dir: Optional[Path] = None,
) -> JsonDict:
    return ObjectMemory(memory_dir).update_from_observation(
        analysis,
        current_cell=current_cell,
        heading=heading,
        step=step,
        navigation_status=navigation_status,
    )


def select_pickup_target(
    *,
    current_cell: str,
    heading: str,
    step: int,
    navigation_status: Optional[JsonDict] = None,
    pickup_surface_policy: str = "floor-only",
    blocked_track_ids: Optional[Iterable[Any]] = None,
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).select_pickup_target(
        current_cell=current_cell,
        heading=heading,
        step=step,
        navigation_status=navigation_status,
        pickup_surface_policy=pickup_surface_policy,
        blocked_track_ids=blocked_track_ids,
    )


def select_receptacle_target(
    *,
    holding_object: bool,
    current_cell: str,
    heading: str,
    step: int,
    navigation_status: Optional[JsonDict] = None,
    held_object_family: Optional[str] = None,
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).select_receptacle_target(
        holding_object=holding_object,
        current_cell=current_cell,
        heading=heading,
        step=step,
        navigation_status=navigation_status,
        held_object_family=held_object_family,
    )


def mark_picked(
    *,
    track_id: Optional[str] = None,
    candidate: Optional[JsonDict] = None,
    step: int = 0,
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).mark_picked(track_id=track_id, candidate=candidate, step=step)


def mark_placed(
    *,
    held_track_id: Optional[str] = None,
    receptacle_track_id: Optional[str] = None,
    held_candidate: Optional[JsonDict] = None,
    receptacle_candidate: Optional[JsonDict] = None,
    step: int = 0,
    memory_dir: Optional[Path] = None,
) -> JsonDict:
    return ObjectMemory(memory_dir).mark_placed(
        held_track_id=held_track_id,
        receptacle_track_id=receptacle_track_id,
        held_candidate=held_candidate,
        receptacle_candidate=receptacle_candidate,
        step=step,
    )


def mark_unreachable(
    *,
    track_id: Optional[str] = None,
    candidate: Optional[JsonDict] = None,
    step: int = 0,
    reason: str = "unreachable",
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).mark_unreachable(track_id=track_id, candidate=candidate, step=step, reason=reason)


def mark_stale(
    *,
    track_id: Optional[str] = None,
    candidate: Optional[JsonDict] = None,
    step: int = 0,
    reason: str = "stale",
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).mark_stale(track_id=track_id, candidate=candidate, step=step, reason=reason)


def mark_rejected_false_positive(
    *,
    track_id: Optional[str] = None,
    candidate: Optional[JsonDict] = None,
    step: int = 0,
    reason: str = "rejected_false_positive",
    memory_dir: Optional[Path] = None,
) -> Optional[JsonDict]:
    return ObjectMemory(memory_dir).mark_rejected_false_positive(
        track_id=track_id,
        candidate=candidate,
        step=step,
        reason=reason,
    )


def parse_json_arg(value: str) -> JsonDict:
    data = json.loads(value or "{}")
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage long-term object memory.")
    parser.add_argument("--memory-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show")
    update = sub.add_parser("update")
    update.add_argument("--analysis-json", required=True)
    update.add_argument("--current-cell", required=True)
    update.add_argument("--heading", required=True)
    update.add_argument("--step", type=int, required=True)
    update.add_argument("--navigation-json", default="{}")
    pick = sub.add_parser("select-pickup")
    pick.add_argument("--current-cell", required=True)
    pick.add_argument("--heading", required=True)
    pick.add_argument("--step", type=int, required=True)
    pick.add_argument("--navigation-json", default="{}")
    pick.add_argument("--pickup-surface-policy", default="floor-only", choices=["floor-only", "any-surface"])
    receptacle = sub.add_parser("select-receptacle")
    receptacle.add_argument("--current-cell", required=True)
    receptacle.add_argument("--heading", required=True)
    receptacle.add_argument("--step", type=int, required=True)
    receptacle.add_argument("--holding-object", action="store_true")
    receptacle.add_argument("--navigation-json", default="{}")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    memory_dir = Path(args.memory_dir) if args.memory_dir else None
    manager = ObjectMemory(memory_dir)
    if args.command == "show":
        result = manager.status()
    elif args.command == "update":
        result = manager.update_from_observation(
            parse_json_arg(args.analysis_json),
            current_cell=args.current_cell,
            heading=args.heading,
            step=int(args.step),
            navigation_status=parse_json_arg(args.navigation_json),
        )
    elif args.command == "select-pickup":
        result = manager.select_pickup_target(
            current_cell=args.current_cell,
            heading=args.heading,
            step=int(args.step),
            navigation_status=parse_json_arg(args.navigation_json),
            pickup_surface_policy=str(args.pickup_surface_policy or "floor-only"),
        ) or {"status": "empty", "result_type": "no_pickup_target"}
    elif args.command == "select-receptacle":
        result = manager.select_receptacle_target(
            holding_object=bool(args.holding_object),
            current_cell=args.current_cell,
            heading=args.heading,
            step=int(args.step),
            navigation_status=parse_json_arg(args.navigation_json),
        ) or {"status": "empty", "result_type": "no_receptacle_target"}
    else:
        raise ValueError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
