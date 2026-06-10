#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Depth local costmap for online-safe robot-cleaner navigation.

The global PositionMap stores action-odometry occupancy over the room.  This
module provides the complementary *local* layer used immediately before a
movement action:

RGB-D frame -> pinhole projection -> robot-local obstacle grid -> footprint
inflation -> action swept-volume checks.

The module intentionally consumes only the sanitized get-vision payload and
runner state.  It never reads AI2-THOR raw metadata.  A held object is modeled
as an expanded footprint so movement and rotation become more conservative
while carrying an item.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
LOCAL_COSTMAP_PATH = MEMORY_DIR / "navigation-costmap.json"
SCHEMA_VERSION = 1

TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
LOOK_ACTIONS = {"LookUp", "LookDown"}
MOVE_ACTIONS = TRANSLATION_ACTIONS | ROTATE_ACTIONS | LOOK_ACTIONS

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


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        for attempt in range(20):
            try:
                os.replace(tmp_name, path)
                tmp_name = ""
                break
            except PermissionError:
                if attempt >= 19:
                    raise
                time.sleep(min(0.5, 0.05 * (attempt + 1)))
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        raw = path.read_text(encoding="utf-8-sig")
        data = json.loads(raw) if raw.strip() else {}
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> Iterable[Tuple[int, int]]:
    """Yield integer cells from start to end inclusive."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


class LocalCostmap:
    """Build and persist a short-range robot-local obstacle costmap."""

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.path = self.memory_dir / "navigation-costmap.json"
        self.resolution_m = max(0.02, env_float("ROBOT_LOCAL_COSTMAP_RESOLUTION_M", 0.05))
        self.x_limit_m = max(0.60, env_float("ROBOT_LOCAL_COSTMAP_X_LIMIT_M", 1.50))
        self.z_back_m = max(0.25, env_float("ROBOT_LOCAL_COSTMAP_Z_BACK_M", 0.75))
        self.z_front_m = max(0.75, env_float("ROBOT_LOCAL_COSTMAP_Z_FRONT_M", 2.50))
        self.sample_stride = max(2, env_int("ROBOT_LOCAL_COSTMAP_SAMPLE_STRIDE", 4))
        self.depth_min_m = max(0.05, env_float("ROBOT_LOCAL_COSTMAP_DEPTH_MIN_M", 0.10))
        self.depth_max_m = max(self.depth_min_m + 0.1, env_float("ROBOT_LOCAL_COSTMAP_DEPTH_MAX_M", 3.00))
        self.floor_height_max_m = env_float("ROBOT_LOCAL_COSTMAP_FLOOR_HEIGHT_MAX_M", 0.08)
        self.obstacle_height_min_m = env_float("ROBOT_LOCAL_COSTMAP_OBSTACLE_HEIGHT_MIN_M", 0.10)
        self.obstacle_height_max_m = env_float("ROBOT_LOCAL_COSTMAP_OBSTACLE_HEIGHT_MAX_M", 1.80)
        self.robot_radius_m = max(0.08, env_float("ROBOT_LOCAL_COSTMAP_ROBOT_RADIUS_M", 0.18))
        self.held_extra_radius_m = max(0.0, env_float("ROBOT_LOCAL_COSTMAP_HELD_EXTRA_RADIUS_M", 0.08))
        self.held_small_extra_radius_m = max(0.0, env_float("ROBOT_LOCAL_COSTMAP_HELD_SMALL_EXTRA_RADIUS_M", 0.06))
        self.held_medium_extra_radius_m = max(0.0, env_float("ROBOT_LOCAL_COSTMAP_HELD_MEDIUM_EXTRA_RADIUS_M", 0.10))
        self.held_large_extra_radius_m = max(0.0, env_float("ROBOT_LOCAL_COSTMAP_HELD_LARGE_EXTRA_RADIUS_M", 0.14))
        self.translation_step_m = max(0.05, env_float("ROBOT_LOCAL_COSTMAP_TRANSLATION_STEP_M", 0.25))
        self.rotation_extra_margin_m = max(0.0, env_float("ROBOT_LOCAL_COSTMAP_ROTATION_EXTRA_MARGIN_M", 0.08))
        self.hard_block_confidence = clamp(env_float("ROBOT_LOCAL_COSTMAP_HARD_BLOCK_CONFIDENCE", 0.20))
        self.min_translation_observed_ratio = clamp(env_float("ROBOT_LOCAL_COSTMAP_MIN_TRANSLATION_OBSERVED_RATIO", 0.15))
        self.rear_min_observed_ratio = clamp(env_float("ROBOT_LOCAL_COSTMAP_REAR_MIN_OBSERVED_RATIO", 0.60))
        self.front_corridor_offset_m = max(
            self.resolution_m,
            env_float("ROBOT_LOCAL_COSTMAP_FRONT_CORRIDOR_OFFSET_M", 0.20),
        )

    def _default(self) -> JsonDict:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "not_observed",
            "result_type": "local_costmap_not_observed",
            "last_updated_at": None,
            "holding_object": False,
            "held_object_labels": [],
            "inflated_robot_radius_m": round(self.robot_radius_m, 4),
            "action_safety": {},
            "blocked_actions": [],
        }

    def reset(self) -> JsonDict:
        payload = self._default()
        payload.update({"status": "reset", "result_type": "local_costmap_reset", "last_updated_at": now_iso()})
        atomic_write_json(self.path, payload)
        return payload

    def status(self) -> JsonDict:
        return load_json(self.path, self._default())

    def _load_depth(self, vision: JsonDict) -> Any:
        depth = vision.get("depth_frame") if isinstance(vision, dict) else None
        if depth is not None:
            return depth
        depth_path = (vision.get("depth_path") or vision.get("depth_npy_path")) if isinstance(vision, dict) else None
        if not depth_path:
            return None
        try:
            import numpy as np  # type: ignore
            return np.load(str(depth_path))
        except Exception:
            return None

    def _camera(self, vision: JsonDict, shape: Tuple[int, int]) -> JsonDict:
        h, w = shape
        camera = vision.get("camera") if isinstance(vision.get("camera"), dict) else {}
        return {
            "fx": float(camera.get("fx") or max(1.0, w / 2.0)),
            "fy": float(camera.get("fy") or max(1.0, h / 2.0)),
            "cx": float(camera.get("cx") or (w / 2.0)),
            "cy": float(camera.get("cy") or (h / 2.0)),
            "camera_height_m": float(camera.get("camera_height_m", 0.901) or 0.901),
            "camera_horizon_deg": float(camera.get("camera_horizon_deg", 0.0) or 0.0),
        }

    def _grid_shape(self) -> Tuple[int, int, int, int]:
        min_ix = int(math.floor(-self.x_limit_m / self.resolution_m))
        max_ix = int(math.ceil(self.x_limit_m / self.resolution_m))
        min_iz = int(math.floor(-self.z_back_m / self.resolution_m))
        max_iz = int(math.ceil(self.z_front_m / self.resolution_m))
        return min_ix, max_ix, min_iz, max_iz

    def _to_grid(self, x_right_m: float, z_forward_m: float) -> Tuple[int, int]:
        return int(round(x_right_m / self.resolution_m)), int(round(z_forward_m / self.resolution_m))

    def _in_bounds(self, ix: int, iz: int, bounds: Tuple[int, int, int, int]) -> bool:
        min_ix, max_ix, min_iz, max_iz = bounds
        return min_ix <= ix <= max_ix and min_iz <= iz <= max_iz

    def _inflate(self, occupied: set[Tuple[int, int]], radius_m: float, bounds: Tuple[int, int, int, int]) -> set[Tuple[int, int]]:
        radius_cells = max(0, int(math.ceil(radius_m / self.resolution_m)))
        inflated = set(occupied)
        if radius_cells <= 0:
            return inflated
        offsets = [
            (dx, dz)
            for dx in range(-radius_cells, radius_cells + 1)
            for dz in range(-radius_cells, radius_cells + 1)
            if math.hypot(dx, dz) <= radius_cells
        ]
        for ox, oz in occupied:
            for dx, dz in offsets:
                cell = (ox + dx, oz + dz)
                if self._in_bounds(cell[0], cell[1], bounds):
                    inflated.add(cell)
        return inflated

    def _inflate_with_sources(
        self,
        occupied: set[Tuple[int, int]],
        radius_m: float,
        bounds: Tuple[int, int, int, int],
    ) -> Tuple[set[Tuple[int, int]], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        radius_cells = max(0, int(math.ceil(radius_m / self.resolution_m)))
        inflated = set(occupied)
        sources: Dict[Tuple[int, int], List[Tuple[int, int]]] = {cell: [cell] for cell in occupied}
        if radius_cells <= 0:
            return inflated, sources
        offsets = [
            (dx, dz)
            for dx in range(-radius_cells, radius_cells + 1)
            for dz in range(-radius_cells, radius_cells + 1)
            if math.hypot(dx, dz) <= radius_cells
        ]
        for ox, oz in occupied:
            for dx, dz in offsets:
                cell = (ox + dx, oz + dz)
                if not self._in_bounds(cell[0], cell[1], bounds):
                    continue
                inflated.add(cell)
                entries = sources.setdefault(cell, [])
                if len(entries) < 6 and (ox, oz) not in entries:
                    entries.append((ox, oz))
        return inflated, sources

    def _footprint_cells(self, radius_cells: int) -> set[Tuple[int, int]]:
        return {
            (ix, iz)
            for ix in range(-radius_cells, radius_cells + 1)
            for iz in range(-radius_cells, radius_cells + 1)
            if math.hypot(ix, iz) <= radius_cells
        }

    def _corridor(self, action: str, radius_m: float) -> set[Tuple[int, int]]:
        """Return robot-center cells swept by one discrete action.

        Obstacles are already inflated by the body/held-object radius.  The
        translation corridor therefore follows the robot-center trajectory;
        applying the radius a second time would overinflate obstacles and make
        narrow but valid passages unusable.  Rotation checks retain a small
        extra sweep disk for carried-object swing margin.
        """
        step_cells = max(1, int(math.ceil(self.translation_step_m / self.resolution_m)))
        rotation_margin_cells = max(1, int(math.ceil(self.rotation_extra_margin_m / self.resolution_m)))
        if action == "MoveAhead":
            return {(0, iz) for iz in range(1, step_cells + 1)}
        if action == "MoveBack":
            return {(0, iz) for iz in range(-step_cells, 0)}
        if action == "MoveLeft":
            return {(ix, 0) for ix in range(-step_cells, 0)}
        if action == "MoveRight":
            return {(ix, 0) for ix in range(1, step_cells + 1)}
        if action in ROTATE_ACTIONS:
            return self._footprint_cells(rotation_margin_cells)
        return set()

    def _held_footprint_extra(self, labels: Sequence[str], analysis: JsonDict) -> Tuple[float, str]:
        """Return a conservative carried-object footprint extension.

        Perception already owns object semantics.  The local controller only
        converts those online-safe labels/families into a collision envelope;
        it never reads simulator object dimensions.
        """
        normalized = {str(item or "").strip().lower().replace("-", "_") for item in labels if str(item or "").strip()}
        family = str(analysis.get("held_object_family") or "").strip().lower()
        large = {"pan", "pot", "plate", "bowl", "kettle"}
        medium = {"book", "bottle", "mug", "cup", "vase", "soap_bottle", "remote", "remote_control"}
        if normalized & large or family in {"cookware", "dishware", "large"}:
            return max(self.held_extra_radius_m, self.held_large_extra_radius_m), "large_carried_object"
        if normalized & medium or family in {"container", "medium"}:
            return max(self.held_extra_radius_m, self.held_medium_extra_radius_m), "medium_carried_object"
        return max(self.held_extra_radius_m, self.held_small_extra_radius_m), "default_carried_object"

    def _held_overlay_boxes(self, analysis: JsonDict, labels: Sequence[str]) -> List[Tuple[int, int, int, int]]:
        """Return image boxes for the visible held-object overlay.

        Held items often occupy the lower center of the first-person RGB-D
        frame.  Those pixels are self geometry, not external scene obstacles.
        Perception identifies them online-safely; the costmap masks their depth
        points and separately expands the robot footprint.
        """
        boxes: List[Tuple[int, int, int, int]] = []
        records: List[Any] = []
        raw = analysis.get("held_object_overlay_candidates")
        if isinstance(raw, list):
            records.extend(raw)
        direct = analysis.get("held_object_overlay_bbox")
        if isinstance(direct, dict):
            records.append({"bbox": direct})
        for item in records:
            rec = item if isinstance(item, dict) else {}
            bbox = rec.get("bbox") if isinstance(rec.get("bbox"), dict) else rec
            try:
                x = int(float(bbox.get("x", 0) or 0))
                y = int(float(bbox.get("y", 0) or 0))
                w = int(float(bbox.get("w", 0) or 0))
                h = int(float(bbox.get("h", 0) or 0))
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _pixel_in_boxes(u: int, v: int, boxes: Sequence[Tuple[int, int, int, int]]) -> bool:
        return any(x0 <= u < x1 and y0 <= v < y1 for x0, y0, x1, y1 in boxes)

    @staticmethod
    def _candidate_bbox(candidate: JsonDict) -> Optional[JsonDict]:
        for key in ("bbox", "region_bbox", "parent_bbox"):
            bbox = candidate.get(key)
            if isinstance(bbox, dict):
                return bbox
        return None

    @staticmethod
    def _bbox_contains_pixel(bbox: JsonDict, u: int, v: int) -> bool:
        try:
            x = float(bbox.get("x", 0.0) or 0.0)
            y = float(bbox.get("y", 0.0) or 0.0)
            w = float(bbox.get("w", 0.0) or 0.0)
            h = float(bbox.get("h", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return bool(w > 0 and h > 0 and x <= float(u) < x + w and y <= float(v) < y + h)

    def _candidate_pixel_attribution(
        self,
        analysis: JsonDict,
        u: int,
        v: int,
        *,
        limit: int = 5,
    ) -> List[JsonDict]:
        """Return visual candidates whose boxes contain one obstacle pixel.

        This is diagnostic only.  The costmap is still driven by RGB-D geometry;
        labels are attached afterward so a blocked MoveAhead can be traced back
        to the visible object/surface region that produced the depth points.
        """
        candidate_keys = (
            "best_pickup_candidate",
            "best_receptacle_candidate",
            "best_surface_candidate",
            "best_rejected_surface_candidate",
            "best_obstacle_candidate",
            "trash_candidates",
            "service_candidates",
            "receptacle_candidates",
            "surface_regions",
            "surface_candidates",
            "visual_ready_surface_regions",
            "placement_avoidance_candidates",
            "top_obstacle_candidates",
            "obstacle_candidates",
        )
        candidates: List[Any] = []
        for key in candidate_keys:
            value = analysis.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif isinstance(value, dict):
                candidates.append(value)

        matches: List[JsonDict] = []
        seen: set[Tuple[str, str, str, str]] = set()
        for item in candidates:
            if not isinstance(item, dict):
                continue
            bbox = self._candidate_bbox(item)
            if not isinstance(bbox, dict) or not self._bbox_contains_pixel(bbox, u, v):
                continue
            label = str(
                item.get("raw_label")
                or item.get("label")
                or item.get("parent_label")
                or item.get("parent_object")
                or item.get("region_type")
                or item.get("source")
                or "unknown"
            )
            task_class = str(item.get("task_semantic_class") or "")
            source = str(item.get("surface_candidate_source") or item.get("source") or "")
            key = (label, task_class, source, json.dumps(bbox, sort_keys=True, separators=(",", ":")))
            if key in seen:
                continue
            seen.add(key)
            match = {
                "label": label,
                "task_semantic_class": task_class or None,
                "source": source or None,
                "position_hint": item.get("position_hint"),
                "surface_hint": item.get("surface_hint"),
                "confidence": item.get("confidence"),
                "bbox": {
                    "x": bbox.get("x"),
                    "y": bbox.get("y"),
                    "w": bbox.get("w"),
                    "h": bbox.get("h"),
                },
            }
            for field in (
                "surface_candidate_id",
                "visual_place_ready",
                "is_support_surface",
                "is_floor_level",
                "reachable",
                "blocked",
                "rejection_reasons",
            ):
                if field in item:
                    match[field] = item.get(field)
            matches.append(match)
            if len(matches) >= limit:
                break
        return matches

    def _add_occupied_record(
        self,
        records: Dict[Tuple[int, int], List[JsonDict]],
        cell: Tuple[int, int],
        record: JsonDict,
        *,
        max_records_per_cell: int = 4,
    ) -> None:
        entries = records.setdefault(cell, [])
        if len(entries) < max_records_per_cell:
            entries.append(record)

    def _blocked_source_records(
        self,
        blocked: set[Tuple[int, int]],
        inflated_sources: Dict[Tuple[int, int], List[Tuple[int, int]]],
        occupied_records: Dict[Tuple[int, int], List[JsonDict]],
        *,
        max_blocked_cells: int = 8,
        max_source_records: int = 12,
    ) -> List[JsonDict]:
        diagnostics: List[JsonDict] = []
        source_record_count = 0
        ordered_blocked = sorted(blocked, key=lambda cell: (abs(cell[1]), abs(cell[0]), cell[1], cell[0]))
        for blocked_cell in ordered_blocked[:max_blocked_cells]:
            source_cells = inflated_sources.get(blocked_cell, [blocked_cell])
            source_entries: List[JsonDict] = []
            for source_cell in source_cells:
                records = occupied_records.get(source_cell, [])
                for record in records:
                    if source_record_count >= max_source_records:
                        break
                    dx = (blocked_cell[0] - source_cell[0]) * self.resolution_m
                    dz = (blocked_cell[1] - source_cell[1]) * self.resolution_m
                    entry = {
                        "source_cell": [source_cell[0], source_cell[1]],
                        "inflation_distance_m": round(math.hypot(dx, dz), 4),
                    }
                    entry.update(record)
                    source_entries.append(entry)
                    source_record_count += 1
                if source_record_count >= max_source_records:
                    break
            diagnostics.append(
                {
                    "blocked_cell": [blocked_cell[0], blocked_cell[1]],
                    "blocked_cell_m": {
                        "x": round(blocked_cell[0] * self.resolution_m, 4),
                        "z": round(blocked_cell[1] * self.resolution_m, 4),
                    },
                    "source_records": source_entries,
                }
            )
            if source_record_count >= max_source_records:
                break
        return diagnostics

    def _clearance(self, occupied: set[Tuple[int, int]], *, side: str) -> float:
        points: List[float] = []
        for ix, iz in occupied:
            x = ix * self.resolution_m
            z = iz * self.resolution_m
            if side == "front" and z > 0 and abs(x) <= max(0.45, self.robot_radius_m * 2.2):
                points.append(math.hypot(x, z))
            elif side == "left" and x < 0 and -0.20 <= z <= 0.85:
                points.append(math.hypot(x, z))
            elif side == "right" and x > 0 and -0.20 <= z <= 0.85:
                points.append(math.hypot(x, z))
        if not points:
            return self.z_front_m if side == "front" else self.x_limit_m
        return min(points)

    def _safety_record(
        self,
        *,
        action: str,
        inflated: set[Tuple[int, int]],
        observed: set[Tuple[int, int]],
        radius_m: float,
        inflated_sources: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
        occupied_records: Optional[Dict[Tuple[int, int], List[JsonDict]]] = None,
    ) -> JsonDict:
        if action in LOOK_ACTIONS:
            return {"safe": True, "confidence": 1.0, "reason": "camera_pitch_action", "blocked_cell_count": 0, "observed_ratio": 1.0}
        corridor = self._corridor(action, radius_m)
        if not corridor:
            return {"safe": True, "confidence": 0.0, "reason": "no_swept_volume", "blocked_cell_count": 0, "observed_ratio": 0.0}
        blocked = corridor & inflated
        observed_ratio = float(len(corridor & observed)) / float(max(1, len(corridor)))
        min_observed = self.rear_min_observed_ratio if action == "MoveBack" else self.min_translation_observed_ratio
        unknown = bool(action in TRANSLATION_ACTIONS and observed_ratio < min_observed)
        confidence = clamp(observed_ratio + (0.35 if blocked else 0.0) + (0.80 if unknown else 0.0))
        safe = bool((not blocked) and not unknown)
        if blocked:
            reason = "inflated_obstacle_in_swept_volume"
        elif unknown:
            reason = "unknown_swept_volume"
        else:
            reason = "clear_swept_volume"
        record = {
            "safe": safe,
            "confidence": round(confidence, 4),
            "reason": reason,
            "blocked_cell_count": len(blocked),
            "observed_ratio": round(observed_ratio, 4),
            "min_observed_ratio": round(min_observed, 4) if action in TRANSLATION_ACTIONS else None,
        }
        if blocked:
            ordered_blocked = sorted(blocked, key=lambda cell: (abs(cell[1]), abs(cell[0]), cell[1], cell[0]))
            record["blocked_cells"] = [[cell[0], cell[1]] for cell in ordered_blocked[:12]]
            record["nearest_blocked_distance_m"] = round(
                min(math.hypot(cell[0] * self.resolution_m, cell[1] * self.resolution_m) for cell in blocked),
                4,
            )
            if inflated_sources is not None and occupied_records is not None:
                record["blocked_sources"] = self._blocked_source_records(
                    blocked,
                    inflated_sources,
                    occupied_records,
                )
        return record

    def _lane_safety_record(
        self,
        *,
        lane: str,
        offset_cells: int,
        inflated: set[Tuple[int, int]],
        observed: set[Tuple[int, int]],
        inflated_sources: Dict[Tuple[int, int], List[Tuple[int, int]]],
        occupied_records: Dict[Tuple[int, int], List[JsonDict]],
    ) -> JsonDict:
        step_cells = max(1, int(math.ceil(self.translation_step_m / self.resolution_m)))
        corridor = {(offset_cells, iz) for iz in range(1, step_cells + 1)}
        blocked = corridor & inflated
        observed_ratio = float(len(corridor & observed)) / float(max(1, len(corridor)))
        unknown = bool(observed_ratio < self.min_translation_observed_ratio)
        safe = bool((not blocked) and not unknown)
        if blocked:
            reason = "inflated_obstacle_in_lane"
        elif unknown:
            reason = "unknown_lane"
        else:
            reason = "clear_lane"
        record: JsonDict = {
            "lane": lane,
            "offset_cell": int(offset_cells),
            "offset_m": round(offset_cells * self.resolution_m, 4),
            "safe": safe,
            "reason": reason,
            "blocked_cell_count": len(blocked),
            "observed_ratio": round(observed_ratio, 4),
            "min_observed_ratio": round(self.min_translation_observed_ratio, 4),
        }
        if blocked:
            record["blocked_cells"] = [[cell[0], cell[1]] for cell in sorted(blocked)[:12]]
            record["nearest_blocked_distance_m"] = round(
                min(math.hypot(cell[0] * self.resolution_m, cell[1] * self.resolution_m) for cell in blocked),
                4,
            )
            record["blocked_sources"] = self._blocked_source_records(
                blocked,
                inflated_sources,
                occupied_records,
                max_blocked_cells=4,
                max_source_records=6,
            )
        return record

    def _dominant_blocker_side(self, blocked_sources: Sequence[JsonDict]) -> str:
        xs: List[float] = []
        for blocker in blocked_sources:
            if not isinstance(blocker, dict):
                continue
            for record in blocker.get("source_records", []) or []:
                if not isinstance(record, dict):
                    continue
                point = record.get("point_m") if isinstance(record.get("point_m"), dict) else {}
                try:
                    xs.append(float(point.get("x")))
                except (TypeError, ValueError):
                    source_cell = record.get("source_cell")
                    if isinstance(source_cell, list) and source_cell:
                        try:
                            xs.append(float(source_cell[0]) * self.resolution_m)
                        except (TypeError, ValueError):
                            pass
        if not xs:
            return "unknown"
        mean_x = sum(xs) / float(len(xs))
        threshold = max(0.05, self.robot_radius_m * 0.35)
        if mean_x > threshold:
            return "right"
        if mean_x < -threshold:
            return "left"
        return "center"

    def _front_corridor_report(
        self,
        *,
        inflated: set[Tuple[int, int]],
        observed: set[Tuple[int, int]],
        inflated_sources: Dict[Tuple[int, int], List[Tuple[int, int]]],
        occupied_records: Dict[Tuple[int, int], List[JsonDict]],
        moveahead_record: JsonDict,
    ) -> JsonDict:
        offset_cells = max(1, int(round(self.front_corridor_offset_m / self.resolution_m)))
        lanes = {
            "left": self._lane_safety_record(
                lane="front_left",
                offset_cells=-offset_cells,
                inflated=inflated,
                observed=observed,
                inflated_sources=inflated_sources,
                occupied_records=occupied_records,
            ),
            "center": self._lane_safety_record(
                lane="front_center",
                offset_cells=0,
                inflated=inflated,
                observed=observed,
                inflated_sources=inflated_sources,
                occupied_records=occupied_records,
            ),
            "right": self._lane_safety_record(
                lane="front_right",
                offset_cells=offset_cells,
                inflated=inflated,
                observed=observed,
                inflated_sources=inflated_sources,
                occupied_records=occupied_records,
            ),
        }
        dominant_side = self._dominant_blocker_side(moveahead_record.get("blocked_sources", []) or [])
        preferred = None
        if moveahead_record.get("safe") is False:
            if dominant_side == "right" and bool(lanes["left"].get("safe")):
                preferred = "left"
            elif dominant_side == "left" and bool(lanes["right"].get("safe")):
                preferred = "right"
            elif bool(lanes["left"].get("safe")) and not bool(lanes["right"].get("safe")):
                preferred = "left"
            elif bool(lanes["right"].get("safe")) and not bool(lanes["left"].get("safe")):
                preferred = "right"
        return {
            "status": "success",
            "offset_m": round(offset_cells * self.resolution_m, 4),
            "dominant_blocker_side": dominant_side,
            "preferred_bypass_side": preferred,
            "asymmetric_bypass_available": bool(preferred),
            "lanes": lanes,
        }

    def update(
        self,
        *,
        vision: JsonDict,
        analysis: Optional[JsonDict] = None,
        holding_object: bool = False,
        held_object_labels: Optional[Sequence[str]] = None,
        step: Optional[int] = None,
        persist: bool = True,
    ) -> JsonDict:
        analysis = analysis if isinstance(analysis, dict) else {}
        labels = [str(item) for item in (held_object_labels or []) if str(item).strip()]
        depth = self._load_depth(vision or {})
        if depth is None:
            payload = self._default()
            payload.update({
                "status": "skipped",
                "result_type": "local_costmap_no_depth",
                "reason": "no_depth_frame",
                "holding_object": bool(holding_object),
                "held_object_labels": labels,
                "last_updated_at": now_iso(),
                "step": step,
            })
            if persist:
                atomic_write_json(self.path, payload)
            return payload
        try:
            import numpy as np  # type: ignore
        except Exception:
            payload = self._default()
            payload.update({"status": "skipped", "result_type": "local_costmap_numpy_unavailable", "last_updated_at": now_iso(), "step": step})
            if persist:
                atomic_write_json(self.path, payload)
            return payload

        arr = np.asarray(depth)
        if arr.ndim != 2:
            payload = self._default()
            payload.update({"status": "skipped", "result_type": "local_costmap_bad_depth_shape", "depth_shape": list(arr.shape), "last_updated_at": now_iso(), "step": step})
            if persist:
                atomic_write_json(self.path, payload)
            return payload

        h, w = arr.shape
        camera = self._camera(vision or {}, (h, w))
        fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]
        camera_height = camera["camera_height_m"]
        pitch = math.radians(camera["camera_horizon_deg"])
        cos_pitch, sin_pitch = math.cos(pitch), math.sin(pitch)
        bounds = self._grid_shape()
        origin = self._to_grid(0.0, 0.0)
        observed: set[Tuple[int, int]] = {origin}
        free: set[Tuple[int, int]] = {origin}
        occupied: set[Tuple[int, int]] = set()
        occupied_records: Dict[Tuple[int, int], List[JsonDict]] = {}
        sampled = 0
        skipped_held_overlay = 0
        held_overlay_boxes = self._held_overlay_boxes(analysis, labels) if holding_object else []

        for v in range(0, h, self.sample_stride):
            for u in range(0, w, self.sample_stride):
                if held_overlay_boxes and self._pixel_in_boxes(u, v, held_overlay_boxes):
                    skipped_held_overlay += 1
                    continue
                z_cam = float(arr[v, u])
                if not math.isfinite(z_cam) or not (self.depth_min_m <= z_cam <= self.depth_max_m):
                    continue
                sampled += 1
                x_right = (float(u) - cx) * z_cam / fx
                y_up_cam = -(float(v) - cy) * z_cam / fy
                z_forward = z_cam * cos_pitch + y_up_cam * sin_pitch
                world_height = camera_height + y_up_cam * cos_pitch - z_cam * sin_pitch
                if z_forward < -self.z_back_m or z_forward > self.z_front_m or abs(x_right) > self.x_limit_m:
                    continue
                target = self._to_grid(x_right, z_forward)
                if not self._in_bounds(target[0], target[1], bounds):
                    continue
                ray = list(_bresenham(origin[0], origin[1], target[0], target[1]))
                for cell in ray[:-1]:
                    if self._in_bounds(cell[0], cell[1], bounds):
                        observed.add(cell)
                        free.add(cell)
                observed.add(target)
                if self.obstacle_height_min_m <= world_height <= self.obstacle_height_max_m:
                    occupied.add(target)
                    self._add_occupied_record(
                        occupied_records,
                        target,
                        {
                            "grid_cell": [target[0], target[1]],
                            "pixel": {"u": int(u), "v": int(v)},
                            "point_m": {
                                "x": round(x_right, 4),
                                "z": round(z_forward, 4),
                                "height": round(world_height, 4),
                                "depth": round(z_cam, 4),
                            },
                            "height_band": "obstacle",
                            "candidate_overlaps": self._candidate_pixel_attribution(analysis, int(u), int(v)),
                        },
                    )
                    free.discard(target)
                elif world_height <= self.floor_height_max_m:
                    free.add(target)

        held_extra, held_profile = self._held_footprint_extra(labels, analysis) if holding_object else (0.0, "empty_hand")
        radius = self.robot_radius_m + held_extra
        inflated, inflated_sources = self._inflate_with_sources(occupied, radius, bounds)
        actions = sorted(MOVE_ACTIONS)
        action_safety = {
            action: self._safety_record(
                action=action,
                inflated=inflated,
                observed=observed,
                radius_m=radius,
                inflated_sources=inflated_sources,
                occupied_records=occupied_records,
            )
            for action in actions
        }
        front_corridor = self._front_corridor_report(
            inflated=inflated,
            observed=observed,
            inflated_sources=inflated_sources,
            occupied_records=occupied_records,
            moveahead_record=action_safety["MoveAhead"],
        )
        blocked_actions = sorted(action for action, rec in action_safety.items() if rec.get("safe") is False)

        payload: JsonDict = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "result_type": "local_costmap_updated",
            "online_safe": True,
            "last_updated_at": now_iso(),
            "step": step,
            "holding_object": bool(holding_object),
            "held_object_labels": labels,
            "grid": {
                "coordinate_frame": "robot_relative_x_right_z_forward",
                "resolution_m": round(self.resolution_m, 4),
                "x_limit_m": round(self.x_limit_m, 4),
                "z_back_m": round(self.z_back_m, 4),
                "z_front_m": round(self.z_front_m, 4),
                "sample_stride": self.sample_stride,
            },
            "camera": camera,
            "sampled_depth_point_count": sampled,
            "skipped_held_overlay_depth_point_count": skipped_held_overlay,
            "held_overlay_bbox_count": len(held_overlay_boxes),
            "observed_cell_count": len(observed),
            "free_cell_count": len(free),
            "occupied_cell_count": len(occupied),
            "inflated_cell_count": len(inflated),
            "obstacle_height_band_m": [
                round(self.obstacle_height_min_m, 4),
                round(self.obstacle_height_max_m, 4),
            ],
            "obstacle_source_record_count": sum(len(entries) for entries in occupied_records.values()),
            "inflated_robot_radius_m": round(radius, 4),
            "base_robot_radius_m": round(self.robot_radius_m, 4),
            "held_extra_radius_m": round(held_extra, 4),
            "held_footprint_profile": held_profile,
            "front_clearance_m": round(self._clearance(occupied, side="front"), 4),
            "left_clearance_m": round(self._clearance(occupied, side="left"), 4),
            "right_clearance_m": round(self._clearance(occupied, side="right"), 4),
            "action_safety": action_safety,
            "front_corridor": front_corridor,
            "blocked_actions": blocked_actions,
            "moveahead_blockers": action_safety["MoveAhead"].get("blocked_sources", []),
            "moveahead_safe": bool(action_safety["MoveAhead"]["safe"]),
            "moveback_safe": bool(action_safety["MoveBack"]["safe"]),
            "moveleft_safe": bool(action_safety["MoveLeft"]["safe"]),
            "moveright_safe": bool(action_safety["MoveRight"]["safe"]),
            "rotate_left_safe": bool(action_safety["RotateLeft"]["safe"]),
            "rotate_right_safe": bool(action_safety["RotateRight"]["safe"]),
            "lookup_safe": True,
            "lookdown_safe": True,
        }
        if persist:
            atomic_write_json(self.path, payload)
        return payload

    def action_record(self, action: str, status: Optional[JsonDict] = None) -> Optional[JsonDict]:
        data = status if isinstance(status, dict) else self.status()
        rec = (data.get("action_safety") or {}).get(str(action))
        return dict(rec) if isinstance(rec, dict) else None

    def known_unsafe(self, action: str, status: Optional[JsonDict] = None) -> bool:
        rec = self.action_record(action, status)
        if not rec:
            return False
        return bool(rec.get("safe") is False and float(rec.get("confidence", 0.0) or 0.0) >= self.hard_block_confidence)


def parse_json_arg(value: Optional[str]) -> JsonDict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an online-safe RGB-D local navigation costmap.")
    parser.add_argument("command", choices=["reset", "status", "update"])
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--vision-json", default="{}")
    parser.add_argument("--analysis-json", default="{}")
    parser.add_argument("--holding-object", action="store_true")
    parser.add_argument("--held-object-labels", default="")
    parser.add_argument("--step", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manager = LocalCostmap(Path(args.memory_dir) if args.memory_dir else None)
    if args.command == "reset":
        result = manager.reset()
    elif args.command == "status":
        result = manager.status()
    else:
        result = manager.update(
            vision=parse_json_arg(args.vision_json),
            analysis=parse_json_arg(args.analysis_json),
            holding_object=bool(args.holding_object),
            held_object_labels=[item.strip() for item in str(args.held_object_labels or "").split(",") if item.strip()],
            step=args.step,
            persist=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
