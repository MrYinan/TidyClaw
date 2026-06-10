#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistent semantic layer over the online-safe PositionMap.

This module implements a SemExp-inspired top-down semantic occupancy layer for
OpenClaw robot-cleaner.  It deliberately does *not* replace position-map.json:

- position-map.json owns geometry: free / occupied / unknown, visited cells,
  action odometry, blocked edges and pose uncertainty;
- semantic-map.json owns persistent semantic evidence attached to map cells;
- object-memory.json owns object tracks and object-goal viewpoints.

The layer fuses two online-safe sources:

1. current YOLO / RGB-D analysis candidates;
2. long-term tracks produced by object_memory_core.py.

No AI2-THOR raw metadata is read.  The resulting map is suitable for global
planning, semantic-frontier scoring and debugging.  It intentionally stores
uncertainty instead of presenting projected object cells as exact ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from scripts.object_memory_core import (
        candidate_bearing_deg,
        candidate_distance_m,
        project_observation_to_cell,
    )
    from scripts.position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        CELL_UNKNOWN,
        PositionMap,
        format_cell,
        four_neighbors,
        manhattan_distance,
        parse_cell,
    )
except ImportError:  # pragma: no cover - direct script execution
    from object_memory_core import candidate_bearing_deg, candidate_distance_m, project_observation_to_cell
    from position_map_core import (
        CELL_FREE,
        CELL_INFLATED,
        CELL_OCCUPIED,
        CELL_UNKNOWN,
        PositionMap,
        format_cell,
        four_neighbors,
        manhattan_distance,
        parse_cell,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
SEMANTIC_MAP_PATH = MEMORY_DIR / "semantic-map.json"
OBJECT_MEMORY_PATH = MEMORY_DIR / "object-memory.json"

SEMANTIC_MAP_SCHEMA_VERSION = 1
DEFAULT_CELL_SIZE_M = 0.25

TASK_PICKUP = "pickup_target"
TASK_RECEPTACLE = "place_receptacle"
TASK_SURFACE = "surface_target"
TASK_OBSTACLE = "obstacle"
TASK_IGNORED = "ignored_object"
SURFACE_SOURCES = {
    "depth_region_geometry",
    "pointcloud_plane",
    "pointcloud_plane_completion",
    "pointcloud_plane_grid_completion",
    "placement_value_map",
}

OBSERVATION_SINGLE_KEYS = (
    "best_pickup_candidate",
    "best_receptacle_candidate",
    "best_surface_candidate",
    "best_place_affordance",
    "best_obstacle_candidate",
)
OBSERVATION_LIST_KEYS = (
    "service_candidates",
    "receptacle_candidates",
    "visual_ready_surface_regions",
    "pointcloud_surface_regions",
    "surface_regions",
    "placement_avoidance_candidates",
    "ignored_candidates",
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


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def normalize_label(label: Any) -> str:
    value = str(label or "").strip()
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def unique_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text) if text.strip() else {}
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


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


def default_semantic_map(room_name: str = "current_room") -> JsonDict:
    return {
        "schema_version": SEMANTIC_MAP_SCHEMA_VERSION,
        "room_name": str(room_name or "current_room"),
        "map_frame": {
            "coordinate_mode": "action_odometry_grid",
            "cell_size_m": DEFAULT_CELL_SIZE_M,
            "origin_cell": "0,0",
            "axis": {"x_cell_positive": "east", "z_cell_positive": "north"},
        },
        "cells": {},
        "track_index": {},
        "frontier_scores": {},
        "stats": {
            "cell_count": 0,
            "semantic_cell_count": 0,
            "label_count": 0,
            "track_count": 0,
            "last_updated_step": None,
            "last_updated_at": None,
        },
    }


def _empty_semantic_cell(cell: str) -> JsonDict:
    x, z = parse_cell(cell)
    return {
        "cell": cell,
        "x": x,
        "z": z,
        "position_state": CELL_UNKNOWN,
        "visited": False,
        "occupancy_confidence": 0.0,
        "semantic_labels": {},
        "track_ids": [],
        "task_scores": {
            "pickup_target_score": 0.0,
            "receptacle_score": 0.0,
            "surface_score": 0.0,
            "obstacle_score": 0.0,
        },
        "last_seen_step": None,
        "last_updated_at": None,
    }


def _candidate_task_class(candidate: JsonDict) -> str:
    source = normalize_label(candidate.get("source") or candidate.get("surface_candidate_source"))
    label = normalize_label(candidate.get("label"))
    if candidate.get("surface_candidate_id") or source in SURFACE_SOURCES or label == "pc_surface":
        return TASK_SURFACE
    task_class = normalize_label(candidate.get("task_semantic_class") or candidate.get("task_class"))
    if task_class:
        return task_class
    if bool(candidate.get("is_support_surface")):
        return TASK_SURFACE
    return "unknown"


def _candidate_label(candidate: JsonDict, task_class: str) -> str:
    if task_class == TASK_SURFACE:
        return normalize_label(candidate.get("parent_label") or candidate.get("parent_object") or candidate.get("raw_label") or "surface")
    return normalize_label(candidate.get("label") or candidate.get("raw_label"))


def _semantic_score_key(task_class: str) -> Optional[str]:
    if task_class == TASK_PICKUP:
        return "pickup_target_score"
    if task_class == TASK_RECEPTACLE:
        return "receptacle_score"
    if task_class == TASK_SURFACE:
        return "surface_score"
    if task_class == TASK_OBSTACLE:
        return "obstacle_score"
    return None


def _track_task_class(track: JsonDict) -> str:
    task_class = normalize_label(track.get("task_class"))
    if task_class in {TASK_PICKUP, TASK_RECEPTACLE, TASK_SURFACE, TASK_OBSTACLE, TASK_IGNORED}:
        return task_class
    label_family = normalize_label(track.get("label_family"))
    if label_family == "receptacle":
        return TASK_RECEPTACLE
    if label_family in {"food", "pickup_target"}:
        return TASK_PICKUP
    return "unknown"


class SemanticMap:
    """Persistent semantic evidence attached to PositionMap cells."""

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.path = self.memory_dir / "semantic-map.json"
        self.object_memory_path = self.memory_dir / "object-memory.json"
        self.position_map = PositionMap(self.memory_dir)
        self.decay_per_step = clamp(env_float("ROBOT_SEMANTIC_MAP_DECAY_PER_STEP", 0.985), 0.80, 1.0)
        self.min_label_confidence = clamp(env_float("ROBOT_SEMANTIC_MAP_MIN_LABEL_CONFIDENCE", 0.03), 0.0, 1.0)
        self.max_labels_per_cell = max(4, env_int("ROBOT_SEMANTIC_MAP_MAX_LABELS_PER_CELL", 24))
        self.frontier_radius_cells = max(1, env_int("ROBOT_SEMANTIC_MAP_FRONTIER_RADIUS", 4))

    def load(self) -> JsonDict:
        return self.normalize(load_json(self.path, default_semantic_map()))

    def save(self, data: JsonDict) -> None:
        atomic_write_json(self.path, self.normalize(data))

    def reset(self, *, room_name: str = "current_room") -> JsonDict:
        data = default_semantic_map(room_name)
        self._sync_position_layer(data, self.position_map.status())
        self._refresh_stats(data)
        self.save(data)
        return self.status()

    def normalize(self, data: JsonDict) -> JsonDict:
        base = default_semantic_map(str(data.get("room_name") or "current_room"))
        if isinstance(data, dict):
            base.update(data)
        frame = base.get("map_frame") if isinstance(base.get("map_frame"), dict) else {}
        frame.setdefault("coordinate_mode", "action_odometry_grid")
        frame.setdefault("cell_size_m", DEFAULT_CELL_SIZE_M)
        frame.setdefault("origin_cell", "0,0")
        frame.setdefault("axis", {"x_cell_positive": "east", "z_cell_positive": "north"})
        base["map_frame"] = frame

        normalized_cells: Dict[str, JsonDict] = {}
        for raw_cell, raw_record in (base.get("cells") or {}).items():
            cell = str(raw_cell)
            try:
                parse_cell(cell)
            except Exception:
                continue
            rec = _empty_semantic_cell(cell)
            if isinstance(raw_record, dict):
                rec.update(raw_record)
            rec["cell"] = cell
            rec["x"], rec["z"] = parse_cell(cell)
            if rec.get("position_state") not in {CELL_UNKNOWN, CELL_FREE, CELL_OCCUPIED, CELL_INFLATED}:
                rec["position_state"] = CELL_UNKNOWN
            rec["visited"] = bool(rec.get("visited", False))
            rec["occupancy_confidence"] = clamp(float(rec.get("occupancy_confidence", 0.0) or 0.0))
            rec["track_ids"] = unique_strings(rec.get("track_ids", []))
            labels: Dict[str, JsonDict] = {}
            for raw_label, raw_label_record in (rec.get("semantic_labels") or {}).items():
                label = normalize_label(raw_label)
                if not label:
                    continue
                label_rec = raw_label_record if isinstance(raw_label_record, dict) else {}
                confidence = clamp(float(label_rec.get("confidence", 0.0) or 0.0))
                if confidence < self.min_label_confidence:
                    continue
                labels[label] = {
                    "confidence": round(confidence, 6),
                    "evidence_count": max(0, int(label_rec.get("evidence_count", 0) or 0)),
                    "task_classes": unique_strings(label_rec.get("task_classes", [])),
                    "sources": unique_strings(label_rec.get("sources", [])),
                    "track_ids": unique_strings(label_rec.get("track_ids", [])),
                    "last_seen_step": label_rec.get("last_seen_step"),
                    "last_decay_step": label_rec.get("last_decay_step", label_rec.get("last_seen_step")),
                    "last_updated_at": label_rec.get("last_updated_at"),
                }
            rec["semantic_labels"] = labels
            task_scores = rec.get("task_scores") if isinstance(rec.get("task_scores"), dict) else {}
            rec["task_scores"] = {
                "pickup_target_score": clamp(float(task_scores.get("pickup_target_score", 0.0) or 0.0)),
                "receptacle_score": clamp(float(task_scores.get("receptacle_score", 0.0) or 0.0)),
                "surface_score": clamp(float(task_scores.get("surface_score", 0.0) or 0.0)),
                "obstacle_score": clamp(float(task_scores.get("obstacle_score", 0.0) or 0.0)),
            }
            normalized_cells[cell] = rec
        base["cells"] = normalized_cells
        base["track_index"] = base.get("track_index") if isinstance(base.get("track_index"), dict) else {}
        base["frontier_scores"] = base.get("frontier_scores") if isinstance(base.get("frontier_scores"), dict) else {}
        base["stats"] = base.get("stats") if isinstance(base.get("stats"), dict) else {}
        return base

    def ensure_cell(self, data: JsonDict, cell: str) -> JsonDict:
        cells = data.setdefault("cells", {})
        if cell not in cells:
            cells[cell] = _empty_semantic_cell(cell)
        return cells[cell]

    def _sync_position_layer(self, data: JsonDict, position_status: JsonDict) -> None:
        if not isinstance(position_status, dict):
            return
        frame = position_status.get("map_frame") if isinstance(position_status.get("map_frame"), dict) else {}
        if frame:
            data["map_frame"] = dict(frame)
        if position_status.get("room_name"):
            data["room_name"] = str(position_status.get("room_name"))
        for cell, position_rec in (position_status.get("cells") or {}).items():
            if not isinstance(position_rec, dict):
                continue
            rec = self.ensure_cell(data, str(cell))
            rec["position_state"] = str(position_rec.get("state") or CELL_UNKNOWN)
            rec["visited"] = bool(position_rec.get("visited", False))
            rec["occupancy_confidence"] = clamp(float(position_rec.get("occupancy_confidence", 0.0) or 0.0))

    def _decay_cell(self, rec: JsonDict, *, step: int) -> None:
        labels = rec.get("semantic_labels") if isinstance(rec.get("semantic_labels"), dict) else {}
        kept: Dict[str, JsonDict] = {}
        for label, label_rec in labels.items():
            last_step = label_rec.get("last_decay_step", label_rec.get("last_seen_step"))
            try:
                age = max(0, int(step) - int(last_step)) if last_step is not None else 0
            except (TypeError, ValueError):
                age = 0
            confidence = clamp(float(label_rec.get("confidence", 0.0) or 0.0) * (self.decay_per_step ** age))
            if confidence < self.min_label_confidence:
                continue
            updated = dict(label_rec)
            updated["confidence"] = round(confidence, 6)
            updated["last_decay_step"] = int(step)
            kept[label] = updated
        rec["semantic_labels"] = kept
        self._recompute_task_scores(rec)

    def _recompute_task_scores(self, rec: JsonDict) -> None:
        scores = {
            "pickup_target_score": 0.0,
            "receptacle_score": 0.0,
            "surface_score": 0.0,
            "obstacle_score": 0.0,
        }
        for label_rec in (rec.get("semantic_labels") or {}).values():
            confidence = clamp(float(label_rec.get("confidence", 0.0) or 0.0))
            for task_class in label_rec.get("task_classes", []) or []:
                key = _semantic_score_key(str(task_class))
                if key:
                    scores[key] = max(scores[key], confidence)
        rec["task_scores"] = {key: round(value, 6) for key, value in scores.items()}

    def _update_label(
        self,
        data: JsonDict,
        *,
        cell: str,
        label: str,
        task_class: str,
        confidence: float,
        step: int,
        source: str,
        track_id: Optional[str] = None,
        accumulate: bool = True,
    ) -> None:
        label = normalize_label(label)
        task_class = normalize_label(task_class)
        if not label or confidence <= 0.0:
            return
        rec = self.ensure_cell(data, cell)
        labels = rec.setdefault("semantic_labels", {})
        old = labels.get(label) if isinstance(labels.get(label), dict) else {}
        old_conf = clamp(float(old.get("confidence", 0.0) or 0.0))
        # Current-frame observations use Noisy-OR accumulation. Persistent
        # object-memory synchronization uses max() so replaying the same track
        # every step does not artificially drive confidence to 1.0.
        if accumulate:
            new_conf = 1.0 - (1.0 - old_conf) * (1.0 - clamp(confidence))
        else:
            new_conf = max(old_conf, clamp(confidence))
        label_rec = {
            "confidence": round(clamp(new_conf), 6),
            "evidence_count": int(old.get("evidence_count", 0) or 0) + 1,
            "task_classes": unique_strings(list(old.get("task_classes", []) or []) + ([task_class] if task_class else [])),
            "sources": unique_strings(list(old.get("sources", []) or []) + [source]),
            "track_ids": unique_strings(list(old.get("track_ids", []) or []) + ([track_id] if track_id else [])),
            "last_seen_step": int(step),
            "last_decay_step": int(step),
            "last_updated_at": now_iso(),
        }
        labels[label] = label_rec
        if len(labels) > self.max_labels_per_cell:
            ordered = sorted(labels.items(), key=lambda item: float(item[1].get("confidence", 0.0) or 0.0), reverse=True)
            rec["semantic_labels"] = dict(ordered[: self.max_labels_per_cell])
        rec["track_ids"] = unique_strings(list(rec.get("track_ids", []) or []) + ([track_id] if track_id else []))
        rec["last_seen_step"] = int(step)
        rec["last_updated_at"] = now_iso()
        self._recompute_task_scores(rec)

    def _analysis_candidates(self, analysis: JsonDict) -> List[Tuple[str, JsonDict]]:
        output: List[Tuple[str, JsonDict]] = []
        signatures = set()
        for key in OBSERVATION_SINGLE_KEYS:
            value = analysis.get(key)
            if isinstance(value, dict):
                signature = str(value.get("id") or value.get("surface_candidate_id") or (key, value.get("label"), value.get("bbox")))
                if signature not in signatures:
                    signatures.add(signature)
                    output.append((key, value))
        for key in OBSERVATION_LIST_KEYS:
            values = analysis.get(key)
            if not isinstance(values, list):
                continue
            for candidate in values:
                if not isinstance(candidate, dict):
                    continue
                signature = str(candidate.get("id") or candidate.get("surface_candidate_id") or (key, candidate.get("label"), candidate.get("bbox")))
                if signature in signatures:
                    continue
                signatures.add(signature)
                output.append((key, candidate))
        return output

    def _candidate_cells(
        self,
        candidate: JsonDict,
        *,
        current_cell: str,
        heading: str,
        cell_size_m: float,
    ) -> List[Tuple[str, float]]:
        explicit = candidate.get("object_memory_estimated_cell") or candidate.get("estimated_object_cell")
        if explicit:
            return [(str(explicit), 1.0)]
        estimated = project_observation_to_cell(
            current_cell,
            heading,
            bearing_deg=candidate_bearing_deg(candidate),
            distance_m=candidate_distance_m(candidate),
            cell_size_m=cell_size_m,
            position_hint=str(candidate.get("position_hint") or "front-center"),
        )
        cells: List[Tuple[str, float]] = []
        for record in estimated.get("candidate_cells", []) or []:
            if not isinstance(record, dict) or not record.get("cell"):
                continue
            cells.append((str(record.get("cell")), clamp(float(record.get("prob", 0.0) or 0.0))))
        if not cells and estimated.get("estimated_object_cell"):
            cells.append((str(estimated.get("estimated_object_cell")), 1.0))
        return cells

    def _fuse_current_analysis(
        self,
        data: JsonDict,
        analysis: JsonDict,
        *,
        navigation_status: JsonDict,
        step: int,
    ) -> int:
        current_cell = str(navigation_status.get("last_cell") or "0,0")
        heading = str(navigation_status.get("last_heading") or "north")
        frame = data.get("map_frame") if isinstance(data.get("map_frame"), dict) else {}
        cell_size_m = float(frame.get("cell_size_m", DEFAULT_CELL_SIZE_M) or DEFAULT_CELL_SIZE_M)
        count = 0
        for source, candidate in self._analysis_candidates(analysis):
            task_class = _candidate_task_class(candidate)
            label = _candidate_label(candidate, task_class)
            if not label or task_class == "unknown":
                continue
            confidence = clamp(float(candidate.get("confidence", candidate.get("score", 0.55)) or 0.55))
            if task_class == TASK_SURFACE:
                confidence = max(confidence, clamp(float(candidate.get("score", 0.0) or 0.0)))
            track_id = candidate.get("object_memory_track_id")
            for cell, probability in self._candidate_cells(
                candidate,
                current_cell=current_cell,
                heading=heading,
                cell_size_m=cell_size_m,
            ):
                self._update_label(
                    data,
                    cell=cell,
                    label=label,
                    task_class=task_class,
                    confidence=confidence * max(0.05, probability),
                    step=step,
                    source=f"analysis:{source}",
                    track_id=str(track_id) if track_id else None,
                )
                count += 1
        return count

    def _fuse_object_memory(self, data: JsonDict, *, step: int) -> int:
        memory = load_json(self.object_memory_path, {})
        tracks = memory.get("tracks") if isinstance(memory.get("tracks"), dict) else {}
        track_index: Dict[str, JsonDict] = {}
        fused = 0
        for track_id, track in tracks.items():
            if not isinstance(track, dict):
                continue
            label = normalize_label(track.get("label"))
            task_class = _track_task_class(track)
            status = str(track.get("status") or "unknown")
            if not label or task_class == "unknown" or status in {"stale", "unreachable"}:
                continue
            base_conf = clamp(float(track.get("confidence", 0.0) or 0.0))
            if status in {"picked", "placed"} and task_class == TASK_PICKUP:
                # Keep history, but do not make completed pickup targets attract
                # new search goals.
                base_conf *= 0.20
            estimated = track.get("estimated_location") if isinstance(track.get("estimated_location"), dict) else {}
            candidate_cells = estimated.get("candidate_cells") if isinstance(estimated.get("candidate_cells"), list) else []
            if not candidate_cells and estimated.get("estimated_object_cell"):
                candidate_cells = [{"cell": estimated.get("estimated_object_cell"), "prob": 1.0}]
            indexed_cells: List[str] = []
            for record in candidate_cells:
                if not isinstance(record, dict) or not record.get("cell"):
                    continue
                cell = str(record.get("cell"))
                probability = clamp(float(record.get("prob", 0.0) or 0.0))
                self._update_label(
                    data,
                    cell=cell,
                    label=label,
                    task_class=task_class,
                    confidence=base_conf * max(0.05, probability),
                    step=step,
                    source="object_memory_track",
                    track_id=str(track_id),
                    accumulate=False,
                )
                indexed_cells.append(cell)
                fused += 1
            track_index[str(track_id)] = {
                "track_id": str(track_id),
                "label": label,
                "task_class": task_class,
                "status": status,
                "confidence": round(base_conf, 6),
                "cells": unique_strings(indexed_cells),
                "recommended_view_cell": (track.get("viewpoint") or {}).get("recommended_view_cell"),
                "last_seen_step": track.get("last_seen_step"),
            }
        data["track_index"] = track_index
        return fused

    def _refresh_frontier_scores(self, data: JsonDict, position_status: JsonDict) -> None:
        scores: Dict[str, JsonDict] = {}
        cells = data.get("cells") if isinstance(data.get("cells"), dict) else {}
        frontiers = position_status.get("frontier_cells") or position_status.get("frontiers") or []
        for frontier in frontiers:
            frontier = str(frontier)
            pickup = receptacle = surface = obstacle = 0.0
            for cell, rec in cells.items():
                if not isinstance(rec, dict):
                    continue
                distance = manhattan_distance(frontier, cell)
                if distance > self.frontier_radius_cells:
                    continue
                discount = 1.0 / float(1 + distance)
                task_scores = rec.get("task_scores") if isinstance(rec.get("task_scores"), dict) else {}
                pickup = max(pickup, discount * float(task_scores.get("pickup_target_score", 0.0) or 0.0))
                receptacle = max(receptacle, discount * float(task_scores.get("receptacle_score", 0.0) or 0.0))
                surface = max(surface, discount * float(task_scores.get("surface_score", 0.0) or 0.0))
                obstacle = max(obstacle, discount * float(task_scores.get("obstacle_score", 0.0) or 0.0))
            unknown_neighbors = sum(1 for nb in four_neighbors(frontier) if nb not in cells or cells[nb].get("position_state") == CELL_UNKNOWN)
            scores[frontier] = {
                "cell": frontier,
                "pickup_target_score": round(clamp(pickup), 6),
                "receptacle_score": round(clamp(max(receptacle, surface)), 6),
                "surface_score": round(clamp(surface), 6),
                "obstacle_score": round(clamp(obstacle), 6),
                "unknown_neighbor_count": int(unknown_neighbors),
                "exploration_score": round(clamp(0.20 * unknown_neighbors + 0.35 * max(pickup, receptacle, surface) - 0.20 * obstacle), 6),
            }
        data["frontier_scores"] = scores

    def _refresh_stats(self, data: JsonDict, *, step: Optional[int] = None) -> None:
        cells = data.get("cells") if isinstance(data.get("cells"), dict) else {}
        labels = set()
        semantic_cells = 0
        for rec in cells.values():
            semantic_labels = rec.get("semantic_labels") if isinstance(rec.get("semantic_labels"), dict) else {}
            if semantic_labels:
                semantic_cells += 1
                labels.update(semantic_labels.keys())
        stats = data.setdefault("stats", {})
        stats["cell_count"] = len(cells)
        stats["semantic_cell_count"] = semantic_cells
        stats["label_count"] = len(labels)
        stats["track_count"] = len(data.get("track_index", {}) or {})
        if step is not None:
            stats["last_updated_step"] = int(step)
        stats["last_updated_at"] = now_iso()

    def update_from_observation(
        self,
        analysis: JsonDict,
        *,
        navigation_status: Optional[JsonDict] = None,
        step: int = 0,
        persist: bool = True,
    ) -> JsonDict:
        data = self.load()
        position_status = self.position_map.status()
        nav = dict(position_status)
        if isinstance(navigation_status, dict):
            nav.update(navigation_status)
        self._sync_position_layer(data, position_status)
        for rec in data.get("cells", {}).values():
            self._decay_cell(rec, step=int(step))
        analysis_update_count = self._fuse_current_analysis(data, analysis or {}, navigation_status=nav, step=int(step))
        track_update_count = self._fuse_object_memory(data, step=int(step))
        self._refresh_frontier_scores(data, position_status)
        self._refresh_stats(data, step=int(step))
        if persist:
            self.save(data)
        result = self.status_from_data(data)
        result.update({
            "result_type": "semantic_map_updated",
            "step": int(step),
            "analysis_update_count": analysis_update_count,
            "track_update_count": track_update_count,
        })
        return result

    def sync_position_layer(self, *, persist: bool = True) -> JsonDict:
        data = self.load()
        position_status = self.position_map.status()
        self._sync_position_layer(data, position_status)
        self._refresh_frontier_scores(data, position_status)
        self._refresh_stats(data)
        if persist:
            self.save(data)
        return self.status_from_data(data)

    def status(self) -> JsonDict:
        data = self.load()
        position_status = self.position_map.status()
        self._sync_position_layer(data, position_status)
        self._refresh_frontier_scores(data, position_status)
        self._refresh_stats(data)
        return self.status_from_data(data)

    def status_from_data(self, data: JsonDict) -> JsonDict:
        return {
            "status": "success",
            "result_type": "semantic_map_status",
            "room_name": data.get("room_name"),
            "map_frame": data.get("map_frame", {}),
            "cells": data.get("cells", {}),
            "track_index": data.get("track_index", {}),
            "frontier_scores": data.get("frontier_scores", {}),
            "stats": data.get("stats", {}),
        }


def parse_json_arg(value: Optional[str]) -> JsonDict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("JSON argument must be an object")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage semantic-map.json over position-map.json.")
    parser.add_argument("command", choices=["reset", "status", "sync-position", "update"])
    parser.add_argument("--memory-dir", default=None)
    parser.add_argument("--room", default="current_room")
    parser.add_argument("--analysis-json", default="{}")
    parser.add_argument("--navigation-json", default="{}")
    parser.add_argument("--step", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    semantic_map = SemanticMap(Path(args.memory_dir) if args.memory_dir else None)
    if args.command == "reset":
        result = semantic_map.reset(room_name=args.room)
    elif args.command == "sync-position":
        result = semantic_map.sync_position_layer()
    elif args.command == "update":
        result = semantic_map.update_from_observation(
            parse_json_arg(args.analysis_json),
            navigation_status=parse_json_arg(args.navigation_json),
            step=args.step,
        )
    else:
        result = semantic_map.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
