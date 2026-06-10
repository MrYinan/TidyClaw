#!/usr/bin/env python3
"""Placement viewpoint planner for tidy-mode household manipulation.

This module sits between object memory and navigation memory.

ObjectMemory answers: "which receptacle should I search for?"
PositionMap answers: "which grid cells look free / occupied / uncertain?"
This planner answers: "from which robot viewpoint should I observe that receptacle
so RGB-D surface reasoning has a fair chance to produce a place-ready region?"

The planner is deliberately online-safe. It consumes only sanitized object-memory
payloads, position-map cells, structured perception output, and action feedback.
It does not read AI2-THOR raw metadata.

Long-term design goals:
- Maintain several standoff viewpoints for every remembered receptacle.
- Limit scans at one viewpoint; never rotate forever at a bad viewpoint.
- Cool down exhausted viewpoints and try alternatives.
- Use current visual context when action-odometry confidence is low.
- Downgrade old map cells from "precise target" to "coarse region" when pose
  uncertainty is high.
- Persist planner state for interrupted / resumed patrol segments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = REPO_ROOT / "memory"
PLACEMENT_VIEWPOINT_STATE_PATH = MEMORY_DIR / "placement-viewpoint-state.json"
POSITION_MAP_PATH = MEMORY_DIR / "position-map.json"

SCHEMA_VERSION = 1
HEADING_ORDER = ["north", "east", "south", "west"]
HEADING_VECTORS = {
    "north": (0, 1),
    "east": (1, 0),
    "south": (0, -1),
    "west": (-1, 0),
}
TRANSLATION_ACTIONS = {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight"}
ROTATE_ACTIONS = {"RotateLeft", "RotateRight"}
LOOK_ACTIONS = {"LookUp", "LookDown"}
MOVE_ACTIONS = TRANSLATION_ACTIONS | ROTATE_ACTIONS | LOOK_ACTIONS
BLOCKED_CELL_STATES = {"occupied", "inflated_occupied"}

JsonDict = Dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def env_bool(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def parse_cell(cell: Any) -> Tuple[int, int]:
    left, right = str(cell or "0,0").split(",", 1)
    return int(left), int(right)


def format_cell(x: int, z: int) -> str:
    return f"{int(x)},{int(z)}"


def manhattan_distance(left: Any, right: Any) -> int:
    lx, lz = parse_cell(left)
    rx, rz = parse_cell(right)
    return abs(lx - rx) + abs(lz - rz)


def heading_between(source: str, target: str, fallback: str = "north") -> str:
    sx, sz = parse_cell(source)
    tx, tz = parse_cell(target)
    dx, dz = tx - sx, tz - sz
    if abs(dx) >= abs(dz) and dx != 0:
        return "east" if dx > 0 else "west"
    if dz != 0:
        return "north" if dz > 0 else "south"
    return fallback if fallback in HEADING_ORDER else "north"


def left_heading(heading: str) -> str:
    heading = heading if heading in HEADING_ORDER else "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) - 1) % 4]


def right_heading(heading: str) -> str:
    heading = heading if heading in HEADING_ORDER else "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) + 1) % 4]


def opposite_heading(heading: str) -> str:
    heading = heading if heading in HEADING_ORDER else "north"
    return HEADING_ORDER[(HEADING_ORDER.index(heading) + 2) % 4]


def neighbor_cell(cell: str, heading: str) -> str:
    x, z = parse_cell(cell)
    dx, dz = HEADING_VECTORS.get(heading, (0, 1))
    return format_cell(x + dx, z + dz)


def unique_strings(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_json(path: Path, default: JsonDict) -> JsonDict:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


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


def pose_trust_from_navigation(navigation_status: Optional[JsonDict]) -> JsonDict:
    nav = navigation_status if isinstance(navigation_status, dict) else {}
    try:
        confidence = float(nav.get("pose_confidence", 1.0) or 1.0)
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        uncertainty = float(nav.get("position_uncertainty_cells", 0.0) or 0.0)
    except (TypeError, ValueError):
        uncertainty = 0.0
    try:
        heading_confidence = float(nav.get("heading_confidence", 1.0) or 1.0)
    except (TypeError, ValueError):
        heading_confidence = 1.0

    low_conf = env_float("ROBOT_VIEWPOINT_LOW_POSE_CONFIDENCE", 0.45)
    low_uncertainty = env_float("ROBOT_VIEWPOINT_HIGH_UNCERTAINTY_CELLS", 3.0)
    medium_conf = env_float("ROBOT_VIEWPOINT_MEDIUM_POSE_CONFIDENCE", 0.68)
    medium_uncertainty = env_float("ROBOT_VIEWPOINT_MEDIUM_UNCERTAINTY_CELLS", 1.75)

    if confidence < low_conf or uncertainty >= low_uncertainty:
        level = "low"
    elif confidence < medium_conf or uncertainty >= medium_uncertainty:
        level = "medium"
    else:
        level = "high"
    return {
        "level": level,
        "pose_confidence": round(clamp(confidence, 0.0, 1.0), 4),
        "position_uncertainty_cells": round(max(0.0, uncertainty), 4),
        "heading_confidence": round(clamp(heading_confidence, 0.0, 1.0), 4),
        "precise_viewpoint_allowed": level == "high",
        "region_only": level == "low",
    }


@dataclass
class ViewpointDecision:
    status: str
    reason: str
    track_id: Optional[str] = None
    viewpoint_id: Optional[str] = None
    target_cell: Optional[str] = None
    target_heading: Optional[str] = None
    action: Optional[str] = None
    pose_trust: Optional[JsonDict] = None
    viewpoint: Optional[JsonDict] = None
    exhausted_viewpoint_id: Optional[str] = None
    planner_mode: str = "placement_viewpoint_v1"

    def as_dict(self) -> JsonDict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}


def default_state() -> JsonDict:
    return {
        "schema_version": SCHEMA_VERSION,
        "active_track_id": None,
        "tracks": {},
        "history": [],
        "stats": {
            "plan_count": 0,
            "viewpoint_exhausted_count": 0,
            "surface_ready_count": 0,
            "place_success_count": 0,
            "last_update": now_iso(),
        },
    }


class PlacementViewpointPlanner:
    """Persistent standoff viewpoint planner for receptacle surface reacquisition."""

    def __init__(self, memory_dir: Optional[Path] = None) -> None:
        self.memory_dir = Path(memory_dir) if memory_dir else MEMORY_DIR
        self.state_path = self.memory_dir / "placement-viewpoint-state.json"
        self.position_map_path = self.memory_dir / "position-map.json"
        self.scan_limit = max(1, env_int("ROBOT_PLACEMENT_VIEWPOINT_SCAN_LIMIT", 3))
        self.cooldown_steps = max(1, env_int("ROBOT_PLACEMENT_VIEWPOINT_COOLDOWN_STEPS", 10))
        self.max_viewpoints_per_track = max(4, env_int("ROBOT_PLACEMENT_VIEWPOINT_MAX_CANDIDATES", 16))
        self.standoff_min_m = env_float("ROBOT_PLACEMENT_VIEWPOINT_STANDOFF_MIN_M", 0.75)
        self.standoff_max_m = env_float("ROBOT_PLACEMENT_VIEWPOINT_STANDOFF_MAX_M", 1.25)
        # Disabled by default to keep RGB-D projection and map fusion on a
        # stable camera horizon. Manual move-robot LookUp / LookDown calls stay
        # supported, and future experiments may opt in explicitly.
        self.allow_autonomous_camera_pitch = env_bool("ROBOT_AUTONOMOUS_CAMERA_PITCH_ENABLED", False)

    def load(self) -> JsonDict:
        state = load_json(self.state_path, default_state())
        if not isinstance(state.get("tracks"), dict):
            state["tracks"] = {}
        if not isinstance(state.get("history"), list):
            state["history"] = []
        if not isinstance(state.get("stats"), dict):
            state["stats"] = default_state()["stats"]
        return state

    def save(self, state: JsonDict) -> None:
        state.setdefault("stats", {})["last_update"] = now_iso()
        atomic_write_json(self.state_path, state)

    def reset(self, *, reason: str = "reset") -> JsonDict:
        state = default_state()
        state["history"].append({"time": now_iso(), "event": "reset", "reason": reason})
        self.save(state)
        return state

    def status(self) -> JsonDict:
        state = self.load()
        return {
            "status": "success",
            "result_type": "placement_viewpoint_status",
            "active_track_id": state.get("active_track_id"),
            "track_count": len(state.get("tracks", {}) or {}),
            "stats": state.get("stats", {}),
        }

    def _position_map(self) -> JsonDict:
        return load_json(self.position_map_path, {})

    def _cell_size(self, navigation_status: Optional[JsonDict], position_map: JsonDict) -> float:
        nav = navigation_status if isinstance(navigation_status, dict) else {}
        frame = position_map.get("map_frame") if isinstance(position_map.get("map_frame"), dict) else {}
        try:
            return float(nav.get("cell_size") or nav.get("cell_size_m") or frame.get("cell_size_m") or 0.25)
        except (TypeError, ValueError):
            return 0.25

    def _cell_record(self, position_map: JsonDict, cell: str) -> JsonDict:
        cells = position_map.get("cells") if isinstance(position_map.get("cells"), dict) else {}
        rec = cells.get(str(cell))
        return rec if isinstance(rec, dict) else {}

    def _cell_traversable(self, position_map: JsonDict, cell: str) -> bool:
        state = str(self._cell_record(position_map, cell).get("state") or "unknown")
        return state not in BLOCKED_CELL_STATES

    def _viewpoint_id(self, track_id: str, cell: str, heading: str) -> str:
        return f"{track_id}|{cell}|{heading}"

    def _track_state(self, state: JsonDict, track_id: str, target: JsonDict) -> JsonDict:
        tracks = state.setdefault("tracks", {})
        track_state = tracks.get(track_id)
        if not isinstance(track_state, dict):
            track_state = {
                "track_id": track_id,
                "label": target.get("label"),
                "active_viewpoint_id": None,
                "candidate_viewpoints": {},
                "status": "searching",
                "last_target_snapshot": {},
                "last_plan": {},
            }
            tracks[track_id] = track_state
        if not isinstance(track_state.get("candidate_viewpoints"), dict):
            track_state["candidate_viewpoints"] = {}
        track_state["label"] = target.get("label")
        track_state["last_target_snapshot"] = {
            "track_id": target.get("track_id"),
            "label": target.get("label"),
            "recommended_view_cell": target.get("recommended_view_cell"),
            "recommended_heading": target.get("recommended_heading"),
            "estimated_object_cell": target.get("estimated_object_cell"),
            "last_seen_step": target.get("last_seen_step"),
            "staleness": target.get("staleness"),
        }
        return track_state

    def _candidate_offsets(self, radius_cells: int) -> List[Tuple[int, int]]:
        offsets = [
            (0, -radius_cells),
            (radius_cells, 0),
            (0, radius_cells),
            (-radius_cells, 0),
        ]
        if radius_cells >= 3:
            diag = max(1, int(round(radius_cells * 0.70)))
            offsets.extend([(diag, diag), (diag, -diag), (-diag, diag), (-diag, -diag)])
        return offsets

    def _generate_candidates(
        self,
        *,
        target: JsonDict,
        current_cell: str,
        current_heading: str,
        navigation_status: Optional[JsonDict],
        pose_trust: JsonDict,
        position_map: JsonDict,
        step: int,
    ) -> List[JsonDict]:
        track_id = str(target.get("track_id") or "receptacle:unknown")
        estimated_object_cell = str(target.get("estimated_object_cell") or target.get("goal_cell") or current_cell)
        recommended_cell = str(target.get("recommended_view_cell") or "")
        recommended_heading = str(target.get("recommended_heading") or current_heading)
        cell_size = self._cell_size(navigation_status, position_map)
        min_radius = max(2, int(round(self.standoff_min_m / max(1e-6, cell_size))))
        max_radius = max(min_radius, int(round(self.standoff_max_m / max(1e-6, cell_size))))

        raw: List[Tuple[str, str, str]] = []
        # Old ObjectMemory viewpoint is valuable only while odometry is still trusted.
        if recommended_cell and not pose_trust.get("region_only"):
            raw.append((recommended_cell, recommended_heading, "object_memory_recommended"))

        try:
            ox, oz = parse_cell(estimated_object_cell)
        except Exception:
            ox, oz = parse_cell(current_cell)
            estimated_object_cell = current_cell
        for radius in range(min_radius, max_radius + 1):
            for dx, dz in self._candidate_offsets(radius):
                cell = format_cell(ox + dx, oz + dz)
                heading = heading_between(cell, estimated_object_cell, fallback=recommended_heading)
                raw.append((cell, heading, "estimated_object_standoff_ring"))

        # With low trust, treat old object cells as a coarse region and actively sample
        # nearby free cells around the current pose. Current visual context should guide
        # the exact direction later in the runner.
        if pose_trust.get("region_only"):
            raw.append((current_cell, current_heading, "low_trust_local_reacquisition"))
            for heading in HEADING_ORDER:
                cell = neighbor_cell(current_cell, heading)
                raw.append((cell, heading_between(cell, estimated_object_cell, fallback=current_heading), "low_trust_local_neighbor"))

        seen = set()
        candidates: List[JsonDict] = []
        for cell, heading, source in raw:
            key = (cell, heading)
            if key in seen:
                continue
            seen.add(key)
            if not self._cell_traversable(position_map, cell):
                continue
            rec = self._cell_record(position_map, cell)
            cell_state = str(rec.get("state") or "unknown")
            visited = bool(rec.get("visited", False))
            distance = manhattan_distance(current_cell, cell)
            score = 0.0
            if source == "object_memory_recommended":
                score += 1.0 if pose_trust.get("precise_viewpoint_allowed") else 0.25
            if source == "low_trust_local_reacquisition":
                score += 0.70
            elif source == "low_trust_local_neighbor":
                score += 0.55
            if cell_state == "free":
                score += 0.75
            elif cell_state == "unknown":
                score -= 0.25
            if visited:
                score += 0.20
            score -= 0.055 * float(distance)
            candidates.append(
                {
                    "viewpoint_id": self._viewpoint_id(track_id, cell, heading),
                    "cell": cell,
                    "heading": heading,
                    "source": source,
                    "score": round(score, 4),
                    "cell_state": cell_state,
                    "visited": visited,
                    "distance_cells": distance,
                    "status": "untried",
                    "scan_count": 0,
                    "failure_count": 0,
                    "cooldown_until_step": 0,
                    "last_failure_reason": None,
                    "last_tried_step": None,
                    "created_step": int(step),
                }
            )
        candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return candidates[: self.max_viewpoints_per_track]

    def _merge_candidates(self, track_state: JsonDict, incoming: List[JsonDict]) -> None:
        existing = track_state.setdefault("candidate_viewpoints", {})
        for candidate in incoming:
            viewpoint_id = str(candidate.get("viewpoint_id"))
            old = existing.get(viewpoint_id)
            if isinstance(old, dict):
                preserved = {
                    "status": old.get("status", "untried"),
                    "scan_count": int(old.get("scan_count", 0) or 0),
                    "failure_count": int(old.get("failure_count", 0) or 0),
                    "cooldown_until_step": int(old.get("cooldown_until_step", 0) or 0),
                    "last_failure_reason": old.get("last_failure_reason"),
                    "last_tried_step": old.get("last_tried_step"),
                }
                old.update(candidate)
                old.update(preserved)
            else:
                existing[viewpoint_id] = candidate

    def _usable_viewpoints(self, track_state: JsonDict, *, step: int) -> List[JsonDict]:
        viewpoints = track_state.get("candidate_viewpoints") if isinstance(track_state.get("candidate_viewpoints"), dict) else {}
        usable: List[JsonDict] = []
        for vp in viewpoints.values():
            if not isinstance(vp, dict):
                continue
            cooldown = int(vp.get("cooldown_until_step", 0) or 0)
            status = str(vp.get("status") or "untried")
            if cooldown > int(step):
                continue
            if status == "completed":
                continue
            usable.append(vp)
        usable.sort(
            key=lambda item: (
                1 if str(item.get("status") or "") == "untried" else 0,
                float(item.get("score", 0.0) or 0.0) - 0.35 * int(item.get("failure_count", 0) or 0),
            ),
            reverse=True,
        )
        return usable

    def _select_active_viewpoint(self, track_state: JsonDict, *, step: int) -> Optional[JsonDict]:
        viewpoints = track_state.get("candidate_viewpoints") if isinstance(track_state.get("candidate_viewpoints"), dict) else {}
        active_id = str(track_state.get("active_viewpoint_id") or "")
        active = viewpoints.get(active_id)
        if isinstance(active, dict):
            cooldown = int(active.get("cooldown_until_step", 0) or 0)
            if cooldown <= int(step) and str(active.get("status") or "") not in {"exhausted", "completed"}:
                return active
        usable = self._usable_viewpoints(track_state, step=step)
        if not usable:
            track_state["active_viewpoint_id"] = None
            return None
        active = usable[0]
        active["status"] = "active"
        track_state["active_viewpoint_id"] = active.get("viewpoint_id")
        return active

    def _exhaust_viewpoint(self, state: JsonDict, track_state: JsonDict, viewpoint: JsonDict, *, step: int, reason: str) -> None:
        viewpoint["status"] = "exhausted"
        viewpoint["failure_count"] = int(viewpoint.get("failure_count", 0) or 0) + 1
        viewpoint["last_failure_reason"] = reason
        viewpoint["last_tried_step"] = int(step)
        viewpoint["cooldown_until_step"] = int(step) + self.cooldown_steps * min(4, int(viewpoint["failure_count"]))
        track_state["active_viewpoint_id"] = None
        stats = state.setdefault("stats", {})
        stats["viewpoint_exhausted_count"] = int(stats.get("viewpoint_exhausted_count", 0) or 0) + 1
        state.setdefault("history", []).append(
            {
                "time": now_iso(),
                "event": "viewpoint_exhausted",
                "step": int(step),
                "track_id": track_state.get("track_id"),
                "viewpoint_id": viewpoint.get("viewpoint_id"),
                "reason": reason,
                "cooldown_until_step": viewpoint.get("cooldown_until_step"),
            }
        )
        state["history"] = state["history"][-120:]

    def _scan_action(self, viewpoint: JsonDict, current_heading: str) -> str:
        scan_count = int(viewpoint.get("scan_count", 0) or 0)
        target_heading = str(viewpoint.get("heading") or current_heading)
        if current_heading != target_heading:
            if left_heading(current_heading) == target_heading:
                return "RotateLeft"
            return "RotateRight"
        # Once aligned, use body rotations by default. Camera pitch is opt-in
        # only, so ordinary tidy runs preserve a stable RGB-D projection frame.
        sequence = ["RotateRight", "RotateLeft"]
        if self.allow_autonomous_camera_pitch:
            sequence.extend(["LookDown", "LookUp"])
        return sequence[(max(1, scan_count) - 1) % len(sequence)]

    def _local_visual_action(self, *, context_candidate: Optional[JsonDict], dominant_rejection_reason: str) -> Optional[str]:
        candidate = context_candidate if isinstance(context_candidate, dict) else {}
        hint = str(candidate.get("position_hint") or "")
        if dominant_rejection_reason == "too_close":
            return "MoveBack"
        if dominant_rejection_reason == "too_far" and hint in {"front-center", ""}:
            return "MoveAhead"
        if hint == "front-left":
            return "MoveLeft"
        if hint == "front-right":
            return "MoveRight"
        if dominant_rejection_reason in {"surface_above_view", "too_high_in_image"}:
            return "LookUp" if self.allow_autonomous_camera_pitch else "MoveBack"
        if dominant_rejection_reason in {"surface_below_view", "too_low_in_image"}:
            return "LookDown" if self.allow_autonomous_camera_pitch else "MoveBack"
        if dominant_rejection_reason in {"touches_image_edge", "touches_parent_edge", "thin_region", "edge_only_region", "single_row_region"}:
            return "MoveBack"
        return None

    def plan(
        self,
        *,
        target: JsonDict,
        current_cell: str,
        current_heading: str,
        step: int,
        navigation_status: Optional[JsonDict],
        analysis: Optional[JsonDict] = None,
        context_candidate: Optional[JsonDict] = None,
        dominant_rejection_reason: str = "no_visual_ready_surface",
        persist: bool = True,
    ) -> JsonDict:
        track_id = str(target.get("track_id") or "")
        if not track_id:
            return ViewpointDecision(status="no_target", reason="missing_receptacle_track_id").as_dict()

        analysis = analysis if isinstance(analysis, dict) else {}
        pose_trust = pose_trust_from_navigation(navigation_status)
        position_map = self._position_map()
        state = self.load()
        state["active_track_id"] = track_id
        track_state = self._track_state(state, track_id, target)
        incoming = self._generate_candidates(
            target=target,
            current_cell=str(current_cell),
            current_heading=str(current_heading),
            navigation_status=navigation_status,
            pose_trust=pose_trust,
            position_map=position_map,
            step=int(step),
        )
        self._merge_candidates(track_state, incoming)

        stats = state.setdefault("stats", {})
        stats["plan_count"] = int(stats.get("plan_count", 0) or 0) + 1

        # At low odometry trust, a current visual receptacle box is more valuable
        # than a stale exact grid cell. Use it to reacquire a better camera view.
        local_action = self._local_visual_action(
            context_candidate=context_candidate,
            dominant_rejection_reason=str(dominant_rejection_reason or "no_visual_ready_surface"),
        )
        if local_action and (pose_trust.get("region_only") or isinstance(context_candidate, dict)):
            decision = ViewpointDecision(
                status="local_visual_reacquisition",
                reason=f"visual_context_adjustment:{dominant_rejection_reason}",
                track_id=track_id,
                action=local_action,
                pose_trust=pose_trust,
                planner_mode="local_visual_reacquisition",
            ).as_dict()
            track_state["last_plan"] = decision
            self.save(state) if persist else None
            return decision

        active = self._select_active_viewpoint(track_state, step=int(step))
        if not isinstance(active, dict):
            decision = ViewpointDecision(
                status="no_usable_viewpoint",
                reason="all_viewpoints_exhausted_or_blocked",
                track_id=track_id,
                pose_trust=pose_trust,
            ).as_dict()
            track_state["status"] = "exhausted"
            track_state["last_plan"] = decision
            self.save(state) if persist else None
            return decision

        active["last_tried_step"] = int(step)
        if str(current_cell) != str(active.get("cell")):
            decision = ViewpointDecision(
                status="navigate_to_viewpoint",
                reason="placement_viewpoint_selected",
                track_id=track_id,
                viewpoint_id=str(active.get("viewpoint_id")),
                target_cell=str(active.get("cell")),
                target_heading=str(active.get("heading")),
                pose_trust=pose_trust,
                viewpoint=dict(active),
            ).as_dict()
            track_state["last_plan"] = decision
            self.save(state) if persist else None
            return decision

        active["scan_count"] = int(active.get("scan_count", 0) or 0) + 1
        if int(active["scan_count"]) <= self.scan_limit:
            action = self._scan_action(active, str(current_heading))
            decision = ViewpointDecision(
                status="scan_viewpoint",
                reason=f"viewpoint_reached_scan:{active['scan_count']}/{self.scan_limit}",
                track_id=track_id,
                viewpoint_id=str(active.get("viewpoint_id")),
                target_cell=str(active.get("cell")),
                target_heading=str(active.get("heading")),
                action=action,
                pose_trust=pose_trust,
                viewpoint=dict(active),
            ).as_dict()
            track_state["last_plan"] = decision
            self.save(state) if persist else None
            return decision

        exhausted_id = str(active.get("viewpoint_id"))
        self._exhaust_viewpoint(
            state,
            track_state,
            active,
            step=int(step),
            reason=f"no_visual_ready_surface_after_{self.scan_limit}_scans:{dominant_rejection_reason}",
        )
        next_viewpoint = self._select_active_viewpoint(track_state, step=int(step))
        if isinstance(next_viewpoint, dict):
            decision = ViewpointDecision(
                status="switch_viewpoint",
                reason="previous_viewpoint_exhausted_select_alternative",
                track_id=track_id,
                viewpoint_id=str(next_viewpoint.get("viewpoint_id")),
                target_cell=str(next_viewpoint.get("cell")),
                target_heading=str(next_viewpoint.get("heading")),
                pose_trust=pose_trust,
                viewpoint=dict(next_viewpoint),
                exhausted_viewpoint_id=exhausted_id,
            ).as_dict()
            track_state["last_plan"] = decision
            self.save(state) if persist else None
            return decision

        decision = ViewpointDecision(
            status="no_usable_viewpoint",
            reason="all_viewpoints_exhausted_after_scan",
            track_id=track_id,
            exhausted_viewpoint_id=exhausted_id,
            pose_trust=pose_trust,
        ).as_dict()
        track_state["status"] = "exhausted"
        track_state["last_plan"] = decision
        self.save(state) if persist else None
        return decision

    def mark_surface_ready(self, *, track_id: Optional[str], step: int, surface_candidate_id: Optional[str] = None) -> None:
        if not track_id:
            return
        state = self.load()
        track_state = (state.get("tracks") or {}).get(str(track_id))
        if not isinstance(track_state, dict):
            return
        track_state["status"] = "surface_ready"
        track_state["surface_ready_step"] = int(step)
        track_state["surface_candidate_id"] = surface_candidate_id
        stats = state.setdefault("stats", {})
        stats["surface_ready_count"] = int(stats.get("surface_ready_count", 0) or 0) + 1
        self.save(state)

    def mark_place_success(self, *, track_id: Optional[str], step: int) -> None:
        state = self.load()
        if track_id:
            track_state = (state.get("tracks") or {}).get(str(track_id))
            if isinstance(track_state, dict):
                track_state["status"] = "completed"
                track_state["completed_step"] = int(step)
                active_id = str(track_state.get("active_viewpoint_id") or "")
                active = (track_state.get("candidate_viewpoints") or {}).get(active_id)
                if isinstance(active, dict):
                    active["status"] = "completed"
        state["active_track_id"] = None
        stats = state.setdefault("stats", {})
        stats["place_success_count"] = int(stats.get("place_success_count", 0) or 0) + 1
        self.save(state)

    def record_move_result(
        self,
        *,
        track_id: Optional[str],
        viewpoint_id: Optional[str],
        action: str,
        success: bool,
        step: int,
        failure_reason: Optional[str] = None,
    ) -> None:
        if not track_id or not viewpoint_id:
            return
        state = self.load()
        track_state = (state.get("tracks") or {}).get(str(track_id))
        if not isinstance(track_state, dict):
            return
        viewpoint = (track_state.get("candidate_viewpoints") or {}).get(str(viewpoint_id))
        if not isinstance(viewpoint, dict):
            return
        viewpoint["last_action"] = str(action)
        viewpoint["last_action_success"] = bool(success)
        viewpoint["last_tried_step"] = int(step)
        if not success:
            self._exhaust_viewpoint(
                state,
                track_state,
                viewpoint,
                step=int(step),
                reason=f"navigation_action_failed:{failure_reason or action}",
            )
        self.save(state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage persistent placement viewpoints.")
    parser.add_argument("--memory-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    reset = sub.add_parser("reset")
    reset.add_argument("--reason", default="manual_reset")
    plan = sub.add_parser("plan")
    plan.add_argument("--target-json", required=True)
    plan.add_argument("--current-cell", required=True)
    plan.add_argument("--current-heading", required=True)
    plan.add_argument("--step", required=True, type=int)
    plan.add_argument("--navigation-json", default="{}")
    plan.add_argument("--analysis-json", default="{}")
    plan.add_argument("--context-json", default="{}")
    plan.add_argument("--dominant-rejection-reason", default="no_visual_ready_surface")
    return parser


def parse_json_arg(text: str) -> JsonDict:
    value = json.loads(text or "{}")
    if not isinstance(value, dict):
        raise ValueError("JSON argument must be an object")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manager = PlacementViewpointPlanner(Path(args.memory_dir) if args.memory_dir else None)
    if args.command == "status":
        result = manager.status()
    elif args.command == "reset":
        result = manager.reset(reason=args.reason)
    elif args.command == "plan":
        result = manager.plan(
            target=parse_json_arg(args.target_json),
            current_cell=args.current_cell,
            current_heading=args.current_heading,
            step=int(args.step),
            navigation_status=parse_json_arg(args.navigation_json),
            analysis=parse_json_arg(args.analysis_json),
            context_candidate=parse_json_arg(args.context_json),
            dominant_rejection_reason=args.dominant_rejection_reason,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
