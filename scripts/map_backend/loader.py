#!/usr/bin/env python3
"""Map backend selection."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from scripts.map_backend.action_odometry import ActionOdometryMapBackend
    from scripts.map_backend.ai2thor_groundtruth import AI2ThorGroundTruthMapBackend
    from scripts.map_backend.base import BackendUnavailableError, MapBackend
    from scripts.map_backend.map_bundle import MapBundleBackend
except ImportError:  # pragma: no cover - direct script execution
    from map_backend.action_odometry import ActionOdometryMapBackend
    from map_backend.ai2thor_groundtruth import AI2ThorGroundTruthMapBackend
    from map_backend.base import BackendUnavailableError, MapBackend
    from map_backend.map_bundle import MapBundleBackend


def normalize_backend_name(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or "action_odometry"


def load_map_backend(memory_dir: Path | str | None = None, *, backend: str | None = None) -> MapBackend:
    name = normalize_backend_name(backend or os.getenv("ROBOT_MAP_BACKEND"))
    if name in {"action_odometry", "action_odom", "position_map", "default"}:
        return ActionOdometryMapBackend(memory_dir)
    if name in {"ai2thor_groundtruth", "ai2thor", "ground_truth", "groundtruth", "sim_groundtruth"}:
        return AI2ThorGroundTruthMapBackend(memory_dir)
    if name in {"map_bundle", "bundle", "frozen_map", "built_map"}:
        return MapBundleBackend(memory_dir)
    raise BackendUnavailableError(f"Unsupported ROBOT_MAP_BACKEND: {name}")
