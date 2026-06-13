#!/usr/bin/env python3
"""Runtime configuration for robot-cleaner tool scripts.

Project configuration is the stable default. Environment variables remain an
explicit operator override for experiments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "robot_cleaner_runtime.json"
RUNTIME_CONFIG_SCHEMA = "robot_cleaner_runtime_config_v1"

JsonDict = dict[str, Any]


def as_dict(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_workspace_path(value: Any, *, base: Path = REPO_ROOT) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = base / path
    return path


def normalize_backend(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or "action_odometry"


def load_runtime_config(path: Path | str = CONFIG_PATH) -> JsonDict:
    config = read_json(Path(path))
    if not config:
        return {"schema": RUNTIME_CONFIG_SCHEMA, "map": {"backend": "action_odometry"}}
    config.setdefault("schema", RUNTIME_CONFIG_SCHEMA)
    config.setdefault("map", {})
    return config


def configured_map_backend(config: JsonDict | None = None) -> JsonDict:
    config = config if isinstance(config, dict) else load_runtime_config()
    map_config = as_dict(config.get("map"))
    backend = normalize_backend(map_config.get("backend") or "action_odometry")
    bundle_path_text = str(map_config.get("bundle_path") or "memory/maps/current")
    bundle_path = resolve_workspace_path(bundle_path_text)
    fallback_backend = normalize_backend(map_config.get("fallback_backend") or "action_odometry")
    fallback_enabled = bool(map_config.get("fallback_if_bundle_missing", True))
    effective_backend = backend
    fallback_reason = None
    if backend in {"map_bundle", "bundle", "frozen_map", "built_map"}:
        snapshot_path = bundle_path if bundle_path.suffix.lower() == ".json" else bundle_path / "map-snapshot.json"
        if fallback_enabled and not snapshot_path.exists():
            effective_backend = fallback_backend
            fallback_reason = "map_bundle_missing"
    return {
        "schema": "robot_cleaner_runtime_map_backend_v1",
        "configured_backend": backend,
        "backend": effective_backend,
        "bundle_path": str(bundle_path),
        "bundle_path_display": display_path(bundle_path),
        "fallback_backend": fallback_backend,
        "fallback_reason": fallback_reason,
        "config_path": display_path(CONFIG_PATH),
    }


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def apply_runtime_environment(config: JsonDict | None = None) -> JsonDict:
    """Apply runtime config to process environment unless explicitly overridden."""

    selection = configured_map_backend(config)
    previous_env = {
        "ROBOT_MAP_BACKEND": os.environ.get("ROBOT_MAP_BACKEND"),
        "ROBOT_MAP_BUNDLE_PATH": os.environ.get("ROBOT_MAP_BUNDLE_PATH"),
    }
    env_overrides: JsonDict = {}
    if not os.getenv("ROBOT_MAP_BACKEND"):
        os.environ["ROBOT_MAP_BACKEND"] = str(selection["backend"])
        env_overrides["ROBOT_MAP_BACKEND"] = str(selection["backend"])
    if not os.getenv("ROBOT_MAP_BUNDLE_PATH"):
        os.environ["ROBOT_MAP_BUNDLE_PATH"] = str(selection["bundle_path"])
        env_overrides["ROBOT_MAP_BUNDLE_PATH"] = str(selection["bundle_path"])
    return {
        "status": "success",
        "result_type": "runtime_environment_applied",
        "map_backend": selection,
        "env_applied": env_overrides,
        "previous_env": previous_env,
        "env_existing": {
            "ROBOT_MAP_BACKEND": os.getenv("ROBOT_MAP_BACKEND"),
            "ROBOT_MAP_BUNDLE_PATH": os.getenv("ROBOT_MAP_BUNDLE_PATH"),
        },
    }


def restore_runtime_environment(previous_env: JsonDict) -> None:
    for key in ("ROBOT_MAP_BACKEND", "ROBOT_MAP_BUNDLE_PATH"):
        value = previous_env.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)


if __name__ == "__main__":
    print(json.dumps(apply_runtime_environment(), ensure_ascii=False, indent=2), flush=True)
