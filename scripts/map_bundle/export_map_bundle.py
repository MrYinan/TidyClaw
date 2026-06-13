#!/usr/bin/env python3
"""Export a stable map bundle from the configured MapBackend."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.map_backend import BackendUnavailableError, load_map_backend
except ImportError:  # pragma: no cover - direct script execution
    from map_backend import BackendUnavailableError, load_map_backend


MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_OUTPUT_DIR = MEMORY_DIR / "maps" / "current"
MAP_BUNDLE_SCHEMA = "robot_cleaner_map_bundle_v1"

JsonDict = dict[str, Any]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, data: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def read_json(path: Path) -> JsonDict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def copy_if_exists(source: Path, target: Path) -> JsonDict | None:
    if not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"source": display_path(source), "path": display_path(target)}


def export_map_bundle(
    *,
    memory_dir: Path | str = MEMORY_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    backend: str | None = None,
) -> JsonDict:
    memory = Path(memory_dir)
    output = Path(output_dir)
    map_backend = load_map_backend(memory, backend=backend)
    snapshot = map_backend.load_snapshot()

    snapshot_payload = snapshot.to_dict(include_raw=True)
    position_status = snapshot.to_position_status()
    room_state = snapshot.to_navigation_room_state()
    manifest: JsonDict = {
        "schema": MAP_BUNDLE_SCHEMA,
        "status": "success",
        "result_type": "map_bundle_exported",
        "generated_at": now_iso(),
        "backend": snapshot.backend,
        "map_snapshot_schema": snapshot.schema,
        "output_dir": display_path(output),
        "files": {
            "map_snapshot": "map-snapshot.json",
            "position_map": "position-map.json",
            "room_state": "room-state.json",
        },
        "map_summary": snapshot.public_summary(),
        "usage": {
            "runtime_backend": "map_bundle",
            "env": {
                "ROBOT_MAP_BACKEND": "map_bundle",
                "ROBOT_MAP_BUNDLE_PATH": display_path(output),
            },
        },
    }

    atomic_write_json(output / "map-snapshot.json", snapshot_payload)
    atomic_write_json(output / "position-map.json", position_status)
    atomic_write_json(output / "room-state.json", room_state)

    copied: JsonDict = {}
    for name in ("object-memory.json", "semantic-map.json"):
        result = copy_if_exists(memory / name, output / name)
        if result:
            copied[name] = result
    if copied:
        manifest["copied_runtime_files"] = copied

    raw_groundtruth = read_json(memory / "ai2thor-groundtruth-map.json")
    if raw_groundtruth:
        atomic_write_json(output / "ai2thor-groundtruth-map.json", raw_groundtruth)
        manifest["files"]["ai2thor_groundtruth_map"] = "ai2thor-groundtruth-map.json"

    atomic_write_json(output / "map-manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a robot-cleaner map bundle from a MapBackend.")
    parser.add_argument("--memory-dir", default=str(MEMORY_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backend", default=None, help="Override ROBOT_MAP_BACKEND for this export.")
    parser.add_argument("--format", choices=("pretty", "compact"), default="pretty")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_map_bundle(
            memory_dir=Path(args.memory_dir),
            output_dir=Path(args.output_dir),
            backend=args.backend,
        )
    except BackendUnavailableError as exc:
        result = {
            "status": "error",
            "result_type": "map_bundle_export_failed",
            "message": str(exc),
            "generated_at": now_iso(),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.format == "pretty" else None), flush=True)
        return 1
    if args.format == "compact":
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
