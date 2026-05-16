#!/usr/bin/env python3
"""
OpenClaw skill entry point for the robot-cleaner state manager.

The implementation lives in scripts/state_manager_core.py so both Python code
and this skill can share the same state logic.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.state_manager_core import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
