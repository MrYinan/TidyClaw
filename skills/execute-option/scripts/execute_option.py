#!/usr/bin/env python3
"""OpenClaw skill wrapper for scripts/execute_option.py."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.execute_option import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
