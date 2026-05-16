#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for collecting the AI2-THOR service YOLO dataset.

This workspace runs AI2-THOR through `back/robot_server.py`, so the practical
collector reuses the running backend and writes RGB images plus YOLO labels from
offline `/eval/state` metadata.
"""

from __future__ import annotations

from collect_ai2thor_service_dataset_backend import main


if __name__ == "__main__":
    raise SystemExit(main())
