"""Offline evaluation/debug adapter for privileged AI2-THOR state.

The online Agent must not call this adapter. It exists for controlled
experiments, debugging, dataset annotation and metric computation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


JsonDict = Dict[str, Any]


class EvalAdapter:
    """Expose AI2-THOR privileged state for offline evaluation only."""

    SCHEMA_VERSION = 2

    def __init__(self, env: Any) -> None:
        self.env = env

    def get_eval_state(self) -> JsonDict:
        snapshot = self.env.get_state_snapshot()
        if not isinstance(snapshot, dict):
            snapshot = {"status": "error", "result_type": "invalid_eval_snapshot"}
        snapshot = dict(snapshot)
        snapshot.setdefault("status", "success")
        snapshot.update(
            {
                "schema_version": self.SCHEMA_VERSION,
                "result_type": snapshot.get("result_type") or "eval_state_snapshot",
                "usage_scope": "offline_evaluation_debug_only",
                "online_safe": False,
                "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "warning": "Do not feed this endpoint into online Agent decisions.",
            }
        )
        return snapshot
