from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


JsonDict = Dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_CONFIG = REPO_ROOT / "configs" / "scenarios_v2.json"


class ScenarioManager:
    """Load V2 benchmark scenarios from JSON config."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = Path(config_path or DEFAULT_SCENARIO_CONFIG)

    def load_config(self) -> JsonDict:
        if not self.config_path.exists():
            return {"schema_version": 2, "scenarios": []}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def list_scenarios(self) -> JsonDict:
        config = self.load_config()
        scenarios = config.get("scenarios", []) if isinstance(config.get("scenarios"), list) else []
        return {
            "status": "success",
            "schema_version": config.get("schema_version", 2),
            "result_type": "scenario_list",
            "config_path": str(self.config_path),
            "count": len(scenarios),
            "scenarios": [
                {
                    "name": item.get("name"),
                    "scene": item.get("scene"),
                    "mode": item.get("mode", "benchmark"),
                    "description": item.get("description", ""),
                    "tags": item.get("tags", []),
                }
                for item in scenarios
                if isinstance(item, dict)
            ],
        }

    def get_scenario(self, name: str) -> Optional[JsonDict]:
        config = self.load_config()
        for item in config.get("scenarios", []) or []:
            if isinstance(item, dict) and item.get("name") == name:
                scenario = dict(item)
                scenario["source"] = str(self.config_path)
                return scenario
        return None
