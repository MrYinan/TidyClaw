from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.object_memory_core import ObjectMemory
from scripts.position_map_core import PositionMap
from scripts.semantic_mapping_core import SemanticMap


def apple_candidate() -> dict:
    return {
        "label": "apple",
        "raw_label": "Apple",
        "task_semantic_class": "pickup_target",
        "confidence": 0.91,
        "bearing_deg": 0.0,
        "ground_distance_m": 0.5,
        "distance_m": 0.5,
        "position_hint": "front-center",
        "bbox": {"x": 280, "y": 510, "w": 30, "h": 35},
    }


def grid_surface_candidate() -> dict:
    return {
        "id": "pc_grid:counter_top:test",
        "surface_candidate_id": "pc_grid:counter_top:test",
        "label": "pc_surface",
        "raw_label": "CounterTop",
        "parent_label": "counter_top",
        "parent_object": "counter_top",
        "source": "pointcloud_plane_grid_completion",
        "surface_candidate_source": "pointcloud_plane_grid_completion",
        "task_semantic_class": "place_receptacle",
        "confidence": 0.88,
        "score": 0.88,
        "bearing_deg": 35.0,
        "ground_distance_m": 1.0,
        "distance_m": 1.0,
        "position_hint": "front-right",
        "visual_place_ready": True,
        "is_support_surface": True,
        "bbox": {"x": 40, "y": 320, "w": 120, "h": 100},
    }


class SemanticMappingCoreTests(unittest.TestCase):
    def test_semantic_layer_fuses_current_analysis_and_object_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            object_memory = ObjectMemory(memory_dir)
            object_memory.update_from_observation(
                {"best_pickup_candidate": apple_candidate()},
                current_cell="0,0",
                heading="north",
                step=1,
                navigation_status=position.status(),
            )
            semantic = SemanticMap(memory_dir)
            result = semantic.update_from_observation(
                {
                    "best_pickup_candidate": apple_candidate(),
                    "best_surface_candidate": grid_surface_candidate(),
                },
                navigation_status=position.status(),
                step=2,
            )
            self.assertGreater(result["stats"]["semantic_cell_count"], 0)
            self.assertGreater(result["analysis_update_count"], 0)
            self.assertGreater(result["track_update_count"], 0)
            labels = {
                label
                for cell in result["cells"].values()
                for label in cell.get("semantic_labels", {}).keys()
            }
            self.assertIn("apple", labels)
            self.assertIn("counter_top", labels)
            surface_scores = [
                float(cell.get("task_scores", {}).get("surface_score", 0.0))
                for cell in result["cells"].values()
            ]
            self.assertGreater(max(surface_scores), 0.0)

    def test_semantic_map_preserves_position_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp)
            position = PositionMap(memory_dir)
            position.reset()
            data = position.load()
            position.mark_occupied(data, "0,1", evidence="test")
            position.save(data)
            semantic = SemanticMap(memory_dir)
            result = semantic.sync_position_layer()
            self.assertEqual(result["cells"]["0,1"]["position_state"], "occupied")


if __name__ == "__main__":
    unittest.main()
