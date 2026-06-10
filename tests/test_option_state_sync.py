import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.option_state_sync import sync_option_result


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".test-tmp"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_memory_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_active_state(memory_dir: Path, *, step_count: int) -> None:
    write_json(
        memory_dir / "patrol-state.json",
        {
            "enabled": True,
            "mode": "SERVICE",
            "step_count": step_count,
            "max_steps": 100,
        },
    )
    write_json(
        memory_dir / "mission-state.json",
        {
            "enabled": True,
            "mode": "SERVICE",
            "current_room": "current_room",
            "total_steps_completed": step_count,
            "max_steps": 100,
        },
    )
    write_json(
        memory_dir / "room-state.json",
        {
            "room_name": "current_room",
            "room_complete": False,
            "explored_steps": step_count,
        },
    )


class OptionStateSyncTests(unittest.TestCase):
    def test_pickup_success_sets_held_object_context(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        memory_dir = make_memory_dir("pickup")
        self.addCleanup(shutil.rmtree, memory_dir, ignore_errors=True)
        write_active_state(memory_dir, step_count=2)
        write_json(
            memory_dir / "object-memory.json",
            {
                "schema_version": 1,
                "tracks": {
                    "objtrk:apple:0001": {
                        "track_id": "objtrk:apple:0001",
                        "label": "apple",
                        "task_class": "pickup_target",
                        "status": "unpicked",
                        "interaction": {},
                    }
                },
            },
        )

        result = sync_option_result(
            context={"generated_at": "2026-06-09T10:00:00+08:00"},
            option={"kind": "service_action", "action": "pick-object"},
            candidate={
                "candidate_id": "apple-front",
                "track_id": "objtrk:apple:0001",
                "label": "apple",
                "raw_label": "Apple",
            },
            execution={"status": "success", "result_type": "pickup_executed", "holding_object": True},
            success=True,
            memory_dir=memory_dir,
        )

        self.assertEqual(result["status"], "success")
        state = json.loads((memory_dir / "service-task-state.json").read_text(encoding="utf-8"))
        self.assertTrue(state["holding_object"])
        self.assertEqual(state["phase"], "SEARCH_RECEPTACLE")
        self.assertEqual(state["held_object_track_id"], "objtrk:apple:0001")
        self.assertEqual(state["held_object_family"], "food")
        memory = json.loads((memory_dir / "object-memory.json").read_text(encoding="utf-8"))
        self.assertEqual(memory["tracks"]["objtrk:apple:0001"]["status"], "picked")

    def test_place_success_clears_held_context_and_records_completion(self) -> None:
        TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        memory_dir = make_memory_dir("place")
        self.addCleanup(shutil.rmtree, memory_dir, ignore_errors=True)
        write_active_state(memory_dir, step_count=3)
        write_json(memory_dir / "place-precheck-cache.json", {"schema": "test"})
        write_json(
            memory_dir / "service-task-state.json",
            {
                "phase": "SEARCH_RECEPTACLE",
                "subgoal_index": 1,
                "target_label": "apple",
                "target_raw_label": "Apple",
                "target_track_id": "objtrk:apple:0001",
                "holding_object": True,
                "held_object_label": "apple",
                "held_object_raw_label": "Apple",
                "held_object_family": "food",
                "held_object_labels": ["apple", "Apple"],
                "held_object_track_id": "objtrk:apple:0001",
                "completed_subgoals": [],
                "recently_placed_tracks": {},
                "history": [],
            },
        )
        write_json(
            memory_dir / "object-memory.json",
            {
                "schema_version": 1,
                "tracks": {
                    "objtrk:apple:0001": {
                        "track_id": "objtrk:apple:0001",
                        "label": "apple",
                        "task_class": "pickup_target",
                        "status": "picked",
                        "last_observation": {"candidate_signature": "apple-sig"},
                        "estimated_location": {"estimated_object_cell": "0,0"},
                        "interaction": {},
                    }
                },
            },
        )

        result = sync_option_result(
            context={"generated_at": "2026-06-09T10:01:00+08:00"},
            option={"kind": "service_action", "action": "place-object"},
            candidate={
                "candidate_id": "counter-surface",
                "label": "pc_surface",
                "raw_label": "CounterTop",
                "task_semantic_class": "place_receptacle",
            },
            execution={"status": "success", "result_type": "place_executed", "holding_object": False},
            success=True,
            memory_dir=memory_dir,
        )

        self.assertEqual(result["status"], "success")
        state = json.loads((memory_dir / "service-task-state.json").read_text(encoding="utf-8"))
        self.assertFalse(state["holding_object"])
        self.assertEqual(state["phase"], "SEARCH_PICKUP_TARGET")
        self.assertIsNone(state["held_object_track_id"])
        self.assertEqual(state["completed_subgoals"][-1]["summary"], "Apple->CounterTop")
        self.assertIn("objtrk:apple:0001", state["recently_placed_tracks"])
        self.assertFalse((memory_dir / "place-precheck-cache.json").exists())

        mission = json.loads((memory_dir / "mission-state.json").read_text(encoding="utf-8"))
        room = json.loads((memory_dir / "room-state.json").read_text(encoding="utf-8"))
        self.assertIn("Apple", mission["objects_placed"])
        self.assertIn("Apple->CounterTop", room["service_tasks_completed"])
        memory = json.loads((memory_dir / "object-memory.json").read_text(encoding="utf-8"))
        self.assertEqual(memory["tracks"]["objtrk:apple:0001"]["status"], "placed")


if __name__ == "__main__":
    unittest.main()
