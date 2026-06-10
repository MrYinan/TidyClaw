import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.robot_report import build_robot_report
from scripts.robot_status import build_robot_status
from scripts.robot_stop import request_stop


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = WORKSPACE_ROOT / "tests" / "_tmp_robot_status_report_stop"


def fresh_memory(name: str) -> Path:
    path = TEST_TMP_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_memory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def seed_memory(memory: Path) -> None:
    write_json(
        memory / "patrol-state.json",
        {
            "schema_version": 1,
            "enabled": True,
            "mode": "SERVICE",
            "step_count": 7,
            "max_steps": 80,
            "last_action": "MoveAhead",
        },
    )
    write_json(
        memory / "mission-state.json",
        {
            "schema_version": 1,
            "enabled": True,
            "mode": "SERVICE",
            "current_room": "current_room",
            "total_steps_completed": 7,
            "max_steps": 80,
            "objects_detected": ["Apple", "Mug"],
            "objects_placed": ["Apple"],
            "service_tasks_completed": ["Apple->CounterTop"],
            "last_result": "service_completed: Apple->CounterTop",
        },
    )
    write_json(
        memory / "room-state.json",
        {
            "schema_version": 1,
            "room_name": "current_room",
            "explored_steps": 7,
            "objects_placed": ["Apple"],
            "service_tasks_completed": ["Apple->CounterTop"],
            "targets_cleaned": [],
            "room_complete": False,
            "last_cell": "0,-2",
            "last_heading": "south",
            "coverage_estimate": 0.25,
            "frontier_cells": ["0,-3"],
            "collision_count": 0,
        },
    )
    write_json(
        memory / "service-task-state.json",
        {
            "schema_version": 1,
            "phase": "SEARCH_PICKUP_TARGET",
            "holding_object": False,
            "completed_subgoals": [{"summary": "Apple->CounterTop"}],
            "pickup_surface_policy": "floor-only",
        },
    )
    write_json(
        memory / "decision-context.json",
        {
            "status": "success",
            "schema": "robot_cleaner_decision_context_v1",
            "generated_at": "2026-06-10T00:00:00+08:00",
            "option_set": {
                "rule_baseline_option_id": "move:moveahead",
                "options": [{"option_id": "move:moveahead"}],
            },
        },
    )
    write_json(
        memory / "yolo-current-rgbd.json",
        {
            "status": "success",
            "result_type": "scene_analyzed_yolo",
            "perception_backend": "yolo",
            "candidate_count": 0,
            "frontier_exists": True,
            "recommended_action": "MoveAhead",
        },
    )


class RobotStatusReportStopTests(unittest.TestCase):
    @patch("scripts.robot_tool_state.runner_process_summary")
    def test_status_returns_bounded_summary(self, runner_summary) -> None:
        runner_summary.return_value = {"pid": None, "alive": False}
        memory = fresh_memory("status")
        try:
            seed_memory(memory)

            status = build_robot_status(memory)
        finally:
            cleanup_memory(memory)

        self.assertEqual(status["status"], "success")
        self.assertEqual(status["task"]["step_count"], 7)
        self.assertEqual(status["service"]["phase"], "SEARCH_PICKUP_TARGET")
        self.assertEqual(status["progress"]["objects_placed_count"], 1)
        self.assertTrue(status["decision_context"]["model_decision_required"])
        self.assertEqual(status["decision_context"]["rule_baseline_option_id"], "move:moveahead")

    @patch("scripts.robot_tool_state.runner_process_summary")
    def test_report_describes_subgoal_progress_without_room_complete(self, runner_summary) -> None:
        runner_summary.return_value = {"pid": None, "alive": False}
        memory = fresh_memory("report")
        try:
            seed_memory(memory)

            report = build_robot_report(memory)
        finally:
            cleanup_memory(memory)

        self.assertEqual(report["status"], "success")
        self.assertEqual(report["report_type"], "subgoal_progress")
        self.assertIn("已放置 1 个物体", report["user_message"])

    @patch("scripts.robot_stop.runner_process_summary")
    @patch("scripts.robot_tool_state.runner_process_summary")
    def test_stop_marks_mission_done(self, state_runner_summary, stop_runner_summary) -> None:
        state_runner_summary.return_value = {"pid": None, "alive": False}
        stop_runner_summary.return_value = {"pid": None, "alive": False}
        memory = fresh_memory("stop")
        try:
            seed_memory(memory)

            result = request_stop("unit_test_stop", memory_dir=memory)

            mission = json.loads((memory / "mission-state.json").read_text(encoding="utf-8"))
            patrol = json.loads((memory / "patrol-state.json").read_text(encoding="utf-8"))
        finally:
            cleanup_memory(memory)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stop_state"], "stopped")
        self.assertFalse(mission["enabled"])
        self.assertEqual(mission["mode"], "DONE")
        self.assertFalse(patrol["enabled"])
        self.assertEqual(patrol["mode"], "DONE")


if __name__ == "__main__":
    unittest.main()
