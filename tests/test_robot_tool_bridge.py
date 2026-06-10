import unittest

from scripts.robot_tool_bridge import ToolRequestError, command_for_request


class RobotToolBridgeTests(unittest.TestCase):
    def test_prepare_endpoint_maps_to_fixed_script(self) -> None:
        command = command_for_request("/tools/prepare-decision-turn", {})

        self.assertEqual(command.tool_name, "robot_cleaner_prepare_decision_turn")
        self.assertEqual(command.script.name, "prepare_decision_turn.py")
        self.assertIn("--task-mode", command.args)
        self.assertIn("tidy", command.args)
        self.assertIn("memory/decision-context.json", command.args)

    def test_execute_endpoint_maps_only_option_id(self) -> None:
        command = command_for_request("/tools/execute-option", {"option_id": "move:moveahead"})

        self.assertEqual(command.tool_name, "robot_cleaner_execute_option")
        self.assertEqual(command.script.name, "execute_option.py")
        self.assertIn("--option-id", command.args)
        self.assertIn("move:moveahead", command.args)
        self.assertNotIn("script_path", command.args)

    def test_execute_endpoint_rejects_multiline_option_id(self) -> None:
        with self.assertRaises(ToolRequestError):
            command_for_request("/tools/execute-option", {"option_id": "move:moveahead\nscript=bad"})

    def test_status_endpoint_maps_to_fixed_script(self) -> None:
        command = command_for_request("/tools/status", {})

        self.assertEqual(command.tool_name, "robot_cleaner_status")
        self.assertEqual(command.script.name, "robot_status.py")
        self.assertEqual(command.args, ["--format", "compact"])

    def test_report_endpoint_maps_to_fixed_script(self) -> None:
        command = command_for_request("/tools/report", {})

        self.assertEqual(command.tool_name, "robot_cleaner_report")
        self.assertEqual(command.script.name, "robot_report.py")
        self.assertEqual(command.args, ["--format", "compact"])

    def test_stop_endpoint_maps_only_reason(self) -> None:
        command = command_for_request("/tools/stop", {"reason": "operator_check"})

        self.assertEqual(command.tool_name, "robot_cleaner_stop")
        self.assertEqual(command.script.name, "robot_stop.py")
        self.assertEqual(command.args, ["--reason", "operator_check", "--format", "compact"])
        self.assertNotIn("script_path", command.args)

    def test_unknown_endpoint_rejected(self) -> None:
        with self.assertRaises(ToolRequestError):
            command_for_request("/tools/not-real", {})


if __name__ == "__main__":
    unittest.main()
