import { describe, expect, it } from "vitest";
import entry from "./index.js";
import { getToolPluginMetadata } from "openclaw/plugin-sdk/tool-plugin";

describe("robot-cleaner-tools", () => {
  it("declares tool metadata", () => {
    expect(getToolPluginMetadata(entry)?.tools.map((tool) => tool.name)).toEqual([
      "robot_cleaner_prepare_decision_turn",
      "robot_cleaner_execute_option",
      "robot_cleaner_status",
      "robot_cleaner_report",
      "robot_cleaner_stop",
    ]);
  });
});
