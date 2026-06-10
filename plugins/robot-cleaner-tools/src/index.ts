import { Type } from "typebox";
import { defineToolPlugin } from "openclaw/plugin-sdk/tool-plugin";
import { callRobotToolBackend } from "./python_runner.js";

const configSchema = Type.Object(
  {
    baseUrl: Type.Optional(
      Type.String({
        description: "Base URL for the local Robot Cleaner tool bridge service.",
      }),
    ),
    timeoutMs: Type.Optional(
      Type.Number({
        description: "Maximum backend script runtime in milliseconds.",
        minimum: 1000,
        maximum: 600000,
      }),
    ),
  },
  { additionalProperties: false },
);

export default defineToolPlugin({
  id: "robot-cleaner-tools",
  name: "Robot Cleaner Tools",
  description: "Stable OpenClaw tools for the household service robot agent.",
  configSchema,
  tools: (tool) => [
    tool({
      name: "robot_cleaner_prepare_decision_turn",
      label: "Prepare Robot Cleaner Decision Turn",
      description:
        "Refresh robot observation/perception and build the bounded decision context for one model decision turn.",
      parameters: Type.Object({}, { additionalProperties: false }),
      execute: async (_params, config, context) => {
        context.signal?.throwIfAborted();
        return callRobotToolBackend({
          toolName: "robot_cleaner_prepare_decision_turn",
          endpoint: "/tools/prepare-decision-turn",
          body: {},
          config,
          signal: context.signal,
        });
      },
    }),
    tool({
      name: "robot_cleaner_execute_option",
      label: "Execute Robot Cleaner Option",
      description:
        "Execute one option_id selected from the current robot_cleaner_decision_context_v1 after local validation.",
      parameters: Type.Object(
        {
          option_id: Type.String({
            description: "The selected option_id from decision_context.option_set.options.",
            minLength: 1,
          }),
        },
        { additionalProperties: false },
      ),
      execute: async ({ option_id }, config, context) => {
        context.signal?.throwIfAborted();
        assertSafeOptionId(option_id);
        return callRobotToolBackend({
          toolName: "robot_cleaner_execute_option",
          endpoint: "/tools/execute-option",
          body: { option_id },
          config,
          signal: context.signal,
        });
      },
    }),
    tool({
      name: "robot_cleaner_status",
      label: "Robot Cleaner Status",
      description:
        "Return the current robot service task state, patrol state, holding state, and recent progress without executing a physical action.",
      parameters: Type.Object({}, { additionalProperties: false }),
      execute: async (_params, config, context) => {
        context.signal?.throwIfAborted();
        return callRobotToolBackend({
          toolName: "robot_cleaner_status",
          endpoint: "/tools/status",
          body: {},
          config,
          signal: context.signal,
        });
      },
    }),
    tool({
      name: "robot_cleaner_report",
      label: "Robot Cleaner Report",
      description:
        "Return a user-facing structured report for the current household service robot run without executing a physical action.",
      parameters: Type.Object({}, { additionalProperties: false }),
      execute: async (_params, config, context) => {
        context.signal?.throwIfAborted();
        return callRobotToolBackend({
          toolName: "robot_cleaner_report",
          endpoint: "/tools/report",
          body: {},
          config,
          signal: context.signal,
        });
      },
    }),
    tool({
      name: "robot_cleaner_stop",
      label: "Stop Robot Cleaner",
      description: "Request a safe stop for the current household service robot run.",
      parameters: Type.Object(
        {
          reason: Type.Optional(
            Type.String({
              description: "Short stop reason recorded in robot state.",
              minLength: 1,
            }),
          ),
        },
        { additionalProperties: false },
      ),
      execute: async ({ reason }, config, context) => {
        context.signal?.throwIfAborted();
        const stopReason = sanitizeReason(reason);
        return callRobotToolBackend({
          toolName: "robot_cleaner_stop",
          endpoint: "/tools/stop",
          body: { reason: stopReason },
          config,
          signal: context.signal,
        });
      },
    }),
  ],
});

function assertSafeOptionId(optionId: string): void {
  if (optionId !== optionId.trim() || /[\r\n\t]/.test(optionId)) {
    throw new Error("option_id must be a single trimmed option id");
  }
}

function sanitizeReason(reason: string | undefined): string {
  const value = (reason || "user_stop").trim();
  if (!value || /[\r\n\t]/.test(value)) {
    return "user_stop";
  }
  return value.slice(0, 200);
}
