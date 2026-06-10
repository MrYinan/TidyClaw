export type RobotCleanerToolConfig = {
  baseUrl?: string;
  timeoutMs?: number;
};

export type RobotToolRequest = {
  toolName: string;
  endpoint: string;
  body?: Record<string, unknown>;
  config: RobotCleanerToolConfig;
  signal?: AbortSignal;
};

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";
const DEFAULT_TIMEOUT_MS = 120_000;

export async function callRobotToolBackend({
  toolName,
  endpoint,
  body = {},
  config,
  signal,
}: RobotToolRequest): Promise<unknown> {
  if (!endpoint.startsWith("/")) {
    return backendError(toolName, "invalid_endpoint", endpoint);
  }

  const baseUrl = normalizeBaseUrl(config.baseUrl);
  const timeoutMs = sanitizeTimeout(config.timeoutMs);
  const url = new URL(endpoint, baseUrl);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const abortHandler = () => controller.abort();

  signal?.addEventListener("abort", abortHandler);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    const text = await response.text();
    const parsed = parseJson(text);
    if (!response.ok) {
      return {
        status: "error",
        result_type: "robot_cleaner_backend_http_error",
        tool: toolName,
        http_status: response.status,
        response: parsed.ok ? parsed.value : undefined,
        response_tail: parsed.ok ? undefined : tail(text),
      };
    }

    if (parsed.ok) {
      return parsed.value;
    }

    return {
      status: "error",
      result_type: "robot_cleaner_backend_invalid_json",
      tool: toolName,
      response_tail: tail(text),
      parse_error: parsed.error,
    };
  } catch (error) {
    return {
      status: "error",
      result_type:
        error instanceof Error && error.name === "AbortError"
          ? "robot_cleaner_backend_timeout"
          : "robot_cleaner_backend_unreachable",
      tool: toolName,
      base_url: baseUrl,
      endpoint,
      timeout_ms: timeoutMs,
      message: error instanceof Error ? error.message : String(error),
      required_next: "start_robot_cleaner_tool_bridge",
    };
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abortHandler);
  }
}

function normalizeBaseUrl(value: string | undefined): string {
  const raw = (value || DEFAULT_BASE_URL).trim();
  return raw.endsWith("/") ? raw : `${raw}/`;
}

function sanitizeTimeout(value: number | undefined): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_TIMEOUT_MS;
  }
  return Math.max(1_000, Math.min(Math.trunc(value), 600_000));
}

function backendError(toolName: string, reason: string, endpoint: string): Record<string, unknown> {
  return {
    status: "error",
    result_type: "robot_cleaner_tool_backend_unavailable",
    tool: toolName,
    reason,
    endpoint,
    required_next: "fix_plugin_backend_mapping",
  };
}

function parseJson(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const trimmed = text.trim();
  if (!trimmed) {
    return { ok: false, error: "empty response body" };
  }

  try {
    return { ok: true, value: JSON.parse(trimmed) };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function tail(value: string): string {
  const maxChars = 4_000;
  if (value.length <= maxChars) {
    return value;
  }
  return value.slice(value.length - maxChars);
}
