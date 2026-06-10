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
export declare function callRobotToolBackend({ toolName, endpoint, body, config, signal, }: RobotToolRequest): Promise<unknown>;
