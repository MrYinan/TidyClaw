# HEARTBEAT.md - Heartbeat Policy

heartbeat 是低频主动触发层，只负责空闲时检查是否需要启动当前房间 tidy 服务巡视。它不直接移动、拾取、放置或清扫。

## OpenClaw Binding

OpenClaw heartbeat 读取本文件。触发时优先执行：

```bash
D:\Anaconda\envs\robot\python.exe skills/heartbeat-watchdog/scripts/heartbeat_watchdog.py --task-mode tidy --launch-mode agent-request --timeout 60
```

必须运行上面的 heartbeat-watchdog 命令。不能只调用 robot_cleaner_status 或 patrol-runner status 后自行判断 heartbeat_idle_no_task。只有 heartbeat-watchdog.py 的 JSON 返回 result_type=heartbeat_idle_no_task 时，才允许回复 NO_REPLY。

## Behavior

- `heartbeat_runner_alive`：回复 `NO_REPLY`
- `heartbeat_idle_no_task`：回复 `NO_REPLY`
- `heartbeat_idle_no_scan`：回复 `NO_REPLY`
- `heartbeat_state_observed_no_action`：回复 `NO_REPLY`
- `heartbeat_tidy_agent_run_requested`：请求启动/交接给非 heartbeat 的 `tidy-room-agent` 任务
- `heartbeat_target_found_started`：转发 `user_message`
- `heartbeat_target_found_start_failed`：转发 `user_message`
- `heartbeat_room_report_ready`：转发 `user_message`
- `heartbeat_room_report_failed`：转发 `user_message`

只有返回值里 `should_notify_user=true` 且 `notify_user=true` 时，才向主会话输出 `user_message`。否则保持安静。

当返回 `heartbeat_tidy_agent_run_requested` 时，heartbeat 回合只做交接：把返回值里的 `agent_run_request` 交给 OpenClaw 的非 heartbeat agent/taskflow 去启动 `tidy-room-agent`。

heartbeat 回合本身不要调用 `robot_cleaner_prepare_decision_turn()`，不要调用 `robot_cleaner_execute_option()`，不要在当前 heartbeat 消息里手写 tidy 循环。真正的整理循环只能由 `tidy-room-agent` 会话执行。

## Role Boundary

heartbeat 只做：

1. 检查是否已有整理任务运行。
2. 空闲时低频感知当前房间。
3. 发现服务整理目标时请求 OpenClaw 启动/交接给 `tidy-room-agent`。
4. 兼容模式下可启动后台 `patrol-runner`，但这不是默认主线。
5. 发现 report ready 时转发报告。

heartbeat 不做：

- 直接调用 `move-robot`
- 直接调用 `pick-object`
- 直接调用 `place-object`
- 直接调用 `clean-garbage`
- 循环推进固定步数
- 绕过 `tidy-room-agent` / `robot_cleaner_*` 工具契约

正式单轮 OpenClaw 决策仍走 `robot_cleaner_prepare_decision_turn()` / `robot_cleaner_execute_option()`。
