---
name: heartbeat-watchdog
description: OpenClaw heartbeat 触发时做空闲感知，并在发现可清扫地面目标时通过 patrol-runner skill 启动后台巡视。
metadata:
  {
    "openclaw":
      {
        "emoji": "heartbeat",
        "requires": { "bins": ["python3"] }
      }
  }
---

# Heartbeat Watchdog

V2.2 note: pass `--task-mode tidy` to let heartbeat start the background
runner when YOLO detects household service candidates such as pickup targets.
The default `clean` mode still starts only on cleanable floor targets.

本 skill 是 robot-cleaner 的主动触发层。

它不直接移动机器人，也不直接清扫。它只在 OpenClaw heartbeat 触发时做低频检查：

```text
heartbeat-watchdog
-> 检查 patrol-runner 是否正在运行
-> 空闲时做一次轻量感知和结构化分析
-> 如果发现可清扫地面目标，调用 patrol-runner skill 的 --command start 启动后台 runner，并返回启动通知
-> 如果上一轮后台巡视已有可汇报结果，调用 patrol-runner skill 的 --command report，并返回用户可见总结
-> runner 正在运行时只观察和记录，不打扰用户
```

## 使用场景

当 OpenClaw heartbeat 被触发时，优先执行：

```bash
python C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\heartbeat-watchdog\scripts\heartbeat_watchdog.py
```

## Runtime JSON Contract

成功返回字段：

- `status = "success"`
- `result_type`
- `runner_alive`
- `action_taken`
- `message`
- `state_mode`

可能的 `result_type`：

- `heartbeat_runner_alive`：runner 正在运行，不做动作
- `heartbeat_target_found_started`：空闲感知发现可清扫地面目标，已调用 `patrol-runner` skill 的 `--command start` 启动后台巡视
- `heartbeat_target_found_start_failed`：空闲感知发现目标，但启动后台巡视失败
- `heartbeat_room_report_ready`：上一轮后台巡视已有可汇报结果，已调用 `patrol-runner --command report` 生成总结
- `heartbeat_room_report_failed`：上一轮后台巡视可能有结果，但 report 命令失败
- `heartbeat_idle_no_task`：空闲感知未发现任务
- `heartbeat_idle_no_scan`：空闲但本次未执行感知
- `heartbeat_state_observed_no_action`：观察到非空闲任务状态，但不恢复、不汇报
- `heartbeat_error` / `heartbeat_timeout`：watchdog 自身异常

可能出现的用户通知字段：

- `should_notify_user = true`：OpenClaw Agent 应把 `user_message` 发到聊天框
- `notify_user = true`：与其他 runner skill 的通知字段保持一致，表示本次结果应通知用户
- `user_message`：用于“主动发现目标并启动 runner”或“后台巡视已有结果”等状态变化
- `runner_start`：当 `result_type = heartbeat_target_found_started` 时出现，包含 `patrol-runner --command start` 的返回结果
- `runner_report`：当 `result_type = heartbeat_room_report_ready` 时出现，包含 `patrol-runner --command report` 的返回结果

## Agent Rule

OpenClaw heartbeat 触发时：

1. 调用本 skill。
2. 如果返回 `should_notify_user = true` 或 `notify_user = true`，优先把 `user_message` 原样或略微润色后发给用户。
3. 如果返回 `heartbeat_runner_alive`、`heartbeat_idle_no_task`、`heartbeat_idle_no_scan`、`heartbeat_state_observed_no_action`，回复 `HEARTBEAT_OK`。
4. 如果返回 `heartbeat_target_found_started`，把 `user_message` 发到聊天框，说明后台 runner 已启动；不要在本 heartbeat 回合继续等待完整巡视结束。
5. 如果返回 `heartbeat_room_report_ready`，把 `user_message` 发到聊天框；该文本来自 `patrol-runner --command report`。

不要在 heartbeat 回合中手工循环调用移动或清扫 skill。
不要在 heartbeat 回合中使用 `patrol-runner --command segment` 做周期续跑；正式路径是启动 continuous runner。
不要在 heartbeat 回合中使用长时间阻塞的 `patrol-runner --command run`，避免 exec completed 事件反复唤醒主会话并再次触发 heartbeat。
