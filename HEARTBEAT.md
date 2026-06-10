# HEARTBEAT.md - 家庭服务机器人 heartbeat policy

heartbeat 是家庭服务机器人（原项目名 `robot-cleaner`）的**低频主动触发层**。

它不负责一步步移动机器人，也不负责把任务拆成短执行段续跑。正式循环巡视交给 `patrol-runner` continuous 执行。heartbeat 主动触发时只做秒级检查：空闲时感知环境；发现服务整理目标后，用 `patrol-runner` skill 的 `--command start` 启动后台 `tidy` runner 并立即返回；后续 heartbeat 或用户询问状态时，再用 `patrol-runner --command report/status` 汇报结果。

## OpenClaw Binding

OpenClaw heartbeat 会周期性触发一个 agent turn，并读取工作区 `HEARTBEAT.md`。

每次 heartbeat 触发时，优先执行：

```bash
python skills/heartbeat-watchdog/scripts/heartbeat_watchdog.py --task-mode tidy
```

根据返回值处理：

- `heartbeat_runner_alive`：runner 正在运行，回复 `NO_REPLY`
- `heartbeat_idle_no_task`：空闲且未发现服务整理目标，回复 `NO_REPLY`
- `heartbeat_idle_no_scan`：空闲但未执行感知，回复 `NO_REPLY`
- `heartbeat_target_found_started`：主动发现服务整理目标，并已通过 `patrol-runner --command start` 启动后台 tidy 巡视
- `heartbeat_target_found_start_failed`：主动发现服务整理目标，但后台 tidy 巡视启动失败
- `heartbeat_room_report_ready`：发现上一轮巡视已有可汇报结果，并已通过 `patrol-runner --command report` 生成用户可见总结
- `heartbeat_room_report_failed`：发现上一轮巡视可能有结果，但 report 命令失败
- `heartbeat_state_observed_no_action`：非空闲任务状态已被观察到，但 heartbeat 不汇报、不恢复；回复 `NO_REPLY`

如果返回值中存在：

```json
{"should_notify_user": true, "notify_user": true, "user_message": "..."}
```

OpenClaw Agent 应优先把 `user_message` 作为本次 heartbeat 的用户可见回复。无该字段或其值为 false 时，默认不打扰用户，按上面的 `NO_REPLY` 规则处理。

## Current Architecture

当前主动任务规划分两层：

```text
heartbeat-watchdog
  -> 低频检查 runner / 空闲环境
  -> 空闲且发现服务整理目标时启动 patrol-runner

patrol-runner
  -> continuous 循环执行 tidy 巡视
  -> get-vision -> perceive-scene-yolo -> decide -> move/pick/place/clean-fallback -> state-manager
  -> status/report 返回任务摘要
```

因此：

- heartbeat 负责“空闲时要不要主动启动服务整理任务”
- patrol-runner 负责“任务启动后如何持续执行”
- patrol-runner 负责“任务完成后如何生成总结”
- state-manager 负责“三份 memory JSON 的一致性”

## Heartbeat Role

heartbeat 只负责：

1. 检查 `patrol-runner` 是否仍在运行
2. 空闲时低频执行一次轻量感知与结构化分析
3. 如果发现服务整理目标，启动 `patrol-runner`
4. 在无事发生时不产生用户可见输出（`NO_REPLY`）

heartbeat 不负责：

- 直接调用 `move-robot`
- 直接调用 `pick-object`
- 直接调用 `place-object`
- 直接调用 `clean-garbage`
- 循环调用单步 skills
- 每次 heartbeat 推进固定 3 步
- 绕过 `patrol-runner` 或 `state-manager`
- 生成房间完成汇报
- 恢复异常退出的 runner

## Active Trigger Rule

当 heartbeat 触发且系统处于 `IDLE` 时：

1. `heartbeat-watchdog` 可执行一次 `get-vision`
2. 继续调用 YOLO 结构化感知
3. 如果结构化分析发现服务整理目标，watchdog 调用 `patrol-runner` skill 入口启动后台 runner

服务整理目标包括：

- `pickup_target_detected = true`
- `direct_pickup_detected = true`
- `service_candidates` 中存在可整理物体或可放置目标
- 兼容的地面清扫目标可作为 tidy 的 fallback，但不再是 heartbeat 的默认基线

启动命令应带 `--task-mode tidy`：

```bash
python skills/patrol-runner/scripts/patrol_runner_skill.py --command start --max-steps 200 --task-mode tidy
```

4. watchdog 返回 `heartbeat_target_found_started`，OpenClaw Agent 只把 `user_message` 发给用户，例如“已启动当前房间服务整理巡视，后台 runner 正在执行”
5. 如果没有发现服务整理目标，保持 `IDLE` 并记录 heartbeat 日志

说明：heartbeat-watchdog 是主动触发 skill，`patrol-runner` 是正式执行 skill。heartbeat 回合不得使用长时间阻塞的 `--command run`，避免 runner 完成事件反复唤醒主会话并再次触发 heartbeat。watchdog 只通过 skill 入口调用 detached `--command start`，不直接调用 `scripts/patrol_runner.py`。

上一轮任务完成并向用户汇报后，必须通过 `patrol-runner --command report` 的默认 finalize 行为，或直接调用 `state-manager finalize-report`，将三份当前状态 JSON 复位到 `IDLE`。否则 heartbeat 会看到 `MISSION_REPORT` / `room_complete=true`，把系统判断为非空闲，从而不执行下一次空闲主动感知。

## Reporting Rule

heartbeat 回合默认不打扰用户。它只在状态发生变化时给用户反馈。

### 主会话隔离原则

heartbeat 对主会话的干扰应当最小化：

| heartbeat 结果 | 行为 | 对主会话的影响 |
|---|---|---|
| `heartbeat_idle_no_task` | `NO_REPLY` | 零输出 |
| `heartbeat_idle_no_scan` | `NO_REPLY` | 零输出 |
| `heartbeat_runner_alive` | `NO_REPLY` | 零输出 |
| `heartbeat_state_observed_no_action` | `NO_REPLY` | 零输出 |
| `heartbeat_target_found_started` | 推送 `user_message` | 一条通知 |
| `heartbeat_target_found_start_failed` | 推送 `user_message` | 一条通知 |
| `heartbeat_room_report_ready` | 推送 `user_message` | 一条通知 |
| `heartbeat_room_report_failed` | 推送 `user_message` | 一条通知 |
| 用户主动询问 heartbeat 状态 | 正常回复 | 用户发起 |

**核心规则：** 仅当 `should_notify_user=true` 且 `notify_user=true` 时，才向主会话推送 `user_message`。否则一律 `NO_REPLY`。

房间完成、停止、RECOVER 的自然语言总结仍由 `patrol-runner --command report` 生成。heartbeat 只是在后续触发时发现 `report_ready=true`，调用 report 并把 `user_message` 转发给用户。用户主动询问结果时，也应调用：

```bash
python skills/patrol-runner/scripts/patrol_runner_skill.py --command report
```

## Debug Segment Rule

`patrol-runner` 仍保留 `--command segment`，但它只用于调试或测试模式。

正式 heartbeat 路径不得使用短执行段续跑。不要再采用：

```text
heartbeat -> segment 3 steps -> next heartbeat -> segment 3 steps
```

正式路径必须是：

```text
heartbeat -> heartbeat-watchdog --task-mode tidy -> patrol-runner skill --command start -> 立即返回启动通知
下次 heartbeat 或用户询问 -> patrol-runner skill --command report/status -> 聊天框汇报
用户主动启动正式巡视 -> OpenClaw 调用 patrol-runner skill --command start 或 run
```

## Boundaries

当前 heartbeat 版本只服务于：

- 当前房间
- 单房间自动服务整理巡视
- 空闲状态下的低频主动感知与任务触发

当前不要在 heartbeat 中加入：

- 多房间调度
- 电量决策
- 复杂物体重排
- 平台系统巡检
- 直接移动
- 直接拾取
- 直接放置
- 直接清扫
