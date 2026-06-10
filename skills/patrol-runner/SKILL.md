---
name: patrol-runner
description: 启动、继续、查看或停止单房间家庭服务巡视 runner。
disable-model-invocation: true
metadata: {"openclaw":{"emoji":"patrol","requires":{"bins":["python3"]}}}
---

# 家庭服务巡视 Runner

该 skill 是 OpenClaw 调用 `scripts/patrol_runner.py` 的正式入口。

它会把已有技能串成一个闭环：

```text
get-vision -> perceive/analyze scene -> local RGB-D costmap -> semantic/A* navigation -> decide -> move/pick/place/clean -> verify -> navigation-memory -> state-manager
```

也就是：

```text
先感知 -> 再结构化分析 -> 再决策 -> 执行一个物理动作 -> 验证结果 -> 更新导航记忆和任务状态
```

## 什么时候使用

用户提出以下需求时，使用该 skill：

- 当前房间服务巡视
- 自动 tidy 整理模式
- 家庭物体拾取/放置任务
- 连续房间检查
- 兼容旧版清扫巡视

不要把它用于单次手动移动、单次拍照或单张图片分析。那些场景应分别使用 `move-robot`、`get-vision` 或感知 skill。

## 模式


家庭服务整理模式：

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --max-steps 200 --task-mode tidy
```

后台启动服务巡视：

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command start --max-steps 200 --task-mode tidy
```

查看状态或报告：

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command status
python skills\patrol-runner\scripts\patrol_runner_skill.py --command report
```

停止巡视：

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command stop --reason user_stop
```

## tidy 语义

`tidy` 不是一次性的“捡起并放下”动作，而是一个可以连续完成多个服务子任务的房间巡视模式。

一个服务子任务的流程是：

```text
寻找可拾取目标
-> 对齐/靠近
-> 拾取
-> 寻找可放置目标
-> 对齐/靠近
-> 放置
-> 记录 objects_placed 和 service_tasks_completed
-> 继续巡视
```

重要规则：

- `place-object` 成功只表示一个整理子任务完成，不表示房间完成。
- runner 会继续运行，直到满足房间完成、达到 max-step、recover failed 或用户停止条件。
- 刚放置成功的物体会被短期抑制，避免机器人马上又把它捡起来。
- tidy 默认拾取策略是 `--pickup-surface-policy floor-only`。
- 桌面、台面、架子等 elevated 候选，只有在明确调试 `--pickup-surface-policy any-surface` 时才进入 pickup 子任务。

## 运行返回

`--command run` 成功返回：

- `status = success`
- `result_type = patrol_runner_run_finished`
- `runner_returncode`
- `events_tail`
- `report`
- `report_ready`
- `notify_user`
- `user_message`

`--command status` 成功返回：

- 后台 runner 进程状态
- state-manager 校验结果
- 是否应该继续运行
- 当前报告
- 可用时包含 service task 进度

## Agent 规则

当用户要求连续巡视、自动整理或当前房间服务检查时，应调用该 skill，而不是在聊天上下文里手工循环调用多个单步技能。

短段调试可以直接调用底层 runner：

```powershell
python scripts\patrol_runner.py --start --segment-steps 20 --max-steps 80 --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden --verbose
```

不要把一个短段执行结果汇报成房间完成，除非状态明确满足：

```text
room_complete = true
```


## Local RGB-D navigation safety

`patrol_runner.py` updates `memory/navigation-costmap.json` before every movement decision.
The costmap masks held-object overlay depth, expands the footprint while carrying an object,
and validates `MoveAhead`, `MoveBack`, `MoveLeft`, `MoveRight`, `RotateLeft`, and `RotateRight`.
`LookUp` / `LookDown` remain camera-only active-perception actions.
