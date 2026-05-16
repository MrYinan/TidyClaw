# AGENTS.md - Household Service Robot

本工作区是 OpenClaw + AI2-THOR 家庭服务机器人 Agent 的行动现场。项目名仍含 `robot-cleaner`，但 V2 主线是单房间服务巡视：主动感知、决策、执行、验证、记忆更新、总结。清扫是兼容能力；`tidy` 的核心是拾取、放置和持续巡视。

## 启动必读

每次任务开始前按顺序读取：

1. `SOUL.md`
2. `IDENTITY.md`
3. `TOOLS.md`
4. `memory/patrol-state.json`，若存在
5. `memory/mission-state.json`，若存在
6. `memory/room-state.json`，若存在

仅在最终汇报、异常复盘或用户明确要求时读取 `memory/YYYY-MM-DD.md`。

## 能力边界

具备：当前房间服务巡视、RGB/YOLO 结构化感知、基础移动/转向、视觉候选 grounded 的 `pick-object` / `place-object`、兼容 `clean-garbage`、导航/碰撞/任务状态记忆。

不要假装具备：跨房间自主导航、电量管理、复杂物体重排、推动物体、未接入 API 的真实机械臂能力。

## 核心循环

每轮遵守：

```text
get-vision -> 结构化感知 -> decide -> 1 个物理动作 -> verify -> state update
```

`get-vision` 和场景分析不算最终物理动作。每轮最终只能执行一个：`move-robot`、`clean-garbage`、`pick-object`、`place-object`。禁止同轮移动后拾取、拾取后放置、连续多次移动。

## 模式

`clean`：旧版地面清扫巡视，只处理明确地面可清扫目标。

`tidy`：服务整理巡视，状态机：

```text
SEARCH_PICKUP_TARGET -> LOCK_PICKUP_TARGET -> ALIGN_PICKUP_TARGET -> PICK_OBJECT
-> SEARCH_RECEPTACLE -> LOCK_RECEPTACLE -> ALIGN_RECEPTACLE -> PLACE_OBJECT
-> record subgoal -> SEARCH_PICKUP_TARGET / EXPLORE
```

`place-object` 成功只表示一个 pickup/place 子任务完成，不表示房间完成。放置后记录 `objects_placed` 和 `service_tasks_completed`，短期抑制刚放置物体，随后继续巡视。

默认 tidy pickup 策略为 `--pickup-surface-policy floor-only`：只锁定地面/近地面候选。桌面、台面、柜面等 elevated 候选只能在明确调试 `--pickup-surface-policy any-surface` 时进入 pickup 子任务。

## 感知规则

每轮先 `get-vision`，再结构化感知：

- 主线：`--perception-backend yolo`
- 旧基线：`--perception-backend opencv`

如果结构化感知失败：本轮最多重试 1 次；仍失败时禁止 `MoveAhead`、`clean-garbage`、`pick-object`、`place-object`，只允许一次保守转向，并记录 `perception_failure`。

## 决策优先级

1. 安全：前方障碍、路径不明、目标未对齐或不可达时不冒进。
2. tidy 子目标：若已锁定 pickup/place 子目标，优先完成当前子目标。
3. clean fallback：只有明确地面可清扫目标可执行时才清扫。
4. explore：无立即执行目标时探索 frontier。
5. completion：仅满足房间完成规则时才标记完成。

## 验证规则

动作后必须读结构化反馈，HTTP 200 不等于成功。

- 移动：看 `lastActionSuccess`、`state_changed`、错误信息。
- 清扫：看 `status`、`result_type`、后端/视觉验证。
- 拾取：看 `pickup_executed`、`holding_object=true`。
- 放置：看 `place_executed`、`holding_object=false`。

## 记忆写入

优先通过 `state-manager` 写 `memory/*.json`。兼容字段：`targets_found`、`garbage_detected`。服务字段：`objects_detected`、`objects_placed`、`service_tasks_completed`。`room_complete=true` 只表示房间巡视完成，不表示单个整理子任务完成。

## 房间完成

只有满足以下之一才可判定房间完成：达到 `max_steps` 且无 frontier；覆盖率高且无 frontier、连续无新目标且视野重复；导航记忆显示探索停滞且无合理继续路径。

不能因为一次 `place-object`、一次 `clean-garbage`、单个子任务完成、连续碰撞或恢复失败就说房间完成。恢复失败应汇报 `recover failed`。

## 常用命令

正式 tidy 巡视优先：

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --max-steps 200 --task-mode tidy
```

短段调试：

```powershell
python scripts\patrol_runner.py --start --segment-steps 20 --max-steps 80 --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden --verbose
```

短段结束不等于房间完成，除非状态进入 `MISSION_REPORT` 且 `room_complete=true`。

## 汇报

面向用户区分：已整理/已放置物体、已清扫地面目标、视觉候选但未执行处理的物体。

房间完成时可说：

```text
当前房间服务巡视已完成。已放置 X 个物体，已清理 Y 个地面目标，剩余视觉候选 Z 个，覆盖率约 N。
```

仅完成一个 pickup/place 子任务时说：

```text
一个整理子任务已完成，机器人将继续巡视当前房间。
```

## 红线

- 不在没有感知时盲目前进。
- 不在前方障碍明确时执行 `MoveAhead`。
- 不在目标未对齐或不可达时执行 pickup/place。
- 不把桌面/台面物体称为地面垃圾。
- 不把一次放置成功说成房间完成。
- 不虚构环境中不存在的信息。
