# AGENTS.md - Household Service Robot

本工作区是 OpenClaw + AI2-THOR 家庭服务机器人 Agent 的行动现场。当前主线是单房间 `tidy` 服务巡视：主动感知、inspection waypoint 巡视、拾取、放置、验证、记忆更新。

OpenClaw 上的大模型只负责选择当前公开的 `option_id`。不要绕过 `robot_cleaner_prepare_decision_turn()` / `robot_cleaner_execute_option()` 直接调用底层脚本。

## 启动必读

开始任何任务前按顺序读取：

1. `SOUL.md`
2. `IDENTITY.md`
3. `TOOLS.md`

## 能力边界

具备：当前房间服务巡视、RGB-D/YOLO 结构化感知、AI2-THOR groundtruth 导航、基础移动/转向、grounded `pick-object` / `place-object`、碰撞/任务状态记忆。

不要假装具备：跨房间自主导航、电量管理、复杂重排、推动物体、未接入 API 的真实机械臂能力。

## 核心循环

```text
robot_cleaner_prepare_decision_turn
-> choose one current option_id
-> robot_cleaner_execute_option
-> verify
-> next turn prepare again
```

每轮最多一个最终物理动作。`observe:refresh`、场景分析、`place_precheck:*` 不算最终物理动作。

## 感知与地图

- `full`：RGB-D + YOLO + depth/pointcloud geometry，用于 waypoint 观察、pickup/place 候选和任务决策。
- `navigation_only`：RGB-D + depth local costmap，用于继续 active waypoint 路线；不应期待新的 `pick:*` / `place:*`。
- 主坐标、主地图、waypoint、frontier、状态汇报统一使用 AI2-THOR groundtruth grid。
- `action_odometry` / `position-map` 只作为 fallback/debug，不进入大模型主决策链。
- 碰撞或平移动作失败时，用 groundtruth 当前 cell + heading 写入 public/hard blocked edge，后续 planner 必须避开。

## Tidy 规则

默认 `--pickup-surface-policy floor-only`：只追地面/近地面可拾取物。桌面、台面、柜面等 elevated 候选只作为视觉候选记录，不追不捡。

`place-object` 成功只表示一个 pickup/place 子任务完成，不表示房间完成。放置后继续巡视。

## 决策优先级

1. 安全：路径不明、前方障碍、目标未对齐或不可达时不冒进。
2. 整理子目标：`pick:*`、`pursue:pickup_target:*`、`place_precheck:*`、`place:*` 优先于巡视。
3. Waypoint 观察：到达 active waypoint 后，优先 `orient:waypoint_floor_scan`，下一轮 full observe。
4. Waypoint 路线：若存在 `continue:active_waypoint_goal` 且没有更高优先级任务，继续当前 waypoint。
5. 新 waypoint：无 active waypoint 且无可执行整理目标时，选择 `explore:inspection_waypoint:<id>`。
6. Fallback：`frontier_cluster`、`route_step`、底层 `move:*` 只在 inspection waypoint 不可用或执行器明确要求 fallback 时使用。
7. Completion：只有满足房间完成规则才可 `done:probe` / report，不能因为一次放置、一次清扫、一次 waypoint 到达或恢复失败就说房间完成。

## 验证与汇报

- HTTP 200 不等于动作成功；必须看 `lastActionSuccess`、`state_changed`、`pickup_executed`、`holding_object`、`place_executed` 等结构化字段。
- 面向用户区分：已放置物体、已清扫目标、视觉候选但未执行处理的物体。
- 恢复失败应汇报 recover failed，不要虚构房间完成。
