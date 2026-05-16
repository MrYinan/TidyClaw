---
name: navigation-memory
description: 查看、重置和更新 robot-cleaner 的轻量导航记忆，包括已访问网格、碰撞边、覆盖率和探索建议。
metadata:
  {
    "openclaw":
      {
        "emoji": "map",
        "requires": { "bins": ["python3"] }
      }
  }
---

# Navigation Memory

本 skill 管理单房间巡视中的轻量导航记忆。

它不是完整 SLAM，而是为当前 AI2-THOR 单房间清扫任务提供：

- 已访问网格 `visited_cells`
- 网格访问次数 `visited_cell_counts`
- 碰撞/不可通行边 `blocked_edges`
- 粗略覆盖率 `coverage_estimate`
- 最近导航动作 `recent_navigation_actions`
- 碰撞次数和振荡次数
- frontier-like 探索建议

正式实现位于：

```text
scripts/navigation_memory_core.py
```

## 查看导航记忆

```bash
python C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\navigation-memory\scripts\navigation_memory.py show
```

或：

```bash
python C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\navigation-memory\scripts\navigation_memory.py status
```

## 重置导航记忆

```bash
python C:\Users\天涯\.openclaw\workspace-robot-cleaner\skills\navigation-memory\scripts\navigation_memory.py reset --room current_room
```

该命令只重置导航记忆字段，不清空任务步数和清扫记录。

## Runner 内部用法

`patrol-runner` 会直接导入 `scripts.navigation_memory_core.NavigationMemory`，在每一轮中：

1. 根据 `get-vision` 位姿记录当前网格
2. 根据 `analyze-scene-opencv` 的 open directions 估计 frontier
3. 根据 move 结果记录碰撞边
4. 用未访问方向优先的策略给出探索建议
5. 将导航记忆写入 `memory/room-state.json`

## Agent Rule

当用户询问“导航记忆 / 走过哪里 / 覆盖率 / 为什么来回走 / 碰撞次数”时，优先调用本 skill。

当用户要求启动正式循环巡视时，仍调用 `patrol-runner`，不要单独调用本 skill 来移动机器人。
