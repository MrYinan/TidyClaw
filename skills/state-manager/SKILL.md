---
name: state-manager
description: 统一管理家庭服务机器人 patrol / mission / room 记忆 JSON。
metadata:
  {
    "openclaw":
      {
        "emoji": "state",
        "requires": { "bins": ["python3"] }
      }
  }
---

# 状态管理器

本技能是家庭服务机器人的唯一记忆写入入口。

它管理以下文件：

- `memory/patrol-state.json`
- `memory/mission-state.json`
- `memory/room-state.json`

核心实现：

```text
scripts/state_manager_core.py
```

## 常用命令

显示状态：

```powershell
python skills\state-manager\scripts\state_manager.py show
```

验证一致性：

```powershell
python skills\state-manager\scripts\state_manager.py validate
```

检查巡视是否应该继续：

```powershell
python skills\state-manager\scripts\state_manager.py should-continue
```

启动或重置任务：

```powershell
python skills\state-manager\scripts\state_manager.py start-mission --room current_room --max-steps 120
```

记录一次物理动作步骤：

```powershell
python skills\state-manager\scripts\state_manager.py record-step --action MoveAhead --mode EXPLORE
```

记录一次服务步骤：

```powershell
python skills\state-manager\scripts\state_manager.py record-step --action place-object --mode SERVICE --placed Apple --service-completed "Apple->CounterTop"
```

停止任务，但不标记房间完成：

```powershell
python skills\state-manager\scripts\state_manager.py stop-mission --reason user_stop
```

标记恢复失败：

```powershell
python skills\state-manager\scripts\state_manager.py mark-recover-failed --reason consecutive_action_failures
```

归档报告并返回空闲状态：

```powershell
python skills\state-manager\scripts\state_manager.py finalize-report --reason report_delivered
```

## 记忆语义

旧版兼容字段：

- `garbage_detected`
- `garbage_cleaned`
- `targets_found`
- `targets_cleaned`

服务模式字段：

- `objects_detected`
- `objects_placed`
- `service_tasks_completed`

`room_complete=true` 表示房间巡视完成，不表示单个 pickup/place 子目标完成。

## Agent 使用规则

不要分别手动编辑这三个状态 JSON 文件。应使用本技能，或导入 `scripts.state_manager_core`。

最终的用户可见汇报交付后，应 finalize 状态，让下一次 heartbeat 看到空闲系统。
