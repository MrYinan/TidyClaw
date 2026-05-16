---
name: state-manager
description: Manage household service robot patrol / mission / room memory JSON consistently.
metadata:
  {
    "openclaw":
      {
        "emoji": "state",
        "requires": { "bins": ["python3"] }
      }
  }
---

# State Manager

This skill is the single memory writer for the household service robot.

It manages:

- `memory/patrol-state.json`
- `memory/mission-state.json`
- `memory/room-state.json`

Implementation:

```text
scripts/state_manager_core.py
```

## Common Commands

Show state:

```powershell
python skills\state-manager\scripts\state_manager.py show
```

Validate consistency:

```powershell
python skills\state-manager\scripts\state_manager.py validate
```

Check whether patrol should continue:

```powershell
python skills\state-manager\scripts\state_manager.py should-continue
```

Start/reset a mission:

```powershell
python skills\state-manager\scripts\state_manager.py start-mission --room current_room --max-steps 120
```

Record one physical-action step:

```powershell
python skills\state-manager\scripts\state_manager.py record-step --action MoveAhead --mode EXPLORE
```

Record a service step:

```powershell
python skills\state-manager\scripts\state_manager.py record-step --action place-object --mode SERVICE --placed Apple --service-completed "Apple->CounterTop"
```

Stop without marking room complete:

```powershell
python skills\state-manager\scripts\state_manager.py stop-mission --reason user_stop
```

Mark recover failed:

```powershell
python skills\state-manager\scripts\state_manager.py mark-recover-failed --reason consecutive_action_failures
```

Archive report and return to idle:

```powershell
python skills\state-manager\scripts\state_manager.py finalize-report --reason report_delivered
```

## Memory Semantics

Legacy compatibility fields:

- `garbage_detected`
- `garbage_cleaned`
- `targets_found`
- `targets_cleaned`

Service-mode fields:

- `objects_detected`
- `objects_placed`
- `service_tasks_completed`

`room_complete=true` means the room patrol is complete. It does not mean a single pickup/place subgoal completed.

## Agent Rule

Do not manually edit the three state JSON files separately. Use this skill or import `scripts.state_manager_core`.

After a final user-visible report is delivered, finalize the state so the next heartbeat sees an idle system.
