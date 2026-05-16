---
name: patrol-runner
description: Start, continue, inspect, or stop the single-room household service patrol runner.
metadata:
  {
    "openclaw":
      {
        "emoji": "patrol",
        "requires": { "bins": ["python3"] }
      }
  }
---

# Household Service Patrol Runner

This skill is the OpenClaw entry point for `scripts/patrol_runner.py`.

It schedules the existing skills in a closed loop:

```text
get-vision -> perceive/analyze scene -> decide -> move/pick/place/clean -> verify -> navigation-memory -> state-manager
```

## When To Use

Use this skill when the user asks for:

- current-room service patrol
- automatic tidy mode
- household-object pickup/place tasks
- continuous room inspection
- legacy cleaning patrol

Do not use it for a single manual movement, one image capture, or one isolated scene analysis.

## Modes

Legacy cleaning:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --max-steps 200 --task-mode clean
```

Household service / tidy:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --max-steps 200 --task-mode tidy
```

Detached background service patrol:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command start --max-steps 200 --task-mode tidy
```

Status/report:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command status
python skills\patrol-runner\scripts\patrol_runner_skill.py --command report
```

Stop:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command stop --reason user_stop
```

## Tidy Semantics

`tidy` mode is not a one-shot pickup/place task. It is a room patrol that can complete multiple service subgoals.

One service subgoal:

```text
find pickup target
-> align/approach
-> pick
-> find receptacle
-> align/approach
-> place
-> record objects_placed and service_tasks_completed
-> continue patrol
```

Important:

- `place-object` success does not mark the room complete.
- The runner continues until room-completion, max-step, recover-failed, or user-stop conditions.
- A just-placed object is temporarily suppressed so the robot does not immediately pick it up again.
- Tidy pickup defaults to `--pickup-surface-policy floor-only`; tabletop/elevated pickup candidates require an explicit `--pickup-surface-policy any-surface` debug run.

## Runtime Contract

Successful `--command run` returns:

- `status = success`
- `result_type = patrol_runner_run_finished`
- `runner_returncode`
- `events_tail`
- `report`
- `report_ready`
- `notify_user`
- `user_message`

Successful `--command status` returns:

- background process state
- state-manager validation
- should-continue result
- current report
- service task progress when available

## Agent Rule

When the user asks for continuous patrol, automatic tidy, or service-room inspection, call this skill rather than manually looping inside chat context.

When only a short debug run is needed, direct runner invocation is acceptable:

```powershell
python scripts\patrol_runner.py --start --segment-steps 20 --max-steps 80 --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden --verbose
```

Do not report a short segment as a completed room unless `room_complete=true`.
