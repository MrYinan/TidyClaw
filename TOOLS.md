# TOOLS.md - Household Service Robot Runtime Notes

This workspace runs an OpenClaw embodied household service robot in AI2-THOR.

The project still contains legacy "robot-cleaner" names, but the active V2 line is a service robot:

- `clean` mode: legacy floor-cleaning patrol
- `tidy` mode: household service patrol with pickup/place, plus optional clean fallback

## Environment

- Framework: OpenClaw
- Agent role: household service robot
- Simulator: AI2-THOR
- Local control service: `http://127.0.0.1:5000`
- Default vision image: `D:\photos\openclaw_robot_vision.jpg`
- Default workspace Python: `D:\Anaconda\envs\robot\python.exe`

## Core Skills And Scripts

- `get-vision`: capture the current RGB observation, optional depth frame, camera info, and action feedback
- `perceive-scene-yolo`: YOLO service perception for pickup targets, receptacles, obstacles, cleanable objects, and depth-derived surface candidates
- `analyze-scene-opencv`: legacy OpenCV floor-cleaning perception baseline
- `move-robot`: execute `MoveAhead`, `MoveBack`, `RotateLeft`, `RotateRight`
- `pick-object`: execute a visual-candidate grounded pickup action
- `place-object`: execute a visual-candidate grounded place action
- `clean-garbage`: legacy floor clean action
- `patrol-runner`: continuous room patrol/service task runner
- `navigation-memory`: lightweight frontier/coverage/collision memory
- `state-manager`: single writer for `memory/*.json`

## Common Commands

Run legacy cleaning patrol:

```powershell
python scripts\patrol_runner.py --start --continuous --perception-backend yolo --task-mode clean
```

Run service robot tidy patrol:

```powershell
python scripts\patrol_runner.py --start --continuous --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden
```

Default tidy pickup policy is `--pickup-surface-policy floor-only`, so tabletop/elevated objects are ignored unless an explicit desktop-object tidy test uses `--pickup-surface-policy any-surface`.

Run one foreground debug segment:

```powershell
python scripts\patrol_runner.py --start --segment-steps 20 --max-steps 80 --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden --verbose
```

Use the OpenClaw patrol-runner skill entry:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --max-steps 200 --task-mode tidy
python skills\patrol-runner\scripts\patrol_runner_skill.py --command start --max-steps 200 --task-mode tidy
python skills\patrol-runner\scripts\patrol_runner_skill.py --command status
python skills\patrol-runner\scripts\patrol_runner_skill.py --command report
python skills\patrol-runner\scripts\patrol_runner_skill.py --command stop --reason user_stop
```

Inspect memory:

```powershell
python skills\state-manager\scripts\state_manager.py show
python skills\state-manager\scripts\state_manager.py validate
python skills\navigation-memory\scripts\navigation_memory.py show
```

## Service Task Semantics

The service ontology lives at:

```text
configs/service_task_ontology_v2.json
```

Important classes:

- `pickup_target`: object that may be picked and tidied
- `place_receptacle`: support surface/container where an object may be placed
- `obstacle`: structure or object that affects navigation
- `cleanable_object`: legacy cleanable floor target
- `ignored_object`: irrelevant object

## Tidy Mode State Rule

`tidy` is a room patrol with repeated service subgoals:

```text
SEARCH_PICKUP_TARGET
-> LOCK_PICKUP_TARGET
-> ALIGN_PICKUP_TARGET
-> PICK_OBJECT
-> SEARCH_RECEPTACLE
-> LOCK_RECEPTACLE
-> ALIGN_RECEPTACLE
-> PLACE_OBJECT
-> record service subgoal complete
-> SEARCH_PICKUP_TARGET / EXPLORE again
```

Important: `place-object` success does not mean room completion. It only records one service subgoal. Room completion still depends on coverage/frontier/no-target/stagnation/max-step rules.

## JSON Memory Fields

Legacy fields are preserved for compatibility:

- `mission.garbage_detected`
- `mission.garbage_cleaned`
- `room.targets_found`
- `room.targets_cleaned`

Service fields are now preferred for tidy mode:

- `mission.objects_detected`
- `mission.objects_placed`
- `mission.service_tasks_completed`
- `room.objects_placed`
- `room.service_tasks_completed`

## Safety Rules

- Do not move without fresh structured perception.
- Do not execute pickup/place unless the candidate is visible, reachable, and actuator-ready according to perception and executor feedback.
- `place-object` must verify that the final object remains on the grounded receptacle and inside the robot's reachable front placement zone; whole-receptacle placement fallback is off by default.
- Do not immediately re-pick a just-placed object; it is temporarily suppressed after a successful place.
- If perception times out twice, use one conservative rotation instead of moving forward.
- If repeated action failures occur, stop as recover failed rather than reporting the room complete.
