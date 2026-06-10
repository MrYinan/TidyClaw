# V2.3 Household Service Patrol Protocol

V2 keeps the existing cleaning stack, but the main line is now household service
patrol. The robot is no longer modeled as only a cleaner. In `tidy` mode it
performs repeated pickup/place service subgoals while continuing to patrol the
current room.

```text
OpenClaw heartbeat
  -> RGB observation
  -> YOLO household-object perception
  -> task semantic mapping
  -> patrol runner tidy policy
  -> move / pick / place / clean
  -> sanitized feedback + memory update
```

## Task Semantics

The service ontology lives in `configs/service_task_ontology_v2.json`.

- `pickup_target`: object that can be tidied with `PickupObject`
- `place_receptacle`: visible support/container for placement
- `obstacle`: object or geometry that constrains movement
- `cleanable_object`: legacy cleanable floor proxy
- `ignored_object`: object irrelevant to the current task

## Backend APIs

V2.2 adds these online-safe endpoints:

- `POST /pick`
- `POST /place`
- `GET /inventory`

By default they do not expose `objectId`, exact object pose, raw
`metadata.objects`, or inventory object details. For offline debug only, use
`?include_eval=1`.

## Runner Mode

Cleaning remains the default:

```powershell
python scripts\patrol_runner.py --start --continuous --perception-backend yolo --task-mode clean
```

Household service mode:

```powershell
python scripts\patrol_runner.py --start --continuous --perception-backend yolo --task-mode tidy
```

Through the OpenClaw skill wrapper:

```powershell
python skills\patrol-runner\scripts\patrol_runner_skill.py --command run --task-mode tidy
```

Heartbeat can also start tidy tasks:

```powershell
python skills\heartbeat-watchdog\scripts\heartbeat_watchdog.py --task-mode tidy
```

## Tidy State Machine

`tidy` mode is a room-level service patrol, not a single ALFRED episode.

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

After `place-object` succeeds and inventory is empty, the runner records:

- `objects_placed`
- `service_tasks_completed`
- service history in `memory/service-task-state.json`

`place-object` success now also requires backend placement verification:

- the held object must be on the grounded target receptacle
- the final object position must remain inside the robot's reachable front zone
- default placement uses a controlled screen point near the visible front area
  of the receptacle
- whole-receptacle simulator-selected fallback is not supported

Then it clears the pickup/receptacle locks and continues room patrol. It does
not mark the room complete just because one object was placed.

## Completion Semantics

These are different:

- service subgoal complete: one object was picked and placed successfully
- room complete: the current room patrol is finished according to coverage,
  frontier, no-target, stagnation, max-step, or user-stop rules

`pickup_place_verified` is no longer a room-completion reason by itself.

To avoid immediate loops, a just-placed label is temporarily suppressed from the
pickup candidate list.

## Dataset Route

The main dataset route is AI2-THOR generated data:

```powershell
python scripts\collect_ai2thor_service_dataset.py --init-only
python scripts\collect_ai2thor_service_dataset.py --count 1000
```

The collector uses AI2-THOR `instance_detections2D` offline to write YOLO
labels. This is allowed for training data generation only. Online Agent
decisions still use RGB-only YOLO output and sanitized action feedback.
