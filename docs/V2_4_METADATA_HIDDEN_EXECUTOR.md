# V2.4 Metadata-Hidden Executor

This project now separates visual decision making from simulator-side action grounding.

## Goal

The online decision module should behave like an ALFRED-style embodied agent:

1. Observe the egocentric RGB frame.
2. Use YOLO perception to select a visible pickup target or receptacle.
3. Predict the high-level interaction action.
4. Send only a sanitized visual candidate, such as label, bbox, center point, and task role, to the executor.

The decision module must not consume AI2-THOR object IDs, global object lists, exact object positions, or raw metadata for pickup/place planning.

## Runtime Boundary

Default runner mode:

```powershell
python scripts\patrol_runner.py --start --segment-steps 20 --max-steps 80 --perception-backend yolo --task-mode tidy --interaction-grounding metadata-hidden --verbose
```

Decision-side input:

- RGB image path from `get-vision`
- YOLO structured candidates
- Action history and service-task phase

Decision-side output:

- `pick-object` or `place-object`
- Sanitized visual candidate:
  - `label`
  - `raw_label`
  - `task_semantic_class`
  - `bbox`
  - `center`
  - `confidence`
  - `position_hint`

Executor-side hidden grounding:

- `/pick` and `/place` receive the visual candidate.
- `RobotEnvironment` uses AI2-THOR metadata internally to map that current-frame visual candidate to an executable `objectId`.
- Online responses expose only action feedback, such as `pickup_executed`, `place_executed`, and `holding_object`.
- Raw `objectId`, metadata object lists, candidate metadata, and exact object positions remain excluded from online responses.

## Why This Is ALFRED-Style

ALFRED agents are expected to act from language and egocentric visual observations, while the simulator/evaluator internally grounds interactions and verifies state changes. This project follows the same boundary at the engineering level:

```text
RGB -> YOLO candidate -> action + visual target -> hidden simulator executor -> success/fail feedback
```

Metadata still exists inside AI2-THOR, but it is no longer an online decision input for pickup/place.

## Legacy Mode

For debugging only:

```powershell
python scripts\patrol_runner.py --task-mode tidy --interaction-grounding legacy-metadata
```

This keeps the old empty `/pick` and `/place` calls where the backend selects the front eligible metadata object directly. Do not use this mode for final project claims about ALFRED-style metadata hiding.
