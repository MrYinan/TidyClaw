# V2.3 ALFRED-Style Tidy State Machine

V2.3 keeps the online policy RGB-only, but borrows ALFRED's subgoal organization:
the runner does not chase a different object every frame. It locks a target,
finishes the current subgoal, verifies feedback, then advances to the next
subgoal.

## Core Phases

```text
SEARCH_PICKUP_TARGET
LOCK_PICKUP_TARGET
ALIGN_PICKUP_TARGET
PICK_OBJECT
VERIFY_HOLDING
SEARCH_RECEPTACLE
LOCK_RECEPTACLE
ALIGN_RECEPTACLE
PLACE_OBJECT
VERIFY_TASK_DONE
TASK_DONE
```

## Runtime State

The runner persists tidy task progress in:

```text
memory/service-task-state.json
```

Important fields:

```json
{
  "task_style": "alfred_like_pick_place",
  "phase": "ALIGN_PICKUP_TARGET",
  "subgoal_index": 0,
  "target_raw_label": "Book",
  "target_signature": "Book/front-left/...",
  "receptacle_raw_label": null,
  "holding_object": false,
  "history": []
}
```

## Decision Rules

- If not holding an object, only pursue `pickup_target`.
- Once a pickup target is locked, keep pursuing the same raw label for a few steps before switching.
- `pick-object` is called when the locked pickup target is visually front-center and actionable.
- After pickup, `/inventory` must report `holding_object=true`.
- If holding an object, ignore pickup targets and only pursue `place_receptacle`.
- `place-object` is called when a locked receptacle is front-center/actionable.
- After place, `/inventory` must report `holding_object=false`; then phase becomes `TASK_DONE`.

## Future Extensions

Future ALFRED-like tasks should add subgoal phases instead of rewriting the runner:

```text
PICK_OBJECT -> CLEAN_OBJECT -> SEARCH_RECEPTACLE -> PLACE_OBJECT
PICK_OBJECT -> HEAT_OBJECT -> SEARCH_RECEPTACLE -> PLACE_OBJECT
PICK_OBJECT -> COOL_OBJECT -> SEARCH_RECEPTACLE -> PLACE_OBJECT
SEARCH_TOGGLE_TARGET -> TOGGLE_OBJECT -> VERIFY_TASK_DONE
```

The online Agent still uses RGB perception and sanitized action feedback. ALFRED
trajectory JSON/PDDL/masks are references for task structure, not online inputs.
