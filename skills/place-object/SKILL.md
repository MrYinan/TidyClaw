---
name: place-object
description: Place the currently held object onto a visible front-center receptacle through the backend /place actuator.
metadata:
  {
    "openclaw": {
      "emoji": "place",
      "requires": { "bins": ["python3"] }
    }
  }
---

# Place Object

Use this skill after the robot is holding an object and fresh RGB perception
indicates a reachable front-center `place_receptacle`.

Run:

```bash
python skills/place-object/scripts/place_object.py
```

The skill calls `POST /place`. The backend may use simulator metadata internally
to execute `PutObject`, but the default response is online-safe.

Default V2 behavior places through a controlled screen point near the visible
front area of the receptacle and verifies that the final object remains on the
target receptacle and inside the robot's reachable front zone. The legacy
whole-receptacle fallback is disabled unless
`ROBOT_ALLOW_RECEPTACLE_WIDE_PLACE_FALLBACK=true`.

Success fields:

- `status = "success"`
- `result_type = "place_executed"`
- `lastActionSuccess`
- `holding_object = false`
- `placement_verified = true`
- `object_reachable_from_agent = true`

Failure result types include:

- `error_no_held_object`
- `error_no_receptacle_in_front`
- `error_receptacle_not_centered`
- `error_receptacle_too_far`
- `error_target_not_receptacle`
- `error_place_no_reachable_point`
- `error_place_position_unreachable`
- `error_place_receptacle_mismatch`
