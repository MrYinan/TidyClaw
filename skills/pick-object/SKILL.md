---
name: pick-object
description: Pick up a visible front-center household service object through the backend /pick actuator.
metadata:
  {
    "openclaw": {
      "emoji": "pick",
      "requires": { "bins": ["python3"] }
    }
  }
---

# Pick Object

Use this skill after fresh RGB perception indicates a reachable front-center
`pickup_target`.

Run:

```bash
python skills/pick-object/scripts/pick_object.py
```

The skill calls `POST /pick`. The backend may use simulator metadata internally
to execute `PickupObject`, but the default response is online-safe and does not
expose `objectId`, exact position, or raw metadata.

Success fields:

- `status = "success"`
- `result_type = "pickup_executed"`
- `lastActionSuccess`
- `holding_object = true`

Failure result types include:

- `error_no_pickup_target_in_front`
- `error_pickup_target_not_centered`
- `error_pickup_target_too_far`
- `error_target_not_pickupable`
- `error_already_holding_object`
