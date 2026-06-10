# Robot Cleaner Tools

Stable OpenClaw tool plugin for the household service robot workspace.

The plugin exposes only the long-lived public tool surface that the model should
use:

- `robot_cleaner_prepare_decision_turn`
- `robot_cleaner_execute_option`
- `robot_cleaner_status`
- `robot_cleaner_report`
- `robot_cleaner_stop`

The model must not call low-level robot scripts directly. Movement, pickup,
placement, cleaning, state updates, and prechecks stay behind the local Robot
Cleaner tool bridge service.

## Configuration

Configure these fields in the OpenClaw plugin entry:

```json
{
  "baseUrl": "http://127.0.0.1:8765",
  "timeoutMs": 120000
}
```

The plugin does not spawn Python or shell commands directly. The local bridge
service owns the fixed Python script calls and returns JSON.

## Build

```bash
npm install
npm run plugin:build
npm run plugin:validate
npm test
```
