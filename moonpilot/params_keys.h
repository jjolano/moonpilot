// moonpilot params. Textually included into the `keys` map in
// openpilot/common/params_keys.h, so this file holds map rows only, in
// upstream's format:
//   {"MoonpilotExample", {PERSISTENT, BOOL}},        // no default
//   {"MoonpilotToggle", {PERSISTENT, BOOL, "1"}},    // default on
{"MoonpilotLeadLateral", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotTorqueLateral", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotLongitudinal", {PERSISTENT, BOOL, "1"}},
    // On by default: `policy` takes the minimum, so the model's ask can only
    // ever add braking, and past the deadband it is bounded by the actuator's
    // own ACCEL_MIN rather than a fork floor.
    {"MoonpilotModelBraking", {PERSISTENT, BOOL, "1"}},
    // Off by default: it changes the speed the planner plans from, and the
    // correction is not validated until someone drives it.
    {"MoonpilotSlam", {PERSISTENT, BOOL, "0"}},
    {"MoonpilotTailscale", {PERSISTENT, BOOL, "0"}},
    // Off by default: it changes the safety model, and the PCM_CRUISE_2 rate on
    // the car is not validated until someone drives it.
    {"MoonpilotLateralEngage", {PERSISTENT, BOOL, "0"}},
    // Off by default: it adds braking authority derived from the model's
    // predicted path, and the curve target is not validated until someone
    // drives it.
    {"MoonpilotCurveSpeed", {PERSISTENT, BOOL, "0"}},
    // Off by default: it changes the steering request, and the preview gain is
    // not validated until someone drives it.
    {"MoonpilotPathPreview", {PERSISTENT, BOOL, "0"}},
    // Tailscaled -> panels, one line: "<state>" or "<state> <detail>", decoded
    // in moonpilot/tailscale.py. CLEAR_ON_MANAGER_START so a reboot cannot
    // leave a stale "running 100.x" on screen.
    {"MoonpilotTailscaleStatus", {CLEAR_ON_MANAGER_START, STRING}},
    // Learned onroad by moonpilot/latency.py, written by the fork longitudinal
    // planner: the command -> delivered-accel lag in seconds. 0.0 means nothing
    // measured yet, which is below the estimator's own ROI floor and so is
    // never applied.
    {"MoonpilotLongLag", {PERSISTENT, FLOAT, "0.0"}},
    // Learned onroad by moonpilot/curve.py: realized lateral acceleration over
    // what the model's path predicted. 1.0 is neutral, and the applied value is
    // clamped at or above it, so a learned bias only ever plans for less speed.
    {"MoonpilotCurveLatScale", {PERSISTENT, FLOAT, "1.0"}},
