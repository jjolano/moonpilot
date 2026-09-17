// moonpilot params. Textually included into the `keys` map in
// openpilot/common/params_keys.h, so this file holds map rows only, in
// upstream's format:
//   {"MoonpilotExample", {PERSISTENT, BOOL}},        // no default
//   {"MoonpilotToggle", {PERSISTENT, BOOL, "1"}},    // default on
{"MoonpilotLeadLateral", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotTailscale", {PERSISTENT, BOOL, "0"}},
    // Tailscaled -> panels, one line: "<state>" or "<state> <detail>", decoded
    // in moonpilot/tailscale.py. CLEAR_ON_MANAGER_START so a reboot cannot
    // leave a stale "running 100.x" on screen.
    {"MoonpilotTailscaleStatus", {CLEAR_ON_MANAGER_START, STRING}},
