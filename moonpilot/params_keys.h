// moonpilot params. Textually included into the `keys` map in openpilot/common/params_keys.h,
// so this file holds map rows only, in upstream's format:
//   {"MoonpilotExample", {PERSISTENT, BOOL}},        // no default
//   {"MoonpilotToggle", {PERSISTENT, BOOL, "1"}},    // default on
{"MoonpilotBranding", {PERSISTENT, BOOL, "1"}},
{"MoonpilotLeadLateral", {PERSISTENT, BOOL, "1"}},
