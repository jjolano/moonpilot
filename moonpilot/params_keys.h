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
    // Off by default: it changes the safety model. The permission is granted in
    // the forked safety layer for every brand, arming on the car's own cruise
    // main switch where the mode decodes it (Toyota, Honda, Volkswagen MQB/MEB)
    // and on openpilot's own engaged heartbeat everywhere else, with neither
    // rate validated on a car until someone drives one.
    {"MoonpilotLateralEngage", {PERSISTENT, BOOL, "0"}},
    // Off by default: it adds braking authority derived from the model's
    // predicted path, and the curve target is not validated until someone
    // drives it.
    {"MoonpilotCurveSpeed", {PERSISTENT, BOOL, "0"}},
    // On by default: it lets a grade spend its own acceleration inside a bounded speed band; the
    // feature is read every frame and safety candidates still win through the planner's minimum.
    {"MoonpilotCoastGrade", {PERSISTENT, BOOL, "1"}},
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
    // Learned onroad from positive command ramps versus `carState.aEgo`.
    // Tightens only the longitudinal comfort-side jerk; the braking and
    // emergency ramps stay validated constants.
    {"MoonpilotLongJerkScale", {PERSISTENT, FLOAT, "1.0"}},
    // Learned onroad by moonpilot/curve.py: realized lateral acceleration over
    // what the model's path predicted. 1.0 is neutral, and the applied value is
    // clamped at or above it, so a learned bias only ever plans for less speed.
    {"MoonpilotCurveLatScale", {PERSISTENT, FLOAT, "1.0"}},
    // Offroad mode, entered from the moonpilot panel's row or a hold on the
    // driving view: while it is set the device stays offroad with the ignition
    // on. CLEAR_ON_IGNITION_ON so a new drive always comes up onroad,
    // CLEAR_ON_MANAGER_START so a reboot does too; both are cleared by
    // openpilot/system/manager/manager.py, not by fork code.
    {"MoonpilotOffroad",
     {CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON, BOOL, "0"}},
    // Model marketplace, moonpilot/models.py. The two desired strings are the
    // driver's selection: "" is the bundled model, a 64-hex digest is one
    // catalog recipe, "c-…" a composition of installed recipes. Persistent, and
    // read by the UI only -- the model processes read the boot snapshot below,
    // never these, so a selection change cannot swap a model under a running
    // prediction (the panel says "restart to apply").
    {"MoonpilotModelsDriving", {PERSISTENT, STRING, ""}},
    {"MoonpilotModelsMonitoring", {PERSISTENT, STRING, ""}},
    // The committed per-boot decision: what each kind actually loads, with
    // every check the boot made recorded beside it. Written once by
    // commit_boot_selection() from openpilot/system/manager/manager.py, after
    // the CLEAR_ON_MANAGER_START clears and before any process starts, and
    // CLEAR_ON_MANAGER_START so a reboot cannot inherit a stale snapshot.
    {"MoonpilotModelsBoot", {CLEAR_ON_MANAGER_START, STRING}},
    // modelsd's inbox (one JSON op: refresh/install/remove/cancel) and the
    // snapshot the panels render. Both CLEAR_ON_MANAGER_START: a request is
    // for this boot, and a status from the last one must never be shown.
    {"MoonpilotModelsRequest", {CLEAR_ON_MANAGER_START, STRING}},
    {"MoonpilotModelsStatus", {CLEAR_ON_MANAGER_START, STRING}},
    // Written by each model process when it loads a custom model, and only
    // then, so the panel can name what is in effect: "<name> (<protocol>)", or
    // "stock: <reason>" when a committed selection failed to load. Two rows
    // because two processes write them, and a param has one writer.
    {"MoonpilotModelsActiveDriving", {CLEAR_ON_MANAGER_START, STRING}},
    {"MoonpilotModelsActiveMonitoring", {CLEAR_ON_MANAGER_START, STRING}},
    // depsd writes this boot's state; the manager clears it at manager start.
    {"MoonpilotDepsStatus", {CLEAR_ON_MANAGER_START, STRING}},
    // The settings panel writes retry requests; depsd consumes and clears them this boot.
    {"MoonpilotDepsRequest", {CLEAR_ON_MANAGER_START, STRING}},
    // The settings panel writes this toggle; a user change or params reset clears it.
    {"MoonpilotCatalogSignatures", {PERSISTENT, BOOL, "0"}},
    // manager writes the boot rollback report; it clears it at manager start.
    {"MoonpilotRollback", {CLEAR_ON_MANAGER_START, STRING}},
