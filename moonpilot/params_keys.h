// moonpilot params. Textually included into the `keys` map in
// openpilot/common/params_keys.h, so this file holds map rows only, in
// upstream's format:
//   {"MoonpilotExample", {PERSISTENT, BOOL}},        // no default
//   {"MoonpilotToggle", {PERSISTENT, BOOL, "1"}},    // default on
{"MoonpilotLeadLateral", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotTorqueLateral", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotLongitudinal", {PERSISTENT, BOOL, "1"}},
    // Off by default: upstream dropped chill mode and its ExperimentalMode
    // param (#39097), so the fork planner's own choice between the model's raw
    // accel and its gated path lives here. Read every frame.
    {"MoonpilotExperimentalMode", {PERSISTENT, BOOL, "0"}},
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
    // Off by default: it adds braking from the model's road edges, and the
    // width threshold is not validated until someone drives it.
    {"MoonpilotSqueeze", {PERSISTENT, BOOL, "0"}},
    // Off by default: it raises a banner from the model path leaving the
    // free corridor, and the fraction threshold is not validated on a car.
    {"MoonpilotPathOutside", {PERSISTENT, BOOL, "0"}},
    // On by default: it lets a grade spend its own acceleration inside a
    // bounded speed band; the feature is read every frame and safety candidates
    // still win through the planner's minimum.
    {"MoonpilotCoastGrade", {PERSISTENT, BOOL, "1"}},
    // On by default: a shallow negative ask above -0.40 m/s^2 is raised to a coast and a deeper one is
    // held as one press, so the car's brake arbitration does not flicker the brake lights; read
    // every frame, so it applies immediately.
    {"MoonpilotBrakeBand", {PERSISTENT, BOOL, "1"}},
    // Off by default, a test toggle: Toyota PERMIT_BRAKING is held off while the request stays at the
    // engine's own coast, and the planner coasts to it; set into CarParams at card start, so restart.
    {"MoonpilotEngineCoast", {PERSISTENT, BOOL, "0"}},
    // Off by default, a test toggle: below 0.5 m/s a gentle brake request eases toward zero before the
    // stop; read once when the longitudinal controller is built, so restart.
    {"MoonpilotCrawlRelease", {PERSISTENT, BOOL, "0"}},
    // On by default: on a hand-back from the driver's gas the positive ask starts low and rebuilds; read every
    // frame, so it applies immediately.
    {"MoonpilotHandbackEase", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotPathLookahead", {PERSISTENT, INT, "0"}},
    {"MoonpilotPathSmooth", {PERSISTENT, INT, "0"}},
    // Off by default: the part of the curvature request beyond the ego lane's own bend is capped
    // at MOONPILOT_RECENTER_LAT_ACCEL above 12 m/s (faded in from 8) on seen lane lines; bound into
    // the curvature reference at construction, so the row takes a restart.
    {"MoonpilotPathRecenter", {PERSISTENT, BOOL, "0"}},
    {"MoonpilotSteerHysteresis", {PERSISTENT, INT, "0"}},
    // Off by default: an index into STEER_DAMPING's choices, the derivative time on the fork torque controller's
    // feedback. Chosen at construction, so the row takes a restart.
    {"MoonpilotSteerDamping", {PERSISTENT, INT, "0"}},
    // Off by default, a test toggle: the fork torque controller's friction compensation may rise
    // no faster than MOONPILOT_FRICTION_RISE, so the torque that breaks the wheel loose is not
    // stored in one step; release is unthrottled. Chosen at construction, so the row takes a restart.
    {"MoonpilotSteerFrictionRamp", {PERSISTENT, BOOL, "0"}},
    // On by default: below the lane-change speed threshold a turn signal
    // feeds a turn desire into the model so it steers into the corner. The
    // model's own turn prediction picks the direction.
    {"MoonpilotTurnDesire", {PERSISTENT, BOOL, "1"}},
    {"MoonpilotTailscaleStatus", {CLEAR_ON_MANAGER_START, STRING}},
    // Learned onroad by moonpilot/latency.py, written by the fork longitudinal
    // planner: the command -> delivered-accel lag in seconds. 0.0 means nothing
    // measured yet, which is below the estimator's own ROI floor and so is
    // never applied.
    {"MoonpilotLongLag", {PERSISTENT, FLOAT, "0.0"}},
    // The evidence behind MoonpilotLongLag, in blocks of accepted estimates.
    // It rides with the value so fragmented engagement accumulates across
    // drives instead of being re-trusted wholesale: below the estimator's own
    // block requirement the carried mean is held back and the stock constant
    // stands. 0 is "no evidence", which is what a value written before this
    // key existed reads as.
    {"MoonpilotLongLagBlocks", {PERSISTENT, INT, "0"}},
    // The completed blocks behind MoonpilotLongLag, oldest first, so a boot
    // resumes the history itself rather than that many copies of its mean.
    // Written beside the value and count, which stay what modeld reads.
    {"MoonpilotLongLagHistory", {PERSISTENT, JSON}},
    // Learned onroad from positive command ramps versus `carState.aEgo`.
    // Tightens only the longitudinal comfort-side jerk; the braking and
    // emergency ramps stay validated constants.
    {"MoonpilotLongJerkScale", {PERSISTENT, FLOAT, "1.0"}},
    // Learned onroad by moonpilot/curve.py: realized lateral acceleration over
    // what the model's path predicted. 1.0 is neutral, and the applied value is
    // clamped at or above it, so a learned bias only ever plans for less speed.
    {"MoonpilotCurveLatScale", {PERSISTENT, FLOAT, "1.0"}},
    // Learned onroad by moonpilot/pitch.py: the standing offset in
    // carControl.orientationNED[1], in radians, subtracted before any grade
    // term reads the pitch. 0.0 means nothing measured yet and corrects
    // nothing; the applied value is clamped so a wrong offset alone cannot
    // take level road past MOONPILOT_COAST_GRADE_MIN.
    {"MoonpilotPitchOffset", {PERSISTENT, FLOAT, "0.0"}},
    // Learned onroad by moonpilot/radar_latency.py: a radar lead's sensor
    // latency in seconds, added to the lead's age when planning. 0.0 means
    // nothing measured yet (pass-through); the applied value is one-sided
    // [0, 0.3]. The evidence count rides with it so partial drives accumulate.
    {"MoonpilotRadarLatency", {PERSISTENT, FLOAT, "0.0"}},
    {"MoonpilotRadarLatencySamples", {PERSISTENT, INT, "0"}},
    {"MoonpilotOffroad",
     {CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON, BOOL, "0"}},
    // Model marketplace, moonpilot/models.py. The two desired strings are the
    // driver's selection: "" is the bundled model, or one 64-hex digest for a
    // catalog recipe. Persistent, and read by the UI only -- the model
    // processes read the boot snapshot below, never these, so a selection
    // change cannot swap a model under a running prediction (the panel says
    // "restart to apply").
    {"MoonpilotModelsDriving", {PERSISTENT, STRING}},
    {"MoonpilotModelsMonitoring", {PERSISTENT, STRING}},
    // The starred models in the pickers, ";"-joined recipe digests, written by
    // the star on a row and read back into the "favorites" group. A digest
    // whose catalog entry is gone simply matches no row, so a stale star costs
    // nothing and is rewritten on the next change.
    {"MoonpilotModelsFavs", {PERSISTENT, STRING}},
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
    // The settings panel writes retry requests; depsd consumes and clears them
    // this boot.
    {"MoonpilotDepsRequest", {CLEAR_ON_MANAGER_START, STRING}},
    {"MoonpilotCatalogSignatures", {PERSISTENT, BOOL, "0"}},
    // manager writes the boot rollback report; it clears it at manager start.
    {"MoonpilotRollback", {CLEAR_ON_MANAGER_START, STRING}},
