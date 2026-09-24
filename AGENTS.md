# moonpilot

Personal fork of [openpilot](https://github.com/commaai/openpilot). The name is lowercase everywhere, like `openpilot` and `comma`. `CLAUDE.md` is a symlink to this file.

## The rule

Fork code lives in `moonpilot/`. Upstream files carry **seams** only: the minimum lines that hook into `moonpilot/`, each marked with a literal `moonpilot` comment or identifier. The fork's whole surface on upstream is the table below — keep it that way and `git merge upstream/master` only ever conflicts on seam lines.

New fork behavior: write it under `moonpilot/`, hook it at the seam that already exists. A genuinely new seam means updating this table *and* `ALLOWED` in `moonpilot/tests/test_upstream_touches.py`, which fails on any other upstream edit.

## Seams

| Upstream file | Seam |
| --- | --- |
| `pyproject.toml` | `moonpilot` in the hatch wheel packages, so `import moonpilot` resolves; `moonpilot/vendor/openmodels/*` in the codespell `skip`, because that copy is third-party text with third-party spelling |
| `scripts/lint/lint.sh` | ruff / ty / `git ls-files` cover `moonpilot/` |
| `tools/test_runner.py` | `moonpilot` in the default test targets |
| `.gitmodules` | relative fork URLs for `panda` and `opendbc_repo` |
| `panda` (submodule) | `HEALTH_FLAG_CONTROLS_ALLOWED_LATERAL` in `board/health.h`, published in `board/main_comms.h` |
| `opendbc_repo` (submodule) | the fork's safety layer: `opendbc/safety/moonpilot/lateral_engage.h`, `PCM_CRUISE_2` in `modes/toyota.h`, the 12 lateral reads in `lateral.h` |
| `openpilot/common/params_keys.h` | `#include "moonpilot/params_keys.h"` — fork params, one row per key |
| `openpilot/system/hardware/hardwared.py` | `onroad_conditions["moonpilot_onroad"]` — offroad mode holds the device offroad while the ignition is on |
| `openpilot/system/manager/process_config.py` | `procs += MOONPILOT_PROCS` from `moonpilot/procs.py` |
| `openpilot/system/manager/manager.py` | `commit_boot_selection(params)` right after the default seeding — the one boot snapshot every model process reads; `moonpilot.boot`'s boot-health token and rollback report after the first `ensure_running` |
| `launch_chffrplus.sh` | `moonpilot/boot.sh recover` before anything else in `launch()`, and the boot token written into the swap record — the boot-success rollback, see **Boot rollback** |
| `openpilot/selfdrive/modeld/modeld.py` | `moonpilot_model_runtime()`, and the runtime's model, smoothing constants and action decode; the learned long-lag horizon (`applied_long_delay`, re-read in the loop); low-speed turn-desire hook after `DesireHelper.update()` |
| `openpilot/selfdrive/modeld/dmonitoringmodeld.py` | `moonpilot_dm_runtime()`, and the runtime's model and output parse |
| `openpilot/cereal/custom.capnp` | `MoonpilotState` (upstream's reserved struct; never change the `@0x…` id) |
| `openpilot/cereal/log.capnp` | `moonpilotState @107` event field; `PandaState.controlsAllowedLateral @38` and `controlsAllowedLongitudinal @39` — upstream's two slots reserved for forks, the first carrying this fork's lateral grant and the second named for the longitudinal half of the split, which this fork never populates; the fork's own `OnroadEvent.EventName` rows — `lateralEngageOff @105`, `turnLeft @106` and `turnRight @107`, above upstream's highest @104, so the next fork event is @108 |
| `openpilot/cereal/services.py` | `moonpilotState` service row |
| `openpilot/common/version.h` | `COMMA_VERSION "<upstream>-moonpilot.<fork revision>"` |
| `openpilot/selfdrive/controls/plannerd.py` | `moonpilotState` subscription; `moonpilot_longitudinal_planner()` picks the longitudinal planner |
| `openpilot/selfdrive/controls/lib/longitudinal_planner.py` | lead danger factor from `moonpilot.lead` |
| `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | `lead_danger_factor` kwarg on `LongitudinalMpc.update` |
| `openpilot/selfdrive/controls/controlsd.py` | `moonpilot_latcontrol()` picks the torque lateral controller, `moonpilot_longcontrol()` the acceleration controller, `moonpilot_curvature()` applies response-aligned curvature, `moonpilot_actuator_gate()` both actuator permissions; `pandaStates` in the subscription |
| `openpilot/selfdrive/car/card.py` | `moonpilot_engage_safety_param()` before `CarParams` is written |
| `openpilot/selfdrive/pandad/pandad.cc` | `PandaState.controlsAllowedLateral` from the panda health flag |
| `openpilot/selfdrive/selfdrived/selfdrived.py` | `moonpilot_engage()`; `LateralEngage.update()` after every event source; the panda cross-check; `is_fork_build()` picks the startup banner; the turn-desire banner, raised from `moonpilot/turn_desire.py`'s `turn_desire_alert` — the fed desire is never published, so the banner recomputes the same predicate modeld feeds the model, and requires `carControl.latActive` so it appears only while openpilot is the thing steering |
| `openpilot/selfdrive/selfdrived/events.py` | `EventName.lateralEngageOff` — the half-engagement banner; the turn-desire banner |
| `openpilot/selfdrive/test/process_replay/process_replay.py` | `moonpilotState` in plannerd's `pubs` |
| `openpilot/selfdrive/ui/ui_state.py` | `moonpilotState` subscription |
| `openpilot/selfdrive/ui/onroad/augmented_road_view.py` | half-engaged border color from `moonpilot/ui/onroad.py`; the disengaged border color moves off blue; the offroad-mode hold gesture |
| `openpilot/selfdrive/ui/onroad/model_renderer.py` | lead path draw (tizi) |
| `openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py` | the offroad-mode hold gesture (mici) |
| `openpilot/selfdrive/ui/mici/onroad/model_renderer.py` | lead path draw (mici); half-engaged lane-line color |
| `openpilot/selfdrive/ui/mici/onroad/alert_renderer.py` | turn-desire banner icon and height (mici) |
| `openpilot/selfdrive/ui/layouts/home.py` | brand string (tizi) |
| `openpilot/selfdrive/ui/mici/layouts/home.py` | brand label (mici) |
| `openpilot/selfdrive/ui/layouts/settings/settings.py` | `PanelType.MOONPILOT` + panel from `moonpilot/ui/settings.py` |
| `openpilot/selfdrive/ui/mici/layouts/settings/settings.py` | settings entry + panel from `moonpilot/ui/settings_mici.py` |

Pick by need — reuse a seam, never invent one:

- Setting or small persisted value → a row in `moonpilot/params_keys.h`, named `Moonpilot*`. Anything bigger goes to a root from `moonpilot/paths.py` — see **Storage** below.
- Long-running work → `MOONPILOT_PROCS` in `moonpilot/procs.py`. The name must be unique (`test_manager.test_duplicate_procs`).
- State other components observe → fill in `MoonpilotState`, then publish `moonpilotState`. Add a row to `openpilot/cereal/services.py` only once a publisher exists — that file then joins the table above and `ALLOWED`.
- Native binary → no seam yet: add a `SConscript([...])` line to `SConstruct`, with its row in the table above and in `ALLOWED`.
- UI → a panel under `moonpilot/ui/`, registered in **both** UI trees.

## Features

A feature is code under `moonpilot/` reached through an existing seam. Whether it also gets a user-facing toggle is a separate call:

- **Shape every seam to delegate, not replace.** `brand = moonpilot_brand() or "openpilot"  # moonpilot` leaves upstream's value in the line, so stock behavior still exists to fall back to. `brand = "moonpilot"` deletes it, and there is nothing left to toggle back to. Both are one line; only one keeps the option.
- **Add a toggle only for behavior you would actually flip.** Every toggle is a second code path you now have to keep working. A feature that is always on costs nothing to leave always on — most should be.
- **Fork identity takes no toggle either.** `brand()`, `version()` and `is_fork_build()` in `moonpilot/features.py` are always-on fork facts the seams read directly. `is_fork_build()` is the one of them that gates behavior: comma tests comma's branches, so upstream's "WARNING: This branch is not tested" startup banner would otherwise sit on the road at the top of every drive, and the seam swaps in upstream's normal startup alert instead.

When a feature does get one, it is a param plus a control, all inside `moonpilot/`:

1. A row in `moonpilot/params_keys.h` — `{"Moonpilot<Feature>", {PERSISTENT, BOOL, "1"}}`. The third element is the default; `"1"` means on.
2. A control in the moonpilot panel: tizi (`moonpilot/ui/settings.py`) uses `toggle_item(...)` with a callback doing `params.put_bool(key, state, block=True)`; mici (`moonpilot/ui/settings_mici.py`) uses `BigParamControl(text, key, description=...)`, which writes the param itself.

   The two panels refresh by different mechanisms, and it is the opposite of the obvious guess. tizi's `title`, `description` and `enabled` take callables that `list_view` re-resolves on every render, so a tizi row needs no refresh loop at all. mici's `set_value` and `set_checked` are pushed rather than resolved, so a mici row whose state can change outside the panel does need one — a `_update_rows()` called from `show_event()` and registered with `ui_state.add_offroad_transition_callback(...)`, the loop upstream's mici `developer.py` uses, plus a re-run from `_update_state()` whenever `ui_state.CP` is a different object: the car gate can change under the panel, and `ui_state` re-parses `CarParams` on its own parameter thread.
3. A read in fork code, with `params.get(key, return_default=True)` — **not** `get_bool`.

That read is the trap: `get_bool` ignores the declared default and reports off for an unset param. The declaration only becomes true because `openpilot/system/manager/manager.py` seeds every unset param from its default at boot. On device both work; in a bare script or a test outside the manager, only `return_default=True` does.

Gate anything that changes driving behavior on `ui_state.is_offroad()` — the `enabled=` argument — and if it needs a restart to take effect, say so in the description the way the alpha-longitudinal toggle does.

Feature toggles go in the **moonpilot panel**, never as new sidebar entries: the tizi sidebar already reaches y 1070 of 1080 with 7 entries, so an 8th clips. The panel body scrolls and takes as many rows as you like — upstream's Developer panel carries 8 in the same widget.

### Storage

`/data/openpilot` is git state — the updater runs `git reset --hard` in the finalized overlay and the installer moves a fresh clone over the directory — so **nothing fork-owned is ever written inside the checkout**. `moonpilot/paths.py` holds the roots, and no upstream seam is involved in using them:

- **A scalar or small JSON value** → a `Moonpilot*` param row. Persistent, atomic, readable by every process. This is the answer for almost everything.
- **Anything bigger or unbounded** → `data_dir(feature)`: `/data/moonpilot/<feature>` on device, `~/.comma/moonpilot/<feature>` on PC, created on demand. Survives updates; a factory reset wipes it.
- **Must survive a factory reset** — a license, an identity, a calibration → `persist_root()`, `/persist/moonpilot`, on the partition a reset leaves alone. Keep it small; the dongle id and RSA key live there too.
- **Per-drive artifacts** → inside the route directory under `Paths.log_root()`, the only tree the drive deleter reclaims when space runs low.

Two traps: loose files in the params directory are unlinked by `Params::clearAll`, which deletes everything there that is not in the key list; and the drive deleter only frees space under `realdata`, so a store in `data_dir()` shares the partition with driver footage and needs its own bound — a size cap, a ring buffer, or retention.

### Dependencies

The device runs the system `python3` straight out of the read-only AGNOS image with `PYTHONPATH` pointed at the checkout: no venv, no `pip`, no `uv`, and nothing in the boot path that would ever read `uv.lock`. `tools/setup_dependencies.sh` — the script that curls uv and runs `uv sync --frozen` into `.venv` — is called only from `tools/op.sh`, so it is the dev-PC and CI flow and it never runs on a car. `uv.lock` therefore describes the PC environment, not the device's, and a module the device needs is not merely slow to arrive: it is absent, and importing it fails where the import is.

That makes the fork's dependency surface the **init path**: whatever is reachable at import time from the modules a seam pulls in — `moonpilot.procs` from the manager at boot, `moonpilot/ui/settings*.py` from the UI at start, `moonpilot.lead` from `longitudinal_planner` (plannerd) and the renderers, `moonpilot.engage` from `card` and `selfdrived`. Under those, import only the standard library, `numpy`, and `openpilot`'s own modules.

Anything else is a runtime import, in one of two shapes:

- **A `PythonProcess` under `MOONPILOT_PROCS`.** The manager imports the process table and nothing behind it, and its `launcher` imports the module inside the child, so a module that fails to import costs that process and no more. The shape for code that a given device may simply never be able to run.
- **An import at the point of use**, inside the function or the `main()` that needs it, with the failure turning the feature off instead of propagating. The shape for an optional capability inside code that must keep running.

The fork side of a seam still has to return upstream's value when the module is missing — see **delegate, not replace** above. Never an import on the init path, and never a row in upstream's `dependencies`: that list only goes down (`scripts/lint/check_dependencies.py` fails against a fixed budget), `uv.lock` is not in `ALLOWED` so the edit would be rejected anyway, and — the reason that actually matters — the device never installs from `uv.lock` at all. A dependency the fork genuinely needs goes through `requires` and the fork's own lock, in the two sections below.

### The gate

A feature that needs a module the lock does not carry declares it: `Feature(..., requires=("PIL",))` in `moonpilot/features.py`. `requires` holds **importable module names**, not install specs. `available(feature)` is `deps.available()` over each of them, `wanted(feature, params)` is the param read, and `enabled(feature, params)` is the two ANDed — so the seam keeps calling `enabled()` unchanged, and a feature whose dependency is missing is off everywhere at once without the planner or the UI needing to know why.

Three consequences:

- An unavailable feature's row is **disabled with the reason where the driver will see it**: in the description for tizi (`moonpilot/ui/settings.py`), in the value line for mici (`moonpilot/ui/settings_mici.py`) — a disabled mici widget cannot open its long-press dialog, so its description is unreachable.
- A feature's `PythonProcess` is gated on the same predicate. `ensure_running` never restarts a child that exited while `should_run` was still True — `start()` no-ops while `self.proc` is set (`openpilot/system/manager/process.py:142-147,166-171`) — so a proc that dies is gone until the next boot. Gating it on the same availability check is what gives it a lifetime that tracks the feature's: started when there is work, stopped the moment there is not, and long-lived processes loop internally rather than exiting.
- Availability is `importlib.util.find_spec`, never a real `import`. The check runs on the init path — `moonpilot.procs`, both panels, `moonpilot.lead` — where importing the module is exactly the thing that must not happen.
- **A feature can also be blocked by the car, and that gate is a second one** because it cannot live in the same place: `available()` takes no `CarParams`, and the seams that call it are running precisely when the answer is needed. So a car requirement is a function beside the requirement, `car_unavailable_reason(feature, CP)` in `moonpilot/engage.py` — the reason string for a car that cannot run it, `None` otherwise, with the private `_unsupported_reason(CP)` as the single source the gate itself reads, so `_available(CP)` and the panel row cannot disagree (`test_engage.py` asserts exactly that over every out-of-scope car). It lives in `moonpilot/engage.py`, where `LATERAL_ENGAGE`'s own requirement already is, rather than as a per-feature branch inside both panels, which keeps the panels generic over `FEATURES`. `ui_state.CP` is `None` until `carParams` lands, which reads as "say nothing" rather than as a verdict on the car. Three features have one so far. `LATERAL_ENGAGE`'s is a safety gate — the `_unsupported_reason(CP)` that `_available` also reads, so a dashcam car, or a car whose PCM does not own its ACC, sees *not in dashcam mode* or *stock ACC only* instead of a row that reads on and does nothing. `TORQUE_LATERAL` (*torque-steered cars only*) and `LONGITUDINAL` (*openpilot-longitudinal cars only*) are panel-facing only: each names a car whose controlsd branch the fork's seam is not in, so the feature is inert rather than gated, and nothing but the row's reason changes. The two gates say the same kind of thing to the driver — tizi puts either in the bold description, mici in the value line — so a feature with both (none today) would report the dependency first, the device being the thing that can change.

### How a dependency reaches the device

There is no venv, no `pip` and no `uv` on device: third-party packages come from the read-only AGNOS image, and `tools/setup_dependencies.sh` only ever runs where there is a network. `/data` is the only writable tree, and it is not on `sys.path`. Nothing inside the checkout survives an update either — the updater runs `git clean -xdff` and `git reset --hard`, then swaps the whole directory.

So `moonpilot/deps.py` builds the environment beside the checkout, not in it: it bootstraps a pinned static `uv` into `/data/moonpilot/deps/bin`, installs `moonpilot/deps.lock` into a release directory named for the lock's own fingerprint (`/data/moonpilot/deps/releases/<sha256 of the lock, first 16 hex>`), points `/data/moonpilot/deps/current` at that release with **one** `os.replace`d symlink once the install has succeeded, and **appends** `current` to `sys.path`. Promotion is that single atomic step and a failed install removes only the release it was writing, so a half-populated package tree is never what a feature imports. Append, never insert: the fork supplies what is absent and never shadows what the OS ships.

That append cannot only happen at import, because the timing is wrong: the UI, plannerd and the manager all import `deps.py` early in the boot, when the tree may not exist yet, and then run for hours. So `activate()` also runs inside `available()` — a package `depsd` installs mid-boot becomes visible to processes that were already running, instead of staying invisible until a restart. A feature module that imports its package at the top level is covered by the same two mechanisms: it must import `moonpilot.features` or `moonpilot.deps` above the third-party import, and a manager-spawned process is forked from a manager that has already called `available()` in its `should_run` evaluation, so the child inherits the wired-up path.

`moonpilot/depsd.py` is the process that does it — registered in `MOONPILOT_PROCS`, it waits on `deviceState.networkType`, skips while `networkMetered`, and backs off exponentially so a device that cannot succeed is not retrying in a tight loop. It publishes one `CLEAR_ON_MANAGER_START` string, `MoonpilotDepsStatus` (`ready` / `waiting-network` / `waiting-metered` / `installing` / `error`, the detail naming the missing modules), and the moonpilot panel's `dependencies` row in both trees reads it through `deps.status_text()`; that row's action writes a one-shot `MoonpilotDepsRequest`, which authorises **one** attempt over a metered connection — the driver's own escape from the ceiling below rather than the fork's decision, consumed only by the attempt that uses it.

Adding a dependency is three things, and they must agree: a `Requirement(module, spec)` row in `moonpilot/deps.py`, the matching hash-pinned line in `moonpilot/deps.lock`, and the module name in the consuming feature's `requires`. `moonpilot/tests/test_deps.py` fails if any of the three disagree. **The lock is generated for the device, not for the PC**: a host-resolved lock carries only x86_64 wheel hashes, and `--require-hashes` on the device then cannot select an aarch64 wheel — so the command recorded in `moonpilot/deps.lock`'s header pins `--python-platform aarch64-unknown-linux-gnu --python-version 3.12` and is the one to re-run. Its first row is `cryptography`, for the catalog signature check (`moonpilot/docs/models.md`), which is what turns the mechanism on: the device fetches it on the first unmetered network, and until then `CATALOG_SIGNATURES` reads as unavailable-with-reason.

Two ceilings, named rather than hidden: a device that only ever sees a metered connection installs nothing until the driver asks for the retry above, and a factory reset wipes `/data/moonpilot`, so the packages are fetched again on the next unmetered network.

### A binary the device doesn't ship

Some capabilities are a native binary rather than a Python package — tailscale is the one so far, `moonpilot/tailscale.py`. The shape is a download through `moonpilot/fetch.py` (sha256 verified before anything runs, `os.replace` for `ETXTBSY`), installed under `data_root()` never the checkout, supervised offroad-only with `sudo -n`, and reported through the one `MoonpilotTailscaleStatus` param. The supervisor must kill its own child on exit and both trees' QR dialogs are fork-owned on purpose. Read `moonpilot/docs/tailscale.md` before touching any of it.

### The lateral controller

`moonpilot/latcontrol.py` is the fork's own torque lateral controller, and the seam in `openpilot/selfdrive/controls/controlsd.py` is a factory call that returns `None` when the toggle is off, so upstream's controller stays on the line as the fallback. Three things about it are not obvious:

- **It is chosen once, at construction.** controlsd builds one `LatControl` in `__init__` and never revisits it, so the toggle takes a restart — the feature description says so, and the row is offroad-only.
- **It only exists for torque-steered cars.** The seam sits in the `lateralTuning == 'torque'` branch, so angle- and curvature-steered cars never reach it and the toggle is a no-op there.
- **It logs into upstream's `LateralTorqueState`**, with `version` in a fork band (1000) so a log says which controller produced it. No new cereal struct, so the plotjuggler and jotpluggler torque-controller layouts keep working.

The control law is upstream's family — feedforward in lateral acceleration, PI on a response-aligned error, friction compensation — because that is the part real miles paid for; the gains and every other number are fork-owned, in that file's header block. The PI setpoint stays delay-matched while the request tightens or holds, but reads at the existing jerk-lookahead sample when its magnitude is unwinding. (That delay is upstream `lagd`'s `lateralDelay` — on a sub-50-mph drive a carried-over estimate the drive cannot validate; see `moonpilot/docs/curvature.md`.) A sign change releases the old turn to zero without applying opposite-sign feedback before the plant delay. That asymmetry starts releasing torque before the downstream car-specific rate limiter can add another measured ~0.3 s of unwind lag, without anticipating a new tightening request. `moonpilot/tests/test_latcontrol.py` pins both sides, the roll and friction terms in the feedforward, the saturation timer, and an integrator that does not survive a disengagement.

### Response-aligned path steering

`moonpilot/curvature.py` aligns the model's path with when steering is expected to respond, as a bounded correction toward the path anchored on `modelV2.action.desiredCurvature`; the seam is `moonpilot_curvature()` in controlsd, chosen once at construction, so the offroad-only row takes a restart. Stale, future-dated, malformed or out-of-grid input returns `desiredCurvature` bit-for-bit. The response horizon runs on upstream `lagd`'s `lateralDelay`, which sub-50-mph driving cannot validate — a coverage statement, not a defect. `moonpilot/tests/test_curvature.py` pins it. Full details: `moonpilot/docs/curvature.md`.

### The longitudinal strategy

`moonpilot/longitudinal.py` is the fork's own planner and `moonpilot/longcontrol.py` its own acceleration controller; the seams are `moonpilot_longitudinal_planner()` in plannerd and `moonpilot_longcontrol()` in controlsd, both factory calls returning `None` when the toggle is off, so upstream's MPC and PI loop stay on the line. The policy is three closed-form candidates (`min` wins): a spacing regulator with a standstill floor under the headway, a TTC approach term, a stopping floor in the relative frame with a bounded lead-brake sustain — plus cruise, grade coasting, curve speed, corridor squeeze braking from `modelV2.roadEdges`, and the model's braking ask through a deadband that, once past it, holds a stop the model began until the model asks to go. Both seams are chosen once at construction, so the row takes a restart and only applies to `openpilotLongitudinalControl` cars. `moonpilot/tests/test_longitudinal.py` and `test_longcontrol.py` pin it, and end by running upstream's maneuver suite against the fork planner. Full details of every non-obvious property: `moonpilot/docs/longitudinal.md`.

### The rolling-window ego correction

`moonpilot/slam.py` blends the wheel-speed prior against camera odometry's forward translation over a 5 s window and publishes the disagreement as `moonpilotState.egoCorrection`, which the longitudinal planner adds to `v_ego` — one added number, so the pass-through case (off, invalid, stale, dead publisher) is the stock planner exactly. One process publishes `moonpilotState` and it is `leadd`; a second publisher would take the leads down with it. The correction is `smoothed minus the raw prior` (sign load-bearing), the gate is live not restart-gated, and a standstill gets nothing. `moonpilot/tests/test_slam.py` pins the sign, clamps, staleness and calibration ingest. Full details: `moonpilot/docs/slam.md`.

### Lateral-only engagement

`moonpilot/engage.py` is the fork's half-engaged state: openpilot steers from the cruise main switch, the car's ACC takes speed only once the driver sets it. It is the fork's one feature that reaches into `opendbc` and `panda`: a second safety flag `controls_allowed_lateral` that only ever gates steering, armed on the main switch (ARM_SWITCH) or openpilot's own heartbeat (ARM_HOST), with `pcmDisable`/`pedalPressed`/`buttonCancel` suppressed and an authoritative disable latched until a re-arm gesture. `ActuatorGate` gates `CC.latActive`/`CC.longActive` on the panda's grants, so half-engagement on an openpilot-longitudinal car is steer-only. Ceiling: every brand whose interface reports `CP.pcmCruise`, minus `ToyotaFlags.UNSUPPORTED_DSU`. Read the whole chain before changing any part of it: `moonpilot/docs/engage.md`.

### Offroad mode

`moonpilot/offroad.py` holds the device offroad while the ignition stays on, so offroad-only work (settings rows, uploads, SSH, tailscale) is reachable without turning the car off. The seam is one dict member, `onroad_conditions["moonpilot_onroad"]`, plus the refresh above `ign_edge`; the mode is the `MoonpilotOffroad` param with `CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON` (deliberately not `CLEAR_ON_OFFROAD_TRANSITION`). Two entries share one `parked()` gate; leaving is never gated; the row is not in `FEATURES`. Cost: while set, no assist and no camera logging, and the car moving does not end it. `moonpilot/tests/test_offroad.py` pins the flags and the two literals in `hardwared.py`. Full details: `moonpilot/docs/offroad.md`.

### Boot rollback

`launch_chffrplus.sh` swaps the checkout for a finalized update by moving the live tree to
`/data/safe_staging/old_openpilot` and the finalized tree into its place, then `exec`s itself — and
upstream's own `# TODO: restore backup?` is what happens when the swapped tree cannot start its
manager. `moonpilot/boot.sh` closes that gap; `moonpilot/boot.py` is the manager's half.

- **The decision is a per-boot token, not a clock.** `/proc/sys/kernel/random/boot_id` is written
  into `moonpilot_swap` at swap time and `moonpilot_boot_ok` by the manager after the first
  `ensure_running`. Same token on re-entry means “this boot just swapped; do nothing”; matching
  health token on a later boot proves success and clears only the swap record; missing/mismatched
  evidence restores the backup and keeps the failed tree as `failed_openpilot`. Missing/empty
  evidence is otherwise a no-op. `MoonpilotRollback` carries the sentence to both settings trees.
- **PC proof has a limit.** `moonpilot/tests/test_boot.py` drives all branches and the marker
  contract in a temporary root; only a device can prove the real updater swap, manager startup and
  reboot transition.

### The model marketplace

A driver may replace the model this car drives: panes in `moonpilot/ui/models.py`/`models_mici.py`, rules in `moonpilot/models.py`, catalog in `moonpilot/modelcatalog.py`, worker in `moonpilot/modelsd.py`, compiler in `moonpilot/modelbuild.py`, runtime in `moonpilot/modelruntime.py`. A selection is a string (`""` bundled, else a 64-hex digest), committed once at boot by `commit_boot_selection` into `MoonpilotModelsBoot`, which the model processes read and nothing else — the row takes a restart. A load failure falls back to the bundled model and takes the whole runtime with it, action decode included. Only four protocols are selectable; the worker never exits on its own and never compiles onroad; nothing on the boot path touches the network. Three ceilings: 300 MB per artifact, 8 GB store, 2 GB free space left alone. `moonpilot/tests/test_models.py`, `test_modelcatalog.py` and `test_modelruntime.py` check it without a car. Full details: `moonpilot/docs/models.md`.

### Forked opendbc and panda

`panda` and `opendbc_repo` point at `jjolano/moonpilot-panda` and `jjolano/moonpilot-opendbc` (relative URLs in `.gitmodules`), each with `upstream` still on comma and fork work committed on `master`. The safety code compiles into the panda firmware, so a fork-owned safety layer means fork-owned submodules, and nothing else reaches them. They sync the same way the superproject does — merge, never rebase, no force-push — with their own `master`:

    cd panda && git fetch upstream && git merge upstream/master
    cd ../opendbc_repo && git fetch upstream && git merge upstream/master

Then record the merge with an ordinary `git add panda opendbc_repo` in the superproject. `git submodule update` on the device fetches the default refspec, so the recorded SHA must be reachable from the fork's default branch — a fork commit left on a side branch does not reach a device. That makes the order matter at the other end too: **push each submodule before the superproject.** A device updates by fetching the superproject and then the submodules, so a pointer pushed ahead of the commit it names is a device that cannot check its tree out; submodules are independent repositories with their own `origin`, and no superproject push ever carries them. Both are forks of commaai's repos in the same fork network as every other account, so `gh repo fork` will refuse once one exists; creating a plain public repo and pushing to it is the equivalent, and the only requirement is that `master` carries the commits.

One consequence of the fork owning the safety layer lands in panda rather than here: **panda's own `./test.sh` cannot build the firmware.** Its `setup.sh` creates a venv whose `pyproject.toml` pins `opendbc @ git+https://github.com/commaai/opendbc.git@master`, so that venv holds *upstream* opendbc with no `opendbc/safety/moonpilot/`, and panda's standalone `scons` stops on `'controls_allowed_lateral' undeclared` before any test runs. Build from here — `tools/op.sh build`, which puts `opendbc_repo` on the include path — or give the standalone run the fork's opendbc: `PYTHONPATH=../opendbc_repo panda/.venv/bin/python -m unittest discover -s panda/tests`. Never repoint panda's `pyproject.toml` at the fork; that line is upstream's, and `panda/AGENTS.md` records this too.

`docs/SAFETY.md` (the checkout root) states four things about forks. Two are unconditional and bind every fork — **do not disable or nerf driver monitoring**, and **do not disable or nerf the excessive-actuation checks**; two attach to editing `opendbc/safety/`, which this fork does:

- **the fork cannot use the openpilot trademark**, which is what makes `moonpilot`'s own brand load-bearing rather than cosmetic.
- **the full safety test suite must be preserved and pass, including any new coverage the fork's changes require.** `opendbc/safety/tests/test.sh` enforces the second half mechanically — 100% line coverage of every non-libsafety safety file — so new C in that tree that no test reaches is a failing gate, not a warning. `moonpilot/tests/test_engage.py` holds the openpilot half of the same feature; the safety half is `opendbc/safety/tests/lateral_engage_common.py` — the mixin both arming modes are tested through, with `LATERAL_ENGAGE_ARM` naming the one a class drives — the mixin-derived count is twenty-six classes across twenty test modules — one or more per brand in `LATERAL_ENGAGE_FLAGS`, across that brand's platforms. `opendbc/safety/tests/mutation.py` is the second half of that gate: it mutates the safety C and fails if the suite leaves a mutant alive, so a new branch in the rule needs a test that fails when the branch is broken, not merely one that passes. **"Preserved" is a statement about the run, not about the diff**: the suite is the evidence, and the one upstream test file whose existing lines the fork changes is `opendbc/safety/tests/common.py`, +28/−2: the new classes are additions; the existing upstream Volkswagen-PQ and GM `issubset({...})` sets are extended to cover their new classes, and `lateral_engage_pairs` is an additional explicit exclusion mechanism for lateral-engage classes whose tx lists overlap their platforms' by construction. Every other test file gains classes and changes nothing.

If you touched the safety layer, the superproject's own loop proves nothing about it: the gates live in `opendbc_repo` (`opendbc/safety/tests/test.sh`, `mutation.py`, and the MISRA pair) and in panda (`PYTHONPATH=opendbc_repo panda/.venv/bin/python -m unittest discover -s panda/tests`). `opendbc_repo/AGENTS.md` has the details, including the two ways the coverage gate lies — gcovr is not in that venv, and it consumes the coverage data it reads.

The two unconditional prohibitions are held by absence, and the one place that is a judgment call is named here rather than left to be discovered. `openpilot/selfdrive/monitoring/` and `openpilot/selfdrive/selfdrived/helpers.py` carry no fork edits at all — no fork commit has touched them, they are not in `ALLOWED`, and `moonpilot/tests/test_upstream_touches.py` refuses both an edit and the row that would sanction one (`NEVER_ALLOWED` and `test_the_unconditional_safety_paths_stay_unreachable`, which also fails if upstream renames either path out from under the guard). The exception is the monitoring **model**: `openpilot/selfdrive/modeld/dmonitoringmodeld.py` is a seam, and the model marketplace can put a different network behind it. That is a driver choosing a model rather than the fork weakening the check — admission requires the heads `openpilot/selfdrive/monitoring/policy.py` actually reads, stock is the default, and the catalog's own `semantics_unverified` rides into the panel verbatim — but nothing in this tree verifies a substituted network's *accuracy*: the gate is structural, so a driver who picks a worse DM model has picked a worse DM model. Read that before loosening how DM models are admitted.

### The forks' own context files

`panda/AGENTS.md` and `opendbc_repo/AGENTS.md`, each with a `CLAUDE.md` symlink beside it, are the two places the fork's rules are restated — and they restate nothing: each is one screen that ends in a heading whose body is the line `@../AGENTS.md`. The harness expands an `@path` import inline before injection, so a session started inside a submodule gets this file whole.

They exist because discovery walks up only as far as the **repository root**, and a submodule is one: `panda/.git` and `opendbc_repo/.git` are `gitdir:` pointers, so a session whose cwd is inside either one finds that repo's own root and stops, never reaching this file. From the superproject the same files are the opposite of hidden — a deeper `AGENTS.md` is listed in the `<dir-context>` block that tells an agent to read it before editing that directory.

What belongs in them is the repo-local delta: what the fork changed there, the invariants that must not break, and the gate commands. What must never go in them is a copy of anything here — a second copy of a claim is a second thing to keep true, and the two drift.

### When moonpilot and upstream converge

- Upstream implements something moonpilot already has → delete moonpilot's version and its toggle row. A switch between two identical behaviors is rot, and it is how a fork accumulates dead weight.
- Moonpilot's version is the one to keep → keep it, and let the seam carry the choice.
- Upstream claims a reserved struct or param name → upstream's wins; the fork moves.

### The registry

`moonpilot/features.py` drives the panel. Adding a feature is three things: a `Feature(...)`, a row in `moonpilot/params_keys.h`, and the code behind it — plus, when it needs a module the device does not ship, `requires=(...)` on the `Feature` and the matching rows in `moonpilot/deps.py` and `moonpilot/deps.lock` that **The gate** and **How a dependency reaches the device** describe. That fourth one is the easy one to forget, because the feature works on your PC without it: the module is already importable there.

```python
LEAD_LATERAL = Feature(
  key="MoonpilotLeadLateral",     # the params_keys.h row, which carries the default
  title="lead lateral prediction",  # what the panel shows
  description="...",                # shown as the item's description / long-press help
)
FEATURES: tuple[Feature, ...] = (LEAD_LATERAL,)
```

Both panels build their rows by iterating `FEATURES`, so a new row appears in tizi and mici without touching either panel file. Read a feature's state with `enabled(feature, params)`, never `get_bool`, for the reason above. When a feature's decision is more than a boolean, add a function next to `enabled` — the seam calls it and no param is needed: `brand()` and `version()` are the pattern, fork identity with nothing to flip.

## Editing upstream

- Seam lines only. Never reformat, reorder, or tidy adjacent upstream code — an incidental whitespace fix turns a one-line merge conflict into a whole-file one.
- **The same rule holds inside `opendbc_repo` and `panda`, where nothing catches a violation.** `test_upstream_touches.py` reads the superproject's diff, so a formatter run over a submodule's upstream file lands as a whole-file delta against comma that no check sees and every future merge pays for. Stage what you touched, not what a tool rewrote.
- When upstream's types or signatures clash with a seam, adapt the fork side. `MOONPILOT_PROCS` is annotated with upstream's process union rather than widening upstream's `procs` list, so `procs += MOONPILOT_PROCS` type-checks with no upstream edit.

### Seam markers

Every fork region inside an upstream file carries the marker `moonpilot seam, see AGENTS.md`, in that file's comment syntax (`#`, `//`, or capnp's trailing `#`) — or, where that line's syntax has no room for a comment, the fork's name in the value. The marker is what tells anyone — human or agent — that the region is fork-owned and why it is there.

    brand = moonpilot_brand() or "openpilot"  # moonpilot seam, see AGENTS.md

**One marker per region, not one per line.** Where it sits inside the region is dictated by syntax, and the check is region-level: `moonpilot/tests/test_upstream_touches.py` runs `git diff -U0` against the merge base, splits the fork's work into contiguous runs of added lines, and requires every run in an allowed upstream file to carry the prose marker — or, in the one path `VALUE_MARKED` names, the value form below. Three shapes that rule takes:

- **New code carries it at the head** — the `def`, the struct, the new statement:

      def _draw_lead_path(self):  # moonpilot seam, see AGENTS.md

- **A rewritten multi-line statement carries it at the end of that statement**, because `#` cannot sit inside a call's parentheses — which is the shape most of the fork's seams in upstream Python take:

      sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', ..., 'radarState',
                                'moonpilotState'],  # moonpilot seam, see AGENTS.md

- **A line whose syntax has no room for a comment carries the fork's name in the value**, with the prose marker on its own line beside it — git-config's `.gitmodules`:

      # moonpilot seam, see AGENTS.md
      url = ../moonpilot-panda.git

That last shape is run by `-U0`: git-config only takes `#` on a line of its own, and `-U0` gives that line its own region, so the value it explains stands alone and `.gitmodules` is the **one** path the test accepts a value form for (`VALUE_MARKED`, matched on `moonpilot-`, the fork's repo URLs).

The test matches the full prose marker `moonpilot seam`, and deliberately not the bare fork name. `'moonpilotState'` is a service name in plannerd's subscription list: a region whose real marker was dropped — by a formatter, or by a hand edit that moved the line — would still contain it, and matching the bare name would report that region as marked. `opendbc/safety`'s `MOONPILOT_*` identifiers are the same trap in C. So a region passes on the prose marker, or on the value form only where `VALUE_MARKED` lists its path; `MOONPILOT_PROCS` in `process_config.py` is not an exception, because that line carries the prose marker like any other.

Two things follow. A region with no marker in it is either a region needing its marker or an upstream edit that should not be there. And because this check reads lines rather than file names, a formatter run over an already-allowed upstream file fails it — the one guard in the tree that sees an edit *inside* a file the seam table has already approved.

A region that only *removes* upstream code is the same question with no line to answer it on — no addition, so no marker can be carried. That direction has its own assertion in the same test class, and it separates the two shapes: a removal-only region inside a file that survives is refused, while a whole-file deletion is excused — git labels it `deleted file mode`, and the file check is what sanctions it, since the deleted path still has to be an `ALLOWED` one. So deleting an upstream file the fork never hooked fails, and retiring a seam it owns passes.

A fork's *own* deletion keeps its row, and keeps it for good. `base` is `merge-base(HEAD, upstream/master)`, which by construction contains no fork commit, so the deleted path never leaves the diff and dropping the row would fail the file check. The sync table's "deletes a file the fork hooked" row is upstream's deletion instead: that commit lands in `base` once the merge is made, the path leaves the diff, and the row goes with it.

Capnp seams carry the invariant they must not break, since a wrong edit there corrupts recorded data:

    moonpilotState @107 :Custom.MoonpilotState;  # moonpilot seam: do not change @107 or which struct it points to. See AGENTS.md before editing.

### Seams carry intent you cannot infer — ask

A seam is one line, but the decision behind it — why the fork diverged, and whether that divergence is still wanted — is not in the code. Before you delete, restructure, or "clean up" a seam, or fold a fork behavior back into upstream's:

- If the intent is legible from `moonpilot/` and the docs, proceed.
- If it is **not** — the fork changed behavior and you cannot tell whether that was deliberate and still wanted, or which side should win after an upstream rework — **ask the developer** with the question tool. Name the file and line, quote the seam, and give the concrete options (keep fork / take upstream / keep both behind a toggle).

Never silently drop a seam to make a merge or a lint error go away. Deleting fork behavior is the developer's call, not an inference to be made from a clean-looking diff.

## Sync with upstream

Fork work lives on `master`. Merge, never rebase; no force-push.

    git fetch upstream
    git merge upstream/master    # conflicts land on seam lines; `grep -n moonpilot <file>` shows the fork's side
    python3 -c "import moonpilot.engage, moonpilot.procs, moonpilot.ui.settings, moonpilot.ui.settings_mici"
    tools/op.sh lint
    tools/op.sh test moonpilot
    tools/op.sh build

Then bump `COMMA_VERSION`'s upstream part to the new upstream version, leaving the `-moonpilot.<n>` suffix alone. See **Versioning** below. The two forked submodules sync on their own schedule — see **Forked opendbc and panda** — and `git submodule update --init --recursive` after a merge is what puts their recorded commits in place.

### Versioning

`COMMA_VERSION` in `openpilot/common/version.h` carries upstream's version and the fork's own, and the two bump separately:

    #define COMMA_VERSION "0.11.2-moonpilot.1"
    #                          ^        ^        ^
    #                          |        |        └ fork revision: count up per fork release
    #                          |        └ fork marker: never dropped
    #                          └ upstream's version: taken verbatim on merge

- **Merging upstream** moves the upstream part to upstream's new version and leaves the fork revision alone. Upstream's release edits this same line, so the merge conflicts — that conflict is the reminder to do it.
- **Releasing the fork** bumps only the fork revision (`.2`, `.3`, …). It is a plain counter, not semver: it answers "which fork state is this?" and nothing orders by it — the updater works on git commits, not version strings.
- **Keep the shape** `<upstream>-moonpilot.<n>`, with no spaces. `updated.py` composes `"version / branch / commit / date"` and mici's `_split_description` requires exactly four parts split on `" / "`, so a version containing `" / "` silently renders as blank in that panel. `moonpilot/tests/test_version.py` fails if the shape drifts.
- Nothing else has to change when the suffix moves. Upstream already ships the accessor for reading the upstream part alone — `OpenpilotMetadata.short_version` is `version.split('-')[0]` (`openpilot/common/version.py:75`) — and `loggerd`'s own test asserts the logged version equals `get_version()`, so the fork string flows through unchanged.

The fork marker also does **not** exist to disguise the fork as upstream. `OpenpilotMetadata.comma_remote` and the git origin feed upstream's release metrics, and upstream asks forks not to touch them (`openpilot/common/version.py:79`). The suffix is moonpilot's own identity, not a way to look like comma.

### A clean merge is not a working seam

The `import` line is not decoration. Upstream renaming a symbol the fork imports merges without a single conflict and passes lint:

| Check | What it can see |
| --- | --- |
| `git merge` | upstream editing the same lines as a seam — nothing else |
| `moonpilot/tests/test_upstream_touches.py` | **only fork edits.** After a merge, upstream's changes are inside the merge-base, so the guard is blind to them by construction |
| `tools/op.sh lint` (ty) | fork-side mistakes, not upstream renames: renaming `text_item` in `openpilot/system/ui/widgets/list_view.py` still reports *All checks passed* |
| the `import moonpilot…` line | upstream renames/removals the fork depends on — it raises `ImportError` on that same rename |
| `tools/op.sh build` | the params and capnp seams (compiles `params.cc`, regenerates cereal) |
| UI boot test | a panel or brand seam that no longer constructs |

### Scenarios

| Upstream does | Merge | Action |
| --- | --- | --- |
| Rewrites a line a seam sits on — the brand string, `COMMA_VERSION`, the wheel `packages` list, a `PanelType` entry | conflict | take upstream's version on their line, keep the fork's intent on ours, re-add the `# moonpilot` marker |
| Adds rows/lines near a seam — a param, a proc, a settings panel | usually clean | rebuild and re-check. A new settings panel is clean to merge but shifts the fork panel down the tizi sidebar (it sits at y 960–1070 of 1080); re-verify it is not clipped. The mici scroller scrolls, so it absorbs the extra entry |
| Renames or removes a symbol the fork imports | clean | the `import moonpilot…` line fails. Fix `moonpilot/`, never upstream |
| Deletes a file the fork hooked | modify/delete conflict | re-attach the seam at the nearest equivalent point, then drop the dead row from the table above and from `ALLOWED` — upstream's commit reaches the merge base with the merge, so the path leaves the diff then and the row goes with it |
| Renames or restructures the safety layer the fork forked | clean in the superproject | the fork's `opendbc_repo` merge is where it lands, and `moonpilot/engage.py` is what breaks if a symbol it imports is gone. Merge the submodule first, then the superproject |
| Adds its own `AGENTS.md` | add/add conflict | this file stays fork-owned; fold in anything useful from upstream's |
| Wants a reserved struct or param name the fork also uses | conflict | upstream's ids and names win — move the fork to the next free `CustomReservedN`, never the reverse |

Last 2000 upstream commits, per seam file: `pyproject.toml` (391) moves almost weekly, so it conflicts most; `cereal/log.capnp` (31), `common/params_keys.h` (18) and `scripts/lint/lint.sh` (16) are moderate; `process_config.py`, the UI settings panels, `home.py`, `version.h`, `custom.capnp` and `controlsd.py` have moved 2–7 times. A new seam is a permanent recurring conflict surface — add one only when no existing seam reaches; `SConstruct` (320) would be as bad as `pyproject.toml`.

## Working here

- `tools/op.sh build | lint | test [target]` — build, lint, test. `tools/op.sh --help` lists the rest.
- UI is Python + raylib; tizi (`openpilot/selfdrive/ui/layouts/`) and mici (`openpilot/selfdrive/ui/mici/layouts/`) are separate trees — a fork panel is registered in both.
- Tests run through a unittest loader (`tools/test_runner.py`), so fork tests subclass `unittest.TestCase`.
- Style is enforced by `scripts/lint/lint.sh` and `pyproject.toml` (ruff, ty, codespell); fork code mirrors the conventions of the upstream file it hooks into.
- `scripts/lint/lint.sh` builds its `${ALL_FILES[@]}` checks — codespell, `check_added_large_files`, the shebang checks — from `git ls-files -z openpilot moonpilot`, so **an untracked file escapes them until it is `git add`ed**. `ruff` and `ty` are handed those two directories instead and do see untracked files. A green run therefore means less than it looks like: stage first, then trust it.
- That list stops at `openpilot/` and `moonpilot/`, so `AGENTS.md` — the file every agent obeys — is linted by nothing. `moonpilot/tests/test_agents_md.py` is its only mechanical check, which is the argument for putting anything here that can be checked by a machine into that test rather than trusting the prose.

## Keeping this file true

Every claim here is a statement about code — a mechanism, an ordering, a ceiling — and the next agent acts on it without checking. So this file is part of the change, not documentation written after it: when a change alters a claim, the claim moves in the same commit, and when you add a mechanism you document the part that is surprising, because that is the only part anyone needs.

The failure mode is writing from memory. An earlier revision of this file stated that the manager restarts a process that dies. It does not: `ensure_running` reaps only on the `should_run`-false path, and `start()` no-ops while `self.proc` is set, so a child that exits is gone until the next boot (`openpilot/system/manager/process.py:142-147,166-171,223-238`). The sentence read as an obvious truth and was wrong, and it was wrong in the direction that breaks things — an agent would have built a self-terminating process and watched it die once, silently. Re-read the sentence you are invalidating against the file it describes.

Where a claim is mechanically checkable, pin it with a test rather than trusting the prose. That is already how much of this file is held up — `test_version.py` (the version shape), `test_paths.py` (the fork state roots), `test_upstream_touches.py` (the seam table against `ALLOWED`), `test_deps.py` (the registry agreeing with the lock) — and `test_agents_md.py` extends the same pattern to the fork paths named here — and to the two submodule context files, which must exist, must import this file, and must stay short enough to be pointers rather than a second copy — so a rename, a deletion, or a paste that misses this file fails the suite instead of misleading the next reader.
