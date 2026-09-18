# moonpilot

Personal fork of [openpilot](https://github.com/commaai/openpilot). The name is lowercase everywhere, like `openpilot` and `comma`. `CLAUDE.md` is a symlink to this file.

## The rule

Fork code lives in `moonpilot/`. Upstream files carry **seams** only: the minimum lines that hook into `moonpilot/`, each marked with a literal `moonpilot` comment or identifier. The fork's whole surface on upstream is the table below — keep it that way and `git merge upstream/master` only ever conflicts on seam lines.

New fork behavior: write it under `moonpilot/`, hook it at the seam that already exists. A genuinely new seam means updating this table *and* `ALLOWED` in `moonpilot/tests/test_upstream_touches.py`, which fails on any other upstream edit.

## Seams

| Upstream file | Seam |
| --- | --- |
| `SConstruct` | `SConscript(['moonpilot/SConscript'])` — fork native targets |
| `pyproject.toml` | `moonpilot` in the hatch wheel packages, so `import moonpilot` resolves |
| `scripts/lint/lint.sh` | ruff / ty / `git ls-files` cover `moonpilot/` |
| `tools/test_runner.py` | `moonpilot` in the default test targets |
| `.gitmodules` | relative fork URLs for `panda` and `opendbc_repo` |
| `panda` (submodule) | `HEALTH_FLAG_CONTROLS_ALLOWED_LATERAL` in `board/health.h`, published in `board/main_comms.h` |
| `opendbc_repo` (submodule) | the fork's safety layer: `opendbc/safety/moonpilot/lateral_engage.h`, `PCM_CRUISE_2` in `modes/toyota.h`, the 12 lateral reads in `lateral.h` |
| `openpilot/common/params_keys.h` | `#include "moonpilot/params_keys.h"` — fork params, one row per key |
| `openpilot/system/manager/process_config.py` | `procs += MOONPILOT_PROCS` from `moonpilot/procs.py` |
| `openpilot/cereal/custom.capnp` | `MoonpilotState` (upstream's reserved struct; never change the `@0x…` id) |
| `openpilot/cereal/log.capnp` | `moonpilotState @107` event field; `PandaState.controlsAllowedLateral @38` |
| `openpilot/cereal/services.py` | `moonpilotState` service row |
| `openpilot/common/version.h` | `COMMA_VERSION "<upstream>-moonpilot.<fork revision>"` |
| `openpilot/selfdrive/controls/plannerd.py` | `moonpilotState` subscription; `moonpilot_longitudinal_planner()` picks the longitudinal planner |
| `openpilot/selfdrive/controls/lib/longitudinal_planner.py` | lead danger factor from `moonpilot.lead` |
| `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | `lead_danger_factor` kwarg on `LongitudinalMpc.update` |
| `openpilot/selfdrive/controls/controlsd.py` | `moonpilot_latcontrol()` picks the torque lateral controller, `moonpilot_longcontrol()` the acceleration controller |
| `openpilot/selfdrive/car/card.py` | `moonpilot_engage_safety_param()` before `CarParams` is written |
| `openpilot/selfdrive/pandad/pandad.cc` | `PandaState.controlsAllowedLateral` from the panda health flag |
| `openpilot/selfdrive/selfdrived/selfdrived.py` | `moonpilot_engage()`; `LateralEngage.update()` after every event source; the panda cross-check |
| `openpilot/selfdrive/test/process_replay/process_replay.py` | `moonpilotState` in plannerd's `pubs` |
| `openpilot/selfdrive/ui/ui_state.py` | `moonpilotState` subscription |
| `openpilot/selfdrive/ui/onroad/model_renderer.py` | lead path draw (tizi) |
| `openpilot/selfdrive/ui/mici/onroad/model_renderer.py` | lead path draw (mici) |
| `openpilot/selfdrive/ui/layouts/home.py` | brand string (tizi) |
| `openpilot/selfdrive/ui/mici/layouts/home.py` | brand label (mici) |
| `openpilot/selfdrive/ui/layouts/settings/settings.py` | `PanelType.MOONPILOT` + panel from `moonpilot/ui/settings.py` |
| `openpilot/selfdrive/ui/mici/layouts/settings/settings.py` | settings entry + panel from `moonpilot/ui/settings_mici.py` |

Pick by need — reuse a seam, never invent one:

- Setting or small persisted value → a row in `moonpilot/params_keys.h`, named `Moonpilot*`. Anything bigger goes to a root from `moonpilot/paths.py` — see **Storage** below.
- Long-running work → `MOONPILOT_PROCS` in `moonpilot/procs.py`. The name must be unique (`test_manager.test_duplicate_procs`).
- State other components observe → fill in `MoonpilotState`, then publish `moonpilotState`. Add a row to `openpilot/cereal/services.py` only once a publisher exists — that file then joins the table above and `ALLOWED`.
- Native binary → `moonpilot/SConscript`.
- UI → a panel under `moonpilot/ui/`, registered in **both** UI trees.

## Features

A feature is code under `moonpilot/` reached through an existing seam. Whether it also gets a user-facing toggle is a separate call:

- **Shape every seam to delegate, not replace.** `brand = moonpilot_brand() or "openpilot"  # moonpilot` leaves upstream's value in the line, so stock behavior still exists to fall back to. `brand = "moonpilot"` deletes it, and there is nothing left to toggle back to. Both are one line; only one keeps the option.
- **Add a toggle only for behavior you would actually flip.** Every toggle is a second code path you now have to keep working. A feature that is always on costs nothing to leave always on — most should be.

When a feature does get one, it is a param plus a control, all inside `moonpilot/`:

1. A row in `moonpilot/params_keys.h` — `{"Moonpilot<Feature>", {PERSISTENT, BOOL, "1"}}`. The third element is the default; `"1"` means on.
2. A control in the moonpilot panel: tizi (`moonpilot/ui/settings.py`) uses `toggle_item(...)` with a callback doing `params.put_bool(key, state, block=True)`; mici (`moonpilot/ui/settings_mici.py`) uses `BigParamControl(text, key, description=...)`, which writes the param itself.

   The two panels refresh by different mechanisms, and it is the opposite of the obvious guess. tizi's `title`, `description` and `enabled` take callables that `list_view` re-resolves on every render, so a tizi row needs no refresh loop at all. mici's `set_value` and `set_checked` are pushed rather than resolved, so a mici row whose state can change outside the panel does need one — a `_update_rows()` called from `show_event()` and registered with `ui_state.add_offroad_transition_callback(...)`, the loop upstream's mici `developer.py` uses.
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

### How a dependency reaches the device

There is no venv, no `pip` and no `uv` on device: third-party packages come from the read-only AGNOS image, and `tools/setup_dependencies.sh` only ever runs where there is a network. `/data` is the only writable tree, and it is not on `sys.path`. Nothing inside the checkout survives an update either — the updater runs `git clean -xdff` and `git reset --hard`, then swaps the whole directory.

So `moonpilot/deps.py` builds the environment beside the checkout, not in it: it bootstraps a pinned static `uv` into `/data/moonpilot/deps/bin`, installs `moonpilot/deps.lock` into `/data/moonpilot/deps/site-packages` with `--require-hashes`, and **appends** that directory to `sys.path`. Append, never insert: the fork supplies what is absent and never shadows what the OS ships.

That append cannot only happen at import, because the timing is wrong: the UI, plannerd and the manager all import `deps.py` early in the boot, when the tree may not exist yet, and then run for hours. So `activate()` also runs inside `available()` — a package `depsd` installs mid-boot becomes visible to processes that were already running, instead of staying invisible until a restart. A feature module that imports its package at the top level is covered by the same two mechanisms: it must import `moonpilot.features` or `moonpilot.deps` above the third-party import, and a manager-spawned process is forked from a manager that has already called `available()` in its `should_run` evaluation, so the child inherits the wired-up path.

`moonpilot/depsd.py` is the process that does it — registered in `MOONPILOT_PROCS`, it waits on `deviceState.networkType`, skips while `networkMetered`, and backs off exponentially so a device that cannot succeed is not retrying in a tight loop. Two ceilings, named rather than hidden: a device that only ever sees a metered connection never installs, and a factory reset wipes `/data/moonpilot`, so the packages are fetched again on the next unmetered network.

Adding a dependency is three things, and they must agree: a `Requirement(module, spec)` row in `moonpilot/deps.py`, the matching hash-pinned line in `moonpilot/deps.lock` (`uv pip compile --generate-hashes --output-file moonpilot/deps.lock ...`), and the module name in the consuming feature's `requires`. `moonpilot/tests/test_deps.py` fails if any of the three disagree. `REQUIREMENTS` is empty today, so the mechanism is inert and structurally correct at the same time — the first row is what turns it on.

### A binary the device doesn't ship

Some capabilities are a native binary rather than a Python package — tailscale is the one so far, `moonpilot/tailscale.py`. The shape is the same as above with the module machinery swapped for a download, and `moonpilot/fetch.py` is the shared half: `download()` verifies the sha256 **before** anything executes the payload (it runs as root), and `extract()` writes each member to `<name>.new` and `os.replace`s it into place, because overwriting a running binary is `ETXTBSY` — an upgrade reinstalls over a live `tailscaled`.

The binaries live under `data_root()` for the reason `deps.py` gives, never in the checkout. A `.version` marker beside them is what makes a version bump reinstall; the pinned `VERSION`/`SHA256` in `moonpilot/tailscale.py` is the **only** thing that updates the client, so `tailscale update` is never called and `tailscale set --auto-update=false` is issued once per run — a tailnet-wide auto-update policy would otherwise have tailscaled replace the binaries it is running from and try to restart itself through systemd or init.d, neither of which the fork installs, leaving new binaries on disk, an old process running, and a marker that lies about both. Because `os.replace` leaves the running process on the old inode, an upgrade must also restart the supervised child; it is therefore offroad-only, and a failed upgrade leaves the working tunnel alone.

Device processes are not root, so a privileged child goes through `sudo -n` — `tailscale.sudo()` prefixes `daemon_args()`, `cli_args()`, `up_args()` and the kill in the supervisor. The supervisor must kill its own child on exit: `ensure_running` will not, and an orphaned root `tailscaled` keeps the tunnel open with nothing supervising it. That kill is scoped to the child's own `--socket` path rather than the binary, because `binaries()` prefers a pair the OS ships and the binary path would take a distro's daemon down with ours. For the same reason `daemon_args()` passes `--statedir` explicitly: tailscaled fills its var root from `--state` only when that file's directory is named `tailscale`, and an empty var root sends certs, Taildrop and `profile-data` to HOME, which on device is the read-only rootfs.

Daemon → UI state travels through one `CLEAR_ON_MANAGER_START` string param, `MoonpilotTailscaleStatus`, decoded by `moonpilot/tailscale.py`, so neither panel ever shells out and a reboot cannot leave a stale `running 100.x` on screen. Both trees' QR sign-in dialogs — `moonpilot/ui/tailscale_qr.py` and `moonpilot/ui/tailscale_qr_mici.py` — are fork-owned and self-contained on purpose: they import `make_texture` from upstream's qrcode module but subclass nothing, because a silently-unused override of a renamed upstream method is exactly the failure a merge cannot see.

### The lateral controller

`moonpilot/latcontrol.py` is the fork's own torque lateral controller, and the seam in `openpilot/selfdrive/controls/controlsd.py` is a factory call that returns `None` when the toggle is off, so upstream's controller stays on the line as the fallback. Three things about it are not obvious:

- **It is chosen once, at construction.** controlsd builds one `LatControl` in `__init__` and never revisits it, so the toggle takes a restart — the feature description says so, and the row is offroad-only.
- **It only exists for torque-steered cars.** The seam sits in the `lateralTuning == 'torque'` branch, so angle- and curvature-steered cars never reach it and the toggle is a no-op there.
- **It logs into upstream's `LateralTorqueState`**, with `version` in a fork band (1000) so a log says which controller produced it. No new cereal struct, so the plotjuggler and jotpluggler torque-controller layouts keep working.

The control law is upstream's family — feedforward in lateral acceleration, PI on the delay-matched error, friction compensation — because that is the part real miles paid for; the gains and every other number are fork-owned, in that file's header block. `moonpilot/tests/test_latcontrol.py` pins what the seam depends on: the delay-matched setpoint, the roll and friction terms in the feedforward, the saturation timer, and an integrator that does not survive a disengagement.

### The longitudinal strategy

`moonpilot/longitudinal.py` is the fork's own planner — what acceleration to ask for — and `moonpilot/longcontrol.py` its own acceleration controller, how that acceleration is tracked. Two seams pick them: `moonpilot_longitudinal_planner()` in plannerd, `moonpilot_longcontrol()` in controlsd, both factory calls returning `None` when the toggle is off, so upstream's acados MPC and upstream's PI loop stay on the line as the fallback. Five things about them are not obvious:

- **The policy is closed-form, and it is three candidates.** A spacing regulator holding `gap == STOP_DISTANCE + t_follow * v_ego` with its braking authority capped at the approach decel; the exact kinematic decel that arrives at `STOP_DISTANCE` with the lead's preview-corrected speed, which binds once it exceeds that cap; and the cruise term, plus the model's own accel in experimental mode. The smallest wins, ties keeping the earlier candidate. The plan's trajectory is the same policy rolled forward over the published horizon.
- **The handover into the kinematic term is a step, and what bounds it is arbitration, not the handover.** Where the required decel reaches the approach decel the regulator's output is whatever the spacing error says — positive while the gap is still wide, and at 25 m/s closing on a 20 m/s lead that is +19.9 m/s², a 20.9 m/s² step in the candidate. It never reaches the output: while a lead candidate asks for more than the cruise candidate, `min` keeps cruise, so the step the output takes is cruise's cap minus the approach decel — 1.8 m/s² at 25 m/s, which the jerk limit turns into a rate. Upstream's maneuver suite has no maneuver in that regime, so `moonpilot/tests/test_longitudinal.py` pins it directly; what the kinematic term buys is the constant-deceleration profile after the handover, which is the part that is self-consistent without a solver.
- **Delay compensation is state prediction, not plan inversion.** The policy is evaluated where the car and the leads will be at `longitudinalActuatorDelay + DT_MDL`, instead of inverting the published plan through `get_accel_from_plan`. That inverse divides by the delay, so a step in the plan comes out amplified by `action_t / dt` — measured at 45 m/s³ and instant `ACCEL_MIN` saturation in simulation, which is why this is the fork's mechanism.
- **It is chosen once, at construction, and only exists for `openpilotLongitudinalControl` cars.** Both seams read the param once, so the toggle takes a restart — the feature description says so, and the row is offroad-only. A car whose stock ACC owns acceleration never reaches the planner seam, and the controller seam is picked there too but inert: controlsd's `CC.longActive` requires `CP.openpilotLongitudinalControl`, so `moonpilot_longcontrol`'s instance stays in `LongCtrlState.off` and returns `0.0`, exactly as upstream's does.
- **`MoonpilotLeadLateral` reaches it through the time gap**, not through an MPC danger factor: the fork planner never calls `lead_danger_factor`, so a lead predicted to leave the path scales `t_follow` (by `MOONPILOT_OUT_OF_PATH_T_FOLLOW`) rather than relaxing a constraint in a solver it does not have.

The control law is upstream's family — PI on the accel error, feedforward of the plan's target, on the car's own `longitudinalTuning.kiV` schedule — because the car is the plant and that table is the only thing with miles behind it. `long_control_state_trans` is imported from upstream rather than copied: it carries no tuning, it is the `LongControlState` contract controlsd, `controlsState` and both UIs read, and upstream's own test already pins it. What the fork owns is the output rate limit, the standstill and pedal overrides' integrator freeze, and a stopping ramp that is a rate limit toward `CP.stopAccel`. Every number in both files is in their header blocks.

One trap is worth naming: `log.LongitudinalPersonality.standard` is a plain `int`, but the same enum read off a message is a capnp `_DynamicEnum` whose hash is not that int, so a dict keyed by the enum's members never matches a value read off `selfdriveState` — `MOONPILOT_T_FOLLOW` is keyed by the raw value for that reason. `moonpilot/tests/test_longitudinal.py` pins it, along with the properties above, and ends by running upstream's own maneuver suite — all 15 maneuvers × 4 `(e2e, force_decel)` combinations from `openpilot/selfdrive/test/longitudinal_maneuvers/test_longitudinal.py`, patched onto the plant — against the fork planner, which is the end-to-end statement that the strategy drives the stock scenarios without a crash, without stalling at a stop, and while still decelerating under `forceDecel`. `moonpilot/tests/test_longcontrol.py` holds the controller to controlsd's call contract and to the three things the fork owns in it.

`process_replay` reference logs differ for plannerd and controlsd with the feature on; that test needs route data and is in `tools/test_runner.py`'s `IGNORED` list, so a run of it would need regenerated refs or the param flipped off.

### Lateral-only engagement

`moonpilot/engage.py` is the fork's half-engaged state: openpilot steers from the car's cruise **main switch**, and the car's own ACC takes over speed only once the driver sets it. It cannot be done on the openpilot side alone — panda blocks steering unless its own `controls_allowed` is true, and that flag is only set on the rising edge of stock ACC — so this is the fork's one feature that reaches into `opendbc` and `panda` as well as the openpilot tree. Read the whole chain before changing any part of it.

The permission itself is in the forked safety layer, `opendbc/safety/moonpilot/lateral_engage.h`, reached through `opendbc/safety/lateral.h`:

- **`controls_allowed` keeps upstream's exact meaning** — stock ACC engagement — and stays the only thing `get_longitudinal_allowed()` reads, so every longitudinal check, the brake/gas/regen exits and the heartbeat logic behave exactly as upstream wrote them. `controls_allowed_lateral` is a second, additive flag that only ever gates steering: all 12 reads of `controls_allowed` in `lateral.h` become `(controls_allowed || controls_allowed_lateral)`. A change that widens the first flag instead of the second hands the fork the longitudinal controls it does not have.
- **What arms it:** the rising edge of the cruise main switch, or of the openpilot heartbeat, while *both* are true, and only when the brand's safety param enabled the rule. What clears it: either of those going false, a rising edge of `steering_disengage`, a disable event in selfdrived, and the lag/validity path in `safety_tick` (both `controls_allowed = false` sites call `lateral_engage_exit()`).
- **Brake and gas deliberately do not appear in the rule.** Keeping steering through a brake tap is the point of the feature, so the fork suppresses two events instead — `pcmDisable`, which is level-triggered whenever stock ACC is off, and `pedalPressed`, the brake/gas disengage. Gas keeps its softer `gasPressedOverride` path, which is an override rather than a disable, so steering continues through it too. Suppressing an event is narrower than changing what raises it: anything else upstream adds in the same event type still disengages.
- **The enable rides the safety param, not `alternative_experience`.** `toyota_init` has to see it to pick its rx-check set, and the param reaches `init` by construction while alt-exp only happens to be set first by pandad. `ToyotaSafetyFlags.LATERAL_ENGAGE` is `16 << 8`, the next free Toyota flag, and a stock config's params and rx-check array are byte-identical to upstream's.

`moonpilot/engage.py` is the openpilot side, wired through three seams: `openpilot/selfdrive/car/card.py` calls `moonpilot_engage_safety_param()` before `CarParams` is written, `openpilot/selfdrive/selfdrived/selfdrived.py` runs `LateralEngage.update()` after every other event source and uses `controls_allowed()` for the panda cross-check, and `openpilot/selfdrive/pandad/pandad.cc` publishes the flag from the panda health bit `HEALTH_FLAG_CONTROLS_ALLOWED_LATERAL`. Four things about it are not obvious:

- **Latching is what makes a disable last, and it is released on an edge.** With `pcmDisable` and `pedalPressed` suppressed, the events that would keep re-disengaging are gone, so an authoritative disable (`USER_DISABLE`, `IMMEDIATE_DISABLE`, `SOFT_DISABLE`) latches the state off in the module and only a re-arm gesture clears it: the cruise main switch coming back on, the LKAS button, or ACC's **rising** edge — not ACC merely being set. Clearing on the level would undo a driver's LKAS press one frame later for anyone cruising with ACC set, which makes their own off switch look broken; `moonpilot/tests/test_engage.py` pins both directions. `SOFT_DISABLE` is included because a soft-disable event without `NO_ENTRY` would otherwise make this an engage/disable loop; the LKAS toggle adds `buttonCancel` — a real `USER_DISABLE` — precisely because the state machine leaves the enabled state on a disable event and on nothing else, so toggling a module flag alone would steer while the driver watched the button do nothing.
- **The engage request is `buttonEnable`, under `not enabled and not NO_ENTRY`, and only while not blocked.** `NO_ENTRY` is what keeps a standstill or an uncalibrated car from nagging with refuse alerts — the attempt simply repeats once the blocker clears — and the `not blocked` half is what stops the latch from being undone by the very next frame's arming request. Upstream's own `pcmEnable` on the same frame counts as the ask, so the fork does not stack a second enable event on it.
- **The panda cross-check has to see the new permission**, or half-engaged reads as `controlsMismatch` and disengages. `LateralEngage.controls_allowed(ps)` accepts either flag; that keeps the mismatch check sharp instead of switching it off.
- **The decision is made once, at construction**, like the fork's other behavior toggles, so the row takes a restart — which its description says.

The ceiling is the car: **Toyota/Lexus with stock longitudinal (`CP.pcmCruise`) that carries `PCM_CRUISE_2`**, not `passive` (a dashcam-mode car has already had its `safetyConfigs` replaced with a single noOutput config by the time the seam runs, so `_available()` checks the flag itself rather than trusting where the call sits), and not `ToyotaFlags.UNSUPPORTED_DSU` — those read the main switch out of `DSU_CRUISE` at 5 Hz, below panda's 10 Hz rx-check minimum, so `_available()` reports the feature unavailable rather than arming a check that would fail on the road. `PCM_CRUISE_2` is an rx check the fork added (33 Hz, checksum and counter ignored on purpose — a mismatch there would clear `controls_allowed` for *every* message, a worse failure than trusting a switch bit), and its real rate is unvalidated until someone drives it: if the car reports `safetyRxChecksInvalid` or `controlsMismatch` with the feature on, measure the rate from a route and set the check's frequency to it rather than relaxing the check. `controlsd` is deliberately untouched: `CC.latActive` already follows `selfdriveState.active`, and `CC.longActive` already requires `CP.openpilotLongitudinalControl`, so the half-engaged state needs no second code path in the acceleration controller.

### Forked opendbc and panda

`panda` and `opendbc_repo` point at `jjolano/moonpilot-panda` and `jjolano/moonpilot-opendbc` (relative URLs in `.gitmodules`), each with `upstream` still on comma and fork work committed on `master`. The safety code compiles into the panda firmware, so a fork-owned safety layer means fork-owned submodules, and nothing else reaches them. They sync the same way the superproject does — merge, never rebase, no force-push — with their own `master`:

    cd panda && git fetch upstream && git merge upstream/master
    cd ../opendbc_repo && git fetch upstream && git merge upstream/master

Then record the merge with an ordinary `git add panda opendbc_repo` in the superproject. `git submodule update` on the device fetches the default refspec, so the recorded SHA must be reachable from the fork's default branch — a fork commit left on a side branch does not reach a device. Both are forks of commaai's repos in the same fork network as every other account, so `gh repo fork` will refuse once one exists; creating a plain public repo and pushing to it is the equivalent, and the only requirement is that `master` carries the commits.

One consequence of the fork owning the safety layer lands in panda rather than here: **panda's own `./test.sh` cannot build the firmware.** Its `setup.sh` creates a venv whose `pyproject.toml` pins `opendbc @ git+https://github.com/commaai/opendbc.git@master`, so that venv holds *upstream* opendbc with no `opendbc/safety/moonpilot/`, and panda's standalone `scons` stops on `'controls_allowed_lateral' undeclared` before any test runs. Build from here — `tools/op.sh build`, which puts `opendbc_repo` on the include path — or give the standalone run the fork's opendbc: `PYTHONPATH=../opendbc_repo panda/.venv/bin/python -m unittest discover -s panda/tests`. Never repoint panda's `pyproject.toml` at the fork; that line is upstream's, and `panda/AGENTS.md` records this too.

Two comma rules come with editing `opendbc/safety/`, both in `docs/SAFETY.md`:

- **the fork cannot use the openpilot trademark**, which is what makes `moonpilot`'s own brand load-bearing rather than cosmetic;
- **the full safety test suite must be preserved and pass, including any new coverage the fork's changes require.** `opendbc/safety/tests/test.sh` enforces the second half mechanically — 100% line coverage of every non-libsafety safety file — so new C in that tree that no test reaches is a failing gate, not a warning. `moonpilot/tests/test_engage.py` holds the openpilot half of the same feature; the safety half is `opendbc/safety/tests/lateral_engage_common.py` plus one class per rx-check branch in `test_toyota.py`.

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
- When upstream's types or signatures clash with a seam, adapt the fork side. `MOONPILOT_PROCS` is annotated with upstream's process union rather than widening upstream's `procs` list, so `procs += MOONPILOT_PROCS` type-checks with no upstream edit.

### Seam markers

Every fork line inside an upstream file carries the marker `moonpilot seam, see AGENTS.md`, in that file's comment syntax (`#`, `//`, or capnp's trailing `#`). The marker is what tells anyone — human or agent — that the line is fork-owned and why it is there.

    brand = moonpilot_brand() or "openpilot"  # moonpilot seam, see AGENTS.md

Capnp seams carry the invariant they must not break, since a wrong edit there corrupts recorded data:

    moonpilotState @107 :Custom.MoonpilotState;  # moonpilot seam: do not change @107 or which struct it points to. See AGENTS.md before editing.

`git diff` showing an unmarked change to a file under `openpilot/` means something is wrong: either the line is a seam and needs the marker, or it is an upstream edit that should not be there.

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
| Adds rows/lines near a seam — a param, a proc, a `SConscript`, a settings panel | usually clean | rebuild and re-check. A new settings panel is clean to merge but shifts the fork panel down the tizi sidebar (it sits at y 960–1070 of 1080); re-verify it is not clipped. The mici scroller scrolls, so it absorbs the extra entry |
| Renames or removes a symbol the fork imports | clean | the `import moonpilot…` line fails. Fix `moonpilot/`, never upstream |
| Deletes a file the fork hooked | modify/delete conflict | re-attach the seam at the nearest equivalent point, then drop the dead row from the table above and from `ALLOWED` |
| Renames or restructures the safety layer the fork forked | clean in the superproject | the fork's `opendbc_repo` merge is where it lands, and `moonpilot/engage.py` is what breaks if a symbol it imports is gone. Merge the submodule first, then the superproject |
| Adds its own `AGENTS.md` | add/add conflict | this file stays fork-owned; fold in anything useful from upstream's |
| Wants a reserved struct or param name the fork also uses | conflict | upstream's ids and names win — move the fork to the next free `CustomReservedN`, never the reverse |

Last 2000 upstream commits, per seam file: `pyproject.toml` (391) and `SConstruct` (320) move almost weekly, so those two conflict most; `cereal/log.capnp` (31), `common/params_keys.h` (18) and `scripts/lint/lint.sh` (16) are moderate; `process_config.py`, the UI settings panels, `home.py`, `version.h`, `custom.capnp` and `controlsd.py` have moved 2–7 times. A new seam is a permanent recurring conflict surface — add one only when no existing seam reaches.

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
