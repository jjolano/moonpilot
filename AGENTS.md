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
| `openpilot/common/params_keys.h` | `#include "moonpilot/params_keys.h"` — fork params, one row per key |
| `openpilot/system/manager/process_config.py` | `procs += MOONPILOT_PROCS` from `moonpilot/procs.py` |
| `openpilot/cereal/custom.capnp` | `MoonpilotState` (upstream's reserved struct; never change the `@0x…` id) |
| `openpilot/cereal/log.capnp` | `moonpilotState @107` event field |
| `openpilot/cereal/services.py` | `moonpilotState` service row |
| `openpilot/common/version.h` | `COMMA_VERSION "<upstream>-moonpilot.<fork revision>"` |
| `openpilot/selfdrive/controls/plannerd.py` | `moonpilotState` subscription |
| `openpilot/selfdrive/controls/lib/longitudinal_planner.py` | lead danger factor from `moonpilot.lead` |
| `openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | `lead_danger_factor` kwarg on `LongitudinalMpc.update` |
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

That makes the fork's dependency surface the **init path**: whatever is reachable at import time from the modules a seam pulls in — `moonpilot.procs` from the manager at boot, `moonpilot/ui/settings*.py` from the UI at start, `moonpilot.lead` from `longitudinal_planner` (plannerd) and the renderers. Under those, import only the standard library, `numpy`, and `openpilot`'s own modules.

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
    python3 -c "import moonpilot.procs, moonpilot.ui.settings, moonpilot.ui.settings_mici"
    tools/op.sh lint
    tools/op.sh test moonpilot
    tools/op.sh build

Then bump `COMMA_VERSION`'s upstream part to the new upstream version, leaving the `-moonpilot.<n>` suffix alone. See **Versioning** below.

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
| Adds its own `AGENTS.md` | add/add conflict | this file stays fork-owned; fold in anything useful from upstream's |
| Wants a reserved struct or param name the fork also uses | conflict | upstream's ids and names win — move the fork to the next free `CustomReservedN`, never the reverse |

Last 2000 upstream commits, per seam file: `pyproject.toml` (391) and `SConstruct` (320) move almost weekly, so those two conflict most; `cereal/log.capnp` (31), `common/params_keys.h` (18) and `scripts/lint/lint.sh` (16) are moderate; `process_config.py`, the UI settings panels, `home.py`, `version.h` and `custom.capnp` have moved 2–7 times. A new seam is a permanent recurring conflict surface — add one only when no existing seam reaches.

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

Where a claim is mechanically checkable, pin it with a test rather than trusting the prose. That is already how much of this file is held up — `test_version.py` (the version shape), `test_paths.py` (the fork state roots), `test_upstream_touches.py` (the seam table against `ALLOWED`), `test_deps.py` (the registry agreeing with the lock) — and `test_agents_md.py` extends the same pattern to the fork paths named here, so a rename that misses this file fails the suite instead of misleading the next reader.
