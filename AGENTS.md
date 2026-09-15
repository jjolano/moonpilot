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
| `openpilot/common/version.h` | `COMMA_VERSION "<upstream>-moonpilot"` |
| `openpilot/selfdrive/ui/layouts/home.py` | brand string (tizi) |
| `openpilot/selfdrive/ui/mici/layouts/home.py` | brand label (mici) |
| `openpilot/selfdrive/ui/layouts/settings/settings.py` | `PanelType.MOONPILOT` + panel from `moonpilot/ui/settings.py` |
| `openpilot/selfdrive/ui/mici/layouts/settings/settings.py` | settings entry + panel from `moonpilot/ui/settings_mici.py` |

Pick by need — reuse a seam, never invent one:

- Setting or persisted state → a row in `moonpilot/params_keys.h`, named `Moonpilot*`.
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
2. A control in the moonpilot panel: tizi (`moonpilot/ui/settings.py`) uses `toggle_item(...)` with a callback doing `params.put_bool(key, state, block=True)`, plus the refresh loop upstream's `developer.py` uses to mirror external changes; mici (`moonpilot/ui/settings_mici.py`) uses `BigParamControl(text, key, description=...)`, which writes the param itself.
3. A read in fork code, with `params.get(key, return_default=True)` — **not** `get_bool`.

That read is the trap: `get_bool` ignores the declared default and reports off for an unset param. The declaration only becomes true because `openpilot/system/manager/manager.py` seeds every unset param from its default at boot. On device both work; in a bare script or a test outside the manager, only `return_default=True` does.

Gate anything that changes driving behavior on `ui_state.is_offroad()` — the `enabled=` argument — and if it needs a restart to take effect, say so in the description the way the alpha-longitudinal toggle does.

Feature toggles go in the **moonpilot panel**, never as new sidebar entries: the tizi sidebar already reaches y 1070 of 1080 with 7 entries, so an 8th clips. The panel body scrolls and takes as many rows as you like — upstream's Developer panel carries 8 in the same widget.

### When moonpilot and upstream converge

- Upstream implements something moonpilot already has → delete moonpilot's version and its toggle row. A switch between two identical behaviors is rot, and it is how a fork accumulates dead weight.
- Moonpilot's version is the one to keep → keep it, and let the seam carry the choice.
- Upstream claims a reserved struct or param name → upstream's wins; the fork moves.

### The registry

`moonpilot/features.py` drives the panel. Adding a feature is three things: a `Feature(...)`, a row in `moonpilot/params_keys.h`, and the code behind it.

```python
BRANDING = Feature(
  key="MoonpilotBranding",      # the params_keys.h row, which carries the default
  title="moonpilot branding",   # what the panel shows
  description="...",            # shown as the item's description / long-press help
)
FEATURES: tuple[Feature, ...] = (BRANDING,)
```

Both panels build their rows by iterating `FEATURES`, so a new row appears in tizi and mici without touching either panel file. Read a feature's state with `enabled(feature, params)`, never `get_bool`, for the reason above. When a feature's decision is more than a boolean, add a function next to `enabled` — `brand(params)` is the pattern: the seam calls it, the feature stays a table row.

## Editing upstream

- Seam lines only. Never reformat, reorder, or tidy adjacent upstream code — an incidental whitespace fix turns a one-line merge conflict into a whole-file one.
- When upstream's types or signatures clash with a seam, adapt the fork side. `MOONPILOT_PROCS` is annotated with upstream's process union rather than widening upstream's `procs` list, so `procs += MOONPILOT_PROCS` type-checks with no upstream edit.

### Seam markers

Every fork line inside an upstream file carries the marker `moonpilot seam, see AGENTS.md`, in that file's comment syntax (`#`, `//`, or capnp's trailing `#`). The marker is what tells anyone — human or agent — that the line is fork-owned and why it is there.

    brand = moonpilot_brand(self.params)  # moonpilot seam, see AGENTS.md

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

Then bump `COMMA_VERSION` to the new upstream version, keeping the `-moonpilot` suffix.

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
- `scripts/lint/lint.sh` walks `git ls-files`, so a new file is invisible to lint until it is tracked — `git add` before trusting a clean run.
