# moonpilot

Personal fork of [openpilot](https://github.com/commaai/openpilot). The name is lowercase everywhere, like `openpilot` and `comma`. `CLAUDE.md` is a symlink to this file.

## The rule

Fork code lives in `moonpilot/`. Upstream files carry **seams** only: a one-line hook into `moonpilot/`, marked with a literal `moonpilot` comment or identifier. The fork's whole surface on upstream is the table below — keep it that way and `git merge upstream/master` only ever conflicts on seam lines.

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

## Sync with upstream

Fork work lives on `master`. Merge, never rebase; no force-push.

    git fetch upstream
    git merge upstream/master
    # conflicts land on seam lines; `grep -n moonpilot <file>` shows what the fork added
    tools/op.sh lint
    tools/op.sh test moonpilot
    tools/op.sh build

Then bump `COMMA_VERSION` to the new upstream version, keeping the `-moonpilot` suffix.

## Working here

- `tools/op.sh build | lint | test [target]` — build, lint, test. `tools/op.sh --help` lists the rest.
- UI is Python + raylib; tizi (`openpilot/selfdrive/ui/layouts/`) and mici (`openpilot/selfdrive/ui/mici/layouts/`) are separate trees — a fork panel is registered in both.
- Tests run through a unittest loader (`tools/test_runner.py`), so fork tests subclass `unittest.TestCase`.
- Style is enforced by `scripts/lint/lint.sh` and `pyproject.toml` (ruff, ty, codespell); fork code mirrors the conventions of the upstream file it hooks into.
