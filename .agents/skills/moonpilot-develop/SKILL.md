---
name: moonpilot-develop
description: Runs the fork's edit loop — lint, fork test suite, build, init-path import, then getting the edit onto the comma 3X and proving it live — and names the failure each step catches. Use when writing or changing code under `moonpilot/`, adding a seam, toggle, param or dependency, merging upstream, building after touching C++ or a submodule, syncing a work-in-progress edit to the device for a quick try, or asking "how do I test this fork change", "which tests are slow", "why did lint pass but the device fail at boot". Not for shipping a release (see `moonpilot-device-deploy`), diagnosing a fault on a running device (`moonpilot-device-debug`), or reading route logs (`moonpilot-log-analysis`).
---

# Developing moonpilot

`AGENTS.md` is the contract, this is the loop. Read **The rule** and **Seams** before touching an
upstream file, **Features** and **Storage** for a toggle or persisted value, **Dependencies** and
**The gate** for a new import, **Forked opendbc and panda** for submodule work.

## The PC loop

```bash
git add -A                      # staging trap, below
tools/op.sh lint
tools/op.sh test moonpilot
tools/op.sh build
.venv/bin/python3 -c "import moonpilot.engage, moonpilot.procs, moonpilot.ui.settings, moonpilot.ui.settings_mici"
```

**Stage first, it silently voids the rest.** `scripts/lint/lint.sh` takes its file list from
`git ls-files -z openpilot moonpilot` and `tools/scripts/devsync.py` walks the same index, so a new
untracked module is linted by nothing, reaches no device, and its green run proves nothing.

- `tools/op.sh lint` — ruff, the `check_*` scripts, then ty and codespell. `--fast` drops the last
  two; `tools/op.sh lint ruff` runs exactly one; `--skip ty` drops one.
- `tools/op.sh test [TARGETS]` — `tools/test_runner.py`. A target is a directory, a file, or
  `path.py::Class::test`; `-k TEXT` filters ids, `-s` shows output live, `-j N` sets workers,
  `-v` lists every test. Bare, it runs `openpilot` + `moonpilot` **and** applies its `IGNORED` list;
  naming any target turns the ignores off.
- `tools/op.sh build` — `scons -u` from the current directory, extra flags passed through
  (`tools/op.sh build -j8`). On AGNOS it runs `openpilot/system/manager/build.py` instead.
- The import line is the only step that **executes the init path**: the chain the manager, both UIs,
  plannerd and card pull in at boot. ruff and ty resolve names statically and pass an upstream module
  that moved or a third-party import that crept in under `moonpilot/`; those fail on the car at boot
  instead. Run it after every `git merge upstream/master`.

## Cheap and not

`moonpilot/tests/` is the fork's own suite and the thing to run per edit. `test_longitudinal.py`,
`test_slam.py`, `test_curve.py` and `test_latency.py` drive upstream's plant and dominate the clock,
so narrow while iterating:

```bash
tools/op.sh test moonpilot/tests/test_engage.py
tools/op.sh test moonpilot -k lateral
```

`process_replay` is outside that budget. `openpilot/selfdrive/test/process_replay/test_processes.py`
sits in `tools/test_runner.py`'s `IGNORED`, so `tools/op.sh test` skips it — and since naming a
target disables the ignores, passing that path explicitly does run it: it fetches route segments and
compares against `ref_commit`, so a fork change to a logged field needs regenerated refs to pass.

## Getting an edit onto the device

```bash
tools/scripts/devsync.py moonpilot                    # --remote defaults to /data/openpilot
tools/scripts/devsync.py 100.94.10.12 -i ~/.ssh/id_rsa
```
It rsyncs `git ls-files --recurse-submodules` to `comma@<ip>:<remote>`, then re-syncs on every
filesystem event. It imports `watchdog`, absent from the fork's lock, so it fails at import until
that is installed — its own first one-shot, by hand:

```bash
git ls-files --recurse-submodules -z | rsync -azn --files-from=- --from0 -e ssh ./ comma@moonpilot:/data/openpilot/
```

`-n` is a dry run. devsync copies files and nothing else — no build, no restart, no submodule pins —
so it suits Python edits under `moonpilot/`; C++, `SConscript`, capnp and submodules need a real
deploy (`moonpilot-device-deploy`). Then prove it live: the device has **no pytest**, and the
interactive `python3` is `/usr/bin/python3` without numpy while the manager runs with
`/usr/local/venv/bin` first on `PATH`, so name that interpreter and keep the fork suite on the PC.
`alive=False` in the block below is the offroad state, not a failure — the fork's publisher runs
onroad.

```bash
ssh moonpilot 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c "
import moonpilot.engage, moonpilot.procs, moonpilot.ui.settings, moonpilot.ui.settings_mici
from openpilot.common.params import Params
print(Params().get(\"MoonpilotSlam\", return_default=True))
import openpilot.cereal.messaging as messaging
sm = messaging.SubMaster([\"moonpilotState\"]); sm.update(500)
print(sm.alive[\"moonpilotState\"], sm[\"moonpilotState\"].egoCorrection.valid)"'
```

## Native and submodule work

This device has no `/data/openpilot/prebuilt`, so `launch_chffrplus.sh` runs `build.py` on every
start and a restart builds whatever arrived. Building by hand is the trap: the launcher also creates
the package symlinks `msgq -> msgq_repo/msgq`, `opendbc -> opendbc_repo/opendbc`, `rednose`,
`teleoprtc`, `tinygrad`, and `panda/SConscript` line 4 is a plain `import opendbc` — so `build.py`
in a tree the launcher has not touched yet (fresh clone, post-update checkout) dies with
`ModuleNotFoundError: No module named 'opendbc'`. On the PC build a submodule change from the repo
root, never from inside `panda/`.

## Verify by change type

- **UI seam** — both settings tables construct the panel eagerly, so a construction error takes the
  whole settings screen down, not just the row. Import is checkable over ssh, construction is not:
  it needs a live window, where fonts are loaded into the dict `font()` indexes
  (`openpilot/system/ui/lib/application.py:220,345,699`). No screenshot is obtainable either —
  rendering is an eyeball check on the device screen.
- **Control path** — the full `tools/op.sh test moonpilot`, then the device import block above plus
  `ssh moonpilot 'pgrep -af moonpilot'` to see the fork's processes by name.
- **Safety layer or firmware (`opendbc_repo`, `panda`)** — the superproject's loop proves nothing
  about either. The gate that matters most is `opendbc_repo`'s mutation run: it mutates the safety C
  and fails if the suite leaves a mutant alive, so a new branch needs a test that dies with it. Then
  the panda suite with the fork's opendbc on the path, and a build that proves the firmware links:

  ```bash
  (cd opendbc_repo && .venv/bin/python opendbc/safety/tests/mutation.py)   # ~35 s, 3133 mutants
  PYTHONPATH=opendbc_repo panda/.venv/bin/python -m unittest discover -s panda/tests
  tools/op.sh build
  ```

  `opendbc_repo/AGENTS.md` holds the suite and the coverage gate, and names both ways that gate lies
  — `gcovr` needs a venv of its own and consumes the data it reads. Keep hunks in those two trees as
  small as the superproject's: nothing checks them for reformatting, where
  `test_upstream_touches.py` checks the superproject.
- **Param or registry row** — the manager seeds a key on disk only when the declaration carries a
  default (`openpilot/system/manager/manager.py`: `get_default_value(k) is not None`), so a
  `{PERSISTENT, STRING}` row with no default reads as absent until something writes it —
  `MoonpilotModelsDriving` and `MoonpilotModelsFavs` both do. To prove a row is *declared* rather
  than merely unwritten, ask the library, and decode: `all_keys()` returns **bytes**, so
  `"Foo" in Params().all_keys()` is silently always False.
  `ssh moonpilot 'cd /data/openpilot && PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c "
  from openpilot.common.params import Params
  keys = [k.decode() for k in Params().all_keys()]
  print(len(keys), [k for k in keys if k.startswith(\"Moonpilot\")])"'`
  Either way the fork's own reads stay correct: they use `get(key, return_default=True)`, which
  answers for an unset key.
