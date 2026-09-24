# The model marketplace
A driver may replace the model this car drives. The panes are `moonpilot/ui/models.py` (tizi) and
`moonpilot/ui/models_mici.py` (mici); the rules are `moonpilot/models.py`; the catalog side is
`moonpilot/modelcatalog.py`; the worker is `moonpilot/modelsd.py`; the compiler is
`moonpilot/modelbuild.py`; the runtime is `moonpilot/modelruntime.py`. The catalog itself —
`https://jjolano.github.io/openmodels/catalog.json` — publishes bytes and a structural identity and
owns no activation, scheduling or qualification, so everything below is the consumer half.

Ten things are not obvious.

- **A selection is a string, and the boot commit is the only gate.** `""` is the bundled model, or a
  64-hex digest for one catalog recipe. The driver's choice goes in
  `MoonpilotModelsDriving`/`MoonpilotModelsMonitoring`; `commit_boot_selection` runs once at boot from
  the manager seam, re-checks the selection against the store (records parse, the protocol is one this
  fork carries, the kind matches the param it came from, every artifact is on disk with its recorded
  size and digest, and a build exists for exactly this selection, protocol and resolution set) and
  writes every check it made into `MoonpilotModelsBoot`. The model processes read that snapshot and
  nothing else — a per-process read of the desired value is not a coherent switch, and
  `dmonitoringmodeld` outlives an onroad cycle. It is `CLEAR_ON_MANAGER_START`, so a reboot cannot
  inherit one. **The row takes a restart**, which both panels say, and the seams read
  `boot(params)` rather than the desired params. The snapshot also carries the selection's declared
  `LAT_SMOOTH_SECONDS`/`LONG_SMOOTH_SECONDS`: `models.boot_configuration` is how `controlsd` reads it
  for the curvature reference and the torque controller, so the fork's own consumers time their
  command against the same number modeld decoded with, and only the load failure below makes the two
  diverge.
- **A load failure is the bundled model, and it takes the whole runtime with it.** `create_model`
  returns `None` for a build that will not load — a truncated or overwritten `model.pkl` — and writes
  the reason to `MoonpilotModelsActive*` for the panel. Both seams then drop the runtime for the rest
  of the process's life, so the model, the smoothing constants and the action decode are all
  upstream's. That last part is the one that bites: the decode is per-selection (a `curvature-100`
  build divides the head by 100 where a `plan` build does not), so a bundled model left talking to a
  custom runtime is a wrong steering request on a model nobody checked. The driver's selection is
  untouched, and the boot commit never sees this failure — it happens later and in another process.
- **Only four protocols are selectable, and what they require is the union over the members.**
  `comma.supercombo.v1` (roles `supercombo`), `comma.split-vision-policy.v1` (`vision`, `on_policy`),
  `comma.split-vision-off-on.v1` (`vision`, `off_policy`, `on_policy`) and `comma.dmonitoring.v1`
  (`dmonitoring`) are pinned in `moonpilot/models.py` with the inputs each role is fed and the output
  names the consumers here read. The slice requirement is the *union*, not a per-role pin: every
  consumer reads names off one dict, so what must hold is that some member produces each name, and a
  per-role pin refuses real sets whose members split them differently — a family-C recipe puts
  `lane_lines`/`lead` in the off-policy member and `plan` in whichever policy member carries it.
  Admission (`moonpilot/modelcatalog.py`) is `Recipe.check` plus this table:
  `configuration_unresolved` and `target_unsupported` refuse, a member whose targets do not include
  `QCOM` refuses, a member declaring an input the protocol does not list refuses (that is what
  refuses the pre-`action_t` monoliths), an artifact over `MAX_ARTIFACT_BYTES` refuses, and a
  multi-member recipe whose profile connection has no recorded port on either side refuses with
  `structure_unknown`. For scale, at the catalog revision the fixture pins (`2026-09-19`): 230 of
  1144 recipes admit — 102 supercombo, 98 vision+policy, 17 off+on+vision, 13 driver monitoring — and
  most refusals are the catalog's own documents rather than a capability limit (689 unresolved
  configuration keys, 121 AMD/big targets, 16 artifacts with no bytes published). The catalog's own
  `semantics_unverified`, `requires_runner_check` and `qualification: consumer_owned` ride into the
  panel verbatim and are never upgraded to "compatible".
- **The action head's meaning changed on 2026-06-01 and nothing in a model records which side of it
  a head was trained on.** `ModelDataV2.action[0,0]` was a curvature times 100
  (`openpilot@9757bf10`) and is now a lateral acceleration (`action[0,0] / max(1, v_ego)**2`, from
  `249cafe897deb433944c84fa6c3c6ce568f727e1`). The protocol pins the convention
  (`action_contract`) and admission refuses a member that declares an `action` slice on the wrong
  side of `LATERAL_ACCEL_SINCE`, by the recipe's recorded publication date: a wrong scale factor here
  is a wrong steering request, silently. The build record carries the contract to the runtime, which
  is why `comma.split-vision-off-on.v1` still decodes the old way.
- **The worker never exits on its own, and never compiles onroad.** `MOONPILOT_PROCS`'s predicate is
  `request present or status missing`, and the manager's stop is the only exit that leads anywhere
  (`ensure_running` does not restart a child that exited while the predicate was still True). So a
  finished job publishes the status and *then* clears the request, which is what lets the manager reap
  it. A download runs on any non-`none` network — the driver asked for it, and the status says when
  the bytes were metered — but the build phase waits in `waiting-offroad`, because the compiler takes a
  whole CPU. `moonpilot/modelsd.py` holds that loop. What the panels render for a running job is
  `models.job_fraction` / `job_progress_text` / `job_text`: a bar only while `downloading` has a
  measured total (a bar for `building` would claim progress no number backs), the bytes beside it, and
  an **elapsed clock** for the four phases that can sit without a fraction — published by the worker as
  `elapsed` seconds in the snapshot (`Job.started_at`), not as a timestamp, so the panels need no clock
  agreement with it and a snapshot without the field renders as the phase alone.
- **Nothing on the boot path touches the network, and nothing pickles a downloaded file.** The boot
  commit is pure file I/O and hashing; the runtime loads only what a build wrote. Model metadata is
  read from downloaded bytes by `moonpilot/vendor/openmodels/metadata.py`, which hand-decodes the
  protobuf and refuses anything but `slice` in the embedded pickle, and every artifact is verified
  against its recorded sha256 before anything executes it (the same rule `moonpilot/fetch.py`
  follows). A package whose bytes disagree with the interface its recipe recorded is removed, not
  kept. Three ceilings are named rather than hidden: 300 MB per artifact, 8 GB for the store under
  `data_dir("models")`, and 2 GB of free space left alone; removal is the driver's action, and the
  panel shows each refusal as a sentence.
- **Catalog bytes can be authenticated, and exact interfaces are enforced.** The `Catalog signatures` toggle is off by default; `modelcatalog.refresh(require_signature=True)` fetches the `.sig` sidecar and verifies it at point of use with the optional `cryptography` dependency, refusing when no pinned `CATALOG_PUBKEY` or signature is available. `verify_downloaded` compares every recorded output slice's start, stop and step, so a shifted or resampled interface is not admitted.
- **The panel names the model actually driving.** The active-model rows use an explicit `stock:` fallback when the selected runtime is unavailable, while the desired selection and boot/restart semantics remain separate.
- **Model jobs are cancellable from both UI trees.** A running download or build exposes cancel through the existing request protocol; tizi and mici both confirm before writing the cancellation request, and the worker publishes the terminal status before it exits.
- **Both trees page the catalog, and the index behind those pages is parsed once per file version.**
  The marketplace, chooser and installed lists render a fixed window of rows (`PAGE_ROWS` 8 on tizi,
  6 on mici) with older/newer paging — the catalog is never rendered whole. What would slow the UI is
  the data behind the rows: both trees re-resolve their row callables every frame, so `models.browse()`
  keeps one stat-keyed slot (path, `mtime_ns`, size, inode — the worker's `atomic_write` rename is what
  moves the key) and `admitted_entries` memoizes its filter per kind, cleared whenever `browse()`
  reloads. A missing file reads as empty rather than serving a stale slot.

The vendored SDK is a copy, not a dependency: `moonpilot/vendor/openmodels/` carries the files and the
sync command in its `README.md`, and is reached only by `moonpilot/modelcatalog.py`, `moonpilot/modelbuild.py`
and the tests, so the manager, the planner and both panels import without it. `moonpilot/models.py` is
stdlib plus `moonpilot/paths.py` for that reason — it is on the init path (`moonpilot/procs.py`, both
panels). The copy and the catalog fixture are the two paths in this tree whose prose is not en-US, and
they are the two fork entries in `pyproject.toml`'s codespell `skip`; `ruff` and `ty` see
both, and `moonpilot/tests/test_modelcatalog.py` reads the fixture through the vendored SDK, so a copy
that drifts fails the suite rather than compiling.

Three ways to check this feature without a car: `.venv/bin/python3 -m unittest` runs of
`moonpilot/tests/test_models.py`, `test_modelcatalog.py` and `test_modelruntime.py`;
`DEV=CPU:LLVM python3 -m moonpilot.modelbuild smoke --onnx openpilot/selfdrive/modeld/models/dmonitoring_model.onnx`,
which compiles, pickles, loads through `moonpilot/modelruntime.py` and runs one frame of zeros; and, for
the whole chain against the real catalog, a scratch script that patches `models.paths.data_dir` to a temp
root, calls `modelsd.Worker` through `models.request`/`serve`, then `commit_boot_selection` and the
runtime. All three were run at the catalog revision above: the pinned stock recipe reaches one frame with
`plan`, `lane_lines`, `lead`, `meta` and `pose` finite, and its build takes ~20 s on the dev PC. The
bundled `driving_supercombo.onnx` is deliberately not a smoke input anymore: upstream's current file
declares the stateful `new_img`/`state_*_q` contract, which `comma.supercombo.v1` refuses rather than
half-feeds, so driving smoke needs a catalog or retained supercombo artifact whose inputs match the fork
protocol.

