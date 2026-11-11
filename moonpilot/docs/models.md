# The model marketplace
A driver may replace the model this car drives. The panes are `moonpilot/ui/models.py` (tizi) and
`moonpilot/ui/models_mici.py` (mici), and the tizi picker they open is `moonpilot/ui/model_picker.py`.
`moonpilot/models.py` holds the store and boot commit; `moonpilot/modelprotocols.py` holds the protocol
table and admission rules; `moonpilot/modelstatus.py` holds the worker job/status protocol and capacity
rules; `moonpilot/modelview.py` holds the panels' chooser, view model and wording. The catalog side is
`moonpilot/modelcatalog.py`; the worker is
`moonpilot/modelsd.py`; the compiler is `moonpilot/modelbuild.py`; the runtime is
`moonpilot/modelruntime.py`. The catalog itself —
`https://jjolano.github.io/openmodels/catalog.json` — publishes bytes and a structural identity and
owns no activation, scheduling or qualification, so everything below is the consumer half.

Notes on behavior and catalog evidence.

- **Boot selection is committed once.** `models.commit_boot_selection` is the only gate between a desired
  selection and model processes; see its implementation for the checks and snapshot fields.
- **A load failure is the bundled model, and it takes the whole runtime with it.** `create_model`
  returns `None` for a build that will not load — a truncated or overwritten `model.pkl` — and writes
  the reason to `MoonpilotModelsActive*` for the panel. Both seams then drop the runtime for the rest
  of the process's life, so the model, the smoothing constants and the action decode are all
  upstream's. That last part is the one that bites: the decode is per-selection (a `curvature-100`
  build divides the head by 100 where a `plan` build does not), so a bundled model left talking to a
  custom runtime is a wrong steering request on a model nobody checked. The driver's selection is
  untouched, and the boot commit never sees this failure — it happens later and in another process.
- **Protocol admission is centralized.** `modelprotocols.py` owns the selectable role, input, output and
  action contracts; catalog admission calls that table. At the pinned 2026-09-19 catalog revision,
  230 of 1144 recipes admit (102 supercombo, 98 vision+policy, 17 off+on+vision, 13 driver monitoring).
  Most refusals are catalog findings: 689 unresolved configuration keys, 121 AMD/big targets and
  16 artifacts with no published bytes. `semantics_unverified`, `requires_runner_check` and
  `qualification: consumer_owned` remain catalog findings, not compatibility claims.
- **The worker publishes job progress and panels format it.** See `modelsd.py` for worker behavior and
  `modelview.py` for progress text and elapsed-time rendering.
- **Store limits:** 300 MB per artifact, 8 GB for the store under `data_dir("models")`, and 2 GB of
  free space left alone. Package/interface verification is implemented in `modelcatalog.py`; the
  boot commit and runtime verification are in `models.py` and `modelruntime.py`.
- **The names and the groups are the catalog's, and the fork names nothing.** openmodels carries a
  model's own display `name`, a `short_name` and a `folder` as reviewed claims from a pinned naming
  source, each checked against an exact commit and artifact set before it is published (`index/names.py`,
  `index/model_names.json`); the browse index copies all three and the picker prints them verbatim.
  A row the catalog has no claims for keeps its generated name and falls back to the group the fork
  *does* own — the protocol the recipe needs, then the model class it was built for — so an older
  snapshot, or the ~1000 models with no published name, still group sensibly. The only fork-side
  edit is a row's name dropping the `Driving · ` prefix a generated name repeats, because the picker
  is already scoped to one kind. **The vendored `contracts.py` line that lets those two keys through
  is load-bearing**: the catalog's `model` object is closed, so a device that refreshed a catalog
  carrying them against a copy without the line would reject the whole snapshot. That is why
  `test_modelcatalog.py` synthesizes a snapshot with the keys and reads it through the vendored loader.
- **The picker is a tree, and the two device trees differ.** tizi
  opens `ModelTreeDialog` — the model in effect pinned at the top, `+`/`-` group headers over
  indented rows, a star per row, a search box, Cancel/Select with Select live only when the pick
  would change something — built on upstream's `MultiOptionDialog`, `Button`, `Scroller` and
  `Keyboard`, so it is fork code with no new seam. mici is a two-level drill-down (groups, then one
  group's models) because a card column has no room for a tree, and a long press is its star. What
  choosing a row uses one shared policy (`modelview.choose_policy`); tizi's footer comes from
  `chooser_hint`, while mici keeps its onroad detail dialog and tizi stays silent when blocked.
  The picker is the catalog browser; the models page handles job, cancel, reboot, storage and refresh.
- **Stars are recipe digests in one param, and a stale one is free.** `MoonpilotModelsFavs` is
  `;`-joined digests, so a favorite whose catalog entry is gone simply matches no row and the string
  is rewritten on the next change; the bundled model is not a recipe and cannot be starred. The
  starred models get their own group *and* stay in the group the catalog gave them, because a group
  is where a model is and a star is a separate fact about it.

The vendored SDK is a copy, not a dependency: `moonpilot/vendor/openmodels/` carries the files and the
sync command in its `README.md`. Catalog access is in `moonpilot/modelcatalog.py`; the compiler also
uses the SDK's metadata reader, and the worker uses its package downloader. Its README names every
local edit, the presentation-key line above among them. The copy and the catalog fixture are the two
paths in this tree whose prose is not en-US, and they are the two fork entries in `pyproject.toml`'s
codespell `skip`; `ruff` and `ty` see both, and `moonpilot/tests/test_modelcatalog.py` reads the
fixture through the vendored SDK, so a copy that drifts fails the suite rather than compiling.

The one ceiling the picker cannot lift: a model reaches a group only if the catalog publishes one, and
at the revision above that is 50 of 1146 entries. The rest group by protocol and class under generated
names, which is the honest answer — a marketing name for a comma commit would be a claim nobody
checked. openmodels is where that would change, through a reviewed naming record.

Local checks: `tools/op.sh test moonpilot/tests/test_models.py moonpilot/tests/test_modelcatalog.py
moonpilot/tests/test_modelruntime.py` runs the focused suite; `moonpilot.modelbuild smoke` compiles
and loads a model for one inference frame.
The bundled `driving_supercombo.onnx` declares the stateful `new_img`/`state_*_q` contract, which
`comma.supercombo.v1` refuses rather than half-feeding. Driving smoke therefore needs a catalog or
retained supercombo artifact whose inputs match the fork protocol.

