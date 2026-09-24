# openmodels, vendored

The catalog client from [`jjolano/openmodels`](https://github.com/jjolano/openmodels), copied in
rather than depended on: the device has no `pip` and no `uv` (AGENTS.md, **Dependencies**), the
wheel is not on the AGNOS image, and the client is stdlib-only — so a copy is both the only way to
reach it and cheaper than the dependency machinery `moonpilot/deps.py` exists for.

|Vendored file|Upstream path|Why|
|---|---|---|
|`contracts.py`|`openmodels/contracts.py`|schema-1 manifest/recipe identity, `Manifest`, `Recipe.check`|
|`client.py`|`openmodels/client.py`|`Catalog`, `ModelStore`, atomic staged installs, cancellation|
|`models.py`|`openmodels/models.py`|the display-group helper behind `Catalog.models()`|
|`metadata.py`|`index/metadata.py`|dependency-free ONNX reader, and the only safe `output_slices` decoder|
|`__init__.py`|`openmodels/__init__.py`|the public names above|
|`LICENSE`, `THIRD_PARTY_NOTICES.md`|the repository root|MIT obligation|

Source revision: `8ec3b2f507744a782e64c7a07c424e37c107e36f`.

To sync, from a checkout of that revision:

```sh
cd /path/to/openmodels && git checkout 8ec3b2f507744a782e64c7a07c424e37c107e36f
for f in openmodels/__init__.py openmodels/contracts.py openmodels/client.py openmodels/models.py; do
  { echo '# ruff: noqa'; cat "$f"; } > /path/to/moonpilot/moonpilot/vendor/openmodels/$(basename "$f")
done
{ echo '# ruff: noqa'; cat index/metadata.py; } > /path/to/moonpilot/moonpilot/vendor/openmodels/metadata.py
cp LICENSE THIRD_PARTY_NOTICES.md /path/to/moonpilot/moonpilot/vendor/openmodels/
```

The only local edits are the first line `# ruff: noqa` -- upstream's own style is not this repo's --
one line in `contracts.py`, `"short_name": TEXT, "folder": TEXT,` in the snapshot's `model` object,
and the `moonpilot/vendor/openmodels/*` and `moonpilot/tests/fixtures/*` entries in `pyproject.toml`'s
codespell `skip`: both are copied third-party text, and this tree's codespell dictionary is en-GB to
en-US, so this copy's `cancelled`/`unparseable` and the catalog fixture's `metres` would otherwise
fail a gate that has nothing to do with either. That one line is not optional and not a fork
extension: the `model` object is closed (`additionalProperties: False`), so once the catalog publishes
the two keys, `Catalog.__init__` rejects the entire snapshot without it -- a picker whose groups
never arrive must not cost the device its catalog. They are optional in the required-key list, so a
snapshot that predates them still validates, and the picker's grouping falls back to protocol and
model class (see `entry_group` in `moonpilot/models.py`). Nothing about the code itself is configured away: `ruff` and `ty` see these files
and `moonpilot/tests/test_modelcatalog.py` exercises them, so a copy that drifts or goes missing
fails rather than compiles. `metadata.py` is
upstream's `index/metadata.py` flattened into this package so the ONNX reader travels with the
client. `moonpilot/tests/test_modelcatalog.py` exercises the copy, so a sync that drops a file
fails the suite rather than the device.

Nothing here may be imported on the init path: `moonpilot/modelcatalog.py` is the only importer,
and it is reached by `modelsd` and the tests, not by the manager, the planner or the UI.
