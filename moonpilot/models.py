"""The model store and boot commit.

A driver may replace the model the car drives with one from the openmodels catalog. The catalog
publishes bytes and a structural identity, the device downloads and compiles them
(`moonpilot/modelcatalog.py`, `moonpilot/modelbuild.py`, `moonpilot/modelsd.py`), and this module
owns the store and says which selection the next boot may run. Protocol and admission rules live in
`moonpilot/modelprotocols.py`, worker request/status rules in `moonpilot/modelstatus.py`, and panel
wording and chooser state in `moonpilot/modelview.py`.

Three things are worth knowing before reading the rest.

**Selections are strings, and `""` is the bundled model.** Non-empty selections are 64-hex recipe
digests from the catalog; the value names the package and build directories.

**The boot snapshot is the model-process input.** `commit_boot_selection` publishes
`MoonpilotModelsBoot` once at manager startup.

**A failure is never silent and never a fallback mid-drive.** Every refusal ends up as a string in
the boot entry's `fallback` and the car runs the bundled model — the driver's selection is
preserved, the panel shows why, and nothing swaps a model under a running prediction.
"""


import hashlib
import json
import os
from typing import Any

from moonpilot import modelprotocols, paths


# --- param rows --------------------------------------------------------------
# Read declared defaults with return_default=True; get_bool ignores an unset row's declared default.
DRIVING_KEY = "MoonpilotModelsDriving"
MONITORING_KEY = "MoonpilotModelsMonitoring"
BOOT_KEY = "MoonpilotModelsBoot"
REQUEST_KEY = "MoonpilotModelsRequest"
STATUS_KEY = "MoonpilotModelsStatus"
ACTIVE_DRIVING_KEY = "MoonpilotModelsActiveDriving"
ACTIVE_MONITORING_KEY = "MoonpilotModelsActiveMonitoring"
FAVS_KEY = "MoonpilotModelsFavs"
ACTIVE_FAILURE_PREFIX = "stock: "


def active_failure(reason: str) -> str:
  return f"{ACTIVE_FAILURE_PREFIX}{reason}"


def active_failure_reason(text: str) -> str | None:
  return text[len(ACTIVE_FAILURE_PREFIX):] if text.startswith(ACTIVE_FAILURE_PREFIX) else None

REBOOT_KEY = "DoReboot"  # upstream's own param, read by the manager power path (manager.py:204)

# Two kinds, because two processes run models and a param has one writer: `modeld` reads the
# driving half of the boot snapshot, `dmonitoringmodeld` the monitoring half.
DESIRED_KEY = {modelprotocols.DRIVING: DRIVING_KEY, modelprotocols.MONITORING: MONITORING_KEY}
ACTIVE_KEY = {modelprotocols.DRIVING: ACTIVE_DRIVING_KEY, modelprotocols.MONITORING: ACTIVE_MONITORING_KEY}

# The camera resolutions a driving build compiles for: the two the device's cameras report
# (`_ar_ox_fisheye`, `_os_fisheye`). A build is only valid for the set it was compiled for, which
# is why the resolutions are part of the build key and the record.
CAMERA_RESOLUTIONS = ((1928, 1208), (1344, 760))


# The compiler and the runtime must share this one definition.
def nv12_copy_size(stride: int, y_height: int, uv_height: int) -> int:
  return stride * (y_height + uv_height)


SCHEMA = 1
NOT_BUILT = "not built yet"
BUILD_FAILED = "the build failed"
BUILD_INCOMPLETE = "the build is incomplete"
BUILD_FAILED_PREFIX = "build failed: "
BUILD_MISMATCH = "the build was made for another protocol"
ARTIFACT_CHANGED = "the installed bytes changed since the build"
UNKNOWN_SELECTION = "not a selection"
WRONG_KIND = "selected for the other kind"
SELECTED_HINT = "switch to another model first"
PACKAGED = "packaged"
BUILT = "built"
FAILED = "failed"
MODEL_STATES = (PACKAGED, BUILT, FAILED)

def short(value: str) -> str:
  return value[:12]


# --- store layout ------------------------------------------------------------
def models_root() -> str:
  return paths.data_dir("models")


def catalog_file() -> str:
  """The last validated catalog snapshot, as fetched. The worker's, never a panel's."""
  return os.path.join(models_root(), "catalog", "current.json")


def browse_file() -> str:
  """The display index the panels page: derived from `catalog_file()` by the worker."""
  return os.path.join(models_root(), "catalog", "browse.json")


def packages_dir() -> str:
  return os.path.join(models_root(), "packages")


def builds_dir() -> str:
  return os.path.join(models_root(), "builds")


def tmp_dir() -> str:
  """Scratch space for the worker; tinygrad's temporary files stay out of the checkout."""
  return os.path.join(models_root(), "tmp")


def build_key(selection: str, protocol: str, resolutions: tuple[tuple[int, int], ...]) -> str:
  """The build directory name: one directory per (selection, protocol, resolution set).

  A changed member, protocol or resolution set is a *new* directory rather than a stale one that
  gets loaded by mistake -- the failure this key exists to make impossible, because the model
  pickle is keyed on nothing else.
  """
  raw = "|".join([selection, protocol, *(f"{w}x{h}" for w, h in resolutions)])
  return hashlib.sha256(raw.encode()).hexdigest()


def build_dir(selection: str, protocol: str, resolutions: tuple[tuple[int, int], ...]) -> str:
  return os.path.join(builds_dir(), build_key(selection, protocol, resolutions))


def build_model(key: str) -> str:
  return os.path.join(builds_dir(), key, "model.pkl")


def build_record(key: str) -> str:
  return os.path.join(builds_dir(), key, "build.json")


def is_recipe(selection: str) -> bool:
  return len(selection) == 64 and all(c in "0123456789abcdef" for c in selection)













# --- store records -----------------------------------------------------------
def read_json(path: str) -> Any:
  try:
    with open(path, encoding="utf-8") as handle:
      return json.loads(handle.read())
  except (OSError, ValueError):
    return None


def write_json(path: str, value: Any) -> None:
  """Atomic write, overwriting. The worker's only write primitive."""
  from openpilot.common.utils import atomic_write

  os.makedirs(os.path.dirname(path), exist_ok=True)
  with atomic_write(path, "w", overwrite=True) as handle:
    json.dump(value, handle, sort_keys=True)


def load_build(key: str) -> dict | None:
  record = read_json(build_record(key))
  return record if isinstance(record, dict) and record.get("schema") == SCHEMA else None


def build_state(key: str) -> tuple[str, str | None]:
  """`(state, error)` for a build directory: `packaged` when nothing is built, `failed` with the
  compiler's message, `built` when `model.pkl` and the record are both there."""
  record = load_build(key)
  if record is None:
    return PACKAGED, None
  if record.get("state") != BUILT:
    return FAILED, record.get("error") if isinstance(record.get("error"), str) else BUILD_FAILED
  if not os.path.isfile(build_model(key)):
    return FAILED, BUILD_INCOMPLETE
  return BUILT, None


def write_build_record(key: str, protocol: str, action_contract: str, version: str) -> None:
  write_json(build_record(key), {
    "schema": SCHEMA,
    "state": BUILT,
    "error": None,
    "protocol": protocol,
    "provenance": {"version": version},
    "action_contract": action_contract,
  })


def write_failed_build(key: str, error: str) -> None:
  write_json(build_record(key), {"schema": SCHEMA, "state": FAILED, "error": error})


def package_dir(recipe: str) -> str:
  return os.path.join(packages_dir(), recipe)


def load_package(recipe: str) -> dict | None:
  """The recipe document of an installed package, and its profile document. `None` when the package
  is absent or its own files disagree with the digest it is stored under -- `ModelStore` writes both
  under `packages/<recipe>/`, and the digest is a hash of the recipe bytes, so this is checkable
  without the catalog."""
  document = read_json(os.path.join(package_dir(recipe), "recipe.json"))
  if not isinstance(document, dict):
    return None
  canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
  if hashlib.sha256(canonical.encode()).hexdigest() != recipe:
    return None
  profile = read_json(os.path.join(package_dir(recipe), "profile.json"))
  if not isinstance(profile, dict):
    return None
  return {"recipe": document, "profile": profile}


def package_members(recipe: str, document: dict) -> dict[str, dict]:
  """`{role: {"path", "sha256", "size", "metadata"}}` for a stored package. Paths are inside the
  package; nothing here checks the bytes (see `verify_members`)."""
  members = document.get("members")
  if not isinstance(members, dict):
    return {}
  result = {}
  for role, member in members.items():
    artifact = member.get("artifact") if isinstance(member, dict) else None
    if not isinstance(artifact, dict):
      return {}
    result[role] = {
      "path": os.path.join(package_dir(recipe), str(artifact.get("sha256", ""))),
      "sha256": str(artifact.get("sha256", "")),
      "size": int(artifact.get("size", 0) or 0),
      "metadata": member.get("metadata") if isinstance(member.get("metadata"), dict) else {},
      "configuration": member.get("configuration") if isinstance(member.get("configuration"), dict) else {},
    }
  return result


def verify_members(members: dict[str, dict]) -> str | None:
  """`None` when every member file is present with its recorded size and digest; otherwise the
  reason. The boot commit and both runtime factories run this: a build is only as good as the bytes
  it was compiled from, and the store is on a partition that can be truncated by a factory reset."""
  for member in members.values():
    try:
      with open(member["path"], "rb") as handle:
        if os.fstat(handle.fileno()).st_size != member["size"] or hashlib.file_digest(handle, "sha256").hexdigest() != member["sha256"]:
          return ARTIFACT_CHANGED
    except OSError:
      return modelprotocols.NOT_INSTALLED
  return None


def member_view(member: Any) -> dict:
  """Normalize one catalog member document to the fields admission reads."""
  if not isinstance(member, dict):
    return {}
  metadata = member.get("metadata") if isinstance(member.get("metadata"), dict) else {}
  artifact = member.get("artifact") if isinstance(member.get("artifact"), dict) else {}
  view = {
    "input_shapes": metadata.get("input_shapes") or {},
    "slices": metadata.get("output_slices") or {},
    "format": artifact.get("format"),
    "size": int(artifact.get("size", 0) or 0),
    "targets": member.get("targets") if isinstance(member.get("targets"), list) else [],
    "configuration": member.get("configuration") if isinstance(member.get("configuration"), dict) else {},
  }
  if isinstance(member.get("published_at"), str):
    view["published_at"] = member["published_at"]
  return view


def selection_record(selection: str) -> dict | None:
  """Resolve a recipe to its protocol, configuration and members, or None when it is not installed,
  its role set matches no protocol, or its own records do not check out. Pure file I/O: no catalog,
  network or tinygrad, which is what lets the manager call it."""
  if not is_recipe(selection):
    return None
  package = load_package(selection)
  if package is None:
    return None
  members = package_members(selection, package["recipe"])
  if not members:
    return None
  proto = modelprotocols.protocol_for_roles(members)
  if proto is None:
    return None
  # The protocol's own required keys, out of the recipe's top-level configuration: the worker's
  # admission already refused a recipe whose values are unresolved, and the members cannot disagree
  # (the same check reads them all), so this is the one set every consumer -- the compiler's queue
  # lengths and smoothing, the boot snapshot's record -- should read.
  recipe_configuration = package["recipe"].get("configuration") or {}
  configuration = {key: recipe_configuration.get(key) for key in proto.required_configuration}
  return {"protocol": proto, "configuration": configuration, "members": members}


def installed_entries() -> list[dict]:
  """Every installed package, with its build state."""
  entries: list[dict] = []
  try:
    recipes = sorted(d for d in os.listdir(packages_dir()) if not d.startswith(".") and os.path.isdir(package_dir(d)))
  except OSError:
    recipes = []
  for recipe in recipes:
    entry = _entry_for(recipe)
    if entry is not None:
      entries.append(entry)
  return entries


def _entry_for(selection: str) -> dict | None:
  resolved = selection_record(selection)
  if resolved is None:
    return None
  proto: modelprotocols.Protocol = resolved["protocol"]
  members: dict[str, dict] = resolved["members"]
  key = build_key(selection, proto.id, CAMERA_RESOLUTIONS)
  state, error = build_state(key)
  provenance = ""
  record = load_build(key) or {}
  if isinstance(record.get("provenance"), dict):
    provenance = str(record["provenance"].get("version", ""))
  return {
    "recipe": selection,
    "kind": proto.kind,
    "protocol": proto.id,
    "roles": sorted(members),
    "size": sum(m["size"] for m in members.values()),
    "state": state,
    "error": error,
    "provenance": provenance,
  }


# --- boot commit -------------------------------------------------------------
def desired(params: Any, kind: str) -> str:
  """The driver's selection for one kind; unset means the bundled model. `return_default=True`
  keeps this read safe in a bare script as well as after manager initialization."""
  value = params.get(DESIRED_KEY[kind], return_default=True)
  return str(value) if value else ""


def select(params: Any, kind: str, selection: str) -> None:
  """Set one kind's selection. Validates the shape only -- whether the selection can run is the
  boot commit's question, and answering it here would mean the panel could refuse a selection the
  car could run."""
  if selection and not is_recipe(selection):
    raise ValueError(f"not a selection: {selection!r}")
  params.put(DESIRED_KEY[kind], selection, block=True)


def boot(params: Any) -> dict:
  """The committed boot snapshot. Absent or unparsable reads as `bundled` for both kinds, which is
  what a device that has never committed one must do."""
  raw = params.get(BOOT_KEY)
  result = {
    "schema": SCHEMA,
    modelprotocols.DRIVING: empty_boot_entry(""),
    modelprotocols.MONITORING: empty_boot_entry(""),
  }
  if not raw:
    return result
  try:
    data = json.loads(raw)
  except (TypeError, ValueError):
    return result
  if not isinstance(data, dict) or data.get("schema") != SCHEMA:
    return result
  for kind in modelprotocols.KINDS:
    entry = data.get(kind)
    if isinstance(entry, dict):
      result[kind] = _boot_entry(entry)
  return result


def boot_configuration(params: Any, key: str, fallback: float) -> float:
  """One declared configuration value from the committed driving selection, `fallback` when the
  selection, the key or the value is absent.

  The value is read out of the same boot snapshot `modeld` hands `DriverRuntime` (`smoothness`), so a
  consumer that times its command against the model's decode cannot disagree with it -- `controlsd`'s
  curvature reference and torque controller are the two that do. The one divergence is a build that
  fails to load: modeld drops the runtime and uses its module constants, while this still reports what
  the selection declared (the panel already says the model fell back).
  """
  configuration = boot(params)[modelprotocols.DRIVING].get("configuration") or {}
  try:
    return float(configuration.get(key, fallback))
  except (TypeError, ValueError):
    return float(fallback)


def _boot_entry(entry: dict) -> dict:
  kind = entry.get("kind")
  if kind not in (modelprotocols.BUNDLED, modelprotocols.CUSTOM):
    kind = modelprotocols.BUNDLED
  members = entry.get("members") if isinstance(entry.get("members"), dict) else {}
  return {
    "kind": kind,
    "id": str(entry.get("id", "")),
    "protocol": str(entry.get("protocol", "")),
    "build": str(entry.get("build", "")),
    "members": {str(r): str(p) for r, p in members.items()},
    "configuration": entry.get("configuration") if isinstance(entry.get("configuration"), dict) else {},
    "requested": str(entry.get("requested", "")),
    "fallback": entry.get("fallback") if isinstance(entry.get("fallback"), str) else None,
  }


def empty_boot_entry(requested: str) -> dict:
  return {"kind": modelprotocols.BUNDLED, "id": "", "protocol": "", "build": "", "members": {}, "configuration": {}, "requested": requested, "fallback": None}


def commit_boot_selection(params: Any) -> dict:
  """Resolve both desired selections against the store and publish the result. Called once per
  boot, from `manager_init` (openpilot/system/manager/manager.py), before any process starts.

  Every check an execution needs happens here, so that a model process reads one param and either
  loads what it names or reports the reason: the selection is installed, its records parse, its
  protocol is one this fork carries, its kind matches the param it came from, every artifact is on
  disk with its recorded size and digest, and the build for exactly this (selection, protocol,
  resolutions) is present. Anything else writes `bundled` with the reason and leaves the driver's
  selection in place for the panel to explain.
  """
  snapshot = {"schema": SCHEMA}
  for kind in modelprotocols.KINDS:
    requested = desired(params, kind)
    entry = empty_boot_entry(requested)
    if requested:
      entry = _resolve(requested, kind)
    snapshot[kind] = entry
  params.put(BOOT_KEY, json.dumps(snapshot), block=True)
  return boot(params)


def _resolve(selection: str, kind: str) -> dict:
  """One kind's entry: the built model, or `bundled` with the reason it is not."""
  entry = empty_boot_entry(selection)
  if not is_recipe(selection):
    entry["fallback"] = UNKNOWN_SELECTION
    return entry
  resolved = selection_record(selection)
  if resolved is None:
    entry["fallback"] = modelprotocols.NOT_INSTALLED
    return entry
  proto: modelprotocols.Protocol = resolved["protocol"]
  if proto.kind != kind:
    entry["fallback"] = WRONG_KIND
    return entry
  members: dict[str, dict] = resolved["members"]
  if (reason := verify_members(members)) is not None:
    entry["fallback"] = reason
    return entry
  key = build_key(selection, proto.id, CAMERA_RESOLUTIONS)
  directory = build_dir(selection, proto.id, CAMERA_RESOLUTIONS)
  if (load_build(key) or {}).get("protocol") not in (None, proto.id):
    # The key already encodes the protocol, so this is a store that was edited under its own name --
    # caught here so the panel can explain it, and again in the runtime, which cannot see the commit.
    entry["fallback"] = BUILD_MISMATCH
    return entry
  state, error = build_state(key)
  if state != BUILT:
    entry["fallback"] = error or NOT_BUILT
    return entry
  entry.update(
    {
      "kind": modelprotocols.CUSTOM,
      "id": selection,
      "protocol": proto.id,
      "build": directory,
      "members": {role: member["path"] for role, member in members.items()},
      "configuration": resolved["configuration"],
    }
  )
  return entry


def restart_needed(params: Any) -> bool:

  return any(restart_pending(params, kind) for kind in modelprotocols.KINDS)


def request_reboot(params: Any) -> None:
  params.put_bool(REBOOT_KEY, True, block=True)


def restart_pending(params: Any, kind: str) -> bool:
  """Whether the driver's selection differs from the one this boot committed -- the reason both
  panels say `(restart to apply)`. The row takes a restart because the model processes read the
  snapshot once, at their own start."""
  return desired(params, kind) != boot(params)[kind]["requested"]


