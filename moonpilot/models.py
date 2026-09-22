"""The model store, and the one place every rule about it lives.

A driver may replace the model the car drives with one from the openmodels catalog. That is a
selection, not a setting: the catalog publishes bytes and a structural identity, the device
downloads and compiles them (moonpilot/modelcatalog.py, moonpilot/modelbuild.py, moonpilot/modelsd.py),
and this module says which selections exist, where their files live, which of them a boot may
actually run, and what every panel prints about them.

Four things are worth knowing before reading the rest.

**Selections are strings, and `""` is the bundled model.** A non-empty selection is one 64-hex recipe
digest from the catalog. The string is the whole identity: it names the package directory and build
directory, so a selection that is not installed is a selection whose directories do not exist.

**Nothing here is on a load path.** The modules the manager and the two UI trees import at boot —
`moonpilot.procs`, `moonpilot.ui.settings*` — import this one, so it is stdlib plus
`moonpilot.paths` and nothing else. The catalog SDK, tinygrad and the runtime are all behind
imports the worker and the seams do themselves.

**The boot snapshot is the seam between the worker and the car.** `commit_boot_selection` runs
once, in the manager, after the `CLEAR_ON_MANAGER_START` clears and the default seeding and before
the first `ensure_running`, and writes every check it made into `MoonpilotModelsBoot`. The model
processes read that param and nothing else, because a per-process read of the desired value is not
a coherent switch: `dmonitoringmodeld` runs during the offroad driver preview and can outlive an
onroad cycle, so two processes reading two params at two times can load two generations.

**A failure is never silent and never a fallback mid-drive.** Every refusal ends up as a string in
the boot entry's `fallback` and the car runs the bundled model — the driver's selection is
preserved, the panel shows why, and nothing swaps a model under a running prediction.
"""

import datetime
import hashlib
import json
import os
import shutil
import time
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from moonpilot import paths

# --- param rows --------------------------------------------------------------
# The declared defaults and flags are in moonpilot/params_keys.h. Read them with
# `params.get(key, return_default=True)` where a default exists -- `get_bool` ignores the declared
# default and reports off for an unset param (AGENTS.md, Features).
DRIVING_KEY = "MoonpilotModelsDriving"
MONITORING_KEY = "MoonpilotModelsMonitoring"
BOOT_KEY = "MoonpilotModelsBoot"
REQUEST_KEY = "MoonpilotModelsRequest"
STATUS_KEY = "MoonpilotModelsStatus"
ACTIVE_DRIVING_KEY = "MoonpilotModelsActiveDriving"
ACTIVE_MONITORING_KEY = "MoonpilotModelsActiveMonitoring"

# Two kinds, because two processes run models and a param has one writer: `modeld` reads the
# driving half of the boot snapshot, `dmonitoringmodeld` the monitoring half.
DRIVING = "driving"
MONITORING = "monitoring"
KINDS = (DRIVING, MONITORING)
DESIRED_KEY = {DRIVING: DRIVING_KEY, MONITORING: MONITORING_KEY}
ACTIVE_KEY = {DRIVING: ACTIVE_DRIVING_KEY, MONITORING: ACTIVE_MONITORING_KEY}

# The camera resolutions a driving build compiles for: the two the device's cameras report
# (`_ar_ox_fisheye`, `_os_fisheye`). A build is only valid for the set it was compiled for, which
# is why the resolutions are part of the build key and the record.
CAMERA_RESOLUTIONS = ((1928, 1208), (1344, 760))
# The DM warp compiles to the model's own input size (openpilot/common/transformations/model.py).
DM_INPUT_SIZE = (1440, 960)


# The compiler and the runtime must share this one definition.
def nv12_copy_size(stride: int, y_height: int, uv_height: int) -> int:
  return stride * (y_height + uv_height)


SCHEMA = 1


# --- store layout ------------------------------------------------------------
# `<models_root>/` is `data_dir()`'s tree: on the data partition, beside the params, and not
# managed by the drive deleter -- so the caps below are the store's own bound (AGENTS.md, Storage).
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


def artifacts_dir() -> str:
  return os.path.join(packages_dir(), ".artifacts")


def builds_dir() -> str:
  return os.path.join(models_root(), "builds")


def tmp_dir() -> str:
  """Scratch space for the worker. Inside the store on purpose: `dump_oob`'s and tinygrad's temp
  files must never land in the checkout, which is git state (AGENTS.md, Storage)."""
  return os.path.join(models_root(), "tmp")


def build_key(selection: str, protocol: str, resolutions: tuple[tuple[int, int], ...] = CAMERA_RESOLUTIONS) -> str:
  """The build directory name: one directory per (selection, protocol, resolution set).

  A changed member, protocol or resolution set is a *new* directory rather than a stale one that
  gets loaded by mistake -- the failure this key exists to make impossible, because the model
  pickle is keyed on nothing else.
  """
  raw = "|".join([selection, protocol, *(f"{w}x{h}" for w, h in resolutions)])
  return hashlib.sha256(raw.encode()).hexdigest()


def build_dir(selection: str, protocol: str, resolutions: tuple[tuple[int, int], ...] = CAMERA_RESOLUTIONS) -> str:
  return os.path.join(builds_dir(), build_key(selection, protocol, resolutions))


def build_model(key: str) -> str:
  return os.path.join(builds_dir(), key, "model.pkl")


def build_record(key: str) -> str:
  return os.path.join(builds_dir(), key, "build.json")


def is_recipe(selection: str) -> bool:
  return len(selection) == 64 and all(c in "0123456789abcdef" for c in selection)


def valid_selection(selection: str) -> bool:
  return is_recipe(selection)


# --- protocols: what may be selected -----------------------------------------
# A protocol is a statement about an *interchange controller*, not about a catalog entry: it names
# the roles a selection must have, the inputs the fork's runtime feeds each role, the outputs a
# consumer in this tree reads, and how the action head is contracted. Admission
# (moonpilot/modelcatalog.py) checks a recipe against exactly this table, and nothing else may
# decide what the device executes.
#
# Two things about the shape:
#
# - **The slice requirement is the union over the members, not a per-role pin.** Every consumer
#   here (`fill_model_msg`, `fill_pose_msg`, `parse_model_output`) reads names off one dict, so what
#   has to hold is that *some* member produces each required name. That is the property the real
#   split generations satisfy: a family-C set puts `lane_lines`/`lead` in the off-policy member and
#   `plan`/`desire_state` in whichever policy member carries them, and pinning the slices per role
#   refuses sets that run fine -- 24 of the catalog's family-C recipes are admissible under the
#   union and would be refused by any per-role pin that also admits them.
# - **`action_contract` is per protocol, and it is the one place a wrong guess is a steering
#   request.** `plan` decodes both lateral and longitudinal from the trajectory through upstream's
#   `get_action_from_model`; `lateral-accel` is that same current function's `action` branch
#   (`action[0,0] / max(1, v_ego)**2`, i.e. the head's second column is an acceleration);
#   `curvature-100` is the older convention (`action[0,0] / 100`, the head's first column is a
#   curvature times 100) that `openpilot@9757bf10` used. Upstream switched the meaning of that head
#   in `249cafe897deb433944c84fa6c3c6ce568f727e1` ("Prereqs deep models", 2026-06-01), and no shape
#   in the catalog distinguishes the two eras, so admission refuses an action-declaring member on
#   the wrong side of that date rather than scale it by the other convention's constant.
LATERAL_ACCEL_SINCE = datetime.date(2026, 6, 1)

# `ModelConstants`-era `STOCK_SLICES` minus `pad`: every name `fill_model_msg`, `fill_pose_msg` and
# the loop's own `prev_feat` read from the model output dict.
DRIVING_SLICES = frozenset(
  {
    "meta",
    "desire_pred",
    "pose",
    "wide_from_device_euler",
    "road_transform",
    "lane_lines",
    "lane_lines_prob",
    "road_edges",
    "lead",
    "lead_prob",
    "hidden_state",
    "plan",
    "desire_state",
  }
)

# What `dmonitoringmodeld`'s `parse_model_output`/`fill_driver_data` read. `sleep_prob_*` and
# `features` are deliberately absent: they are newer heads, nothing in this tree reads
# `driverStateV2.sleepProb` (`openpilot/selfdrive/monitoring/policy.py`), and a legacy 551-slice
# model that lacks them is safe rather than degraded.
DM_SLICES = frozenset(
  {
    "face_descs_lhd",
    "face_descs_rhd",
    "face_prob_lhd",
    "face_prob_rhd",
    "left_eye_prob_lhd",
    "left_eye_prob_rhd",
    "right_eye_prob_lhd",
    "right_eye_prob_rhd",
    "left_blink_prob_lhd",
    "left_blink_prob_rhd",
    "right_blink_prob_lhd",
    "right_blink_prob_rhd",
    "sunglasses_prob_lhd",
    "sunglasses_prob_rhd",
    "using_phone_prob_lhd",
    "using_phone_prob_rhd",
    "wheel_on_right",
  }
)

CONFIGURATION_KEYS = ("frame_skip", "LAT_SMOOTH_SECONDS", "LONG_SMOOTH_SECONDS")


@dataclass(frozen=True)
class Protocol:
  """One way of running a model, pinned to the roles, inputs and outputs a consumer here needs."""

  id: str
  kind: str
  roles: tuple[str, ...]
  role_inputs: dict[str, frozenset[str]]  # required ONNX input names, per role
  optional_inputs: dict[str, frozenset[str]]  # fed and packed when the member declares them
  slices: frozenset[str]  # required output names, somewhere in the member set
  optional_slices: frozenset[str]  # published when present, unread either way
  action_contract: str
  required_configuration: tuple[str, ...] = CONFIGURATION_KEYS


SUPERCOMBO = Protocol(
  id="comma.supercombo.v1",
  kind=DRIVING,
  roles=("supercombo",),
  role_inputs={
    "supercombo": frozenset({"img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"}),
  },
  optional_inputs={"supercombo": frozenset()},
  slices=DRIVING_SLICES,
  optional_slices=frozenset({"action", "desired_curvature", "sim_pose", "stop_lines", "stop_lines_prob", "pad"}),
  action_contract="lateral-accel",
)

SPLIT_VISION_POLICY = Protocol(
  id="comma.split-vision-policy.v1",
  kind=DRIVING,
  roles=("vision", "on_policy"),
  role_inputs={
    "vision": frozenset({"img", "big_img"}),
    "on_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
  },
  optional_inputs={"vision": frozenset(), "on_policy": frozenset({"action_t"})},
  slices=DRIVING_SLICES,
  optional_slices=frozenset({"action", "desired_curvature", "sim_pose", "pad"}),
  action_contract="plan",
)

SPLIT_VISION_OFF_ON = Protocol(
  id="comma.split-vision-off-on.v1",
  kind=DRIVING,
  roles=("vision", "off_policy", "on_policy"),
  role_inputs={
    "vision": frozenset({"img", "big_img"}),
    "off_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
    "on_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
  },
  optional_inputs={
    "vision": frozenset(),
    "off_policy": frozenset({"action_t"}),
    "on_policy": frozenset({"action_t"}),
  },
  slices=DRIVING_SLICES,
  optional_slices=frozenset({"desired_curvature", "sim_pose", "pad"}),
  action_contract="curvature-100",
)

DMONITORING = Protocol(
  id="comma.dmonitoring.v1",
  kind=MONITORING,
  roles=("dmonitoring",),
  role_inputs={"dmonitoring": frozenset({"input_img", "calib"})},
  optional_inputs={"dmonitoring": frozenset()},
  slices=DM_SLICES,
  optional_slices=frozenset({"sleep_prob_lhd", "sleep_prob_rhd", "features"}),
  action_contract="plan",
  required_configuration=(),
)

PROTOCOLS: tuple[Protocol, ...] = (SUPERCOMBO, SPLIT_VISION_POLICY, SPLIT_VISION_OFF_ON, DMONITORING)


def protocol(protocol_id: str) -> Protocol | None:
  return next((p for p in PROTOCOLS if p.id == protocol_id), None)


def protocol_for_roles(roles: Any) -> Protocol | None:
  """The protocol whose role set is exactly `roles`. A role set no protocol claims is refused
  rather than mapped onto the nearest one."""
  wanted = frozenset(roles)
  return next((p for p in PROTOCOLS if frozenset(p.roles) == wanted), None)


def features_role(proto: Protocol) -> str:
  """The role whose output carries `hidden_state`, i.e. the one whose feature the runtime feeds
  back as `prev_feat`. Every protocol has exactly one, which admission enforces."""
  return proto.roles[0] if len(proto.roles) == 1 else "vision"


# --- what a runnable member set looks like -----------------------------------
# Admission (moonpilot/modelcatalog.py) checks a catalog recipe with these, and modelbuild checks
# the stored members again before compiling. One rule, two callers, so a recipe the panel offers is
# a recipe the worker will accept.
#
# Nothing here needs the catalog SDK: every input is a plain dict of what a document recorded --
# `input_shapes`, `output_slices`, `targets`, `format` -- which is exactly what the catalog's
# metadata reader and the stored `recipe.json` both carry.
#
# A member that declares an input the protocol does not list is refused rather than half-fed. That
# is the rule that refuses the pre-`action_t` monoliths (`lateral_control_params`,
# `prev_desired_curv`, `nav_features`), and it is also what would refuse a member declaring
# `prev_action`: no recipe in the 2026-09-19 catalog declares it, so nothing here carries the
# queue for it, and a model that needs one would be a model this runtime cannot run yet.
UNSUPPORTED_INPUT = "needs inputs this runtime does not supply"
UNKNOWN_ROLE = "a role this protocol does not have"
MISSING_OUTPUTS = "missing outputs this runtime publishes"
HIDDEN_STATE_MISMATCH = "the vision feature width does not match the policy's features_buffer"
NOT_ONNX = "not an ONNX model"
NOT_QCOM = "chestnut/AMD models are not selectable"
NO_METADATA = "the catalog records no interface for this member"
ACTION_ERA = "predates the fork's action contract"


def member_reason(role: str, proto: Protocol, member: dict) -> str | None:
  """Why one member cannot be run as this protocol's `role`, or `None`.

  `member` carries `input_shapes` (name -> shape), `slices` (name -> [start, stop, step]), `format`,
  `targets` and `published_at` (ISO 8601, the catalog's own observation of when the artifact was
  published; absent for a stored package, where the check has already been made at install time).
  """
  required = proto.role_inputs.get(role)
  if required is None:
    return UNKNOWN_ROLE
  shapes = member.get("input_shapes")
  if not isinstance(shapes, dict) or not shapes:
    return NO_METADATA
  if member.get("format") != "onnx":
    return NOT_ONNX
  targets = member.get("targets")
  if isinstance(targets, list) and "QCOM" not in targets:
    return NOT_QCOM
  unsupported = set(shapes) - required - proto.optional_inputs.get(role, frozenset())
  if unsupported:
    return UNSUPPORTED_INPUT
  if not required <= set(shapes):
    return UNSUPPORTED_INPUT
  return _action_era_reason(proto, member)


def _action_era_reason(proto: Protocol, member: dict) -> str | None:
  """The action head's *meaning* changed on 2026-06-01, and nothing in a model's structure records
  which side of it a head was trained on. The protocol pins the convention it decodes, so a member
  that declares an `action` slice on the wrong side of the change is refused rather than scaled by
  the other era's constant -- a wrong scale factor here is a wrong steering request, silently."""
  if "action" not in (member.get("slices") or {}):
    return None
  published = member.get("published_at")
  if not published:
    return None
  try:
    published_date = datetime.datetime.fromisoformat(published).date()
  except (TypeError, ValueError):
    return None
  # `plan` and `lateral-accel` both end in upstream's current function, which decodes the `action`
  # head when one is present -- so an action head is only admissible there after the change, and only
  # before it under the family-C protocol, whose decode is pinned to the old convention.
  old_head = published_date < LATERAL_ACCEL_SINCE
  if proto.action_contract == "curvature-100":
    return ACTION_ERA if not old_head else None
  return ACTION_ERA if old_head else None


def set_reason(proto: Protocol, members: dict[str, dict]) -> str | None:
  """Why a whole member set cannot be run, or `None`: the role set must be the protocol's, every
  member must pass `member_reason`, the union of their outputs must cover what the consumers read,
  and the feature the runtime feeds back must have the width the policy's buffer expects.

  The union is the right shape here rather than a per-role pin -- see the comment above `PROTOCOLS`.
  """
  if frozenset(members) != frozenset(proto.roles):
    return NOT_INSTALLED
  for role, member in members.items():
    if (reason := member_reason(role, proto, member)) is not None:
      return f"{role}: {reason}"
  published = set()
  for member in members.values():
    published |= set(member.get("slices") or {})
  if missing := proto.slices - published:
    return f"{MISSING_OUTPUTS}: {', '.join(sorted(missing))}"
  vision = members.get(features_role(proto))
  if vision is None:
    return NOT_INSTALLED
  hidden = (vision.get("slices") or {}).get("hidden_state")
  width = _slice_width(hidden)
  policy = members.get("on_policy") or members.get("supercombo") or vision
  buffer_shape = (policy.get("input_shapes") or {}).get("features_buffer")
  if width is not None and isinstance(buffer_shape, list) and buffer_shape:
    if width != buffer_shape[-1]:
      return HIDDEN_STATE_MISMATCH
  return None


def _slice_width(bounds: Any) -> int | None:
  try:
    start, stop = bounds[0], bounds[1]
  except (TypeError, IndexError):
    return None
  if not isinstance(start, int) or not isinstance(stop, int):
    return None
  return stop - start


# --- capacity ----------------------------------------------------------------
MAX_ARTIFACT_BYTES = 300 * 1024 * 1024  # a member bigger than this is refused before any download
MIN_FREE_BYTES = 2 * 1024**3  # headroom the store leaves the rest of /data
MAX_STORE_BYTES = 8 * 1024**3  # packages + builds together; removal is the driver's action


def _tree_bytes(root: str) -> int:
  total = 0
  for dirpath, _dirnames, filenames in os.walk(root):
    for name in filenames:
      path = os.path.join(dirpath, name)
      try:
        total += os.lstat(path).st_size
      except OSError:
        continue
  return total


def store_usage() -> int:
  """Bytes under `packages/` and `builds/`. The digest cache counts, being hard links into it."""
  return sum(_tree_bytes(d) for d in (packages_dir(), builds_dir()))


def storage_status() -> dict:
  """The `storage` member of the status snapshot: used, free, limit -- all in bytes."""
  return {"used": store_usage(), "free": free_bytes(), "limit": MAX_STORE_BYTES}


def free_bytes() -> int:
  try:
    return int(shutil.disk_usage(models_root()).free)
  except OSError:
    return 0


def capacity_reason(declared_bytes: int) -> str | None:
  """Why a package cannot be fetched now, or None. Called before any work starts, so a refusal is
  a driver-visible sentence and never a download that dies at 90 %."""
  if declared_bytes > MAX_ARTIFACT_BYTES:
    return TOO_LARGE
  # Three copies while it is in flight: the staging file, the digest-cache hard link and the build
  # output, none of which can be assumed to share blocks.
  demand = declared_bytes * 3 + MIN_FREE_BYTES
  if free_bytes() - demand < 0:
    return NO_SPACE
  if store_usage() + declared_bytes > MAX_STORE_BYTES:
    return STORE_FULL
  return None


# --- request and status codecs -----------------------------------------------
# One JSON op in, one JSON snapshot out, both on params: the panels never touch the store, and the
# worker never touches a widget. A malformed request reads as None -- dropped, never retried,
# because a retry of something unparsable is a loop.
OPS = ("refresh", "install", "remove", "cancel")
PHASES = ("waiting-network", "waiting-offroad", "downloading", "verifying", "building", "done", "error", "canceled")


def job_active(job: dict | None) -> bool:
  """Whether a published job can still be canceled."""
  return bool(job) and job.get("phase") in PHASES[:5]


def request(params: Any, op: str, selection: str = "") -> str:
  """Write a request and return its id. The id is what makes a request cancellable: the worker's
  `canceled()` re-reads the param and stops when it no longer holds this id."""
  request_id = str(time.monotonic_ns())
  params.put(REQUEST_KEY, json.dumps({"schema": SCHEMA, "id": request_id, "op": op, "recipe": selection}), block=True)
  return request_id


def read_request(params: Any) -> dict | None:
  raw = params.get(REQUEST_KEY)
  if not raw:
    return None
  try:
    data = json.loads(raw)
  except (TypeError, ValueError):
    return None
  if not isinstance(data, dict) or data.get("schema") != SCHEMA:
    return None
  if data.get("op") not in OPS or not isinstance(data.get("id"), str) or not data["id"]:
    return None
  selection = data.get("recipe", "")
  if not isinstance(selection, str) or (selection and not valid_selection(selection)):
    return None
  return {"schema": SCHEMA, "id": data["id"], "op": data["op"], "recipe": selection}


def clear_request(params: Any) -> None:
  params.remove(REQUEST_KEY)


@dataclass
class Job:
  """The worker's own view of the request it is serving. `PHASES` is the closed set the panels
  switch on, so an unknown phase is a bug and not a rendering case."""

  id: str
  op: str
  selection: str = ""
  phase: str = "waiting-network"
  received: int = 0
  total: int = 0
  error: str | None = None
  metered: bool = False
  # The worker's own monotonic clock, when the job started. It is published as *elapsed seconds*
  # rather than as a timestamp: the panels then need no clock agreement with the worker, and a job
  # whose elapsed is missing (an older worker's snapshot) simply reads without a clock.
  started_at: float = 0.0

  def to_dict(self) -> dict:
    snapshot = {
      "id": self.id,
      "op": self.op,
      "recipe": self.selection,
      "phase": self.phase,
      "received": self.received,
      "total": self.total,
      "error": self.error,
      "metered": self.metered,
    }
    if self.started_at:
      snapshot["elapsed"] = max(0.0, time.monotonic() - self.started_at)
    return snapshot


def _job_from_dict(data: Any) -> dict | None:
  if not isinstance(data, dict):
    return None
  phase = data.get("phase")
  if phase not in PHASES:
    return None
  elapsed = data.get("elapsed")
  return {
    "id": str(data.get("id", "")),
    "op": str(data.get("op", "")),
    "recipe": str(data.get("recipe", "")),
    "phase": phase,
    "received": int(data.get("received", 0) or 0),
    "total": int(data.get("total", 0) or 0),
    "error": data.get("error") if isinstance(data.get("error"), str) else None,
    "metered": bool(data.get("metered", False)),
    "elapsed": max(0.0, float(elapsed)) if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) else None,
  }


def empty_status() -> dict:
  """The status every reader gets when nothing has been published: one shape, so the panels have
  one code path and an absent worker looks exactly like a worker with nothing to say."""
  return {
    "schema": SCHEMA,
    "catalog": {"revision": "", "generated_at": "", "total": 0, "error": None},
    "job": None,
    "installed": [],
    "storage": {"used": 0, "free": 0, "limit": MAX_STORE_BYTES},
  }


_STATUS_CACHE: tuple[str, dict] | None = None


def status(params: Any) -> dict:
  """The worker's snapshot, or the empty one. Never raises: a param written by an older fork must
  not take a panel down at boot.

  One slot of cache, keyed on the raw param text: both trees read this on every frame (their values
  are pushed, not resolved) and the worker writes it at most once a second, so without the cache
  every frame would re-parse the same JSON.
  """
  global _STATUS_CACHE

  raw = params.get(STATUS_KEY)
  if not raw:
    return empty_status()
  if _STATUS_CACHE is not None and _STATUS_CACHE[0] == raw:
    return _STATUS_CACHE[1]
  data = None
  try:
    data = json.loads(raw)
  except (TypeError, ValueError):
    data = None
  result = _status_from(data)
  _STATUS_CACHE = (raw, result)
  return result


def _status_from(data: Any) -> dict:
  """The fixed shape, from whatever the param held. Anything unparsable reads as the empty status:
  one code path for every reader."""
  result = empty_status()
  if not isinstance(data, dict) or data.get("schema") != SCHEMA:
    return result

  catalog = data.get("catalog")
  if isinstance(catalog, dict):
    result["catalog"] = {
      "revision": str(catalog.get("revision", "")),
      "generated_at": str(catalog.get("generated_at", "")),
      "total": int(catalog.get("total", 0) or 0),
      "error": catalog.get("error") if isinstance(catalog.get("error"), str) else None,
    }
  result["job"] = _job_from_dict(data.get("job"))
  installed = data.get("installed")
  if isinstance(installed, list):
    result["installed"] = [entry for entry in (_installed_from_dict(e) for e in installed) if entry is not None]
  storage = data.get("storage")
  if isinstance(storage, dict):
    result["storage"] = {
      "used": int(storage.get("used", 0) or 0),
      "free": int(storage.get("free", 0) or 0),
      "limit": int(storage.get("limit", MAX_STORE_BYTES) or MAX_STORE_BYTES),
    }
  return result


def _installed_from_dict(data: Any) -> dict | None:
  if not isinstance(data, dict):
    return None
  selection = data.get("recipe", "")
  if not isinstance(selection, str) or not selection or not valid_selection(selection):
    return None
  state = data.get("state")
  if state not in ("packaged", "built", "failed"):
    state = "packaged"
  kind = data.get("kind")
  return {
    "recipe": selection,
    "name": str(data.get("name", "")),
    "kind": kind if kind in KINDS else DRIVING,
    "protocol": str(data.get("protocol", "")),
    "roles": [str(r) for r in data.get("roles", [])] if isinstance(data.get("roles"), list) else [],
    "size": int(data.get("size", 0) or 0),
    "state": state,
    "error": data.get("error") if isinstance(data.get("error"), str) else None,
    "provenance": str(data.get("provenance", "")),
  }


def selection_of(entry: dict) -> str:
  """The recipe digest an entry names."""
  return entry.get("recipe") or ""


def publish_status(
  params: Any, *, catalog: dict | None = None, job: Job | None = None, installed: list[dict] | None = None, storage: dict | None = None
) -> None:
  """Write the snapshot the panels render."""
  snapshot = {
    "schema": SCHEMA,
    "catalog": catalog if catalog is not None else empty_status()["catalog"],
    "job": job.to_dict() if job is not None else None,
    "installed": installed if installed is not None else [],
    "storage": storage if storage is not None else storage_status(),
  }
  params.put(STATUS_KEY, json.dumps(snapshot), block=True)


# --- store records -----------------------------------------------------------
def read_json(path: str) -> Any:
  try:
    with open(path, encoding="utf-8") as handle:
      return json.loads(handle.read())
  except (OSError, ValueError):
    return None


def write_json(path: str, value: Any) -> None:
  """Atomic write, overwriting. The worker's only write primitive, so a reader either sees the
  previous file or the whole new one -- never a half-written record."""
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
    return "packaged", None
  if record.get("state") != "built":
    return "failed", record.get("error") if isinstance(record.get("error"), str) else BUILD_FAILED
  if not os.path.isfile(build_model(key)):
    return "failed", BUILD_INCOMPLETE
  return "built", None


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
  return {"recipe": document, "profile": profile if isinstance(profile, dict) else None}


def package_members(recipe: str, document: dict | None = None) -> dict[str, dict]:
  """`{role: {"path", "sha256", "size", "metadata"}}` for a stored package, from its own recipe
  document. Paths are inside the package; nothing here checks the bytes (see `verify_members`)."""
  recipe_document = document if document is not None else (load_package(recipe) or {}).get("recipe")
  members = (recipe_document or {}).get("members")
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
      return NOT_INSTALLED
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
  if not valid_selection(selection):
    return None
  package = load_package(selection)
  if package is None:
    return None
  members = package_members(selection, package["recipe"])
  if not members:
    return None
  proto = protocol_for_roles(members)
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
  proto: Protocol = resolved["protocol"]
  members: dict[str, dict] = resolved["members"]
  key = build_key(selection, proto.id)
  state, error = build_state(key)
  provenance = ""
  record = load_build(key) or {}
  if isinstance(record.get("provenance"), dict):
    provenance = str(record["provenance"].get("version", ""))
  return {
    "recipe": selection,
    "name": entry_name(selection),
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
  if selection and not valid_selection(selection):
    raise ValueError(f"not a selection: {selection!r}")
  params.put(DESIRED_KEY[kind], selection, block=True)


def boot(params: Any) -> dict:
  """The committed boot snapshot. Absent or unparsable reads as `bundled` for both kinds, which is
  what a device that has never committed one must do."""
  raw = params.get(BOOT_KEY)
  result = {
    "schema": SCHEMA,
    DRIVING: {"kind": "bundled", "id": "", "protocol": "", "build": "", "members": {}, "configuration": {}, "requested": "", "fallback": None},
    MONITORING: {"kind": "bundled", "id": "", "protocol": "", "build": "", "members": {}, "configuration": {}, "requested": "", "fallback": None},
  }
  if not raw:
    return result
  try:
    data = json.loads(raw)
  except (TypeError, ValueError):
    return result
  if not isinstance(data, dict) or data.get("schema") != SCHEMA:
    return result
  for kind in KINDS:
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
  configuration = boot(params)[DRIVING].get("configuration") or {}
  try:
    return float(configuration.get(key, fallback))
  except (TypeError, ValueError):
    return float(fallback)


def _boot_entry(entry: dict) -> dict:
  kind = entry.get("kind")
  if kind not in ("bundled", "custom"):
    kind = "bundled"
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


def boot_model(kind: str, entry: dict) -> str:
  """The `model.pkl` path inside a custom boot entry's build."""
  key = os.path.basename(entry.get("build", ""))
  return build_model(key)


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
  for kind in KINDS:
    requested = desired(params, kind)
    entry = {"kind": "bundled", "id": "", "protocol": "", "build": "", "members": {}, "configuration": {}, "requested": requested, "fallback": None}
    if requested:
      entry = _resolve(requested, kind)
    snapshot[kind] = entry
  params.put(BOOT_KEY, json.dumps(snapshot), block=True)
  return boot(params)


def _resolve(selection: str, kind: str) -> dict:
  """One kind's entry: the built model, or `bundled` with the reason it is not."""
  entry = {"kind": "bundled", "id": "", "protocol": "", "build": "", "members": {}, "configuration": {}, "requested": selection, "fallback": None}
  if not valid_selection(selection):
    entry["fallback"] = UNKNOWN_SELECTION
    return entry
  resolved = selection_record(selection)
  if resolved is None:
    entry["fallback"] = NOT_INSTALLED
    return entry
  proto: Protocol = resolved["protocol"]
  if proto.kind != kind:
    entry["fallback"] = WRONG_KIND
    return entry
  members: dict[str, dict] = resolved["members"]
  if frozenset(members) != frozenset(proto.roles):
    entry["fallback"] = NOT_INSTALLED
    return entry
  if (reason := verify_members(members)) is not None:
    entry["fallback"] = reason
    return entry
  key = build_key(selection, proto.id)
  directory = build_dir(selection, proto.id)
  if (load_build(key) or {}).get("protocol") not in (None, proto.id):
    # The key already encodes the protocol, so this is a store that was edited under its own name --
    # caught here so the panel can explain it, and again in the runtime, which cannot see the commit.
    entry["fallback"] = BUILD_MISMATCH
    return entry
  state, error = build_state(key)
  if state != "built":
    entry["fallback"] = error or NOT_BUILT
    return entry
  entry.update(
    {
      "kind": "custom",
      "id": selection,
      "protocol": proto.id,
      "build": directory,
      "members": {role: member["path"] for role, member in members.items()},
      "configuration": resolved["configuration"],
    }
  )
  return entry


def restart_needed(params: Any) -> bool:
  """Whether either kind owes a restart -- the condition the panes' reboot row appears on.

  `commit_boot_selection` runs once per boot, so a selection change is only a *desired* value until
  the device restarts: this is the one place that difference is summarized for the driver.
  """
  return any(restart_pending(params, kind) for kind in KINDS)


def request_reboot(params: Any) -> None:
  """Ask the manager to restart the device, through upstream's own param."""
  params.put_bool(REBOOT_KEY, True, block=True)


def restart_pending(params: Any, kind: str) -> bool:
  """Whether the driver's selection differs from the one this boot committed -- the reason both
  panels say `(restart to apply)`. The row takes a restart because the model processes read the
  snapshot once, at their own start."""
  return desired(params, kind) != boot(params)[kind]["requested"]


# --- what the panels print ---------------------------------------------------
# Both trees say the same thing because the strings live here, like offroad.py's (AGENTS.md,
# Settings organization). Panels choose layout; they never choose wording.
BUNDLED_LABEL = "stock"
RESTART_NOTE = "(restart to apply)"
TITLE_MODELS = "models"
TITLE_CURRENT_MODEL = "current model"
DESCRIPTION_CURRENT_MODEL = "Select the model used for driving. Driver monitoring uses a separate model."
TITLE_BROWSE = "model marketplace"
TITLE_INSTALLED = "downloaded models"
TITLE_STORAGE = "storage"
TITLE_DRIVING = "driving model"
TITLE_MONITORING = "driver monitoring"
TITLE_REFRESH = "refresh catalog"
DESCRIPTION_MODELS = "Choose the model used for driving and the separate model used to watch \
driver. Driving models can change steering and slowing; monitoring models only watch attention."
DESCRIPTION_BROWSE = "Models available to install. Choose one from the model chooser after it is built."
DESCRIPTION_INSTALLED = "Models installed on this device. Manage or remove packages here."
DESCRIPTION_STORAGE = "Space the model store uses on the data partition. Installing needs room for \
the package, the digest cache and the build."
DESCRIPTION_REFRESH = "Fetch the catalog again. Needs a network; a failed refresh keeps the copy \
this device already has."
LABEL_REFRESH = "REFRESH"
LABEL_REBOOT = "REBOOT NOW"
LABEL_INSTALL = "INSTALL"
LABEL_INSTALLED = "INSTALLED"
LABEL_SELECT = "SELECT"
LABEL_DESELECT = "DESELECT"
LABEL_REBUILD = "REBUILD"
LABEL_REMOVE = "REMOVE"
LABEL_CANCEL = "CANCEL"
LABEL_NEWER = "NEWER"
LABEL_OLDER = "OLDER"
FILTER_ALL = "all"
FILTER_DRIVING = "driving"
# Short on purpose: the filter row's buttons are 250 px wide, and "driver monitoring" overflows
# them (the label is drawn centered and clipped). The kind itself is named in full elsewhere.
FILTER_MONITORING = "monitoring"
FILTER_RUNNABLE = "runnable only"
REBOOT_KEY = "DoReboot"  # upstream's own, read by the manager's power path (manager.py:199)
CONFIRM_SELECT = "Use this model?"
CONFIRM_REMOVE = "Remove this model?"
CONFIRM_CANCEL = "Cancel the current model job?"
SELECT_TEXT = "This model has not been tested by comma or moonpilot. Select it, then restart to use \
it."
REMOVE_TEXT = "This deletes the package and its build from the device. The model can be installed \
again from the catalog."
DESCRIPTION_CANCEL = "Canceling stops the current model job. You can start it again later."
DESCRIPTION_REBOOT = "A selected model starts after restart. Restart after installing and selecting."
REASON_SELECTED = "in effect"
REASON_INSTALLED = "installed"
DRIVING_ONLY = "PARKED ONLY"
LABEL_BACK = "BACK"
LABEL_OPEN = "OPEN"
BROWSE_EMPTY = "no catalog yet"
BROWSE_NONE = "nothing matches"
BROWSE_STALE = "refresh the catalog first"
INSTALL_TEXT = "Download and build this model. Then choose it from the model chooser and restart to use it."
# The protocol ids are precise and unreadable; these are what a driver picks between.
PROTOCOL_LABELS = {
  "comma.supercombo.v1": "supercombo",
  "comma.split-vision-policy.v1": "vision + policy",
  "comma.split-vision-off-on.v1": "vision + off + on",
  "comma.dmonitoring.v1": "driver monitoring",
}
OFFROAD_NOTE = "Installation runs while driving; selecting and removing wait until the car is parked \
and openpilot is off."
PHASE_TEXT = {
  "waiting-network": "waiting for network",
  "waiting-offroad": "waiting for offroad",
  "downloading": "downloading",
  "verifying": "verifying",
  "building": "building",
  "done": "done",
  "error": "failed",
  "canceled": "canceled",
}
TOO_LARGE = "too large to build on this device"
NO_SPACE = "not enough free space on the data partition"
STORE_FULL = "the model store is full; remove a model first"
NOT_INSTALLED = "not installed"
NOT_BUILT = "not built yet"
BUILD_FAILED = "the build failed"
BUILD_INCOMPLETE = "the build is incomplete"
BUILD_FAILED_PREFIX = "build failed: "
BUILD_MISMATCH = "the build was made for another protocol"
ARTIFACT_CHANGED = "the installed bytes changed since the build"
UNKNOWN_SELECTION = "not a selection"
WRONG_KIND = "selected for the other kind"
SELECTED_HINT = "switch to another model first"


def protocol_label(protocol_id: str) -> str:
  return PROTOCOL_LABELS.get(protocol_id, protocol_id)


def human_size(size: int) -> str:
  for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
    if size >= scale:
      return f"{size / scale:.1f} {unit}"
  return f"{size} B"


def storage_text(storage: dict) -> str:
  return f"{human_size(storage.get('used', 0))} used, {human_size(storage.get('free', 0))} free"


# The phases that can sit for a while without a fraction to show. `downloading` is not among them:
# it has bytes, and a clock beside them would be noise.
PATIENT_PHASES = ("waiting-network", "waiting-offroad", "verifying", "building")


def clock(seconds: float) -> str:
  """`m:ss`, or `h:mm:ss` past an hour -- the same in both trees, and short enough for a row."""
  total = max(0, int(seconds))
  hours, minutes, secs = total // 3600, (total % 3600) // 60, total % 60
  return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def job_fraction(job: dict | None) -> float | None:
  """How far along a download is, 0..1, or `None` when there is nothing to measure.

  A bar is only honest where a total is known: `building`, `verifying` and the two waiting phases
  have none, and drawing a full or empty bar for them would be a claim about progress that no
  number backs.
  """
  if not job or job.get("phase") != "downloading":
    return None
  total = int(job.get("total") or 0)
  if total <= 0:
    return None
  return min(1.0, max(0.0, int(job.get("received") or 0) / total))


def job_text(job: dict | None) -> str:
  """The one line the panels show for a running job, in the driver's terms."""
  if not job:
    return ""
  phase = PHASE_TEXT.get(str(job.get("phase", "")), str(job.get("phase", "")))
  if job.get("phase") == "downloading" and job.get("total"):
    return f"{phase} {human_size(int(job.get('received', 0) or 0))} of {human_size(int(job['total']))}"
  if job.get("phase") in PATIENT_PHASES and job.get("elapsed") is not None:
    # The one answer a stalled build needs: how long it has been going. The worker refreshes this
    # with the rest of the snapshot, so it keeps counting.
    return f"{phase} {clock(float(job['elapsed']))}"
  return phase


def job_progress_text(job: dict | None) -> str:
  """The label on the bar: the percentage beside the bytes, because a bar shows the shape of the
  wait and only the numbers show the size of it."""
  fraction = job_fraction(job)
  if fraction is None or not job:
    return job_text(job)
  return f"{round(fraction * 100)}%  {human_size(int(job.get('received', 0) or 0))} of {human_size(int(job['total']))}"


def job_description(job: dict | None, params: Any) -> str:
  """The job row's detail: the phase, whether the bytes are metered, and how to stop it.

  Canceling is a *new* request (models.OPS), not a flag: the worker's `canceled()` callback reads
  the request param and stops the moment it holds anything else, so a cancel is the same mechanism
  as changing your mind.
  """
  if not job:
    return ""
  parts = [f"{job.get('op', '')} {str(job.get('recipe', ''))[:12]}".strip(), PHASE_TEXT.get(str(job.get("phase", "")), "")]
  if job.get("metered"):
    parts.append("over a metered connection")
  if job.get("error"):
    parts.append(str(job["error"]))
  if job.get("phase") in ("waiting-network", "waiting-offroad", "downloading", "verifying", "building"):
    parts.append("canceling starts a new request")
  return ". ".join(p for p in parts if p) + "."


def catalog_text(catalog: dict) -> str:
  """The refresh row's value: the snapshot this device holds, and the last refresh's error."""
  if not catalog.get("revision"):
    return "never"
  if catalog.get("error"):
    return "stale"
  return str(catalog.get("generated_at", ""))[:10] or catalog["revision"][:12]


def catalog_description(catalog: dict) -> str:
  parts = [DESCRIPTION_REFRESH]
  if catalog.get("revision"):
    revision, total = str(catalog["revision"])[:12], int(catalog.get("total", 0) or 0)
    parts.append(f"holding {revision}, {total} models, generated {str(catalog.get('generated_at', ''))[:10]}")
  if catalog.get("error"):
    parts.append(f"last refresh failed: {catalog['error']}")
  return " ".join(parts)


def installed_action(entry: dict, params: Any) -> tuple[str, str]:
  """What an installed model's first action row offers: the label, and which way it toggles.
  `select` is per kind, and the desired value is the recipe digest.
  """
  selection = selection_of(entry)
  kind = entry.get("kind", DRIVING)
  if desired(params, kind) == selection:
    return LABEL_DESELECT, "deselect"
  return LABEL_SELECT, "select"


def installed_detail(entry: dict) -> str:
  parts = [f"{entry.get('protocol', '')}", f"roles: {', '.join(entry.get('roles', []))}", human_size(int(entry.get("size", 0))), str(entry.get("state", ""))]
  if entry.get("provenance"):
    parts.append(f"built by {entry['provenance']}")
  if entry.get("error"):
    parts.append(str(entry["error"]))
  return ". ".join(p for p in parts if p) + "."


def model_label(params: Any, kind: str) -> str:
  """The model row's value: the selection in effect, and whether a restart is owed."""
  entry = boot(params)[kind]
  if entry["kind"] == "bundled":
    name = BUNDLED_LABEL
    if entry.get("fallback"):
      name = f"{BUNDLED_LABEL}, {entry['requested'][:12]} refused"
  else:
    active = params.get(ACTIVE_KEY[kind]) or ""
    name = entry_name(entry["id"]) or entry["id"][:12]
    name = f"{BUNDLED_LABEL}, {name} did not load" if active.startswith("stock: ") else f"{name} ({protocol_label(entry['protocol'])})"
  return f"{name} {RESTART_NOTE}" if restart_pending(params, kind) else name


def model_description(params: Any, kind: str) -> str:
  """The model row's description: what is in effect, and the reason when a selection did not run.

  Two reasons can stop a selection, and they are different moments: the boot commit refused it (the
  entry's `fallback`, known before any process started) or the model process failed to load it
  (`MoonpilotModelsActive*`, written by `moonpilot/modelruntime.py` -- the boot commit cannot see
  that one, because it happens later and in another process). Both are shown here, because a driver
  looking at a selection that is not driving the car needs the sentence either way.
  """
  entry = boot(params)[kind]
  if entry.get("fallback"):
    return f"{entry['requested'][:12]} is not in use: {entry['fallback']}."
  if entry["kind"] == "bundled":
    return DESCRIPTION_MODELS
  active = params.get(ACTIVE_KEY[kind]) or ""
  if active.startswith("stock: "):
    return f"{entry['id'][:12]} did not load: {active[len('stock: ') :]}. {DESCRIPTION_MODELS}"
  return f"in effect since this boot, built from {entry['id'][:12]}. {DESCRIPTION_MODELS}"


def entry_name(selection: str) -> str:
  """The catalog's display name for a recipe, from the browse index the worker wrote, and `""`
  when the index does not know it -- neither does a recipe whose catalog entry is gone."""
  for entry in browse().get("entries", []):
    if selection_of(entry) == selection:
      return str(entry.get("name", ""))
  return ""


def browse() -> dict:
  """The worker's display index: `{"revision", "generated_at", "entries"}`. Stdlib JSON only, so a
  panel can read it; entries keep the shape the worker wrote."""
  document = read_json(browse_file())
  if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
    return {"revision": "", "generated_at": "", "entries": []}
  return {
    "revision": str(document.get("revision", "")),
    "generated_at": str(document.get("generated_at", "")),
    "entries": [e for e in document["entries"] if isinstance(e, dict)],
  }


def chooser_entries(params: Any, kind: str) -> list[dict | None]:
  """Stock, built installed entries, then admitted catalog entries for one model kind."""
  installed = status(params)["installed"]
  installed_recipes = {selection_of(entry) for entry in installed}
  entries: list[dict | None] = [None]
  seen = set()
  for entry in installed:
    recipe = selection_of(entry)
    if recipe in seen or entry.get("kind") != kind or entry.get("state") != "built":
      continue
    seen.add(recipe)
    entries.append(entry)
  for entry in browse().get("entries", []):
    recipe = selection_of(entry)
    if not recipe or recipe in installed_recipes or recipe in seen or entry.get("kind") != kind or not entry.get("admitted"):
      continue
    seen.add(recipe)
    entries.append(entry)
  return entries


def entry_action(entry: dict, installed_selections: Collection[str]) -> str:
  """What the catalog row offers: an action for a runnable, uninstalled model, `INSTALLED`, or the
  refusal reason. Dimmed rows are refusals -- the panel grays them and shows this string."""
  selection = selection_of(entry)
  if selection in installed_selections:
    return LABEL_INSTALLED
  if not entry.get("admitted"):
    return str(entry.get("reason") or "not runnable here")
  return LABEL_INSTALL


def entry_detail(entry: dict) -> str:
  """The catalog row's description: roles, size, and the honest limits of a protocol match. The
  catalog's own findings ride along verbatim -- `semantics_unverified`, `requires_runner_check`,
  `qualification` -- because a protocol match proves structures agree and nothing more."""
  parts = [f"roles: {', '.join(entry.get('roles', []))}", f"{human_size(int(entry.get('size', 0)))}", str(entry.get("protocol", ""))]
  if entry.get("updated_at"):
    parts.append(f"catalogued {str(entry['updated_at'])[:10]}")
  if entry.get("notes"):
    parts.extend(str(note) for note in entry["notes"])
  if entry.get("reason"):
    parts.append(str(entry["reason"]))
  return ". ".join(p for p in parts if p) + "."
