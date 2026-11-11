"""Request, job and status protocol for the model worker and panels."""

import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any

from moonpilot import modelprotocols, models
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
  return sum(_tree_bytes(d) for d in (models.packages_dir(), models.builds_dir()))


def storage_status() -> dict:
  """The `storage` member of the status snapshot: used, free, limit -- all in bytes."""
  return {"used": store_usage(), "free": free_bytes(), "limit": MAX_STORE_BYTES}


def free_bytes() -> int:
  try:
    return int(shutil.disk_usage(models.models_root()).free)
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
TOO_LARGE = "too large to build on this device"
NO_SPACE = "not enough free space on the data partition"
STORE_FULL = "the model store is full; remove a model first"

# --- request and status codecs -----------------------------------------------
# One JSON op in, one JSON snapshot out, both on params: the panels never touch the store, and the
# worker never touches a widget. A malformed request reads as None -- dropped, never retried,
# because a retry of something unparsable is a loop.
OP_REFRESH = "refresh"
OP_INSTALL = "install"
OP_REMOVE = "remove"
OP_CANCEL = "cancel"
OPS = (OP_REFRESH, OP_INSTALL, OP_REMOVE, OP_CANCEL)
PHASE_WAITING_NETWORK = "waiting-network"
PHASE_WAITING_OFFROAD = "waiting-offroad"
PHASE_DOWNLOADING = "downloading"
PHASE_VERIFYING = "verifying"
PHASE_BUILDING = "building"
PHASE_DONE = "done"
PHASE_ERROR = "error"
PHASE_CANCELED = "canceled"
PHASES = (
  PHASE_WAITING_NETWORK, PHASE_WAITING_OFFROAD, PHASE_DOWNLOADING, PHASE_VERIFYING, PHASE_BUILDING,
  PHASE_DONE, PHASE_ERROR, PHASE_CANCELED,
)
ACTIVE_PHASES = tuple(phase for phase in PHASES if phase not in (PHASE_DONE, PHASE_ERROR, PHASE_CANCELED))
PATIENT_PHASES = tuple(phase for phase in ACTIVE_PHASES if phase != PHASE_DOWNLOADING)


def job_active(job: dict | None) -> bool:
  """Whether a published job can still be canceled."""
  return bool(job) and job.get("phase") in ACTIVE_PHASES


def request(params: Any, op: str, selection: str = "") -> str | None:
  """Write a request and return its id. The request id lets the worker detect cancellation through
  `Worker._canceled`."""
  if op != OP_CANCEL and read_request(params) is not None:
    return None
  request_id = str(time.monotonic_ns())
  params.put(models.REQUEST_KEY, json.dumps({"schema": models.SCHEMA, "id": request_id, "op": op, "recipe": selection}), block=True)
  return request_id


def read_request(params: Any) -> dict | None:
  raw = params.get(models.REQUEST_KEY)
  if not raw:
    return None
  try:
    data = json.loads(raw)
  except (TypeError, ValueError):
    return None
  if not isinstance(data, dict) or data.get("schema") != models.SCHEMA:
    return None
  if data.get("op") not in OPS or not isinstance(data.get("id"), str) or not data["id"]:
    return None
  selection = data.get("recipe", "")
  if not isinstance(selection, str) or (selection and not models.is_recipe(selection)):
    return None
  return {"schema": models.SCHEMA, "id": data["id"], "op": data["op"], "recipe": selection}


def clear_request(params: Any, request_id: str | None = None) -> None:
  current = read_request(params)
  if request_id is None or current is None or current["id"] == request_id:
    params.remove(models.REQUEST_KEY)


@dataclass
class Job:
  """The worker's own view of the request it is serving. `PHASES` is the closed set the panels
  switch on, so an unknown phase is a bug and not a rendering case."""

  id: str
  op: str
  selection: str = ""
  phase: str = PHASE_WAITING_NETWORK
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
  """The status every reader gets when nothing has been published: one fixed shape."""
  return {
    "schema": models.SCHEMA,
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

  raw = params.get(models.STATUS_KEY)
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


def worker_wanted(params: Any) -> bool:
  return params.get(models.REQUEST_KEY) is not None or params.get(models.STATUS_KEY) is None


def _status_from(data: Any) -> dict:

  result = empty_status()
  if not isinstance(data, dict) or data.get("schema") != models.SCHEMA:
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
  if not isinstance(selection, str) or not selection or not models.is_recipe(selection):
    return None
  state = data.get("state")
  if state not in models.MODEL_STATES:
    state = models.PACKAGED
  kind = data.get("kind")
  return {
    "recipe": selection,
    "name": str(data.get("name", "")),
    "kind": kind if kind in modelprotocols.KINDS else modelprotocols.DRIVING,
    "protocol": str(data.get("protocol", "")),
    "roles": [str(r) for r in data.get("roles", [])] if isinstance(data.get("roles"), list) else [],
    "size": int(data.get("size", 0) or 0),
    "state": state,
    "error": data.get("error") if isinstance(data.get("error"), str) else None,
    "provenance": str(data.get("provenance", "")),
  }

def publish_status(
  params: Any, *, catalog: dict | None = None, job: Job | None = None, installed: list[dict] | None = None, storage: dict | None = None
) -> None:
  """Write the snapshot the panels render."""
  snapshot = {
    "schema": models.SCHEMA,
    "catalog": catalog if catalog is not None else empty_status()["catalog"],
    "job": job.to_dict() if job is not None else None,
    "installed": installed if installed is not None else [],
    "storage": storage if storage is not None else storage_status(),
  }
  params.put(models.STATUS_KEY, json.dumps(snapshot), block=True)


