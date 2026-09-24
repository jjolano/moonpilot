"""The catalog side of the model marketplace: read it, decide what this device can run.

`moonpilot/models.py` holds the rules about a *stored* model; this module holds the rules about a
*catalog* one, and it is the only importer of the vendored SDK (`moonpilot/vendor/openmodels/`).
Three things are worth knowing.

**The catalog is a feed, not an authority.** It publishes bytes and a structural identity and owns
no activation, scheduling or qualification, so a recipe is only "selectable" here because this
module says the fork can execute it: `admit` checks the profile's role set against the frozen
protocol table, the declared inputs and outputs against what the consumers in this tree read, the
artifact formats and targets, and the action head's era. A recipe that no protocol admits is
browsable and not selectable, with the reason the panel prints.

**Downloaded bytes are never trusted.** `verify_onnx` re-derives an interface from the file with
`moonpilot/vendor/openmodels/metadata.py` -- a dependency-free reader that refuses anything but
`slice` in the embedded pickle -- and `verify_downloaded` compares that with what the recipe
recorded, so the artifact a build compiles is the artifact the admission decision was made about.

`admit` is the worker's and the panels' shared answer: it decides whether a catalog recipe is
runnable here and carries the reason and technical notes the panels show.
"""

import os
from typing import Any

from moonpilot import models
from moonpilot.vendor.openmodels import client, contracts, metadata

CATALOG_PUBKEY: str = ""
SIGNATURE_SUFFIX = ".sig"


def verify_signature(payload: bytes, signature: bytes, pubkey_pem: str | None = None) -> None:
  """Verify a catalog sidecar with the pinned Ed25519 key."""
  key = CATALOG_PUBKEY if pubkey_pem is None else pubkey_pem
  if not key:
    raise ValueError("catalog signature requirement cannot be met: no public key is pinned")
  try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
  except ImportError as exc:
    raise ValueError("catalog signature verification cannot run: cryptography is not installed") from exc
  try:
    public_key = serialization.load_pem_public_key(key.encode("ascii"))
    if not isinstance(public_key, Ed25519PublicKey):
      raise ValueError("not an Ed25519 public key")
    public_key.verify(signature, payload)
  except Exception as exc:
    raise ValueError("catalog signature verification failed: invalid signature or public key") from exc


CATALOG_URL = "https://jjolano.github.io/openmodels/catalog.json"
INTERFACE_MISMATCH = "downloaded artifact does not match its recorded interface"
ARTIFACT_UNAVAILABLE = "artifact {availability}: this catalog has no bytes for it"


def load_catalog(path: str | None = None):
  """The snapshot this device holds. No stale-cache fallback: a file that does not validate raises,
  and the caller decides what to do about it (the worker keeps the previous one)."""
  return client.Catalog.load(path or models.catalog_file())


def refresh(url: str = CATALOG_URL, *, timeout: int = 30, require_signature: bool = False):
  """Fetch the catalog and store it. Returns `(revision, Catalog)`.

  The bytes are read into memory and written only once they validate, so a fetch that fails halfway
  leaves the previous snapshot exactly as it was -- which is what makes a failed refresh harmless.
  """
  if require_signature:
    signature = client.read_url(url + SIGNATURE_SUFFIX, client.MAX_SNAPSHOT, timeout=timeout)
    if not CATALOG_PUBKEY:
      raise ValueError("catalog signature requirement cannot be met: no public key is pinned")
    raw = client.read_url(url, client.MAX_SNAPSHOT, timeout=timeout)
    verify_signature(raw, signature)
  else:
    raw = client.read_url(url, client.MAX_SNAPSHOT, timeout=timeout)
  catalog = client.Catalog(raw, base_url=url)
  from openpilot.common.utils import atomic_write

  os.makedirs(os.path.dirname(models.catalog_file()), exist_ok=True)
  with atomic_write(models.catalog_file(), "wb", overwrite=True) as handle:
    handle.write(raw)
  return catalog.revision, catalog


def _entry_dates(catalog) -> dict[str, str]:
  """The newest date the catalog recorded for each recipe, which is the only publication evidence a
  snapshot carries -- `admit` needs it for the action head's era."""
  dates: dict[str, str] = {}
  for entry in catalog.data["entries"]:
    date = max((str(o.get("date", "")) for o in entry["occurrences"] if o.get("date")), default="")
    if date > dates.get(entry["recipe"], ""):
      dates[entry["recipe"]] = date
  return dates


def _members_with_dates(catalog, recipe_object, dates: dict[str, str]) -> dict[str, dict]:
  """`{role: member_view}` for one recipe, with `published_at` added so `models.member_reason` can
  check the action head's era."""
  views = {}
  for role, member in recipe_object.data["members"].items():
    view = models.member_view(member)
    view["published_at"] = dates.get(recipe_object.id, "")
    view["availability"] = catalog._data["locations"].get(member["artifact"]["sha256"], {}).get("availability", "unknown")
    views[role] = view
  return views


def admit(catalog, recipe: str) -> dict:
  """Whether this fork can run `recipe`, and why not when it cannot.

  Returns `{"protocol", "admitted", "reason", "notes"}`. The order matters: the catalog's own
  findings first (they are about the document), then the protocol table (about the device), then the
  members. `notes` carries the findings that are neither refusals nor endorsements -- the honest
  statement of what a protocol match does and does not prove.
  """
  result: dict[str, Any] = {"protocol": "", "admitted": False, "reason": "", "notes": []}
  try:
    recipe_object = catalog.resolve(recipe)
  except contracts.ContractError as exc:
    result["reason"] = str(exc)
    return result

  notes = []
  try:
    findings = recipe_object.check(target={"backend": "QCOM", "hardware": "comma3x", "os": "moonpilot", "runtime": "local-build", "options": {}})
  except contracts.ContractError as exc:
    result["reason"] = str(exc)
    return result
  for finding in findings["findings"]:
    code = finding.get("code")
    if code in ("configuration_unresolved", "target_unsupported"):
      result["reason"] = f"{code}: {finding.get('key', finding.get('connection', ''))}".strip(": ")
      return result
    if code == "structure_unknown":
      # A missing connection port on a *multi-member* recipe is a set that cannot be wired; on a
      # single-member recipe it is a profile edge that does not apply to what would run.
      if len(recipe_object.data["members"]) > 1:
        result["reason"] = f"structure_unknown: {finding.get('connection')}"
        return result
      notes.append(f"structure_unknown: {finding.get('connection')}")
    elif code == "semantics_unverified":
      notes.append(f"semantics_unverified: {finding.get('connection')}")
  notes.append(f"requires_runner_check: {findings['execution_support']}")
  notes.append(f"qualification: {findings['qualification']}")
  result["notes"] = notes

  dates = _entry_dates(catalog)
  members = _members_with_dates(catalog, recipe_object, dates)
  proto = models.protocol_for_roles(members)
  if proto is None:
    result["reason"] = f"{models.UNKNOWN_ROLE}: {', '.join(sorted(members))}"
    return result
  result["protocol"] = proto.id

  for member in members.values():
    if member.get("availability") != "available":
      result["reason"] = ARTIFACT_UNAVAILABLE.format(availability=member.get("availability", "unknown"))
      return result
    if member["size"] > models.MAX_ARTIFACT_BYTES:
      result["reason"] = models.TOO_LARGE
      return result
  if (reason := models.set_reason(proto, members)) is not None:
    result["reason"] = reason
    return result

  result["admitted"] = True
  return result


def _model_row(candidates: list[dict]) -> dict | None:
  """The one variant a model group shows: an admitted recipe first, then the newest.

  Preferring admit over recency is deliberate -- a newer refused variant of the same model is not
  a second thing to pick, it is the same model the driver cannot run.
  """
  if not candidates:
    return None
  return max(candidates, key=lambda e: (e["admitted"], e["updated_at"], e["name"], e["recipe"]))


def browse_index(catalog) -> dict:
  """The display index the panels page: one entry per *model group* -- the admitted variant when
  any of its recipes admit, else the newest -- sorted newest first, with the verdict the panel
  prints and, in `variants`, every recipe of the group so a variant the row does not show still has
  its name. The catalog groups recipe variants under one model id; a flat per-recipe list showed the
  same name many times and put every refused twin in front of a driver who can only pick one."""
  entries = []
  for model in catalog.models(include_archive=True):
    candidates = []
    for variant in model["variants"]:
      recipe = variant["recipe"]
      try:
        recipe_object = catalog.resolve(recipe)
      except contracts.ContractError:
        continue
      verdict = admit(catalog, recipe)
      candidates.append(
        {
          "recipe": recipe,
          "name": str(model.get("name", "")),
          "kind": str(model.get("kind", "")),
          "family": str(model.get("family", "")),
          "model_class": str(model.get("model_class", "")),
          "archived": bool(model.get("archived", False)),
          "roles": sorted(recipe_object.data["members"]),
          "size": int(variant.get("size", 0)),
          "targets": sorted(str(target) for target in variant.get("targets", [])),
          "available": bool(variant.get("available", False)),
          "updated_at": str(variant.get("updated_at", "")),
          "protocol": verdict["protocol"],
          "admitted": verdict["admitted"],
          "reason": verdict["reason"],
          "notes": verdict["notes"],
        }
      )
    if (row := _model_row(candidates)) is not None:
      row["variants"] = sorted(c["recipe"] for c in candidates)
      entries.append(row)
  entries.sort(key=lambda e: (e["updated_at"], e["name"], e["recipe"]), reverse=True)
  return {"schema": models.SCHEMA, "revision": catalog.revision, "generated_at": catalog.data["generated_at"], "total": len(entries), "entries": entries}


def write_browse(catalog) -> dict:
  index = browse_index(catalog)
  models.write_json(models.browse_file(), index)
  return index


def verify_onnx(path: str) -> dict:
  """The interface a downloaded ONNX actually declares, read from its bytes.

  Structural only, and deliberately dependency-free: nothing here executes anything in the file, and
  the embedded `output_slices` pickle is decoded by an unpickler that can construct nothing but
  `slice`. This is the only safe way to read a downloaded artifact -- and never `pickle.load` one."""
  record = metadata.parse(path)
  if record.get("output_slices_error") is not None:
    raise ValueError(f"{os.path.basename(path)}: {record['output_slices_error']}")
  if record.get("output_slices") is None:
    raise ValueError(f"{os.path.basename(path)}: no output_slices in metadata")
  return {"input_shapes": record["input_shapes"], "slices": record["output_slices"], "output_shapes": record.get("output_shapes") or {}}


def _canonical_slice_bounds(value: Any) -> tuple[int | None, int | None, int | None] | None:
  if not isinstance(value, (list, tuple)) or len(value) != 3 or any(part is not None and type(part) is not int for part in value):
    return None
  start, stop, step = value
  return (None if start is None else int(start), None if stop is None else int(stop), None if step is None else int(step))


def verify_downloaded(catalog, recipe: str) -> str | None:
  """Re-derive every member's interface from the bytes on disk and compare it with what the recipe
  recorded. `None` when they agree; the mismatch reason otherwise.

  A file that disagrees is a file the admission decision was not made about, so it is removed rather
  than built from (`moonpilot/modelsd.py`).
  """
  package = models.load_package(recipe)
  if package is None:
    return models.NOT_INSTALLED
  stored = models.package_members(recipe, package["recipe"])
  for role, member in package["recipe"]["members"].items():
    path = (stored.get(role) or {}).get("path")
    recorded = member.get("metadata") or {}
    if not path or not os.path.isfile(path):
      return INTERFACE_MISMATCH
    try:
      actual = verify_onnx(path)
    except (ValueError, OSError):
      return INTERFACE_MISMATCH
    if set(actual["input_shapes"]) != set(recorded.get("input_shapes") or {}):
      return INTERFACE_MISMATCH
    for name, shape in (recorded.get("input_shapes") or {}).items():
      if list(actual["input_shapes"].get(name) or []) != list(shape):
        return INTERFACE_MISMATCH
    recorded_slices = recorded.get("output_slices") or {}
    if not isinstance(recorded_slices, dict) or set(actual["slices"]) != set(recorded_slices):
      return INTERFACE_MISMATCH
    for name, bounds in recorded_slices.items():
      actual_bounds = _canonical_slice_bounds(actual["slices"].get(name))
      if actual_bounds is None or actual_bounds != _canonical_slice_bounds(bounds):
        return INTERFACE_MISMATCH
  return None


def artifact_bytes(catalog, recipe: str) -> int:
  """What a recipe would put on disk, for the capacity check before any download starts."""
  recipe_object = catalog.resolve(recipe)
  return sum(int(member["artifact"]["size"]) for member in recipe_object.data["members"].values())
