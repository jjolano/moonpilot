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

**`admit` and `compose` are the worker's and the panels' shared answer.** Both panels compute a
composition's verdict from stored packages (`models.picks_reason`); `compose` here is the same rule
with catalog documents, which is what the worker would check before recording one. There is no
second rule.
"""

import os
from typing import Any

from moonpilot import models
from moonpilot.vendor.openmodels import client, contracts, metadata

CATALOG_URL = "https://jjolano.github.io/openmodels/catalog.json"
INTERFACE_MISMATCH = "downloaded artifact does not match its recorded interface"
ARTIFACT_UNAVAILABLE = "artifact {availability}: this catalog has no bytes for it"


def load_catalog(path: str | None = None):
  """The snapshot this device holds. No stale-cache fallback: a file that does not validate raises,
  and the caller decides what to do about it (the worker keeps the previous one)."""
  return client.Catalog.load(path or models.catalog_file())


def refresh(url: str = CATALOG_URL, *, timeout: int = 30):
  """Fetch the catalog and store it. Returns `(revision, Catalog)`.

  The bytes are read into memory and written only once they validate, so a fetch that fails halfway
  leaves the previous snapshot exactly as it was -- which is what makes a failed refresh harmless.
  """
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


def browse_index(catalog) -> dict:
  """The display index the panels page: one entry per recipe variant, in the order a driver wants
  them -- newest first -- with the verdict the panel prints."""
  entries = []
  for model in catalog.models(include_archive=True):
    for variant in model["variants"]:
      recipe = variant["recipe"]
      try:
        recipe_object = catalog.resolve(recipe)
      except contracts.ContractError:
        continue
      verdict = admit(catalog, recipe)
      entries.append(
        {
          "recipe": recipe,
          "composition": "",
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
    if set(actual["slices"]) != set(recorded.get("output_slices") or {}):
      return INTERFACE_MISMATCH
  return None


def artifact_bytes(catalog, recipe: str) -> int:
  """What a recipe would put on disk, for the capacity check before any download starts."""
  recipe_object = catalog.resolve(recipe)
  return sum(int(member["artifact"]["size"]) for member in recipe_object.data["members"].values())


def compose(catalog, protocol_id: str, picks: dict[str, str]) -> dict:
  """The composition rule over *catalog* recipes: one pick per role, all admitted to the same
  protocol. `{"protocol", "record", "reason", "notes"}`.

  The rule itself is `models.picks_reason` -- the same function the panels run over stored packages
  and the worker checks a record against -- so this adds only the catalog's own findings to it.
  """
  proto = models.protocol(protocol_id)
  result: dict[str, Any] = {"protocol": protocol_id, "record": None, "reason": None, "notes": []}
  if proto is None:
    result["reason"] = models.UNKNOWN_ROLE
    return result
  if frozenset(picks) != frozenset(proto.roles):
    result["reason"] = f"{models.UNKNOWN_ROLE}: {', '.join(sorted(set(picks) ^ set(proto.roles)))}"
    return result
  if picks and models.picks_are_one_recipe(picks):
    result["reason"] = models.SINGLE_RECIPE
    return result

  dates = _entry_dates(catalog)
  members: dict[str, dict] = {}
  profiles: list[dict] = []
  for role, recipe in picks.items():
    verdict = admit(catalog, recipe)
    if not verdict["admitted"] or verdict["protocol"] != protocol_id:
      result["reason"] = verdict["reason"] or f"{recipe[:12]} is not admitted to {protocol_id}"
      return result
    result["notes"].extend(verdict["notes"])
    recipe_object = catalog.resolve(recipe)
    view = _members_with_dates(catalog, recipe_object, dates).get(role)
    if view is None:
      result["reason"] = f"{recipe[:12]} does not carry the {role} role"
      return result
    view["configuration"] = recipe_object.data.get("configuration") or {}
    members[role] = view
    profiles.append(recipe_object.profile.data)

  reason = models.picks_reason(proto, members, profiles)
  if reason is not None:
    result["reason"] = reason
    return result
  result["record"] = models.composition_record(protocol_id, picks)
  return result
