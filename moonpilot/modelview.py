"""Driver-facing model labels, descriptions, and chooser rows."""

import os
from collections.abc import Sequence
from typing import Any

from moonpilot import modelprotocols, modelstatus, models
# --- what the panels print ---------------------------------------------------
# Both trees say the same thing because the strings live here. Panels choose layout; they never choose wording.
BUNDLED_LABEL = "stock"
RESTART_NOTE = "(restart to apply)"
TITLE_MODELS = "models"
TITLE_CURRENT_MODEL = "current model"
TITLE_SELECT_MODEL = "select a model"
LABEL_SEARCH = "SEARCH"
DESCRIPTION_CURRENT_MODEL = "Select the model used for driving. Driver monitoring uses a separate model."

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
LABEL_REBOOT = "REBOOT NOW"
LABEL_INSTALL = "INSTALL"
LABEL_SELECT = "SELECT"
LABEL_REBUILD = "REBUILD"
LABEL_REMOVE = "REMOVE"
LABEL_CANCEL = "CANCEL"
CONFIRM_SELECT = "Use this model?"
CONFIRM_CANCEL = "Cancel the current model job?"
REMOVE_TEXT = "This deletes the package and its build from the device. The model can be installed \
again from the catalog."
DESCRIPTION_CANCEL = "Canceling stops the current model job. You can start it again later."
DESCRIPTION_REBOOT = "A selected model starts after restart. Restart after installing and selecting."
REASON_SELECTED = "in effect"
CHOOSER_HINT_STOCK = "the bundled model. Applies after a restart."
CHOOSER_HINT_SELECTED = "in effect since this boot."
CHOOSER_HINT_BUILT = "built and ready. Applies after a restart."
CHOOSER_HINT_INSTALL = "not built yet: this downloads and builds it, then applies after a restart."
CHOOSER_HINT_OFFROAD = "available when the device is offroad."
BROWSE_EMPTY = "no catalog yet"
BROWSE_NONE = "nothing matches"
INSTALL_TEXT = "Download and build this model. Then choose it from the model chooser and restart to use it."
PHASE_TEXT = {
  modelstatus.PHASE_WAITING_NETWORK: "waiting for network",
  modelstatus.PHASE_WAITING_OFFROAD: "waiting for offroad",
  modelstatus.PHASE_DOWNLOADING: "downloading",
  modelstatus.PHASE_VERIFYING: "verifying",
  modelstatus.PHASE_BUILDING: "building",
  modelstatus.PHASE_DONE: "done",
  modelstatus.PHASE_ERROR: "failed",
  modelstatus.PHASE_CANCELED: "canceled",
}


def protocol_label(protocol_id: str) -> str:
  proto = modelprotocols.protocol(protocol_id)
  return proto.label if proto is not None else protocol_id


def human_size(size: int) -> str:
  for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("kB", 1024)):
    if size >= scale:
      return f"{size / scale:.1f} {unit}"
  return f"{size} B"


def storage_text(storage: dict) -> str:
  return f"{human_size(storage.get('used', 0))} used, {human_size(storage.get('free', 0))} free"




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
  if not job or job.get("phase") != modelstatus.PHASE_DOWNLOADING:
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
  if job.get("phase") == modelstatus.PHASE_DOWNLOADING and job.get("total"):
    return f"{phase} {human_size(int(job.get('received', 0) or 0))} of {human_size(int(job['total']))}"
  if job.get("phase") in modelstatus.PATIENT_PHASES and job.get("elapsed") is not None:
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


def job_description(job: dict | None) -> str:
  """The job row's detail: the phase, whether the bytes are metered, and how to stop it.

  Canceling is a *new* request (modelstatus.OPS), not a flag: the worker's `canceled()` callback reads
  the request param and stops the moment it holds anything else, so a cancel is the same mechanism
  as changing your mind.
  """
  if not job:
    return ""
  parts = [f"{job.get('op', '')} {models.short(str(job.get('recipe', '')))}".strip(), PHASE_TEXT.get(str(job.get("phase", "")), "")]
  if job.get("metered"):
    parts.append("over a metered connection")
  if job.get("error"):
    parts.append(str(job["error"]))
  if job.get("phase") in modelstatus.ACTIVE_PHASES:
    parts.append("canceling starts a new request")
  return ". ".join(p for p in parts if p) + "."


def catalog_text(catalog: dict) -> str:
  """The refresh row's value: the snapshot this device holds, and the last refresh's error."""
  if not catalog.get("revision"):
    return "never"
  if catalog.get("error"):
    return "stale"
  return str(catalog.get("generated_at", ""))[:10] or models.short(catalog["revision"])


def catalog_description(catalog: dict) -> str:
  parts = [DESCRIPTION_REFRESH]
  if catalog.get("revision"):
    revision, total = models.short(str(catalog["revision"])), int(catalog.get("total", 0) or 0)
    parts.append(f"holding {revision}, {total} models, generated {str(catalog.get('generated_at', ''))[:10]}")
  if catalog.get("error"):
    parts.append(f"last refresh failed: {catalog['error']}")
  return " ".join(parts)


def installed_detail(entry: dict) -> str:
  parts = [f"{entry.get('protocol', '')}", f"roles: {', '.join(entry.get('roles', []))}", human_size(int(entry.get("size", 0))), str(entry.get("state", ""))]
  if entry.get("provenance"):
    parts.append(f"built by {entry['provenance']}")
  if entry.get("error"):
    parts.append(str(entry["error"]))
  return ". ".join(p for p in parts if p) + "."


def model_label(params: Any, kind: str) -> str:
  """The model row's value: the selection in effect, and whether a restart is owed."""
  entry = models.boot(params)[kind]
  if entry["kind"] == modelprotocols.BUNDLED:
    name = BUNDLED_LABEL
    if entry.get("fallback"):
      name = f"{BUNDLED_LABEL}, {selection_name(entry['requested'])} refused"
  else:
    failure = models.active_failure_reason(params.get(models.ACTIVE_KEY[kind]) or "")
    name = selection_name(entry["id"])
    name = f"{BUNDLED_LABEL}, {name} did not load" if failure else f"{name} ({protocol_label(entry['protocol'])})"
  return f"{name} {RESTART_NOTE}" if models.restart_pending(params, kind) else name


def model_description(params: Any, kind: str) -> str:
  """The model row's description: what is in effect, and the reason when a selection did not run.

  Two reasons can stop a selection, and they are different moments: the boot commit refused it (the
  entry's `fallback`, known before any process started) or the model process failed to load it
  (`MoonpilotModelsActive*`, written by `moonpilot/modelruntime.py` -- the boot commit cannot see
  that one, because it happens later and in another process). Both are shown here, because a driver
  looking at a selection that is not driving the car needs the sentence either way.
  """
  entry = models.boot(params)[kind]
  if entry.get("fallback"):
    return f"{selection_name(entry['requested'])} is not in use: {entry['fallback']}."
  if entry["kind"] == modelprotocols.BUNDLED:
    return DESCRIPTION_MODELS
  active = params.get(models.ACTIVE_KEY[kind]) or ""
  if failure := models.active_failure_reason(active):
    return f"{selection_name(entry['id'])} did not load: {failure}. {DESCRIPTION_MODELS}"
  return f"in effect since this boot, built from {selection_name(entry['id'])}. {DESCRIPTION_MODELS}"

def model_page_state(params: Any, status: dict, offroad: bool) -> dict[str, bool]:
  job_present = status["job"] is not None
  busy = modelstatus.job_active(status["job"])
  return {
    "job_present": job_present,
    "cancel": busy,
    "refresh": not busy,
    "reboot": models.restart_needed(params),
    "reboot_enabled": offroad,
  }


def installed_row_state(entry: dict | None, status: dict, offroad: bool) -> dict[str, bool]:
  if entry is None:
    return {"present": False, "remove": False, "rebuild": False, "rebuild_enabled": False}
  available = offroad and not modelstatus.job_active(status["job"])
  return {
    "present": True,
    "remove": available,
    "rebuild": entry.get("state") != models.BUILT,
    "rebuild_enabled": available,
  }



def browse_row(selection: str) -> dict:
  """The index row that names one selection, or `{}`."""
  for entry in browse().get("entries", []):
    if row_selection(entry) == selection or selection in entry.get("variants", ()):
      return entry
  return {}


def entry_name(selection: str) -> str:
  """The catalog's display name for a recipe, or `""` when the index does not know it."""
  return str(browse_row(selection).get("name", ""))

def selection_name(selection: str) -> str:
  return entry_name(selection) or models.short(selection)


def display_name(entry: dict) -> str:
  return str(entry.get("name") or selection_name(row_selection(entry)))

def row_selection(entry: dict | None) -> str:
  return "" if entry is None else entry.get("recipe") or ""

def install(params: Any, selection: str) -> None:
  modelstatus.request(params, modelstatus.OP_INSTALL, selection)


def request_remove(params: Any, entry: dict) -> None:
  modelstatus.request(params, modelstatus.OP_REMOVE, row_selection(entry))


def cancel(params: Any) -> None:
  modelstatus.request(params, modelstatus.OP_CANCEL)





# One slot of cache, like `_STATUS_CACHE`: the pickers re-resolve their rows on every tap and the
# panels re-resolve their row callables every frame, and without this every resolution would open, read
# and re-parse the whole index. The key is the file's stat -- the worker writes through `atomic_write`
# (a rename), so a refresh changes the inode and the panels pick the new index up on their next frame.
# A missing file reads as empty and never serves a stale slot.
_BROWSE_CACHE: tuple[tuple | None, dict] | None = None


def browse() -> dict:
  """The worker's display index: `{"revision", "generated_at", "entries"}`. Stdlib JSON only."""
  global _BROWSE_CACHE
  path = models.browse_file()
  try:
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size, stat.st_ino)
  except OSError:
    key = None
  if _BROWSE_CACHE is not None and _BROWSE_CACHE[0] == key:
    return _BROWSE_CACHE[1]
  document = models.read_json(path)
  if not isinstance(document, dict) or not isinstance(document.get("entries"), list):
    result = {"revision": "", "generated_at": "", "entries": []}
  else:
    result = {
      "revision": str(document.get("revision", "")),
      "generated_at": str(document.get("generated_at", "")),
      "entries": [e for e in document["entries"] if isinstance(e, dict)],
    }
  _BROWSE_CACHE = (key, result)
  return result


def chooser_entries(params: Any, kind: str) -> list[dict | None]:
  """Stock, built installed entries, then admitted catalog entries for one model kind.

  An installed entry is merged over its index row (`{**row, **entry}`) so a model the driver has
  already built carries the same folder, short name and model class as the catalog row beside it --
  without that, the same model would land in two different groups depending on which list found it.
  """
  installed = modelstatus.status(params)["installed"]
  installed_recipes = {row_selection(entry) for entry in installed}
  entries: list[dict | None] = [None]
  seen = set()
  for entry in installed:
    recipe = row_selection(entry)
    if recipe in seen or entry.get("kind") != kind or entry.get("state") != models.BUILT:
      continue
    seen.add(recipe)
    row = browse_row(recipe)
    entries.append({**row, **entry, "name": entry.get("name") or row.get("name", "")})
  for entry in admitted_entries(kind):
    recipe = row_selection(entry)
    if not recipe or recipe in installed_recipes or recipe in seen:
      continue
    seen.add(recipe)
    entries.append(entry)
  return entries


# --- what the pickers group and search -----------------------------------------
# Grouping uses the catalog fields emitted by `modelcatalog.browse_index`.
GROUP_STOCK = ""  # no header: the bundled model leads the picker, as it leads sunnypilot's
GROUP_FAVORITES = "favorites"
KIND_TITLES = {modelprotocols.DRIVING: "Driving", modelprotocols.MONITORING: "Driver monitoring"}


def is_built(params: Any, kind: str, selection: str) -> bool:
  """Whether a selection can be chosen right now: the bundled model always can, and a recipe needs
  a build of its own kind in the store. Whether that build may *run* is the boot commit's question."""
  if not selection:
    return True
  return any(entry.get("kind") == kind and entry.get("state") == models.BUILT and row_selection(entry) == selection
             for entry in modelstatus.status(params)["installed"])

CHOICE_INSTALL = "install"
CHOICE_SELECT = "select"
CHOICE_CURRENT = "current"
CHOICE_BLOCKED = "blocked"

def choose_policy(selection: str, kind: str, params: Any, offroad: bool, busy: bool) -> str:
  if busy:
    return CHOICE_BLOCKED
  if selection and not is_built(params, kind, selection):
    return CHOICE_INSTALL
  if selection == models.desired(params, kind):
    return CHOICE_CURRENT
  return CHOICE_SELECT if offroad else CHOICE_BLOCKED



def entry_group(entry: dict | None) -> str:
  """The group one chooser row belongs under. `None` is the bundled model.

  Whichever parts the row does carry are the group, so an installed model whose catalog entry is
  gone still reads as the protocol it needs rather than as a row of unknowns."""
  if entry is None:
    return GROUP_STOCK
  folder = str(entry.get("folder", ""))
  if folder:
    return folder
  parts = [part for part in (protocol_label(str(entry.get("protocol", ""))), str(entry.get("model_class") or "")) if part]
  return " · ".join(parts) or KIND_TITLES.get(str(entry.get("kind", "")), "models")


def favorites(params: Any) -> set[str]:
  """The driver's starred recipes. A stale digest -- one the catalog no longer lists -- matches no
  row and is rewritten on the next change, so a star can outlive its model without cost."""
  value = params.get(models.FAVS_KEY, return_default=True) or ""
  return {part for part in str(value).split(";") if models.is_recipe(part)}


def set_favorite(params: Any, selection: str, starred: bool) -> None:
  """Star or unstar one recipe. The bundled model is not a recipe and cannot be starred."""
  if not models.is_recipe(selection):
    return
  current = favorites(params)
  current.add(selection) if starred else current.discard(selection)
  params.put(models.FAVS_KEY, ";".join(sorted(current)), block=True)

def toggle_favorite(params: Any, selection: str) -> None:
  set_favorite(params, selection, selection not in favorites(params))


def chooser_groups(params: Any, kind: str) -> list[tuple[str, list[dict | None]]]:
  """`[(group, rows)]` for one kind: the bundled model first, starred models duplicated into
  Favorites, then the rest newest group first. Inside a group the chooser's own order stands, so an
  installed model leads its group and the catalog follows newest first.

  Group order is the order the rows were first seen in, which is `chooser_entries`' order -- and that
  is the catalog's own newest-first order, so the groups come out newest first without a second sort.
  """
  entries = chooser_entries(params, kind)
  starred = favorites(params)
  order: list[str] = []
  buckets: dict[str, list[dict | None]] = {}

  def add(group: str, entry: dict | None) -> None:
    if group not in buckets:
      buckets[group] = []
      order.append(group)
    buckets[group].append(entry)

  for entry in entries:
    selection = row_selection(entry)
    group = entry_group(entry)
    add(group, entry)
    if selection in starred and group != GROUP_FAVORITES:
      add(GROUP_FAVORITES, entry)
  leading = [group for group in (GROUP_STOCK, GROUP_FAVORITES) if group in buckets]
  return [(group, buckets[group]) for group in leading + [group for group in order if group not in leading]]


def entry_display(entry: dict | None, kind: str) -> str:
  """One row's name. The catalog owns it; the only edit is dropping the kind prefix a generated name
  repeats, because the picker is already scoped to one kind."""
  if entry is None:
    return BUNDLED_LABEL
  name = str(entry.get("name", ""))
  prefix = f"{KIND_TITLES.get(kind, '')} · "
  if entry.get("name_kind") == "generated" and name.startswith(prefix):
    name = name[len(prefix):]
  return name or selection_name(row_selection(entry))


def _subsequence(query: str, text: str) -> bool:
  """`query` in `text`, casefolded, as substring or as a subsequence -- so `wsplit` finds
  `WMI V12`. The iterator is the trick: `in` on an iterator consumes up to the match."""
  query, text = query.casefold().strip(), text.casefold()
  if not query:
    return True
  if query in text:
    return True
  rest = iter(text)
  return all(character in rest for character in query)


def matches_query(entry: dict | None, query: str, kind: str = modelprotocols.DRIVING) -> bool:
  """Whether one row matches the picker's search: its short code, its name, its group or its date."""
  if not query.strip():
    return True
  if entry is None:
    return _subsequence(query, BUNDLED_LABEL)
  haystacks = (str(entry.get("short_name", "")), entry_display(entry, kind), entry_group(entry), str(entry.get("updated_at", "")))
  return any(_subsequence(query, text) for text in haystacks)


def search_groups(groups: Sequence[tuple[str, list]], query: str, kind: str = modelprotocols.DRIVING) -> list[tuple[str, list]]:
  """The groups with their matching rows, and no group left empty. An empty query is the list
  unchanged -- searching never hides a group that has rows."""
  if not query.strip():
    return list(groups)
  result = []
  for label, rows in groups:
    kept = [row for row in rows if matches_query(row, query, kind)]
    if kept:
      result.append((label, kept))
  return result


def chooser_hint(entry: dict | None, kind: str, params: Any, offroad: bool) -> str:
  """The picker's one-line footer: what choosing the highlighted row does. A tree has no room for a
  verdict on every row, so the row under the driver's finger carries the sentence instead."""
  action = choose_policy(
    row_selection(entry), kind, params, offroad,
    modelstatus.job_active(modelstatus.status(params)["job"]),
  )
  if entry is None:
    return CHOOSER_HINT_STOCK
  if action == CHOICE_CURRENT:
    return CHOOSER_HINT_SELECTED
  if action == CHOICE_INSTALL:
    return CHOOSER_HINT_INSTALL
  if action == CHOICE_BLOCKED:
    return f"{CHOOSER_HINT_BUILT} {CHOOSER_HINT_OFFROAD}" if not offroad else CHOOSER_HINT_BUILT
  return CHOOSER_HINT_BUILT


def admitted_entries(kind: str) -> list[dict]:
  return [entry for entry in browse()["entries"] if entry.get("admitted") and entry.get("kind") == kind]




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
