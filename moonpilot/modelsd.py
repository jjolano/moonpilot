#!/usr/bin/env python3
"""The model worker: fetch, verify and build what the driver asked for, and publish what it found.

The device has no installer and the panels have no business doing I/O, so both halves of the model
marketplace meet at two params: `MoonpilotModelsRequest` goes in, `MoonpilotModelsStatus` comes out
(moonpilot/models.py). This process is the only writer of either, and it never interprets a request
-- it decodes one op, does it, and publishes.

Four things about it are not obvious.

**It never returns on its own.** The manager's `should_run` predicate is
`request present or status missing`, and its stop is the only exit that leads anywhere:
`ensure_running` does not restart a child that exited while the predicate was still True
(`openpilot/system/manager/process.py:142-147,166-171` -- `start()` no-ops while `self.proc` is
set), so a process that returned after finishing a job would be gone until the next boot. So the
loop idles instead, and finishing a job means publishing the status and *then* clearing the
request, which is what makes the predicate false and lets the manager reap it.

**A rebuild is not automatic.** Nothing here watches for a version bump or a tinygrad change: the
build directory is keyed on (selection, protocol, camera resolutions) only, so a build stays valid
across a fork update, and the panel's rebuild is an `install` of a model that is already installed.
The alternative -- a build that invalidates itself -- would recompile on the road, and the compiler
is not allowed to run onroad (see below).

**The compiler runs offroad only, the download does not.** Fetching is what the driver asked for and
works on any non-`none` network (the status says when the bytes came over cellular); compiling
takes a CPU this car may want for driving, so a job that reaches the build phase while onroad sits
in `waiting-offroad` until the car is parked. That is also why the process is alive but idle, and
why the loop checks for cancellation while it waits.

**A refusal is a sentence, never a silent no-op.** Capacity, a catalog it cannot reach, an artifact
that disagrees with its own recorded interface, a build that fails to compile: each one lands in
`MoonpilotModelsStatus` as `job.error` or as an installed entry's `error`, and the car keeps running
the bundled model (moonpilot/models.py's boot commit).
"""

import json
import os
import shutil
import time

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from moonpilot import models

IDLE_INTERVAL = 1.0  # nothing to do: re-check the store and the request once a second
NETWORK_INTERVAL = 5.0  # the network is unusable right now
PROGRESS_INTERVAL = 1.0  # at most one status publish per second while downloading


class _Canceled(Exception):
  """The driver moved on: a different request, or none. Never a partial package -- the SDK's staging
  directory is removed by its own `finally`."""


class _Refused(Exception):
  """A driver-visible refusal. The string goes into the status; nothing was half-done."""


class Worker:
  def __init__(self) -> None:
    self.params = Params()
    self.sm = messaging.SubMaster(["deviceState"])
    self.catalog = None  # the vendored Catalog, when the store has a validated snapshot
    self.catalog_error: str | None = None
    self.job: models.Job | None = None
    self._published = ""
    self._completed = 0
    self._last_progress = 0.0
    self._load_catalog()

  # --- status ------------------------------------------------------------------------------------
  def _load_catalog(self) -> None:
    """The last validated snapshot, if the store has one. Both the status and an install can be
    served from it without a refresh."""
    from moonpilot import modelcatalog

    if not os.path.isfile(models.catalog_file()):
      return
    try:
      self.catalog = modelcatalog.load_catalog()
    except Exception as exc:
      self.catalog_error = f"stored catalog unusable: {exc}"
      cloudlog.exception("moonpilot models: stored catalog unusable")

  def _catalog_status(self) -> dict:
    index = models.read_json(models.browse_file()) or {}
    return {
      "revision": self.catalog.revision if self.catalog is not None else "",
      "generated_at": self.catalog.data.get("generated_at", "") if self.catalog is not None else "",
      "total": int(index.get("total", 0) or 0) if isinstance(index, dict) else 0,
      "error": self.catalog_error,
    }

  def publish(self, *, force: bool = False) -> None:
    """Write the status, skipping the write when nothing changed: the idle loop runs at 1 Hz and a
    param write per second is a write no reader can see."""
    installed = models.installed_entries()
    storage = models.storage_status()
    catalog = self._catalog_status()
    snapshot = {"catalog": catalog, "job": self.job.to_dict() if self.job is not None else None, "installed": installed, "storage": storage}
    signature = json.dumps(snapshot, sort_keys=True)
    if signature == self._published and not force:
      return
    self._published = signature
    models.publish_status(self.params, catalog=catalog, job=self.job, installed=installed, storage=storage)

  # --- requests ----------------------------------------------------------------------------------
  def run(self) -> None:
    self.publish(force=True)
    while True:
      self.sm.update(0)
      request = models.read_request(self.params)
      if request is None:
        self.publish()
        time.sleep(IDLE_INTERVAL)
        continue
      self.serve(request)

  def serve(self, request: dict) -> None:
    """Serve one request to completion, then hand the process back to the manager by clearing it."""
    job = models.Job(id=request["id"], op=request["op"], selection=request["recipe"], phase="waiting-network")
    job.started_at = time.monotonic()  # published as `elapsed`, so the panels can show a clock
    self.job = job
    self._completed = 0
    self.publish(force=True)
    try:
      if request["op"] == "refresh":
        self._refresh(job)
      elif request["op"] == "install":
        self._install(job)
      elif request["op"] == "remove":
        self._remove(job)
      # "cancel" with nothing running is already done: the request it would cancel is gone.
    except _Canceled:
      job.phase = "canceled"
      cloudlog.info("moonpilot models: canceled")
    except _Refused as exc:
      job.phase = "error"
      job.error = str(exc)
      cloudlog.warning(f"moonpilot models: refused: {exc}")
    except Exception:
      job.phase = "error"
      job.error = "unexpected failure, see the log"
      cloudlog.exception("moonpilot models: job failed")
    self.publish(force=True)
    models.clear_request(self.params)
    self.job = None
    self.publish(force=True)

  def _canceled(self, job: models.Job) -> bool:
    """Has the driver moved on? True when the request is gone, is a different one, or is a cancel.

    The SDK's only abort path is its `DownloadCanceled` exception raised from here, which makes a partial
    package impossible: the staging directory is removed on the way out.
    """
    request = models.read_request(self.params)
    return request is None or request["id"] != job.id or request["op"] == "cancel"

  def _network_usable(self) -> bool:
    state = self.sm["deviceState"]
    return state.networkType != log.DeviceState.NetworkType.none and not state.networkMetered

  # --- ops ---------------------------------------------------------------------------------------
  def _refresh(self, job: models.Job) -> None:
    from moonpilot import deps, features, modelcatalog

    require_signature = features.wanted(features.CATALOG_SIGNATURES, self.params)
    if require_signature and not deps.available("cryptography"):
      raise _Refused("catalog signature requirement cannot be met: cryptography package is not installed")

    # A catalog is 7 MB and not worth a metered connection, so an unusable network holds the
    # request rather than dropping it: the driver asked for this refresh, and it runs the moment
    # there is a connection worth using. The process stays alive while a request exists
    # (`moonpilot/procs.py::_models_wanted`), which is what makes waiting here possible.
    while not self._network_usable():
      job.phase = "waiting-network"
      job.metered = bool(self.sm["deviceState"].networkMetered)
      self.publish(force=True)
      time.sleep(NETWORK_INTERVAL)
      self.sm.update(0)
      if self._canceled(job):
        raise _Canceled
    job.phase = "downloading"
    self.publish(force=True)
    try:
      _revision, catalog = modelcatalog.refresh(require_signature=require_signature)
    except Exception as exc:
      self.catalog_error = str(exc)
      raise _Refused(f"could not reach the catalog: {exc}") from exc
    self.catalog = catalog
    self.catalog_error = None
    modelcatalog.write_browse(catalog)
    job.phase = "done"
    cloudlog.info(f"moonpilot models: catalog {catalog.revision[:12]} refreshed")

  def _install(self, job: models.Job) -> None:
    from moonpilot import modelbuild, modelcatalog

    selection = job.selection
    if self.catalog is None:
      raise _Refused("no catalog on this device yet; refresh it first")
    if models.is_recipe(selection) and models.load_package(selection) is None:
      verdict = modelcatalog.admit(self.catalog, selection)
      if not verdict["admitted"]:
        raise _Refused(str(verdict["reason"]) or "this model cannot run here")
      declared = modelcatalog.artifact_bytes(self.catalog, selection)
      if (reason := models.capacity_reason(declared)) is not None:
        raise _Refused(reason)
      self._fetch(job, self.catalog, selection, declared)
      if (reason := modelcatalog.verify_downloaded(self.catalog, selection)) is not None:
        self._discard(selection)
        raise _Refused(reason)

    resolved = models.selection_record(selection)
    if resolved is None:
      raise _Refused("this model is not installed and could not be resolved")
    if (reason := models.verify_members(resolved["members"])) is not None:
      # A truncated package must not be built from; a fresh install replaces it.
      self._discard(selection)
      raise _Refused(reason)
    proto = resolved["protocol"]
    # A build is a package again: the same copies in flight, and it is larger than its members, so
    # the estimate is their sum (the panel shows the store's own numbers, this only decides).
    build_bytes = sum(member["size"] for member in resolved["members"].values())
    if (reason := models.capacity_reason(build_bytes)) is not None:
      raise _Refused(reason)
    self._build(job, modelbuild, selection, resolved, proto)
    job.phase = "done"

  def _fetch(self, job: models.Job, catalog, selection: str, declared: int) -> None:
    from moonpilot.vendor.openmodels.client import DownloadCancelled, ModelStore  # codespell:ignore cancelled

    job.phase = "downloading"
    job.total = declared
    job.metered = bool(self.sm["deviceState"].networkMetered)
    self.publish(force=True)

    def on_progress(_sha256: str, received: int, total: int) -> None:
      if received >= total:
        self._completed += total
      job.received = self._completed + received
      now = time.monotonic()
      if now - self._last_progress >= PROGRESS_INTERVAL:
        self._last_progress = now
        self.publish(force=True)

    store = ModelStore(models.packages_dir(), catalog)
    try:
      # The abort callback is passed under the SDK's own keyword spelling
      # (`moonpilot/vendor/openmodels/client.py`); elsewhere this tree spells the word with one l.
      store.fetch(catalog.resolve(selection), on_progress=on_progress, cancelled=lambda: self._canceled(job))  # codespell:ignore cancelled
    except DownloadCancelled as exc:  # codespell:ignore cancelled
      raise _Canceled from exc
    job.phase = "verifying"
    self.publish(force=True)

  def _discard(self, selection: str) -> None:
    """A package that failed verification is worse than no package: the boot commit would fall back
    and the panel would show a model that cannot run. Remove it whole."""
    shutil.rmtree(models.package_dir(selection), ignore_errors=True)

  def _build(self, job: models.Job, modelbuild, selection: str, resolved: dict, proto) -> None:
    directory = models.build_dir(selection, proto.id)
    while self.sm["deviceState"].started:
      # The compiler is a whole CPU and the car may want it: a build waits for offroad.
      job.phase = "waiting-offroad"
      self.publish()
      time.sleep(IDLE_INTERVAL)
      self.sm.update(0)
      if self._canceled(job):
        raise _Canceled
    job.phase = "building"
    self.publish(force=True)
    started = time.monotonic()
    try:
      modelbuild.build(
        proto,
        {role: member["path"] for role, member in resolved["members"].items()},
        resolved["configuration"],
        models.CAMERA_RESOLUTIONS,
        directory,
        selection=selection,
      )
    except Exception as exc:
      self._failed_build(modelbuild, directory, str(exc))
      raise _Refused(f"{models.BUILD_FAILED_PREFIX}{exc}") from exc
    cloudlog.info(f"moonpilot models: built {os.path.basename(directory)[:12]} in {time.monotonic() - started:.0f}s")

  def _failed_build(self, modelbuild, directory: str, error: str) -> None:
    """Record the compiler's own message where `models.build_state` and the panel read it, and leave
    no `model.pkl` behind: a build that parses as built while holding garbage is the one thing the
    boot commit must never accept."""
    os.makedirs(directory, exist_ok=True)
    try:
      os.remove(os.path.join(directory, "model.pkl"))
    except OSError:
      pass
    models.write_json(models.build_record(os.path.basename(directory)), modelbuild.failure_record(error))

  def _remove(self, job: models.Job) -> None:
    selection = job.selection
    if not models.valid_selection(selection):
      raise _Refused(models.UNKNOWN_SELECTION)
    # Anything in effect this boot, or wanted for the next one, is off limits: the driver switches
    # first, and the panel says so (models.SELECTED_HINT).
    live = {models.boot(self.params)[kind]["id"] for kind in models.KINDS}
    live |= {models.desired(self.params, kind) for kind in models.KINDS}
    if selection in live:
      raise _Refused(models.SELECTED_HINT)
    resolved = models.selection_record(selection)
    if resolved is not None:
      shutil.rmtree(models.build_dir(selection, resolved["protocol"].id), ignore_errors=True)
    shutil.rmtree(models.package_dir(selection), ignore_errors=True)
    job.phase = "done"


def main() -> None:
  cloudlog.warning("moonpilot modelsd init")
  Worker().run()


if __name__ == "__main__":
  main()
