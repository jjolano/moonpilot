#!/usr/bin/env python3
"""Install moonpilot's external Python requirements once the device has a network.

The device boots with no network and has no installer (moonpilot/deps.py), so a feature whose
package is absent stays unavailable until something fetches it. That something is this process:
the manager runs it whenever `deps.missing()` is non-empty, it waits for a connection it is happy
and it installs moonpilot/deps.lock into a hash-named release, then atomically promotes
`<data_root>/deps/current`.

Observation only with respect to driving: it writes Python packages under /data and touches
nothing in the control path. Features only pick up what it installs on the next process start.
"""

import time

from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from moonpilot import deps

IDLE_INTERVAL = 300.0  # nothing to do: re-check missing() every 5 min
WAIT_INTERVAL = 30.0   # no usable network yet
BACKOFF_START = 30.0
BACKOFF_MAX = 1800.0   # 30 min


def _missing_names(missing: list[deps.Requirement]) -> str:
  return ", ".join(requirement.module for requirement in missing)


def main() -> None:
  sm = messaging.SubMaster(['deviceState'])
  params = Params()
  backoff = BACKOFF_START

  # Never returns: `ensure_running` does not restart a process that exited while `should_run` is
  # still True (openpilot/system/manager/process.py:142-147,166-171 — `start()` no-ops while
  # `self.proc` is set), so exiting on success or failure would wedge installs until the next
  # boot. The manager stops this process when nothing is missing.
  while True:
    sm.update(0)
    missing = deps.missing()

    if not missing:
      deps.put_status(params, deps.READY, "All required modules are installed.")
      time.sleep(IDLE_INTERVAL)
      continue

    names = _missing_names(missing)
    device_state = sm['deviceState']
    if device_state.networkType == log.DeviceState.NetworkType.none:
      deps.put_status(params, deps.WAITING_NETWORK, f"Waiting for a network to install: {names}.")
      time.sleep(WAIT_INTERVAL)
      continue

    if device_state.networkMetered and not deps.take_retry(params):
      deps.put_status(params, deps.WAITING_METERED, f"Waiting for unmetered network to install: {names}.")
      time.sleep(WAIT_INTERVAL)
      continue

    deps.put_status(params, deps.INSTALLING, f"Installing missing modules: {names}.")
    try:
      deps.install()
      cloudlog.event("moonpilot deps installed", target=deps.site_dir())
      deps.put_status(params, deps.READY, f"Installed modules: {names}.")
      backoff = BACKOFF_START
      time.sleep(WAIT_INTERVAL)
    except Exception as exc:
      detail = str(exc).strip().replace("\n", " ")
      suffix = f": {detail}" if detail else ""
      deps.put_status(params, deps.ERROR, f"Failed to install modules: {names}{suffix}.")
      cloudlog.exception("moonpilot deps install failed")
      time.sleep(backoff)
      backoff = min(backoff * 2, BACKOFF_MAX)


if __name__ == "__main__":
  main()
