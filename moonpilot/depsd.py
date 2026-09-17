#!/usr/bin/env python3
"""Install moonpilot's external Python requirements once the device has a network.

The device boots with no network and has no installer (moonpilot/deps.py), so a feature whose
package is absent stays unavailable until something fetches it. That something is this process:
the manager runs it whenever `deps.missing()` is non-empty, it waits for a connection it is happy
to use, and it installs moonpilot/deps.lock into `<data_root>/deps/site-packages`.

Observation only with respect to driving: it writes Python packages under /data and touches
nothing in the control path. Features only pick up what it installs on the next process start.
"""

import time

from openpilot.cereal import log, messaging
from openpilot.common.swaglog import cloudlog

from moonpilot import deps

IDLE_INTERVAL = 300.0  # nothing to do: re-check missing() every 5 min
WAIT_INTERVAL = 30.0   # no usable network yet
BACKOFF_START = 30.0
BACKOFF_MAX = 1800.0   # 30 min


def main() -> None:
  sm = messaging.SubMaster(['deviceState'])
  backoff = BACKOFF_START

  # Never returns: `ensure_running` does not restart a process that exited while `should_run` is
  # still True (openpilot/system/manager/process.py:142-147,166-171 — `start()` no-ops while
  # `self.proc` is set), so exiting on success or failure would wedge installs until the next
  # boot. The manager stops this process when nothing is missing.
  while True:
    sm.update(0)

    if not deps.missing():
      time.sleep(IDLE_INTERVAL)
      continue

    device_state = sm['deviceState']
    if device_state.networkType == log.DeviceState.NetworkType.none:
      time.sleep(WAIT_INTERVAL)
      continue

    # ponytail: metered always defers, so a device that only ever sees cellular never installs.
    # Add updated.py's escape (3 days since the last attempt, or a user-requested fetch) if a
    # dependency ever has to land over cellular.
    if device_state.networkMetered:
      time.sleep(WAIT_INTERVAL)
      continue

    try:
      deps.install()
      cloudlog.event("moonpilot deps installed", target=deps.site_dir())
      backoff = BACKOFF_START
      time.sleep(WAIT_INTERVAL)
    except Exception:
      cloudlog.exception("moonpilot deps install failed")
      time.sleep(backoff)
      backoff = min(backoff * 2, BACKOFF_MAX)


if __name__ == "__main__":
  main()
