#!/usr/bin/env python3
"""Run tailscaled, drive its login, and publish MoonpilotTailscaleStatus for the UI.

The manager starts this process when the `MoonpilotTailscale` toggle is on — offroad and onroad,
since remote access is wanted while driving — and stops it when the toggle goes off. `main()`
never returns: `ensure_running` does not restart a process that exited while `should_run` was
still True (`openpilot/system/manager/process.py:142-147,166-171` — `start()` no-ops while
`self.proc` is set), so exiting on success or failure would wedge the feature until the next boot.
Same shape and same reason as moonpilot/depsd.py.

Installing, upgrading and supervising all live here rather than in a second process: the daemon is
the thing that has to be restarted after an upgrade, and the status param is the thing the panels
read.
"""

import subprocess
import time

from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from moonpilot import tailscale

POLL = 10.0  # status poll while the daemon is up
WAIT = 30.0  # no network yet
CHECK_INTERVAL = 6 * 3600  # stable-track version-check cadence
BACKOFF_START = 30.0
BACKOFF_MAX = 1800.0  # 30 min
UP_RETRY = 30.0  # do not respawn `tailscale up` faster than this


def _stop(proc: subprocess.Popen | None) -> None:
  """Kill a child, hard if it will not go. `sudo` relays the signal to the command it runs."""
  if proc is None:
    return

  proc.terminate()
  try:
    proc.wait(timeout=3)
  except subprocess.TimeoutExpired:
    # Match on our own `--socket` path, never the binary path: binaries() prefers a pair the OS
    # ships, and `pkill -f <tailscaled path>` would take a distro's daemon down with ours. The
    # socket is in daemon_args()' argv, so this still catches the orphaned root child — the case
    # this fallback exists for — and matches nothing else.
    subprocess.run([*tailscale.sudo(), "pkill", "-9", "-f", tailscale.socket_path()], check=False)


def _query_status() -> str:
  """stdout of `tailscale status --json`, or "" when the daemon cannot answer yet."""
  try:
    result = subprocess.run(tailscale.cli_args("status", "--json"), capture_output=True, text=True, timeout=10)
  except (subprocess.TimeoutExpired, OSError):
    return ""
  return result.stdout if result.returncode == 0 else ""


def main() -> None:
  params = Params()
  sm = messaging.SubMaster(['deviceState'])
  daemon: subprocess.Popen | None = None
  login: subprocess.Popen | None = None
  last_up = 0.0
  last_check = -CHECK_INTERVAL  # first offroad+online tick checks immediately; gate is then "6 h between checks"
  backoff = BACKOFF_START
  autoupdate_disabled = False

  try:
    while True:
      sm.update(0)
      online = sm['deviceState'].networkType != log.DeviceState.NetworkType.none

      if tailscale.binaries() is None:
        if not online:
          tailscale.put_status(params, tailscale.OFFLINE)
          time.sleep(WAIT)
          continue

        # Unlike depsd, a metered connection does not defer this install: the driver asked for
        # remote access by flipping the toggle, and an LTE-only device is the one that wants it.
        tailscale.put_status(params, tailscale.INSTALLING)
        try:
          latest = tailscale.latest_release()
          tailscale.install(*latest)
          cloudlog.event("moonpilot tailscale installed", version=latest[0])
          backoff = BACKOFF_START
        except Exception:
          cloudlog.exception("moonpilot tailscale install failed")
          tailscale.put_status(params, tailscale.ERROR, "download failed")
          time.sleep(backoff)
          backoff = min(backoff * 2, BACKOFF_MAX)
      elif online and params.get_bool("IsOffroad") and time.monotonic() - last_check >= CHECK_INTERVAL:
        # Offroad only: this is ~35 MB down and a daemon restart, neither of which belongs
        # mid-drive. The old binary keeps serving until the new one is on disk. The latest
        # fetch happens first so a network/parse failure leaves status and daemon untouched.
        try:
          latest = tailscale.latest_release()
          last_check = time.monotonic()
        except Exception:
          cloudlog.exception("moonpilot tailscale version check failed")
          time.sleep(backoff)
          backoff = min(backoff * 2, BACKOFF_MAX)
          continue
        if tailscale.upgrade_available(latest[0]):
          was = tailscale.marker_version()
          tailscale.put_status(params, tailscale.INSTALLING)
          try:
            tailscale.install(*latest)
            cloudlog.event("moonpilot tailscale upgraded", was=was, now=latest[0])
            backoff = BACKOFF_START
            _stop(daemon)
            daemon = None  # os.replace leaves a running process on the old inode
          except Exception:
            cloudlog.exception("moonpilot tailscale upgrade failed")
            time.sleep(backoff)
            backoff = min(backoff * 2, BACKOFF_MAX)
            # Deliberately no put_status(ERROR): a failed upgrade must not cost a working tunnel,
            # and the poll below overwrites the status with what the daemon is actually doing.

      if daemon is None or daemon.poll() is not None:
        if daemon is not None:
          cloudlog.event("moonpilot tailscaled exited", code=daemon.returncode)
          tailscale.put_status(params, tailscale.ERROR, "tailscaled exited")
          time.sleep(backoff)
          backoff = min(backoff * 2, BACKOFF_MAX)

        # ponytail: tailscaled's own output goes to /dev/null — the exit code lands in cloudlog,
        # and a launch that never succeeds is debugged by running daemon_args() by hand over ssh.
        # A log file here would grow without a bound.
        daemon = subprocess.Popen(tailscale.daemon_args(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tailscale.put_status(params, tailscale.STARTING)

      output = _query_status()

      if output and not autoupdate_disabled:
        # The fork tracks TRACK itself (offroad, hash-verified, supervised restart), so tailscale's
        # own updater stays off: it would replace the running binaries and restart via systemd or
        # init.d, absent here — leaving new bytes on disk, the old process running, and the marker
        # lying.
        try:
          done = subprocess.run(tailscale.cli_args("set", "--auto-update=false"), check=False, capture_output=True, timeout=10)
          autoupdate_disabled = done.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
          pass

      state, detail = tailscale.parse_status(output)
      if state == tailscale.RUNNING:
        backoff = BACKOFF_START  # something works, so a later failure starts over
      elif (login is None or login.poll() is not None) and time.monotonic() - last_up > UP_RETRY:
        # `tailscale up` blocks until the node is authenticated, so the live child is what holds
        # the interactive login open; the URL is read from status --json, never from its stdout.
        # This one branch covers every non-running state: NeedsLogin (first auth), Stopped
        # (WantRunning=false after an earlier `down`), NoState.
        login = subprocess.Popen(tailscale.up_args(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        last_up = time.monotonic()

      tailscale.put_status(params, state, detail)
      time.sleep(POLL)
  finally:
    # The manager sends SIGINT on stop (process.py:82), which lands here as KeyboardInterrupt.
    _stop(login)
    _stop(daemon)
    tailscale.put_status(params, tailscale.STOPPED)


if __name__ == "__main__":
  main()
