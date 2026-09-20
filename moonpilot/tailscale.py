"""tailscale on the device: where its binaries live, how they are run, and what state means.

The device has no tailscale and no package manager (AGENTS.md, How a dependency reaches the
device), so `install()` fetches the stable-track static tarball into `<data_root>/tailscale`
and `moonpilot/tailscaled.py` supervises it. Binaries, the control socket and tailscaled's var
root stay there; its node identity, login session and prefs live at `<persist_root>/tailscaled.state`
so a factory reset does not log the device out. This module is the shared vocabulary for both,
plus the encode/decode of the one string param the supervisor and the settings panels talk
through.

It is imported by the UI and by the supervisor, so: standard library plus `openpilot` modules
only, and nothing here creates a directory or runs a process — the path helpers are computed,
the same rule and the same reason as deps.site_dir().

"""


import json
import os
import shutil

from openpilot.common.hardware import PC
from openpilot.common.params import Params

from moonpilot import paths
from moonpilot.features import TAILSCALE

# The stable track. The supervisor resolves it via `latest_release()` at upgrade time; no
# checkout change is needed to move the device's client forward.
TRACK = "stable"
PKGS_BASE = "https://pkgs.tailscale.com"
ARCH = "arm64"

BINARIES = ("tailscale", "tailscaled")
HOSTNAME = "moonpilot"  # control suffixes a collision inside one tailnet, so no dongle id is sent

STATUS_KEY = "MoonpilotTailscaleStatus"
ENABLE_KEY = TAILSCALE.key

# Status param shape: "<state>" or "<state> <detail>". Decoded by read_status().
OFFLINE = "offline"
INSTALLING = "installing"
STARTING = "starting"
LOGIN = "login"
RUNNING = "running"
ERROR = "error"
STOPPED = "stopped"

_MACHINE_AUTH_DETAIL = "approve this device in the tailscale admin console"


_state_fallback: tuple[str, str] | None = None


def root() -> str:
  """`<data_root>/tailscale`: binaries, the control socket and tailscaled's var root."""
  return os.path.join(paths.data_root(), "tailscale")


def legacy_state_path() -> str:
  """The pre-persistence state path, kept as a migration fallback."""
  return os.path.join(root(), "tailscaled.state")


def persistent_state_path() -> str:
  """The state path that survives a factory reset."""
  return os.path.join(paths.persist_root(), "tailscaled.state")


def _persist_writable() -> bool:
  """Whether the existing persistent directory can accept a new state file."""
  try:
    mode = os.stat(paths.persist_root()).st_mode
  except OSError:
    return False
  return bool(mode & 0o222) and os.access(paths.persist_root(), os.W_OK)


def bin_dir() -> str:
  """Where `install()` puts the two binaries."""
  return os.path.join(root(), "bin")


def marker_path() -> str:
  """The file naming the version `install()` last wrote. Read, never executed."""
  return os.path.join(bin_dir(), ".version")


def socket_path() -> str:
  """tailscaled's control socket, also how the CLI finds the running daemon."""
  return os.path.join(root(), "tailscaled.sock")


def state_path() -> str:
  """tailscaled's persistent state, or the legacy path when migration cannot use `/persist`."""
  persistent = persistent_state_path()
  legacy = legacy_state_path()
  if _state_fallback == (persistent, legacy) or (os.path.isfile(legacy) and not _persist_writable()):
    return legacy
  return persistent


def use_legacy_state() -> None:
  """Keep the daemon on its old state file after a failed persistent migration."""
  global _state_fallback
  _state_fallback = (persistent_state_path(), legacy_state_path())


def use_persistent_state() -> None:
  """Clear a previous migration fallback after persistence is available again."""
  global _state_fallback
  _state_fallback = None


def binaries() -> tuple[str, str] | None:
  """`(tailscale, tailscaled)` to run, or None when there is no pair at all.

  A pair the OS ships wins, mirroring deps.uv()'s preference for a system uv: that is what makes
  the dev PC and the smoke test work without a download. The installed pair is returned whatever
  its version — whether a newer stable release exists is a separate question, `upgrade_available()`.
  """
  system = (shutil.which("tailscale"), shutil.which("tailscaled"))
  if system[0] is not None and system[1] is not None:
    return system[0], system[1]

  installed = tuple(os.path.join(bin_dir(), name) for name in BINARIES)
  if os.path.isfile(installed[0]) and os.path.isfile(installed[1]):
    return installed[0], installed[1]

  return None


def marker_version() -> str:
  """What the installed binaries claim to be, or "" when nothing is installed here."""
  try:
    with open(marker_path()) as f:
      return f.read().strip()
  except OSError:
    return ""


def installed() -> bool:
  """Is there a pair to run at all?"""
  return binaries() is not None


def latest_release(timeout: float = 10.0) -> tuple[str, str, str]:
  """`(version, tarball_url, sha256)` of the latest `TRACK` release for `ARCH`. Raises."""
  import urllib.request

  with urllib.request.urlopen(f"{PKGS_BASE}/{TRACK}/?mode=json&os=linux", timeout=timeout) as r:
    payload = json.loads(r.read().decode())
  version = payload["TarballsVersion"]
  asset = payload["Tarballs"][ARCH]
  url = f"{PKGS_BASE}/{TRACK}/{asset}"
  with urllib.request.urlopen(f"{url}.sha256", timeout=timeout) as r:
    sha256 = r.read().decode().strip()
  return version, url, sha256


def upgrade_available(latest: str) -> bool:
  """Installed under `bin_dir()` and not `latest`.

  False for a pair the OS shipped: the fork does not manage an install it did not make.
  """
  pair = binaries()
  if pair is None or os.path.dirname(pair[0]) != bin_dir():
    return False
  return marker_version() != latest


def install(version: str, url: str, sha256: str) -> None:
  """Download the given tarball and put both binaries in `bin_dir()`. Raises; the caller handles it."""
  from moonpilot import fetch  # fetch's heavy stdlib stays behind this import

  paths.data_dir("tailscale")  # the one place this tree is created
  fetch.extract(fetch.download(url, sha256), BINARIES, bin_dir())
  with open(marker_path(), "w") as f:
    f.write(version)


def sudo() -> list[str]:
  """Prefix for a privileged argv.

  Device processes are not root, so privileged work goes through passwordless sudo, like
  openpilot/common/utils.py:35-52. `-n` so a misconfigured sudoers fails instead of hanging on a
  password prompt. Nothing is prefixed on PC, which is what makes the smoke test runnable.
  """
  return [] if PC else ["sudo", "-n"]


def daemon_args() -> list[str]:
  """argv for tailscaled."""
  pair = binaries()
  assert pair is not None, "daemon_args() called with no tailscaled installed"
  # `--statedir` is passed rather than left to be derived. Derivation is conditional: without it
  # tailscaled sets its var root from `--state` only when that file's directory is named
  # `tailscale` (cmd/tailscaled/tailscaled.go, `ipnServerOpts`). Naming a directory is not a
  # contract, and an empty var root sends certs, Taildrop and profile-data to HOME — the
  # read-only rootfs on device. The state *store* is the reset-surviving `--state` path; binaries,
  # the socket and this var root remain under data-root `root()`.
  args = [*sudo(), pair[1], "--state", state_path(), "--statedir", root(), "--socket", socket_path(), "--no-logs-no-support"]
  if not os.path.exists("/dev/net/tun"):
    # comma 3X's kernel has CONFIG_TUN=y; the comma four kernel is a different tree, so probe
    # rather than assume. Userspace mode still forwards inbound tailnet connections to localhost,
    # which is the point of this feature (ssh); it only loses the tailscale0 interface.
    args += ["--tun", "userspace-networking"]
  return args


def cli_args(*args: str) -> list[str]:
  """argv for the tailscale CLI, against our own socket."""
  pair = binaries()
  assert pair is not None, "cli_args() called with no tailscale installed"
  return [*sudo(), pair[0], "--socket", socket_path(), *args]


def up_args() -> list[str]:
  """argv for the interactive login. `up` blocks until the node is authenticated."""
  return cli_args("up", "--accept-dns=false", "--accept-routes=false", "--netfilter-mode=off", f"--hostname={HOSTNAME}")


def parse_status(payload: str) -> tuple[str, str]:
  """`(state, detail)` from `tailscale status --json` output. Unknown input reads as starting."""
  try:
    status = json.loads(payload)
    backend = status["BackendState"]
    ips = status.get("TailscaleIPs") or []
  except (TypeError, ValueError, KeyError, AttributeError):
    return STARTING, ""

  if backend == "Running":
    return RUNNING, ips[0] if ips else ""
  if backend == "NeedsMachineAuth":
    return ERROR, _MACHINE_AUTH_DETAIL
  if backend in ("NeedsLogin", "NoState", "Stopped"):
    url = status.get("AuthURL") or ""
    return (LOGIN, url) if url else (STARTING, "")
  return STARTING, ""


def put_status(params: Params, state: str, detail: str = "") -> None:
  """Publish `(state, detail)` for the panels."""
  params.put(STATUS_KEY, f"{state} {detail}".strip())


def read_status(params: Params) -> tuple[str, str]:
  """`(state, detail)` as last published, or `("", "")` when nothing has been."""
  state, _, detail = (params.get(STATUS_KEY) or "").partition(" ")
  return state, detail


def status_text(params: Params) -> tuple[str, str]:
  """`(value, detail)` for a settings row. Both panels render this and nothing else.

  The toggle is checked first, so a supervisor that stopped never leaves a stale state on screen.
  """
  if not params.get(ENABLE_KEY, return_default=True):
    return "off", "Turn on to join this device to your tailnet."

  state, detail = read_status(params)
  if state == OFFLINE:
    return "offline", "Waiting for a network."
  if state == INSTALLING:
    return "installing", "Downloading tailscale (~35 MB)."
  if state == LOGIN:
    return "sign in", detail
  if state == RUNNING:
    return detail or "running", f"Connected as {detail}." if detail else ""
  if state == ERROR:
    return "error", detail
  return "starting", ""


def auth_url(params: Params) -> str:
  """The interactive login URL while sign-in is pending, else "". The QR dialog's only input.

  The toggle is checked here too, not just in the supervisor: a supervisor killed without running
  its cleanup would otherwise leave a login line behind while the feature is off, and the panel
  would offer a sign-in for a daemon that is not running. It also closes an open dialog the moment
  the toggle goes off, without waiting for the process to notice.
  """
  if not params.get(ENABLE_KEY, return_default=True):
    return ""
  state, detail = read_status(params)
  return detail if state == LOGIN and detail.startswith("https://") else ""
