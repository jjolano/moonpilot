"""External Python packages a fork feature needs, and the installer the device doesn't have.

AGNOS ships a read-only rootfs with a system `python3.12` and no venv, no `pip` and no `uv`:
`tools/setup_dependencies.sh` installs `uv.lock` where there is a network, and nothing installs
anything afterwards. So a fork feature that needs a package off that lock has to bring its own
installer, and that is what `uv()` + `install()` are.

The target directory is `<data_root>/deps/site-packages`, outside the checkout on purpose. The
updater runs `git clean -xdff` + `git reset --hard` and then swaps the whole checkout directory,
so a package installed under `/data/openpilot` is gone on the next update; `/data` is the only
writable tree and survives (see moonpilot/paths.py). It is not on `sys.path`, hence the
import-time append below.

This module is imported on init paths (the manager's process table, the UI, plannerd), so it
imports only the standard library: everything heavier lives inside the function that needs it.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from moonpilot import paths

# The uv release we bootstrap on device. The asset is the glibc aarch64 build, which matches the
# device's `comma_arm64` target (SConstruct:179-182 selects clang and /usr/lib/aarch64-linux-gnu
# for that arch). UV_SHA256 is the sha256 of that exact asset, checked before it is ever executed.
UV_VERSION = "0.12.13"
UV_SHA256 = "2eaa5d94f5db7b3a1a092156b9420459e42ab0217d917fe74a876309cef9b5e9"
UV_ASSET = "uv-aarch64-unknown-linux-gnu.tar.gz"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{UV_ASSET}"

_DEPS_NAME = "deps"


@dataclass(frozen=True)
class Requirement:
  """One external package: the name you import, and the pin the installer resolves."""

  module: str
  spec: str


# Every package a fork feature may need. Empty today: a row here and a line in deps.lock land
# together, or `--require-hashes` fails the install.
REQUIREMENTS: tuple[Requirement, ...] = ()


def site_dir() -> str:
  """`<data_root>/deps/site-packages`. Computed, never created — creating it is `install()`'s job.

  Deliberately not `paths.data_dir()`: that mkdirs, and this module is imported by the UI and
  plannerd, where a read-only or unmounted /data would turn a missing feature into a crash at
  import time.
  """
  return os.path.join(paths.data_root(), _DEPS_NAME, "site-packages")


def lock_path() -> str:
  """`moonpilot/deps.lock` in this checkout — the install source of truth."""
  return os.path.join(os.path.dirname(os.path.abspath(__file__)), "deps.lock")


def _bin_dir() -> str:
  return os.path.join(paths.data_root(), _DEPS_NAME, "bin")


def activate() -> None:
  """Put `site_dir()` on `sys.path` if it exists. Idempotent, and safe to call at any time.

  Called at import *and* from `available()`, because those are different moments: the UI,
  plannerd and the manager all import this module early, when the tree may not exist yet, and
  then keep running for the whole boot. Without the second call, a package `depsd` installs
  mid-boot would be invisible to them — the files present, `available()` still False — and the
  feature would stay off until something restarted those processes.
  """
  target = site_dir()
  if target not in sys.path and os.path.isdir(target):
    sys.path.append(target)


def available(module: str) -> bool:
  """Is `module` importable? Answered from the finders, without importing it.

  Runs in the manager's ~1 Hz should_run evaluation, in the UI and in plannerd, and a declared
  requirement may be a heavy package — so never `import`.

  ponytail: presence only. A different version already shipped by the system satisfies this; the
  exact `spec` pin is enforced by the installer, not at runtime. Add a version compare here if a
  requirement ever needs one.
  """
  activate()
  try:
    # A nested name raises ModuleNotFoundError when the parent package is absent, and a bad name
    # raises ValueError.
    return importlib.util.find_spec(module) is not None
  except (ImportError, ValueError):
    return False


def missing() -> list[Requirement]:
  """The requirements this device cannot import yet."""
  return [r for r in REQUIREMENTS if not available(r.module)]


def uv() -> str:
  """Path to a usable uv, downloading the pinned static build if the system has none."""
  system_uv = shutil.which("uv")
  if system_uv is not None:
    return system_uv

  uv_path = os.path.join(_bin_dir(), "uv")
  if os.path.isfile(uv_path):
    return uv_path

  from moonpilot import fetch  # fetch's heavy stdlib stays behind this import; deps.py is on the init path

  fetch.extract(fetch.download(UV_URL, UV_SHA256), ("uv",), _bin_dir())
  return uv_path


def install() -> None:
  """Install everything in deps.lock into `site_dir()`. Raises on failure; the caller handles it."""
  target = site_dir()
  os.makedirs(target, mode=0o775, exist_ok=True)

  env = dict(os.environ)
  # uv's cache and /data are different filesystems, which is exactly the hardlink warning uv emits.
  env["UV_LINK_MODE"] = "copy"

  subprocess.run(
    [uv(), "pip", "install", "--target", target, "--require-hashes", "-r", lock_path()],
    env=env,
    check=True,
    capture_output=True,
    text=True,
  )

  # Anything that already looked for a package under `target` cached the miss. Drop those, or
  # `available()` keeps reporting what was true before the install.
  importlib.invalidate_caches()
  sys.path_importer_cache.pop(target, None)


# Populates sys.path for everything that imports this module before the tree exists; see
# activate(). Append, never insert: we only supply what the system is missing, so an installed
# package must never shadow one the OS or AGNOS already ships.
activate()
