"""Where moonpilot keeps state that outlives a process, a boot and an update.

`/data/openpilot` is git state: the installer moves a fresh clone over it and the updater runs
`git reset --hard` in the finalized overlay, so anything written inside the checkout is gone on the
next update. Fork state goes under a root of its own, split the same way upstream splits its own:

- `data_root()` — the data partition, beside /data/params, /data/log and /data/safe_staging.
  Survives updates; a factory reset wipes it (`rm -rf /data/*` + mkfs.ext4 in tici_reset.py).
- `persist_root()` — /persist, which a factory reset leaves alone. It also holds the dongle id and
  the RSA key, so keep it to what genuinely has to survive a reset.

Blobs, datasets and anything unbounded belong in `data_root()`. A scalar or small JSON value
belongs in a `Moonpilot*` param row instead (AGENTS.md): already persistent, atomic, and readable
by every process.

`data_dir()`'s tree is not managed by the drive deleter, which only reclaims space under
`Paths.log_root()`. A store that grows needs its own bound — size cap, ring buffer or retention.
"""

import os

from openpilot.common.hardware import PC
from openpilot.common.hardware.hw import Paths

DEVICE_DATA_ROOT = "/data/moonpilot"


def data_root() -> str:
  """Fork state on the data partition: /data/moonpilot on device, ~/.comma/moonpilot on PC."""
  return os.path.join(Paths.comma_home(), "moonpilot") if PC else DEVICE_DATA_ROOT


def persist_root() -> str:
  """Fork state that must survive a factory reset: /persist/moonpilot on device. Keep it small."""
  return os.path.join(Paths.persist_root(), "moonpilot")


def data_dir(feature: str) -> str:
  """`<data_root>/<feature>`, created on demand. One directory per feature, named for it."""
  path = os.path.join(data_root(), feature)
  os.makedirs(path, mode=0o775, exist_ok=True)
  return path
