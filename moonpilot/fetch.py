"""Fetch a pinned artifact and install the binaries inside it.

Two fork features need a binary the device does not ship — the uv that installs Python packages
(`moonpilot/deps.py`) and tailscale (`moonpilot/tailscale.py`) — so download, verify and extract
live here rather than inline in each.

The heavy stdlib is imported at module scope on purpose: nothing on the init path imports this
module. `deps.py` and `tailscale.py` both import it *inside* the function that downloads, which
is what keeps `hashlib`/`tarfile`/`urllib` off the boot path (see AGENTS.md, Dependencies).
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request


def download(url: str, sha256: str, timeout: float = 120.0) -> bytes:
  """The payload at `url`, or ValueError if it is not the artifact `sha256` names.

  The check is not a nicety: what comes back is executed, usually as root.
  """
  with urllib.request.urlopen(url, timeout=timeout) as resp:
    payload = resp.read()

  digest = hashlib.sha256(payload).hexdigest()
  if digest != sha256:
    raise ValueError(f"{url} sha256 mismatch: got {digest}, want {sha256}")

  return payload


def extract(payload: bytes, names: tuple[str, ...], dest: str) -> None:
  """Write the tar members whose basename is in `names` into `dest`, mode 0o755.

  Members are selected by basename, which is unambiguous for the archives here — the systemd unit
  inside tailscale's tarball has basename `tailscaled.service`, not `tailscaled`.

  Each binary is written to `<name>.new` and moved onto its final path: overwriting a running
  binary in place fails with ETXTBSY, and this function does reinstall over a live `tailscaled`.
  """
  os.makedirs(dest, mode=0o775, exist_ok=True)
  missing = set(names)

  with tempfile.TemporaryDirectory() as tmp:
    archive = os.path.join(tmp, "payload.tar")
    with open(archive, "wb") as f:
      f.write(payload)

    with tarfile.open(archive) as tar:
      for member in tar.getmembers():
        name = os.path.basename(member.name)
        if not member.isfile() or name not in missing:
          continue

        source = tar.extractfile(member)
        if source is None:
          continue

        staged = os.path.join(dest, name + ".new")
        with open(staged, "wb") as f:
          shutil.copyfileobj(source, f)
        os.chmod(staged, 0o755)
        os.replace(staged, os.path.join(dest, name))
        missing.discard(name)

  if missing:
    raise ValueError(f"missing from archive: {sorted(missing)}")
