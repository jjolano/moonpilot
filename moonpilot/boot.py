"""Boot health and update rollback markers shared by the manager and launcher."""

import os

STAGING_ROOT = os.environ.get("STAGING_ROOT", "/data/safe_staging")
BOOT_ID_PATH = os.environ.get("MOONPILOT_BOOT_ID_PATH", "/proc/sys/kernel/random/boot_id")
SWAP_MARKER = "moonpilot_swap"
BOOT_OK_MARKER = "moonpilot_boot_ok"
ROLLBACK_MARKER = "moonpilot_rollback"
FAILED_OPENPILOT = "failed_openpilot"
ROLLBACK_KEY = "MoonpilotRollback"


def boot_id() -> str:
  try:
    with open(BOOT_ID_PATH, encoding="utf-8") as source:
      return source.read().strip()
  except OSError:
    return ""


def _path(name: str) -> str:
  return os.path.join(STAGING_ROOT, name)


def mark_boot_healthy() -> None:
  """Write the current boot token, ignoring unavailable staging storage."""
  token = boot_id()
  if not token or not os.path.isdir(STAGING_ROOT):
    return
  try:
    with open(_path(BOOT_OK_MARKER), "w", encoding="utf-8") as marker:
      marker.write(token + "\n")
  except OSError:
    pass


def read_rollback() -> str:
  """Return the launcher's rollback sentence, or an empty string if absent."""
  try:
    with open(_path(ROLLBACK_MARKER), encoding="utf-8") as record:
      return record.readline().strip()
  except OSError:
    return ""


def rollback_text(params) -> str:
  """Return the rollback sentence currently published for this manager boot."""
  try:
    value = params.get(ROLLBACK_KEY)
  except Exception:
    return ""
  if not value:
    return ""
  return value.decode() if isinstance(value, bytes) else str(value)
