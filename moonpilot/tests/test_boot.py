import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from moonpilot import boot

ROOT = Path(__file__).resolve().parents[2]
BOOT_SH = ROOT / "moonpilot" / "boot.sh"


class FakeParams:
  def __init__(self, values=None):
    self.values = values or {}

  def get(self, key):
    return self.values.get(key)

  def put(self, key, value, block=False):
    self.values[key] = value


class TestBootRecovery(unittest.TestCase):
  def recover(self, staging: Path, checkout: Path, token: str | None) -> str:
    boot_id_file = staging / "boot_id"
    env = os.environ.copy()
    if token is None:
      env["MOONPILOT_BOOT_ID_PATH"] = str(staging / "missing_boot_id")
    else:
      boot_id_file.write_text(token + "\n", encoding="utf-8")
      env["MOONPILOT_BOOT_ID_PATH"] = str(boot_id_file)
    result = subprocess.run(
      ("bash", str(BOOT_SH), "recover", str(staging), str(checkout)),
      cwd=ROOT,
      capture_output=True,
      text=True,
      env=env,
      check=True,
    )
    self.assertEqual(result.stderr, "")
    return result.stdout.strip()

  def test_mid_swap_crash_restores_the_old_tree(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      old = staging / "old_openpilot"
      old.mkdir()
      (staging / boot.SWAP_MARKER).write_text("boot-a\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, "boot-b"), "restart")
      self.assertTrue(checkout.is_dir())
      self.assertFalse(old.exists())
      self.assertFalse((staging / boot.SWAP_MARKER).exists())

  def test_same_boot_swap_waits_for_manager(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      (staging / boot.SWAP_MARKER).write_text("boot-a\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, "boot-a"), "none")
      self.assertTrue(checkout.is_dir())
      self.assertTrue(old.is_dir())
      self.assertTrue((staging / boot.SWAP_MARKER).exists())

  def test_matching_health_token_cleans_later_boot(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      (staging / boot.SWAP_MARKER).write_text("boot-a\n", encoding="utf-8")
      (staging / boot.BOOT_OK_MARKER).write_text("boot-a\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, "boot-b"), "cleaned")
      self.assertTrue(old.is_dir())
      self.assertTrue(checkout.is_dir())
      self.assertFalse((staging / boot.SWAP_MARKER).exists())

  def test_missing_health_token_rolls_back(self):
    self._assert_rolls_back("boot-b", None)

  def test_mismatched_health_token_rolls_back(self):
    self._assert_rolls_back("boot-b", "boot-c")

  def _assert_rolls_back(self, current_token: str, health_token: str | None):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      (checkout / "new").write_text("new", encoding="utf-8")
      old = staging / "old_openpilot"
      old.mkdir()
      (old / "old").write_text("old", encoding="utf-8")
      failed = staging / boot.FAILED_OPENPILOT
      failed.mkdir()
      (failed / "stale").touch()
      (staging / boot.SWAP_MARKER).write_text("boot-a\n", encoding="utf-8")
      if health_token is not None:
        (staging / boot.BOOT_OK_MARKER).write_text(health_token + "\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, current_token), "restart")
      self.assertEqual((checkout / "old").read_text(encoding="utf-8"), "old")
      self.assertEqual((failed / "new").read_text(encoding="utf-8"), "new")
      self.assertFalse((staging / boot.SWAP_MARKER).exists())
      record = (staging / boot.ROLLBACK_MARKER).read_text(encoding="utf-8")
      self.assertRegex(record, r"^update rolled back at .+: the new tree never reached a healthy manager\n$")

  def test_health_marker_pairs_with_shell_token_contract(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      swap = staging / boot.SWAP_MARKER
      swap.write_text("boot-a\n", encoding="utf-8")
      boot_id_file = staging / "boot_id"
      boot_id_file.write_text("boot-a\n", encoding="utf-8")

      with (
        mock.patch.object(boot, "STAGING_ROOT", str(staging)),
        mock.patch.object(boot, "BOOT_ID_PATH", str(boot_id_file)),
      ):
        boot.mark_boot_healthy()
      self.assertEqual((staging / boot.BOOT_OK_MARKER).read_text(encoding="utf-8"), "boot-a\n")

      self.assertEqual(self.recover(staging, checkout, "boot-b"), "cleaned")
      self.assertTrue(old.is_dir())
      self.assertFalse(swap.exists())

  def test_empty_swap_token_does_nothing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      swap = staging / boot.SWAP_MARKER
      swap.touch()

      self.assertEqual(self.recover(staging, checkout, "boot-a"), "none")
      self.assertTrue(old.is_dir())
      self.assertTrue(checkout.is_dir())
      self.assertTrue(swap.exists())

  def test_empty_boot_id_does_nothing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      swap = staging / boot.SWAP_MARKER
      swap.write_text("boot-a\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, ""), "none")
      self.assertTrue(old.is_dir())
      self.assertTrue(checkout.is_dir())
      self.assertTrue(swap.exists())

  def test_unreadable_boot_id_does_nothing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      swap = staging / boot.SWAP_MARKER
      swap.write_text("boot-a\n", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, None), "none")
      self.assertTrue(old.is_dir())
      self.assertTrue(checkout.is_dir())
      self.assertTrue(swap.exists())

  def test_no_evidence_does_nothing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      staging = root / "staging"
      staging.mkdir()
      checkout = root / "openpilot"
      checkout.mkdir()
      old = staging / "old_openpilot"
      old.mkdir()
      (checkout / "sentinel").write_text("keep", encoding="utf-8")

      self.assertEqual(self.recover(staging, checkout, "boot-a"), "none")
      self.assertEqual((checkout / "sentinel").read_text(encoding="utf-8"), "keep")
      self.assertTrue(old.is_dir())

  def test_health_marker_write_ignores_a_missing_root(self):
    with tempfile.TemporaryDirectory() as tmp:
      missing = Path(tmp) / "missing"
      with (
        mock.patch.object(boot, "STAGING_ROOT", str(missing)),
        mock.patch.object(boot, "BOOT_ID_PATH", str(missing / "boot_id")),
      ):
        boot.mark_boot_healthy()
      self.assertFalse(missing.exists())

  def test_rollback_text_reads_the_current_boot_param(self):
    self.assertEqual(boot.rollback_text(FakeParams()), "")
    sentence = "update rolled back at now: the new tree never reached a healthy manager"
    self.assertEqual(boot.rollback_text(FakeParams({boot.ROLLBACK_KEY: sentence})), sentence)


if __name__ == "__main__":
  unittest.main()
