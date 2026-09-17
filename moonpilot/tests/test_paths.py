import os
import tempfile
import unittest
from unittest import mock

from openpilot.common.hardware import hw

from moonpilot import paths


class TestPaths(unittest.TestCase):
  def test_device_roots_are_pinned(self):
    # The standard itself. These literals are what keeps fork state alive across an update: a
    # root under the checkout is erased by `git reset --hard`, and one under /data/params is
    # unlinked by Params::clearAll, which deletes files there that are not in the key list.
    # Both PCs are patched because each function delegates to whoever owns the location: our own
    # constant for /data, upstream's Paths.persist_root() for /persist. Both are module globals
    # bound at import, so neither follows from patching the other.
    with mock.patch.object(paths, "PC", False), mock.patch.object(hw, "PC", False):
      self.assertEqual(paths.data_root(), "/data/moonpilot")
      self.assertEqual(paths.persist_root(), "/persist/moonpilot")

  def test_pc_roots_stay_under_comma_home(self):
    # A dev box has no /data, and comma_home carries OPENPILOT_PREFIX, so prefixes stay isolated.
    with mock.patch.object(paths, "PC", True), mock.patch.object(hw, "PC", True):
      for root in (paths.data_root(), paths.persist_root()):
        self.assertTrue(root.startswith(hw.Paths.comma_home() + os.sep), root)

  def test_data_dir_creates_the_feature_directory(self):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      self.assertEqual(paths.data_dir("leadd"), os.path.join(tmp, "leadd"))
      self.assertTrue(os.path.isdir(paths.data_dir("leadd")))  # a second call at boot must not raise


if __name__ == "__main__":
  unittest.main()
