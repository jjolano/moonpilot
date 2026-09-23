import hashlib
import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

from moonpilot import deps, paths
from moonpilot.features import FEATURES, Feature, available, enabled, missing_modules, wanted
from moonpilot.tests.fakes import _params

# A name no distribution can provide, so `available()` is False without depending on the
# environment. 'json' is stdlib and always present.
ABSENT = "moonpilot_no_such_package_anywhere"
PRESENT = "json"


class ParamStore:
  def __init__(self):
    self.values: dict[str, str] = {}
    self.writes: list[tuple[str, str]] = []

  def get(self, key, return_default=False):
    return self.values.get(key)

  def put(self, key, value, block=False):
    self.values[key] = value
    self.writes.append((key, value))

  def remove(self, key):
    self.values.pop(key, None)


def _feature(requires: tuple[str, ...]) -> Feature:
  return Feature(key="MoonpilotTest", title="test", description="test", requires=requires)


class TestAvailability(unittest.TestCase):
  def test_available_answers_without_importing(self):
    # The property the whole init path rests on: a declared requirement may be a heavy package,
    # and this runs in the manager's ~1 Hz should_run, in the UI and in plannerd. find_spec must
    # answer from the finders, so a module that would blow up on execution still reads present.
    with tempfile.TemporaryDirectory() as tmp:
      with open(os.path.join(tmp, "moonpilot_import_probe.py"), "w") as f:
        f.write("raise RuntimeError('imported')\n")
      sys.path.insert(0, tmp)
      try:
        self.assertTrue(deps.available("moonpilot_import_probe"))
        self.assertNotIn("moonpilot_import_probe", sys.modules)
      finally:
        sys.path.remove(tmp)
        importlib.invalidate_caches()

  def test_absent_and_present_modules(self):
    self.assertFalse(deps.available(ABSENT))
    self.assertTrue(deps.available(PRESENT))

  def test_availability_needs_every_declared_module(self):
    self.assertEqual(missing_modules(_feature((PRESENT, ABSENT))), (ABSENT,))
    self.assertFalse(available(_feature((PRESENT, ABSENT))))
    self.assertTrue(available(_feature((PRESENT,))))
    self.assertTrue(available(_feature(())))


class TestGate(unittest.TestCase):
  def test_missing_dependency_turns_the_feature_off(self):
    # The bug this whole change exists to prevent: a feature the driver asked for, whose package
    # is not installed, must read as off everywhere the seam looks — while still recording that
    # the driver wanted it, so the toggle does not silently flip itself back.
    feature = _feature((ABSENT,))
    self.assertTrue(wanted(feature, _params(True)))
    self.assertFalse(enabled(feature, _params(True)))

  def test_present_dependency_leaves_the_decision_to_the_driver(self):
    feature = _feature((PRESENT,))
    self.assertTrue(enabled(feature, _params(True)))
    self.assertFalse(enabled(feature, _params(False)))


class TestPaths(unittest.TestCase):
  def test_site_dir_is_computed_not_created(self):
    # deps.py is imported by the UI and by plannerd. Creating the tree here would turn a
    # read-only or unmounted /data into a crash at import time.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      target = deps.site_dir()
      self.assertEqual(target, os.path.join(tmp, "deps", "current"))
      self.assertFalse(os.path.lexists(target))


class TestActivation(unittest.TestCase):
  """`activate()` is what makes an install visible to a process that was already running."""

  def _cleanup(self, *entries: str) -> None:
    for entry in entries:
      if entry in sys.path:
        sys.path.remove(entry)
      sys.path_importer_cache.pop(entry, None)
    importlib.invalidate_caches()

  def test_available_finds_a_package_installed_after_import(self):
    # The boot-order trap: the UI, plannerd and the manager import this module while the tree
    # still does not exist, then run for hours. A package depsd installs mid-boot has to become
    # visible to them, not wait for a restart to be noticed.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      target = deps.site_dir()
      release = os.path.join(tmp, "deps", "releases", "first")
      os.makedirs(release)
      os.symlink(release, target)
      self.assertNotIn(target, sys.path)  # imported before depsd created anything
      with open(os.path.join(release, "moonpilot_activation_probe.py"), "w") as f:
        f.write("value = 'installed'\n")
      try:
        self.assertTrue(deps.available("moonpilot_activation_probe"))
        self.assertTrue(target in sys.path)
      finally:
        self._cleanup(target)

  def test_an_install_never_shadows_what_the_system_ships(self):
    # Append, not insert. The same module in two places on sys.path must keep resolving to the
    # earlier entry, or a fork package would silently replace one AGNOS already provides.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      system = tempfile.mkdtemp()
      release = os.path.join(tmp, "deps", "releases", "first")
      os.makedirs(release)
      os.symlink(release, deps.site_dir())
      for directory, value in ((system, "system"), (release, "fork")):
        with open(os.path.join(directory, "moonpilot_shadow_probe.py"), "w") as f:
          f.write(f"value = {value!r}\n")
      sys.path.insert(0, system)
      try:
        deps.activate()
        self.assertEqual(importlib.import_module("moonpilot_shadow_probe").value, "system")
      finally:
        sys.modules.pop("moonpilot_shadow_probe", None)
        self._cleanup(system, deps.site_dir())


class TestStatusAndRetry(unittest.TestCase):
  def test_status_writes_only_when_changed(self):
    params = ParamStore()
    deps.put_status(params, deps.WAITING_NETWORK, "Waiting for a network.")
    deps.put_status(params, deps.WAITING_NETWORK, "Waiting for a network.")
    self.assertEqual(params.writes, [(deps.STATUS_KEY, "waiting-network Waiting for a network.")])
    self.assertEqual(deps.read_status(params), (deps.WAITING_NETWORK, "Waiting for a network."))
    self.assertEqual(deps.status_text(params), ("waiting for network", "Waiting for a network."))

  def test_status_transitions_keep_state_and_detail(self):
    params = ParamStore()
    for state in (deps.READY, deps.WAITING_NETWORK, deps.WAITING_METERED, deps.INSTALLING, deps.ERROR):
      with self.subTest(state=state):
        deps.put_status(params, state, "cryptography is missing.")
        self.assertEqual(deps.read_status(params), (state, "cryptography is missing."))

  def test_empty_status_reads_ready_when_nothing_is_missing(self):
    # depsd never starts when missing() is empty, so MoonpilotDepsStatus stays cleared at
    # boot — the row used to sit on the unknown-state fallback ("starting") forever.
    params = ParamStore()
    with mock.patch.object(deps, "missing", return_value=[]):
      self.assertEqual(deps.status_text(params), ("ready", "All required modules are installed."))

  def test_empty_status_stays_starting_while_something_is_missing(self):
    params = ParamStore()
    with mock.patch.object(deps, "missing", return_value=[deps.Requirement(ABSENT, "x==1")]):
      self.assertEqual(deps.status_text(params), ("starting", ""))

  def test_retry_request_is_consumed_once(self):
    params = ParamStore()
    deps.request_retry(params)
    self.assertTrue(deps.take_retry(params))
    self.assertFalse(deps.take_retry(params))


class TestInstall(unittest.TestCase):
  def _fingerprint(self) -> str:
    with open(deps.lock_path(), "rb") as lock:
      return hashlib.sha256(lock.read()).hexdigest()[:16]

  def test_failed_install_preserves_current_and_releases(self):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      release = os.path.join(tmp, "deps", "releases", "old")
      os.makedirs(release)
      with open(os.path.join(release, "marker"), "w") as f:
        f.write("old")
      current = deps.site_dir()
      os.symlink(release, current)
      with (
        mock.patch.object(deps, "uv", return_value="/usr/bin/uv"),
        mock.patch.object(deps.subprocess, "run", side_effect=RuntimeError("offline")),
      ):
        with self.assertRaisesRegex(RuntimeError, "offline"):
          deps.install()
      self.assertEqual(os.readlink(current), release)
      self.assertEqual(os.listdir(os.path.join(tmp, "deps", "releases")), ["old"])
      with open(os.path.join(release, "marker")) as f:
        self.assertEqual(f.read(), "old")

  def test_success_promotes_release_and_removes_stale_state(self):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      old = os.path.join(tmp, "deps", "releases", "old")
      os.makedirs(old)
      os.makedirs(os.path.join(tmp, "deps", "site-packages"))
      with open(os.path.join(tmp, "deps", "site-packages", "stale"), "w") as f:
        f.write("stale")

      def install_into_target(argv, **kwargs):
        target = argv[argv.index("--target") + 1]
        with open(os.path.join(target, "moonpilot_install_probe.py"), "w") as f:
          f.write("value = 'installed'\n")

      with (
        mock.patch.object(deps, "uv", return_value="/usr/bin/uv"),
        mock.patch.object(deps.subprocess, "run", side_effect=install_into_target),
      ):
        deps.install()

      release = os.path.join(tmp, "deps", "releases", self._fingerprint())
      self.assertEqual(os.path.realpath(deps.site_dir()), release)
      self.assertTrue(os.path.isfile(os.path.join(release, "moonpilot_install_probe.py")))
      self.assertEqual(os.listdir(os.path.join(tmp, "deps", "releases")), [self._fingerprint()])
      self.assertFalse(os.path.exists(os.path.join(tmp, "deps", "site-packages")))
      self.assertTrue(deps.available("moonpilot_install_probe"))

  def test_existing_release_is_reused_without_reinstall(self):
    # The fingerprint names an immutable successful install. Re-running depsd must promote it,
    # not install into a throwaway directory and then point `current` at the old contents.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      release = os.path.join(tmp, "deps", "releases", self._fingerprint())
      os.makedirs(release)
      with open(os.path.join(release, "marker"), "w") as marker:
        marker.write("complete")
      with (
        mock.patch.object(deps, "uv", return_value="/usr/bin/uv"),
        mock.patch.object(deps.subprocess, "run") as run,
        mock.patch.object(deps, "activate"),
      ):
        deps.install()
      run.assert_not_called()
      self.assertEqual(os.path.realpath(deps.site_dir()), release)
      with open(os.path.join(release, "marker")) as marker:
        self.assertEqual(marker.read(), "complete")


def _normalize(name: str) -> str:
  # PEP 503, so `Foo_Bar` and `foo-bar` are the same distribution.
  return name.strip().lower().replace("_", "-").replace(".", "-")


def _locked_distributions() -> set[str]:
  names: set[str] = set()
  with open(deps.lock_path()) as f:
    for line in f:
      spec = line.split("#", 1)[0].split("--", 1)[0].strip()
      if not spec:
        continue
      for sep in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in spec:
          spec = spec.split(sep, 1)[0]
          break
      names.add(_normalize(spec))
  return names


def _check_registry(requirements: tuple[deps.Requirement, ...], features: tuple[Feature, ...]) -> None:
  """Adding a dependency is three things, and they have to agree.

  A `Requirement` the lock does not carry never installs; a module a feature `requires` that no
  `Requirement` names can never become available, so the feature is dead weight nobody can turn
  on. Both are silent at runtime, which is why they are checked here.

  A function over the tables rather than assertions over the live ones, because the live ones are
  empty today: `REQUIREMENTS` being empty must not be the reason the guard cannot fail.
  """
  locked = _locked_distributions()
  for requirement in requirements:
    dist = _normalize(requirement.spec.split("==", 1)[0])
    assert dist in locked, f"{requirement.spec} is in REQUIREMENTS but not in moonpilot/deps.lock"

  declared = {r.module for r in requirements}
  for feature in features:
    for module in feature.requires:
      assert module in declared, f"{feature.key} requires {module}, which no Requirement installs"


class TestRegistryAgreesWithTheLock(unittest.TestCase):
  def test_shipped_registry_agrees(self):
    _check_registry(deps.REQUIREMENTS, FEATURES)

  def test_a_requirement_missing_from_the_lock_is_caught(self):
    with self.assertRaises(AssertionError):
      _check_registry((deps.Requirement(module="json", spec="definitely-not-locked==1.0.0"),), ())

  def test_a_feature_requiring_an_undeclared_module_is_caught(self):
    with self.assertRaises(AssertionError):
      _check_registry((), (_feature((ABSENT,)),))


if __name__ == "__main__":
  unittest.main()
