import io
import json
import os
import shutil
import signal
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

from openpilot.common.params import Params

from moonpilot import fetch, paths, procs, tailscale
from moonpilot.features import FEATURES

ROOT = Path(__file__).resolve().parents[2]

# A URL-shaped login link, which is what `auth_url` insists on before the QR dialog opens.
LOGIN_URL = "https://login.tailscale.com/a/abc123"

# A fake stable release: install() takes whatever latest_release() returns, so the tests pin
# nothing and assert the passed-through values land where they should.
_VERSION = "9.9.9"
_ASSET = f"tailscale_{_VERSION}_arm64.tgz"
_URL = f"https://pkgs.tailscale.com/stable/{_ASSET}"
_SHA = "ab12" * 16


def _quick_install(payload: bytes | None = None) -> None:
  """install() with the download mocked: the argv/binaries tests need files, not the network."""
  with mock.patch.object(fetch, "download", return_value=payload or _tarball(tailscale.BINARIES)):
    tailscale.install(_VERSION, _URL, _SHA)


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store."""

  def __init__(self, values=None):
    self._values: dict[str, object] = values or {}

  def get(self, key, block=False, return_default=False):
    return self._values.get(key)

  def put(self, key, value, block=False):
    self._values[key] = value

  def get_bool(self, key, block=False):
    return bool(self._values.get(key))


def _params(**values) -> Params:
  # Same duck-typed cast as test_deps.py: the code under test only calls get()/put().
  return cast(Params, FakeParams(values))


def _status(state: str, detail: str = "") -> Params:
  return _params(**{tailscale.STATUS_KEY: f"{state} {detail}".strip(), tailscale.ENABLE_KEY: True})


def _tarball(names: tuple[str, ...]) -> bytes:
  """A tailscale-shaped tarball: both binaries under a versioned directory, plus a systemd unit."""
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w") as tar:
    for name in (*names, "tailscaled.service"):
      data = b"#!/bin/sh\n"
      info = tarfile.TarInfo(f"tailscale_x/{name}")
      info.size = len(data)
      info.mode = 0o644  # deliberately not executable: extract() is what sets the mode
      tar.addfile(info, io.BytesIO(data))
  return buf.getvalue()


class TestInstall(unittest.TestCase):
  def test_install_lands_both_binaries_and_the_marker(self):
    payload = _tarball(tailscale.BINARIES)
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(fetch, "download", return_value=payload) as download,
    ):
      tailscale.install(_VERSION, _URL, _SHA)

      self.assertEqual(download.call_args[0], (_URL, _SHA))
      for name in tailscale.BINARIES:
        path = os.path.join(tailscale.bin_dir(), name)
        self.assertTrue(os.path.isfile(path), path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o755, path)
      # The unit file inside the archive shares no basename with either binary, so it is skipped.
      self.assertFalse(os.path.exists(os.path.join(tailscale.bin_dir(), "tailscaled.service")))
      self.assertEqual(tailscale.marker_version(), _VERSION)
      self.assertTrue(tailscale.installed())

  def test_install_replaces_a_binary_that_is_being_executed(self):
    # The upgrade case: the daemon is up, running the old inode, on the old path, while the new
    # bytes land. Writing in place would raise ETXTBSY — the kernel refuses writes to a file under
    # deny_write_access — which is why extract() stages a `.new` file and os.replace()s it.
    #
    # bash is the vehicle because it ignores an unexpected argv[0]; a multicall binary renamed to
    # `tailscaled` exits instead ("unknown program"), which would leave the file executable-free
    # and the test passing for the wrong reason. Hence the check that it really is running our path.
    shell = shutil.which("bash")
    if shell is None:
      self.skipTest("no bash to execute from the installed path")

    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(fetch, "download", return_value=_tarball(tailscale.BINARIES)),
    ):
      os.makedirs(tailscale.bin_dir(), exist_ok=True)
      target = os.path.join(tailscale.bin_dir(), "tailscaled")
      shutil.copy(shell, target)
      running = subprocess.Popen([target, "-c", "while :; do :; done"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      try:
        self.assertEqual(os.readlink(f"/proc/{running.pid}/exe"), target, "not executing the installed path")
        tailscale.install(_VERSION, _URL, _SHA)
      finally:
        running.kill()
        running.wait(timeout=10)

      # New bytes on disk, and the process executing the old inode was never disturbed.
      with open(target, "rb") as f:
        self.assertEqual(f.read(), b"#!/bin/sh\n")
      self.assertEqual(running.returncode, -signal.SIGKILL)

class TestLatestRelease(unittest.TestCase):
  def test_track_json_and_sidecar_resolve_without_the_network(self):
    track = json.dumps({"TarballsVersion": _VERSION, "Tarballs": {"arm64": _ASSET}}).encode()
    with mock.patch.object(fetch.urllib.request, "urlopen", side_effect=[io.BytesIO(track), io.BytesIO(f"{_SHA}\n".encode())]) as urlopen:
      self.assertEqual(tailscale.latest_release(), (_VERSION, _URL, _SHA))
    called = [call[0][0] for call in urlopen.call_args_list]
    self.assertEqual(called, [f"{tailscale.PKGS_BASE}/{tailscale.TRACK}/?mode=json&os=linux", f"{_URL}.sha256"])

  def test_the_release_tuple_splats_straight_into_install(self):
    # install() takes (version, url, sha256) in latest_release()'s order, so the supervisor's
    # install(*latest_release()) cannot silently swap the url and the digest.
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(tailscale, "latest_release", return_value=(_VERSION, _URL, _SHA)),
      mock.patch.object(fetch, "download", return_value=_tarball(tailscale.BINARIES)) as download,
    ):
      tailscale.install(*tailscale.latest_release())
      self.assertEqual(download.call_args[0], (_URL, _SHA))
      self.assertEqual(tailscale.marker_version(), _VERSION)
      self.assertTrue(tailscale.installed())


class TestFetch(unittest.TestCase):
  def test_download_rejects_a_payload_the_digest_does_not_name(self):
    # Security-critical: the artifact is executed, usually as root.
    with mock.patch.object(fetch.urllib.request, "urlopen", return_value=io.BytesIO(b"not the artifact")):
      with self.assertRaises(ValueError) as caught:
        fetch.download("https://example.invalid/x.tgz", "0" * 64)
    self.assertTrue("sha256 mismatch" in str(caught.exception))

  def test_extract_reports_a_name_the_archive_does_not_carry(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaises(ValueError) as caught:
        fetch.extract(_tarball(("tailscale",)), tailscale.BINARIES, tmp)
      self.assertTrue("tailscaled" in str(caught.exception), "the missing name must be reported")


class TestArgv(unittest.TestCase):
  def _with_tun(self, present: bool):
    return mock.patch.object(tailscale.os.path, "exists", return_value=present)

  def test_daemon_args_carry_state_socket_and_no_logs(self):
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp), self._with_tun(True):
      _quick_install()
      args = tailscale.daemon_args()
      self.assertTrue("--state" in args)
      self.assertTrue(tailscale.state_path() in args)
      self.assertTrue("--socket" in args)
      self.assertTrue(tailscale.socket_path() in args)
      self.assertTrue("--no-logs-no-support" in args)

  def test_the_var_root_is_passed_not_derived(self):
    # Derivation from --state is conditional on the state file's directory being named `tailscale`
    # (cmd/tailscaled/tailscaled.go, ipnServerOpts). Left to that, a rename silently sends certs,
    # Taildrop and profile-data to HOME — the read-only rootfs on device.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp), self._with_tun(True):
      _quick_install()
      args = tailscale.daemon_args()
      self.assertTrue("--statedir" in args)
      self.assertEqual(args[args.index("--statedir") + 1], tailscale.root())
      self.assertEqual(os.path.dirname(tailscale.state_path()), tailscale.root())

  def test_tun_mode_is_probed_not_assumed(self):
    # comma 3X's kernel has CONFIG_TUN=y; the comma four kernel is a different tree.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      _quick_install()
      with self._with_tun(False):
        self.assertTrue("userspace-networking" in tailscale.daemon_args())
      with self._with_tun(True):
        self.assertFalse("userspace-networking" in tailscale.daemon_args())

  def test_sudo_prefix_is_device_only(self):
    with mock.patch.object(tailscale, "PC", False):
      self.assertEqual(tailscale.sudo(), ["sudo", "-n"])
    with mock.patch.object(tailscale, "PC", True):
      self.assertEqual(tailscale.sudo(), [])

  def test_up_disables_dns_and_netfilter(self):
    # --accept-dns=false against a read-only rootfs: letting tailscaled own /etc/resolv.conf can
    # only fail. --netfilter-mode=off drops the need for an iptables binary on AGNOS.
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp):
      _quick_install()
      args = tailscale.up_args()
      self.assertTrue("--accept-dns=false" in args, "a read-only rootfs cannot survive tailscaled owning resolv.conf")
      self.assertTrue("--netfilter-mode=off" in args)
      self.assertTrue(f"--hostname={tailscale.HOSTNAME}" in args)
      self.assertEqual(args[-5:], ["up", "--accept-dns=false", "--accept-routes=false", "--netfilter-mode=off", f"--hostname={tailscale.HOSTNAME}"])


class TestParseStatus(unittest.TestCase):
  def test_every_backend_state(self):
    cases = {
      '{"BackendState": "Running", "TailscaleIPs": ["100.64.0.7"]}': (tailscale.RUNNING, "100.64.0.7"),
      '{"BackendState": "Running"}': (tailscale.RUNNING, ""),
      f'{{"BackendState": "NeedsLogin", "AuthURL": "{LOGIN_URL}"}}': (tailscale.LOGIN, LOGIN_URL),
      '{"BackendState": "NeedsLogin"}': (tailscale.STARTING, ""),
      '{"BackendState": "NoState"}': (tailscale.STARTING, ""),
      '{"BackendState": "Stopped"}': (tailscale.STARTING, ""),
      '{"BackendState": "Starting"}': (tailscale.STARTING, ""),
      '{"BackendState": "NeedsMachineAuth"}': (tailscale.ERROR, "approve this device in the tailscale admin console"),
      '{"BackendState": "SomethingNew"}': (tailscale.STARTING, ""),
      "": (tailscale.STARTING, ""),
      "not json at all": (tailscale.STARTING, ""),
    }
    for payload, expected in cases.items():
      with self.subTest(payload=payload):
        self.assertEqual(tailscale.parse_status(payload), expected)


class TestStatusText(unittest.TestCase):
  def test_toggle_off_wins_over_a_stale_status(self):
    params = _params(**{tailscale.STATUS_KEY: f"{tailscale.RUNNING} 100.64.0.7"})
    self.assertEqual(tailscale.status_text(params), ("off", "Turn on to join this device to your tailnet."))

  def test_each_state_has_a_row(self):
    self.assertEqual(tailscale.status_text(_status(tailscale.OFFLINE)), ("offline", "Waiting for a network."))
    self.assertEqual(tailscale.status_text(_status(tailscale.INSTALLING))[0], "installing")
    self.assertEqual(tailscale.status_text(_status(tailscale.LOGIN, LOGIN_URL)), ("sign in", LOGIN_URL))
    self.assertEqual(tailscale.status_text(_status(tailscale.RUNNING, "100.64.0.7")), ("100.64.0.7", "Connected as 100.64.0.7."))
    self.assertEqual(tailscale.status_text(_status(tailscale.ERROR, "download failed")), ("error", "download failed"))
    self.assertEqual(tailscale.status_text(_status(tailscale.STOPPED))[0], "starting")
    self.assertEqual(tailscale.status_text(_params(**{tailscale.ENABLE_KEY: True}))[0], "starting")

  def test_round_trip_through_the_param(self):
    params = FakeParams()
    tailscale.put_status(cast(Params, params), tailscale.LOGIN, LOGIN_URL)
    self.assertEqual(tailscale.read_status(cast(Params, params)), (tailscale.LOGIN, LOGIN_URL))

    tailscale.put_status(cast(Params, params), tailscale.STARTING)
    self.assertEqual(tailscale.read_status(cast(Params, params)), (tailscale.STARTING, ""))


class TestAuthUrl(unittest.TestCase):
  def test_only_a_login_line_carrying_an_https_url_opens_the_dialog(self):
    # Load-bearing: anything else would render a QR code of something that is not a login URL,
    # and the dialog would sit there unscannable instead of staying closed.
    closed = {
      "running": _status(tailscale.RUNNING, "100.64.0.7"),
      "installing": _status(tailscale.INSTALLING),
      "error": _status(tailscale.ERROR, "download failed"),
      "login holding a non-url": _status(tailscale.LOGIN, "not a url"),
      "login with no detail": _status(tailscale.LOGIN),
      "toggle off": _params(**{tailscale.STATUS_KEY: f"{tailscale.LOGIN} {LOGIN_URL}"}),
    }
    for name, params in closed.items():
      with self.subTest(case=name):
        self.assertEqual(tailscale.auth_url(params), "")

    self.assertEqual(tailscale.auth_url(_status(tailscale.LOGIN, LOGIN_URL)), LOGIN_URL)


class TestBinaries(unittest.TestCase):
  def test_a_system_pair_wins_and_is_never_managed(self):
    # A distro's /usr/bin/tailscale is not ours to overwrite, so upgrade_available() must stay False.
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(tailscale.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
    ):
      self.assertEqual(tailscale.binaries(), ("/usr/bin/tailscale", "/usr/bin/tailscaled"))
      self.assertFalse(tailscale.upgrade_available("9.9.99"))

  def test_a_stale_install_still_runs_while_the_upgrade_is_pending(self):
    # "Can we run?" and "is this the latest release?" are deliberately two questions: collapsing
    # them would take the tunnel down the moment a new release landed, before the replacement landed.
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(tailscale.shutil, "which", return_value=None),
    ):
      _quick_install()
      self.assertFalse(tailscale.upgrade_available(_VERSION))

      with open(tailscale.marker_path(), "w") as f:
        f.write("0.0.0")
      self.assertTrue(tailscale.upgrade_available(_VERSION))
      self.assertTrue(tailscale.installed())
      pair = tailscale.binaries()
      self.assertIsNotNone(pair)
      self.assertEqual(cast(tuple[str, str], pair)[0], os.path.join(tailscale.bin_dir(), "tailscale"))

  def test_nothing_installed_is_not_a_pair(self):
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(tailscale.shutil, "which", return_value=None),
    ):
      self.assertIsNone(tailscale.binaries())
      self.assertFalse(tailscale.installed())
      self.assertFalse(tailscale.upgrade_available(_VERSION))


class TestSupervisorKill(unittest.TestCase):
  """`_stop`'s last resort is the only thing that ends a root tailscaled, so what it matches matters."""

  def _stop_fallback_argv(self) -> tuple[list[str], str]:
    from moonpilot import tailscaled

    proc = mock.Mock()
    proc.wait.side_effect = subprocess.TimeoutExpired("tailscaled", 3)
    with tempfile.TemporaryDirectory() as tmp, mock.patch.object(paths, "data_root", return_value=tmp), mock.patch.object(tailscaled.subprocess, "run") as run:
      socket = tailscale.socket_path()  # read inside the patch: it is derived from data_root()
      tailscaled._stop(proc)
      proc.terminate.assert_called_once()
      self.assertEqual(run.call_count, 1)
      return list(run.call_args[0][0]), socket

  def test_the_fallback_kill_is_scoped_to_our_own_socket(self):
    # Not the binary path: binaries() prefers a pair the OS ships, so `pkill -f <tailscaled path>`
    # would kill a distro's daemon along with ours. The socket path is unique to this fork's child,
    # which is why the pattern must equal it and not a binary — asserted by equality, so this test
    # needs no tailscale on the host it runs on.
    with mock.patch.object(tailscale, "PC", True):
      argv, socket = self._stop_fallback_argv()

    self.assertEqual(argv, ["pkill", "-9", "-f", socket])
    self.assertTrue(socket.endswith("tailscaled.sock"))
    # ...and it is not a path under anything the OS ships, which is what makes it collision-free.
    for binary in ("/usr/bin/tailscale", "/usr/sbin/tailscaled"):
      self.assertFalse(binary in argv)

  def test_the_fallback_kill_is_privileged_on_device(self):
    with mock.patch.object(tailscale, "PC", False):
      argv, _ = self._stop_fallback_argv()
    self.assertEqual(argv[:2], ["sudo", "-n"])

  def test_every_child_the_supervisor_stops_carries_the_socket(self):
    # The pattern only finds the child because daemon_args() puts the socket in its argv; that
    # coupling is what makes the scoped kill work at all.
    with (
      tempfile.TemporaryDirectory() as tmp,
      mock.patch.object(paths, "data_root", return_value=tmp),
      mock.patch.object(tailscale.shutil, "which", return_value=None),
    ):
      _quick_install()
      self.assertTrue(tailscale.socket_path() in tailscale.daemon_args())


class TestWiring(unittest.TestCase):
  def test_the_param_rows_exist(self):
    # A missing row is a device-only UnknownKeyName, raised the first time the supervisor writes
    # status or the panel reads the toggle.
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('"MoonpilotTailscale"' in text)
    self.assertTrue('"MoonpilotTailscaleStatus"' in text)

  def test_the_feature_and_its_process_are_registered(self):
    self.assertTrue(tailscale.ENABLE_KEY in [f.key for f in FEATURES])
    matches = [p for p in procs.MOONPILOT_PROCS if p.name == "tailscaled"]
    self.assertEqual(len(matches), 1)
    self.assertIsInstance(matches[0], procs.PythonProcess)
    self.assertEqual(cast(procs.PythonProcess, matches[0]).module, "moonpilot.tailscaled")


if __name__ == "__main__":
  unittest.main()
