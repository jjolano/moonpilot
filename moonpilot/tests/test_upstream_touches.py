import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "upstream/master"

# moonpilot owns these outright.
FORK_OWNED = ("moonpilot/", "AGENTS.md", "CLAUDE.md")

# Upstream files moonpilot may modify, and the seam each one carries.
ALLOWED = {
  "SConstruct": "moonpilot/SConscript registration",
  "pyproject.toml": "moonpilot in the editable install",
  "scripts/lint/lint.sh": "lint moonpilot/",
  "tools/test_runner.py": "test moonpilot/ by default",
  "openpilot/cereal/custom.capnp": "MoonpilotState struct",
  "openpilot/cereal/log.capnp": "moonpilotState event field",
  "openpilot/common/params_keys.h": "moonpilot/params_keys.h include",
  "openpilot/common/version.h": "fork version string",
  "openpilot/selfdrive/ui/layouts/home.py": "brand string",
  "openpilot/selfdrive/ui/layouts/settings/settings.py": "moonpilot panel",
  "openpilot/selfdrive/ui/mici/layouts/home.py": "brand string",
  "openpilot/selfdrive/ui/mici/layouts/settings/settings.py": "moonpilot panel",
  "openpilot/system/manager/process_config.py": "MOONPILOT_PROCS",
}


def git(*args: str) -> subprocess.CompletedProcess:
  return subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True, check=False)


class TestUpstreamTouches(unittest.TestCase):
  def test_touches_stay_in_seams(self):
    if git("rev-parse", "--verify", "--quiet", UPSTREAM).returncode != 0:
      self.skipTest(f"no {UPSTREAM} ref; run: git fetch upstream")

    base = git("merge-base", "HEAD", UPSTREAM).stdout.strip()
    changed = git("diff", "--name-only", base).stdout.split()  # committed + working tree
    touched = {f for f in changed if not f.startswith(FORK_OWNED)}
    unexpected = sorted(touched - ALLOWED.keys())
    assert not unexpected, "upstream files touched outside the moonpilot seams:\n  " + "\n  ".join(unexpected)


if __name__ == "__main__":
  unittest.main()
