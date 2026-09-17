"""Guards AGENTS.md against drift.

The file names fork files by path, and the next agent follows those names without checking them.
A rename that misses the doc leaves instructions pointing at something that is gone, which is only
discovered by whoever follows them, mid-task. This is the cheap half of that: every fork path the
doc mentions must exist.

Upstream paths are deliberately not checked. Upstream renames files on its own schedule, and the
line references this doc makes to upstream drift with them, so pinning those would turn every
merge into a doc failure for no signal the fork can act on.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Backticked `moonpilot/...` tokens. Bare directory mentions end in "/" and are skipped.
FORK_PATH = re.compile(r"`(moonpilot/[A-Za-z0-9_./-]+)`")


class TestAgentsMd(unittest.TestCase):
  def test_every_named_fork_path_exists(self):
    text = (ROOT / "AGENTS.md").read_text()
    named = {p for p in FORK_PATH.findall(text) if not p.endswith("/")}

    # Guards the guard: if the extractor or the doc's formatting changes shape, this test would
    # otherwise pass by finding nothing at all.
    self.assertTrue(named, "no fork paths found in AGENTS.md; the extractor is broken")

    missing = sorted(p for p in named if not (ROOT / p).exists())
    assert not missing, "AGENTS.md names fork paths that do not exist:\n  " + "\n  ".join(missing)


if __name__ == "__main__":
  unittest.main()
