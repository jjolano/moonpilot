"""Guards AGENTS.md against drift.

The file names fork files by path, and the next agent follows those names without checking them.
A rename that misses the doc leaves instructions pointing at something that is gone, which is only
discovered by whoever follows them, mid-task. This is the cheap half of that: every fork path the
doc mentions must exist.

Upstream paths are deliberately not checked. Upstream renames files on its own schedule, and the
line references this doc makes to upstream drift with them, so pinning those would turn every
merge into a doc failure for no signal the fork can act on.

The two forked submodules carry their own context files for the same reason, and the second test
here holds them to the one shape that keeps them from becoming a second copy of this file: they
must exist, they must import this file, and they must stay short.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Backticked `moonpilot/...` tokens. Bare directory mentions end in "/" and are skipped.
FORK_PATH = re.compile(r"`(moonpilot/[A-Za-z0-9_./-]+)`")

# The forked submodules, each its own repository, so a session started inside one walks up to that
# repo's root and never reaches the superproject's AGENTS.md
SUBMODULES = ("panda", "opendbc_repo")
# How long a pointer may be. The root file is ~300 lines; a submodule file approaching that is a
# copy, which is exactly what these are not allowed to become.
MAX_POINTER_LINES = 44


class TestAgentsMd(unittest.TestCase):
  def test_every_named_fork_path_exists(self):
    text = (ROOT / "AGENTS.md").read_text()
    named = {p for p in FORK_PATH.findall(text) if not p.endswith("/")}

    # Guards the guard: if the extractor or the doc's formatting changes shape, this test would
    # otherwise pass by finding nothing at all.
    self.assertTrue(named, "no fork paths found in AGENTS.md; the extractor is broken")

    missing = sorted(p for p in named if not (ROOT / p).exists())
    assert not missing, "AGENTS.md names fork paths that do not exist:\n  " + "\n  ".join(missing)


class TestSubmoduleContextFiles(unittest.TestCase):
  def test_each_fork_carries_a_pointer_not_a_copy(self):
    for name in SUBMODULES:
      with self.subTest(submodule=name):
        context = ROOT / name / "AGENTS.md"
        if not context.exists():
          # The submodules are separate clones; a checkout without them is not a doc failure
          self.skipTest(f"{name} is not checked out")

        text = context.read_text()
        # The import is what makes a session inside the submodule inherit the fork's rules, and
        # the path is relative to the importing file, i.e. the superproject's own AGENTS.md
        self.assertTrue("@../AGENTS.md" in text, f"{name}/AGENTS.md does not import the superproject's rules")
        self.assertLessEqual(len(text.splitlines()), MAX_POINTER_LINES, f"{name}/AGENTS.md is long enough to be a second copy of the real one")
        # A copy would restate the parts that change with upstream; a pointer names them at most
        self.assertTrue("COMMA_VERSION" not in text, f"{name}/AGENTS.md restates versioning")
        self.assertTrue("| Upstream file | Seam |" not in text, f"{name}/AGENTS.md restates the seam table")

        # The symlink is the same convention the superproject uses, so a tool looking for either
        # name finds the same bytes
        claude = ROOT / name / "CLAUDE.md"
        self.assertTrue(claude.is_symlink(), f"{name}/CLAUDE.md is not a symlink")
        self.assertEqual(claude.resolve(), context.resolve(), f"{name}/CLAUDE.md does not point at its AGENTS.md")


if __name__ == "__main__":
  unittest.main()
