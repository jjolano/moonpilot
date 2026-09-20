import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "upstream/master"

# moonpilot owns these outright. `.agents/` has never existed upstream -- it is the fork's own
# agent skills -- so a path under it is not an upstream file by construction. If upstream ever
# grows one, this line is what would have to move.
FORK_OWNED = ("moonpilot/", ".agents/", "AGENTS.md", "CLAUDE.md")

# The prose marker. Deliberately not the bare fork name: `'moonpilotState'` is a service name in
# plannerd's subscription list, so matching `moonpilot` would let a region whose real marker was
# dropped -- by a formatter, say -- pass on a data name that upstream code would never contain.
MARKER = "moonpilot seam"
# The one path whose fork regions cannot carry prose: git-config only takes `#` on a line of its own,
# and `-U0` gives that line its own region, so the value it explains stands alone. It names the fork
# in the URL instead. AGENTS.md's Seam markers section is where the convention lives.
VALUE_MARKED = {".gitmodules": ("moonpilot-",)}
SUBMODULES = ("panda", "opendbc_repo")

# The two checks docs/SAFETY.md forbids nerfing unconditionally, as path prefixes the fork may never
# gain a seam in -- driver monitoring, and the excessive-actuation checks. AGENTS.md states the
# claim; this is what holds it. The guard below reads `ALLOWED`, so it catches the drift the file
# check cannot: a future fork adding a row here would have to delete this constant and that passage
# together, which is a decision rather than a merge. The paths are also asserted to still exist, so
# an upstream rename fails loudly instead of quietly leaving the guard pointed at nothing.
NEVER_ALLOWED = (
  "openpilot/selfdrive/monitoring/",
  "openpilot/selfdrive/selfdrived/helpers.py",
)

# Upstream files moonpilot may modify, and the seam each one carries.
ALLOWED = {
  "launch_chffrplus.sh": "boot-success rollback for the finalized swap",
  ".gitmodules": "forked opendbc/panda submodules",
  "SConstruct": "moonpilot/SConscript registration",
  "pyproject.toml": "moonpilot in the editable install",
  "scripts/lint/lint.sh": "lint moonpilot/",
  "tools/test_runner.py": "test moonpilot/ by default",
  "openpilot/cereal/custom.capnp": "MoonpilotState struct",
  "openpilot/cereal/log.capnp": "moonpilotState event field; the PandaState lateral-controls field",
  "openpilot/cereal/services.py": "moonpilotState service row",
  "openpilot/common/params_keys.h": "moonpilot/params_keys.h include",
  "openpilot/common/version.h": "fork version string",
  "openpilot/selfdrive/car/card.py": "lateral engagement safety param",
  "openpilot/selfdrive/controls/plannerd.py": "moonpilotState subscription; moonpilot longitudinal planner",
  "openpilot/selfdrive/controls/lib/longitudinal_planner.py": "lead danger factor from moonpilot",
  "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py": "lead_danger_factor kwarg",
  "openpilot/selfdrive/controls/controlsd.py": "moonpilot torque lateral controller, response-aligned curvature, acceleration controller, and actuator gate",
  "openpilot/selfdrive/pandad/pandad.cc": "lateral-controls health flag",
  "openpilot/selfdrive/selfdrived/selfdrived.py": "lateral engagement events and panda cross-check, and the fork's startup alert",
  "openpilot/selfdrive/selfdrived/events.py": "the half-engagement banner",
  "openpilot/selfdrive/test/process_replay/process_replay.py": "moonpilotState in plannerd pubs",
  "openpilot/selfdrive/ui/ui_state.py": "moonpilotState subscription",
  "openpilot/selfdrive/ui/onroad/augmented_road_view.py": "half-engaged border color; offroad-mode hold gesture",
  "openpilot/selfdrive/ui/onroad/model_renderer.py": "lead path draw (tizi)",
  "openpilot/selfdrive/ui/mici/onroad/augmented_road_view.py": "offroad-mode hold gesture (mici)",
  "openpilot/selfdrive/ui/mici/onroad/model_renderer.py": "lead path draw (mici)",
  "openpilot/selfdrive/ui/layouts/home.py": "brand string",
  "openpilot/selfdrive/ui/layouts/settings/settings.py": "moonpilot panel",
  "openpilot/selfdrive/ui/mici/layouts/home.py": "brand string",
  "openpilot/selfdrive/ui/mici/layouts/settings/settings.py": "moonpilot panel",
  "openpilot/system/hardware/hardwared.py": "offroad mode member in onroad_conditions",
  "openpilot/system/manager/manager.py": "the model store's boot selection commit",
  "openpilot/selfdrive/modeld/modeld.py": "the fork's driving model runtime; low-speed turn-desire hook after DesireHelper.update()",
  "openpilot/selfdrive/modeld/dmonitoringmodeld.py": "the fork's monitoring model runtime",
  "openpilot/system/manager/process_config.py": "MOONPILOT_PROCS",
  # The fork's own opendbc/panda: recorded as submodule pointer moves, so the paths of the
  # submodules themselves are the upstream paths here.
  "opendbc_repo": "forked safety layer, on jjolano/moonpilot-opendbc",
  "panda": "forked health flag, on jjolano/moonpilot-panda",
}


def git(*args: str) -> subprocess.CompletedProcess:
  return subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True, check=False)


def hunks(diff_text: str) -> list[tuple[str, list[str], list[str]]]:
  """One `(path, added, removed)` per hunk of a `git diff -U0`.

  A hunk is one region of fork work -- a struct body, a method, a rewritten call -- and one marker
  per region is the convention AGENTS.md states. A hunk that replaces lines carries both lists; one
  that only adds upstream behavior carries additions alone, and one that only removes it carries
  removals alone. All three are fork edits, which is why both directions are checked below.

  The path comes from whichever side names a real file, so a deleted file is attributed to the path
  it had -- `+++ /dev/null` there names no file, and a whole deleted upstream file would otherwise
  slip past both guards as an unknown path. A file the fork adds is the mirror case, resolved the
  same way.
  """
  out: list[tuple[str, list[str], list[str]]] = []
  path = old_path = ""
  for line in diff_text.splitlines():
    if line.startswith("--- "):
      old_path = line[4:].strip().removeprefix("a/")
    elif line.startswith("+++ "):
      new_path = line[4:].strip().removeprefix("b/")
      path = old_path if new_path == "/dev/null" else new_path
    elif line.startswith("@@"):
      out.append((path, [], []))
    elif out and line.startswith("+") and not line.startswith("+++"):
      out[-1][1].append(line[1:])
    elif out and line.startswith("-") and not line.startswith("---"):
      out[-1][2].append(line[1:])
  return out


def unmarked_hunks(diff_text: str, files) -> list[tuple[str, list[str]]]:
  """Regions added to an upstream file with no marker in them.

  A region passes on the prose marker, or -- in the paths `VALUE_MARKED` names -- on the value form
  its syntax leaves room for. Submodules are skipped: their diff is the recorded SHA, one line that
  cannot carry either.
  """
  bad: list[tuple[str, list[str]]] = []
  for path, added, _ in hunks(diff_text):
    if path not in files or path in SUBMODULES or not added:
      continue
    tokens = (MARKER, *VALUE_MARKED.get(path, ()))
    if not any(token in line.lower() for token in tokens for line in added):
      bad.append((path, added))
  return bad


def deleted_paths(base: str) -> set[str]:
  """Paths this diff deletes outright.

  The removal check exempts these and nothing else. A whole deletion is not the silent edit that
  check exists to catch -- git labels it `deleted file mode`, and the file-name guard is what
  sanctions it, since the path still has to be an ALLOWED one. A *partial* removal leaves the file
  in place, is an ordinary-looking edit, and is refused as before.
  """
  return set(git("diff", "--name-only", "--diff-filter=D", base).stdout.split())


def silent_removal_hunks(diff_text: str, files, skip=()) -> list[tuple[str, list[str]]]:
  """Regions that *remove* upstream behavior from an allowed file and add nothing back.

  The marker check reads additions, and a hunk with no additions has no line to carry a marker, so a
  pure deletion of upstream behavior -- a dropped alert, a dropped check -- would pass both guards
  while being exactly the upstream edit they exist to catch. `skip` is the whole-file deletions above,
  and it is deliberately not a general escape: the fork replaces rather than deletes in every path
  that survives, which is what makes this assertion affordable.
  """
  return [
    (path, removed) for path, added, removed in hunks(diff_text) if path in files and path not in SUBMODULES and path not in skip and removed and not added
  ]


class TestUpstreamTouches(unittest.TestCase):
  def test_touches_stay_in_seams(self):
    if git("rev-parse", "--verify", "--quiet", UPSTREAM).returncode != 0:
      self.skipTest(f"no {UPSTREAM} ref; run: git fetch upstream")

    base = git("merge-base", "HEAD", UPSTREAM).stdout.strip()
    # Deletions are listed, not filtered out: a deleted path has to be an ALLOWED one, which is what
    # keeps the guard on a fork that deletes an upstream file it never hooked, and what keeps the row
    # of a hooked file alive while the deletion is still in this diff. The removal check exempts the
    # whole-file case explicitly -- see `deleted_paths` -- so the two guards agree rather than
    # deadlock: keep the row and the deletion passes both, drop it and this check fails.
    changed = git("diff", "--name-only", base).stdout.split()  # committed + working tree
    touched = {f for f in changed if not f.startswith(FORK_OWNED)}
    unexpected = sorted(touched - ALLOWED.keys())
    assert not unexpected, "upstream files touched outside the moonpilot seams:\n  " + "\n  ".join(unexpected)

  def test_the_unconditional_safety_paths_stay_unreachable(self):
    """Driver monitoring and the excessive-actuation checks are the two prohibitions that attach to
    every fork rather than to editing `opendbc/safety/`, and this fork honors them by absence.

    Both halves matter. The paths must exist, or an upstream rename retires the guard while leaving
    it green; and no `ALLOWED` row may reach them, which is the drift the file-name check above
    cannot see -- that check reads a diff, this one reads the table that would sanction the diff.
    `openpilot/selfdrive/modeld/dmonitoringmodeld.py` is deliberately not in this tuple: it is the
    monitoring *model's* seam, and the marketplace substituting a network is a driver's choice with
    its own admission gate, not a fork edit to the monitoring policy. AGENTS.md says so in prose.
    """
    for path in NEVER_ALLOWED:
      with self.subTest(path=path):
        self.assertTrue((ROOT / path.rstrip("/")).exists(), f"{path} is gone upstream; re-point this guard")
    reaching = sorted(row for row in ALLOWED if any(row.startswith(path) for path in NEVER_ALLOWED))
    assert not reaching, "an ALLOWED row reaches a path docs/SAFETY.md forbids nerfing:\n  " + "\n  ".join(reaching)


class TestSeamMarkers(unittest.TestCase):
  """Every region of upstream code the fork edits -- added, replaced or removed -- is fork-owned.

  `TestUpstreamTouches` compares file *names*, so an edit inside an already-allowed file passes it
  silently: a formatter run over an upstream file rewrites dozens of lines, merges clean, and is
  invisible to every guard in this tree. This is the half that reads the lines. One marker per
  region -- not per line, and not necessarily at the region's head, since a rewritten multi-line
  statement can only carry a comment at its end and a git-config `url =` cannot carry one at all
  (AGENTS.md, Seam markers) -- is what makes a markerless region either a missing marker or an
  upstream edit that should not be there, which is the only question worth asking of a diff. The
  removal-only case is the same question with no line to answer it on, and gets its own test.
  """

  def test_every_added_region_carries_the_marker(self):
    diff = self.tree_diff()
    bad = unmarked_hunks(diff, ALLOWED)
    assert not bad, "fork lines in upstream files with no moonpilot marker:\n  " + "\n  ".join(f"{path}: {lines[0].strip()[:90]}" for path, lines in bad)

  def test_no_region_only_removes_upstream_behavior(self):
    # A hunk with no additions has no line to carry a marker, so a pure deletion is invisible to the
    # check above -- both halves of the same question ("is this edit fork-owned?") answer yes by
    # default. Whole-file deletions are the one shape excused, and the file-name guard is what keeps
    # those honest: the path it lists has to be an ALLOWED one. Measured clean when this was written,
    # and the number lives here rather than in AGENTS.md so there is one copy of it: 61 regions in
    # allowed files, 21 carrying removals, none of them removal-only.
    base = git("merge-base", "HEAD", UPSTREAM).stdout.strip()
    bad = silent_removal_hunks(self.tree_diff(), ALLOWED, skip=deleted_paths(base))
    assert not bad, "upstream behavior removed with nothing added back:\n  " + "\n  ".join(f"{path}: {lines[0].strip()[:90]}" for path, lines in bad)

  @staticmethod
  def tree_diff() -> str:
    if git("rev-parse", "--verify", "--quiet", UPSTREAM).returncode != 0:
      raise unittest.SkipTest(f"no {UPSTREAM} ref; run: git fetch upstream")
    base = git("merge-base", "HEAD", UPSTREAM).stdout.strip()
    return git("diff", "-U0", base).stdout  # committed + working tree

  def test_an_unmarked_region_is_caught(self):
    # Eight regions, in the shapes the tree actually has: the marker at a new block's head (first,
    # fifth), the marker ending a rewritten multi-line statement -- the common case, since `#` cannot
    # sit inside a call's parentheses (second); two ways a region can be mistaken for marked -- a
    # data name that looks like one, and the value form (third, seventh); two regions that dropped
    # upstream code and added nothing back, the last of them the whole file (sixth, eighth); and one
    # that simply forgot (fourth).
    diff = (
      "+++ b/openpilot/foo.py\n"
      + "@@ -1 +1,2 @@\n"
      + "+def f():  # moonpilot seam, see AGENTS.md\n"
      + "+  return 1\n"
      + "@@ -10 +10,2 @@\n"
      + "+sm = messaging.SubMaster(['carState', 'radarState',\n"
      + "+                          'moonpilotState'],  # moonpilot seam, see AGENTS.md\n"
      + "@@ -20 +20 @@\n"
      + "+sm = messaging.SubMaster(['carState', 'radarState', 'moonpilotState'])\n"
      + "@@ -40 +40,2 @@\n"
      + "+def g():\n"
      + "+  return 2\n"
      + "@@ -50 +50 @@\n"
      + "-def upstream_check():\n"
      + "+def upstream_check():  # moonpilot seam, see AGENTS.md\n"
      + "@@ -60 +60 @@\n"
      + "-def upstream_alert():\n"
      + "+++ b/.gitmodules\n"
      + "@@ -3 +4 @@\n"
      + "+  url = ../moonpilot-panda.git\n"
      + "diff --git a/openpilot/gone.py b/openpilot/gone.py\n"
      + "deleted file mode 100644\n"
      + "--- a/openpilot/gone.py\n"
      + "+++ /dev/null\n"
      + "@@ -1 +0,0 @@\n"
      + "-def upstream_gone():\n"
    )
    hole, forgot = ("sm = messaging.SubMaster(['carState', 'radarState', 'moonpilotState'])", "def g():")

    # Every hunk is a region, replaced or not: six in the Python file, one in the git config, one
    # deleted file. The last is the case `+++ /dev/null` would otherwise hide: it resolves to the
    # path the file had, which is what puts it in scope for the removal check below.
    self.assertEqual([path for path, _, _ in hunks(diff)], ["openpilot/foo.py"] * 6 + [".gitmodules", "openpilot/gone.py"])
    self.assertEqual([len(added) + len(removed) for _, added, removed in hunks(diff)], [2, 2, 1, 2, 2, 1, 1, 1])

    # The third region is the masking hole the prose token closes: `'moonpilotState'` is a service
    # name, not a marker, so a reformat that moved the real marker off this line has to fail. The
    # fourth has a marker nowhere at all.
    self.assertEqual([lines[0].strip() for _, lines in unmarked_hunks(diff, {"openpilot/foo.py"})], [hole, forgot])

    # The url region passes only where the value form is accepted, which is what the per-path map is
    # for: prose cannot live in a git-config value line, so `.gitmodules` says it in the URL.
    self.assertEqual(unmarked_hunks(diff, {".gitmodules"}), [])
    self.assertEqual([lines[0].strip() for _, lines in unmarked_hunks(diff, {".gitmodules", "openpilot/foo.py"})], [hole, forgot])
    self.assertEqual(unmarked_hunks(diff, {"openpilot/other.py"}), [])

    # The sixth and eighth regions, and only those: the fifth rewrites an upstream function and adds
    # a marked line, which is a fork edit like any other, and the seventh adds one. A deleted file is
    # in this set, or its path would be the one place the two guards disagreed. (`removed` holds the
    # text with the diff's `-` stripped.)
    both = {"openpilot/foo.py", ".gitmodules", "openpilot/gone.py"}
    self.assertEqual(
      [lines[0].strip() for _, lines in silent_removal_hunks(diff, both)],
      ["def upstream_alert():", "def upstream_gone():"],
    )
    self.assertEqual(silent_removal_hunks(diff, {"openpilot/other.py"}), [])

    # A whole-file deletion is the one shape excused, and the exemption is per path: the deleted file
    # drops out while the partial removal in the surviving file stays, and skipping a *surviving* path
    # leaves the deleted one flagged rather than switching the check off. The file-name guard is what
    # keeps the exemption honest -- the deleted path still has to be an ALLOWED one.
    self.assertEqual([lines[0].strip() for _, lines in silent_removal_hunks(diff, both, skip={"openpilot/gone.py"})], ["def upstream_alert():"])
    self.assertEqual([lines[0].strip() for _, lines in silent_removal_hunks(diff, both, skip={"openpilot/foo.py"})], ["def upstream_gone():"])


if __name__ == "__main__":
  unittest.main()
