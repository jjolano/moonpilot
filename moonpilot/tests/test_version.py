import re
import unittest
from unittest import mock

from moonpilot import features
from openpilot.common.version import OpenpilotMetadata, get_version

# <upstream semver>-moonpilot.<fork revision>, e.g. "0.11.2-moonpilot.1". See AGENTS.md, Versioning.
VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)-moonpilot\.(\d+)$")


class TestForkVersion(unittest.TestCase):
  def test_shape(self):
    # Catches the merge that takes upstream's version.h wholesale and silently drops the fork's
    # identity from telemetry, both UIs and the API User-Agent -- a clean merge, green lint, and
    # no other check in the tree notices.
    version = get_version()
    match = VERSION_RE.match(version)
    assert match is not None, f"COMMA_VERSION must be <upstream>-moonpilot.<fork revision>, got {version!r}"

    # A counter, not a placeholder.
    assert int(match.group(2)) >= 1, f"fork revision must be >= 1, got {version!r}"

  def test_upstream_part_matches_upstreams_own_accessor(self):
    # Upstream strips the suffix with version.split('-')[0]; the fork's format has to stay
    # compatible with that accessor, not merely with our regex.
    version = get_version()
    match = VERSION_RE.match(version)
    assert match is not None
    metadata = OpenpilotMetadata(version=version, release_notes="", git_commit="", git_origin="",
                                 git_commit_date="", build_style="", is_dirty=False)
    self.assertEqual(metadata.short_version, match.group(1))

  def test_survives_the_update_description_format(self):
    # updated.py builds "version / branch / commit / date" and mici's _split_description
    # (openpilot/selfdrive/ui/mici/layouts/settings/software.py) requires exactly four parts
    # split on " / ". The version must not introduce a fifth, or that panel renders blank.
    version = get_version()
    description = f"{version} / master / abc1234 / Aug 12"
    parts = [p.strip() for p in description.split(" / ")]

    assert len(parts) == 4, f"version upset the update description format: {description!r}"
    self.assertEqual(parts[0], version)

  def test_is_fork_build_agrees_with_the_marker(self):
    # is_fork_build() is fork identity that gates behavior: selfdrived reads it to keep upstream's
    # "WARNING: This branch is not tested" banner off every build of this tree, because comma tests
    # comma's branches and never ours (AGENTS.md, Features). A merge that drops the marker — or a
    # version.h taken from upstream wholesale — would put that warning back on the road, so the two
    # are pinned together here.
    self.assertTrue(features.is_fork_build())

    with mock.patch.object(features, "version", return_value="0.11.2"):
      self.assertFalse(features.is_fork_build())


if __name__ == "__main__":
  unittest.main()
