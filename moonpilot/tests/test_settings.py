"""What the two settings panels put in front of the driver: the mici value line, and the tizi
description.

Both panels are push-only in the widget sense -- the mici row's value and the tizi row's
description are computed by plain functions of a feature and the car -- so the check is on the
string, which is the only thing either panel exposes. The reason it is worth pinning: the mici
panel's line is derived from two gates that can each be missing, and a regression that drops one
of them labels a working feature "unavailable" on every row, which no import or type check would
notice.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from opendbc.car.structs import car

from moonpilot import models
from moonpilot.features import FEATURES, GROUPS, LATERAL_ENGAGE, Feature
from moonpilot.ui import settings, settings_mici

# A name no distribution can provide, so `available()` is False without depending on the
# environment (see test_deps.py, which pins the same contract from the other side).
ABSENT = "moonpilot_no_such_package_anywhere"


def _feature(requires: tuple[str, ...] = ()) -> Feature:
  return Feature(key="MoonpilotTest", title="test", description="test description", requires=requires)


def _cp(brand="toyota", pcm_cruise=True, flags=0, passive=False):
  """A real CarParams message, the same shape test_engage.py builds for the car gate."""
  cp = car.CarParams.new_message()
  cp.brand = brand
  cp.pcmCruise = pcm_cruise
  cp.passive = passive
  cp.flags = int(flags)
  return cp


class TestMiciValue(unittest.TestCase):
  """`_unavailable_value` is the mici row's sub-label, the only place a reason reaches the driver."""

  def test_available_feature_on_supported_car_is_blank(self):
    # A bug here is the regression: every live row reading "unavailable".
    with mock.patch.object(settings_mici, "ui_state", SimpleNamespace(CP=_cp())):
      self.assertEqual(settings_mici._unavailable_value(_feature()), "")

  def test_missing_dependency_reads_unavailable(self):
    # A bug here is a row whose feature cannot run saying nothing about it.
    with mock.patch.object(settings_mici, "ui_state", SimpleNamespace(CP=_cp())):
      self.assertEqual(settings_mici._unavailable_value(_feature((ABSENT,))), "unavailable")

  def test_unsupported_car_carries_the_cars_own_reason(self):
    # A bug here is the car gate falling back to the placeholder, which tells the driver nothing.
    # A dashcam-mode car is the case that survives every brand being in scope: the feature is off
    # there because the panda has no car config to arm, not because of a dependency.
    with mock.patch.object(settings_mici, "ui_state", SimpleNamespace(CP=_cp(passive=True))):
      self.assertEqual(settings_mici._unavailable_value(LATERAL_ENGAGE), "not in dashcam mode")

  def test_unloaded_car_params_is_not_a_verdict(self):
    # The panel renders before CarParams lands; a bug here grays out and mislabels every row at boot.
    with mock.patch.object(settings_mici, "ui_state", SimpleNamespace(CP=None)):
      self.assertEqual(settings_mici._unavailable_value(LATERAL_ENGAGE), "")


class TestTiziDescription(unittest.TestCase):
  """The tizi panel has room to wrap, so the same two gates ride the description instead."""

  def test_available_feature_shows_its_own_description(self):
    # A bug here is every row's dialog prefixed with a reason it does not have.
    with mock.patch.object(settings, "ui_state", SimpleNamespace(CP=_cp())):
      feature = _feature()
      self.assertEqual(settings._description(feature), feature.description)

  def test_missing_dependency_is_bolded_above_the_description(self):
    # A bug here is a driver opening a dead row and not being told which package is missing.
    with mock.patch.object(settings, "ui_state", SimpleNamespace(CP=_cp())):
      text = settings._description(_feature((ABSENT,)))
      self.assertTrue(text.startswith("<b>"))
      self.assertTrue(ABSENT in text)
      self.assertTrue("test description" in text)


if __name__ == "__main__":
  unittest.main()

class TestGrouping(unittest.TestCase):
  """The panel's pages are the feature table's groups, so these two properties are what keeps a
  regroup from losing a row: every feature is on exactly one page, and the flat `FEATURES` the old
  panel iterated is the flattening of the pages."""

  def test_every_feature_sits_on_exactly_one_page(self):
    grouped = [feature for group in GROUPS for feature in group.features]
    self.assertEqual(len(grouped), len(set(grouped)), "a feature is on two pages")
    self.assertEqual(set(grouped), set(FEATURES), "FEATURES and the groups disagree")
    self.assertEqual(tuple(grouped), FEATURES, "FEATURES is not the flattening of GROUPS")

  def test_every_page_says_what_its_rows_have_in_common(self):
    # A page with no description is a page whose rows have no reason to be together: the description
    # is the only thing that tells a driver why they are looking at these toggles.
    for group in GROUPS:
      with self.subTest(group=group.title):
        self.assertTrue(group.features)
        self.assertTrue(group.description)

  def test_the_pages_are_the_three_the_panel_shows(self):
    self.assertEqual([group.title for group in GROUPS], ["steering", "speed & distance", "device"])


class TestModelsStrings(unittest.TestCase):
  """What the models panes print, which both trees share. The strings are the contract between the
  two panels and `moonpilot/models.py`, so they are pinned here and not in either tree."""

  def test_a_bundled_selection_reads_as_stock(self):
    params = SimpleNamespace(get=lambda key, block=False, return_default=False: "" if key.startswith("MoonpilotModels") else None)
    self.assertEqual(models.model_label(params, models.DRIVING), models.BUNDLED_LABEL)

  def test_a_job_with_progress_names_both_numbers(self):
    job = {"phase": "downloading", "received": 5 * 1024**2, "total": 10 * 1024**2, "op": "install", "recipe": "a" * 64}
    text = models.job_text(job)
    self.assertTrue("downloading" in text)
    self.assertTrue(models.human_size(5 * 1024**2) in text)
    self.assertTrue(models.human_size(10 * 1024**2) in text)
    self.assertEqual(models.job_text({"phase": "building"}), "building")
    self.assertEqual(models.job_text(None), "")

  def test_a_catalog_with_no_revision_reads_as_never(self):
    self.assertEqual(models.catalog_text({"revision": "", "generated_at": "", "error": None}), "never")
    self.assertEqual(models.catalog_text({"revision": "a" * 64, "generated_at": "2026-09-19T06:48:29Z", "error": "boom"}), "stale")
    self.assertEqual(models.catalog_text({"revision": "a" * 64, "generated_at": "2026-09-19T06:48:29Z", "error": None}), "2026-09-19")

  def test_the_protocol_label_is_shorter_than_the_id(self):
    for proto in models.PROTOCOLS:
      with self.subTest(protocol=proto.id):
        self.assertNotEqual(models.protocol_label(proto.id), proto.id)


if __name__ == "__main__":
  unittest.main()
