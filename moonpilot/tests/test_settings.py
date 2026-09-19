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

from moonpilot.features import LATERAL_ENGAGE, Feature
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
    with mock.patch.object(settings_mici, "ui_state", SimpleNamespace(CP=_cp(brand="hyundai"))):
      self.assertEqual(settings_mici._unavailable_value(LATERAL_ENGAGE), "Toyota, Lexus, Honda or Volkswagen only")

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
