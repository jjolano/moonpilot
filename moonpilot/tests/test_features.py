"""The feature registry against the compiled param table, and the predicates every seam and panel reads."""

import re
import unittest
from pathlib import Path

from openpilot.common.params import ParamKeyType, Params
from openpilot.common.realtime import Priority

from moonpilot.features import FEATURES, GROUPS, car_reason, choice, enabled, in_force, missing_modules, ui_priority, value, wanted
from moonpilot.tests.fakes import ABSENT, PRESENT, feature as make_feature, params

ROOT = Path(__file__).resolve().parents[2]


class TestRegistryTable(unittest.TestCase):
  def test_every_row_is_declared_with_its_type_and_a_default(self):
    table = Params()
    for feature in FEATURES:
      with self.subTest(key=feature.key):
        self.assertEqual(table.get_type(feature.key), ParamKeyType.INT if feature.choices else ParamKeyType.BOOL)
        self.assertIsNotNone(table.get_default_value(feature.key))

  def test_every_feature_is_on_one_page_once(self):
    keys = [f.key for group in GROUPS for f in group.features]
    self.assertEqual(len(keys), len(set(keys)))

  def test_a_choice_row_ships_off(self):
    """A choice row's `0` is its off choice, so its declared default has to be `0`: one shipping at another index would
    silently turn a feature on, and the tables are read by index."""
    table = Params()
    for feature in FEATURES:
      if feature.choices:
        with self.subTest(key=feature.key):
          self.assertEqual(table.get_default_value(feature.key), 0)

  def test_a_choice_rows_first_value_is_off(self):
    for feature in FEATURES:
      if feature.choices:
        with self.subTest(key=feature.key):
          self.assertFalse(feature.values[0])


  def test_every_moonpilot_key_the_code_names_is_declared(self):
    """A `Moonpilot*` string in fork code that no row declares raises `UnknownKeyName` on device -- in the panel, in
    the planner, in whichever process reads it -- and reads as nothing at all in a bare test. The header's own
    documentation examples are the only names allowed to be undeclared."""
    table = Params()
    documented = {"MoonpilotExample", "MoonpilotToggle"}  # params_keys.h's format sample
    named = set()
    for path in sorted((ROOT / "moonpilot").rglob("*.py")):
      if "/tests/" in str(path) or "params_keys" in path.name:
        continue
      named.update(re.findall(r"[\"'](Moonpilot[A-Za-z]+)[\"']", path.read_text()))
    undeclared = sorted(name for name in named - documented if name not in {f.key for f in FEATURES} and not _declared(table, name))
    self.assertEqual(undeclared, [])


def _declared(table: Params, key: str) -> bool:
  try:
    table.check_key(key)
  except Exception:
    return False
  return True


class TestChoice(unittest.TestCase):
  ROW = make_feature(choices=(("off", 0.0), ("a", 0.1), ("b", 0.2)))

  def test_the_selected_index_picks_its_value(self):
    self.assertEqual(value(self.ROW, params(overrides={self.ROW.key: 2})), 0.2)

  def test_an_out_of_range_or_missing_index_is_off(self):
    for index in (-1, 3, None, "1"):
      with self.subTest(index=index):
        self.assertEqual(choice(self.ROW, params(overrides={self.ROW.key: index})), 0)
        self.assertEqual(value(self.ROW, params(overrides={self.ROW.key: index})), 0.0)

  def test_a_missing_dependency_is_off_whatever_is_chosen(self):
    row = make_feature(requires=(ABSENT,), choices=self.ROW.choices)
    self.assertEqual(value(row, params(overrides={row.key: 2})), 0.0)


class TestGate(unittest.TestCase):
  def test_a_missing_dependency_turns_the_feature_off_but_keeps_the_drivers_choice(self):
    feature = make_feature(requires=(PRESENT, ABSENT))
    self.assertEqual(missing_modules(feature), (ABSENT,))
    self.assertTrue(wanted(feature, params(True)))
    self.assertFalse(enabled(feature, params(True)))

  def test_a_present_dependency_leaves_the_decision_to_the_driver(self):
    feature = make_feature(requires=(PRESENT,))
    self.assertTrue(enabled(feature, params(True)))
    self.assertFalse(enabled(feature, params(False)))

  def test_a_car_gate_holds_the_feature_off_on_that_car_only(self):
    feature = make_feature(car=lambda CP: None if CP == "fits" else "not this car")
    self.assertTrue(in_force(feature, params(True), "fits"))
    self.assertFalse(in_force(feature, params(True), "other"))
    self.assertFalse(in_force(feature, params(False), "fits"))
    self.assertEqual(car_reason(feature, "other"), "not this car")

  def test_no_carparams_yet_is_not_a_verdict_on_the_car(self):
    for feature in FEATURES:
      with self.subTest(key=feature.key):
        self.assertIsNone(car_reason(feature, None))


class TestUiPriority(unittest.TestCase):
  def test_rendering_yields_to_lead_processing_without_raising_an_already_lower_priority(self):
    for upstream in (1, Priority.CTRL_LOW - 1, Priority.CTRL_LOW, Priority.CTRL_HIGH):
      with self.subTest(upstream=upstream):
        self.assertLess(ui_priority(upstream), Priority.CTRL_LOW)
        self.assertLessEqual(ui_priority(upstream), upstream)


if __name__ == "__main__":
  unittest.main()
