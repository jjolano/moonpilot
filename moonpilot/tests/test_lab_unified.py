import inspect
import unittest
from unittest import mock

import moonpilot.longitudinal as longitudinal
from moonpilot.lab.unified import unified_lead_accel
from moonpilot.longitudinal import MOONPILOT_CRAWL_REST, MOONPILOT_STOP_REST, lead_accel
from moonpilot.tests.test_longitudinal import _fly


class TestUnifiedLead(unittest.TestCase):
  def test_it_is_a_drop_in_for_lead_accel(self):
    """`policy` calls `lead_accel` by global name, so the swap is only honest if the arguments line up."""
    self.assertEqual(inspect.signature(unified_lead_accel), inspect.signature(lead_accel))

  def test_the_law_stops_behind_a_stopped_lead_and_a_braking_one_without_contact(self):
    with mock.patch.object(longitudinal, "lead_accel", unified_lead_accel):
      stopped = _fly(25.0, 90.0, 200.0, 0.0, lambda t: 0.0, 60.0)
      braking = _fly(20.0, 72.0, 29.0, 20.0, lambda t: -3.5 if t > 1 else 0.0, 25.0)
    self.assertGreater(stopped["end_gap"], MOONPILOT_STOP_REST - 1.0)
    self.assertLess(stopped["v"][-1], 0.05)
    self.assertFalse(braking["contact"])
    self.assertGreater(braking["min_gap"], MOONPILOT_CRAWL_REST - 0.2)

  def test_a_lead_pulling_away_is_owed_a_positive_ask(self):
    self.assertLess(unified_lead_accel(18.0, 90.0, 12.5, 0.0, 1.45), 0.0)
    self.assertGreater(unified_lead_accel(18.0, 90.0, 12.5, 1.0, 1.45), 0.0)


if __name__ == "__main__":
  unittest.main()
