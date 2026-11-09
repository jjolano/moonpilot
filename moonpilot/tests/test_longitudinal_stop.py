"""The lead law's finish behind a lead at rest (`MOONPILOT_ARRIVE_MIN_T_REST`), and the moving-lead law it leaves alone.

Its own module: `test_longitudinal.py` sits at the repo's 120 KB file cap.
"""
import unittest
from unittest import mock

import numpy as np

from openpilot.cereal import log

from moonpilot.longitudinal import (
  MOONPILOT_ARRIVE_MIN_T,
  MOONPILOT_ARRIVE_MIN_T_REST,
  MOONPILOT_CREEP_SPEED,
  MOONPILOT_MOVING_STOP_RESERVE,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_STOP_REST,
  MOONPILOT_T_FOLLOW,
  lead_accel,
)


class TestRestFinish(unittest.TestCase):
  """Behind a lead at rest the near edge's arrival runs on a shorter minimum closing time, blended in by the lead's
  predicted speed: inside that minimum the brake is `closing / (2 min_t)`, so it finishes a stop the shipped 1.5 s
  crawls (`python -m moonpilot.lab stop-finish`). A lead at creep speed or above is the shipped law exactly."""

  T_FOLLOW = MOONPILOT_T_FOLLOW[int(log.LongitudinalPersonality.standard)]

  def test_resting_lead_finishes_firmer_and_never_less_firm(self):
    v_ego, gaps = 1.0, np.arange(MOONPILOT_STOP_REST + 0.1, 12.0, 0.1)
    now = np.array([lead_accel(v_ego, g, 0.0, 0.0, self.T_FOLLOW) for g in gaps])
    with mock.patch("moonpilot.longitudinal.MOONPILOT_ARRIVE_MIN_T_REST", MOONPILOT_ARRIVE_MIN_T):
      shipped = np.array([lead_accel(v_ego, g, 0.0, 0.0, self.T_FOLLOW) for g in gaps])
    self.assertLessEqual(float(np.max(now - shipped)), 1e-12)  # never less braking in the last meters
    self.assertLess(float(np.min(now - shipped)), -0.3)  # and clearly firmer at the rest edge, where the crawl was
    self.assertLess(MOONPILOT_ARRIVE_MIN_T_REST, MOONPILOT_ARRIVE_MIN_T)

  def test_a_moving_lead_never_sees_the_rest_value(self):
    """The blend reaches only a lead under creep speed. Flown closing on each moving lead, the law is invariant across
    three very different rest minima: a blend that leaked to moving leads would move with them (checked by mutation —
    dropping the `moving` weight fails this test).
    """
    for v_lead in (MOONPILOT_CREEP_SPEED, 3.0, 8.0):
      with self.subTest(v_lead=v_lead):
        v_ego = v_lead + 6.0
        rest = MOONPILOT_STOP_REST + (MOONPILOT_STOP_DISTANCE + MOONPILOT_MOVING_STOP_RESERVE - MOONPILOT_STOP_REST)
        near = max(rest, self.T_FOLLOW * v_lead)
        g = near + (v_ego - v_lead)  # room == closing: the minimum decides, not the room
        asks = []
        for value in (MOONPILOT_ARRIVE_MIN_T_REST, MOONPILOT_ARRIVE_MIN_T, 4.0):
          with mock.patch("moonpilot.longitudinal.MOONPILOT_ARRIVE_MIN_T_REST", value):
            asks.append(lead_accel(v_ego, g, v_lead, 0.0, self.T_FOLLOW))
        self.assertLess(asks[0], -0.2)  # the near-edge arrival is active in this state
        self.assertEqual(len(set(asks)), 1, asks)


if __name__ == "__main__":
  unittest.main()
