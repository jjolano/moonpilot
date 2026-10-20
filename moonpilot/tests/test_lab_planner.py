import math
import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL

from moonpilot.lab.planner import HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, _LaggedPlant, _rest_gaps, control_rate_dt


class TestControlRateGuard(unittest.TestCase):
  """The acceleration controller advances per call, so a replay has to tick it at its own rate.

  A 20 Hz feed is one fifth the ramp per second of a 100 Hz one: the harness must refuse to print a
  controller number rather than quietly reporting the wrong controller.
  """

  def test_accepts_the_control_rate(self):
    t = np.arange(0, 3_000_000_000, int(DT_CTRL * 1e9))
    self.assertAlmostEqual(control_rate_dt(t), DT_CTRL, places=6)

  def test_refuses_a_planner_rate_feed(self):
    t = np.arange(0, 3_000_000_000, int(0.05 * 1e9))  # 20 Hz, the planner's tick rate
    with self.assertRaises(ValueError):
      control_rate_dt(t)

  def test_refuses_too_few_frames(self):
    with self.assertRaises(ValueError):
      control_rate_dt(np.arange(0, int(0.05 * 1e9), int(DT_CTRL * 1e9)))


class TestLaggedPlant(unittest.TestCase):
  def test_a_step_waits_the_dead_time_then_rises_on_the_lag(self):
    """The flights' `0.2 s dead time + 0.25 s lag` is the plant's, to the frame: a queue appended before it is read
    delivered the step one frame early (0.15 s)."""
    plant = _LaggedPlant()
    a = [plant.step(1.0) for _ in range(20)]
    dead = round(HOLD_FLY_DEAD_S / DT_MDL)
    self.assertEqual(a[:dead], [0.0] * dead)
    self.assertGreater(a[dead], 0.0)
    self.assertAlmostEqual(a[dead - 1 + round(HOLD_FLY_TAU_S / DT_MDL)], 1.0 - math.exp(-1.0), places=9)


def _frames(*pieces):
  """Concatenated (frames, route, v, lead present, gap) pieces, the lead stopped throughout."""
  route, v, lead, gap = (np.concatenate([np.full(p[0], p[i]) for p in pieces]) for i in range(1, 5))
  return route, v, lead.astype(bool), np.zeros(len(v)), gap


class TestRestGaps(unittest.TestCase):
  def test_a_radar_dropout_during_one_standstill_is_one_stop(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 5.0), (5, 0, 0.0, False, 5.0), (40, 0, 0.0, True, 5.5)))
    self.assertEqual(gaps.tolist(), [5.0])

  def test_the_lead_may_be_read_just_after_the_stop_frame(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (10, 0, 0.0, False, 0.0), (40, 0, 0.0, True, 4.0)))
    self.assertEqual(gaps.tolist(), [4.0])

  def test_a_creep_restop_is_the_same_stop_and_only_real_motion_rearms(self):
    creep = _frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 6.0), (20, 0, 1.0, True, 5.0), (20, 0, 0.0, True, 4.0))
    self.assertEqual(_rest_gaps(*creep).tolist(), [6.0])
    again = _frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 6.0), (20, 0, 3.0, True, 9.0), (20, 0, 0.0, True, 4.0))
    self.assertEqual(_rest_gaps(*again).tolist(), [6.0, 4.0])

  def test_motion_in_another_route_does_not_arm_a_stop(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (40, 1, 0.0, True, 5.0)))
    self.assertEqual(gaps.tolist(), [])


if __name__ == "__main__":
  unittest.main()
