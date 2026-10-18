import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL

from moonpilot.lab.planner import control_rate_dt


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


if __name__ == "__main__":
  unittest.main()
