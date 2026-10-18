import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL

from moonpilot.lab import lag_s, reversals_per_s


class TestLabMetrics(unittest.TestCase):
  def test_reversals_count_direction_changes_and_skip_holds(self):
    # up, hold, up, down, hold, down, up: two direction changes in one run of 7 frames.
    x = np.array([0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.0, 3.0])
    self.assertAlmostEqual(reversals_per_s(x, np.ones(len(x), dtype=bool)), 2 / (len(x) * DT_CTRL))

  def test_reversals_never_count_across_a_masked_gap(self):
    x = np.array([0.0, 1.0, 2.0, 0.0, -1.0, -2.0])
    mask = np.array([True, True, True, False, True, True])
    # inside each run the command only moves one way; the drop across the gap is not a reversal.
    self.assertEqual(reversals_per_s(x, mask), 0.0)

  def test_lag_recovers_a_known_delay_below_one_frame(self):
    t = np.arange(0, 60, DT_CTRL)
    ref = np.sin(2 * np.pi * 0.3 * t) + 0.5 * np.sin(2 * np.pi * 0.11 * t)
    delay = 0.037
    x = np.interp(t - delay, t, ref)
    self.assertAlmostEqual(lag_s(ref, x, np.ones(len(t), dtype=bool)), delay, delta=0.002)


if __name__ == "__main__":
  unittest.main()
