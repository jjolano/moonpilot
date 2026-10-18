import unittest

import numpy as np

from moonpilot.lab.lookahead import turn_values


class TestLookaheadMetrics(unittest.TestCase):
  def test_advance_apex_and_drift_on_a_synthetic_turn(self):
    base = np.zeros(120)
    for i in range(40, 90):
      base[i] = (i - 39) * 6e-5
    changed = np.where(base != 0, 2 * base, 0.0)
    mask = np.ones(120, dtype=bool)
    speed = np.full(120, 15.0)

    band, advance, apex, inside, outward = turn_values(base, changed, speed, mask)[0]
    self.assertEqual(band, "10-17")
    self.assertAlmostEqual(advance, 0.05)
    self.assertAlmostEqual(apex, 2.0)
    self.assertGreater(inside, 0.0)
    self.assertEqual(outward, 0.0)
