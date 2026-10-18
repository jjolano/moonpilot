import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL

from moonpilot.lab.damping import _curve_windows, _shape, _stretches, _turns


class TestTurnDetection(unittest.TestCase):
  def setUp(self):
    self.t = np.arange(1000) * DT_CTRL
    self.la = np.zeros(1000)
    self.la[200:401] = np.r_[np.linspace(0, 1.5, 101), np.linspace(1.5, 0, 101)[1:]]
    self.mask = np.ones(1000, dtype=bool)

  def test_turns(self):
    self.assertEqual(_turns(self.la, self.mask), [300])
    self.la[800] = 0.9
    self.assertEqual(_turns(self.la, self.mask), [300])  # below 1 m/s^2
    self.la[500:701] = np.linspace(0, 1.2, 201)
    self.assertEqual(_turns(self.la, self.mask), [300, 700])  # more than +-2 s apart, so both count
    self.mask[700] = False
    self.assertEqual(_turns(self.la, self.mask), [300])  # the peak itself must be under the mask

  def test_stretches(self):
    self.assertEqual(_stretches(self.la, 300, self.mask), [(210, 300), (300, 391)])
    self.mask[250] = False
    self.assertEqual(_stretches(self.la, 300, self.mask), [None, (300, 391)])  # only the side leaving the mask is dropped
    self.mask[350] = False
    self.assertEqual(_stretches(self.la, 300, self.mask), [None, None])

  def test_shape(self):
    la = np.zeros(1000)
    la[100:301] = np.linspace(0, 1, 201)  # a linear 2 s build: steepest rate 0.5 /s of a peak of 1
    self.assertAlmostEqual(_shape(la, self.t, 300, -1, self.mask), 0.5)
    self.mask[200] = False
    self.assertIsNone(_shape(la, self.t, 300, -1, self.mask))

  def test_curve_windows(self):
    t = np.arange(3000) * DT_CTRL
    d = {
      "t": t,
      "v_ego": np.full(3000, 10.0),
      "lat_active": np.r_[np.zeros(500, bool), np.ones(2000, bool), np.zeros(500, bool)],
      "pressed": np.zeros(3000, dtype=bool),
    }
    la = np.zeros(3000)
    la[600:700] = 1.2
    la[1200:1300] = 1.2
    self.assertEqual(_curve_windows(d, la), [600, 900, 1200])  # the 300-frame step grid's windows over the two bumps
    d["pressed"][1250] = True  # a press kills the windows containing it, not the earlier one
    self.assertEqual(_curve_windows(d, la), [600])
    d["v_ego"][:] = 20.0
    self.assertEqual(_curve_windows(d, la), [])  # outside the measured 3-17 m/s


if __name__ == "__main__":
  unittest.main()
