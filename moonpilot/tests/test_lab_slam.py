import unittest

import numpy as np

from moonpilot.lab.slam import _nearest


class TestNearestLoggedSample(unittest.TestCase):
  def test_picks_the_nearer_neighbour(self):
    logged = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    # 0.2 is nearer 0.0; 0.6 is nearer 1.0; 1.6 is nearer 2.0; a tie at 1.5 goes left.
    np.testing.assert_array_equal(_nearest(np.array([0.2, 0.6, 1.6]), logged)[:, 1], [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(_nearest(np.array([0.9, 1.1, 1.5]), logged)[:, 1], [20.0, 20.0, 20.0])

  def test_exact_timestamps_and_the_ends(self):
    logged = np.array([[0.0, 1.0], [5.0, 2.0], [9.0, 3.0]])
    np.testing.assert_array_equal(_nearest(np.array([5.0]), logged)[:, 1], [2.0])
    # Outside the span, the nearest end is the answer (searchsorted is clipped, never extrapolated).
    np.testing.assert_array_equal(_nearest(np.array([-3.0, 4.0, 100.0]), logged)[:, 1], [1.0, 2.0, 3.0])

  def test_single_sample_and_empty(self):
    logged = np.array([[1.0, 7.0]])
    np.testing.assert_array_equal(_nearest(np.array([0.0, 9.0]), logged)[:, 1], [7.0, 7.0])
    self.assertEqual(len(_nearest(np.array([1.0]), np.empty((0, 2)))), 0)


if __name__ == "__main__":
  unittest.main()
