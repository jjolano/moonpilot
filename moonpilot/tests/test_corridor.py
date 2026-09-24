import unittest

import numpy as np

from moonpilot.corridor import corridor_bounds, corridor_width, path_in_corridor


def _edge(x, y):
  return (list(x), list(y))


class TestCorridorBounds(unittest.TestCase):
  def test_orders_by_y_not_by_edge_index(self):
    # Swapped arguments: bounds must still be left ≤ right in the device frame.
    xs = [0.0, 100.0]
    left, right = corridor_bounds(_edge(xs, [-4.0, -4.0]), _edge(xs, [4.0, 4.0]), [0.0, 50.0])
    self.assertTrue(np.allclose(left, -4.0))
    self.assertTrue(np.allclose(right, 4.0))
    left2, right2 = corridor_bounds(_edge(xs, [4.0, 4.0]), _edge(xs, [-4.0, -4.0]), [0.0, 50.0])
    self.assertTrue(np.allclose(left2, -4.0))
    self.assertTrue(np.allclose(right2, 4.0))

  def test_sample_outside_edge_span_is_unknown(self):
    xs = [10.0, 100.0]
    left, right = corridor_bounds(_edge(xs, [-3.0, -3.0]), _edge(xs, [3.0, 3.0]), [0.0, 50.0, 150.0])
    self.assertTrue(np.isnan(left[0]))
    self.assertTrue(np.isfinite(left[1]))
    self.assertTrue(np.isnan(left[2]))

  def test_empty_or_mismatched_edge_is_all_unknown(self):
    left, right = corridor_bounds(_edge([0.0], [-1.0]), _edge([0.0, 1.0], [1.0, 1.0]), [0.0])
    self.assertTrue(np.isnan(left[0]) and np.isnan(right[0]))

  def test_width_is_positive_and_nan_off_span(self):
    xs = [0.0, 192.0]
    w = corridor_width(_edge(xs, [-5.0, -6.0]), _edge(xs, [5.0, 6.0]), [0.0, 96.0, 200.0])
    self.assertAlmostEqual(w[0], 10.0)
    self.assertAlmostEqual(w[1], 11.0)
    self.assertTrue(np.isnan(w[2]))


class TestPathInCorridor(unittest.TestCase):
  def _straight(self):
    # Parallel edges at ±4 m, path down the middle.
    xs = [0.0, 100.0]
    return _edge(xs, [-4.0, -4.0]), _edge(xs, [4.0, 4.0])

  def test_centered_path_is_inside(self):
    left, right = self._straight()
    flags = path_in_corridor([0.0, 50.0], [0.0, 0.0], left, right)
    self.assertTrue(np.all(flags == 1.0))

  def test_path_beyond_edge_is_outside(self):
    left, right = self._straight()
    flags = path_in_corridor([50.0], [5.0], left, right)
    self.assertEqual(flags[0], 0.0)

  def test_path_on_edge_is_inside_within_margin(self):
    left, right = self._straight()
    # Exactly on the left edge: inside by construction (margin ≥ 0).
    flags = path_in_corridor([50.0], [-4.0], left, right)
    self.assertEqual(flags[0], 1.0)
    # Past the margin: outside.
    flags = path_in_corridor([50.0], [-4.2], left, right, margin=0.15)
    self.assertEqual(flags[0], 0.0)

  def test_unknown_span_is_nan_not_free(self):
    left, right = self._straight()
    flags = path_in_corridor([150.0], [0.0], left, right)
    self.assertTrue(np.isnan(flags[0]))

  def test_empty_path_is_empty(self):
    left, right = self._straight()
    self.assertEqual(path_in_corridor([], [], left, right).size, 0)


if __name__ == "__main__":
  unittest.main()
