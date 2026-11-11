import unittest

import numpy as np

from opendbc.car.interfaces import ACCEL_MAX

from moonpilot.corridor import (
  MOONPILOT_CORRIDOR_HISTORY_S,
  MOONPILOT_SQUEEZE_ACCEL_MIN,
  MOONPILOT_SQUEEZE_FULL_WIDTH,
  MOONPILOT_SQUEEZE_MIN_WIDTH,
  RollingCorridor,
  corridor_bounds,
  corridor_width,
  path_in_corridor,
  path_outside_alert,
  path_outside_fraction,
  squeeze_accel,
)
from moonpilot.tests.fakes import params
from moonpilot.tests.harness import inputs
from openpilot.cereal import log




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


class TestSqueezeAccel(unittest.TestCase):
  def _edges(self, width, xs=None):
    xs = list(np.asarray(xs if xs is not None else [0.0, 192.0], dtype=float))
    half = float(width) / 2.0
    return _edge(xs, [-half] * len(xs)), _edge(xs, [half] * len(xs))

  def test_wide_enough_is_inactive(self):
    left, right = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH + 0.5)
    self.assertEqual(squeeze_accel(left, right, [0.0, 50.0, 100.0]), ACCEL_MAX)

  def test_narrow_adds_a_bounded_brake(self):
    left, right = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH - 1.0)
    a = squeeze_accel(left, right, [0.0, 20.0, 40.0])
    self.assertLess(a, 0.0)
    self.assertGreaterEqual(a, MOONPILOT_SQUEEZE_ACCEL_MIN)

  def test_at_or_under_min_width_is_the_floor(self):
    for width in (MOONPILOT_SQUEEZE_MIN_WIDTH, MOONPILOT_SQUEEZE_MIN_WIDTH - 1.0, 0.0):
      with self.subTest(width=width):
        left, right = self._edges(width)
        self.assertAlmostEqual(squeeze_accel(left, right, [0.0, 10.0]), MOONPILOT_SQUEEZE_ACCEL_MIN, delta=1e-9)

  def test_narrow_outside_the_lookahead_is_inactive(self):
    # Pinch only after the 40 m window: the sample still spans it, but the binding width is past it.
    xs = [0.0, 45.0, 100.0]
    left, right = _edge(xs, [-6.0, -1.0, -1.0]), _edge(xs, [6.0, 1.0, 1.0])
    self.assertEqual(squeeze_accel(left, right, xs, x_ego=0.0), ACCEL_MAX)
    # Same edges with the car already past the wide part: the pinch is now inside the window.
    self.assertLess(squeeze_accel(left, right, xs, x_ego=40.0), 0.0)

  def test_unknown_span_is_inactive_not_free_braking(self):
    # Edges that only start at 50 m: samples 0…40 are NaN, so no candidate.
    left, right = self._edges(3.0, xs=[50.0, 192.0])
    self.assertEqual(squeeze_accel(left, right, [0.0, 10.0, 40.0]), ACCEL_MAX)

  def test_empty_edges_are_inactive(self):
    self.assertEqual(squeeze_accel(_edge([], []), _edge([], []), [0.0]), ACCEL_MAX)


class TestPathOutside(unittest.TestCase):
  def _straight(self, half=4.0):
    xs = [0.0, 100.0]
    return _edge(xs, [-half] * 2), _edge(xs, [half] * 2)

  def test_centered_path_is_not_outside(self):
    left, right = self._straight()
    self.assertEqual(path_outside_fraction([0.0, 20.0, 40.0], [0.0, 0.0, 0.0], left, right), 0.0)

  def test_path_beyond_edge_counts(self):
    left, right = self._straight()
    # Half the known samples outside → fraction 0.5 ≥ 0.25.
    frac = path_outside_fraction([10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 0.0, 0.0], left, right)
    self.assertAlmostEqual(frac, 0.5)

  def test_unknown_samples_are_excluded_not_counted_as_inside(self):
    left, right = self._straight()
    # Only one known sample, and it is outside: fraction is 1.0, not 0.25 of four.
    frac = path_outside_fraction([10.0, 150.0], [5.0, 0.0], left, right)
    self.assertEqual(frac, 1.0)

  def test_no_known_samples_is_zero(self):
    left, right = self._straight()
    self.assertEqual(path_outside_fraction([150.0], [0.0], left, right), 0.0)

  def test_empty_path_is_zero(self):
    left, right = self._straight()
    self.assertEqual(path_outside_fraction([], [], left, right), 0.0)

  def test_lookahead_window_excludes_far_samples(self):
    left, right = self._straight()
    # Far sample outside, near samples inside: window is 40 m, so far does not count.
    frac = path_outside_fraction([10.0, 20.0, 80.0], [0.0, 0.0, 99.0], left, right)
    self.assertEqual(frac, 0.0)



def _outside_inputs():
  # Path half outside a ±4 m corridor over the near horizon.
  xs = [float(x) for x in np.linspace(0.0, 40.0, 9)]
  ys = [5.0 if x < 20.0 else 0.0 for x in xs]
  edge_xs = [0.0, 100.0]
  edges = [_edge(edge_xs, [-4.0, -4.0]), _edge(edge_xs, [4.0, 4.0])]
  sm = inputs(road_edges=edges)
  sm["modelV2"].position = log.XYZTData.new_message(x=xs, y=ys)
  return sm


class TestPathOutsideAlert(unittest.TestCase):
  def test_feature_off_returns_none(self):
    self.assertIsNone(path_outside_alert(_outside_inputs(), params(overrides={"MoonpilotPathOutside": False})))

  def test_lat_inactive_returns_none(self):
    sm = _outside_inputs()
    sm["carControl"].latActive = False
    self.assertIsNone(path_outside_alert(sm, params(overrides={"MoonpilotPathOutside": True})))


  def test_missing_edges_returns_none(self):
    sm = inputs()
    sm["modelV2"].position = log.XYZTData.new_message(x=[0.0], y=[0.0])
    self.assertIsNone(path_outside_alert(sm, params(overrides={"MoonpilotPathOutside": True})))


  def test_inside_path_returns_none(self):
    xs = [float(x) for x in np.linspace(0.0, 40.0, 9)]
    ys = [0.0] * 9
    edges = [_edge([0.0, 100.0], [-4.0, -4.0]), _edge([0.0, 100.0], [4.0, 4.0])]
    sm = inputs(road_edges=edges)
    sm["modelV2"].position = log.XYZTData.new_message(x=xs, y=ys)
    self.assertIsNone(path_outside_alert(sm, params(overrides={"MoonpilotPathOutside": True})))

  def test_outside_path_returns_event(self):
    sm = _outside_inputs()
    self.assertEqual(
      path_outside_alert(sm, params(overrides={"MoonpilotPathOutside": True})),
      log.OnroadEvent.EventName.pathOutside,
    )


class TestRollingCorridor(unittest.TestCase):
  def _edges(self, width, xs=None):
    xs = list(np.asarray(xs if xs is not None else [0.0, 192.0], dtype=float))
    half = float(width) / 2.0
    return _edge(xs, [-half] * len(xs)), _edge(xs, [half] * len(xs))


  def test_clear_drops_history(self):
    rc = RollingCorridor()
    left, right = self._edges(MOONPILOT_SQUEEZE_MIN_WIDTH)
    rc.push(0.0, (0.0, 0.0, 0.0), left, right)
    self.assertLess(rc.squeeze_accel((0.0, 0.0, 0.0), [0.0, 10.0]), 0.0)
    rc.clear()
    self.assertEqual(rc.squeeze_accel((0.0, 0.0, 0.0), [0.0, 10.0]), ACCEL_MAX)

  def test_history_expires_after_window(self):
    rc = RollingCorridor()
    left, right = self._edges(MOONPILOT_SQUEEZE_MIN_WIDTH)
    rc.push(0.0, (0.0, 0.0, 0.0), left, right)
    # Query far past the history window with a fresh push that also expires the old one.
    rc.push(MOONPILOT_CORRIDOR_HISTORY_S + 1.0, (100.0, 0.0, 0.0), *self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH + 1.0))
    # Only the wide strip remains: inactive.
    self.assertEqual(rc.squeeze_accel((100.0, 0.0, 0.0), [0.0, 10.0]), ACCEL_MAX)

  def test_single_frame_matches_free_function(self):
    rc = RollingCorridor()
    left, right = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH - 1.0)
    rc.push(0.0, (0.0, 0.0, 0.0), left, right)
    a_roll = rc.squeeze_accel((0.0, 0.0, 0.0), [0.0, 20.0, 40.0])
    a_free = squeeze_accel(left, right, [0.0, 20.0, 40.0])
    self.assertAlmostEqual(a_roll, a_free, delta=1e-12)

  def test_intersection_keeps_narrowest_strip(self):
    # Frame 0: wide. Frame 1 (same pose): narrow. Intersection must be narrow.
    rc = RollingCorridor()
    wide = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH + 1.0)
    narrow = self._edges(MOONPILOT_SQUEEZE_MIN_WIDTH)
    rc.push(0.0, (0.0, 0.0, 0.0), *wide)
    rc.push(0.05, (0.0, 0.0, 0.0), *narrow)
    a = rc.squeeze_accel((0.0, 0.0, 0.0), [0.0, 10.0])
    self.assertAlmostEqual(a, MOONPILOT_SQUEEZE_ACCEL_MIN, delta=1e-9)

  def test_translation_warp_preserves_strip_under_straight_motion(self):
    # Car moves +10 m in x; strip stored at origin should still be under the car when queried
    # at the new pose after warp (relative SE2, identity yaw).
    rc = RollingCorridor()
    left, right = self._edges(MOONPILOT_SQUEEZE_MIN_WIDTH, xs=[0.0, 192.0])
    rc.push(0.0, (0.0, 0.0, 0.0), left, right)
    # Query at pose (10, 0, 0): warped strip is x' = x - 10, so sample_x=0 is still near the
    # original geometry's x=10 (inside the strip's span and still narrow).
    a = rc.squeeze_accel((10.0, 0.0, 0.0), [0.0, 10.0])
    self.assertLess(a, 0.0)



if __name__ == "__main__":
  unittest.main()
