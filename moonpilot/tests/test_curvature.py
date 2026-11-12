"""The smoothing lead, the look-ahead lead-in, the re-centering cap and the curvature request filter: behavior and their
restart-gated factories."""

import functools
import math
import struct
import unittest
from typing import Any
from unittest import mock

import numpy as np

import moonpilot.curvature as curvature_mod

from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.curvature import (
  MOONPILOT_MAX_MODEL_AGE,
  MOONPILOT_PREVIEW_MIN_SPEED,
  MOONPILOT_RECENTER_LAT_ACCEL,
  MOONPILOT_SMOOTH_LEAD_MAX_LAT_ACCEL,
  PathSmooth,
  lane_bend,
  moonpilot_curvature,
  moonpilot_path_smooth,
  curvature_reference,
)
from moonpilot.features import PATH_LOOKAHEAD, PATH_RECENTER, PATH_SMOOTH
from moonpilot.tests.fakes import params

LOOKAHEAD = PATH_LOOKAHEAD.values
TAUS = PATH_SMOOTH.values


T_IDXS = np.asarray(ModelConstants.T_IDXS)
NOW = 10.0
CAPTURE_AGE = 0.08
RECEIVE_AGE = 0.02
LAT_DELAY = 0.2
V_EGO = 25.0
LEAD = 0.05


def _model(psi_rate, psi=None, *, timestamp=NOW - CAPTURE_AGE):
  """Build model yaw and yaw-rate arrays on their published time grid."""
  psi_rate = np.broadcast_to(np.asarray(psi_rate, dtype=float), T_IDXS.shape).copy()
  if psi is None:
    psi = np.concatenate(([0.0], np.cumsum(0.5 * (psi_rate[1:] + psi_rate[:-1]) * np.diff(T_IDXS))))
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = int(timestamp * 1e9)
  model.orientation = log.XYZTData.new_message(t=T_IDXS.tolist(), z=np.asarray(psi, dtype=float).tolist())
  model.orientationRate = log.XYZTData.new_message(t=T_IDXS.tolist(), z=psi_rate.tolist())
  return model


def _corner(kappa, *, t_start=0.0, t_end=float("inf")):
  rate = np.where((T_IDXS >= t_start) & (T_IDXS < t_end), kappa * V_EGO, 0.0)
  return _model(rate)


def _ramp(kappa0=0.001, slope=0.002, **kwargs):
  """A turn tightening at `slope` 1/m per second: the shape a first-order filter lags by exactly tau."""
  return _model(V_EGO * (kappa0 + slope * T_IDXS), **kwargs)


def _update(model, desired_curvature=0.0, **overrides):
  arguments: dict[str, Any] = {
    "v_ego": V_EGO,
    "lat_delay": LAT_DELAY,
    "model_recv_time": NOW - RECEIVE_AGE,
    "now": NOW,
    "model_valid": True,
    "lead": LEAD,
  }
  arguments.update(overrides)
  return curvature_reference(model, desired_curvature, **arguments)


def _bits(value):
  return struct.pack("!d", value)


class TestFactory(unittest.TestCase):
  def test_all_off_is_none(self):
    self.assertIsNone(moonpilot_curvature(params(0)))

  def test_the_look_ahead_alone_binds_no_lead(self):
    reference = moonpilot_curvature(params(0, {PATH_LOOKAHEAD.key: 2}))
    assert isinstance(reference, functools.partial)
    self.assertEqual(reference.keywords, {"lookahead": LOOKAHEAD[2], "lead": 0.0, "recenter": 0.0})

  def test_smoothing_alone_binds_its_tau_as_the_lead(self):
    for index in range(1, len(TAUS)):
      with self.subTest(index=index):
        reference = moonpilot_curvature(params(0, {PATH_SMOOTH.key: index}))
        assert isinstance(reference, functools.partial)
        self.assertEqual(reference.keywords, {"lookahead": None, "lead": TAUS[index], "recenter": 0.0})

  def test_recentering_alone_binds_its_cap(self):
    reference = moonpilot_curvature(params(0, {PATH_RECENTER.key: True}))
    assert isinstance(reference, functools.partial)
    self.assertEqual(reference.keywords, {"lookahead": None, "lead": 0.0, "recenter": MOONPILOT_RECENTER_LAT_ACCEL})


class TestLookahead(unittest.TestCase):
  """The look-ahead window: a coming turn pulls the request in early; it never takes turn away from today's request."""

  HORIZON = CAPTURE_AGE + LAT_DELAY

  def _out(self, model, level, desired=0.0, v_ego=V_EGO):
    return _update(model, desired, v_ego=v_ego, lead=0.0, lookahead=LOOKAHEAD[level])

  def test_the_lead_in_fades_out_below_corner_speeds(self):
    model = _corner(0.002, t_start=self.HORIZON + 0.5)
    for v in (6.0, 7.9):
      with self.subTest(v=v):
        self.assertEqual(_bits(self._out(model, 3, v_ego=v)), _bits(self._out(model, 0, v_ego=v)))
    today, faded = self._out(model, 0, v_ego=10.0), self._out(model, 3, v_ego=10.0)
    with mock.patch.object(curvature_mod, "MOONPILOT_CORRECTION_SPEEDS", ([0.0, 1.0], [1.0, 1.0])):
      full = self._out(model, 3, v_ego=10.0)
    self.assertGreater(full, today)
    self.assertAlmostEqual(faded - today, 0.5 * (full - today), delta=1e-15)

  def test_a_turn_beyond_the_single_sample_is_entered_early_and_more_with_each_level(self):
    model = _corner(0.002, t_start=self.HORIZON + 0.5)  # 0.5 s after the point today's sample reads
    self.assertEqual(_bits(self._out(model, 0)), _bits(0.0))  # off: nothing yet, exactly today's request
    light, medium, strong = (self._out(model, level) for level in (1, 2, 3))
    self.assertGreater(light, 0.0)
    self.assertLess(light, medium)
    self.assertLess(medium, strong)
    self.assertLess(strong, 0.002)  # a lead-in, never the turn itself

  def test_an_easing_or_ending_turn_keeps_todays_request_so_it_never_runs_wide(self):
    for model in (_corner(0.002, t_end=self.HORIZON + 0.5), _model(np.interp(T_IDXS, [0.0, 1.0], [0.002 * V_EGO, 0.0005 * V_EGO]))):
      for level in (1, 2, 3):
        with self.subTest(level=level):
          self.assertEqual(_bits(self._out(model, level, desired=0.002)), _bits(0.002))

  def test_the_lead_in_is_capped_per_level(self):
    model = _corner(0.05, t_start=self.HORIZON + 0.1)
    for level in (1, 2, 3):
      lookahead = LOOKAHEAD[level]
      assert lookahead is not None
      with self.subTest(level=level):
        self.assertAlmostEqual(self._out(model, level), lookahead[2] / V_EGO ** 2, delta=1e-12)

  def test_a_window_past_the_published_path_keeps_todays_request(self):
    model = _corner(0.002)
    lookahead = LOOKAHEAD[3]
    assert lookahead is not None
    late = T_IDXS[-1] - CAPTURE_AGE - lookahead[0] + 0.01
    self.assertEqual(_bits(_update(model, 0.03125, lat_delay=late, lead=0.0, lookahead=lookahead)), _bits(0.03125))


def _lanes(bend=0.0, *, probs=(0.9, 0.9), heading=0.0, narrowing=0.0, offset=0.0, edges=None, lane_change=log.LaneChangeState.off):
  """A straight path whose ego lane lines bend at `bend` 1/m (positive right), 3.6 m apart, the car `offset` m right of the
  lane's center and each line turned `narrowing` rad toward it (negative: the lane widens); `edges` is (left y, right y, std)
  of straight road edges in the car's frame, none published by default."""
  model = _corner(0.0)
  x = np.asarray(ModelConstants.X_IDXS)
  model.laneLines = [log.XYZTData.new_message(x=x.tolist(), y=(line - offset + (heading - narrowing * math.copysign(1.0, line)) * x
                                                                + 0.5 * bend * x**2).tolist())
                     for line in (-5.4, -1.8, 1.8, 5.4)]
  model.laneLineProbs = [0.5, probs[0], probs[1], 0.5]
  if edges is not None:
    model.roadEdges = [log.XYZTData.new_message(x=x.tolist(), y=[edge] * len(x)) for edge in edges[:2]]
    model.roadEdgeStds = [edges[2], edges[2]]
  model.meta.laneChangeState = lane_change
  return model


class TestRecenter(unittest.TestCase):
  """MoonpilotPathRecenter: the request may follow the lane's bend at any pace, but on straight-ish lanes its part beyond
  the bend is capped; the cap only ever takes turn away, and only touches a move whose recovery side is known to be clear."""

  CAP = MOONPILOT_RECENTER_LAT_ACCEL / V_EGO**2
  SLIGHT = 0.2 / V_EGO**2  # a bend under the curve fade

  def _out(self, model, desired, **overrides):
    return _update(model, desired, lead=0.0, recenter=MOONPILOT_RECENTER_LAT_ACCEL, **overrides)

  def test_the_lane_is_read_off_its_lines_and_one_seen_line_is_enough(self):
    lane = lane_bend(_lanes(0.002, offset=0.5))
    self.assertAlmostEqual(lane.bend, 0.002, delta=1e-8)  # lines are float32
    self.assertAlmostEqual(lane.left, -2.3, delta=1e-6)
    self.assertAlmostEqual(lane.right, 1.3, delta=1e-6)
    self.assertAlmostEqual(lane_bend(_lanes(0.002, heading=0.03)).bend, 0.002, delta=1e-9)  # the car's heading is not a bend
    one = lane_bend(_lanes(-0.001, probs=(0.9, 0.1)))
    self.assertAlmostEqual(one.bend, -0.001, delta=1e-9)
    self.assertTrue(math.isnan(one.right))
    self.assertEqual(lane_bend(_lanes(0.002, probs=(0.3, 0.3))).weight, 0.0)

  def test_a_sideways_move_on_a_straight_lane_is_held_to_the_cap(self):
    self.assertAlmostEqual(self._out(_lanes(), -4 * self.CAP), -self.CAP, delta=1e-12)
    self.assertAlmostEqual(self._out(_lanes(), 4 * self.CAP), self.CAP, delta=1e-12)
    self.assertEqual(_bits(self._out(_lanes(), 0.5 * self.CAP)), _bits(0.5 * self.CAP))  # inside the cap: untouched

  def test_a_slight_bend_is_followed_and_a_curve_is_left_alone(self):
    self.assertEqual(_bits(self._out(_lanes(self.SLIGHT), self.SLIGHT)), _bits(self.SLIGHT))
    self.assertAlmostEqual(self._out(_lanes(self.SLIGHT), self.SLIGHT + 3 * self.CAP), self.SLIGHT + self.CAP, delta=1e-8)  # lines are float32
    curve = 1.5 / V_EGO**2
    self.assertEqual(_bits(self._out(_lanes(curve), curve + 3 * self.CAP)), _bits(curve + 3 * self.CAP))  # taking turn away runs wide

  def test_it_never_adds_turn_or_reverses_the_request(self):
    with mock.patch.object(curvature_mod, "MOONPILOT_RECENTER_BENDS", ([0.0, 1.0], [1.0, 1.0])):  # the guard, not the curve fade
      bend = 1.5 / V_EGO**2
      self.assertEqual(_bits(self._out(_lanes(bend), 0.2 * bend)), _bits(0.2 * bend))  # drifting wide: not pulled in
      self.assertEqual(self._out(_lanes(-bend), 2 * self.CAP), 0.0)  # a right move in a left curve stops at zero

  def test_a_move_away_from_a_close_line_is_never_capped(self):
    near_left = _lanes(offset=-0.7)  # 1.1 m from the left line
    self.assertEqual(_bits(self._out(near_left, 4 * self.CAP)), _bits(4 * self.CAP))  # back toward the center: free
    self.assertAlmostEqual(self._out(near_left, -4 * self.CAP), -self.CAP, delta=1e-12)  # further toward the line: capped

  def test_a_correction_the_cap_could_not_finish_short_of_the_line_is_never_capped(self):
    # Centered, 1.8 m from each line, steering responding 0.28 s out (capture age + lateral delay): at 25 m/s the car keeps
    # 1.3 m only from an approach below 0.47 m/s, a heading of 0.019 rad.
    toward_right = _lanes(heading=-0.03)  # the right line closes on the car at 0.75 m/s
    self.assertEqual(_bits(self._out(toward_right, -4 * self.CAP)), _bits(-4 * self.CAP))  # the left move recovers: free
    self.assertAlmostEqual(self._out(toward_right, 4 * self.CAP), self.CAP, delta=1e-8)  # further toward it: capped
    self.assertAlmostEqual(self._out(_lanes(heading=-0.01), -4 * self.CAP), -self.CAP, delta=1e-8)  # the cap stops 0.25 m/s in time
    # 0.5 m/s: the cap alone would stop it 1.38 m short, but not after the 0.14 m covered before steering responds
    self.assertEqual(_bits(self._out(_lanes(heading=-0.02), -4 * self.CAP)), _bits(-4 * self.CAP))
    # each line's own slope: a narrowing lane closes on both sides, a widening one on neither
    for move in (-4 * self.CAP, 4 * self.CAP):
      self.assertEqual(_bits(self._out(_lanes(narrowing=0.03), move)), _bits(move))
      self.assertAlmostEqual(self._out(_lanes(narrowing=-0.03), move), math.copysign(self.CAP, move), delta=1e-8)

  def test_a_move_away_from_an_unseen_line_is_never_capped_whatever_the_road_edge_says(self):
    # 1.1 m from a left line the model does not see: the right move may be the recovery from it, so it passes...
    for edges in (None, (-5.0, 5.0, 0.1), (-1.2, 1.2, 0.0)):
      with self.subTest(edges=edges):
        self.assertEqual(_bits(self._out(_lanes(offset=-0.7, probs=(0.1, 0.9), edges=edges), 4 * self.CAP)), _bits(4 * self.CAP))
    # ...and a far road edge certifies nothing: it is the road's boundary, not the lane's, so it cannot re-enable the cap
    self.assertEqual(_bits(self._out(_lanes(offset=-0.7, probs=(0.1, 0.9), edges=(-5.0, 5.0, 0.1)), 4 * self.CAP)), _bits(4 * self.CAP))
    # the same road edges with the line seen do not change a decision the line already makes
    self.assertAlmostEqual(self._out(_lanes(offset=-0.7, edges=(-5.0, 5.0, 0.1)), 4 * self.CAP), 4 * self.CAP, delta=1e-12)

  def test_lane_changes_unseen_lines_and_corner_speeds_are_left_alone(self):
    for model, v in ((_lanes(lane_change=log.LaneChangeState.laneChangeStarting), V_EGO), (_lanes(probs=(0.3, 0.3)), V_EGO), (_lanes(), 7.9)):
      with self.subTest(v=v):
        self.assertEqual(_bits(self._out(model, 0.004, v_ego=v)), _bits(0.004))

  def test_line_confidence_speed_and_bend_fade_it_in_continuously(self):
    request = 4 * self.CAP
    half = self._out(_lanes(probs=(0.5, 0.5)), request)
    self.assertAlmostEqual(half, request + 0.5 * (self.CAP - request), delta=1e-12)
    v = 10.0
    self.assertAlmostEqual(self._out(_lanes(), 1.0 / v**2, v_ego=v), 1.0 / v**2 + 0.5 * (MOONPILOT_RECENTER_LAT_ACCEL - 1.0) / v**2, delta=1e-12)
    bend = 0.45 / V_EGO**2  # halfway through the curve fade
    capped = bend + self.CAP
    self.assertAlmostEqual(self._out(_lanes(bend), bend + 4 * self.CAP), bend + 4 * self.CAP + 0.5 * (capped - bend - 4 * self.CAP), delta=1e-8)


class TestSmoothingLead(unittest.TestCase):
  """The lead that cancels MoonpilotPathSmooth's lag, and the request it leaves alone."""

  def test_without_a_lead_or_look_ahead_the_action_is_bit_exact(self):
    # The response-horizon re-sample is not applied: modeld's action already reads the plan there.
    self.assertEqual(_bits(_update(_corner(0.004, t_start=0.3), 0.03125, lead=0.0)), _bits(0.03125))

  def test_a_tightening_ramp_is_led_by_exactly_slope_times_tau(self):
    for tau in TAUS[1:]:
      with self.subTest(tau=tau):
        self.assertAlmostEqual(_update(_ramp(slope=0.002), 0.01, lead=tau) - 0.01, 0.002 * tau, delta=1e-12)

  def test_the_lead_cancels_the_filters_lag_on_a_ramp(self):
    # Command rising at `slope` per second, led by tau, then filtered: it settles on the unfiltered
    # command, short only the half frame a discrete IIR lags by. Without the lead it trails by tau.
    tau, slope = 0.10, 0.002
    led, plain = PathSmooth(tau), PathSmooth(tau)
    led.update(0.0, False)
    plain.update(0.0, False)
    for n in range(1, 400):
      request = slope * n * 0.01
      out_led = led.update(request + slope * tau, True)
      out_plain = plain.update(request, True)
    self.assertAlmostEqual(out_led, request, delta=slope * 0.006)
    self.assertAlmostEqual(request - out_plain, slope * tau, delta=slope * 0.006)

  def test_an_unwinding_turn_is_led_out_too(self):
    self.assertLess(_update(_ramp(kappa0=0.004, slope=-0.002), 0.003), 0.003)

  def test_a_steady_radius_is_left_alone(self):
    self.assertAlmostEqual(_update(_corner(0.004), 0.004), 0.004, delta=1e-15)

  def test_the_lead_is_bounded_in_lateral_acceleration(self):
    for v_ego in (10.0, 20.0, 30.0):
      with self.subTest(v_ego=v_ego):
        out = _update(_ramp(slope=5.0), v_ego=v_ego, lead=0.15)
        self.assertAlmostEqual(out * v_ego**2, MOONPILOT_SMOOTH_LEAD_MAX_LAT_ACCEL, delta=1e-9)

  def test_the_lead_reads_from_the_response_horizon(self):
    # The 0.28 s horizon plus the 0.05 s lead: a turn the path's grid starts inside that window is led
    # into, one the grid starts after it is not seen yet.
    self.assertGreater(_update(_corner(0.004, t_start=0.3)), 0.0)
    self.assertEqual(_bits(_update(_corner(0.004, t_start=0.36))), _bits(0.0))


class TestExactFallback(unittest.TestCase):
  def assert_pass_through(self, model, base=0.03125, **overrides):
    out = _update(model, base, **overrides)
    self.assertEqual(_bits(out), _bits(base))

  def test_the_fixtures_are_corrected_when_the_input_is_good(self):
    # Each pass-through below would otherwise be vacuous.
    for model in (_ramp(), _corner(0.004, t_start=0.3)):
      self.assertNotEqual(_bits(_update(model, 0.03125)), _bits(0.03125))

  def test_an_invalid_or_dead_model_is_pass_through(self):
    self.assert_pass_through(_corner(0.004, t_start=0.3), model_valid=False)

  def test_stale_and_future_capture_or_receive_times_are_pass_through(self):
    self.assertEqual(MOONPILOT_MAX_MODEL_AGE, 2 * DT_MDL)
    cases = (
      ("stale capture", _ramp(timestamp=NOW - MOONPILOT_MAX_MODEL_AGE - 0.001), {}),
      ("future capture", _ramp(timestamp=NOW + 0.001), {}),
      ("zero capture timestamp", _ramp(timestamp=0.0), {}),
      ("stale receive", _ramp(), {"model_recv_time": NOW - MOONPILOT_MAX_MODEL_AGE - 0.001}),
      ("future receive", _ramp(), {"model_recv_time": NOW + 0.001}),
      ("non-finite receive", _ramp(), {"model_recv_time": float("nan")}),
      ("non-finite control time", _ramp(), {"now": float("nan")}),
    )
    for name, model, arguments in cases:
      with self.subTest(name=name):
        self.assert_pass_through(model, **arguments)

  def test_low_or_non_finite_speed_is_pass_through(self):
    model = _corner(0.004, t_start=0.3)
    for v_ego in (MOONPILOT_PREVIEW_MIN_SPEED, 0.0, float("nan"), float("inf"), -float("inf")):
      with self.subTest(v_ego=v_ego):
        self.assert_pass_through(model, v_ego=v_ego)

  def test_negative_or_non_finite_delay_is_pass_through(self):
    model = _corner(0.004, t_start=0.3)
    for delay in (-0.001, float("nan"), float("inf"), -float("inf")):
      with self.subTest(delay=delay):
        self.assert_pass_through(model, lat_delay=delay)

  def test_a_non_positive_or_out_of_grid_horizon_is_pass_through(self):
    self.assert_pass_through(_ramp(timestamp=NOW), lat_delay=0.0)
    self.assert_pass_through(_ramp(), lat_delay=float(T_IDXS[-1]) + 0.01)

  def test_empty_or_mismatched_arrays_are_pass_through(self):
    bare = messaging.new_message("modelV2").modelV2
    bare.timestampEof = int((NOW - CAPTURE_AGE) * 1e9)
    self.assert_pass_through(bare)

    mismatched = _ramp()
    mismatched.orientation.z = list(mismatched.orientation.z)[:-1]
    self.assert_pass_through(mismatched)

  def test_different_or_non_increasing_time_grids_are_pass_through(self):
    different = _ramp()
    rate_times = list(different.orientationRate.t)
    rate_times[2] += 0.001
    different.orientationRate.t = rate_times
    self.assert_pass_through(different)

    non_increasing = _ramp()
    bad_times = list(non_increasing.orientation.t)
    bad_times[2] = bad_times[1]
    non_increasing.orientation.t = bad_times
    non_increasing.orientationRate.t = bad_times
    self.assert_pass_through(non_increasing)

  def test_non_finite_trajectory_arrays_are_pass_through(self):
    for field in ("orientation.z", "orientationRate.z", "orientation.t", "orientationRate.t"):
      with self.subTest(field=field):
        model = _ramp()
        owner_name, field_name = field.split(".")
        owner = getattr(model, owner_name)
        values = list(getattr(owner, field_name))
        values[2] = float("nan")
        setattr(owner, field_name, values)
        self.assert_pass_through(model)

  def test_a_non_finite_base_request_is_returned_unchanged(self):
    model = _corner(0.004, t_start=0.3)
    for base in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(base=base):
        self.assert_pass_through(model, base=base)


class TestSmoothFactory(unittest.TestCase):
  def test_the_choice_off_is_none(self):
    self.assertIsNone(moonpilot_path_smooth(params(0)))

  def test_each_choice_builds_the_filter_at_its_tau(self):
    for index in range(1, len(TAUS)):
      with self.subTest(index=index):
        smooth = moonpilot_path_smooth(params(0, {PATH_SMOOTH.key: index}))
        assert isinstance(smooth, PathSmooth)
        self.assertEqual(smooth.tau, TAUS[index])

  def test_an_out_of_range_choice_reads_as_off(self):
    for value in (-1, len(TAUS), None):
      with self.subTest(value=value):
        self.assertIsNone(moonpilot_path_smooth(params(0, {PATH_SMOOTH.key: value})))


class TestPathSmooth(unittest.TestCase):
  """The request filter: the exact step response, engage continuity, and the pass-through edges."""

  TAU = 0.15  # the step response below counts 15 frames to one tau
  ALPHA = 1.0 - math.exp(-0.01 / TAU)

  def test_a_step_follows_the_closed_form_response(self):
    smooth = PathSmooth(self.TAU)
    smooth.update(0.0, False)  # seed the state at rest, as controlsd does while not steering
    outs = [smooth.update(1.0, True) for _ in range(15)]
    self.assertAlmostEqual(outs[0], self.ALPHA, delta=1e-12)
    # 15 control frames at dt=0.01 is tau: (1 - alpha)^n is exp(-n*dt/tau) exactly.
    self.assertAlmostEqual(outs[-1], 1.0 - math.exp(-1.0), delta=1e-12)
    self.assertTrue(all(b >= a for a, b in zip(outs, outs[1:], strict=False)))

  def test_a_constant_input_is_reached_and_held(self):
    smooth = PathSmooth(self.TAU)
    out = 0.0
    for _ in range(2000):
      out = smooth.update(0.25, True)
    self.assertAlmostEqual(out, 0.25, delta=1e-12)

  def test_while_not_steering_the_input_passes_through_bit_exact(self):
    smooth = PathSmooth(self.TAU)
    smooth.update(1.0, True)  # a dirty state from before the disengage
    for value in (0.0, -0.4, 1e-6):
      with self.subTest(value=value):
        self.assertEqual(smooth.update(value, False), value)

  def test_a_disengage_resets_the_state_to_the_car_s_path(self):
    smooth = PathSmooth(self.TAU)
    smooth.update(1.0, True)
    smooth.update(0.02, False)  # while not steering the state tracks the actual curvature
    self.assertEqual(smooth.update(0.02, True), 0.02)  # re-engage never re-injects the 1.0

  def test_the_first_call_on_a_fresh_filter_returns_the_input(self):
    self.assertEqual(PathSmooth(self.TAU).update(0.3, True), 0.3)

  def test_a_non_finite_request_passes_through_and_holds_the_state(self):
    smooth = PathSmooth(self.TAU)
    smooth.update(0.1, True)  # the first call passes through and lands the state on 0.1
    for value in (float("nan"), float("inf"), -float("inf")):
      with self.subTest(value=value):
        out = smooth.update(value, True)
        self.assertTrue(math.isnan(out) if math.isnan(value) else out == value)
    # The next finite frame filters from the held 0.1: a poisoned state would come back nan.
    self.assertAlmostEqual(smooth.update(0.2, True), 0.1 + self.ALPHA * 0.1, delta=1e-12)


if __name__ == "__main__":
  unittest.main()
