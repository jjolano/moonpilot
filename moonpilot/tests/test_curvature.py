"""Response-aligned steering and the curvature request filter: behavior and their restart-gated factories."""

import math
import struct
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

import moonpilot.curvature as curvature_mod
from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.curvature import (
  MOONPILOT_LOOKAHEAD,
  MOONPILOT_MAX_MODEL_AGE,
  MOONPILOT_PREVIEW_GAIN,
  MOONPILOT_PREVIEW_MAX_LAT_ACCEL,
  MOONPILOT_PREVIEW_MIN_SPEED,
  MOONPILOT_SMOOTH_TAUS,
  PathSmooth,
  moonpilot_curvature,
  moonpilot_path_smooth,
  response_aligned_curvature,
)
from moonpilot.features import PATH_LOOKAHEAD, PATH_SMOOTH
from moonpilot.tests.fakes import FakeParams, _params

NO_LOOKAHEAD = {PATH_LOOKAHEAD.key: 0}

ROOT = Path(__file__).resolve().parents[2]
T_IDXS = np.asarray(ModelConstants.T_IDXS)
NOW = 10.0
CAPTURE_AGE = 0.08
RECEIVE_AGE = 0.02
LAT_DELAY = 0.2
V_EGO = 25.0


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


def _request(model, v_ego=V_EGO, action_t=LAT_DELAY):
  return float(get_curvature_from_plan(model.orientation.z, model.orientationRate.z, model.orientation.t, v_ego, action_t))


def _response(model, *, v_ego=V_EGO, now=NOW, lat_delay=LAT_DELAY):
  horizon = now - model.timestampEof * 1e-9 + lat_delay
  return float(get_curvature_from_plan(model.orientation.z, model.orientationRate.z, model.orientation.t, v_ego, horizon))


def _update(model, desired_curvature=0.0, **overrides):
  arguments: dict[str, Any] = {
    "v_ego": V_EGO,
    "lat_delay": LAT_DELAY,
    "model_recv_time": NOW - RECEIVE_AGE,
    "now": NOW,
    "model_valid": True,
  }
  arguments.update(overrides)
  return response_aligned_curvature(model, desired_curvature, **arguments)


def _bits(value):
  return struct.pack("!d", value)


class TestFactory(unittest.TestCase):
  def test_the_toggle_off_is_none(self):
    self.assertIsNone(moonpilot_curvature(_params(False)))

  def test_the_toggle_on_builds_the_reference(self):
    self.assertIs(moonpilot_curvature(_params(True, NO_LOOKAHEAD)), response_aligned_curvature)

  def test_it_is_chosen_once_so_the_toggle_takes_a_restart(self):
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(False)):
      self.assertIsNone(moonpilot_curvature())
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(True, NO_LOOKAHEAD)):
      self.assertIs(moonpilot_curvature(), response_aligned_curvature)


class TestLookahead(unittest.TestCase):
  """The look-ahead window: a coming turn pulls the request in early; it never takes turn away from today's request."""

  HORIZON = CAPTURE_AGE + LAT_DELAY

  def _out(self, model, level, desired=0.0):
    reference = moonpilot_curvature(_params(True, {PATH_LOOKAHEAD.key: level}))
    assert reference is not None
    return reference(model, desired, v_ego=V_EGO, lat_delay=LAT_DELAY, model_recv_time=NOW - RECEIVE_AGE, now=NOW, model_valid=True)

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
      today = self._out(model, 0, desired=0.002)
      for level in (1, 2, 3):
        with self.subTest(level=level):
          self.assertEqual(_bits(self._out(model, level, desired=0.002)), _bits(today))

  def test_the_lead_in_is_capped_per_level(self):
    model = _corner(0.05, t_start=self.HORIZON + 0.1)
    for level in (1, 2, 3):
      lookahead = MOONPILOT_LOOKAHEAD[level]
      assert lookahead is not None
      with self.subTest(level=level):
        self.assertAlmostEqual(self._out(model, level), lookahead[2] / V_EGO ** 2, delta=1e-12)

  def test_a_window_past_the_published_path_keeps_todays_request(self):
    model = _corner(0.002)
    lookahead = MOONPILOT_LOOKAHEAD[3]
    assert lookahead is not None
    late = T_IDXS[-1] - CAPTURE_AGE - lookahead[0] + 0.01
    today = _update(model, 0.03125, lat_delay=late)
    out = _update(model, 0.03125, lat_delay=late, lookahead=lookahead)
    self.assertEqual(_bits(out), _bits(today))

  def test_without_response_aligned_steering_there_is_no_look_ahead(self):
    self.assertIsNone(moonpilot_curvature(_params(False, {PATH_LOOKAHEAD.key: 3})))


class TestResponseReference(unittest.TestCase):
  def test_capture_age_and_delay_define_the_response_horizon(self):
    model = _corner(0.004, t_start=0.3)
    base = _request(model)
    response_horizon = NOW - model.timestampEof * 1e-9 + LAT_DELAY
    response = _response(model)
    limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / V_EGO**2
    expected = base + float(np.clip(MOONPILOT_PREVIEW_GAIN * (response - base), -limit, limit))

    with mock.patch.object(curvature_mod, "get_curvature_from_plan", wraps=get_curvature_from_plan) as sample:
      out = _update(model, base)

    self.assertAlmostEqual(response_horizon, 0.28, delta=1e-12)
    self.assertAlmostEqual(sample.call_args.args[4], 0.28, delta=1e-12)
    self.assertAlmostEqual(out, expected, delta=1e-12)

  def test_control_time_advances_tightening_and_exiting_references(self):
    tightening = _corner(0.004, t_start=0.3)
    tightening_base = _request(tightening, action_t=0.25)
    tightening_now = _update(tightening, tightening_base, lat_delay=0.25)
    tightening_later = _update(tightening, tightening_base, lat_delay=0.25, now=NOW + 0.01)
    self.assertGreater(tightening_later, tightening_now)

    exiting = _corner(0.004, t_end=0.3)
    exiting_base = _request(exiting, action_t=0.25)
    exiting_now = _update(exiting, exiting_base, lat_delay=0.25)
    exiting_later = _update(exiting, exiting_base, lat_delay=0.25, now=NOW + 0.01)
    self.assertLess(exiting_later, exiting_now)

  def test_a_steady_radius_is_unchanged_as_time_advances(self):
    model = _corner(0.004)
    base = _request(model)
    self.assertAlmostEqual(_update(model, base), base, delta=1e-12)
    self.assertAlmostEqual(_update(model, base, now=NOW + 0.01), base, delta=1e-12)

  def test_the_first_call_returns_the_full_bounded_correction(self):
    model = _corner(0.2, t_start=0.3)
    out = _update(model)
    limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / V_EGO**2
    self.assertAlmostEqual(abs(out), limit, delta=1e-12)

  def test_the_bound_scales_in_lateral_acceleration(self):
    model = _corner(0.2, t_start=0.3)
    for v_ego in (10.0, 20.0, 30.0):
      with self.subTest(v_ego=v_ego):
        out = _update(model, v_ego=v_ego)
        self.assertLessEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL + 1e-12)
        self.assertAlmostEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL, delta=1e-9)

  def test_an_action_head_remains_the_anchor_when_it_differs_from_the_path(self):
    model = _corner(0.004, t_start=0.3)
    response = _response(model)
    base = response + 0.002
    out = _update(model, base)
    expected = base + MOONPILOT_PREVIEW_GAIN * (response - base)

    self.assertAlmostEqual(out, expected, delta=1e-12)
    self.assertNotAlmostEqual(out, response, delta=1e-9)
    self.assertLess(response, out)
    self.assertLess(out, base)


class TestExactFallback(unittest.TestCase):
  def assert_pass_through(self, model, base=0.03125, **overrides):
    out = _update(model, base, **overrides)
    self.assertEqual(_bits(out), _bits(base))

  def test_an_invalid_or_dead_model_is_pass_through(self):
    self.assert_pass_through(_corner(0.004, t_start=0.3), model_valid=False)

  def test_stale_and_future_capture_or_receive_times_are_pass_through(self):
    self.assertEqual(MOONPILOT_MAX_MODEL_AGE, 2 * DT_MDL)
    cases = (
      ("stale capture", _model(0.02, timestamp=NOW - MOONPILOT_MAX_MODEL_AGE - 0.001), {}),
      ("future capture", _model(0.02, timestamp=NOW + 0.001), {}),
      ("zero capture timestamp", _model(0.02, timestamp=0.0), {}),
      ("stale receive", _model(0.02), {"model_recv_time": NOW - MOONPILOT_MAX_MODEL_AGE - 0.001}),
      ("future receive", _model(0.02), {"model_recv_time": NOW + 0.001}),
      ("non-finite receive", _model(0.02), {"model_recv_time": float("nan")}),
      ("non-finite control time", _model(0.02), {"now": float("nan")}),
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
    self.assert_pass_through(_model(0.02, timestamp=NOW), lat_delay=0.0)
    self.assert_pass_through(_model(0.02), lat_delay=float(T_IDXS[-1]) + 0.01)

  def test_empty_or_mismatched_arrays_are_pass_through(self):
    bare = messaging.new_message("modelV2").modelV2
    bare.timestampEof = int((NOW - CAPTURE_AGE) * 1e9)
    self.assert_pass_through(bare)

    mismatched = _model(0.02)
    mismatched.orientation.z = list(mismatched.orientation.z)[:-1]
    self.assert_pass_through(mismatched)

  def test_different_or_non_increasing_time_grids_are_pass_through(self):
    different = _model(0.02)
    rate_times = list(different.orientationRate.t)
    rate_times[2] += 0.001
    different.orientationRate.t = rate_times
    self.assert_pass_through(different)

    non_increasing = _model(0.02)
    bad_times = list(non_increasing.orientation.t)
    bad_times[2] = bad_times[1]
    non_increasing.orientation.t = bad_times
    non_increasing.orientationRate.t = bad_times
    self.assert_pass_through(non_increasing)

  def test_non_finite_trajectory_arrays_are_pass_through(self):
    for field in ("orientation.z", "orientationRate.z", "orientation.t", "orientationRate.t"):
      with self.subTest(field=field):
        model = _model(0.02)
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
    self.assertIsNone(moonpilot_path_smooth(_params(0)))

  def test_each_choice_builds_the_filter_its_label_names(self):
    self.assertEqual(len(MOONPILOT_SMOOTH_TAUS), len(PATH_SMOOTH.choices))
    for index, label in enumerate(PATH_SMOOTH.choices[1:], start=1):
      with self.subTest(label=label):
        smooth = moonpilot_path_smooth(_params(index))
        assert isinstance(smooth, PathSmooth)
        self.assertEqual(smooth.tau, MOONPILOT_SMOOTH_TAUS[index])
        self.assertEqual(label, f"{smooth.tau:.2f} s")

  def test_an_out_of_range_choice_reads_as_off(self):
    for value in (-1, len(PATH_SMOOTH.choices), None):
      with self.subTest(value=value):
        self.assertIsNone(moonpilot_path_smooth(_params(value)))

  def test_it_is_chosen_once_so_the_choice_takes_a_restart(self):
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(0)):
      self.assertIsNone(moonpilot_path_smooth())
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(2)):
      self.assertEqual(getattr(moonpilot_path_smooth(), "tau", None), 0.10)


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
