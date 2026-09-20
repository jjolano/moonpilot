"""Response-aligned steering behavior and its restart-gated factory."""

import struct
import unittest
from typing import cast
from unittest import mock

import numpy as np

import moonpilot.curvature as curvature_mod
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.curvature import (
  MOONPILOT_MAX_MODEL_AGE,
  MOONPILOT_PREVIEW_GAIN,
  MOONPILOT_PREVIEW_MAX_LAT_ACCEL,
  MOONPILOT_PREVIEW_MIN_SPEED,
  MoonpilotCurvature,
  moonpilot_curvature,
)

T_IDXS = np.asarray(ModelConstants.T_IDXS)
NOW = 10.0
CAPTURE_AGE = 0.08
RECEIVE_AGE = 0.02
LAT_DELAY = 0.2
V_EGO = 25.0


class FakeParams:
  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


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


def _update(reference, model, desired_curvature=0.0, **overrides):
  arguments = {
    "v_ego": V_EGO,
    "lat_delay": LAT_DELAY,
    "model_recv_time": NOW - RECEIVE_AGE,
    "now": NOW,
    "model_valid": True,
  }
  arguments.update(overrides)
  return reference.update(model, desired_curvature, **arguments)


def _bits(value):
  return struct.pack("!d", value)


class TestFactory(unittest.TestCase):
  def test_the_toggle_off_is_none(self):
    self.assertIsNone(moonpilot_curvature(_params(False)))

  def test_the_toggle_on_builds_the_reference(self):
    self.assertIsInstance(moonpilot_curvature(_params(True)), MoonpilotCurvature)

  def test_it_is_chosen_once_so_the_toggle_takes_a_restart(self):
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(False)):
      self.assertIsNone(moonpilot_curvature())
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(True)):
      self.assertIsInstance(moonpilot_curvature(), MoonpilotCurvature)


class TestResponseReference(unittest.TestCase):
  def test_capture_age_and_delay_define_the_response_horizon(self):
    model = _corner(0.004, t_start=0.3)
    base = _request(model)
    response_horizon = NOW - model.timestampEof * 1e-9 + LAT_DELAY
    response = _response(model)
    limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / V_EGO**2
    expected = base + float(np.clip(MOONPILOT_PREVIEW_GAIN * (response - base), -limit, limit))

    with mock.patch.object(curvature_mod, "get_curvature_from_plan", wraps=get_curvature_from_plan) as sample:
      out = _update(MoonpilotCurvature(), model, base)

    self.assertAlmostEqual(response_horizon, 0.28, delta=1e-12)
    self.assertAlmostEqual(sample.call_args.args[4], 0.28, delta=1e-12)
    self.assertAlmostEqual(out, expected, delta=1e-12)

  def test_control_time_advances_tightening_and_exiting_references(self):
    tightening = _corner(0.004, t_start=0.3)
    tightening_base = _request(tightening, action_t=0.25)
    tightening_now = _update(MoonpilotCurvature(), tightening, tightening_base, lat_delay=0.25)
    tightening_later = _update(MoonpilotCurvature(), tightening, tightening_base, lat_delay=0.25, now=NOW + 0.01)
    self.assertGreater(tightening_later, tightening_now)

    exiting = _corner(0.004, t_end=0.3)
    exiting_base = _request(exiting, action_t=0.25)
    exiting_now = _update(MoonpilotCurvature(), exiting, exiting_base, lat_delay=0.25)
    exiting_later = _update(MoonpilotCurvature(), exiting, exiting_base, lat_delay=0.25, now=NOW + 0.01)
    self.assertLess(exiting_later, exiting_now)

  def test_a_steady_radius_is_unchanged_as_time_advances(self):
    model = _corner(0.004)
    base = _request(model)
    self.assertAlmostEqual(_update(MoonpilotCurvature(), model, base), base, delta=1e-12)
    self.assertAlmostEqual(_update(MoonpilotCurvature(), model, base, now=NOW + 0.01), base, delta=1e-12)

  def test_the_first_call_returns_the_full_bounded_correction(self):
    model = _corner(0.2, t_start=0.3)
    out = _update(MoonpilotCurvature(), model)
    limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / V_EGO**2
    self.assertAlmostEqual(abs(out), limit, delta=1e-12)

  def test_the_bound_scales_in_lateral_acceleration(self):
    model = _corner(0.2, t_start=0.3)
    for v_ego in (10.0, 20.0, 30.0):
      with self.subTest(v_ego=v_ego):
        out = _update(MoonpilotCurvature(), model, v_ego=v_ego)
        self.assertLessEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL + 1e-12)
        self.assertAlmostEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL, delta=1e-9)

  def test_an_action_head_remains_the_anchor_when_it_differs_from_the_path(self):
    model = _corner(0.004, t_start=0.3)
    response = _response(model)
    base = response + 0.002
    out = _update(MoonpilotCurvature(), model, base)
    expected = base + MOONPILOT_PREVIEW_GAIN * (response - base)

    self.assertAlmostEqual(out, expected, delta=1e-12)
    self.assertNotAlmostEqual(out, response, delta=1e-9)
    self.assertLess(response, out)
    self.assertLess(out, base)


class TestExactFallback(unittest.TestCase):
  def assert_pass_through(self, model, base=0.03125, **overrides):
    out = _update(MoonpilotCurvature(), model, base, **overrides)
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


if __name__ == "__main__":
  unittest.main()
