"""The fork's path preview (`moonpilot/curvature.py`), and the seam that reaches it.

What is pinned here is the contract controlsd depends on: the input curvature is the base and only a
bounded delta is ever added, the delta is bounded in lateral-accel terms rather than curvature terms,
the preview really is upstream's own `get_curvature_from_plan` read at a longer action time, and every
state where the preview cannot speak — lateral off, too slow, no yaw arrays, a non-finite path — is a
pass-through rather than the last value held.

`_model` builds a `modelV2` the way modeld fills one: yaw is the model's ORIENTATION, its rate is
ORIENTATION_RATE, and both are on `ModelConstants.T_IDXS`. The request the model would publish is that
same function at the delay-matched action time, so the expected delta is computed here the same way
the module computes it — and the case that matters is the one where the two horizons *disagree*,
which is a curve that is still tightening out at the preview.
"""

import math
import unittest
from typing import cast
from unittest import mock

import numpy as np

import moonpilot.curvature as curvature_mod
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.curvature import (
  MOONPILOT_PREVIEW_GAIN,
  MOONPILOT_PREVIEW_MAX_LAT_ACCEL,
  MOONPILOT_PREVIEW_MIN_SPEED,
  MOONPILOT_PREVIEW_SMOOTH_T,
  MOONPILOT_PREVIEW_T,
  MoonpilotCurvature,
  moonpilot_curvature,
)

T_IDXS = np.array(ModelConstants.T_IDXS)
LAT_DELAY = 0.2


class FakeParams:
  """Duck-typed Params, as `moonpilot/tests/test_longitudinal.py` uses: the factory only ever reads
  one key, and this test must not touch the real param store."""

  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


def _model(psi_rate, psi=None):
  """A `modelV2` with yaw and yaw rate on the model's own grid. `psi` defaults to the integral of
  `psi_rate`, which is the relation the two arrays have in a real message."""
  psi_rate = np.broadcast_to(np.asarray(psi_rate, dtype=float), T_IDXS.shape)
  psi = np.concatenate([[0.0], np.cumsum(0.5 * (psi_rate[1:] + psi_rate[:-1]) * np.diff(T_IDXS))]) if psi is None else psi
  model = messaging.new_message("modelV2").modelV2
  model.orientation = log.XYZTData.new_message(z=[float(v) for v in psi], t=T_IDXS.tolist())
  model.orientationRate = log.XYZTData.new_message(z=[float(v) for v in psi_rate], t=T_IDXS.tolist())
  return model


def _request(model, v_ego, lat_delay=LAT_DELAY):
  """What upstream would command: the same plan read at the delay-matched action time, which is the
  quantity modeld's `desiredCurvature` is built from."""
  return float(get_curvature_from_plan(model.orientation.z, model.orientationRate.z, ModelConstants.T_IDXS, v_ego, lat_delay))


def _corner(kappa, v_path=25.0, t_start=0.0, t_end=float("inf")):
  """A path that is straight, then a constant-radius corner, then straight again: yaw is the integral of
  the yaw rate, which is `kappa * v` between `t_start` and `t_end`.

  This is the shape that separates the two horizons. A *steady* corner reads the same at both, because
  `get_curvature_from_plan` is `2 * psi(t_a) / (v * t_a) - psi_rate[0] / v` and a constant curvature is
  its own fixed point. A corner that has not started yet reads deeper at the longer action time (entry),
  and one that has already ended reads shallower (exit) — which is the whole feature.
  """
  psi_rate = np.where((T_IDXS >= t_start) & (T_IDXS < t_end), kappa * v_path, 0.0)
  return _model(psi_rate)


def _settled(preview, model, v_ego, lat_delay=LAT_DELAY, frames=1000, lat_active=True, request=None):
  """Drive `update` to its fixed point at `DT_CTRL`. A first-order filter on a constant input settles
  in a few time constants, and 1000 frames is 10 s against a 0.25 s constant — 40 time constants, so
  the leftover is below float noise and the settled value *is* the gain."""
  request = _request(model, v_ego, lat_delay) if request is None else request
  out = request
  for _ in range(frames):
    out = preview.update(model, v_ego, lat_delay, lat_active, request)
  return out


class TestFactory(unittest.TestCase):
  def test_the_toggle_off_is_none(self):
    self.assertIsNone(moonpilot_curvature(_params(False)))

  def test_the_toggle_on_builds_the_preview(self):
    self.assertIsInstance(moonpilot_curvature(_params(True)), MoonpilotCurvature)

  def test_it_is_chosen_once_so_the_toggle_takes_a_restart(self):
    """The factory's whole contract: one call, one object, and `controlsd` holds whatever it returned.
    A param flipped mid-drive cannot reach a preview that was never built."""
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(False)):
      self.assertIsNone(moonpilot_curvature())
    with mock.patch.object(curvature_mod, "Params", lambda: FakeParams(True)):
      self.assertIsInstance(moonpilot_curvature(), MoonpilotCurvature)


class TestPassThrough(unittest.TestCase):
  def test_a_bare_message_is_pass_through(self):
    """No yaw arrays at all — the shape a test harness hands in — adds nothing."""
    preview = MoonpilotCurvature()
    bare = messaging.new_message("modelV2").modelV2
    for _ in range(100):
      self.assertEqual(preview.update(bare, 25.0, LAT_DELAY, True, 0.03), 0.03)
    self.assertEqual(preview.delta, 0.0)

  def test_a_straight_path_is_pass_through(self):
    """The preview of a straight road is zero, and the request here is zero too, so the delta is zero
    — exactly, not nearly. (A *non-zero* request against a straight path is the exit side of the same
    mechanism and is a real, bounded delta towards the path: `test_a_corner_that_has_ended_is_given_back`
    is that case. It matters because modeld's request is not always a reading of these arrays — it is the
    model's own `action` head when the model ships one (`modeld.py`) — so the two can legitimately
    disagree about the same moment.)"""
    preview = MoonpilotCurvature()
    model = _model(0.0)
    for _ in range(100):
      self.assertEqual(preview.update(model, 25.0, LAT_DELAY, True, 0.0), 0.0)
    self.assertEqual(preview.delta, 0.0)

  def test_a_non_finite_path_is_pass_through(self):
    """One NaN in the yaw array would otherwise travel through the filter and into the command."""
    preview = MoonpilotCurvature()
    psi_rate = np.full(len(T_IDXS), 0.02)
    psi_rate[5] = np.nan
    request = 0.01
    for _ in range(100):
      self.assertEqual(preview.update(_model(psi_rate), 25.0, LAT_DELAY, True, request), request)
    self.assertEqual(preview.delta, 0.0)

  def test_a_yaw_array_of_the_wrong_length_is_refused(self):
    """`get_curvature_from_plan` interpolates over the model's own grid, so a yaw array of another
    length is not the same quantity read at the right times."""
    preview = MoonpilotCurvature()
    model = _model(0.02)
    model.orientation.z = [0.0, 0.01]
    for _ in range(20):
      self.assertEqual(preview.update(model, 25.0, LAT_DELAY, True, 0.03), 0.03)
    self.assertEqual(preview.delta, 0.0)

  def test_below_the_minimum_speed_the_delta_decays(self):
    """Settled at 25 m/s, then a crawl: the delta shrinks toward zero monotonically rather than
    stepping, no frame moves it by more than the filter's own first-order step, and it really does
    reach zero — 1000 frames is 10 s against the 0.25 s constant, 40 time constants."""
    preview = MoonpilotCurvature()
    model = _corner(0.004, t_start=0.3)
    self.assertNotEqual(_settled(preview, model, 25.0), _request(model, 25.0))

    alpha = 1 - math.exp(-DT_CTRL / MOONPILOT_PREVIEW_SMOOTH_T)
    previous = preview.delta
    for _ in range(1000):
      preview.update(model, 3.0, LAT_DELAY, True, _request(model, 3.0))
      self.assertLess(abs(preview.delta), abs(previous))
      self.assertLessEqual(abs(preview.delta - previous), abs(previous) * alpha + 1e-12)
      previous = preview.delta
    self.assertAlmostEqual(preview.delta, 0.0, delta=1e-12)
    self.assertLess(3.0, MOONPILOT_PREVIEW_MIN_SPEED)

  def test_an_inactive_frame_returns_the_input_and_clears_the_delta(self):
    """`controlsd` hands in the *measured* curvature while lateral is off, so a delta carried across
    a re-engage would arrive as a step."""
    preview = MoonpilotCurvature()
    model = _model(0.0, psi=0.006 * T_IDXS**2)
    _settled(preview, model, 25.0)
    self.assertNotEqual(preview.delta, 0.0)

    request = 0.02
    self.assertEqual(preview.update(model, 25.0, LAT_DELAY, False, request), request)
    self.assertEqual(preview.delta, 0.0)
    # and the first re-engaged frame adds only what one frame of the filter allows
    out = preview.update(model, 25.0, LAT_DELAY, True, request)
    alpha = 1 - math.exp(-DT_CTRL / MOONPILOT_PREVIEW_SMOOTH_T)
    self.assertLessEqual(abs(out - request), abs(preview._raw_delta(model, 25.0, LAT_DELAY, request)) * alpha + 1e-12)


class TestPreview(unittest.TestCase):
  def test_a_tightening_curve_is_entered_earlier(self):
    """The case the feature exists for: a corner that has not started at the delay-matched action time
    but is inside the preview window, so the fork's reading is deeper than the model's. The value is the
    gain on the difference of two readings of upstream's own function — pinned exactly rather than by
    sign, so a changed horizon or a changed gain fails here.
    """
    preview = MoonpilotCurvature()
    model = _corner(0.004, t_start=0.3)
    request = _request(model, 25.0)
    out = _settled(preview, model, 25.0)
    preview_curv = get_curvature_from_plan(model.orientation.z, model.orientationRate.z, ModelConstants.T_IDXS, 25.0, LAT_DELAY + MOONPILOT_PREVIEW_T)
    self.assertGreater(abs(preview_curv), abs(request))
    self.assertAlmostEqual(out - request, MOONPILOT_PREVIEW_GAIN * (preview_curv - request), delta=1e-9)
    self.assertGreater(abs(out), abs(request))
    self.assertEqual(math.copysign(1.0, out), math.copysign(1.0, request))

  def test_a_corner_that_has_ended_is_given_back(self):
    """The sign is not one-way: a corner the model is *leaving* reads shallower out at the preview, so
    the delta comes back the other way and the car does not keep turning into a road that has
    straightened."""
    preview = MoonpilotCurvature()
    model = _corner(0.004, t_start=0.0, t_end=0.3)
    request = _request(model, 25.0)
    preview_curv = get_curvature_from_plan(model.orientation.z, model.orientationRate.z, ModelConstants.T_IDXS, 25.0, LAT_DELAY + MOONPILOT_PREVIEW_T)
    self.assertGreater(request, 0.0)
    self.assertLess(preview_curv, request)
    out = _settled(preview, model, 25.0)
    self.assertLess(out, request)
    self.assertGreater(out, 0.0)  # still a turn, just a shallower one

  def test_a_steady_corner_is_left_alone(self):
    """A constant-radius corner is `get_curvature_from_plan`'s own fixed point — `2 * psi(t_a) / (v *
    t_a) - psi_rate[0] / v` reads the same at both horizons — so the feature has nothing to say about
    one. That is what keeps the bounded cases below about *changes* in curvature rather than curvature
    itself."""
    preview = MoonpilotCurvature()
    model = _corner(0.004)
    request = _request(model, 25.0)
    self.assertAlmostEqual(_settled(preview, model, 25.0), request, delta=1e-9)

  def test_the_delta_is_bounded_in_lateral_accel(self):
    """What bounds the feature is a lateral accel, not a curvature: the gain on this raw difference is
    far past the ceiling at 30 m/s, and the ceiling is what is delivered."""
    v_ego = 30.0
    model = _corner(0.2, t_start=0.3)
    self.assertGreater(abs(MOONPILOT_PREVIEW_GAIN * _request(model, v_ego)), MOONPILOT_PREVIEW_MAX_LAT_ACCEL / v_ego**2)
    preview = MoonpilotCurvature()
    out = _settled(preview, model, v_ego, request=0.0)
    self.assertAlmostEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL, delta=1e-6)

  def test_the_bound_scales_with_the_speed(self):
    """The ceiling is a fixed lateral accel, so at a quarter of the speed it admits sixteen times the
    curvature — which is what keeps the bound meaningful at both ends of the speed range."""
    model = _corner(0.2, t_start=0.3)
    for v_ego in (10.0, 20.0):
      with self.subTest(v_ego=v_ego):
        preview = MoonpilotCurvature()
        out = _settled(preview, model, v_ego, request=0.0)
        self.assertAlmostEqual(abs(out) * v_ego**2, MOONPILOT_PREVIEW_MAX_LAT_ACCEL, delta=1e-6)

  def test_the_smoothing_runs_at_the_control_rate(self):
    """`smooth_value`'s default `dt` is `DT_MDL`, and controlsd runs at 100 Hz: with the default the
    filter would move five times too fast. One frame's step is `alpha` of the raw delta at `DT_CTRL` —
    pinned, because nothing else in the module would notice the wrong constant."""
    preview = MoonpilotCurvature()
    model = _corner(0.004, t_start=0.3)
    request = _request(model, 25.0)
    raw = preview._raw_delta(model, 25.0, LAT_DELAY, request)
    out = preview.update(model, 25.0, LAT_DELAY, True, request)
    alpha = 1 - math.exp(-DT_CTRL / MOONPILOT_PREVIEW_SMOOTH_T)
    self.assertAlmostEqual(out - request, raw * alpha, delta=1e-12)
    # the default-dt step, for the reader: five times more of the delta in the same frame
    self.assertNotAlmostEqual(alpha, 1 - math.exp(-0.05 / MOONPILOT_PREVIEW_SMOOTH_T), delta=1e-6)

  def test_the_delta_converges_on_the_gain(self):
    """A first-order filter on a constant input converges to the input, and each frame closes the gap
    by `alpha` — so after N frames the leftover is `(1-alpha)^N` of the first delta, which is what makes
    the settled value the gain rather than something else."""
    preview = MoonpilotCurvature()
    model = _corner(0.004, t_start=0.3)
    request = _request(model, 25.0)
    raw = preview._raw_delta(model, 25.0, LAT_DELAY, request)
    alpha = 1 - math.exp(-DT_CTRL / MOONPILOT_PREVIEW_SMOOTH_T)
    for n in range(1, 6):
      preview.delta = 0.0
      for _ in range(n):
        preview.update(model, 25.0, LAT_DELAY, True, request)
      self.assertAlmostEqual(preview.delta, raw * (1 - (1 - alpha) ** n), delta=1e-12)

  def test_a_nan_request_cannot_produce_a_delta(self):
    """A NaN makes every comparison false, so nothing upstream would catch one — and this module's
    guarantee is that it never *adds* one: the preview refuses to score itself against a request it
    cannot read, so the delta stays finite and the caller's own value is what comes back."""
    preview = MoonpilotCurvature()
    out = preview.update(_corner(0.004, t_start=0.3), 25.0, LAT_DELAY, True, float("nan"))
    self.assertEqual(preview.delta, 0.0)
    self.assertTrue(math.isnan(out))  # the caller's value, unchanged, not something invented here


if __name__ == "__main__":
  unittest.main()
