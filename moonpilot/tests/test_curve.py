"""The fork's curve math (`moonpilot/curve.py`): the budget, the targets, the three terms, and the
learned scale.

Every expected value here is computed from the module's own constants, so a constant that moves takes
its test with it rather than leaving a number behind that no longer means anything. The properties the
planner's seam depends on are the ones pinned: `None` for a model message with no path, `ACCEL_MAX` as
the inactive sentinel for each term, the bank correction copied from `clip_curvature`, and
`applied()`'s one-sided clamp — the rule that a learned scale can only ever make the plan more
conservative.
"""

import math
import unittest

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.interfaces import ACCEL_MAX

from moonpilot.curve import (
  MOONPILOT_CURVE_A_LAT,
  MOONPILOT_CURVE_A_LAT_MIN,
  MOONPILOT_CURVE_ACCEL_MIN,
  MOONPILOT_CURVE_BIAS_MAX,
  MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL,
  MOONPILOT_CURVE_BIAS_MIN_SAMPLES,
  MOONPILOT_CURVE_BIAS_RC,
  MOONPILOT_CURVE_BIAS_TRACKING_TOLERANCE,
  MOONPILOT_CURVE_HOLD_MARGIN,
  MOONPILOT_CURVE_HOLD_MIN_CURVATURE,
  MOONPILOT_CURVE_HOLD_MIN_SPEED,
  MOONPILOT_CURVE_J_LAT,
  MOONPILOT_CURVE_MIN_PATH_SPEED,
  MOONPILOT_CURVE_MIN_SLACK,
  MOONPILOT_CURVE_PREVIEW_T,
  MOONPILOT_CURVE_T_IDX,
  MOONPILOT_CURVE_V_MIN,
  LatAccelBiasEstimator,
  curve_accel,
  curve_targets,
  hold_speed,
  lat_accel_budget,
  lat_accel_hold,
  predicted_lat_accel,
)

T_IDXS = np.array(ModelConstants.T_IDXS)
X_IDXS = np.array(ModelConstants.X_IDXS)


def _path(curvature, v_path=20.0, x=None):
  """A `modelV2` message carrying the path arrays `curve_targets` reads. `curvature` is a scalar or an
  array over the samples; `orientationRate.z` is it times the path speed, which is how modeld fills
  the two."""
  x = X_IDXS if x is None else np.asarray(x, dtype=float)
  curv = np.broadcast_to(np.asarray(curvature, dtype=float), x.shape)
  v_path = np.broadcast_to(np.asarray(v_path, dtype=float), x.shape)
  model = messaging.new_message("modelV2").modelV2
  model.position = log.XYZTData.new_message(x=[float(v) for v in x], t=T_IDXS.tolist())
  model.velocity = log.XYZTData.new_message(x=[float(v) for v in v_path], t=T_IDXS.tolist())
  model.orientationRate = log.XYZTData.new_message(z=[float(v) for v in curv * v_path], t=T_IDXS.tolist())
  return model


class TestLatAccelBudget(unittest.TestCase):
  def test_flat_road_is_the_budget_itself(self):
    self.assertAlmostEqual(float(lat_accel_budget(1.0, 0.0)), MOONPILOT_CURVE_A_LAT, delta=1e-12)
    self.assertAlmostEqual(float(lat_accel_budget(-1.0, 0.0)), MOONPILOT_CURVE_A_LAT, delta=1e-12)

  def test_bank_is_credited_the_way_clip_curvature_credits_it(self):
    """A positive roll adds budget to a left turn and takes it from a right one, by exactly
    `g * roll` — `drive_helpers.clip_curvature`'s own shift of the achievable band."""
    roll = 0.05
    self.assertAlmostEqual(float(lat_accel_budget(1.0, roll)), MOONPILOT_CURVE_A_LAT + ACCELERATION_DUE_TO_GRAVITY * roll, delta=1e-12)
    self.assertAlmostEqual(float(lat_accel_budget(-1.0, roll)), MOONPILOT_CURVE_A_LAT - ACCELERATION_DUE_TO_GRAVITY * roll, delta=1e-12)

  def test_a_bank_large_enough_to_go_negative_floors(self):
    """A downhill right-hander in a steep bank would otherwise ask for zero lateral accel, i.e. a
    stop; the floor is what keeps the budget a budget."""
    roll = -(MOONPILOT_CURVE_A_LAT + 0.5) / ACCELERATION_DUE_TO_GRAVITY
    self.assertEqual(float(lat_accel_budget(1.0, roll)), MOONPILOT_CURVE_A_LAT_MIN)

  def test_the_scale_only_ever_divides(self):
    self.assertAlmostEqual(float(lat_accel_budget(1.0, 0.0, 1.25)), MOONPILOT_CURVE_A_LAT / 1.25, delta=1e-12)
    # below 1.0 is not a scale this can be handed: the one-sided clamp is applied here too
    self.assertAlmostEqual(float(lat_accel_budget(1.0, 0.0, 0.8)), MOONPILOT_CURVE_A_LAT, delta=1e-12)

  def test_it_broadcasts_over_per_sample_directions(self):
    """`curve_targets` passes the array of per-sample turn directions and the in-curve term a scalar
    from the same formula."""
    out = lat_accel_budget(np.array([-1.0, 0.0, 1.0]), 0.05)
    self.assertEqual(out.shape, (3,))
    self.assertAlmostEqual(float(out[2] - out[0]), 2 * ACCELERATION_DUE_TO_GRAVITY * 0.05, delta=1e-12)


class TestCurveTargets(unittest.TestCase):
  def test_no_path_arrays_is_no_candidate(self):
    """The inertness statement: a model message with no path takes this branch, which is exactly the
    planner this fork had before the feature existed."""
    self.assertIsNone(curve_targets(messaging.new_message("modelV2").modelV2, True))
    self.assertIsNone(curve_targets(_path(0.0), False))

  def test_the_path_is_referenced_to_the_car_now(self):
    """`position.x` is measured from the pose of the frame the model saw, so `ahead` is the travel the
    caller can account for since then: the returned `x` is exactly that much closer, which is what
    makes it meters ahead of the car now for the command and the published rollout alike."""
    model = _path(0.004)
    base = curve_targets(model, True)
    shifted = curve_targets(model, True, ahead=2.0)
    np.testing.assert_allclose(shifted.x, base.x - 2.0, atol=1e-12)
    np.testing.assert_allclose(shifted.v, base.v, atol=1e-12)
    for ahead in (0.0, -1.0, float("nan"), float("inf")):
      with self.subTest(ahead=ahead):
        np.testing.assert_allclose(curve_targets(model, True, ahead=ahead).x, base.x, atol=1e-12)

  def test_a_constant_radius_gives_the_budget_speed(self):
    curv = 0.004
    target = curve_targets(_path(curv), True)
    self.assertIsNotNone(target)
    expected = math.sqrt(MOONPILOT_CURVE_A_LAT / curv)
    self.assertTrue(np.allclose(target.v, expected, atol=1e-9))

  def test_a_spurious_curvature_cannot_ask_for_a_stop(self):
    target = curve_targets(_path(0.5), True)
    self.assertTrue(np.allclose(target.v, MOONPILOT_CURVE_V_MIN, atol=1e-9))

  def test_the_preview_window_is_bounded(self):
    """Past `MOONPILOT_CURVE_PREVIEW_T` the prediction is not worth braking on, so those samples are
    not admitted at all."""
    target = curve_targets(_path(0.004), True)
    keep = T_IDXS <= MOONPILOT_CURVE_PREVIEW_T
    self.assertEqual(len(target.x), int(keep.sum()))
    self.assertTrue(np.all(T_IDXS[keep] <= MOONPILOT_CURVE_PREVIEW_T))

  def test_a_path_speed_below_the_floor_does_not_divide_by_zero(self):
    """A path whose own speed is ~0 divides by the floor rather than by zero, so the curvature stays
    finite and the target is the one that curvature implies rather than an inf."""
    psi_rate = 0.02
    model = messaging.new_message("modelV2").modelV2
    model.position = log.XYZTData.new_message(x=X_IDXS.tolist())
    model.velocity = log.XYZTData.new_message(x=np.zeros(len(X_IDXS)).tolist())
    model.orientationRate = log.XYZTData.new_message(z=np.full(len(X_IDXS), psi_rate).tolist())
    target = curve_targets(model, True)
    self.assertTrue(np.all(np.isfinite(target.v)))
    expected = math.sqrt(MOONPILOT_CURVE_A_LAT / (psi_rate / MOONPILOT_CURVE_MIN_PATH_SPEED))
    self.assertTrue(np.allclose(target.v, expected, rtol=1e-5))

  def test_the_jerk_ceiling_bounds_a_tightening_entry(self):
    """Curvature ramping with distance at `dκ/ds` makes the jerk ceiling the speed ceiling wherever
    the accel target is above it. At `dκ/ds = 1e-4` 1/m^2 that is `cbrt(J_LAT / 1e-4)`, and the model
    grid's early samples are well under the 19.7 m where the two cross."""
    dk_ds = 1e-4
    curv = dk_ds * X_IDXS  # κ linear in x, so np.gradient reads dk_ds exactly
    keep = T_IDXS <= MOONPILOT_CURVE_PREVIEW_T
    target = curve_targets(_path(curv), True)
    self.assertEqual(len(target.v), int(keep.sum()))
    v_jerk = (MOONPILOT_CURVE_J_LAT / dk_ds) ** (1.0 / 3.0)
    v_accel = np.sqrt(MOONPILOT_CURVE_A_LAT / np.maximum(curv[keep], 1e-6))
    early = v_accel > v_jerk
    self.assertTrue(early.any(), "the sweep never reached the jerk-governed region")
    self.assertTrue(np.allclose(target.v[early], v_jerk, atol=1e-6))

  def test_a_constant_radius_has_no_jerk_ceiling(self):
    """`np.gradient` of a constant is zero, so the ceiling stops binding and the accel target is what
    governs — the far end of the ramp above, checked as its own case."""
    curv = 0.002
    target = curve_targets(_path(curv), True)
    self.assertAlmostEqual(float(target.v[-1]), math.sqrt(MOONPILOT_CURVE_A_LAT / curv), delta=1e-5)

  def test_a_ten_metre_ramp_still_governs(self):
    """The window does not lose the entry this ceiling exists for. A 10 m ramp to a 250 m radius is
    `dk/ds = 4e-4`, i.e. `cbrt(3.0 / 4e-4)` = 19.6 m/s — and it is only two or three samples wide on
    the model's own grid at 20 m/s, so the interpolated curvature under-reads its slope by ~10 %
    (measured here: 20.7 m/s against the budget's 42 at that sample). It still governs, which is the
    point: the entry is slowed before the budget would slow it."""
    ramp = np.clip((X_IDXS - 20.0) / 10.0, 0.0, 1.0) * 0.004
    target = curve_targets(_path(ramp), True)
    keep = T_IDXS <= MOONPILOT_CURVE_PREVIEW_T
    v_jerk = (MOONPILOT_CURVE_J_LAT / 4e-4) ** (1.0 / 3.0)
    inside = (target.x >= 20.0) & (target.x <= 30.0) & (ramp[keep] < 0.004)
    v_budget_here = np.sqrt(MOONPILOT_CURVE_A_LAT / np.maximum(ramp[keep][inside], 1e-6))
    self.assertTrue(inside.any(), "the model grid has no sample inside the ramp")
    self.assertTrue(np.all(target.v[inside] < v_budget_here))
    self.assertLessEqual(float(target.v[inside].min()), v_jerk * 1.15)

  def test_one_sample_of_curvature_noise_does_not_govern_the_ceiling(self):
    """The ceiling is differenced over `MOONPILOT_CURVE_JERK_STEP` meters rather than between the
    model's own samples, because the pre-brake takes the *deepest* sample of the window: at the sample
    scale one bad sample governs the whole term. The corpus says which samples those are — of the 741
    frames where only this ceiling asks for braking, the 33 % the window drops sit at a median 0.47 m
    spacing with incoherent steps (net over summed |dκ| 0.42) and |κ| at 0.6x its own neighbourhood,
    while the 67 % it keeps are coherent ramps at 1.7 m spacing with |κ| at 1.7x.

    Here the step is 4e-4 across the 0.19 m between two of the model grid's first samples — the
    sub-metre class — so every sample keeps the speed the *budget* gives it, where the sample-scale
    gradient reads that step as a ramp and caps three of them. The budget's own per-sample dip at the
    spike is untouched: it is a level rather than a derivative.
    """
    curv = np.full(len(X_IDXS), 0.001)
    curv[1] += 4e-4  # X_IDXS[1] = 0.19 m, 0.19 m from the sample before it and 0.56 m from the one after
    target = curve_targets(_path(curv), True)
    keep = T_IDXS <= MOONPILOT_CURVE_PREVIEW_T
    v_budget = np.sqrt(MOONPILOT_CURVE_A_LAT / np.maximum(curv[keep], 1e-6))
    self.assertEqual(len(target.v), int(keep.sum()))
    self.assertTrue(np.allclose(target.v, v_budget, rtol=1e-4))

    # what the sample-scale gradient made of the same step: a ramp, capped below the budget
    with np.errstate(divide='ignore', invalid='ignore'):
      dk_ds = np.abs(np.gradient(curv[keep], X_IDXS[keep]))
    v_jerk_raw = np.cbrt(MOONPILOT_CURVE_J_LAT / np.maximum(dk_ds, 1e-9))
    self.assertTrue(np.any(v_jerk_raw < v_budget), "the step was too small to have governed at the sample scale")

  def test_a_nan_sample_is_dropped_rather_than_planned_on(self):
    curv = np.full(len(X_IDXS), 0.004)
    curv[3] = np.nan
    target = curve_targets(_path(curv), True)
    self.assertLess(len(target.x), len(X_IDXS))
    self.assertTrue(np.all(np.isfinite(target.x)) and np.all(np.isfinite(target.v)))

  def test_the_scale_slows_the_target(self):
    curv = 0.004
    plain = float(curve_targets(_path(curv), True).v[0])
    scaled = float(curve_targets(_path(curv), True, scale=1.25).v[0])
    # the message's own arrays are Float32, so the readback carries its precision
    self.assertTrue(math.isclose(plain, math.sqrt(MOONPILOT_CURVE_A_LAT / curv), rel_tol=1e-5))
    self.assertTrue(math.isclose(scaled, math.sqrt(MOONPILOT_CURVE_A_LAT / 1.25 / curv), rel_tol=1e-5))
    self.assertLess(scaled, plain)


class TestCurveAccel(unittest.TestCase):
  def test_the_pre_brake_arrives_at_the_curve_at_its_target(self):
    curv, gap = 0.004, 150.0
    target = curve_targets(_path(curv), True)
    x_ego = float(target.x[0]) - gap
    self.assertAlmostEqual(curve_accel(25.0, x_ego, target), (math.sqrt(MOONPILOT_CURVE_A_LAT / curv) ** 2 - 25.0**2) / (2 * gap), delta=1e-6)

  def test_a_tight_curve_close_in_clamps_to_the_terms_floor(self):
    target = curve_targets(_path(0.05), True)
    self.assertEqual(curve_accel(30.0, float(target.x[0]) - 10.0, target), MOONPILOT_CURVE_ACCEL_MIN)

  def test_the_denominator_is_floored(self):
    """A sample a hair ahead of the car divides by `MOONPILOT_CURVE_MIN_SLACK`, not by ~0."""
    target = curve_targets(_path(0.004), True)
    self.assertEqual(curve_accel(30.0, float(target.x[0]) - 1e-9, target), MOONPILOT_CURVE_ACCEL_MIN)
    self.assertTrue(MOONPILOT_CURVE_MIN_SLACK > 0.0)

  def test_the_inactive_sentinel_is_accel_max(self):
    target = curve_targets(_path(0.004), True)
    self.assertEqual(curve_accel(25.0, 0.0, None), ACCEL_MAX)  # no candidate
    self.assertEqual(curve_accel(5.0, 0.0, target), ACCEL_MAX)  # every target above the ego speed
    self.assertEqual(curve_accel(25.0, float(target.x[-1]) + 1.0, target), ACCEL_MAX)  # all behind us


class TestHoldSpeed(unittest.TestCase):
  def test_it_holds_the_budget_speed_with_the_margin(self):
    curv = 0.008
    self.assertAlmostEqual(
      hold_speed(curv, 20.0, MOONPILOT_CURVE_A_LAT, False, True), math.sqrt(MOONPILOT_CURVE_HOLD_MARGIN * MOONPILOT_CURVE_A_LAT / curv), delta=1e-9
    )

  def test_every_state_where_the_measurement_cannot_speak(self):
    """`inf` is the term's off switch, and each of these is a reason to trust something else instead:
    a standstill, a paramsd that has not validated its own calibration (`vehicleParameters.valid` is
    false in a bare capnp message, which is how the maneuver plant and the existing tests stay
    inert), a speed too low for the steer angle to mean anything, and a curvature at the noise
    floor."""
    tau = 0.008
    self.assertEqual(hold_speed(tau, 20.0, MOONPILOT_CURVE_A_LAT, True, True), math.inf)
    self.assertEqual(hold_speed(tau, 20.0, MOONPILOT_CURVE_A_LAT, False, False), math.inf)
    self.assertEqual(hold_speed(tau, MOONPILOT_CURVE_HOLD_MIN_SPEED, MOONPILOT_CURVE_A_LAT, False, True), math.inf)
    self.assertEqual(hold_speed(MOONPILOT_CURVE_HOLD_MIN_CURVATURE / 2, 20.0, MOONPILOT_CURVE_A_LAT, False, True), math.inf)
    self.assertEqual(hold_speed(float("nan"), 20.0, MOONPILOT_CURVE_A_LAT, False, True), math.inf)


class TestLatAccelHold(unittest.TestCase):
  def test_it_is_proportional_on_the_speed_error(self):
    curv = 0.008
    v_hold = hold_speed(curv, 20.0, MOONPILOT_CURVE_A_LAT, False, True)
    self.assertAlmostEqual(lat_accel_hold(18.0, v_hold), 0.6 * (v_hold - 18.0), delta=1e-9)

  def test_it_clamps_at_the_terms_floor(self):
    v_hold = hold_speed(0.008, 20.0, MOONPILOT_CURVE_A_LAT, False, True)
    self.assertEqual(lat_accel_hold(20.0, v_hold), MOONPILOT_CURVE_ACCEL_MIN)

  def test_it_is_inactive_at_or_below_the_setpoint(self):
    self.assertEqual(lat_accel_hold(16.0, 16.0), ACCEL_MAX)
    self.assertEqual(lat_accel_hold(10.0, math.inf), ACCEL_MAX)


class TestPredictedLatAccel(unittest.TestCase):
  def test_it_reads_the_message_arrays(self):
    """`|κ| v^2` is `|orientationRate.z| * velocity.x`, evaluated at the estimator's pairing time."""
    model = _path(0.004, v_path=np.full(len(X_IDXS), 20.0))
    t = 1.0
    expected = 0.004 * float(np.interp(t, T_IDXS, np.full(len(T_IDXS), 20.0))) ** 2
    self.assertAlmostEqual(predicted_lat_accel(model, t), expected, delta=1e-6)

  def test_a_short_path_predicts_nothing(self):
    self.assertEqual(predicted_lat_accel(messaging.new_message("modelV2").modelV2), 0.0)


class TestLatAccelBiasEstimator(unittest.TestCase):
  def _run(self, est, predicted, measured, frames):
    for _ in range(frames):
      est.update(predicted, measured, True)
    return est

  def test_it_recovers_a_car_that_pulls_harder_than_predicted(self):
    """2.0 predicted against 2.5 realized is a ratio of 1.25, and the filter is the one-order lag the
    constants describe, so the value after N pairs is the closed form rather than a tolerance."""
    est = LatAccelBiasEstimator()
    delay_frames = est.delay_frames
    pairs = MOONPILOT_CURVE_BIAS_MIN_SAMPLES + int(4 * MOONPILOT_CURVE_BIAS_RC / DT_MDL)
    self._run(est, 2.0, 2.5, delay_frames + pairs)
    alpha = DT_MDL / (MOONPILOT_CURVE_BIAS_RC + DT_MDL)
    ratio = 2.5 / 2.0
    self.assertEqual(est.samples, pairs)
    self.assertAlmostEqual(est.estimate, ratio - (ratio - 1.0) * (1 - alpha) ** pairs, delta=1e-9)
    self.assertAlmostEqual(est.applied(), est.estimate, delta=1e-9)

  def test_before_the_sample_threshold_nothing_is_applied(self):
    est = LatAccelBiasEstimator()
    self._run(est, 2.0, 2.5, MOONPILOT_CURVE_BIAS_MIN_SAMPLES - 1)
    self.assertEqual(est.status, "measuring")
    self.assertEqual(est.applied(), 1.0)
    self.assertGreater(est.estimate, 1.0)  # the filter really has moved; it is simply not trusted

  def test_a_car_that_pulls_less_learns_a_value_it_cannot_use(self):
    """The one-sided clamp. A measurement below the prediction is the lateral controller failing to
    track or the model over-predicting, and either way the fork may not take *less* braking than the
    geometry says — so `estimate` records it and `applied()` stays 1.0."""
    est = LatAccelBiasEstimator()
    self._run(est, 2.0, 1.6, est.delay_frames + MOONPILOT_CURVE_BIAS_MIN_SAMPLES + int(4 * MOONPILOT_CURVE_BIAS_RC / DT_MDL))
    self.assertEqual(est.status, "estimated")
    self.assertLess(est.estimate, 1.0)
    self.assertEqual(est.applied(), 1.0)

  def test_the_applied_scale_is_bounded_above(self):
    est = LatAccelBiasEstimator()
    est.seed(5.0, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    self.assertEqual(est.applied(), MOONPILOT_CURVE_BIAS_MAX)

  def test_a_seed_is_used_from_the_first_frame(self):
    est = LatAccelBiasEstimator()
    est.seed(1.3, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.applied(), 1.3, delta=1e-12)

  def test_an_invalid_frame_empties_the_queue(self):
    """Otherwise a prediction from before a disengage or a driver override would score a measurement
    from after it — two different cars in the same ratio."""
    est = LatAccelBiasEstimator()
    est.update(2.0, 2.0, True)
    self.assertEqual(len(est.predicted), 1)
    est.update(2.0, 2.0, False)
    self.assertEqual(len(est.predicted), 0)
    self.assertEqual(est.samples, 0)

  def test_straight_frames_teach_nothing(self):
    """Both magnitudes have to clear the floor: a pair of near-zero values is a ratio of noise."""
    below = MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL - 0.1
    est = LatAccelBiasEstimator()
    self._run(est, below, below, 2 * (est.delay_frames + 50))
    self.assertEqual(est.samples, 0)
    self.assertEqual(est.applied(), 1.0)

  def test_the_pairing_really_is_the_configured_delay_apart(self):
    """The queue's head, not the current sample: a prediction spike scores the measurement
    `delay_frames` later, so a prediction made about now is scored against what the car is doing when
    it arrives there."""
    est = LatAccelBiasEstimator()
    delay_frames = est.delay_frames
    est.update(5.0, 5.0, True)  # the spike, and the queue's head for the first pair
    for _ in range(delay_frames - 1):
      est.update(2.0, 5.0, True)
    self.assertEqual(est.samples, 0, "the queue filled early and a pair was scored off the wrong delay")
    est.update(2.0, 5.0, True)  # the measurement that pairs with the spike
    self.assertEqual(est.samples, 1)
    self.assertAlmostEqual(est.estimate, 5.0 / 5.0, delta=1e-12)

  def test_the_model_grid_is_the_paths_own(self):
    """The target arrays are indexed on the model's grid, which is what makes `x` a distance rather
    than a sample count."""
    self.assertEqual(len(MOONPILOT_CURVE_T_IDX), len(T_IDXS))
    self.assertAlmostEqual(MOONPILOT_CURVE_MIN_PATH_SPEED, 1.0, delta=1e-12)


class TestLatAccelBiasTrackingGate(unittest.TestCase):
  """The planner only learns from the fork's torque logger when that logger says it tracked."""

  @staticmethod
  def _inputs():
    from moonpilot.tests.test_longitudinal import _inputs, _path

    return _inputs(
      v_ego=20.0,
      path=_path(20.0, 0.006),
      steer_angle_deg=10.0,
      vp_valid=True,
      steer_ratio=15.0,
      stiffness_factor=1.0,
    )

  @classmethod
  def _samples(cls, torque=None):
    from moonpilot.tests.test_longitudinal import _planner

    sm = cls._inputs()
    if torque is not None:
      sm['controlsState'].lateralControlState.torqueState = torque
    planner = _planner()
    for _ in range(planner.lat_bias.delay_frames + MOONPILOT_CURVE_BIAS_MIN_SAMPLES):
      planner.update(sm)
    return planner.lat_bias.samples

  def test_a_saturated_torque_log_does_not_teach_bias(self):
    torque = log.ControlsState.LateralTorqueState.new_message(
      version=1000,
      active=True,
      saturated=True,
      actualLateralAccel=2.0,
      desiredLateralAccel=2.0,
    )
    self.assertEqual(self._samples(torque), 0)

  def test_an_in_tolerance_active_torque_log_teaches_bias(self):
    torque = log.ControlsState.LateralTorqueState.new_message(
      version=1000,
      active=True,
      saturated=False,
      actualLateralAccel=2.0,
      desiredLateralAccel=2.0 + MOONPILOT_CURVE_BIAS_TRACKING_TOLERANCE / 2,
    )
    self.assertGreater(self._samples(torque), 0)

  def test_a_non_torque_union_member_keeps_learning(self):
    self.assertGreater(self._samples(), 0)


if __name__ == "__main__":
  unittest.main()
