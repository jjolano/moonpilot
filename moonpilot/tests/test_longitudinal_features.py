"""Curve-speed and squeeze policy tests, through the planner seam.

Split out of `test_longitudinal.py` for the 120 KB size gate; helpers live there.
"""

import math
import unittest

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import custom
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.corridor import MOONPILOT_SQUEEZE_ACCEL_MIN, MOONPILOT_SQUEEZE_FULL_WIDTH, MOONPILOT_SQUEEZE_MIN_WIDTH
from moonpilot.curve import (
  MOONPILOT_CURVE_A_LAT,
  MOONPILOT_CURVE_ACCEL_MIN,
  MOONPILOT_CURVE_BIAS_KEY,
  MOONPILOT_CURVE_BIAS_MIN_SAMPLES,
  MOONPILOT_CURVE_BIAS_PERSIST_EVERY,
  MOONPILOT_CURVE_HOLD_MARGIN,
  MOONPILOT_CURVE_K_HOLD,
  MOONPILOT_CURVE_PATH_MAX_AGE,
  curve_targets,
  lat_accel_hold,
)
from moonpilot.jerk import (
  MOONPILOT_LONG_JERK_MIN_SAMPLES,
  MOONPILOT_LONG_JERK_PERSIST_EVERY,
  MOONPILOT_LONG_JERK_SCALE_KEY,
)
from moonpilot.features import FEATURES
from moonpilot.tests.test_longitudinal import ROOT, Source, _cp, _inputs, _path, _planner


class TestCurveSpeed(unittest.TestCase):
  """`MoonpilotCurveSpeed`, through the planner's own seam (`moonpilot/tests/test_curve.py` holds the
  pure math). The two things worth pinning here are the inertness — no path arrays, or a paramsd that
  has not validated its calibration, is exactly the planner this fork had before this existed — and
  that both terms reach the command *and* the published rollout.

  `_inputs` leaves `vehicleParameters.valid` false and the path arrays empty by default, which is also
  why the maneuver suite in `test_longitudinal.py` is untouched: its plant fills `position` and `velocity` but never
  `orientationRate`, and its `vehicleParameters` message is bare.
  """

  V_EGO = 20.0
  # A curve the car can hold at the budget: 1 / 0.006 is a 167 m radius, and at 20 m/s the in-curve
  # setpoint lands below the ego speed by ~1.34 m/s, i.e. inside the term's proportional range rather
  # than pinned on its floor — so the test reads the gain, not the clamp.
  KAPPA = 0.006

  @staticmethod
  def _steer_angle_for(curvature: float, v_ego: float) -> float:
    """The steering angle whose *measured* curvature is `curvature`: `controlsd` and this planner both
    read `-VM.calc_curvature(radians(angle), v, roll)`, and `get_steer_from_curvature` is that model's
    own inverse, so the angle is exact rather than fitted."""
    vm = VehicleModel(_cp())
    vm.update_params(1.0, 15.0)
    return math.degrees(vm.get_steer_from_curvature(-curvature, v_ego, 0.0))

  def _ramp_curve(self, v_path, onset, ramp=40.0, curvature=0.004):
    """A curve that ramps in over `ramp` meters from `onset` and then holds — the shape a path has, and
    the shape the jerk ceiling is meant for. A step would be read as a jerk spike by `np.gradient`."""
    x, _, _ = _path(v_path, 0.0)
    return _path(v_path, np.clip((x - onset) / ramp, 0.0, 1.0) * curvature)

  # The curve the closed loop flies at, and the road layout it sits on: the car starts `START_ONSET`
  # meters short of a curve that ramps in over `RAMP` and then holds. The set speed is the starting
  # speed, so nothing accelerates on the way in and the approach is the feature's alone.
  CLOSED_LOOP_ONSET = 150.0
  CLOSED_LOOP_V0 = 25.0
  CLOSED_LOOP_RAMP = 40.0
  CLOSED_LOOP_KAPPA = 0.004

  def _fly_to_the_curve(self, scale=None, curvature=None):
    """The closed loop: the path is rebuilt each frame from the car's own travel, so the curve is a
    fixed piece of road, and the speed is integrated from the command. Returns the speed where the
    curve reaches full strength (the end of its ramp — measuring at the ramp's *start* would measure a
    target speed that is still rising), the deepest command, and the speeds along the way."""
    curvature = self.CLOSED_LOOP_KAPPA if curvature is None else curvature
    v_ego, travel = self.CLOSED_LOOP_V0, 0.0
    planner = _planner()
    if scale is not None:
      planner.lat_bias.seed(scale, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    speeds, deepest = [v_ego], 0.0
    for _ in range(4000):
      x, _, _ = _path(v_ego, 0.0)
      path = _path(v_ego, np.clip((x + travel - self.CLOSED_LOOP_ONSET) / self.CLOSED_LOOP_RAMP, 0.0, 1.0) * curvature)
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=self.CLOSED_LOOP_V0 * 3.6, path=path, standstill=v_ego < 0.05))
      self.assertGreaterEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
      deepest = min(deepest, planner.output_a_target)
      v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
      travel += v_ego * DT_MDL
      speeds.append(v_ego)
      if travel >= self.CLOSED_LOOP_ONSET + self.CLOSED_LOOP_RAMP:
        return v_ego, deepest, np.array(speeds)
    self.fail("the car never reached the curve")

  def test_no_path_arrays_is_the_planner_without_the_feature(self):
    """The inertness statement the maneuver suite rests on: a model message with no path gives no
    candidate at all, so the command is identical with the feature on and off."""
    self.assertIsNone(curve_targets(_inputs()['modelV2'], True))
    on, off = _planner(), _planner(params_on=False)
    for _ in range(20):
      on.update(_inputs())
      off.update(_inputs())
    self.assertAlmostEqual(on.output_a_target, off.output_a_target, delta=1e-12)
    self.assertEqual(on.source, off.source)

  def test_the_toggle_off_is_the_cruise_planner_with_a_path_present(self):
    """The other half of the inertness: the same curved path with the feature off is the planner
    without the feature, still cruising to its set speed."""
    path = self._ramp_curve(30.0, 60.0)
    off = _planner(params_on=False)
    for _ in range(20):
      off.update(_inputs(v_ego=30.0, v_cruise_kph=130.0, path=path))
    self.assertGreater(off.output_a_target, 0.0)
    self.assertEqual(off.source, Source.cruise)

  def test_it_pre_brakes_before_the_curve(self):
    """The pre-brake is the stopping floor's own kinematics against the curve's distance: from 30 m/s
    towards a target of `sqrt(A_LAT / 0.004)` it is negative at any distance the preview admits, and
    bounded by the term's own floor rather than by `ACCEL_MIN`."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertLess(planner.output_a_target, 0.0)
    self.assertGreaterEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
    self.assertEqual(planner.source, Source.cruise)

  def test_the_path_age_is_the_consuming_tick_not_the_publish_stamp(self):
    """`recv_time` is stamped when the planner reads the message — msgq buffers, so it is the age the
    shift must use — and the publish lag is only the fallback for a clock that does not line up with
    the stamps. The two are not the same number: measured over the corpus, the interval from the
    model publish to the plan's own send is ~16 ms median on top of the ~31 ms publish lag."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(**ages):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, **ages))
      return planner.output_a_target

    plain = command(model_age_s=0.05)
    tick = command(model_age_s=0.05, recv_age_s=0.03)  # the tick sees 0.08 s
    self.assertLess(tick, plain, "the tick anchor did not deepen the pre-brake")
    self.assertAlmostEqual(tick, command(model_age_s=0.08), delta=1e-12)

  def test_an_unusable_clock_falls_back_to_the_publish_lag(self):
    """A replay reads wall time against a log's stamps, so its tick age is absurd; the publish lag is
    computable from the stamps alone and is the fallback. A tick that is merely *late* — past
    `MOONPILOT_CURVE_PATH_MAX_AGE` but inside the sanity bound — is a real staleness and drops the
    shift instead, the same direction the bound always had."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(**ages):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, **ages))
      return planner.output_a_target

    plain = command(model_age_s=0.05)
    self.assertAlmostEqual(command(model_age_s=0.05, recv_age_s=1e6), plain, delta=1e-12)
    self.assertAlmostEqual(command(model_age_s=0.05, recv_age_s=0.5), command(model_age_s=0.0), delta=1e-12)

  def test_the_pre_brake_is_the_terms_own_floor_at_the_deepest_point(self):
    """At 30 m/s even a 60 m approach to the 250 m-radius curve's ~20.6 m/s target needs more than the term's authority, so
    what bounds this is `MOONPILOT_CURVE_ACCEL_MIN` and not the geometry — the actuator's own
    `ACCEL_MIN` is never reached, which is the point of the term's floor."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertAlmostEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN, delta=1e-6)
    self.assertGreater(MOONPILOT_CURVE_ACCEL_MIN, ACCEL_MIN)

  def test_the_closed_loop_arrives_at_the_target(self):
    """Flown, not sampled: the arrival bound is the target plus 1 m/s of slack for the jerk-limited
    onset, and the floor is well below it — the term is a braking authority that converges on the
    target, not a speed limiter with its own dynamics.

    `MOONPILOT_CURVE_PREVIEW_T` admits 4 s of path, ~100 m at 25 m/s, and slowing 25 to 20.62 at the
    term's -1.5 m/s^2 floor needs about 67 m, so the approach is inside the term's authority. At 30 m/s the
    same arithmetic does not close — 158 m of braking against a 117 m preview — which is why this
    flies at 25.
    """
    v_target = math.sqrt(MOONPILOT_CURVE_A_LAT / self.CLOSED_LOOP_KAPPA)
    arrival, deepest, speeds = self._fly_to_the_curve()
    self.assertLessEqual(arrival, v_target + 1.0)
    self.assertGreaterEqual(arrival, 14.0)
    self.assertGreaterEqual(deepest, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
    # it braked on the way in rather than arriving at the curve at its starting speed
    self.assertLess(arrival, self.CLOSED_LOOP_V0 - 1.0)
    self.assertTrue(np.all(np.diff(speeds) <= 1e-9), "the speed rose on the way in")

  def test_a_learned_scale_slows_the_arrival_and_below_one_is_ignored(self):
    """The one-sided clamp, end to end: seeded above 1.0 the car plans for a tighter curve and arrives
    slower, and seeded below it the scale is not a scale — `applied()` floors at 1.0, because the
    opposite direction would hand the feature less braking than the geometry justifies."""
    plain, _, _ = self._fly_to_the_curve()
    biased, _, _ = self._fly_to_the_curve(scale=1.4)
    ignored, _, _ = self._fly_to_the_curve(scale=0.7)
    self.assertLess(biased, plain)
    self.assertAlmostEqual(ignored, plain, delta=1e-9)

  def test_the_in_curve_hold_brakes_with_no_path_at_all(self):
    """The in-curve regulator needs no preview: the curvature is the one the car is *pulling*, from the
    steer angle and the vehicle model. This input carries no path arrays, so nothing but this term can
    be braking.

    The command is the term evaluated where the car will be at the actuator delay — every candidate in
    this planner is, and here that is visible in the magnitude: a proportional gain on a speed that the
    command itself moves solves to `K * (v_hold - v_ego) / (1 + K * action_t)`, i.e. 0.536 of the error
    rather than 0.6. Pinned rather than tolerated, because it is the composition the whole planner
    rests on.
    """
    planner = _planner()
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    sm = _inputs(v_ego=self.V_EGO, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0)
    self.assertEqual(len(sm['modelV2'].position.x), 0, "the input was meant to carry no path at all")
    for _ in range(60):
      planner.update(sm)
    v_hold = math.sqrt(MOONPILOT_CURVE_HOLD_MARGIN * MOONPILOT_CURVE_A_LAT / self.KAPPA)
    gain, action_t = MOONPILOT_CURVE_K_HOLD, _cp().longitudinalActuatorDelay + DT_MDL
    expected = max(gain * (v_hold - self.V_EGO) / (1.0 + gain * action_t), MOONPILOT_CURVE_ACCEL_MIN)
    self.assertAlmostEqual(planner.output_a_target, expected, delta=0.01)
    # the unprojected value the term itself computes, for the reader: deeper, and not what is delivered
    self.assertLess(lat_accel_hold(self.V_EGO, v_hold), planner.output_a_target)

  def test_the_in_curve_hold_needs_a_validated_paramsd(self):
    """`vehicleParameters.valid` is paramsd's own composition of its sensor, angle-offset and roll
    validity, and it is false in a bare capnp message — which is what keeps this term out of the
    maneuver plant and out of every existing test that passes a steer angle. Unvalidated, this input
    is the planner with the feature off: the steer angle still reaches `cruise_accel`'s cornering
    budget, and nothing else.
    """
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    inputs = {"v_ego": self.V_EGO, "steer_angle_deg": angle, "steer_ratio": 15.0, "stiffness_factor": 1.0}
    held = _planner()
    for _ in range(60):
      held.update(_inputs(vp_valid=True, **inputs))
    self.assertLess(held.output_a_target, 0.0)

    unvalidated = _planner()
    for _ in range(60):
      unvalidated.update(_inputs(vp_valid=False, **inputs))
    off = _planner(params_overrides={"MoonpilotCurveSpeed": False})
    for _ in range(60):
      off.update(_inputs(vp_valid=True, **inputs))
    self.assertAlmostEqual(unvalidated.output_a_target, off.output_a_target, delta=1e-12)

  def test_the_in_curve_hold_is_inert_at_low_speed(self):
    """Below `MOONPILOT_CURVE_HOLD_MIN_SPEED` the steer angle implies curvatures no plan should chase,
    so the term is off and the cruise term governs."""
    planner = _planner()
    angle = self._steer_angle_for(self.KAPPA, 3.0)
    for _ in range(60):
      planner.update(_inputs(v_ego=3.0, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0))
    self.assertGreater(planner.output_a_target, 0.0)

  def test_the_bench_correction_reaches_the_budget(self):
    """A bank adds budget to a left turn and takes it from a right one, so the same measured curvature
    on a banked road asks for a different speed. One angle, two rolls, two commands."""
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    commands = []
    for roll in (-0.05, 0.05):
      planner = _planner()
      for _ in range(60):
        planner.update(_inputs(v_ego=self.V_EGO, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0, roll=roll))
      commands.append(planner.output_a_target)
    self.assertNotAlmostEqual(commands[0], commands[1], delta=0.01)

  def test_the_bias_only_learns_while_lateral_is_ours(self):
    """What makes the learned scale a measurement of this car rather than of whoever is steering: a
    disengaged or overridden frame pairs a prediction about the model's path with a measurement of the
    driver's own steering, so it teaches nothing."""
    path = _path(self.V_EGO, self.KAPPA)
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    inputs = {"v_ego": self.V_EGO, "path": path, "steer_angle_deg": angle, "vp_valid": True, "steer_ratio": 15.0, "stiffness_factor": 1.0}

    learned = _planner()
    for _ in range(100):
      learned.update(_inputs(**inputs))
    self.assertGreater(learned.lat_bias.samples, 0)

    for disengaged_kwargs in ({"lat_active": False}, {"steering_pressed": True}):
      with self.subTest(**disengaged_kwargs):
        planner = _planner()
        for _ in range(100):
          planner.update(_inputs(**inputs, **disengaged_kwargs))
        self.assertEqual(planner.lat_bias.samples, 0)

  def test_the_path_is_referenced_from_the_frame_the_model_saw(self):
    """`modelV2.position.x` is measured from the pose of the frame the model saw — `T_IDXS` is
    published unshifted (`fill_model_msg.py:84`) — while every candidate in the `min` is evaluated at
    model publish + `action_t`, which is where `lead_age` puts the lead pair. So the pre-brake has to
    read the path `v_ego * (logMonoTime - timestampEof)` of the way in, or it brakes as if the car
    were still where that frame was taken. Measured publish lag over seven corpus segments: 29 ms
    median, 34 p95, 38 max — ~0.9 m at 30 m/s.

    The geometry is chosen so the answer is the term's own arithmetic rather than its floor: a 40 m
    ramp to a 250 m radius starting 20 m out, at 30 m/s, where one sample binds at ~66 m and the ask
    is ~-0.25 m/s². A 3 m shift (the bound, 0.1 s) is then `|a| * 3 / 66` ≈ 0.013 deeper.
    """
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def settle(age):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, model_age_s=age))
      return planner

    plain, shifted = settle(0.0), settle(0.1)
    delta = plain.output_a_target - shifted.output_a_target
    self.assertEqual(shifted.source, plain.source)
    self.assertGreater(delta, 0.005, "the shift did not reach the command")
    self.assertLess(delta, 0.05, "the shift moved the command by more than the term's own arithmetic")
    self.assertLess(float(shifted.a_desired_trajectory.min()), float(plain.a_desired_trajectory.min()))

  def test_the_path_age_is_bounded_and_absent_stamps_are_the_old_planner(self):
    """Two model periods, the same bound `moonpilot/curvature.py` puts on the age of this same
    message: past it the shift is dropped rather than extrapolated, and a missing, zero or impossible
    stamp is the zero shift this planner had before it existed. That direction matters — a shift that
    is too large empties the pre-brake's binding set, which stops it asking at all — so the guard is a
    window rather than `max(0.0, ...)`."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(sm):
      planner = _planner()
      for _ in range(40):
        planner.update(sm)
      return planner.output_a_target

    plain = command(_inputs(v_ego=30.0, path=path))
    for age in (MOONPILOT_CURVE_PATH_MAX_AGE + 1e-9, 0.5):
      with self.subTest(model_age_s=age):
        self.assertAlmostEqual(command(_inputs(v_ego=30.0, path=path, model_age_s=age)), plain, delta=1e-9)

    unset = _inputs(v_ego=30.0, path=path, model_age_s=0.1)
    unset['modelV2'].timestampEof = 0  # a message that never carried one
    self.assertAlmostEqual(command(unset), plain, delta=1e-9)

    no_stamp = _inputs(v_ego=30.0, path=path, model_age_s=0.1)
    no_stamp.logMonoTime.pop('modelV2')  # a bare dict, as the maneuver plant hands the planner
    no_stamp.recv_time.pop('modelV2')  # and no tick stamp: the tick anchor is the primary one now
    self.assertAlmostEqual(command(no_stamp), plain, delta=1e-9)

  def test_the_rollout_carries_the_curve(self):
    """The published plan is the policy the command came from, so the rollout has to see the same
    terms — otherwise the plan reads as if the car were still cruising."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertLess(float(planner.a_desired_trajectory.min()), 0.0)
    self.assertTrue(np.all(np.diff(planner.v_desired_trajectory) <= 1e-9))

    without = _planner()
    for _ in range(40):
      without.update(_inputs(v_ego=30.0))
    self.assertTrue(np.all(without.a_desired_trajectory >= 0.0))
    self.assertTrue(np.all(np.diff(without.v_desired_trajectory) >= -1e-9))

  def test_the_learned_scale_persists(self):
    """The value handed to the next drive, on the estimator's own cadence and gate: a trusted
    estimate only, so an unestimated one never becomes the next boot's scale."""
    planner = _planner()
    planner.lat_bias.seed(1.2, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    for _ in range(MOONPILOT_CURVE_BIAS_PERSIST_EVERY):
      planner.update(_inputs())
    self.assertEqual(planner.params.puts.get(MOONPILOT_CURVE_BIAS_KEY), 1.2)

    measuring = _planner()
    for _ in range(MOONPILOT_CURVE_BIAS_PERSIST_EVERY):
      measuring.update(_inputs())
    self.assertNotIn(MOONPILOT_CURVE_BIAS_KEY, measuring.params.puts)

  def test_the_longitudinal_jerk_scale_persists_only_after_estimation(self):
    planner = _planner()
    planner.long_jerk.seed(0.75, MOONPILOT_LONG_JERK_MIN_SAMPLES)
    for _ in range(MOONPILOT_LONG_JERK_PERSIST_EVERY):
      planner.update(_inputs())
    self.assertEqual(planner.params.puts.get(MOONPILOT_LONG_JERK_SCALE_KEY), 0.75)

    measuring = _planner()
    for _ in range(MOONPILOT_LONG_JERK_PERSIST_EVERY):
      measuring.update(_inputs())
    self.assertNotIn(MOONPILOT_LONG_JERK_SCALE_KEY, measuring.params.puts)

  def test_the_feature_rows_match_the_params_defaults(self):
    """These behaviors ship off: each is unvalidated on a car, and each is read with
    `enabled()`, so the row and the param default have to agree."""
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    for key in ("MoonpilotCurveSpeed", "MoonpilotPathPreview", "MoonpilotPathSmooth", "MoonpilotSqueeze", "MoonpilotPathOutside"):
      with self.subTest(key=key):
        feature = next(f for f in FEATURES if f.key == key)
        self.assertTrue(feature.offroad_only)
        self.assertFalse(feature.requires)
        self.assertTrue(f'{{"{key}", {{PERSISTENT, BOOL, "0"}}}}' in text)
    # The learned scale is a value, not a toggle: nothing may gate on it, and its neutral default has
    # to be the neutral ratio rather than the "unset" the lag param uses.
    self.assertTrue('{"MoonpilotCurveLatScale", {PERSISTENT, FLOAT, "1.0"}}' in text)
    self.assertTrue('{"MoonpilotLongJerkScale", {PERSISTENT, FLOAT, "1.0"}}' in text)


class TestSqueeze(unittest.TestCase):
  """`MoonpilotSqueeze`, through the planner's own seam (`test_corridor.py` holds the pure math).

  The inertness statement is the same one curve rests on: no road edges (the default `_inputs`,
  which is also what upstream's maneuver plant fills) is exactly the planner this fork had before
  this existed, feature on or off — and a pinched corridor with the toggle off is too.
  """

  @staticmethod
  def _edges(width, xs=None):
    xs = list(np.asarray(xs if xs is not None else ModelConstants.X_IDXS, dtype=float))
    half = float(width) / 2.0
    return (xs, [-half] * len(xs)), (xs, [half] * len(xs))

  def test_narrow_corridor_adds_bounded_braking(self):
    edges = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH - 1.0)
    on = _planner()
    off = _planner(params_overrides={"MoonpilotSqueeze": False})
    for _ in range(20):
      on.update(_inputs(road_edges=edges))
      off.update(_inputs(road_edges=edges))
    self.assertLess(on.output_a_target, off.output_a_target)
    self.assertGreaterEqual(on.output_a_target, MOONPILOT_SQUEEZE_ACCEL_MIN - 1e-9)
    self.assertEqual(on.source, off.source)  # still the cruise slot

  def test_wide_corridor_is_the_planner_without_the_feature(self):
    edges = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH + 2.0)
    on, off = _planner(), _planner(params_overrides={"MoonpilotSqueeze": False})
    for _ in range(20):
      on.update(_inputs(road_edges=edges))
      off.update(_inputs(road_edges=edges))
    self.assertAlmostEqual(on.output_a_target, off.output_a_target, delta=1e-12)

  def test_no_road_edges_is_inert_even_when_on(self):
    on, off = _planner(), _planner(params_overrides={"MoonpilotSqueeze": False})
    for _ in range(20):
      on.update(_inputs())
      off.update(_inputs())
    self.assertAlmostEqual(on.output_a_target, off.output_a_target, delta=1e-12)

  def test_toggle_off_is_inert_on_a_pinch(self):
    edges = self._edges(MOONPILOT_SQUEEZE_MIN_WIDTH)
    off = _planner(params_overrides={"MoonpilotSqueeze": False})
    baseline = _planner(params_overrides={"MoonpilotSqueeze": False})
    for _ in range(20):
      off.update(_inputs(road_edges=edges))
      baseline.update(_inputs())
    self.assertAlmostEqual(off.output_a_target, baseline.output_a_target, delta=1e-12)

  def test_valid_ego_pose_uses_the_rolling_history(self):
    """With a valid pose the planner pushes into `RollingCorridor`; without one it clears and
    falls back to single-frame. Both must still add the bounded brake on a pinch."""
    edges = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH - 1.0)
    on = _planner()
    # Give the harness's moonpilotState a valid egoPose at the origin.
    pose_msg = custom.MoonpilotState.new_message()
    pose_msg.egoPose.valid = True
    pose_msg.egoPose.x = 0.0
    pose_msg.egoPose.y = 0.0
    pose_msg.egoPose.yaw = 0.0
    for _ in range(20):
      sm = _inputs(road_edges=edges)
      sm["moonpilotState"] = pose_msg
      sm.valid["moonpilotState"] = True
      sm.alive["moonpilotState"] = True
      on.update(sm)
    self.assertLess(on.output_a_target, 0.0)
    self.assertGreaterEqual(on.output_a_target, MOONPILOT_SQUEEZE_ACCEL_MIN - 1e-9)
    self.assertEqual(len(on.corridor._hist), 20)

  def test_invalid_pose_falls_back_to_single_frame(self):
    edges = self._edges(MOONPILOT_SQUEEZE_FULL_WIDTH - 1.0)
    on = _planner()
    # moonpilotState present but egoPose.valid false (the default SubMaster case).
    for _ in range(5):
      on.update(_inputs(road_edges=edges))
    self.assertEqual(len(on.corridor._hist), 0)
    self.assertLess(on.output_a_target, 0.0)


if __name__ == "__main__":
  unittest.main()
