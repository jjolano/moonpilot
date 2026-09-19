"""The rolling-window ego correction: the smoother, its clamps, and the planner that consumes it.

The harness is moonpilot/tests/test_longitudinal.py's — the same duck-typed SubMaster and the same
inputs builder — so this file tests the correction against the planner it is wired into rather than
against a second fake of it.
"""

import math
import time
import unittest
from pathlib import Path
from unittest import mock

import openpilot.selfdrive.test.longitudinal_maneuvers.plant as plant_mod
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers

from moonpilot import procs
from moonpilot.features import FEATURES
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.slam import (
  MOONPILOT_SLAM_CORR_STD_MAX,
  MOONPILOT_SLAM_CORR_STD_MIN,
  MOONPILOT_SLAM_MAX_AGE,
  MOONPILOT_SLAM_MAX_CORR_POS,
  MOONPILOT_SLAM_MAX_CORR_VEL,
  MOONPILOT_SLAM_MAX_CORR_YAW,
  MOONPILOT_SLAM_MIN_NODES,
  MOONPILOT_SLAM_MIN_STD,
  MOONPILOT_SLAM_POSE_DELAY,
  MOONPILOT_SLAM_PRIOR_WINDOW_S,
  MOONPILOT_SLAM_RATE_HZ,
  MOONPILOT_SLAM_ROT_STD_MULT,
  MOONPILOT_SLAM_TRANS_STD_MULT,
  MOONPILOT_SLAM_WINDOW_S,
  Node,
  PriorChannel,
  RotatingPoseWindow,
  ego_speed_correction,
)
from moonpilot.tests.test_longitudinal import _inputs, _lead, _planner

ROOT = Path(__file__).resolve().parents[2]
DT = 1.0 / MOONPILOT_SLAM_RATE_HZ
# An arbitrary monotonic time in the same order of magnitude as the runtime's, so the pose-time
# subtraction and the nanosecond round trip are exercised at the size they happen at.
ODOMETRY_MONO = 5_980_000.0


def _node(t, v_ego, trans_x, trans_y=0.0, yaw_rate=0.0, rot_z=0.0, trans_std=0.02, rot_std=0.02):
  """One window sample, in the car frame the window is written in. The odometry stds are
  posenet-sized: locationd multiplies them by 4 and 10, which the smoother mirrors."""
  return Node(
    mono_time=t,
    v_ego=v_ego,
    yaw_rate=yaw_rate,
    trans_x=trans_x,
    trans_y=trans_y,
    rot_z=rot_z,
    trans_std_x=trans_std,
    rot_std_z=rot_std,
  )


def _window(nodes):
  window = RotatingPoseWindow()
  for node in nodes:
    window.push(node)
  return window


def _straight(t0, v_ego=20.0, trans_x=20.0, n=int(MOONPILOT_SLAM_WINDOW_S * MOONPILOT_SLAM_RATE_HZ)):
  return [_node(t0 + i * DT, v_ego, trans_x) for i in range(n)]


class TestWindow(unittest.TestCase):
  def test_a_slow_wheel_speed_reads_back_as_a_correction_toward_the_odometry(self):
    """The wheel speed reads 0.5 m/s low -- a 2.5 % scale error -- and the odometry reads the truth.

    The convention is *smoothed minus the raw prior*, which is the only one a consumer can use: the
    correction is added to the wheel speed it was computed against, so it has to be positive when
    that speed is the low one. The plan pinned the opposite sign (`dVel ~ -0.5` for a VO reading
    high), which would add the wheel's own scale error back rather than remove it -- truth 20, wheel
    20, VO 20.5 comes out 19.5. The replay check against `gpsLocationExternal` in the plan's
    Verification is the arbiter if these two ever disagree again.

    The odometry's own std is 0.08 m/s here against the prior's 1.0, so the blend is nearly the
    odometry's and the applied speed lands on the truth rather than between the two.
    """
    corr = _window(_straight(time.monotonic(), v_ego=19.5, trans_x=20.0)).update()
    self.assertTrue(corr["valid"])
    self.assertAlmostEqual(corr["dVel"], 0.5, delta=0.05)
    self.assertAlmostEqual(19.5 + corr["dVel"], 20.0, delta=0.05)  # the consumer's arithmetic
    # The wheel speed was low for the whole window, so the position the two chains disagree by is
    # that bias times the window's span.
    self.assertAlmostEqual(corr["dPos"], 0.5 * (MOONPILOT_SLAM_WINDOW_S - DT), delta=0.05)

  def test_the_applied_speed_is_closer_to_truth_than_the_wheel_speed(self):
    """The feature's reason to exist, in the currency the plan's replay check measures: across the
    scale errors a wheel speed is actually wrong by, the corrected speed is nearer the truth than
    the raw one. A negative bias is the same statement with the sign flipped, and zero is the
    no-harm case -- an honest wheel speed costs a few centimeters per second of blend, not a bias.
    """
    for bias in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
      with self.subTest(bias=bias):
        wheel, truth = 20.0, 20.0 + bias
        corr = _window(_straight(time.monotonic(), v_ego=wheel, trans_x=truth)).update()
        self.assertTrue(corr["valid"])
        applied = wheel + corr["dVel"]
        self.assertLessEqual(abs(applied - truth), abs(wheel - truth))  # never worse than the wheel
        if bias == 0.0:
          self.assertLess(abs(applied - truth), 0.05)  # and an honest wheel speed costs nothing

  def test_a_matching_prior_corrects_nothing(self):
    """The ceiling on the correction is what it must be worth when the two sources agree: nothing."""
    corr = _window(_straight(time.monotonic(), v_ego=20.0, trans_x=20.0)).update()
    self.assertTrue(corr["valid"])
    self.assertAlmostEqual(corr["dVel"], 0.0, delta=1e-6)
    self.assertAlmostEqual(corr["dPos"], 0.0, delta=1e-6)
    self.assertAlmostEqual(corr["dYaw"], 0.0, delta=1e-9)

  def test_a_healthy_yaw_rate_is_left_alone(self):
    """The gyro's plausible error is far below the odometry's own rot std, so the weights hand this
    channel to the prior: 0.02 rad of drift over the window reaches a consumer as ~1 % of itself.
    That is the channel working -- the fork does not inject posenet's yaw noise into a good sensor."""
    t0 = time.monotonic()
    nodes = [_node(t0 + i * DT, 10.0, 10.0, yaw_rate=0.204, rot_z=0.2) for i in range(100)]
    corr = _window(nodes).update()
    self.assertTrue(corr["valid"])
    self.assertLess(abs(corr["dYaw"]), 0.1 * 0.02)

  def test_a_broken_yaw_rate_is_pulled_toward_the_odometry(self):
    """The other side of the same weights: a yaw rate wrong by far more than the odometry's own
    uncertainty is still the prior's error to give up, and the clamp is what bounds what it gives."""
    t0 = time.monotonic()
    nodes = [_node(t0 + i * DT, 10.0, 10.0, yaw_rate=2.0, rot_z=0.2) for i in range(100)]
    corr = _window(nodes).update()
    self.assertTrue(corr["valid"])
    self.assertLess(corr["dYaw"], 0.0)  # toward the odometry's 0.2 rad/s, not away from it
    self.assertAlmostEqual(corr["dYaw"], -MOONPILOT_SLAM_MAX_CORR_YAW, delta=1e-6)

  def test_a_wide_disagreement_is_clamped_not_trusted(self):
    """A 10 m/s disagreement is a broken sensor, not a scale error: it reaches a consumer as the
    clamp, and the clamps are what bound every term the planner reads."""
    corr = _window(_straight(time.monotonic(), v_ego=20.0, trans_x=30.0)).update()
    self.assertTrue(corr["valid"])
    self.assertAlmostEqual(abs(corr["dPos"]), MOONPILOT_SLAM_MAX_CORR_POS, delta=1e-6)
    self.assertAlmostEqual(abs(corr["dVel"]), MOONPILOT_SLAM_MAX_CORR_VEL, delta=1e-6)

  def test_a_dropped_tick_stream_keeps_the_correction_finite(self):
    """Half the odometry frames lost is a real thing on a device under load, and the honest answers
    are "a correction from the frames that arrived" or "nothing": never a non-finite one. The
    intervals are twice as long but the weights are inverse variances of the same currency, so the
    estimate is the one the full stream gives."""
    t0 = time.monotonic()
    nodes = [_node(t0 + i * 2 * DT, 19.5, 20.0) for i in range(int(MOONPILOT_SLAM_WINDOW_S * MOONPILOT_SLAM_RATE_HZ / 2))]
    corr = _window(nodes).update()
    self.assertTrue(corr["valid"])
    self.assertAlmostEqual(corr["dVel"], 0.5, delta=0.05)
    self.assertTrue(all(math.isfinite(v) for v in (corr["dPos"], corr["dVel"], corr["dYaw"], corr["corrStd"])))

  def test_a_non_finite_sample_is_dropped_and_not_carried(self):
    """One bad frame must cost one frame. A NaN that reached the least squares would poison every
    correction the window produces until the sample fell off the end of it."""
    t0 = time.monotonic()
    nodes = _straight(t0)
    nodes[40] = nodes[40]._replace(trans_x=float("nan"))
    corr = _window(nodes).update()
    self.assertTrue(corr["valid"])
    self.assertTrue(all(math.isfinite(v) for v in (corr["dPos"], corr["dVel"], corr["dYaw"], corr["corrStd"])))

  def test_a_window_that_cannot_speak_says_nothing(self):
    """Fewer than MIN_NODES samples, a repeated timestamp, or a time running backwards: all three
    are one answer -- invalid and zero -- so a consumer needs no case for a partial one."""
    t0 = time.monotonic()
    for nodes in (
      [],
      _straight(t0, n=MOONPILOT_SLAM_MIN_NODES - 1),
      [_node(t0 + i * DT, 20.0, 20.0) for i in range(MOONPILOT_SLAM_MIN_NODES - 1)] + [_node(t0, 20.0, 20.0)],
      [_node(t0 - i * DT, 20.0, 20.0) for i in range(MOONPILOT_SLAM_MIN_NODES + 1)],
    ):
      with self.subTest(nodes=len(nodes)):
        corr = _window(nodes).update()
        self.assertFalse(corr["valid"])
        self.assertEqual((corr["dPos"], corr["dVel"], corr["dYaw"]), (0.0, 0.0, 0.0))

  def test_the_window_forgets_what_left_it(self):
    """The rotating part: a sample older than the window is gone, so the estimate tracks the car
    rather than averaging the whole drive."""
    window = _window(_straight(0.0))
    self.assertEqual(len(window.window), int(MOONPILOT_SLAM_WINDOW_S * MOONPILOT_SLAM_RATE_HZ))
    for i in range(100):
      window.push(_node(100.0 + i * DT, 20.0, 20.0))
    self.assertGreaterEqual(window.window[0].mono_time, 100.0 - MOONPILOT_SLAM_WINDOW_S)

  def test_the_age_is_the_window_end_to_now(self):
    """Honest, and it includes the pose delay: the window end *is* the pose time, MOONPILOT_SLAM_POSE_DELAY
    in the past, so a correction is never fresher than that. The consumer's budget accounts for it
    rather than the publisher papering over it."""
    corr = _window(_straight(time.monotonic())).update()
    self.assertLess(corr["age"], MOONPILOT_SLAM_MAX_AGE)


def _with_correction(sm, d_vel=0.0, d_pos=0.0, d_yaw=0.0, age=0.0, corr_std=MOONPILOT_SLAM_CORR_STD_MIN, valid=True, alive=True):
  """The correction a publisher would have left on `sm`, and its validity flags."""
  corr = sm["moonpilotState"].egoCorrection
  corr.valid = valid
  corr.monoTime = int(time.monotonic() * 1e9)
  corr.age = age
  corr.dPos = d_pos
  corr.dVel = d_vel
  corr.dYaw = d_yaw
  corr.corrStd = corr_std
  sm.valid["moonpilotState"] = True
  sm.alive["moonpilotState"] = alive
  return sm


class TestConsumer(unittest.TestCase):
  def test_nothing_published_is_no_correction(self):
    """The baseline: no daemon, no correction, and the planner's inputs are the raw ones."""
    self.assertEqual(ego_speed_correction(_inputs(v_ego=20.0)), 0.0)
    # The maneuver harness hands the planner a plain dict with no moonpilotState at all
    self.assertEqual(ego_speed_correction({}), 0.0)

  def test_a_stale_or_invalid_correction_is_no_correction(self):
    for kwargs in (
      {"age": MOONPILOT_SLAM_MAX_AGE + MOONPILOT_SLAM_POSE_DELAY + 0.01},
      {"valid": False},
      {"alive": False},
      {"d_vel": float("nan")},
      {"corr_std": float("inf")},
    ):
      with self.subTest(**kwargs):
        sm = _with_correction(_inputs(v_ego=20.0), **{"d_vel": -0.4, **kwargs})
        self.assertEqual(ego_speed_correction(sm), 0.0)

  def test_a_fresh_correction_is_scaled_by_its_own_confidence(self):
    # float32 on the wire, hence the loose delta
    sm = _with_correction(_inputs(v_ego=20.0), d_vel=-0.4, corr_std=MOONPILOT_SLAM_CORR_STD_MIN)
    self.assertAlmostEqual(ego_speed_correction(sm), -0.4 / (1.0 + MOONPILOT_SLAM_CORR_STD_MIN), delta=1e-6)

    sm = _with_correction(_inputs(v_ego=20.0), d_vel=-0.4, corr_std=MOONPILOT_SLAM_CORR_STD_MAX)
    self.assertAlmostEqual(ego_speed_correction(sm), -0.4 / (1.0 + MOONPILOT_SLAM_CORR_STD_MAX), delta=1e-6)

    # Never amplified: the largest correction any published std can produce is the clamp itself
    sm = _with_correction(_inputs(v_ego=20.0), d_vel=-MOONPILOT_SLAM_MAX_CORR_VEL)
    self.assertGreaterEqual(ego_speed_correction(sm), -MOONPILOT_SLAM_MAX_CORR_VEL)
    self.assertLess(ego_speed_correction(sm), 0.0)


def _stop_and_go(planner, correction=None, frames=420):
  """20 m/s behind a lead 40 m ahead that brakes to a stop at 2 m/s^2 and stays there.

  The correction moves the speed the planner plans from and nothing else, so the two runs are the
  same drive seen through two speed estimates: what is asserted is that neither the plan's shape nor
  the following distance runs away, and that both come to rest behind the lead.
  """
  v_ego, v_lead, gap = 20.0, 20.0, 40.0
  peak_decel, commands = 0.0, []
  for frame in range(frames):
    if gap < 2.0:
      raise AssertionError(f"contact at frame {frame}: the plan stopped following")
    sm = _inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead, model_prob=0.95), standstill=v_ego < 0.1)
    if correction is not None:
      _with_correction(sm, **correction)
    planner.update(sm)
    assert math.isfinite(planner.output_a_target), f"non-finite command at frame {frame}"
    assert all(math.isfinite(float(v)) for v in planner.v_desired_trajectory), f"non-finite plan at frame {frame}"
    commands.append(planner.output_a_target)
    peak_decel = min(peak_decel, planner.output_a_target)
    v_ego = max(0.0, v_ego + planner.output_a_target * DT)
    v_lead = max(0.0, v_lead - 2.0 * DT) if frame < 480 else v_lead
    gap = max(0.0, gap - (v_ego - v_lead) * DT)
  return commands, peak_decel, v_ego


class TestPlannerCorrection(unittest.TestCase):
  """The planner is the one consumer: the correction moves the speed it plans from, and nothing
  else about the plan. A lead braking to a stop is where that either helps or shows up as a
  different plan, which is the difference this class bounds."""

  def test_a_correction_refines_the_plan_rather_than_reshaping_it(self):
    base_commands, base_peak, base_v = _stop_and_go(_planner())
    corr_commands, corr_peak, corr_v = _stop_and_go(_planner(), correction={"d_vel": 0.5})

    self.assertNotEqual(base_commands, corr_commands)  # the correction really reaches the plan
    self.assertLess(abs(corr_peak - base_peak), 0.5)  # ... by less than the correction's own size
    self.assertAlmostEqual(corr_v, base_v, delta=0.5)  # and it neither stalls nor closes the gap

  def test_an_invalid_correction_leaves_the_plan_identical(self):
    base_commands, _, _ = _stop_and_go(_planner())
    for kwargs in ({"valid": False}, {"age": MOONPILOT_SLAM_MAX_AGE + MOONPILOT_SLAM_POSE_DELAY + 0.5}, {"alive": False}, {"corr_std": float("nan")}):
      with self.subTest(**kwargs):
        sm_commands, _, _ = _stop_and_go(_planner(), correction={"d_vel": 0.5, **kwargs})
        self.assertEqual(base_commands, sm_commands)


class TestUpstreamManeuverWithCorrection(unittest.TestCase):
  """Upstream's own plant, driven with a correction and without one, on the maneuver where the
  correction has something to say: following at 20 m/s while the lead brakes to a stop.

  The plant hands its planner a plain dict and nothing publishes a correction into it, so the
  correction is injected at the seam the plant's own patched class name gives -- the same trick
  test_longitudinal.py uses to swap the planner in, one layer down.
  """

  TITLE = "steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2"

  class _Sm(dict):
    """The plant's dict, with the two duck-typed attributes the correction reader probes."""

    def __init__(self, data, correction):
      super().__init__(data)
      msg = messaging.new_message("moonpilotState")
      msg.valid = True
      if correction is not None:
        corr = msg.moonpilotState.egoCorrection
        corr.valid = True
        corr.monoTime = int(time.monotonic() * 1e9)
        corr.age = 0.0
        corr.corrStd = MOONPILOT_SLAM_CORR_STD_MIN
        corr.dVel = correction
      self["moonpilotState"] = msg.moonpilotState
      self.valid = {"moonpilotState": True}
      self.alive = {"moonpilotState": True}

  @classmethod
  def _planner(cls, correction):
    class CorrectedPlanner(MoonpilotLongitudinalPlanner):
      def update(self, sm):
        super().update(cls._Sm(sm, correction))

    def factory(CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
      return CorrectedPlanner(CP, init_v=init_v, init_a=init_a, dt=dt)

    return factory, CorrectedPlanner

  def _run(self, correction):
    factory, _ = self._planner(correction)
    maneuver = next(m for m in create_maneuvers({"e2e": False, "force_decel": False}) if m.title == self.TITLE)
    with mock.patch.object(plant_mod, "LongitudinalPlanner", factory):
      valid, logs = maneuver.evaluate()
    self.assertTrue(valid)  # no contact, no stall, and still decelerating to the stop
    return logs

  def test_the_correction_moves_the_maneuver_without_reshaping_it(self):
    base = self._run(None)
    corrected = self._run(0.5)  # the wheel speed reads 0.5 m/s low: where a closing lead's money is

    self.assertNotEqual(list(base[:, 5]), list(corrected[:, 5]))
    # Refine, not reshape: the correction cannot reorder the candidates the policy arbitrates
    # between, and the depth of the maneuver is essentially the same one (0.18 m/s^2 between the two
    # minima). What the 1.5 m/s^2 bound on the instantaneous difference allows is the *onset* moving,
    # which is the whole point of the correction: the two runs are 0.25 s apart on the jerk-limited
    # ramp into the lead's brake, and the peak difference (measured 1.00 m/s^2) lands on that ramp's
    # steep part rather than on a different ramp. It was 0.43 m/s^2 before the headway floor moved the
    # follow 6 m closer, where the same speed error acts through a smaller slack.
    self.assertLess(abs(corrected[:, 5].min() - base[:, 5].min()), 0.5)
    self.assertLess(max(abs(a - b) for a, b in zip(base[:, 5], corrected[:, 5], strict=True)), 1.5)
    for logs in (base, corrected):
      self.assertAlmostEqual(logs[-1, 3], 0.0, delta=0.05)  # it still comes to rest
      self.assertGreater(logs[:, 6].min(), 2.0)  # and behind the lead, never through it
    # The one visible difference, and the direction the correction's own sign implies: the same gap
    # believed to be closing 0.5 m/s faster is planned for earlier, so the corrected run settles
    # further back. That is the whole trade this feature makes, visible in one maneuver.
    self.assertGreater(corrected[-1, 6], base[-1, 6], "the corrected run must settle further back: a correction the plan cannot act on is not reaching it")


class TestMirroredConstants(unittest.TestCase):
  """The smoother trusts the two sensors on locationd's terms, not on its own: the stds it refuses
  below and the multipliers it discounts temporally correlated odometry noise by. Mirrored rather
  than imported for the init-path reason -- locationd is a daemon, and the planner's seam path
  imports this module -- so a change there has to fail here instead of silently retuning the fork."""

  def test_the_stds_and_multipliers_are_locationds(self):
    from openpilot.selfdrive.locationd import locationd

    self.assertEqual(MOONPILOT_SLAM_MIN_STD, locationd.MIN_STD_SANITY_CHECK)
    self.assertEqual(MOONPILOT_SLAM_TRANS_STD_MULT, locationd.CAM_ODO_TRANS_STD_MULT)
    self.assertEqual(MOONPILOT_SLAM_ROT_STD_MULT, locationd.CAM_ODO_ROT_STD_MULT)


class TestPriorChannel(unittest.TestCase):
  def test_reads_the_pose_time_not_the_latest_sample(self):
    """The whole point of the channel: `at` interpolates to a time in the past, so the prior covers
    the same span the odometry's pose does rather than the span since the last frame."""
    channel = PriorChannel()
    for i in range(20):  # 0.2 s at 100 Hz
      channel.push(i * 0.01, 20.0 + i)
    at_pose_time = channel.at(0.15)
    assert at_pose_time is not None  # the samples span it, which is the claim under test
    self.assertAlmostEqual(at_pose_time, 35.0, delta=1e-9)
    self.assertTrue(channel.at(0.0) is not None)

  def test_a_time_it_cannot_span_is_not_extrapolated(self):
    channel = PriorChannel()
    for i in range(10):
      channel.push(i * 0.01, 20.0)
    self.assertIsNone(channel.at(5.0))  # after the samples
    self.assertIsNone(channel.at(-1.0))  # before them
    self.assertIsNone(channel.at(float("nan")))
    self.assertIsNone(PriorChannel().at(0.0))  # nothing pushed at all

  def test_a_non_finite_sample_never_enters_the_channel(self):
    channel = PriorChannel()
    for i in range(10):
      channel.push(i * 0.01, float("nan") if i == 5 else 20.0)
    self.assertNotIn(float("nan"), [value for _, value in channel.window])
    at_pose_time = channel.at(0.05)
    assert at_pose_time is not None
    self.assertAlmostEqual(at_pose_time, 20.0, delta=1e-9)

  def test_the_channel_forgets_what_left_it(self):
    channel = PriorChannel()
    for i in range(300):
      channel.push(i * 0.01, 20.0)
    self.assertGreaterEqual(channel.window[0][0], 2.99 - MOONPILOT_SLAM_PRIOR_WINDOW_S)


class TestIngest(unittest.TestCase):
  """Calibration-frame odometry is rotated into the device frame before the window's car-frame
  conversion.

  The device frame is x forward, y right, z down (`openpilot/common/transformations/camera.py`).
  The window is written in the car's frame, x forward, y left and yaw left-positive. A dropped
  negation here is invisible to every other test in this file -- they build `Node`s directly,
  already in car-frame units -- and on the road it makes a left turn's gyro and odometry cancel
  each other instead of agreeing.
  """

  class _Sm(dict):
    def __init__(self, odometry, device_motion, car_state, extrinsics_calibration=None, calibration_valid=True):
      super().__init__(cameraOdometry=odometry, deviceMotion=device_motion, carState=car_state)
      if extrinsics_calibration is not None:
        self["extrinsicsCalibration"] = extrinsics_calibration
      self.updated = {"cameraOdometry": True, "deviceMotion": True, "carState": True}
      self.valid = {"cameraOdometry": True, "deviceMotion": True, "carState": True}
      if extrinsics_calibration is not None:
        self.valid["extrinsicsCalibration"] = calibration_valid
      self.logMonoTime = {"cameraOdometry": int(ODOMETRY_MONO * 1e9), "deviceMotion": int(ODOMETRY_MONO * 1e9), "carState": int(ODOMETRY_MONO * 1e9)}

  @staticmethod
  def _calibration(rpy_calib):
    calibration = messaging.new_message("extrinsicsCalibration")
    calibration.extrinsicsCalibration.rpyCalib = rpy_calib
    return calibration.extrinsicsCalibration

  def test_calibration_rotates_vectors_and_stds_before_sign_conversion(self):
    from moonpilot.leadd import _slam_node

    odometry = messaging.new_message("cameraOdometry")
    odometry.cameraOdometry.trans = [20.0, 1.0, 2.0]
    odometry.cameraOdometry.rot = [0.1, 0.3, -0.2]
    odometry.cameraOdometry.transStd = [0.02, 0.04, 0.06]
    odometry.cameraOdometry.rotStd = [0.03, 0.04, 0.05]
    device_motion = messaging.new_message("deviceMotion")
    device_motion.deviceMotion.angularVelocityDevice.z = -0.2
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = 19.5
    sm = self._Sm(
      odometry.cameraOdometry,
      device_motion.deviceMotion,
      car_state.carState,
      self._calibration([0.0, 0.1, 0.0]),
    )

    node = _slam_node(sm, self._priors(v_ego=19.5))
    assert node is not None
    c, s = math.cos(0.1), math.sin(0.1)
    self.assertAlmostEqual(node.trans_x, c * 20.0 + s * 2.0)
    self.assertAlmostEqual(node.trans_y, -1.0)
    self.assertAlmostEqual(node.rot_z, s * 0.1 + c * 0.2)
    self.assertAlmostEqual(node.trans_std_x, math.sqrt((c * 0.02) ** 2 + (s * 0.06) ** 2))
    self.assertAlmostEqual(node.rot_std_z, math.sqrt((s * 0.03) ** 2 + (c * 0.05) ** 2))
    self.assertAlmostEqual(node.yaw_rate, 0.2)  # gyro path remains device-frame and unchanged

  def test_identity_and_untrusted_calibration_fall_back_to_identity(self):
    from moonpilot.leadd import _slam_node

    odometry = messaging.new_message("cameraOdometry")
    odometry.cameraOdometry.trans = [20.0, 1.0, 2.0]
    odometry.cameraOdometry.rot = [0.1, 0.3, -0.2]
    odometry.cameraOdometry.transStd = [0.02, 0.04, 0.06]
    odometry.cameraOdometry.rotStd = [0.03, 0.04, 0.05]
    device_motion = messaging.new_message("deviceMotion")
    device_motion.deviceMotion.angularVelocityDevice.z = -0.2
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = 19.5
    priors = self._priors(v_ego=19.5)
    common = (odometry.cameraOdometry, device_motion.deviceMotion, car_state.carState)
    expected = _slam_node(self._Sm(*common), priors)
    identity = _slam_node(self._Sm(*common, self._calibration([0.0, 0.0, 0.0])), self._priors(v_ego=19.5))
    assert expected is not None and identity is not None
    self.assertEqual(identity, expected)

    for rpy_calib, calibration_valid in (
      ([0.0, 0.1, 0.0], False),
      ([0.0, 0.1], True),
      ([0.0, 0.6, 0.0], True),
    ):
      actual = _slam_node(
        self._Sm(*common, self._calibration(rpy_calib), calibration_valid),
        self._priors(v_ego=19.5),
      )
      self.assertEqual(actual, expected)



  @staticmethod
  def _priors(v_ego=20.0, gyro_z=-0.2):
    priors = {"carState": PriorChannel(), "deviceMotion": PriorChannel()}
    for i in range(20):
      t = ODOMETRY_MONO - 0.2 + i * 0.01
      priors["carState"].push(t, v_ego)
      priors["deviceMotion"].push(t, gyro_z)
    return priors

  def test_a_left_turn_arrives_left_positive(self):
    from moonpilot.leadd import _slam_node

    odometry = messaging.new_message("cameraOdometry")
    odometry.cameraOdometry.trans = [20.0, 1.0, 0.0]  # y rightward in the device frame
    odometry.cameraOdometry.rot = [0.0, 0.0, -0.2]  # z down: a left turn is negative
    odometry.cameraOdometry.transStd = [0.02, 0.02, 0.02]
    odometry.cameraOdometry.rotStd = [0.02, 0.02, 0.02]

    device_motion = messaging.new_message("deviceMotion")
    device_motion.deviceMotion.angularVelocityDevice.z = -0.2
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = 19.5

    node = _slam_node(self._Sm(odometry.cameraOdometry, device_motion.deviceMotion, car_state.carState), self._priors(v_ego=19.5))
    assert node is not None
    self.assertAlmostEqual(node.mono_time, ODOMETRY_MONO - MOONPILOT_SLAM_POSE_DELAY, delta=1e-9)
    self.assertGreater(node.yaw_rate, 0.0)  # the gyro, left-positive
    self.assertGreater(node.rot_z, 0.0)  # the odometry agrees with it instead of canceling
    self.assertLess(node.trans_y, 0.0)  # leftward
    self.assertAlmostEqual(node.trans_x, 20.0, delta=1e-9)
    self.assertAlmostEqual(node.v_ego, 19.5, delta=1e-9)

  def test_a_frame_the_prior_cannot_cover_is_skipped(self):
    from moonpilot.leadd import _slam_node

    def node_for(priors, odometry_valid=True):
      odometry = messaging.new_message("cameraOdometry")
      odometry.cameraOdometry.trans = [20.0, 0.0, 0.0]
      odometry.cameraOdometry.rot = [0.0, 0.0, 0.0]
      odometry.cameraOdometry.transStd = [0.02, 0.02, 0.02]
      odometry.cameraOdometry.rotStd = [0.02, 0.02, 0.02]
      device_motion = messaging.new_message("deviceMotion")
      car_state = messaging.new_message("carState")
      car_state.carState.vEgo = 19.5
      sm = self._Sm(odometry.cameraOdometry, device_motion.deviceMotion, car_state.carState)
      sm.valid["cameraOdometry"] = odometry_valid
      return _slam_node(sm, priors)

    self.assertIsNotNone(node_for(self._priors()))  # the control: this frame does produce a node
    # Nothing pushed at all: there is no prior to read at the pose time
    self.assertIsNone(node_for({"carState": PriorChannel(), "deviceMotion": PriorChannel()}))
    # A prior that starts at the pose time and never before it: read, never extrapolated backwards
    late = {"carState": PriorChannel(), "deviceMotion": PriorChannel()}
    for channel in late.values():
      channel.push(ODOMETRY_MONO, 20.0)
    self.assertIsNone(node_for(late))
    # An odometry the model itself does not trust is not an observation
    self.assertIsNone(node_for(self._priors(), odometry_valid=False))
    # A gyro that never produced a usable sample leaves the second channel with nothing to offer
    nan_gyro = self._priors(gyro_z=float("nan"))
    self.assertEqual(len(nan_gyro["deviceMotion"].window), 0)
    self.assertIsNone(node_for(nan_gyro))


class TestWiring(unittest.TestCase):
  def test_the_feature_its_param_row_and_its_reader_agree(self):
    feature = next(f for f in FEATURES if f.key == "MoonpilotSlam")
    # It changes a speed the planner plans from, so the row is offroad-only, and the declared
    # default is the off state: a missing row is a device-only UnknownKeyName.
    self.assertTrue(feature.offroad_only)
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('{"MoonpilotSlam", {PERSISTENT, BOOL, "0"}}' in text)

  def test_one_process_writes_the_service(self):
    """msgq allows exactly one publisher per service, so the correction and the leads have to share
    a process: a second publisher kills the first with EADDRINUSE on its next send, and the manager
    never restarts a process that exited."""
    writers = [p for p in procs.MOONPILOT_PROCS if p.name == "leadd"]
    self.assertEqual(len(writers), 1)
    self.assertEqual(writers[0].module, "moonpilot.leadd")
    self.assertNotIn("slamd", [p.name for p in procs.MOONPILOT_PROCS])


if __name__ == "__main__":
  unittest.main()
