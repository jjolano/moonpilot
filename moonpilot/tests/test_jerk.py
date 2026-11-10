import math
import unittest

from openpilot.common.realtime import DT_MDL

from moonpilot.jerk import (
  MOONPILOT_LONG_JERK_MIN_SAMPLES,
  MOONPILOT_LONG_JERK_SCALE_MIN,
  LongitudinalComfortJerkEstimator,
)
from moonpilot.longitudinal import (
  MOONPILOT_JERK_DOWN,
  MOONPILOT_JERK_LAUNCH_SPEED,
  MOONPILOT_JERK_ONSET,
  MOONPILOT_JERK_RAMP_GAP,
  MOONPILOT_JERK_RAMP_HEADWAY,
  MOONPILOT_JERK_LAUNCH_A_BP,
  MOONPILOT_JERK_LAUNCH_V,
  MOONPILOT_JERK_RAMP_V,
  MOONPILOT_JERK_UP,
  positive_ramp,
  jerk_limit,
  ramp_jerk,
)


class TestLongitudinalComfortJerkEstimator(unittest.TestCase):
  def _run(self, gain: float, base: float = 0.75, amplitude: float = 0.75, noise: float = 0.0):
    estimator = LongitudinalComfortJerkEstimator()
    delay_frames = 3
    command = [base + amplitude * math.sin(2.0 * math.pi * i / 40.0) for i in range(600)]
    actual = [gain * command[i - delay_frames] + noise * math.sin(2.0 * math.pi * i / 7.0) if i >= delay_frames else 0.0 for i in range(len(command))]
    for cmd, acc in zip(command, actual, strict=True):
      estimator.update(cmd, acc, delay_frames * DT_MDL, True)
    return estimator

  def test_only_a_high_delivered_comfort_ramp_tightens(self):
    estimator = self._run(2.0)
    self.assertEqual(estimator.status, "estimated")
    self.assertLess(estimator.applied(), 1.0)
    self.assertGreaterEqual(estimator.applied(), MOONPILOT_LONG_JERK_SCALE_MIN)

    gentle = self._run(0.5)
    self.assertEqual(gentle.status, "estimated")
    self.assertAlmostEqual(gentle.applied(), 1.0, places=3)

  def test_unit_gain_with_zero_mean_slope_noise_stays_neutral(self):
    estimator = self._run(1.0, noise=0.005)
    self.assertEqual(estimator.status, "estimated")
    self.assertAlmostEqual(estimator.applied(), 1.0, places=3)

  def test_invalid_stretch_clears_pairing_history(self):
    estimator = LongitudinalComfortJerkEstimator()
    for _ in range(20):
      estimator.update(0.0, 0.0, 0.15, True)
    estimator.update(0.0, 0.0, 0.15, False)
    self.assertEqual(len(estimator.commands), 0)
    self.assertEqual(len(estimator.actuals), 0)
    self.assertEqual(len(estimator.comfort), 0)

  def test_emergency_braking_never_teaches_the_scale(self):
    estimator = LongitudinalComfortJerkEstimator()
    delay_frames = 3
    command = [-4.0 + 1.5 * (i % 40) / 39.0 for i in range(600)]
    actual = [2.0 * command[i - delay_frames] if i >= delay_frames else 0.0 for i in range(len(command))]
    for cmd, acc in zip(command, actual, strict=True):
      estimator.update(cmd, acc, delay_frames * DT_MDL, True)
    self.assertEqual(estimator.status, "measuring")
    self.assertEqual(estimator.applied(), 1.0)

  def test_seed_is_tighten_only(self):
    estimator = LongitudinalComfortJerkEstimator()
    estimator.seed(2.0, MOONPILOT_LONG_JERK_MIN_SAMPLES)
    self.assertEqual(estimator.applied(), 1.0)
    estimator.seed(0.25, MOONPILOT_LONG_JERK_MIN_SAMPLES)
    self.assertEqual(estimator.applied(), MOONPILOT_LONG_JERK_SCALE_MIN)


class TestCalibratedJerkLimit(unittest.TestCase):
  def test_scale_only_changes_positive_comfort_ramp(self):
    baseline = jerk_limit(1.0, 0.0, DT_MDL, 0.0)
    tightened = jerk_limit(1.0, 0.0, DT_MDL, 0.0, comfort_scale=0.5)
    self.assertAlmostEqual(baseline, 1.5 * DT_MDL, places=9)
    self.assertAlmostEqual(tightened, 0.5 * baseline, places=9)

  def test_emergency_braking_ramp_is_unchanged(self):
    baseline = jerk_limit(-3.0, -1.0, DT_MDL, 20.0)
    tightened = jerk_limit(-3.0, -1.0, DT_MDL, 20.0, comfort_scale=0.5)
    self.assertEqual(tightened, baseline)


class TestPositiveRamp(unittest.TestCase):
  """`jerk_limit`'s up side and `ramp_jerk`, which `moonpilot/docs/longitudinal.md` states. The state's first field is the
  source label, which the ramp never reads, so a bare 0 stands in for it."""

  def test_a_positive_command_ramps_at_every_speed_and_a_brake_unwinds_faster(self):
    """The up limit is the positive ramp at every speed, so what the car builds is never a step to the ladder's rung. A
    command at or below zero rises at `MOONPILOT_JERK_UP` instead: letting off a brake is not a build, and one comfort step
    cannot cross `should_stop`'s 0.1, so a parked car does not lose its brake hold to a lead whose reported speed jitters
    (`test_the_stop_gate_follows_the_governing_candidate_not_every_lead` in `test_longitudinal.py` pins that from the other
    end). The down side reads the *command*: a shallow ask onsets at `MOONPILOT_JERK_ONSET` above the creep band, and keeps
    `MOONPILOT_JERK_DOWN` at a crawl, while a lead brakes, and from -2.0 on, where it interpolates into the untouched
    emergency ramp.
    """
    ramp = MOONPILOT_JERK_RAMP_V[-1]
    self.assertLess(ramp, MOONPILOT_JERK_UP)
    for v_ego in (0.0, MOONPILOT_JERK_LAUNCH_SPEED, 25.0):
      self.assertAlmostEqual(jerk_limit(10.0, 0.05, DT_MDL, v_ego), 0.05 + ramp * DT_MDL, places=9)
      self.assertAlmostEqual(jerk_limit(10.0, 0.05, DT_MDL, v_ego, ramp=0.3), 0.05 + 0.3 * DT_MDL, places=9)
      self.assertAlmostEqual(jerk_limit(10.0, 0.0, DT_MDL, v_ego), MOONPILOT_JERK_UP * DT_MDL, places=9)
      self.assertAlmostEqual(jerk_limit(10.0, -0.5, DT_MDL, v_ego), -0.5 + MOONPILOT_JERK_UP * DT_MDL, places=9)
    for v_ego in (0.0, MOONPILOT_JERK_LAUNCH_SPEED):
      self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, v_ego), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, 25.0), -MOONPILOT_JERK_ONSET * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, 25.0, lead_braking=True), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-2.0, 0.0, DT_MDL, 25.0), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)

  def test_the_positive_ramp_follows_the_slowest_close_lead_and_never_passes_the_lead_free_one(self):
    """How hard a lead close ahead accelerates sets the ramp: the floor behind one that is not, faster behind a brisker one,
    the lead-free ramp behind a brisk one and never more. Of two close leads the slower sets it. Close is
    `MOONPILOT_JERK_RAMP_GAP` or `MOONPILOT_JERK_RAMP_HEADWAY` of the ego's speed, whichever is farther, so a lead being
    followed at speed still sets it while one past 3 s of headway is not the lead being followed."""
    def state(gap, a_lead):
      return (0, gap, 0.0, a_lead, 2.0, a_lead)

    floor, lead_free = MOONPILOT_JERK_RAMP_V[0], MOONPILOT_JERK_RAMP_V[-1]
    gentle, brisker = ramp_jerk([state(5.0, 0.5)], 0.0), ramp_jerk([state(5.0, 1.0)], 0.0)
    self.assertLess(floor, gentle)
    self.assertLess(gentle, brisker)
    self.assertLess(brisker, lead_free)
    self.assertEqual(ramp_jerk([state(5.0, 0.0)], 0.0), floor)
    self.assertEqual(ramp_jerk([state(5.0, 3.0)], 0.0), lead_free)
    self.assertEqual(ramp_jerk([], 0.0), lead_free)
    self.assertEqual(ramp_jerk([state(5.0, 3.0), state(15.0, 0.5)], 0.0), gentle)
    self.assertEqual(ramp_jerk([state(MOONPILOT_JERK_RAMP_GAP - 1.0, 0.0)], 0.0), floor)
    self.assertEqual(ramp_jerk([state(MOONPILOT_JERK_RAMP_GAP + 1.0, 0.0)], 0.0), lead_free)
    # 20 m/s: 3 s of headway is 60 m, so a lead at 50 m is being followed and sets the ramp
    self.assertEqual(ramp_jerk([state(50.0, 0.0)], 20.0), floor)
    self.assertEqual(ramp_jerk([state(MOONPILOT_JERK_RAMP_HEADWAY * 20.0 + 1.0, 0.0)], 20.0), lead_free)

  def test_a_standstill_launch_uses_the_launch_band_and_speed_returns_to_the_ordinary_ramp(self):
    """From rest the positive ramp is the driver's own launch build, `MOONPILOT_JERK_LAUNCH_V` on the close lead's launch;
    by twice `MOONPILOT_JERK_LAUNCH_SPEED` it is `ramp_jerk`'s and the launch band has no say. The ordinary ramp alone is
    2.7-4x slower than the driver's build (`python -m moonpilot.lab accel-ramp`), which moved the car within 0.15 s of the
    ask: the ramp, not the car, is what a launch waits on."""
    def state(gap, a_lead):
      return (0, gap, 0.0, a_lead, 2.0, a_lead)

    gentle_lead, brisk_lead = MOONPILOT_JERK_LAUNCH_A_BP[0], MOONPILOT_JERK_LAUNCH_A_BP[-1]
    self.assertAlmostEqual(positive_ramp([], 0.0), MOONPILOT_JERK_LAUNCH_V[-1], places=9)
    self.assertAlmostEqual(positive_ramp([state(5.0, gentle_lead)], 0.0), MOONPILOT_JERK_LAUNCH_V[0], places=9)
    self.assertAlmostEqual(positive_ramp([state(5.0, brisk_lead)], 0.0), MOONPILOT_JERK_LAUNCH_V[-1], places=9)
    self.assertGreater(MOONPILOT_JERK_LAUNCH_V[0], MOONPILOT_JERK_RAMP_V[-1])  # faster than the at-speed ramp, at every lead
    for v_ego in (2 * MOONPILOT_JERK_LAUNCH_SPEED, 25.0):
      self.assertAlmostEqual(positive_ramp([state(5.0, 0.0)], v_ego), ramp_jerk([state(5.0, 0.0)], v_ego), places=9)
      self.assertAlmostEqual(positive_ramp([], v_ego), ramp_jerk([], v_ego), places=9)
    # monotone through the blend, and the launch band owns the whole standstill
    asks = [positive_ramp([state(5.0, 0.5)], v) for v in (0.0, 1.0, 2.5, 4.0, 5.0, 10.0)]
    self.assertGreater(asks[0], asks[-1])
    self.assertTrue(all(later <= earlier + 1e-12 for earlier, later in zip(asks, asks[1:], strict=False)))


if __name__ == "__main__":
  unittest.main()
