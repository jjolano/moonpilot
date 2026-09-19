import math
import unittest

from openpilot.common.realtime import DT_MDL

from moonpilot.jerk import (
  MOONPILOT_LONG_JERK_MIN_SAMPLES,
  MOONPILOT_LONG_JERK_SCALE_MIN,
  LongitudinalComfortJerkEstimator,
)
from moonpilot.longitudinal import jerk_limit


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


if __name__ == "__main__":
  unittest.main()
