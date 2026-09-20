"""The learned pitch offset: what it corrects, what it refuses to learn, and its bound.

Every number quoted here comes from the latest 23-segment route (`000003bc--f85dbc2922--*`), where
the reported pitch is negative on 0.23 % of 136,440 `carControl` frames with a +2.27 deg median,
the drive ends 71 m from where it started, and integrating `sin(pitch) * ds` over 7.93 km claims
+298 m of climb — 2.16 deg of bias, -0.21 m/s^2 of phantom uphill in `coast_accel`'s road term.
Replayed through this estimator at plannerd's cadence, that route's 555 s of driving above
`MOONPILOT_PITCH_MIN_SPEED` reaches `estimated` at 300 s and 1.30 deg by the end, which is
`1 - exp(-555/600)` of the truth: under-converged, and therefore under-corrected.
"""

import math
import unittest

from openpilot.common.realtime import DT_MDL

from moonpilot.longitudinal import MOONPILOT_COAST_FLAT_ACCEL, coast_accel
from moonpilot.pitch import (
  MOONPILOT_PITCH_MIN_SAMPLES,
  MOONPILOT_PITCH_OFFSET_MAX,
  MOONPILOT_PITCH_RC,
  MOONPILOT_PITCH_SANITY,
  PitchOffsetEstimator,
)

ROUTE_BIAS = math.radians(2.16)  # the measured offset on the route this feature was written for


def _drive(estimator, pitch, seconds, valid=True):
  for _ in range(int(round(seconds / DT_MDL))):
    estimator.update(pitch, valid)
  return estimator


class TestPitchOffsetEstimator(unittest.TestCase):
  def test_a_standing_bias_is_learned_and_removed_from_the_grade_term(self):
    """The feature's reason to exist: level road reported as a climb stops being a climb.

    Driven for 5 * RC so the filter is converged rather than merely trusted — the partial case is
    the next test.
    """
    estimator = _drive(PitchOffsetEstimator(), ROUTE_BIAS, 5 * MOONPILOT_PITCH_RC)
    self.assertEqual(estimator.status, "estimated")
    self.assertAlmostEqual(estimator.applied(), ROUTE_BIAS, delta=math.radians(0.05))

    corrected = coast_accel(ROUTE_BIAS - estimator.applied())
    self.assertAlmostEqual(corrected, MOONPILOT_COAST_FLAT_ACCEL, delta=0.01)
    # and what it was worth: the uncorrected road term this route reported on level ground
    self.assertAlmostEqual(coast_accel(ROUTE_BIAS) - MOONPILOT_COAST_FLAT_ACCEL, -0.213, delta=0.005)

  def test_nothing_is_applied_before_it_is_trusted_and_a_partial_estimate_under_corrects(self):
    """Zero until `MOONPILOT_PITCH_MIN_SAMPLES`, which is exactly the planner this fork had before,
    and the first trusted value is a fraction of the truth rather than an overshoot of it."""
    estimator = PitchOffsetEstimator()
    _drive(estimator, ROUTE_BIAS, (MOONPILOT_PITCH_MIN_SAMPLES - 1) * DT_MDL)
    self.assertEqual(estimator.status, "measuring")
    self.assertEqual(estimator.applied(), 0.0)

    estimator.update(ROUTE_BIAS, True)
    self.assertEqual(estimator.status, "estimated")
    self.assertGreater(estimator.applied(), 0.0)
    self.assertLess(estimator.applied(), ROUTE_BIAS)  # under-corrected, which leaves the band conservative

  def test_the_clamp_is_above_the_bias_it_has_to_correct(self):
    """A clamp that binds on a real device saturates silently and under-corrects forever, so the
    ordering against the measured bias is the constraint — and tightening it to chase
    `MOONPILOT_COAST_GRADE_MIN` is the wrong fix, which is what this refuses.

    The other side is a named ceiling rather than an impossibility: a wrongly saturated offset is
    worth 0.296 m/s^2 of road term, enough to open the coast band on level road at a gate below
    that, and what bounds the overspeed there is `MOONPILOT_COAST_BAND`.
    """
    self.assertGreater(MOONPILOT_PITCH_OFFSET_MAX, ROUTE_BIAS)
    saturated = abs(coast_accel(-MOONPILOT_PITCH_OFFSET_MAX) - MOONPILOT_COAST_FLAT_ACCEL)
    self.assertLess(saturated, abs(coast_accel(-MOONPILOT_PITCH_SANITY) - MOONPILOT_COAST_FLAT_ACCEL))

  def test_a_large_real_grade_is_clamped_rather_than_believed(self):
    """A genuinely long climb is the window's failure mode, so the clamp is what bounds it."""
    estimator = _drive(PitchOffsetEstimator(), math.radians(10.0), 5 * MOONPILOT_PITCH_RC)
    self.assertAlmostEqual(estimator.applied(), MOONPILOT_PITCH_OFFSET_MAX, delta=1e-9)

  def test_invalid_implausible_and_non_finite_frames_teach_nothing(self):
    """A parked car, a broken pose and a NaN each leave the estimate and the sample count alone."""
    for pitch, valid in ((ROUTE_BIAS, False), (MOONPILOT_PITCH_SANITY * 2, True), (float("nan"), True)):
      with self.subTest(pitch=pitch, valid=valid):
        estimator = _drive(PitchOffsetEstimator(), pitch, 5 * MOONPILOT_PITCH_RC, valid=valid)
        self.assertEqual(estimator.samples, 0)
        self.assertEqual(estimator.estimate, 0.0)
        self.assertEqual(estimator.applied(), 0.0)

  def test_a_seed_is_trusted_immediately_and_clamped(self):
    """The window is longer than most drives, so the next boot has to continue rather than restart."""
    estimator = PitchOffsetEstimator()
    estimator.seed(ROUTE_BIAS, MOONPILOT_PITCH_MIN_SAMPLES)
    self.assertEqual(estimator.status, "estimated")
    self.assertAlmostEqual(estimator.applied(), ROUTE_BIAS, delta=1e-9)

    estimator.seed(math.radians(45.0), MOONPILOT_PITCH_MIN_SAMPLES)
    self.assertAlmostEqual(estimator.applied(), MOONPILOT_PITCH_OFFSET_MAX, delta=1e-9)

  def test_a_route_that_comes_back_down_is_not_an_offset(self):
    """The assumption stated as a test: equal time up and down averages to zero, so real terrain
    does not become a correction."""
    estimator = PitchOffsetEstimator()
    for _ in range(20):
      _drive(estimator, math.radians(4.0), MOONPILOT_PITCH_RC / 4)
      _drive(estimator, math.radians(-4.0), MOONPILOT_PITCH_RC / 4)
    self.assertLess(abs(estimator.applied()), math.radians(0.5))


if __name__ == "__main__":
  unittest.main()
