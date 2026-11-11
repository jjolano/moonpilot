"""The learned radar sensor latency (`moonpilot/radar_latency.py`) and its planner wiring.

The estimator's physics is one ratio on gated frames: radard composes `vLead` with an ego speed
that is `CP.radarDelay` older than "now", so on a car with `radarDelay = 0` a stationary target
reads `vLead ≈ aEgo · τ`. Every number quoted here is either the RAV4 route measurement (0.136 s
over 429 near-rest samples) or a synthetic frame that clears or misses the module's own gates.
"""

import unittest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR

from moonpilot.learned import PERSIST_EVERY
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.radar_latency import (
  MOONPILOT_RADAR_LATENCY_KEY,
  MOONPILOT_RADAR_LATENCY_MAX,
  MOONPILOT_RADAR_LATENCY_MIN_AEGO,
  MOONPILOT_RADAR_LATENCY_MIN_SAMPLES,
  MOONPILOT_RADAR_LATENCY_SAMPLES_KEY,
  RadarLatencyEstimator,
  persisted_seed,
)
from moonpilot.tests.fakes import params

from moonpilot.tests.harness import inputs, lead as make_lead, planner as make_planner

CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)

ROUTE_TAU = 0.136  # s; the RAV4 route's measured ratio


def _frame(v_lead: float, a_ego: float, radar: bool = True, present: bool = True):
  lead = make_lead(30.0, v_lead)
  lead.radar = radar
  lead.present = present
  return lead, a_ego


class TestRadarLatencyEstimator(unittest.TestCase):
  def test_a_stationary_lead_under_excitation_recovers_the_route_tau(self):
    """The feature's reason to exist: alternating brake and accel frames around a near-rest lead
    read back the RAV4's measured 0.136 s, and only once the sample count clears."""
    est = RadarLatencyEstimator()
    for i in range(MOONPILOT_RADAR_LATENCY_MIN_SAMPLES - 1):
      a = -2.0 if i % 2 == 0 else 2.0
      est.update(*_frame(ROUTE_TAU * a, a))
    self.assertEqual(est.status, "measuring")
    self.assertEqual(est.applied(), 0.0)

    est.update(*_frame(ROUTE_TAU * 2.0, 2.0))
    self.assertEqual(est.status, "estimated")
    # capnp stores float32, so the mean recovers ROUTE_TAU only to single precision
    self.assertAlmostEqual(est.applied(), ROUTE_TAU, delta=1e-6)

  def test_vision_absent_and_unexcited_frames_teach_nothing(self):
    for lead, a_ego in (
      _frame(ROUTE_TAU * 2.0, 2.0, radar=False),
      _frame(ROUTE_TAU * 2.0, 2.0, present=False),
      _frame(ROUTE_TAU * 2.0, MOONPILOT_RADAR_LATENCY_MIN_AEGO - 0.01),
      _frame(ROUTE_TAU * 2.0, float("nan")),
      _frame(float("nan"), 2.0),
    ):
      with self.subTest(v_lead=lead.vLead, a_ego=a_ego):
        est = RadarLatencyEstimator()
        for _ in range(100):
          est.update(lead, a_ego)
        self.assertEqual(est.samples, 0)
        self.assertEqual(est.applied(), 0.0)

  def test_a_moving_lead_is_rejected_by_the_tau_window(self):
    """A lead travelling at absolute speed `v` reads `τ_est = v/aEgo + τ`, which leaves the
    one-sided ceiling; during braking the same contamination goes negative and is dropped."""
    est = RadarLatencyEstimator()
    for _ in range(100):
      est.update(*_frame(5.0 + ROUTE_TAU * 2.0, 2.0))  # lead doing 5 m/s while ego accelerates
      est.update(*_frame(5.0 + ROUTE_TAU * -2.0, -2.0))  # lead doing 5 m/s while ego brakes
    self.assertEqual(est.samples, 0)
    self.assertEqual(est.applied(), 0.0)

  def test_the_applied_value_is_one_sided_and_clamped(self):
    est = RadarLatencyEstimator()
    est.seed(5.0, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)
    self.assertAlmostEqual(est.applied(), MOONPILOT_RADAR_LATENCY_MAX, delta=1e-9)
    est.seed(-1.0, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)
    self.assertEqual(est.applied(), 0.0)

  def test_a_seed_is_trusted_immediately(self):
    est = RadarLatencyEstimator()
    est.seed(ROUTE_TAU, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.applied(), ROUTE_TAU, delta=1e-9)

  def test_a_partial_seed_carries_samples_without_trusting(self):
    est = RadarLatencyEstimator()
    est.seed(ROUTE_TAU, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES - 1)
    self.assertEqual(est.status, "measuring")
    self.assertEqual(est.applied(), 0.0)
    est.update(*_frame(ROUTE_TAU * 2.0, 2.0))
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.applied(), ROUTE_TAU, delta=0.01)


  def test_a_float_value_and_count_seed(self):
    store = params(on=False, overrides={
      MOONPILOT_RADAR_LATENCY_KEY: ROUTE_TAU,
      MOONPILOT_RADAR_LATENCY_SAMPLES_KEY: 80,
    })
    self.assertEqual(persisted_seed(store), (ROUTE_TAU, 80))

  def test_a_zero_default_does_not_seed(self):
    self.assertIsNone(persisted_seed(params(on=False)))

  def test_a_missing_count_reads_as_trusted(self):
    """A value written before the count existed was only ever written when trusted."""
    seeded = persisted_seed(params(on=False, overrides={MOONPILOT_RADAR_LATENCY_KEY: ROUTE_TAU}))
    assert seeded is not None
    latency, samples = seeded
    self.assertAlmostEqual(latency, ROUTE_TAU, delta=1e-9)
    self.assertEqual(samples, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)


class TestPlannerWiring(unittest.TestCase):
  def test_an_unseeded_planner_applies_nothing(self):
    planner = make_planner()
    self.assertEqual(planner.radar_latency.applied(), 0.0)
    self.assertEqual(planner.lead_age(make_lead(30.0, 0.0), 0.05), 0.05)

  def test_a_seeded_latency_reaches_only_radar_leads(self):
    planner = make_planner()
    planner.radar_latency.seed(ROUTE_TAU, MOONPILOT_RADAR_LATENCY_MIN_SAMPLES)
    radar = make_lead(30.0, 0.0)
    vision = make_lead(30.0, 0.0)
    vision.radar = False
    self.assertAlmostEqual(planner.lead_age(radar, 0.05), 0.05 + ROUTE_TAU, delta=1e-9)
    self.assertAlmostEqual(planner.lead_age(vision, 0.05), 0.05, delta=1e-9)

  def test_gated_frames_reach_the_estimator_through_update(self):
    planner = make_planner()
    tau = ROUTE_TAU
    for i in range(MOONPILOT_RADAR_LATENCY_MIN_SAMPLES):
      a = -2.0 if i % 2 == 0 else 2.0
      lead = make_lead(30.0, tau * a)
      planner.update(inputs(v_ego=10.0, v_cruise_kph=72.0, a_ego=a, lead=lead))
    self.assertEqual(planner.radar_latency.status, "estimated")
    self.assertAlmostEqual(planner.radar_latency.applied(), tau, delta=0.02)

  def test_a_persisted_value_is_seeded_on_construction(self):
    store = params(on=False, overrides={
      MOONPILOT_RADAR_LATENCY_KEY: ROUTE_TAU,
      MOONPILOT_RADAR_LATENCY_SAMPLES_KEY: MOONPILOT_RADAR_LATENCY_MIN_SAMPLES,
    })
    seeded = MoonpilotLongitudinalPlanner(CP, params=store)
    self.assertEqual(seeded.radar_latency.status, "estimated")
    self.assertAlmostEqual(seeded.radar_latency.applied(), ROUTE_TAU, delta=1e-9)

  def test_the_value_and_its_evidence_persist_together(self):
    planner = make_planner()
    planner.radar_latency.seed(ROUTE_TAU, 10)
    for _ in range(PERSIST_EVERY):
      planner.update(inputs())
    self.assertAlmostEqual(planner.params.puts.get(MOONPILOT_RADAR_LATENCY_KEY), ROUTE_TAU, places=4)
    self.assertEqual(planner.params.puts.get(MOONPILOT_RADAR_LATENCY_SAMPLES_KEY), 10)



if __name__ == "__main__":
  unittest.main()
