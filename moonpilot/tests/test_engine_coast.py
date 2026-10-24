"""The engine-braking coast (`MoonpilotEngineCoast`): the car side's PERMIT_BRAKING hold, the card hook
that turns it on, and the planner coasting to it — with the default untouched when it is off."""

import unittest
from unittest import mock

import numpy as np
from opendbc.car.moonpilot.engine_coast import ENGINE_COAST_BP, ENGINE_COAST_EXIT, ENGINE_COAST_V, EngineCoast, engine_coast_floor
from opendbc.car.toyota.interface import CarInterface
from opendbc.car.toyota.values import CAR, ToyotaFlags

import moonpilot.longitudinal as longitudinal
from moonpilot.brake_band import MOONPILOT_BRAKE_BAND_FLOOR
from moonpilot.engage import car_unavailable_reason, moonpilot_engage_safety_param
from moonpilot.engine_coast import MOONPILOT_ENGINE_COAST_LEAD_T, MOONPILOT_ENGINE_COAST_MARGIN, coast_budget, coast_margin
from moonpilot.features import ENGINE_COAST
from moonpilot.longitudinal import MOONPILOT_ARRIVAL_HOLD_FULL, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V, MoonpilotLongitudinalPlanner, lead_accel
from moonpilot.tests.fakes import _params
from moonpilot.tests.test_engage import _cp
from moonpilot.tests.test_longitudinal import DT_MDL, _inputs, _lead

FLAG = int(ToyotaFlags.MOONPILOT_ENGINE_COAST)


def _planner(flag: bool, overrides: dict | None = None) -> MoonpilotLongitudinalPlanner:
  CP = CarInterface.get_non_essential_params(CAR.TOYOTA_RAV4_TSS2)
  CP.openpilotLongitudinalControl = True
  if flag:
    CP.flags |= FLAG
  fake = _params(on=True, overrides=overrides)
  with mock.patch("moonpilot.longitudinal.Params", lambda: fake):
    planner = MoonpilotLongitudinalPlanner(CP)
  planner.params = fake
  return planner


def _follow(flag: bool, v_lead: float, gap: float, brake: float = 0.0) -> list[float]:
  """3 s behind a lead at 20 m/s ego, the lead braking at `brake` from the second second."""
  planner, v_ego, out = _planner(flag), 20.0, []
  for i in range(60):
    v_lead = max(v_lead + (brake * DT_MDL if i >= 20 else 0.0), 0.0)
    planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead)))
    out.append(planner.output_a_target)
    gap -= (v_ego - v_lead) * DT_MDL
  return out


class TestEngineCoast(unittest.TestCase):
  def test_the_car_side_holds_the_brakes_off_only_at_the_engine_coast(self):
    coast = EngineCoast()
    floor = engine_coast_floor(20.0)
    self.assertFalse(coast.permit(True, floor, 20.0, True))  # at the coast: engine only
    self.assertFalse(coast.permit(True, floor - ENGINE_COAST_EXIT + 0.01, 20.0, True))  # inside the hysteresis: held
    self.assertTrue(coast.permit(True, floor - ENGINE_COAST_EXIT - 0.01, 20.0, True))  # past it: brakes back at once
    self.assertTrue(coast.permit(True, floor - ENGINE_COAST_EXIT + 0.01, 20.0, True))  # and stays back until the coast
    self.assertFalse(coast.permit(True, 0.25, 20.0, True))  # a positive ask needs no brakes either
    # upstream's own decision stands below the speed floor, while stopping, and without longitudinal control
    self.assertTrue(coast.permit(True, -0.05, 4.0, True))
    self.assertTrue(coast.permit(True, -0.05, 20.0, False))

  def test_both_sides_read_the_drivers_measured_coast(self):
    np.testing.assert_allclose(ENGINE_COAST_V, np.interp(ENGINE_COAST_BP, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V))

  def test_the_headway_check_holds_at_every_moment_of_the_coast(self):
    """The margin is the minimum over the whole coast, not its end points: at 1 m/s of closing and
    20.65 m the speed-match point clears the 1.0 s floor by 0.13 m while the gap dips 0.03 m under it
    at 2.03 s. Checked against a dense simulation over random follows."""
    self.assertAlmostEqual(coast_margin(20.0, 20.65, 19.0, 0.0, -0.33, 1.0, 6.0), -0.0302, delta=1e-3)
    rng = np.random.default_rng(0)
    dt = 1e-3
    for _ in range(300):
      v_ego, v_lead = rng.uniform(5, 30), rng.uniform(0, 30)
      gap, a_lead, coast = rng.uniform(3, 60), rng.uniform(-3, 0.5), rng.uniform(-0.4, -0.1)
      ve, vl, g, t, worst = v_ego, v_lead, gap, 0.0, np.inf
      while True:
        worst = min(worst, g - max(ve, 6.0))
        if t >= MOONPILOT_ENGINE_COAST_LEAD_T and ve <= vl:
          break
        vl_next = max(vl + (min(a_lead, 0.0) if t < MOONPILOT_ENGINE_COAST_LEAD_T else 0.0) * dt, 0.0)
        ve_next = ve + coast * dt
        g += 0.5 * (vl + vl_next - ve - ve_next) * dt
        ve, vl, t = ve_next, vl_next, t + dt
      with self.subTest(state=(v_ego, gap, v_lead, a_lead, coast)):
        self.assertAlmostEqual(coast_margin(v_ego, gap, v_lead, a_lead, coast, 1.0, 6.0), worst, delta=0.02)

  def test_card_flags_only_a_toyota_on_openpilot_longitudinal_with_the_toggle(self):
    for brand, op_long, toggle, flagged in (
      ("toyota", True, True, True),
      ("toyota", True, False, False),
      ("toyota", False, True, False),
      ("honda", True, True, False),
    ):
      with self.subTest(brand=brand, op_long=op_long, toggle=toggle):
        cp = _cp(brand=brand, openpilot_longitudinal=op_long)
        moonpilot_engage_safety_param(cp, _params(on=False, overrides={ENGINE_COAST.key: toggle}))
        self.assertEqual(bool(cp.flags & FLAG), flagged)
        self.assertEqual(car_unavailable_reason(ENGINE_COAST, cp) is None, brand == "toyota" and op_long)

  def test_the_planner_coasts_to_the_engine_where_it_fits_and_brakes_where_it_does_not(self):
    floor = engine_coast_floor(20.0)
    # 1 m/s of closing at 1.3 s: the default presses, the engine coast absorbs it
    self.assertLess(min(_follow(False, 19.0, 26.0)), -0.5)
    self.assertGreaterEqual(min(_follow(True, 19.0, 26.0)), floor - 1e-6)
    # the same closing at 1.05 s, and a lead braking hard: the coast cannot fit, so the brake is as deep
    # (the shifted band starts its ramp from the engine coast, so the path there differs, not the depth)
    for args in ((19.0, 21.0), (20.0, 26.0, -3.0)):
      with self.subTest(args=args):
        self.assertAlmostEqual(min(_follow(True, *args)), min(_follow(False, *args)), delta=1e-6)

  def test_a_far_shallow_arrival_holds_speed_only_with_the_engine_coast(self):
    """Route 0000044d: a lead 5 m/s slower at 119 m drew -0.13 from first sight. With the engine coast the
    arrival holds there, then ramps in continuously and is whole past HOLD_FULL; default mode, whose only
    lamp-free decel is -0.10, still coasts from first sight."""
    floor = engine_coast_floor(20.0)

    def ask(gap, coast=floor, hold=True):
      if hold:
        return lead_accel(20.0, gap, 15.0, 0.0, 1.45, coast=coast)
      with mock.patch.multiple(longitudinal, MOONPILOT_ARRIVAL_HOLD_START=0.0, MOONPILOT_ARRIVAL_HOLD_FULL=0.0):
        return lead_accel(20.0, gap, 15.0, 0.0, 1.45, coast=coast)

    self.assertLess(ask(119.0, hold=False), -0.12)
    self.assertEqual(ask(119.0), 0.0)
    self.assertEqual(ask(119.0, coast=MOONPILOT_BRAKE_BAND_FLOOR), ask(119.0, coast=MOONPILOT_BRAKE_BAND_FLOOR, hold=False))
    self.assertLess(ask(60.0), MOONPILOT_ARRIVAL_HOLD_FULL * floor)
    self.assertEqual(ask(60.0), ask(60.0, hold=False))
    asks = np.array([ask(gap) for gap in np.arange(130.0, 40.0, -0.05)])
    self.assertLessEqual(np.diff(asks).max(), 0.0)  # never eases as the gap closes
    self.assertLess(np.abs(np.diff(asks)).max(), 0.01)  # and never steps

  def test_the_coast_budget_is_a_blend_not_a_switch(self):
    """As a switch a steady -1.0 ask stepped to the coast and back with the margin's noise. Blended, the output never
    leaves [ask, coast], is the ask where the coast does not fit and the coast past MOONPILOT_ENGINE_COAST_MARGIN of
    margin, and moves continuously with the gap and the ask (halving the step halves the largest change)."""
    floor = engine_coast_floor(15.0)

    def budget(ask, gap):
      return coast_budget(ask, 15.0, gap, 14.0, -0.3, floor, 1.0, 1.0, 4.0)
    margins = [coast_margin(15.0, gap, 14.0, -0.3, floor, 1.0, 4.0) for gap in (10.0, 60.0)]
    self.assertLess(margins[0], 0.0)
    self.assertGreater(margins[1], MOONPILOT_ENGINE_COAST_MARGIN)
    self.assertEqual(budget(-1.0, 10.0), -1.0)
    self.assertAlmostEqual(budget(-1.0, 60.0), floor, places=9)
    for sweep, lo, hi in ((lambda x: budget(-1.0, x), 10.0, 60.0), (lambda x: budget(x, 60.0), -1.5, floor)):
      outs = [[sweep(x) for x in np.linspace(lo, hi, n)] for n in (201, 401)]
      self.assertTrue(all(-1.5 - 1e-9 <= o <= floor + 1e-9 for o in outs[1]))
      self.assertLess(np.abs(np.diff(outs[1])).max(), 0.6 * np.abs(np.diff(outs[0])).max())

  def test_the_flag_means_nothing_on_another_brand(self):
    """Bit 20 belongs to `ToyotaFlags` only; the same bit on another brand's flags is that brand's own."""
    planner = _planner(False)
    planner.CP.brand = "hyundai"
    planner.CP.flags |= FLAG
    out = []
    v_lead, gap = 19.0, 26.0
    for _ in range(60):
      planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(gap, v_lead)))
      out.append(planner.output_a_target)
      gap -= (20.0 - v_lead) * DT_MDL
    self.assertEqual(out, _follow(False, 19.0, 26.0))


if __name__ == "__main__":
  unittest.main()
