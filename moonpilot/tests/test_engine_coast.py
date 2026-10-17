"""The engine-braking coast (`MoonpilotEngineCoast`): the car side's PERMIT_BRAKING hold, the card hook
that turns it on, and the planner coasting to it — with the default untouched when it is off."""

import unittest
from unittest import mock

import numpy as np
from opendbc.car.moonpilot.engine_coast import ENGINE_COAST_BP, ENGINE_COAST_EXIT, ENGINE_COAST_V, EngineCoast, engine_coast_floor
from opendbc.car.toyota.interface import CarInterface
from opendbc.car.toyota.values import CAR, ToyotaFlags

from moonpilot.engage import car_unavailable_reason, moonpilot_engage_safety_param
from moonpilot.features import ENGINE_COAST
from moonpilot.longitudinal import MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V, MoonpilotLongitudinalPlanner
from moonpilot.tests.fakes import _params
from moonpilot.tests.test_engage import _cp
from moonpilot.tests.test_longitudinal import DT_MDL, _inputs, _lead

FLAG = int(ToyotaFlags.MOONPILOT_ENGINE_COAST)


def _planner(flag: bool) -> MoonpilotLongitudinalPlanner:
  CP = CarInterface.get_non_essential_params(CAR.TOYOTA_RAV4_TSS2)
  CP.openpilotLongitudinalControl = True
  if flag:
    CP.flags |= FLAG
  fake = _params(on=True)
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
