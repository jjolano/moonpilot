"""The longitudinal brake band: shallow negative asks coast, deep ones are one committed press."""

import unittest

from opendbc.car.structs import car
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET

from moonpilot.brake_band import (
  MOONPILOT_BRAKE_BAND_ENTER,
  MOONPILOT_BRAKE_BAND_ENTER_LEAD,
  MOONPILOT_BRAKE_BAND_FLOOR,
  MOONPILOT_BRAKE_BAND_HOLD,
  MOONPILOT_BRAKE_BAND_MIN_PRESS_T,
  MOONPILOT_BRAKE_BAND_MIN_SPEED,
  MOONPILOT_BRAKE_BAND_RELEASE,
  MOONPILOT_BRAKE_BAND_RELEASE_T,
  BrakeBand,
)
from moonpilot.features import FEATURES
from moonpilot.tests.test_longitudinal import COAST_SET_SPEED, ROOT, _inputs, _lead, _planner


class TestBrakeBand(unittest.TestCase):
  """Shallow negative asks coast; deep ones are one committed press; nothing changes at a crawl."""

  V = 15.0

  def _run(self, asks, v=V, band=None):
    band = band or BrakeBand()
    return [band.update(a, v, DT_MDL) for a in asks]

  def test_a_shallow_ask_is_raised_to_the_coast_floor_and_others_pass_bit_exact(self):
    for ask in (-0.15, -0.25, MOONPILOT_BRAKE_BAND_ENTER):
      with self.subTest(ask=ask):
        self.assertEqual(self._run([ask] * 40), [MOONPILOT_BRAKE_BAND_FLOOR] * 40)
    for ask in (0.3, 0.0, -0.05, MOONPILOT_BRAKE_BAND_FLOOR, -0.6):
      with self.subTest(ask=ask):
        self.assertEqual(self._run([ask])[0], ask)

  def test_behind_a_slowing_lead_the_press_commits_earlier(self):
    ask = (MOONPILOT_BRAKE_BAND_ENTER_LEAD + MOONPILOT_BRAKE_BAND_ENTER) / 2  # between the two commit points
    self.assertEqual(BrakeBand().update(ask, self.V, DT_MDL, urgent=False), MOONPILOT_BRAKE_BAND_FLOOR)
    band = BrakeBand()
    self.assertEqual(band.update(ask, self.V, DT_MDL, urgent=True), min(ask, MOONPILOT_BRAKE_BAND_HOLD))
    # Once committed, the lead's own reading no longer decides: a shallower ask is still the press.
    self.assertEqual(band.update(-0.15, self.V, DT_MDL, urgent=False), MOONPILOT_BRAKE_BAND_HOLD)
    # Short of the earlier commit point it is still a coast, slowing lead or not.
    self.assertEqual(BrakeBand().update(MOONPILOT_BRAKE_BAND_ENTER_LEAD + 0.01, self.V, DT_MDL, urgent=True), MOONPILOT_BRAKE_BAND_FLOOR)

  def test_a_deep_ask_commits_a_press_that_holds_then_lets_go(self):
    hold_frames = int(round(MOONPILOT_BRAKE_BAND_MIN_PRESS_T / DT_MDL))
    out = self._run([-0.6] + [-0.1] * (hold_frames + 2))
    self.assertEqual(out[0], -0.6)
    # Inside the minimum press the ask is held at the firm level the car keeps its brakes on at.
    self.assertTrue(all(x == MOONPILOT_BRAKE_BAND_HOLD for x in out[1:hold_frames]))
    # Past it, an ask that has stayed above the release lets go straight to the coast floor: no second press.
    self.assertEqual(out[-1], MOONPILOT_BRAKE_BAND_FLOOR)

  def test_the_press_lets_go_only_once_the_ask_stays_above_release(self):
    over, under = MOONPILOT_BRAKE_BAND_RELEASE + 0.05, MOONPILOT_BRAKE_BAND_RELEASE - 0.05
    release_frames = int(round(MOONPILOT_BRAKE_BAND_RELEASE_T / DT_MDL))
    # Past the minimum press, an ask hovering across the release is still one press, deeper asks followed.
    out = self._run([-0.6, -1.2] + [-0.35] * 20 + [over, under] * 10 + [over] * release_frames)
    self.assertEqual(out[:3], [-0.6, -1.2, -0.35])
    self.assertTrue(all(x == MOONPILOT_BRAKE_BAND_HOLD for x in out[22:-1]))
    self.assertEqual(out[-1], MOONPILOT_BRAKE_BAND_FLOOR)

  def test_a_positive_ask_ends_a_press_at_once(self):
    self.assertEqual(self._run([-0.6, 0.2]), [-0.6, 0.2])

  def test_below_the_speed_gate_nothing_is_shaped_and_no_press_survives(self):
    band = BrakeBand()
    self.assertEqual(self._run([-0.6], band=band), [-0.6])
    slow = MOONPILOT_BRAKE_BAND_MIN_SPEED - 0.1
    self.assertEqual(self._run([-0.2, -0.1], v=slow, band=band), [-0.2, -0.1])
    self.assertEqual(self._run([-0.1], band=band), [MOONPILOT_BRAKE_BAND_FLOOR])  # back above: coasting, not pressing

  def test_the_planner_coasts_a_slight_overspeed_and_the_row_turns_it_off(self):
    """1 m/s over the set speed the cruise rung asks for about -0.2: shaped, it settles on the coast
    floor; with the row off it settles on the rung."""
    settled = {}
    for on in (True, False):
      planner = _planner(params_overrides={"MoonpilotBrakeBand": on})
      sm = _inputs(v_ego=COAST_SET_SPEED + 1.0, v_cruise_kph=COAST_SET_SPEED * 3.6)
      for _ in range(100):
        planner.update(sm)
      settled[on] = planner.output_a_target
    self.assertAlmostEqual(settled[True], MOONPILOT_BRAKE_BAND_FLOOR, delta=1e-9)
    self.assertLess(settled[False], MOONPILOT_BRAKE_BAND_FLOOR - 0.05)
    self.assertGreater(settled[False], MOONPILOT_BRAKE_BAND_ENTER)

  def test_disengagement_cannot_arm_a_press_for_reengagement(self):
    for reset in ("long_off", "unset_cruise"):
      for previously_active in (False, True):
        with self.subTest(reset=reset, previously_active=previously_active):
          planner = _planner()
          deep = _inputs(v_ego=COAST_SET_SPEED + 1.0, v_cruise_kph=COAST_SET_SPEED * 3.6, lead=_lead(12.0, 0.0))
          if previously_active:
            planner.update(deep)
            self.assertTrue(planner.brake_band.pressing)
          if reset == "long_off":
            # Lateral-only engagement still reports selfdriveState.enabled.
            deep["controlsState"].longControlState = car.CarControl.Actuators.LongControlState.off
          else:
            deep["carState"].vCruise = V_CRUISE_UNSET
          for _ in range(3):
            planner.update(deep)
            self.assertFalse(planner.brake_band.pressing)
            self.assertEqual(planner.brake_band.held, 0.0)
            self.assertEqual(planner.brake_band.above, 0.0)
          shallow = _inputs(v_ego=COAST_SET_SPEED + 1.0, v_cruise_kph=COAST_SET_SPEED * 3.6)
          # Allow the ordinary jerk ramp, but stay inside the stale press's one-second hold.
          for _ in range(8):
            planner.update(shallow)
          self.assertAlmostEqual(planner.output_a_target, MOONPILOT_BRAKE_BAND_FLOOR, delta=1e-9)

  def test_sustained_gas_denial_keeps_its_coast_ceiling(self):
    """The model expects the driver off the gas: `cruise_cap` holds every ask at the coast accel, and
    nothing downstream enforces `allowThrottle` — only `aTarget` reaches the car. The band must not raise
    the emitted target above that ceiling, or the car would hold -0.10 on throttle."""
    settled = {}
    for on in (True, False):
      planner = _planner(params_overrides={"MoonpilotBrakeBand": on})
      sm = _inputs(v_ego=COAST_SET_SPEED, v_cruise_kph=COAST_SET_SPEED * 3.6, throttle_prob=0.0)
      for _ in range(100):
        planner.update(sm)
      self.assertFalse(planner.allow_throttle)
      settled[on] = planner.output_a_target
    self.assertLess(settled[False], MOONPILOT_BRAKE_BAND_FLOOR)  # the ceiling is a real coast, inside the band's raise
    self.assertAlmostEqual(settled[True], settled[False], delta=1e-9)

  def test_feature_row_is_on_and_registered(self):
    feature = next(f for f in FEATURES if f.key == "MoonpilotBrakeBand")
    self.assertTrue(feature.offroad_only)
    self.assertFalse(feature.requires)
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('{"MoonpilotBrakeBand", {PERSISTENT, BOOL, "1"}}' in text)


if __name__ == "__main__":
  unittest.main()
