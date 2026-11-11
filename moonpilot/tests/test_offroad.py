"""Offroad mode: the param, the gate both entries share, and the one line in hardwared.

The policy is pure -- three booleans in, one out -- so it is tested directly, and the only thing
that is not is the seam: `hardwared` is the sole reader of the param on the device, nothing in this
tree imports it, and a merge that drops its two lines deletes the feature silently. That pair is
pinned by reading the file, the way params_keys.h and the version string already are.
"""

import unittest
from pathlib import Path

from moonpilot import offroad
from moonpilot.tests.fakes import FakeParams

ROOT = Path(__file__).resolve().parents[2]

def _params(requested: bool | None = None) -> FakeParams:
  return FakeParams(on=False, overrides={offroad.MOONPILOT_OFFROAD_KEY: requested} if requested is not None else {})





class TestPolicy(unittest.TestCase):

  def test_can_enter_requires_the_car_on_parked_and_disengaged(self):
    self.assertTrue(offroad.can_enter(True, 0.0, False))
    # A car crawling in traffic at a standstill is engaged; a parked car with the engine running
    # is neither.
    self.assertFalse(offroad.can_enter(True, 0.0, True))
    self.assertFalse(offroad.can_enter(True, 0.5, False))
    self.assertFalse(offroad.can_enter(False, 0.0, False))
    # The boundary is strict, so the speed is the last one that counts as parked.
    self.assertFalse(offroad.can_enter(True, offroad.MOONPILOT_OFFROAD_SPEED, False))
    self.assertTrue(offroad.can_enter(True, offroad.MOONPILOT_OFFROAD_SPEED - 1e-6, False))

  def test_the_label_names_why_the_action_is_unavailable(self):
    self.assertEqual(offroad.button_label(_params(True), started=False, parked=False), offroad.EXIT_LABEL)
    self.assertEqual(offroad.button_label(_params(True), started=True, parked=False), offroad.EXIT_LABEL)
    self.assertEqual(offroad.button_label(_params(), started=False, parked=False), offroad.OFFROAD_LABEL)
    self.assertEqual(offroad.button_label(_params(), started=True, parked=False), offroad.NOT_PARKED_LABEL)
    self.assertEqual(offroad.button_label(_params(), started=True, parked=True), offroad.ENTER_LABEL)

  def test_leaving_offroad_is_never_gated(self):
    """The device has to stay recoverable from the state the mode puts it in, and the mode's own
    state is `started` false: gating the way out on that would strand the car."""
    for parked in (True, False):
      with self.subTest(parked=parked):
        self.assertTrue(offroad.enabled(_params(True), parked))
    self.assertTrue(offroad.enabled(_params(), parked=True))
    self.assertFalse(offroad.enabled(_params(), parked=False))



class TestWiring(unittest.TestCase):
  def test_the_seam_is_in_the_onroad_conditions(self):
    """`onroad_conditions` is upstream's own "may this device be onroad" dict, and
    `should_start = all(onroad_conditions.values())` is what reads it: one member whose value is
    the inverse of the request is the entire mechanism. The refresh sits before the ignition-edge
    test, so a flip of the param is published on that tick rather than at the next 2 Hz one."""
    text = (ROOT / "openpilot" / "system" / "hardware" / "hardwared.py").read_text()
    self.assertTrue('"moonpilot_onroad": True,' in text)
    self.assertTrue('onroad_conditions["moonpilot_onroad"] = onroad_condition(params)' in text)
    self.assertLess(text.index('onroad_conditions["moonpilot_onroad"] = onroad_condition(params)'), text.index("ign_edge = "))


if __name__ == "__main__":
  unittest.main()
