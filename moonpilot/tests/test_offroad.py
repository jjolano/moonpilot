"""Offroad mode: the param, the gate both entries share, and the one line in hardwared.

The policy is pure -- three booleans in, one out -- so it is tested directly, and the only thing
that is not is the seam: `hardwared` is the sole reader of the param on the device, nothing in this
tree imports it, and a merge that drops its two lines deletes the feature silently. That pair is
pinned by reading the file, the way params_keys.h and the version string already are.
"""

import unittest
from pathlib import Path

from moonpilot import offroad
from moonpilot.longcontrol import MOONPILOT_STANDSTILL_SPEED
from moonpilot.longitudinal import MOONPILOT_SHOULD_STOP_SPEED

ROOT = Path(__file__).resolve().parents[2]


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store.

  Models both reads faithfully: `return_default=True` sees the row's declared default while nothing
  has been written, and a plain read returns None for an unset key, as `Params.get` does.
  """

  def __init__(self, default=False):
    self._default = default
    self.written = None

  def get(self, key, return_default=False):
    if self.written is None:
      return self._default if return_default else None
    return self.written

  def put_bool(self, key, value, block=False):
    self.written = value


def _params(requested: bool | None = None) -> FakeParams:
  """The row with `"0"` declared, as it is in params_keys.h, and the fork's request in it."""
  params = FakeParams(default=False)
  if requested is not None:
    params.put_bool(offroad.MOONPILOT_OFFROAD_KEY, requested)
  return params


class TestPolicy(unittest.TestCase):
  def test_the_param_row_is_declared_and_cleared_by_both_edges(self):
    """The default is the off state, and both edges that end the mode are in the flags: a new
    ignition cycle and a manager start -- so neither a fresh drive nor a reboot can inherit it.
    The row is what declares them: a read of a key with no row is an UnknownKeyName out of the
    compiled param table, which a fake params in a test never sees.
    Whitespace-normalized, because the row is too long for one 80-column line in this file and
    the formatter wraps it."""
    text = " ".join((ROOT / "moonpilot" / "params_keys.h").read_text().split())
    self.assertTrue('{"MoonpilotOffroad", {CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON, BOOL, "0"}}' in text)

  def test_the_speed_is_the_forks_standstill_convention(self):
    """ "Parked" has to mean here what it means in the planner, or a car the planner calls at rest
    could be one this refuses to enter offroad (or the reverse)."""
    self.assertEqual(offroad.MOONPILOT_OFFROAD_SPEED, MOONPILOT_STANDSTILL_SPEED)
    self.assertEqual(offroad.MOONPILOT_OFFROAD_SPEED, MOONPILOT_SHOULD_STOP_SPEED)

  def test_requested_reads_the_declared_default(self):
    self.assertFalse(offroad.requested(_params()))
    self.assertTrue(offroad.requested(_params(True)))
    self.assertFalse(offroad.requested(_params(False)))

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

  def test_the_seam_condition_is_the_inverse_of_the_request(self):
    self.assertFalse(offroad.onroad_condition(_params(True)))
    self.assertTrue(offroad.onroad_condition(_params()))


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
