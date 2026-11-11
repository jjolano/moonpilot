"""Turn desire at low speed: blinker feeds a turn desire into the model."""

import unittest

from openpilot.cereal import log

from moonpilot.tests.harness import inputs

from moonpilot.tests.fakes import params
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN
from moonpilot.turn_desire import turn_desire, turn_desire_alert



def _desire_state(left=0.0, right=0.0):
  desire_state = [0.0] * 8
  desire_state[log.Desire.turnLeft] = left
  desire_state[log.Desire.turnRight] = right
  return desire_state


class TestTurnDesire(unittest.TestCase):
  def test_gate_blinker_and_model_override(self):
    store = params(on=True)
    sm = inputs(v_ego=5.0)
    sm["carState"].leftBlinker = True
    cs = sm["carState"]
    self.assertEqual(turn_desire(cs, None, store), log.Desire.turnLeft)
    self.assertEqual(turn_desire(cs, _desire_state(right=0.9), store), log.Desire.turnRight)
    slow_cs = inputs(v_ego=LANE_CHANGE_SPEED_MIN + 0.5)["carState"]  # above the gate: carState rounds a float32
    slow_cs.leftBlinker = True
    self.assertIsNone(turn_desire(slow_cs, None, store))
    self.assertIsNone(turn_desire(cs, None, params(on=False)))

  def test_banner_requires_active_lateral_and_uses_explicit_car_state(self):
    cs = inputs(v_ego=5.0)["carState"]
    cs.leftBlinker = True
    sm = inputs(lat_active=True)
    self.assertEqual(turn_desire_alert(sm, params(on=True), cs), log.OnroadEvent.EventName.turnLeft)
    sm = inputs(lat_active=False)
    self.assertIsNone(turn_desire_alert(sm, params(on=True), cs))


if __name__ == "__main__":
  unittest.main()
