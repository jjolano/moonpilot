"""Turn desire at low speed: blinker feeds a turn desire into the model."""

import unittest
from unittest import mock

from openpilot.cereal import log
from openpilot.common.constants import CV

from moonpilot.turn_desire import LANE_CHANGE_SPEED_MIN, turn_desire


class FakeParams:
  """Duck-typed Params with explicit and declared-default reads."""

  def __init__(self, values=None, default=None):
    self._values = dict(values or {})
    self._default = default

  def get(self, key, block=False, return_default=False):
    if key in self._values:
      return self._values[key]
    return self._default if return_default else None


def _params(values=None, default=None):
  return FakeParams(values, default=default)


class _FakeCarState:
  def __init__(self, vEgo=0.0, leftBlinker=False, rightBlinker=False):
    self.vEgo = vEgo
    self.leftBlinker = leftBlinker
    self.rightBlinker = rightBlinker


class _FakeSubMaster:
  """Only exposes the carState consumed by turn_desire."""

  def __init__(self, car_state=None):
    self.carState = car_state or _FakeCarState()

  def __getitem__(self, key):
    if key != "carState":
      raise KeyError(key)
    return self.carState


def _sm(vEgo=0.0, leftBlinker=False, rightBlinker=False):
  return _FakeSubMaster(_FakeCarState(vEgo, leftBlinker, rightBlinker))


def _desire_state(left=0.0, right=0.0):
  desire_state = [0.0] * 8
  desire_state[log.Desire.turnLeft] = left
  desire_state[log.Desire.turnRight] = right
  return desire_state


class TestTurnDesire(unittest.TestCase):
  def test_feature_off_returns_none(self):
    sm = _sm(vEgo=10.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 0})
    self.assertIsNone(turn_desire(sm, _desire_state(left=0.9), params))

  def test_speed_above_threshold_returns_none(self):
    sm = _sm(vEgo=LANE_CHANGE_SPEED_MIN + 0.1, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertIsNone(turn_desire(sm, params=params))

  def test_no_blinker_returns_none(self):
    sm = _sm(vEgo=5.0)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertIsNone(turn_desire(sm, params=params))

  def test_left_blinker_returns_turn_left(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertEqual(turn_desire(sm, params=params), log.Desire.turnLeft)

  def test_right_blinker_returns_turn_right(self):
    sm = _sm(vEgo=5.0, rightBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertEqual(turn_desire(sm, params=params), log.Desire.turnRight)

  def test_model_override_opposes_left_blinker(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    desire_state = _desire_state(left=0.05, right=0.9)
    self.assertEqual(turn_desire(sm, desire_state, params), log.Desire.turnRight)

  def test_model_override_opposes_right_blinker(self):
    sm = _sm(vEgo=5.0, rightBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    desire_state = _desire_state(left=0.9, right=0.05)
    self.assertEqual(turn_desire(sm, desire_state, params), log.Desire.turnLeft)

  def test_model_fallback_to_blinker(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    desire_state = _desire_state(left=0.5, right=0.45)
    self.assertEqual(turn_desire(sm, desire_state, params), log.Desire.turnLeft)

  def test_model_threshold_is_strict(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    desire_state = _desire_state(left=0.2, right=0.4)
    self.assertEqual(turn_desire(sm, desire_state, params), log.Desire.turnLeft)

  def test_missing_desire_state_uses_blinker(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertEqual(turn_desire(sm, params=params), log.Desire.turnLeft)

  def test_none_desire_state_uses_blinker(self):
    sm = _sm(vEgo=5.0, rightBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertEqual(turn_desire(sm, None, params), log.Desire.turnRight)

  def test_exact_threshold_returns_none(self):
    sm = _sm(vEgo=LANE_CHANGE_SPEED_MIN, leftBlinker=True)
    params = _params({"MoonpilotTurnDesire": 1})
    self.assertIsNone(turn_desire(sm, params=params))

  def test_default_params_honor_registered_default_on(self):
    sm = _sm(vEgo=5.0, leftBlinker=True)
    params = _params(default=1)
    with mock.patch("openpilot.common.params.Params", return_value=params) as params_factory:
      self.assertEqual(turn_desire(sm), log.Desire.turnLeft)
    params_factory.assert_called_once_with()


class TestLaneChangeSpeedMin(unittest.TestCase):
  def test_matches_upstream(self):
    self.assertEqual(LANE_CHANGE_SPEED_MIN, 20 * CV.MPH_TO_MS)
    self.assertAlmostEqual(LANE_CHANGE_SPEED_MIN * CV.MS_TO_KPH, 32.0, delta=0.5)


if __name__ == "__main__":
  unittest.main()
