"""Turn desire at low speed: blinker feeds a turn desire into the model.

Tests the moonpilot/turn_desire.py function directly — the feature is offroad-only,
but the function itself is tested in isolation with duck-typed SubMaster mocks.
"""

import unittest
from types import SimpleNamespace
from openpilot.cereal import log
from openpilot.common.constants import CV

from moonpilot.turn_desire import turn_desire, LANE_CHANGE_SPEED_MIN


class FakeParams:
    """Duck-typed stand-in for Params."""

    def __init__(self, values=None):
        self._values = dict(values or {})

    def get(self, key, block=False, return_default=False):
        return self._values.get(key)


def _params(values=None):
    return FakeParams(values)


class _FakeCarState:
    """Duck-typed stand-in for carState."""

    def __init__(self, vEgo=0.0, leftBlinker=False, rightBlinker=False):
        self.vEgo = vEgo
        self.leftBlinker = leftBlinker
        self.rightBlinker = rightBlinker


class _FakeModelV2:
    """Duck-typed stand-in for modelV2 with meta.desireState."""

    def __init__(self, desire_state=None):
        self.meta = SimpleNamespace(desireState=desire_state or [0.0] * 7)


class _FakeSubMaster:
    """SubMaster mock supporting both attribute and subscript access."""
    def __init__(self, car_state=None, model_v2=None):
        self.carState = car_state or _FakeCarState()
        self.modelV2 = model_v2

    def __getitem__(self, key):
        return getattr(self, key)


def _sm(vEgo=0.0, leftBlinker=False, rightBlinker=False, desire_state=None, modelv2_enabled=True):
    """Build a duck-typed SubMaster mock."""
    cs = _FakeCarState(vEgo=vEgo, leftBlinker=leftBlinker, rightBlinker=rightBlinker)
    mv2 = _FakeModelV2(desire_state=desire_state) if modelv2_enabled else None
    sm = _FakeSubMaster(car_state=cs, model_v2=mv2)
    return sm


class TestTurnDesire(unittest.TestCase):
    """The turn_desire function: feature-gated, speed-gated, blinker-gated,
    and model-informed."""

    def test_feature_off_returns_none(self):
        """When the MoonpilotTurnDesire param is 0, turn_desire is always None."""
        sm = _sm(vEgo=10.0, leftBlinker=True)
        params = _params({"MoonpilotTurnDesire": 0})
        self.assertIsNone(turn_desire(sm, params))

    def test_speed_above_threshold_returns_none(self):
        """At or above LANE_CHANGE_SPEED_MIN, turn_desire is always None."""
        sm = _sm(vEgo=LANE_CHANGE_SPEED_MIN + 0.1, leftBlinker=True)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertIsNone(turn_desire(sm, params))

    def test_no_blinker_returns_none(self):
        """No blinker on means no turn desire regardless of speed."""
        sm = _sm(vEgo=5.0, leftBlinker=False, rightBlinker=False)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertIsNone(turn_desire(sm, params))

    def test_left_blinker_returns_turn_left(self):
        """Left blinker at low speed returns turnLeft."""
        sm = _sm(vEgo=5.0, leftBlinker=True, rightBlinker=False)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertEqual(turn_desire(sm, params), log.Desire.turnLeft)

    def test_right_blinker_returns_turn_right(self):
        """Right blinker at low speed returns turnRight."""
        sm = _sm(vEgo=5.0, leftBlinker=False, rightBlinker=True)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertEqual(turn_desire(sm, params), log.Desire.turnRight)

    def test_model_override_left(self):
        """When the model strongly predicts turnLeft, it overrides the blinker."""
        sm = _sm(vEgo=5.0, leftBlinker=True, rightBlinker=False,
                 desire_state=[0.0, 0.9, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = _params({"MoonpilotTurnDesire": 1})
        # turnLeft prob (0.9) > turnRight prob (0.05) + 0.2 → use model's turnLeft
        self.assertEqual(turn_desire(sm, params), log.Desire.turnLeft)

    def test_model_override_right(self):
        """When the model strongly predicts turnRight, it overrides the blinker."""
        sm = _sm(vEgo=5.0, leftBlinker=True, rightBlinker=False,
                 desire_state=[0.0, 0.05, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = _params({"MoonpilotTurnDesire": 1})
        # turnRight prob (0.9) > turnLeft prob (0.05) + 0.2 → use model's turnRight
        self.assertEqual(turn_desire(sm, params), log.Desire.turnRight)

    def test_model_fallback_to_blinker(self):
        """When model probs are close (within 0.2), fall back to the blinker."""
        sm = _sm(vEgo=5.0, leftBlinker=True, rightBlinker=False,
                 desire_state=[0.0, 0.5, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = _params({"MoonpilotTurnDesire": 1})
        # 0.5 vs 0.45 → diff is 0.05 < 0.2 → fall back to leftBlinker
        self.assertEqual(turn_desire(sm, params), log.Desire.turnLeft)

    def test_model_not_available_uses_blinker(self):
        """When modelV2 is unavailable, fall back to the blinker."""
        sm = _sm(vEgo=5.0, leftBlinker=True, rightBlinker=False, modelv2_enabled=False)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertEqual(turn_desire(sm, params), log.Desire.turnLeft)

    def test_exact_threshold_returns_none(self):
        """Exactly at LANE_CHANGE_SPEED_MIN returns None (the boundary is >=)."""
        sm = _sm(vEgo=LANE_CHANGE_SPEED_MIN, leftBlinker=True)
        params = _params({"MoonpilotTurnDesire": 1})
        self.assertIsNone(turn_desire(sm, params))

    def test_default_params_returns_none(self):
        """When no params passed, feature defaults to off (return_default=True not used)."""
        sm = _sm(vEgo=5.0, leftBlinker=True)
        # Params() without MoonpilotTurnDesire key → enabled returns False
        self.assertIsNone(turn_desire(sm))

    def test_desire_values_are_correct_enum(self):
        """The returned desire values are valid log.Desire enum values."""
        sm = _sm(vEgo=5.0, leftBlinker=True)
        params = _params({"MoonpilotTurnDesire": 1})
        result = turn_desire(sm, params)
        self.assertEqual(result, log.Desire.turnLeft)
        self.assertIn(result, (log.Desire.turnLeft, log.Desire.turnRight))


class TestLaneChangeSpeedMin(unittest.TestCase):
    """LANE_CHANGE_SPEED_MIN matches the upstream definition."""

    def test_matches_upstream(self):
        """20 mph should be ~32 km/h."""
        from openpilot.common.constants import CV
        expected = 20 * CV.MPH_TO_MS
        self.assertEqual(LANE_CHANGE_SPEED_MIN, expected)
        # Verify it's ~32 km/h: m/s * MS_TO_KPH
        self.assertAlmostEqual(LANE_CHANGE_SPEED_MIN * CV.MS_TO_KPH, 32.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()