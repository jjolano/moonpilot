"""Low-speed turn desire from the blinker and model prediction."""

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.desire_helper import LANE_CHANGE_SPEED_MIN

from moonpilot.features import TURN_DESIRE, enabled


def turn_desire(car_state, desire_state, params) -> int | None:
  if not enabled(TURN_DESIRE, params):
    return None

  v_ego = max(car_state.vEgo, 0.0)
  if v_ego >= LANE_CHANGE_SPEED_MIN:
    return None

  one_blinker = car_state.leftBlinker != car_state.rightBlinker
  if not one_blinker:
    return None

  if desire_state is not None:
    l_prob = desire_state[log.Desire.turnLeft]
    r_prob = desire_state[log.Desire.turnRight]
    if l_prob > r_prob + 0.2:
      return log.Desire.turnLeft
    if r_prob > l_prob + 0.2:
      return log.Desire.turnRight

  return log.Desire.turnLeft if car_state.leftBlinker else log.Desire.turnRight


def turn_desire_alert(sm, params: Params, car_state) -> int | None:
  if not sm["carControl"].latActive:
    return None
  # A SubMaster that has never received modelV2 hands back a default message whose desireState is empty: read that
  # as "no prediction", which turn_desire falls back to the blinker for.
  desire_state = sm["modelV2"].meta.desireState
  if len(desire_state) <= log.Desire.turnRight:
    desire_state = None
  desire = turn_desire(car_state, desire_state, params)
  if desire == log.Desire.turnLeft:
    return log.OnroadEvent.EventName.turnLeft
  if desire == log.Desire.turnRight:
    return log.OnroadEvent.EventName.turnRight
  return None
