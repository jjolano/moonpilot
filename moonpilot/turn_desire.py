"""Low-speed turn desire: at speeds below LANE_CHANGE_SPEED_MIN, blinker on
maps to a turn desire (turnLeft/turnRight) rather than being ignored.
"""

from openpilot.cereal import log
from openpilot.common.constants import CV

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS  # ~32 km/h, same as upstream


def turn_desire(sm, params=None) -> int | None:
    """Return a turn desire (log.Desire.turnLeft / turnRight) or None.

    Conditions:
    - Feature must be enabled: enabled(TURN_DESIRE, params or Params())
    - Speed must be below LANE_CHANGE_SPEED_MIN
    - A blinker must be on (leftBlinker != rightBlinker)
    - Falls back to the model's own desireState prediction if available
    """
    from openpilot.common.params import Params
    from moonpilot.features import TURN_DESIRE, enabled

    if not params:
      return None
    if not enabled(TURN_DESIRE, params):
      return None

    v_ego = max(sm["carState"].vEgo, 0.0)
    if v_ego >= LANE_CHANGE_SPEED_MIN:
      return None

    one_blinker = sm["carState"].leftBlinker != sm["carState"].rightBlinker
    if not one_blinker:
      return None

    # Read model's own desire prediction; if available and one direction has
    # significantly higher probability, use that; otherwise fall back to blinker
    desire_state = None
    try:
      desire_state = sm["modelV2"].meta.desireState
    except (KeyError, AttributeError):
      pass

    if desire_state is not None:
      l_prob = desire_state[log.Desire.turnLeft]
      r_prob = desire_state[log.Desire.turnRight]
      # If one is clearly preferred, use it; otherwise fall back to blinker
      if l_prob > r_prob + 0.2:
        return log.Desire.turnLeft
      elif r_prob > l_prob + 0.2:
        return log.Desire.turnRight
      # Fall through to blinker

    # Blinker fallback: left = turnLeft, right = turnRight
    if sm["carState"].leftBlinker:
      return log.Desire.turnLeft
    elif sm["carState"].rightBlinker:
      return log.Desire.turnRight

    return None