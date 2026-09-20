"""Low-speed turn desire: blinker on below LANE_CHANGE_SPEED_MIN feeds
turnLeft/turnRight into the model rather than being ignored.
"""

from openpilot.cereal import log
from openpilot.common.constants import CV

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS  # ~32 km/h, same as upstream


def turn_desire(sm, desire_state=None, params=None) -> int | None:
    """Return a turn desire (log.Desire.turnLeft / turnRight) or None.

    Conditions:
    - Feature must be enabled: enabled(TURN_DESIRE, Params() when omitted)
    - Speed must be below LANE_CHANGE_SPEED_MIN
    - A blinker must be on (leftBlinker != rightBlinker)
    - A strong current model desireState prediction overrides the blinker
    """
    from openpilot.common.params import Params
    from moonpilot.features import TURN_DESIRE, enabled

    if params is None:
        params = Params()
    if not enabled(TURN_DESIRE, params):
        return None

    v_ego = max(sm["carState"].vEgo, 0.0)
    if v_ego >= LANE_CHANGE_SPEED_MIN:
        return None

    one_blinker = sm["carState"].leftBlinker != sm["carState"].rightBlinker
    if not one_blinker:
        return None

    # If one direction is clearly preferred, use the current model prediction;
    # otherwise fall back to the blinker.
    if desire_state is not None:
        l_prob = desire_state[log.Desire.turnLeft]
        r_prob = desire_state[log.Desire.turnRight]
        if l_prob > r_prob + 0.2:
            return log.Desire.turnLeft
        elif r_prob > l_prob + 0.2:
            return log.Desire.turnRight

    # Blinker fallback: left = turnLeft, right = turnRight
    if sm["carState"].leftBlinker:
        return log.Desire.turnLeft
    elif sm["carState"].rightBlinker:
        return log.Desire.turnRight

    return None
