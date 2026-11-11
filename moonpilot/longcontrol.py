"""moonpilot's acceleration controller: the fork's own answer to "how do we track that?".

Upstream's LongControl is a PI loop on the accel error with a feedforward of the plan's target, and
that is the right family — the car is the plant, and the only thing that makes it work is the gain
schedule the car ships. The fork keeps the shape and the car's own `longitudinalTuning.kiV` table
(the same reason `moonpilot/latcontrol.py` keeps `lateralTuning.torque`) and owns the rest:

  - the commanded accel is rate limited, so a step in the plan is not a step at the actuator — except
    at a positive handover out of the stopping state, where the held brake command is not a plan
    value;
  - the integrator freezes at standstill and while the driver is on the pedals, where neither the
    error nor the measurement means anything;
  - while rolling into a stop, braking follows the plan's nonpositive target with a small minimum
    deceleration; only physical standstill ramps toward `CP.stopAccel`.

`long_control_state_trans` is upstream's, imported and reused: it carries no tuning, it is the
LongControlState contract that controlsd, controlsState and both UIs read; upstream's own test pins it.
"""

import numpy as np

from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.pid import PIDController
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import long_control_state_trans

from moonpilot.features import CRAWL_RELEASE, LONGITUDINAL, in_force

LongCtrlState = car.CarControl.Actuators.LongControlState

MOONPILOT_ACCEL_JERK = 8.0
MOONPILOT_STOPPING_JERK = 1.0
MOONPILOT_MIN_ROLLING_STOP_DECEL = 0.2
MOONPILOT_STANDSTILL_SPEED = 0.1
MOONPILOT_CRAWL_RELEASE_SPEED = 0.5  # m/s
MOONPILOT_CRAWL_RELEASE_FULL_ASK = 0.2  # m/s^2
MOONPILOT_CRAWL_RELEASE_MAX_ASK = 0.3  # m/s^2; a deeper ask is a stop that needs its braking, untouched
MOONPILOT_CRAWL_RELEASE_MIN = 0.3  # share of the ask (and of the rolling stop's floor) kept at the end


class MoonpilotLongControl:
  """Same public surface as upstream's controller — `long_control_state`, `reset`, `update` with the
  same positional contract — because controlsd's seam line calls it and reads the state."""

  def __init__(self, CP, crawl_release=False):
    self.CP = CP
    self.crawl_release = crawl_release
    self.long_control_state = LongCtrlState.off
    self.last_output_accel = 0.0
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV), rate=1 / DT_CTRL)

  def _crawl_share(self, CS, a_target) -> float:
    """How much of a gentle braking ask, and of the rolling stop's floor, is kept in the last of a stop: 1 unless
    `MoonpilotCrawlRelease` is on and the car is still rolling under MOONPILOT_CRAWL_RELEASE_SPEED. Continuous in both:
    the speed share fades out between FULL_ASK and MAX_ASK of depth, and an ask at or above zero gets the full share, so
    the stopping floor does not snap back when the plan crosses zero."""
    if not self.crawl_release or CS.standstill or CS.vEgo >= MOONPILOT_CRAWL_RELEASE_SPEED:
      return 1.0
    speed_share = max(CS.vEgo / MOONPILOT_CRAWL_RELEASE_SPEED, MOONPILOT_CRAWL_RELEASE_MIN)
    depth = (a_target + MOONPILOT_CRAWL_RELEASE_MAX_ASK) / (MOONPILOT_CRAWL_RELEASE_MAX_ASK - MOONPILOT_CRAWL_RELEASE_FULL_ASK)
    return 1.0 - (1.0 - speed_share) * min(max(depth, 0.0), 1.0)

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits) -> float:
    self.pid.neg_limit, self.pid.pos_limit = accel_limits

    previous_state = self.long_control_state
    share = self._crawl_share(CS, a_target)
    if a_target < 0.0:
      a_target *= share  # a positive ask is never scaled; the share still eases the stopping floor below
    self.long_control_state = long_control_state_trans(active, self.long_control_state, should_stop, CS.brakePressed, CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      self.last_output_accel = 0.0
      return 0.0

    if previous_state == LongCtrlState.stopping and self.long_control_state == LongCtrlState.pid:
      # A positive plan releases a held brake immediately; negative requests retain their braking
      # baseline across the state edge instead of being reset toward zero.
      if a_target > 0.0:
        self.last_output_accel = 0.0
      self.reset()

    if self.long_control_state == LongCtrlState.stopping:
      if CS.standstill:
        output_accel = max(min(self.last_output_accel, 0.0) - MOONPILOT_STOPPING_JERK * DT_CTRL, self.CP.stopAccel)
        output_accel = float(
          np.clip(output_accel, self.last_output_accel - MOONPILOT_ACCEL_JERK * DT_CTRL, self.last_output_accel + MOONPILOT_ACCEL_JERK * DT_CTRL)
        )
      else:
        rolling_target = min(a_target, -MOONPILOT_MIN_ROLLING_STOP_DECEL * share)
        output_accel = float(
          np.clip(rolling_target, self.last_output_accel - MOONPILOT_ACCEL_JERK * DT_CTRL, self.last_output_accel + MOONPILOT_ACCEL_JERK * DT_CTRL)
        )
      self.reset()
      self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
      return self.last_output_accel

    if share < 1.0 and a_target < 0.0:
      # Under the crawl release the eased ask is commanded as is: this car decelerates past a gentle crawl ask, so tracking
      # it would wind toward throttle, and an integral carried in would cancel it. kp is 0, so the PID is i + feedforward
      # and resetting it leaves the feedforward — and resumes from zero integral as the release fades out.
      self.reset()
      output_accel = a_target
    else:
      error = a_target - CS.aEgo
      freeze = CS.vEgo < MOONPILOT_STANDSTILL_SPEED or CS.brakePressed or CS.gasPressed
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target, freeze_integrator=freeze)

    output_accel = float(
      np.clip(output_accel, self.last_output_accel - MOONPILOT_ACCEL_JERK * DT_CTRL, self.last_output_accel + MOONPILOT_ACCEL_JERK * DT_CTRL)
    )
    self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
    return self.last_output_accel


def moonpilot_longcontrol(CP, params: Params | None = None) -> MoonpilotLongControl | None:
  """The fork controller when the driver wants it on a car it can run on, None to leave upstream's in place; read
  once, so a restart."""
  if not in_force(LONGITUDINAL, params := params or Params(), CP):
    return None
  return MoonpilotLongControl(CP, crawl_release=in_force(CRAWL_RELEASE, params, CP))
