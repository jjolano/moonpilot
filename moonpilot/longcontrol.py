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
LongControlState contract that controlsd, controlsState and both UIs read, and upstream's own test
already pins it — a second copy would only drift. The controller is picked once, at construction,
so the toggle needs a restart.
"""

import numpy as np

from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.pid import PIDController
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import long_control_state_trans

from moonpilot.features import CRAWL_RELEASE, LONGITUDINAL, enabled

LongCtrlState = car.CarControl.Actuators.LongControlState

# Starting points, all fork-owned. Tune against logs.
MOONPILOT_ACCEL_JERK = 8.0  # m/s^3 limit on the commanded accel, so a plan step is not an actuator step
MOONPILOT_STOPPING_JERK = 1.0  # m/s^3 ramp toward CP.stopAccel once physically stopped
MOONPILOT_MIN_ROLLING_STOP_DECEL = 0.2  # m/s^2 ensures a zero-target rolling stop can finish
MOONPILOT_STANDSTILL_SPEED = 0.1  # m/s; below it aEgo is noise, so the integrator freezes
# `MoonpilotCrawlRelease` (test, off): below this speed, still rolling, a braking ask no deeper than MAX_ASK is scaled by
# the speed's share of it, never below MIN — the way a driver lets off just before the car stops. On route 00000468
# the car's own decel deepened from -0.14 to -0.28 m/s^2 between 0.37 and 0.25 m/s while the request sat at -0.16
# (wheel speeds and the VSC G sensor agree), and the rolling stop's floor stepped the request to -0.20 at 0.27 m/s.
# Whether a lighter request takes that out is the drive's answer. Open-loop from 0.4 m/s it doubles the crawl's own stopping
# distance — 1.0 m against 0.5 behind a -0.16 ask, 0.54 against 0.27 behind -0.3 — and a gap that closes faster than the
# plan wanted draws an ask past MAX_ASK, which passes untouched.
MOONPILOT_CRAWL_RELEASE_SPEED = 0.5  # m/s
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
    `MoonpilotCrawlRelease` is on and the car is still rolling under MOONPILOT_CRAWL_RELEASE_SPEED behind an ask no deeper
    than MOONPILOT_CRAWL_RELEASE_MAX_ASK."""
    if not self.crawl_release or CS.standstill or CS.vEgo >= MOONPILOT_CRAWL_RELEASE_SPEED or not -MOONPILOT_CRAWL_RELEASE_MAX_ASK <= a_target < 0.0:
      return 1.0
    return max(CS.vEgo / MOONPILOT_CRAWL_RELEASE_SPEED, MOONPILOT_CRAWL_RELEASE_MIN)

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits) -> float:
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit, self.pid.pos_limit = accel_limits

    previous_state = self.long_control_state
    share = self._crawl_share(CS, a_target)
    a_target *= share
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

    # LongCtrlState.pid
    error = a_target - CS.aEgo
    # Standing still, or with the driver on a pedal, the integrator would wind up on error that
    # is a measurement artifact or a command the driver is overriding. Under the crawl release the eased ask is not a
    # target to track — this car decelerates past a gentle ask at a crawl, so tracking it would wind toward throttle.
    freeze = CS.vEgo < MOONPILOT_STANDSTILL_SPEED or CS.brakePressed or CS.gasPressed or share < 1.0
    output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target, freeze_integrator=freeze)
    if share < 1.0:
      output_accel = min(output_accel, 0.0)  # the release lightens a braking ask; it never turns one into throttle

    output_accel = float(
      np.clip(output_accel, self.last_output_accel - MOONPILOT_ACCEL_JERK * DT_CTRL, self.last_output_accel + MOONPILOT_ACCEL_JERK * DT_CTRL)
    )
    self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
    return self.last_output_accel


def moonpilot_longcontrol(CP, params: Params | None = None) -> MoonpilotLongControl | None:
  """The seam's fork side: the fork controller when the driver wants it, None to leave upstream's in
  place. The param is read once, here, so the toggle takes a restart."""
  if not enabled(LONGITUDINAL, params := params or Params()):
    return None
  return MoonpilotLongControl(CP, crawl_release=enabled(CRAWL_RELEASE, params))
