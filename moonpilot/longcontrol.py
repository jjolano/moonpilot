"""moonpilot's acceleration controller: the fork's own answer to "how do we track that?".

Upstream's LongControl is a PI loop on the accel error with a feedforward of the plan's target, and
that is the right family — the car is the plant, and the only thing that makes it work is the gain
schedule the car ships. The fork keeps the shape and the car's own `longitudinalTuning.kiV` table
(the same reason `moonpilot/latcontrol.py` keeps `lateralTuning.torque`) and owns the rest:

  - the commanded accel is rate limited, so a step in the plan is not a step at the actuator — except
    at the handover out of the stopping state, whose held value is a brake command rather than one
    being tracked, so the baseline resets instead;
  - the integrator freezes at standstill and while the driver is on the pedals, where neither the
    error nor the measurement means anything;
  - the stopping ramp is a rate limit toward `CP.stopAccel` rather than a fixed step per frame.

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

from moonpilot.features import LONGITUDINAL, enabled

LongCtrlState = car.CarControl.Actuators.LongControlState

# Starting points, all fork-owned. Tune against logs.
MOONPILOT_ACCEL_JERK = 8.0  # m/s^3 limit on the commanded accel, so a plan step is not an actuator step
MOONPILOT_STOPPING_JERK = 1.0  # m/s^3 ramp toward CP.stopAccel while stopping
MOONPILOT_STANDSTILL_SPEED = 0.1  # m/s; below it aEgo is noise, so the integrator freezes


class MoonpilotLongControl:
  """Same public surface as upstream's controller — `long_control_state`, `reset`, `update` with the
  same positional contract — because controlsd's seam line calls it and reads the state."""

  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.last_output_accel = 0.0
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV), rate=1 / DT_CTRL)

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits) -> float:
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit, self.pid.pos_limit = accel_limits

    previous_state = self.long_control_state
    self.long_control_state = long_control_state_trans(active, self.long_control_state, should_stop, CS.brakePressed, CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      self.last_output_accel = 0.0
      return 0.0

    if previous_state == LongCtrlState.stopping and self.long_control_state == LongCtrlState.pid:
      # Leaving the stop hold, and the one place the fork's rate limit is the wrong shape. Everything
      # the stopping branch produces is a brake command, so carrying `last_output_accel` into the pid
      # branch commands braking while the plan is already asking for motion — at the floor, -2.0 m/s^2
      # walked off at MOONPILOT_ACCEL_JERK * DT_CTRL, 25 frames and 240 ms. Upstream has no rate limit
      # here at all; it steps straight to the feedforward.
      #
      # Both of `long_control_state_trans`'s pins — `brakePressed` and `cruiseState.standstill` — hold
      # the state in stopping outright, so the hold sits at the floor for as long as either is set and
      # the release pays the full 240 ms the moment it clears. That is a car waiting at a light with
      # its own ACC holding the standstill bit, which is exactly when the driver is watching for it to
      # move: measured from the `stopping -> pid` edge, pre-fix commands -1.92, -1.84, -1.76, -1.68
      # where the plan asks +0.15, and turns positive 240 ms later; with the reset it commands +0.08,
      # already on the plan's side, on the first frame. Same on the free path from a 10 s dwell up.
      # The depth does not change the rule: a half-ramped hold is still a brake command, so the
      # mid-ramp window is the same defect, just a smaller one, and it clears too — measured from the
      # same edge, 100 ms at a 10 s dwell and 230 ms at 11 s.
      #
      # The cost is one MOONPILOT_ACCEL_JERK frame at each edge of the plan's creep dither, which was
      # where the state flipped stopping<->pid ~13 times a second behind a lead creeping below ~0.33 m/s
      # (peak jerk 22.0 m/s^3 against 8.0 without the reset, a 3 mm/s velocity ripple). That dither was
      # the planner's and is now gated there — `MoonpilotLongitudinalPlanner` does not declare
      # should-stop while it is trailing a moving lead — so on that regime this edge no longer arises.
      # The reset is still what the saturated hold needs.
      self.last_output_accel = 0.0

    if self.long_control_state == LongCtrlState.stopping:
      output_accel = max(min(self.last_output_accel, 0.0) - MOONPILOT_STOPPING_JERK * DT_CTRL, self.CP.stopAccel)
      self.reset()
    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      # Standing still, or with the driver on a pedal, the integrator would wind up on error that
      # is a measurement artifact or a command the driver is overriding.
      freeze = CS.vEgo < MOONPILOT_STANDSTILL_SPEED or CS.brakePressed or CS.gasPressed
      output_accel = self.pid.update(error, speed=CS.vEgo, feedforward=a_target, freeze_integrator=freeze)

    output_accel = float(
      np.clip(output_accel, self.last_output_accel - MOONPILOT_ACCEL_JERK * DT_CTRL, self.last_output_accel + MOONPILOT_ACCEL_JERK * DT_CTRL)
    )
    self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
    return self.last_output_accel


def moonpilot_longcontrol(CP, params: Params | None = None) -> MoonpilotLongControl | None:
  """The seam's fork side: the fork controller when the driver wants it, None to leave upstream's in
  place. The param is read once, here, so the toggle takes a restart."""
  if not enabled(LONGITUDINAL, params or Params()):
    return None
  return MoonpilotLongControl(CP)
