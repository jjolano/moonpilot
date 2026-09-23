"""moonpilot's lateral control: a fork-owned torque controller, chosen at the controlsd seam.

The control law is the family upstream validated on cars — feedforward in lateral acceleration,
PI on the delay-matched error, friction compensation — because that is the part real miles paid
for. What the fork owns is the code and every number in it, so tuning happens here instead of in
an upstream file.

Two mechanisms are deliberately not upstream's:

  - the setpoint is read out of the request buffer at a fractional frame, so a lateralDelay that
    is not a whole number of frames (it moves continuously) does not step the setpoint by a whole
    10 ms as it drifts;
  - the desired jerk is a centered difference at the same fractional resolution as the setpoint, one
    0.19 s lookahead ahead of it, rather than at an integer frame index derived from the delay. The
    two agree while the delay is near 0.2 s; only the fractional form holds the lookahead at 0.19 s
    on a car whose estimated delay is larger.

The controller is picked once, at construction, so the toggle needs a restart, and it only runs
on a car whose lateralTuning is torque — angle and curvature cars never reach the seam.
"""

import math
from collections import deque

import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from openpilot.cereal import log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.latcontrol import LatControl

from moonpilot.features import TORQUE_LATERAL, enabled

# Error correction runs in lateral-acceleration space, where the same curvature error means less
# lateral acceleration the slower the car goes; the speed schedule is what puts that back. These
# start as upstream's values, the only ones with miles behind them, and are the fork's to tune.
MOONPILOT_KP_BP = [1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0]  # m/s
MOONPILOT_KP_V = [250.0, 120.0, 65.0, 30.0, 11.5, 5.5, 3.5, 2.0, 0.8]
MOONPILOT_KI = 0.15
MOONPILOT_JERK_CUTOFF_HZ = 1.2  # a 100 Hz plan's own jerk is mostly noise above this
MOONPILOT_JERK_GAIN = 0.3  # how much anticipated jerk counts as error when breaking friction
MOONPILOT_JERK_LOOKAHEAD_T = 0.19  # s ahead of the delayed setpoint the jerk is read at
MOONPILOT_BUFFER_SECONDS = 1.0  # ceiling on the steering delay this can match
MOONPILOT_INTEGRATOR_MIN_SPEED = 5.0  # m/s; below it the angle measurement is too coarse to integrate
# Road steps kick steeringAngleDeg hard enough that the friction feedforward (corr ~0.9 with
# output on a rough-highway route) chases them and weaves; a few Hz still tracks real cornering.
MOONPILOT_MEAS_CUTOFF_HZ = 8.0
MOONPILOT_VERSION = 1000  # logged; a fork band upstream's counter will not reach


class MoonpilotLatControlTorque(LatControl):
  def __init__(self, CP, CI, dt):
    super().__init__(CP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    self.pid = PIDController([MOONPILOT_KP_BP, MOONPILOT_KP_V], MOONPILOT_KI, rate=1 / self.dt)
    self.update_limits()

    # Lateral acceleration requests, newest last: the delay line the setpoint is read out of.
    self.buffer_len = int(MOONPILOT_BUFFER_SECONDS / self.dt)
    self.requests = deque([0.0] * self.buffer_len, maxlen=self.buffer_len)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MOONPILOT_JERK_CUTOFF_HZ), self.dt)
    self.meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MOONPILOT_MEAS_CUTOFF_HZ), self.dt)

  def update_torque_parameters(self, latAccelFactor, latAccelOffset, friction):
    # controlsd calls this whenever torqued publishes a new fit; the limits move with the factor.
    self.torque_params.latAccelFactor = latAccelFactor
    self.torque_params.latAccelOffset = latAccelOffset
    self.torque_params.friction = friction
    self.update_limits()

  def update_limits(self):
    # The PID works in lateral acceleration, so its limits are full steer torque mapped back.
    self.pid.set_limits(self.lateral_accel_from_torque(self.steer_max, self.torque_params), self.lateral_accel_from_torque(-self.steer_max, self.torque_params))

  def reset(self):
    # controlsd calls this while lateral is inactive. The integrator and the jerk filter are
    # stale the moment someone else is steering; the delay line is not — it keeps tracking what
    # was asked for, so re-engaging mid-curve starts with real history.
    super().reset()
    self.pid.reset()
    self.jerk_filter.x = 0.0
    # The measurement filter is not reset: it keeps tracking the wheel while inactive, the same
    # way the delay line does, so re-engaging does not start from a stale zero.

  def _setpoint(self, lat_delay: float) -> float:
    # The request that should be showing up in the measurement now. lateralDelay is continuous,
    # so interpolate between frames instead of rounding to one.
    frames = float(np.clip(lat_delay / self.dt, 0.0, self.buffer_len - 1))
    whole = int(frames)
    older = min(whole + 1, self.buffer_len - 1)
    newer_accel = self.requests[-1 - whole]
    older_accel = self.requests[-1 - older]
    return newer_accel + (frames - whole) * (older_accel - newer_accel)

  def _desired_jerk(self, lat_delay: float) -> float:
    # Read one lookahead ahead of the setpoint, not at the newest request: with the buffer read at a
    # fractional frame this is upstream's centered difference exactly, and it keeps the anticipation
    # at the lookahead on a car whose estimated delay is large (lagd publishes up to 0.65 s).
    delay = max(lat_delay - MOONPILOT_JERK_LOOKAHEAD_T, self.dt)
    return self.jerk_filter.update((self._setpoint(delay - self.dt) - self._setpoint(delay + self.dt)) / (2 * self.dt))

  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    torque_log = log.ControlsState.LateralTorqueState.new_message()
    torque_log.version = MOONPILOT_VERSION

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    # Raw is what the road put on the sensor; the filter is what feedback (P and friction) sees.
    measured_lat_accel_raw = measured_curvature * CS.vEgo**2
    measured_lat_accel = self.meas_filter.update(measured_lat_accel_raw)
    desired_lat_accel = desired_curvature * CS.vEgo**2
    # Fed whether or not lateral is active, so the delay line is warm on engage.
    self.requests.append(desired_lat_accel)

    # Feedback follows the delayed request while tightening. Unwind can start at the jerk-lookahead
    # sample, but a sign change only releases the old turn: opposite feedback still waits for the plant.
    delayed_setpoint = self._setpoint(lat_delay)
    lookahead_setpoint = self._setpoint(max(lat_delay - MOONPILOT_JERK_LOOKAHEAD_T, 0.0))
    if delayed_setpoint * lookahead_setpoint < 0:
      setpoint = 0.0
    elif abs(lookahead_setpoint) < abs(delayed_setpoint):
      setpoint = lookahead_setpoint
    else:
      setpoint = delayed_setpoint
    error = setpoint - measured_lat_accel
    desired_jerk = self._desired_jerk(lat_delay)

    # Friction only applies outside the steering's own deadzone, in lateral acceleration units.
    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    deadzone = curvature_deadzone * CS.vEgo**2

    # Feedforward: what the plan asks for, less the lateral acceleration the road's own bank
    # already supplies, less the learned bias in that roll estimate, plus enough to move the
    # steering against its friction.
    feedforward = desired_lat_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY - self.torque_params.latAccelOffset
    feedforward += get_friction(error + MOONPILOT_JERK_GAIN * desired_jerk, deadzone, FRICTION_THRESHOLD, self.torque_params)

    if not active:
      output_torque = 0.0
      torque_log.active = False
    else:
      # Integrating is only honest while the output is ours and the measurement is usable.
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < MOONPILOT_INTEGRATOR_MIN_SPEED
      command = self.pid.update(error, speed=CS.vEgo, feedforward=feedforward, freeze_integrator=freeze_integrator)
      # Clipped here as well as by the PID's limits: a car's torque map need not be monotonic.
      output_torque = float(np.clip(self.torque_from_lateral_accel(command, self.torque_params), -self.steer_max, self.steer_max))

      torque_log.active = True
      torque_log.p = float(self.pid.p)
      torque_log.i = float(self.pid.i)
      torque_log.d = float(self.pid.d)
      torque_log.f = float(self.pid.f)
      torque_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))

    # Logged in both branches, unlike upstream's: the plotjuggler and jotpluggler torque-controller
    # layouts then show tracking across a disengagement instead of freezing at the last engaged frame.
    torque_log.error = float(error)
    torque_log.output = float(-output_torque)
    torque_log.actualLateralAccel = float(measured_lat_accel)
    torque_log.desiredLateralAccel = float(setpoint)
    torque_log.desiredLateralJerk = float(desired_jerk)

    # Left is positive in this convention, the opposite sign of the torque the car is given.
    return -output_torque, 0.0, torque_log


def moonpilot_latcontrol(CP, CI, dt, params: Params | None = None) -> LatControl | None:
  """The seam's fork side: the fork controller when the driver wants it, None to leave upstream's
  in place. The param is read once, here, so the toggle takes a restart."""
  if not enabled(TORQUE_LATERAL, params or Params()):
    return None
  return MoonpilotLatControlTorque(CP, CI, dt)
