"""moonpilot's lateral control: a fork-owned torque controller, chosen at the controlsd seam.

The control law is the family upstream validated on cars — feedforward in lateral acceleration,
PI on the delay-matched error, friction compensation — because that is the part real miles paid
for. What the fork owns is the code and every number in it, so tuning happens here instead of in
an upstream file.

Seven mechanisms differ from upstream: fractional delay interpolation, fractional jerk lookahead, continuous friction deadband,
an 8 Hz steering-angle filter, output hysteresis, plan-rate damping and a rising friction-compensation ramp.
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

from moonpilot.features import STEER_DAMPING, STEER_FRICTION_RAMP, STEER_HYSTERESIS, TORQUE_LATERAL, enabled, value

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
MOONPILOT_FRICTION_DEADBAND = 0.05  # m/s^2; keep near-zero noise out of friction compensation
# MoonpilotSteerFrictionRamp: friction compensation may rise no faster than this, m/s^2 per second; it may always
# fall as fast as the error does. The term is what breaks the wheel loose, so letting it arrive in ~0.1 s stores the
# torque that then overshoots; a ramp gives the wheel time to follow before the next increment. It is also the one term
# that turns a road bump's short error spike, or the error flip after a snap, into a full-size torque step at once.
MOONPILOT_FRICTION_RISE = 0.3
MOONPILOT_VERSION = 1001  # logged; a fork band upstream's counter will not reach
MOONPILOT_VERSION_HYSTERESIS = 1002  # the same law with output hysteresis on
# Where the weave was measured, and where the angle and rate sensors are fine enough to damp (the levels are
# STEER_DAMPING's choices): off below 3 m/s (KP 30 and up), full from 5 to 15 m/s, faded out by 20 m/s, where the
# wheel already matched the driver.
MOONPILOT_DAMPING_SPEEDS = ([3.0, 5.0, 15.0, 20.0], [0.0, 1.0, 1.0, 0.0])


def friction_compensation(error: float, desired_jerk: float, deadzone: float, torque_params) -> float:
  friction_input = error + MOONPILOT_JERK_GAIN * desired_jerk
  quiet_band = max(deadzone, MOONPILOT_FRICTION_DEADBAND)
  friction_input = math.copysign(max(abs(friction_input) - quiet_band, 0.0), friction_input)
  return get_friction(friction_input, 0.0, FRICTION_THRESHOLD - MOONPILOT_FRICTION_DEADBAND, torque_params)


class MoonpilotLatControlTorque(LatControl):
  def __init__(self, CP, CI, dt, hysteresis: float = 0.0, damping_t: float = 0.0, friction_rise: float = 0.0):
    super().__init__(CP, CI, dt)
    self.torque_params = CP.lateralTuning.torque.as_builder()
    self.torque_from_lateral_accel = CI.torque_from_lateral_accel()
    self.lateral_accel_from_torque = CI.lateral_accel_from_torque()
    self.steering_angle_deadzone_deg = self.torque_params.steeringAngleDeadzoneDeg

    # D's gain is KP times the derivative time, at every breakpoint of either schedule.
    kd_bp = sorted(set(MOONPILOT_KP_BP + MOONPILOT_DAMPING_SPEEDS[0]))
    kd_v = [float(np.interp(v, MOONPILOT_KP_BP, MOONPILOT_KP_V) * damping_t * np.interp(v, *MOONPILOT_DAMPING_SPEEDS)) for v in kd_bp]
    self.pid = PIDController([MOONPILOT_KP_BP, MOONPILOT_KP_V], MOONPILOT_KI, k_d=[kd_bp, kd_v], rate=1 / self.dt)
    self.update_limits()

    # Lateral acceleration requests, newest last: the delay line the setpoint is read out of.
    self.buffer_len = int(MOONPILOT_BUFFER_SECONDS / self.dt)
    self.requests = deque([0.0] * self.buffer_len, maxlen=self.buffer_len)
    self.jerk_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MOONPILOT_JERK_CUTOFF_HZ), self.dt)
    self.meas_filter = FirstOrderFilter(0.0, 1 / (2 * np.pi * MOONPILOT_MEAS_CUTOFF_HZ), self.dt)
    self.hysteresis = hysteresis
    self.held_torque: float | None = None
    self.friction_rise = friction_rise
    self.friction_state = 0.0

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
    self.held_torque = None  # a re-engage starts from the controller's own output, not a stale hold
    self.friction_state = 0.0
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

  def _hold(self, torque: float) -> float:
    if self.held_torque is None:
      self.held_torque = torque
    elif torque > self.held_torque + self.hysteresis:
      self.held_torque = torque - self.hysteresis
    elif torque < self.held_torque - self.hysteresis:
      self.held_torque = torque + self.hysteresis
    return self.held_torque

  def _desired_jerk(self, lookahead_delay: float) -> float:
    delay = max(lookahead_delay, self.dt)
    return self.jerk_filter.update((self._setpoint(delay - self.dt) - self._setpoint(delay + self.dt)) / (2 * self.dt))


  def update(self, active, CS, VM, params, steer_limited_by_safety, desired_curvature, curvature_limited, lat_delay):
    torque_log = log.ControlsState.LateralTorqueState.new_message()
    torque_log.version = MOONPILOT_VERSION_HYSTERESIS if self.hysteresis else MOONPILOT_VERSION

    measured_curvature = -VM.calc_curvature(math.radians(CS.steeringAngleDeg - params.angleOffsetDeg), CS.vEgo, params.roll)
    # Raw is what the road put on the sensor; the filter is what feedback (P and friction) sees.
    measured_lat_accel_raw = measured_curvature * CS.vEgo**2
    measured_lat_accel = self.meas_filter.update(measured_lat_accel_raw)
    desired_lat_accel = desired_curvature * CS.vEgo**2
    self.requests.append(desired_lat_accel)

    # Feedback follows the delayed request while tightening. Unwind can start at the jerk-lookahead
    # sample, but a sign change only releases the old turn: opposite feedback still waits for the plant.
    delayed_setpoint = self._setpoint(lat_delay)
    lookahead_delay = max(lat_delay - MOONPILOT_JERK_LOOKAHEAD_T, 0.0)
    lookahead_setpoint = self._setpoint(lookahead_delay)
    if delayed_setpoint * lookahead_setpoint < 0:
      setpoint = 0.0
    elif abs(lookahead_setpoint) < abs(delayed_setpoint):
      setpoint = lookahead_setpoint
    else:
      setpoint = delayed_setpoint
    error = setpoint - measured_lat_accel
    desired_jerk = self._desired_jerk(lookahead_delay)
    # The wheel's own rate from the rate sensor (it trails the angle's derivative by ~20 ms on this car and
    # does not differentiate the angle's 0.1 deg steps), against the plan's rate: zero while the wheel
    # follows a turn the plan asks for, so only a departure from it is damped.
    measured_jerk = -VM.calc_curvature(math.radians(CS.steeringRateDeg), CS.vEgo, 0.0) * CS.vEgo**2
    error_rate = desired_jerk - measured_jerk

    curvature_deadzone = abs(VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0))
    deadzone = curvature_deadzone * CS.vEgo**2

    feedforward = desired_lat_accel - params.roll * ACCELERATION_DUE_TO_GRAVITY - self.torque_params.latAccelOffset
    friction = friction_compensation(error, desired_jerk, deadzone, self.torque_params)
    if self.friction_rise > 0.0 and active:
      step = self.friction_rise * self.dt
      if friction == 0.0:
        self.friction_state = 0.0
      elif friction * self.friction_state > 0.0:
        if abs(friction) <= abs(self.friction_state):
          self.friction_state = friction
        else:
          self.friction_state += math.copysign(min(abs(friction) - abs(self.friction_state), step), friction)
      else:
        self.friction_state = math.copysign(min(abs(friction), step), friction)
    else:
      self.friction_state = friction
    feedforward += self.friction_state

    if not active:
      output_torque = 0.0
      torque_log.active = False
    else:
      freeze_integrator = steer_limited_by_safety or CS.steeringPressed or CS.vEgo < MOONPILOT_INTEGRATOR_MIN_SPEED
      command = self.pid.update(error, error_rate=error_rate, speed=CS.vEgo, feedforward=feedforward, freeze_integrator=freeze_integrator)
      # Clipped here as well as by the PID's limits: a car's torque map need not be monotonic.
      output_torque = float(np.clip(self.torque_from_lateral_accel(command, self.torque_params), -self.steer_max, self.steer_max))

      torque_log.active = True
      torque_log.p = float(self.pid.p)
      torque_log.i = float(self.pid.i)
      torque_log.d = float(self.pid.d)
      torque_log.f = float(self.pid.f)
      torque_log.saturated = bool(self._check_saturation(self.steer_max - abs(output_torque) < 1e-3, CS, steer_limited_by_safety, curvature_limited))
      if self.hysteresis:
        output_torque = self._hold(output_torque)

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
  """The fork controller when the driver wants it, None to leave upstream's in place; read once, so a restart."""
  params = params or Params()
  if not enabled(TORQUE_LATERAL, params):
    return None
  friction_rise = MOONPILOT_FRICTION_RISE if enabled(STEER_FRICTION_RAMP, params) else 0.0
  return MoonpilotLatControlTorque(CP, CI, dt, value(STEER_HYSTERESIS, params), value(STEER_DAMPING, params), friction_rise)
