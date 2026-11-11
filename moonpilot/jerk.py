"""Vehicle-response calibration for longitudinal comfort jerk.

This observes positive command ramps against delayed ``carState.aEgo`` and only tightens the
comfort-side acceleration ramp when the vehicle delivers more jerk than requested. It never changes
the braking-side or emergency jerk ramp: those values are part of the stopping-distance evidence.

No trustworthy estimate means scale 1.0, so the uncalibrated planner remains the fallback.
"""

import math
from collections import deque
from typing import Any

import numpy as np
from opendbc.car.interfaces import ACCEL_MIN


from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

MOONPILOT_LONG_JERK_SCALE_KEY = "MoonpilotLongJerkScale"
MOONPILOT_LONG_JERK_SCALE_MIN = 0.5  # never make the comfort ramp less than half its validated value
MOONPILOT_LONG_JERK_SCALE_MAX = 1.0  # calibration can tighten only; it cannot raise a limit
MOONPILOT_LONG_JERK_RC = 20.0  # s; a trusted estimate moves slowly against noisy aEgo derivatives
MOONPILOT_LONG_JERK_WINDOW = 0.15  # s; slope span, longer than one noisy 20 Hz difference
MOONPILOT_LONG_JERK_EXCITATION = 0.5  # m/s^3; both command and realized slopes must clear the noise floor
MOONPILOT_LONG_JERK_MIN_SAMPLES = 100  # paired ramps before the scale is applied
MOONPILOT_LONG_JERK_COMFORT_ACCEL = -2.0  # m/s^2; do not learn across the emergency ramp
MOONPILOT_LONG_JERK_MAX_RATIO = 4.0  # reject an acceleration-sensor spike rather than over-tighten forever


MOONPILOT_JERK_LAUNCH_V = [1.3, 2.1]  # m/s^3

MOONPILOT_JERK_UP = 1.5  # m/s^3; a command at or below zero rises this fast: a brake unwinds, and a ramp leaves zero
MOONPILOT_JERK_LAUNCH_SPEED = 2.5  # m/s
MOONPILOT_JERK_DOWN = 2.0  # m/s^3 at a -2.0 ask; 6.05 m end gap, no contact at 90-144 kph
MOONPILOT_JERK_EMERGENCY = 10.0  # m/s^3, reached at ACCEL_MIN
MOONPILOT_JERK_ONSET = 1.0  # m/s^3 above JERK_ONSET_A; this and below: docs/longitudinal.md
MOONPILOT_JERK_ONSET_A = -1.0  # m/s^2
# m/s^3; a braking ask past MOONPILOT_JERK_EASE_HOLD_A eases no faster than this while any lead brakes — a driver does not let
# off while the car ahead is still braking. The headway-scheduled braking match lets closing build behind a braking lead, and
# the arrival's closing^2 term then follows every wobble of the lead's estimate: through stop-shape's 57 decels it eased and
# re-deepened (a pump) 79 times against the flat match's 51; at 0.5 it is 52, with the stops' peak still 36% in, and 0.3
# over-brakes, deepening the slowdowns' last third. `python -m moonpilot.lab stop-shape` flies all four.
MOONPILOT_JERK_EASE_LEAD_BRAKING = 0.5
MOONPILOT_JERK_EASE_HOLD_A = -0.3  # m/s^2
MOONPILOT_JERK_ONSET_LEAD_A = -0.2  # m/s^2; a lead slowing past this keeps JERK_DOWN
MOONPILOT_BRAKE_EXIT_A = -0.5  # m/s^2; braking past this holds the positive ask, as curve braking does
MOONPILOT_BRAKE_EXIT_JERK = 0.5  # m/s^3; its rebuild with the wheel straight, and every positive ramp's with no lead close ahead
# A positive command follows the slowest-accelerating lead within the farther of the gap and headway limits.
MOONPILOT_JERK_RAMP_A_BP = [0.0, 1.25]  # m/s^2 of the lead's accel
MOONPILOT_JERK_RAMP_V = [0.25, MOONPILOT_BRAKE_EXIT_JERK]  # m/s^3
MOONPILOT_JERK_RAMP_GAP = 25.0  # m
MOONPILOT_JERK_RAMP_HEADWAY = 3.0  # s
# The launch ramp is faster, not a floor: a floor's first step above `should_stop`'s 0.1 would drop a parked car's brake hold.
MOONPILOT_JERK_LAUNCH_A_BP = [0.0, 1.5]  # m/s^2 of the close lead's launch
def lead_braking(lead_states) -> bool:
  return any(a_lead < MOONPILOT_JERK_ONSET_LEAD_A for _, _, _, a_lead, *_ in lead_states)


def _close_lead_accel(lead_states, v_ego) -> float | None:
  """The slowest-accelerating lead within `MOONPILOT_JERK_RAMP_GAP` or `MOONPILOT_JERK_RAMP_HEADWAY` of `v_ego`, whichever
  is farther; None with none there, so each band supplies its own lead-free end."""
  near = max(MOONPILOT_JERK_RAMP_GAP, MOONPILOT_JERK_RAMP_HEADWAY * v_ego)
  return min((a_lead for _, gap, _, a_lead, *_ in lead_states if gap < near), default=None)


def ramp_jerk(lead_states, v_ego) -> float:
  """The positive ramp's jerk at speed: `MOONPILOT_JERK_RAMP_V` on the slowest-accelerating close lead, the lead-free ramp
  with none there."""
  a = _close_lead_accel(lead_states, v_ego)
  return float(np.interp(MOONPILOT_JERK_RAMP_A_BP[-1] if a is None else a, MOONPILOT_JERK_RAMP_A_BP, MOONPILOT_JERK_RAMP_V))


def launch_ramp(lead_states, v_ego) -> float:
  """The launch ramp follows the same close-lead estimate as `ramp_jerk`."""
  a = _close_lead_accel(lead_states, v_ego)
  if a is None:
    return MOONPILOT_JERK_LAUNCH_V[-1]
  return float(np.interp(max(a, 0.0), MOONPILOT_JERK_LAUNCH_A_BP, MOONPILOT_JERK_LAUNCH_V))


def positive_ramp(lead_states, v_ego) -> float:
  """The ramp a command above zero may build at: the launch band's at a standstill, `ramp_jerk`'s by twice
  `MOONPILOT_JERK_LAUNCH_SPEED`, blended across it."""
  w = float(np.clip(1.0 - v_ego / (2.0 * MOONPILOT_JERK_LAUNCH_SPEED), 0.0, 1.0))
  return launch_ramp(lead_states, v_ego) * w + ramp_jerk(lead_states, v_ego) * (1.0 - w)


def jerk_limit(a_cmd, a_prev, dt, v_ego, comfort_scale=1.0, lead_braking=False, ramp=MOONPILOT_JERK_RAMP_V[-1]) -> float:
  """Rate limit, asymmetric and urgency-scaled. A command above zero rises at `ramp` (`ramp_jerk`), at or below it at
  `MOONPILOT_JERK_UP`, so letting off a brake is never the slow part.

  """
  up = (MOONPILOT_JERK_UP if a_prev <= 0.0 else ramp) * comfort_scale
  # The gentle onset is off at a crawl and while a lead brakes, where a shallow ask is urgent.
  onset = float(np.interp(v_ego, [MOONPILOT_JERK_LAUNCH_SPEED, 2 * MOONPILOT_JERK_LAUNCH_SPEED], [MOONPILOT_JERK_DOWN, MOONPILOT_JERK_ONSET]))
  onset = MOONPILOT_JERK_DOWN if lead_braking else onset
  down = float(np.interp(a_cmd, [ACCEL_MIN, MOONPILOT_LONG_JERK_COMFORT_ACCEL, MOONPILOT_JERK_ONSET_A], [MOONPILOT_JERK_EMERGENCY, MOONPILOT_JERK_DOWN, onset]))
  if lead_braking and a_prev < MOONPILOT_JERK_EASE_HOLD_A:
    up = min(up, MOONPILOT_JERK_EASE_LEAD_BRAKING)
  return float(np.clip(a_cmd, a_prev - down * dt, a_prev + up * dt))
class LongitudinalComfortJerkEstimator:
  """Learn a command-to-road jerk ratio from valid positive acceleration ramps."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = float(dt)
    self.span_frames = max(2, int(round(MOONPILOT_LONG_JERK_WINDOW / self.dt)))
    self.history_len = int(round(0.8 / self.dt)) + self.span_frames + 2
    self.commands: deque[float] = deque(maxlen=self.history_len)
    self.actuals: deque[float] = deque(maxlen=self.history_len)
    self.comfort: deque[bool] = deque(maxlen=self.history_len)
    self.filter = FirstOrderFilter(1.0, MOONPILOT_LONG_JERK_RC, self.dt)
    self.samples = 0

  def seed(self, scale: float, samples: int) -> None:
    self.filter.x = min(max(float(scale), MOONPILOT_LONG_JERK_SCALE_MIN), MOONPILOT_LONG_JERK_SCALE_MAX)
    self.filter.initialized = True
    self.samples = int(samples)

  def restore(self, params: Any) -> None:
    scale = params.get(MOONPILOT_LONG_JERK_SCALE_KEY, return_default=True)
    if scale < 1.0:
      self.seed(scale, MOONPILOT_LONG_JERK_MIN_SAMPLES)

  def persist(self, params: Any) -> None:
    if self.status == "estimated":
      params.put(MOONPILOT_LONG_JERK_SCALE_KEY, round(self.applied(), 3))


  @property
  def status(self) -> str:
    return "estimated" if self.samples >= MOONPILOT_LONG_JERK_MIN_SAMPLES else "measuring"

  @property
  def estimate(self) -> float:
    return float(self.filter.x)

  def applied(self) -> float:
    """Return only a trusted tighten-only comfort scale."""
    if self.status != "estimated":
      return MOONPILOT_LONG_JERK_SCALE_MAX
    return min(max(self.estimate, MOONPILOT_LONG_JERK_SCALE_MIN), MOONPILOT_LONG_JERK_SCALE_MAX)

  def update(self, command: float, actual: float, delay: float, valid: bool) -> None:
    """Pair a delayed command slope with the delivered acceleration slope.

    Invalid stretches clear the history so a driver-controlled or braking event cannot be paired
    across a disengagement. Only positive command ramps inside the comfort region are eligible.
    """
    command = float(command)
    actual = float(actual)
    if not valid or not (math.isfinite(command) and math.isfinite(actual)):
      self.commands.clear()
      self.actuals.clear()
      self.comfort.clear()
      return

    self.commands.append(command)
    self.actuals.append(actual)
    self.comfort.append(command > MOONPILOT_LONG_JERK_COMFORT_ACCEL)

    delay_frames = max(1, int(round(max(float(delay), 0.0) / self.dt)))
    end = len(self.commands) - 1 - delay_frames
    start = end - self.span_frames
    if start < 0 or not (self.comfort[start] and self.comfort[end]):
      return

    span = self.span_frames * self.dt
    command_jerk = (self.commands[end] - self.commands[start]) / span
    actual_jerk = (self.actuals[-1] - self.actuals[-1 - self.span_frames]) / span
    if command_jerk < MOONPILOT_LONG_JERK_EXCITATION or actual_jerk < MOONPILOT_LONG_JERK_EXCITATION:
      return
    ratio = actual_jerk / command_jerk
    if not math.isfinite(ratio):
      return
    ratio = min(max(ratio, 1.0 / MOONPILOT_LONG_JERK_MAX_RATIO), MOONPILOT_LONG_JERK_MAX_RATIO)
    self.filter.update(1.0 / ratio)
    self.samples += 1
