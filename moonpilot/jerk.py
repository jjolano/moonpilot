"""Vehicle-response calibration for longitudinal comfort jerk.

This observes positive command ramps against delayed ``carState.aEgo`` and only tightens the
comfort-side acceleration ramp when the vehicle delivers more jerk than requested. It never changes
the braking-side or emergency jerk ramp: those values are part of the stopping-distance evidence.

No trustworthy estimate means scale 1.0, so the uncalibrated planner remains the fallback.
"""

import math
from collections import deque


from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

MOONPILOT_LONG_JERK_SCALE_KEY = "MoonpilotLongJerkScale"
MOONPILOT_LONG_JERK_SCALE_MIN = 0.5  # never make the comfort ramp less than half its validated value
MOONPILOT_LONG_JERK_SCALE_MAX = 1.0  # calibration can tighten only; it cannot raise a limit
MOONPILOT_LONG_JERK_RC = 20.0  # s; a trusted estimate moves slowly against noisy aEgo derivatives
MOONPILOT_LONG_JERK_WINDOW = 0.15  # s; slope span, longer than one noisy 20 Hz difference
MOONPILOT_LONG_JERK_EXCITATION = 0.5  # m/s^3; command-side slope must be real, not dither
MOONPILOT_LONG_JERK_MIN_SAMPLES = 100  # paired ramps before the scale is applied
MOONPILOT_LONG_JERK_PERSIST_EVERY = 1200  # frames, 60 s at DT_MDL
MOONPILOT_LONG_JERK_COMFORT_ACCEL = -2.0  # m/s^2; do not learn across the emergency ramp
MOONPILOT_LONG_JERK_MAX_RATIO = 4.0  # reject an acceleration-sensor spike rather than over-tighten forever


class LongitudinalComfortJerkEstimator:
  """Learn a one-sided command-to-road jerk ratio from valid positive acceleration ramps."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = float(dt)
    self.span_frames = max(2, int(round(MOONPILOT_LONG_JERK_WINDOW / self.dt)))
    self.history_len = int(round(0.8 / self.dt)) + self.span_frames + 2
    self.commands: deque[float] = deque(maxlen=self.history_len)
    self.actuals: deque[float] = deque(maxlen=self.history_len)
    self.comfort: deque[bool] = deque(maxlen=self.history_len)
    self.filter = FirstOrderFilter(1.0, MOONPILOT_LONG_JERK_RC, self.dt)
    self.samples = 0
    self.frames = 0

  def seed(self, scale: float, samples: int) -> None:
    """Resume a previously trusted scale without allowing it to raise a limit."""
    self.filter.x = min(max(float(scale), MOONPILOT_LONG_JERK_SCALE_MIN), MOONPILOT_LONG_JERK_SCALE_MAX)
    self.filter.initialized = True
    self.samples = int(samples)

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
    self.frames += 1
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
    if command_jerk < MOONPILOT_LONG_JERK_EXCITATION or actual_jerk <= 0.0:
      return
    ratio = actual_jerk / command_jerk
    if not math.isfinite(ratio) or ratio <= 1.0:
      return

    ratio = min(ratio, MOONPILOT_LONG_JERK_MAX_RATIO)
    self.filter.update(1.0 / ratio)
    self.samples += 1


if __name__ == "__main__":
  # Tiny smoke check for the pure estimator: twice the requested positive jerk tightens to <= 1.
  estimator = LongitudinalComfortJerkEstimator()
  delay_frames = 3
  command = [0.5 + 0.75 * math.sin(2.0 * math.pi * i / 40.0) for i in range(500)]
  actual = [2.0 * command[i - delay_frames] if i >= delay_frames else 0.0 for i in range(500)]
  for cmd, acc in zip(command, actual, strict=True):
    estimator.update(cmd, acc, delay_frames * estimator.dt, True)
  assert estimator.estimate <= 1.0
