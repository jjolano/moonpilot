"""The longitudinal planner's brake band: a shallow negative ask goes to the actuator as a coast, a deep
one as one committed press. See `MOONPILOT_BRAKE_BAND_*` below and docs/longitudinal.md."""

import math

# Brake band: a shallow negative ask goes to the actuator as a coast, a deep one as a committed press.
# On this car (RAV4 TSS2, `VSC1S07.BRKON` under openpilot-only control, 14 routes, 21 min above 5 m/s)
# the brakes almost never switch on at a sent command of -0.10 or above (0.1 % per 50 ms at -0.10), switch
# on at 3-18 % per 50 ms between -0.15 and -0.35, and once on drop out 6-8 % per 50 ms at -0.15..-0.20 but
# hold (<= 0.5 %) at -0.25 and deeper. A command hovering in that zone is the flicker: 1.3 lamp blips
# shorter than 1 s per minute. The driver coasts first (92 % of slowing brake presses follow no gas)
# and presses the pedal for over 1 s (626 of 639), the lightest presses averaging -0.35. So: a negative
# ask above `ENTER` is raised to `FLOOR`; one below commits a press that holds at `HOLD` or deeper for
# at least `MIN_PRESS_T` and lets go once the ask has stayed above `RELEASE` for `RELEASE_T` (the policy's
# ask jitters by hundredths a frame), or at once when it turns positive. `FLOOR` is -0.10, not shallower:
# the arrival term's own coast asks about -0.09, and raising it starves the approach into a firm press
# later. Off below `MIN_SPEED`, where stops and crawls need their shallow asks, and in experimental mode,
# whose ask is the model's raw. Behind a slowing lead (`lead_braking`, the test `jerk_limit` reads) the
# press commits from `ENTER_LEAD`: there the onset is the safety margin (`ENTER` alone cost a -3.5 m/s^2
# lead one frame of its -0.5 onset), and `ENTER_LEAD` stays below `RELEASE` so a press cannot end and
# restart on the same ask. The planner bounds the result by `cruise_cap`, so with throttle denied the
# coast accel stays a ceiling. Replayed through the planner (15 routes) and those switching rates,
# re-lights within 2 s go 0.94 -> 0.80 /min; docs/longitudinal.md has the rest. TTC, the stopping floor
# and real braking are deeper than `ENTER` and pass untouched.
MOONPILOT_BRAKE_BAND_FLOOR = -0.10  # m/s^2
MOONPILOT_BRAKE_BAND_ENTER = -0.40  # m/s^2
MOONPILOT_BRAKE_BAND_HOLD = -0.30  # m/s^2
MOONPILOT_BRAKE_BAND_RELEASE = -0.20  # m/s^2
MOONPILOT_BRAKE_BAND_MIN_PRESS_T = 1.0  # s
MOONPILOT_BRAKE_BAND_MIN_SPEED = 5.0  # m/s
MOONPILOT_BRAKE_BAND_ENTER_LEAD = -0.25  # m/s^2
MOONPILOT_BRAKE_BAND_RELEASE_T = 0.5  # s


class BrakeBand:
  """Coast-or-press shaping of the negative ask, so the car's own brake arbitration is not left to
  dither (see `MOONPILOT_BRAKE_BAND_*`). State: whether a press is committed, how long it has held, and how
  long the ask has stayed above the release."""

  def __init__(self):
    self.reset()

  def reset(self):
    self.pressing = False
    self.held = 0.0
    self.above = 0.0

  def update(self, a_cmd: float, v_ego: float, dt: float, urgent: bool = False, floor: float = MOONPILOT_BRAKE_BAND_FLOOR) -> float:
    """`urgent`: a lead is slowing, so the press commits from `ENTER_LEAD` instead of `ENTER`. `floor`: the
    lamp-free coast; every threshold moves with it (`moonpilot/engine_coast.py`)."""
    shift = floor - MOONPILOT_BRAKE_BAND_FLOOR
    if v_ego < MOONPILOT_BRAKE_BAND_MIN_SPEED or not math.isfinite(a_cmd):
      self.reset()
      return a_cmd
    if self.pressing:
      self.held += dt
      # The policy's ask is not rate-limited and jitters by hundredths a frame: one frame over the release
      # is not the need ending, and letting go on it re-lit the lamps 0.2 s later on route 0000042b.
      self.above = self.above + dt if a_cmd > MOONPILOT_BRAKE_BAND_RELEASE + shift else 0.0
      if a_cmd >= 0.0 or (self.held >= MOONPILOT_BRAKE_BAND_MIN_PRESS_T and self.above >= MOONPILOT_BRAKE_BAND_RELEASE_T - 1e-9):
        self.reset()
      else:
        return min(a_cmd, MOONPILOT_BRAKE_BAND_HOLD + shift)
    if a_cmd < (MOONPILOT_BRAKE_BAND_ENTER_LEAD if urgent else MOONPILOT_BRAKE_BAND_ENTER) + shift:
      self.pressing = True
      return min(a_cmd, MOONPILOT_BRAKE_BAND_HOLD + shift)
    return max(a_cmd, floor) if a_cmd < 0.0 else a_cmd

