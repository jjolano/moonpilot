"""The standing error in the pitch every grade feature reads.

`carControl.orientationNED[1]` is locationd's calibrated pose, and it is the fork's only grade
input: `coast_accel` turns it into the no-throttle acceleration that `cruise_accel`'s coast band
tapers toward and that `cruise_cap` uses as its no-throttle ceiling. Nothing in that chain is
closed loop, so whatever standing offset the signal carries is a phantom hill the planner believes
in forever.

That offset is not hypothetical. On the latest 23-segment route (136,440 `carControl` frames) the
reported pitch is negative on **0.23 %** of frames, median **+2.27 deg**; removing its correlation
with `carState.aEgo` (r = 0.145, so not body pitch under acceleration) leaves the median at
+2.29 deg. The drive ends 71 m from where it started — a GPS fix at both ends — so its true net
elevation change is ~0, while integrating `sin(pitch) * ds` over the same 7.93 km claims +298 m of
climb. That is `asin(298 / 7927)` = 2.16 deg of bias, which `coast_accel`'s road term reads as
**-0.21 m/s^2 of permanent uphill**. Its cost is the descent branch: `_coast_applies` needs a
negative pitch, so on that device a real descent has to exceed ~6.3 deg before the band can open at
all, and Toyota's own `sin(min(pitch, 0)) * g` (`opendbc/car/toyota/carcontroller.py`) is clamped to
zero for the same reason.

So this learns the offset rather than the disturbance, which is the cheaper half and the one
everything else is downstream of. Four things about it:

- **It is the mean, and the assumption is that routes come back down.** Real driving averages to
  level, so a long-window average of the pitch is the offset. ``ponytail: a windowed mean assumes
  the window is not one long climb; a route that genuinely ascends for MOONPILOT_PITCH_RC biases it,
  bounded by MOONPILOT_PITCH_OFFSET_MAX. Pair it with an altitude source if that bound ever binds.``
- **The bound is sized above the bias, and what it bounds is the damage rather than the gate.**
  `MOONPILOT_PITCH_OFFSET_MAX` is 3 deg, comfortably over the 2.16 deg measured here, because a
  clamp that binds on a real device saturates silently and under-corrects forever —
  `moonpilot/tests/test_pitch.py` pins that ordering, since chasing a tighter number is exactly the
  wrong fix. The named ceiling on the other side: a *wrongly* saturated offset is
  `sin(MOONPILOT_PITCH_OFFSET_MAX) * 5.65` = 0.296 m/s^2 of road term, which is enough to open the
  coast band on level road at a `MOONPILOT_COAST_GRADE_MIN` below that, and the overspeed it can
  then permit is `MOONPILOT_COAST_BAND`'s — bounded, not impossible. Reaching that state takes a
  window that is one long climb, which is the same assumption as the bullet above.
- **It learns whenever the car is driving, not only while engaged.** The pose is published onroad
  regardless of who is steering, and the bias is a property of the device and the car rather than of
  the controller — which is the whole reason this one is measurable where
  `moonpilot/latency.py`'s estimate is not: the same corpus holds ~2.4 min of engaged longitudinal
  control against 136k pose frames.
- **No trusted estimate is exactly zero**, so an uncalibrated device is the planner this fork had
  before this existed. It persists in `MoonpilotPitchOffset` on the fork planner's own cadence; a
  learned value is not a behavior the driver flips, so there is no row in the registry.
"""

import math

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

MOONPILOT_PITCH_OFFSET_KEY = "MoonpilotPitchOffset"
# rad; 3 deg, over the 2.16 deg measured here so the clamp does not bind on a real device and
# saturate silently. A wrongly saturated offset is worth 0.296 m/s^2 of road term — see the header.
MOONPILOT_PITCH_OFFSET_MAX = math.radians(3.0)
MOONPILOT_PITCH_RC = 600.0  # s of valid driving; long enough that one hill is not the estimate
MOONPILOT_PITCH_MIN_SPEED = 5.0  # m/s; a parked or creeping car is a driveway, not a road
MOONPILOT_PITCH_SANITY = math.radians(20.0)  # rad; past this the pose is broken, not steep
MOONPILOT_PITCH_MIN_SAMPLES = 6000  # frames, 5 min at DT_MDL, before the offset is applied
MOONPILOT_PITCH_PERSIST_EVERY = 1200  # frames, 60 s at DT_MDL


class PitchOffsetEstimator:
  """The standing offset in the reported pitch, from its long-window mean while driving."""

  def __init__(self, dt: float = DT_MDL):
    self.filter = FirstOrderFilter(0.0, MOONPILOT_PITCH_RC, float(dt))
    self.samples = 0

  def seed(self, offset: float, samples: int) -> None:
    """Resume the last drive's offset, clamped, so the next boot corrects from its first frame."""
    self.filter.x = min(max(float(offset), -MOONPILOT_PITCH_OFFSET_MAX), MOONPILOT_PITCH_OFFSET_MAX)
    self.samples = int(samples)

  @property
  def status(self) -> str:
    return "estimated" if self.samples >= MOONPILOT_PITCH_MIN_SAMPLES else "measuring"

  @property
  def estimate(self) -> float:
    return float(self.filter.x)

  def applied(self) -> float:
    """What to subtract from the reported pitch. Zero until trusted, and bounded after."""
    if self.status != "estimated":
      return 0.0
    return min(max(self.estimate, -MOONPILOT_PITCH_OFFSET_MAX), MOONPILOT_PITCH_OFFSET_MAX)

  def update(self, pitch: float, valid: bool) -> None:
    """One pose sample. An invalid or implausible frame teaches nothing and is not counted."""
    if not valid or not math.isfinite(pitch) or abs(pitch) > MOONPILOT_PITCH_SANITY:
      return
    self.filter.update(float(pitch))
    self.samples += 1
