"""Ride through a transient localization-validity dropout, and only a transient one.

Upstream raises `locationdTemporaryError` the moment `deviceMotion.inputsOK` reads false, and that
event is `noEntry` + `softDisable`. Measured on route 000003f8 (65 min, commit 954f67d), one such
read ended the drive:

  t=343066.296  modeld logs its only error of the route, `frames out of sync!` -- the main and extra
                camera streams' SOF timestamps 50 ms apart, past modeld's 10 ms threshold
  t=343066.447  exactly one `cameraOdometry` message (frameId 72338) published with the capnp
                validity bit false: 1 invalid of 1199 in that segment
  t=343066.456  locationd's next 20 Hz tick computes `inputs_valid = sm.all_valid() and
                critical_service_inputs_valid` -> False, so it publishes `inputsOK=False`
  ...           for exactly ONE frame, 59 ms. 78,029 deviceMotion samples over the route, and this is
                the only non-startup false one (the 19 others are the 0.9 s before the filter
                initializes, not engaged)
  t=343066.466  soft-disable; the driver disengaged 1.9 s later

So a single 59 ms validity hole -- one bad camera frame inside a camera-stream desync -- is enough to
end a drive. The gate is a real safety input and stays: this rides through the transient, and a
dropout that outlives `MOONPILOT_LOCATION_RIDE_THROUGH_S` raises exactly as upstream does.

The window is 0.3 s. The event measured 59 ms, and `deviceMotion` publishes at 20 Hz, so one bad
sample stays visible to selfdrived for up to 50 ms after it was taken -- 0.3 s is more than five
times the observed event. Upstream is already tolerant at a coarser scale than this:
`locationd.INPUT_INVALID_LIMIT` is 2.0 s of bad sensor input ignored before locationd counts it at
all. A localization loss that is real lasts far longer than 0.3 s, so the gate still closes inside a
third of a second; the cost is bounded entirely to the transient case this module exists for.
"""

import time

# Seconds a `deviceMotion.inputsOK` false run may last before `locationdTemporaryError` is raised.
# See the module docstring for the measurement this is sized against.
MOONPILOT_LOCATION_RIDE_THROUGH_S = 0.3

# When `inputsOK` first read false, and since when it last read true. Latch state rather than a
# counter, because the publisher's rate is not the rate selfdrived runs at and a frame count would
# silently mean different durations on different cars.
_bad_since = None
_good_since = None


def reset() -> None:
  """Forget the latch: the next false read opens a new window. For a state machine restart."""
  global _bad_since, _good_since
  _bad_since = None
  _good_since = None


def moonpilot_locationd_ok(inputs_ok: bool, now: float | None = None) -> bool:
  """Whether `locationdTemporaryError` should stay unraised for this frame.

  Upstream's own predicate with a bounded transient ride-through, so the seam's value is
  `inputs_ok` itself whenever the ride-through does not apply -- this widens nothing except the
  window, and is upstream's answer verbatim on every other frame.

  A false run shorter than the window is ridden through. A run that outlives it returns False and
  upstream raises. Inputs reading true for a full window close the latch, so a later transient is
  ridden through again; a signal that flaps inside the window never accumulates a clean stretch, so
  its false run keeps ageing and the event does eventually fire rather than being starved.
  """
  global _bad_since, _good_since
  now = time.monotonic() if now is None else float(now)

  if inputs_ok:
    if _good_since is None:
      _good_since = now
    if _bad_since is not None and now - _good_since >= MOONPILOT_LOCATION_RIDE_THROUGH_S:
      _bad_since = None
    return True

  _good_since = None
  if _bad_since is None:
    _bad_since = now
  return now - _bad_since < MOONPILOT_LOCATION_RIDE_THROUGH_S
