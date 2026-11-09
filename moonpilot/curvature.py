"""Plan-timed corrections to the model's curvature request: the look-ahead lead-in, the smoothing lead and the re-centering cap.

The decoded `modelV2.action.desiredCurvature` stays authoritative. modeld already samples the plan at
`lateralDelay + 75 ms`, so a re-sample at the live capture age moves the request by well under
0.01 m/s^2 and is not applied. What this module adds is time-shift, read from the model's path at
the horizon the steering responds on (`capture age + lat_delay`):

- `lead`: the `MoonpilotPathSmooth` filter's own time constant. A first-order filter lags a ramp by
  tau, so adding the planned curvature's change over the next tau (`orientationRate.z / v` at
  `horizon + tau` minus at `horizon`) in full before the filter cancels its lag on a curvature ramp.
- `lookahead`: the turn-in lead-in, a mean of the planned curvature over a window ahead.
- `recenter`: a cap, in m/s^2, on the part of the request beyond the ego lane's own bend (`lane_bend`), applied
  last: the plan may follow the road at any pace but move the car sideways within its lane only gently.

`clip_curvature` downstream still owns the command's jerk, lateral-acceleration, and maximum-curvature
envelope. Capture age and receive age are checked independently. Any stale, future-dated, malformed,
slow or out-of-grid input returns the decoded action exactly; no previous correction is retained.
"""

import functools
import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED

from moonpilot.features import PATH_LOOKAHEAD, PATH_RECENTER, PATH_SMOOTH, choice, enabled

MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the path curvature is noise
MOONPILOT_SMOOTH_LEAD_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the smoothing lead itself may ask for
MOONPILOT_MAX_MODEL_AGE = 2 * DT_MDL  # s; two missed 20 Hz model periods disable the correction
MOONPILOT_SMOOTH_TAUS = (0.0, 0.05, 0.10, 0.15)  # s; first-order time constant per MoonpilotPathSmooth choice, 0 = off
# Look-ahead per MoonpilotPathLookahead choice: (window s, gain, cap m/s^2); None = off.
# The mean planned curvature over [horizon, horizon + window] may only add turn to today's request, never take
# any away, so a turn-in starts early while the apex and the exit stay as today's: replayed on the recorded turns
# of routes 3f8-410 (python -m moonpilot.lab lookahead) the request started 0.00/0.10/0.12 s (light), 0.00/0.20/0.20 s
# (medium) and 0.00/0.35/0.24 s (strong) earlier at 5-10/10-17/>17 m/s with every turn's apex at 100% of today's (p10
# included) and no outward drift at the median; the 5-10 column is after the fade below. Unrestricted, the same
# averages cut the apex to 82-96% and let turns out early -- running wide; that is the pre-guard variant, which this
# checkout can no longer run. The cost is a turn-in that begins inside the lane: 0.4-0.6 / 0.8-1.2 / 1.1-1.5 m median
# open-loop, before the model's own re-planning. None makes highway builds lighter: the controller adds that abruptness.
MOONPILOT_LOOKAHEAD: tuple[tuple[float, float, float] | None, ...] = (None, (1.0, 0.3, 0.8), (1.0, 0.6, 1.5), (1.5, 0.6, 1.5))
# Below about 36 km/h, residential corners and intersections, an early turn-in is a cut toward the inside -- the curb
# on a right turn -- for little lead (up to ~0.1 s of median lead at 5-10 m/s, unfaded). Faded out there: at 5-10 m/s
# the lead-in's median open-loop inside drift fell from 0.2-0.5 m with the fade disabled to 0.00 m, and the 10-17 and
# >17 m/s leads were unchanged.
MOONPILOT_LOOKAHEAD_SPEEDS = ([8.0, 12.0], [0.0, 1.0])  # m/s: none below 8, full from 12
# MoonpilotPathRecenter: the part of the request beyond the ego lane's own bend -- the plan moving the car sideways
# within the lane rather than following the road -- is capped in lateral acceleration on straight-ish lanes. The bend is
# the lane lines' curvature from a quadratic fit over the first MOONPILOT_RECENTER_FIT_X m, weighted by how well each line
# is seen. The cap only ever takes turn away from the request, never adds any or reverses it; it fades out in curves,
# where taking turn away runs wide; and it only touches a move whose recovery side -- the side the car is leaving -- has a
# seen lane line at least MOONPILOT_RECENTER_NEAR_LINE away. A lane line is the only thing that can certify that: a road
# edge is the road's boundary, not the lane's, so it says nothing about a boundary the car might be recovering from.
# `python -m moonpilot.lab recenter`: above 12 m/s on well-seen lanes the request goes beyond the bend by p99 0.40 m/s^2
# against the driver's own 0.34; capped at 0.3 it is lowered by > 0.05 on 0.22% of straight-ish frames and 0.03% in
# curves. It does not act on 0000046b's lurch, whose recovery side (the right lane line) is unseen throughout.
MOONPILOT_RECENTER_LAT_ACCEL = 0.3  # m/s^2
MOONPILOT_RECENTER_SPEEDS = ([8.0, 12.0], [0.0, 1.0])  # m/s: none below 8 (corners, intersections), full from 12
MOONPILOT_RECENTER_PROBS = ([0.4, 0.6], [0.0, 1.0])  # laneLineProbs: a line earns its weight across this band
MOONPILOT_RECENTER_BENDS = ([0.3, 0.6], [1.0, 0.0])  # m/s^2 of the lane's own bend: full on straights, none in curves
MOONPILOT_RECENTER_NEAR_LINE = 1.3  # m from the car's center to the boundary it moves away from: closer is never capped
MOONPILOT_RECENTER_FIT_X = 20.0  # m


class LaneBend(NamedTuple):
  bend: float  # 1/m, positive right like desiredCurvature; nan when no line is seen
  weight: float  # 0..1, what the better-seen line earns
  left: float  # m, the seen left line's offset at the car (negative); nan when not seen
  right: float  # m, the seen right line's offset (positive); nan when not seen


@functools.lru_cache(maxsize=4)
def _curvature_weights(x: tuple[float, ...]) -> np.ndarray:
  """Least-squares weights that read twice the quadratic coefficient of y(x) off the samples at `x`: the curvature."""
  return 2.0 * np.linalg.pinv(np.vander(np.asarray(x), 3))[0]


def lane_bend(model) -> LaneBend:
  """The ego lane's bend from laneLines 1 and 2, confidence-weighted, and where each seen line is."""
  lines, probs = model.laneLines, model.laneLineProbs
  if len(lines) != 4 or len(probs) != 4:
    return LaneBend(math.nan, 0.0, math.nan, math.nan)
  bends, weights, offsets = [], [], [math.nan, math.nan]
  for side, i in enumerate((1, 2)):
    x, y = np.asarray(lines[i].x, dtype=float), np.asarray(lines[i].y, dtype=float)
    near = np.flatnonzero((x >= 0.0) & (x <= MOONPILOT_RECENTER_FIT_X))
    weight = float(np.interp(probs[i], *MOONPILOT_RECENTER_PROBS))
    if len(near) < 5 or len(x) != len(y) or weight <= 0.0 or not np.isfinite(y[near]).all():
      continue
    bends.append(float(_curvature_weights(tuple(x[near])) @ y[near]))
    weights.append(weight)
    offsets[side] = float(y[near[0]])
  if not weights:
    return LaneBend(math.nan, 0.0, math.nan, math.nan)
  return LaneBend(float(np.average(bends, weights=weights)), max(weights), *offsets)


def recentered(request: float, lane: LaneBend, v_ego: float, cap: float = MOONPILOT_RECENTER_LAT_ACCEL) -> float:
  """The request with its part beyond the lane's bend held to `cap` m/s^2, faded by speed, by the lines' weight and out
  in curves, and left alone unless the side the car moves away from is known to be clear. Only ever moves it toward zero."""
  if not math.isfinite(lane.bend):
    return request
  beyond = request - lane.bend
  recovery_side = lane.left if beyond > 0.0 else lane.right  # positive is right, so a right move leaves the left side
  if not abs(recovery_side) >= MOONPILOT_RECENTER_NEAR_LINE:  # close, or nan -- not established: this may be a correction
    return request
  v = max(v_ego, MIN_SPEED)
  fade = float(np.interp(v_ego, *MOONPILOT_RECENTER_SPEEDS) * np.interp(abs(lane.bend) * v**2, *MOONPILOT_RECENTER_BENDS)) * lane.weight
  if fade <= 0.0:
    return request
  limit = cap / v**2
  bounded = float(np.clip(lane.bend + float(np.clip(beyond, -limit, limit)), min(0.0, request), max(0.0, request)))
  return request + fade * (bounded - request)


def _toward(desired_curvature: float, target: float, v_ego: float, gain: float, cap: float) -> float:
  """The request plus `gain` of the way to `target`, the step capped at `cap` m/s^2; the request if `target` is not finite."""
  if not math.isfinite(target):
    return desired_curvature
  limit = cap / max(v_ego, MIN_SPEED) ** 2
  return desired_curvature + float(np.clip(gain * (target - desired_curvature), -limit, limit))


def response_aligned_curvature(model, desired_curvature: float, *, v_ego: float, lat_delay: float, model_recv_time: float, now: float,
                               model_valid: bool, lookahead: tuple[float, float, float] | None = None, lead: float = 0.0,
                               recenter: float = 0.0) -> float:
  """Return the decoded request plus the smoothing lead and the look-ahead lead-in, re-centering capped, or the
  request unchanged. Called once per control frame."""
  if not model_valid:
    return desired_curvature

  capture_time = model.timestampEof * 1e-9
  if not all(math.isfinite(value) for value in (desired_curvature, v_ego, lat_delay, capture_time, model_recv_time, now)):
    return desired_curvature
  if v_ego <= MOONPILOT_PREVIEW_MIN_SPEED or lat_delay < 0.0 or capture_time <= 0.0:
    return desired_curvature

  capture_age = now - capture_time
  message_age = now - model_recv_time
  if not 0.0 <= capture_age <= MOONPILOT_MAX_MODEL_AGE or not 0.0 <= message_age <= MOONPILOT_MAX_MODEL_AGE:
    return desired_curvature

  response_horizon = capture_age + lat_delay
  yaws = np.asarray(model.orientation.z, dtype=float)
  yaw_rates = np.asarray(model.orientationRate.z, dtype=float)
  t_idxs = np.asarray(model.orientation.t, dtype=float)
  rate_t_idxs = np.asarray(model.orientationRate.t, dtype=float)
  if not len(yaws) or not (len(yaws) == len(yaw_rates) == len(t_idxs) == len(rate_t_idxs)):
    return desired_curvature
  if not np.array_equal(t_idxs, rate_t_idxs) or np.any(np.diff(t_idxs) <= 0.0):
    return desired_curvature
  if not all(np.isfinite(values).all() for values in (yaws, yaw_rates, t_idxs, rate_t_idxs)):
    return desired_curvature
  if response_horizon <= 0.0 or response_horizon + lead > t_idxs[-1]:
    return desired_curvature

  today = desired_curvature
  if lead > 0.0:
    # The planned curvature `lead` later than at the response horizon: exactly `slope * lead` on a ramp.
    shift = float(np.interp(response_horizon + lead, t_idxs, yaw_rates) - np.interp(response_horizon, t_idxs, yaw_rates)) / max(v_ego, MIN_SPEED)
    limit = MOONPILOT_SMOOTH_LEAD_MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2
    today = desired_curvature + float(np.clip(shift, -limit, limit))
  start = response_horizon + lead  # the filter lags the lead-in too
  ahead = today
  if lookahead is not None and start + lookahead[0] <= t_idxs[-1]:
    window, gain, cap = lookahead
    # The planned curvature (yaw rate / speed) averaged over the window ahead of the response: a turn coming up
    # pulls the request in early and gradually.
    samples = np.interp(np.arange(start, start + window + 1e-9, 0.05), t_idxs, yaw_rates)
    ahead = _toward(today, float(np.mean(samples)) / max(v_ego, MIN_SPEED), v_ego, gain, cap)
    ahead = today + float(np.interp(v_ego, *MOONPILOT_LOOKAHEAD_SPEEDS)) * (ahead - today)
    # Never less turn than today's request: averaging in a turn's easing would cut its apex and let it out
    # early, and both run wide.
    ahead = today if (ahead - today) * today < 0.0 else ahead
  if recenter <= 0.0 or model.meta.laneChangeState.raw != 0:  # a lane change is a sideways move the driver asked for
    return ahead
  return recentered(ahead, lane_bend(model), v_ego, recenter)


def _smooth_tau(params: Params) -> float:
  return MOONPILOT_SMOOTH_TAUS[choice(PATH_SMOOTH, params)] if enabled(PATH_SMOOTH, params) else 0.0


def moonpilot_curvature(params: Params | None = None) -> Callable[..., float] | None:
  """Return the reference with its look-ahead, smoothing lead and re-centering cap bound, or None when all three
  are off; fixed until restart."""
  params = params or Params()
  lookahead = MOONPILOT_LOOKAHEAD[choice(PATH_LOOKAHEAD, params) if enabled(PATH_LOOKAHEAD, params) else 0]
  lead = _smooth_tau(params)
  recenter = MOONPILOT_RECENTER_LAT_ACCEL if enabled(PATH_RECENTER, params) else 0.0
  if lookahead is None and lead == 0.0 and recenter == 0.0:
    return None
  return functools.partial(response_aligned_curvature, lookahead=lookahead, lead=lead, recenter=recenter)


class PathSmooth:
  """First-order lag on the curvature request: an IIR at the control rate, `x += alpha * (u - x)`.

  While not steering it tracks the input, so a re-engage starts from the path the car is already
  on — the same reset `clip_curvature`'s engage path makes, and what keeps the filter from
  re-injecting its pre-disengage state. A non-finite request passes through untouched with the
  state held: this stage must never be the thing that turns a good value bad.

  The filter lags by `tau`. controlsd does not add it to the controller's `lat_delay` (the request
  buffer already holds the filtered request); `moonpilot_curvature` binds it as the reference's `lead`,
  which time-shifts the request ahead by the same `tau` above `MOONPILOT_PREVIEW_MIN_SPEED`."""

  def __init__(self, tau: float, dt: float = DT_CTRL):
    self.tau = tau
    self._alpha = 1.0 - math.exp(-dt / tau)
    self._x: float | None = None

  def update(self, curvature: float, active: bool) -> float:
    if not math.isfinite(curvature):
      return curvature
    if not active or self._x is None:
      self._x = curvature
      return curvature
    self._x += self._alpha * (curvature - self._x)
    return self._x


def moonpilot_path_smooth(params: Params | None = None) -> PathSmooth | None:
  """Return the curvature request filter at the chosen tau, or None when off; fixed until restart."""
  params = params or Params()
  tau = _smooth_tau(params)
  return PathSmooth(tau) if tau > 0.0 else None
