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

from moonpilot.features import PATH_LOOKAHEAD, PATH_RECENTER, PATH_SMOOTH, enabled, value

MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the path curvature is noise
MOONPILOT_SMOOTH_LEAD_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the smoothing lead itself may ask for
MOONPILOT_MAX_MODEL_AGE = 2 * DT_MDL  # s; two missed 20 Hz model periods disable the correction
# Below about 36 km/h, residential corners and intersections, an early turn-in (PATH_LOOKAHEAD) is a cut toward the
# inside -- the curb on a right turn -- for little lead (up to ~0.1 s of median lead at 5-10 m/s, unfaded). Faded out
# there: at 5-10 m/s the lead-in's median open-loop inside drift fell from 0.2-0.5 m with the fade disabled to 0.00 m,
# and the 10-17 and >17 m/s leads were unchanged.
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
MOONPILOT_CORRECTION_SPEEDS = ([8.0, 12.0], [0.0, 1.0])  # m/s
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


def recentered(request: float, lane: LaneBend, v_ego: float, cap: float) -> float:
  """See the comment above `MOONPILOT_RECENTER_LAT_ACCEL`."""
  if not math.isfinite(lane.bend):
    return request
  beyond = request - lane.bend
  recovery_side = lane.left if beyond > 0.0 else lane.right  # positive is right, so a right move leaves the left side
  if not abs(recovery_side) >= MOONPILOT_RECENTER_NEAR_LINE:  # close, or nan -- not established: this may be a correction
    return request
  v = max(v_ego, MIN_SPEED)
  fade = float(np.interp(v_ego, *MOONPILOT_CORRECTION_SPEEDS) * np.interp(abs(lane.bend) * v**2, *MOONPILOT_RECENTER_BENDS)) * lane.weight
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


def curvature_reference(model, desired_curvature: float, *, v_ego: float, lat_delay: float, model_recv_time: float, now: float,
                               model_valid: bool, lookahead: tuple[float, float, float] | None = None, lead: float = 0.0,
                               recenter: float = 0.0) -> float:
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
    samples = np.interp(np.arange(start, start + window + 1e-9, 0.05), t_idxs, yaw_rates)
    ahead = _toward(today, float(np.mean(samples)) / max(v_ego, MIN_SPEED), v_ego, gain, cap)
    ahead = today + float(np.interp(v_ego, *MOONPILOT_CORRECTION_SPEEDS)) * (ahead - today)
    # Never less turn than today's request: averaging in a turn's easing would cut its apex and let it out
    # early, and both run wide.
    ahead = today if (ahead - today) * today < 0.0 else ahead
  if recenter <= 0.0 or model.meta.laneChangeState.raw != 0:  # a lane change is a sideways move the driver asked for
    return ahead
  return recentered(ahead, lane_bend(model), v_ego, recenter)


def moonpilot_curvature(params: Params | None = None) -> Callable[..., float] | None:
  """Return the reference with its look-ahead, smoothing lead and re-centering cap bound, or None when all three
  are off; fixed until restart."""
  params = params or Params()
  lookahead, lead = value(PATH_LOOKAHEAD, params), value(PATH_SMOOTH, params)
  recenter = MOONPILOT_RECENTER_LAT_ACCEL if enabled(PATH_RECENTER, params) else 0.0
  if lookahead is None and lead == 0.0 and recenter == 0.0:
    return None
  return functools.partial(curvature_reference, lookahead=lookahead, lead=lead, recenter=recenter)


class PathSmooth:
  """A first-order lag on the curvature request.

  controlsd tracks the filter while lateral is inactive and leads its response horizon by `tau`.
  """

  def __init__(self, tau: float):
    self.tau = tau
    self._alpha = 1.0 - math.exp(-DT_CTRL / tau)
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
  tau = value(PATH_SMOOTH, params or Params())
  return PathSmooth(tau) if tau > 0.0 else None
