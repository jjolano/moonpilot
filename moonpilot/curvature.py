"""Response-aligned steering from the model's timestamped path.

The decoded `modelV2.action.desiredCurvature` stays authoritative. This module samples the model
path at the time the steering response is expected, then adds only a bounded share of the
difference. `clip_curvature` downstream still owns the command's jerk, lateral-acceleration, and
maximum-curvature envelope.

Capture age and receive age are checked independently. Any stale, future-dated, malformed, or
out-of-grid input returns the decoded action exactly; no previous correction is retained.

`LAT_SMOOTH_SECONDS` is deliberately not imported: the seam passes the already-summed delay, which
keeps this module off modeld's import graph (modeld imports its constants).
"""

import functools
import math
from collections.abc import Callable

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_curvature_from_plan

from moonpilot.features import PATH_LOOKAHEAD, PATH_PREVIEW, PATH_SMOOTH, choice, enabled

MOONPILOT_PREVIEW_GAIN = 0.3  # share of (response-aligned path - model request) added to the command
MOONPILOT_PREVIEW_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the delta itself may ask for
MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the path curvature is noise
MOONPILOT_MAX_MODEL_AGE = 2 * DT_MDL  # s; two missed 20 Hz model periods disable the correction
MOONPILOT_SMOOTH_TAUS = (0.0, 0.05, 0.10, 0.15)  # s; first-order time constant per MoonpilotPathSmooth choice, 0 = off
# Look-ahead per MoonpilotPathLookahead choice: (window s, gain, cap m/s^2); None = off, the single sample above.
# The mean planned curvature over [horizon, horizon + window] may only add turn to today's request, never take
# any away, so a turn-in starts early while the apex and the exit stay as today's: on openpilot's recorded turns
# (routes 3f8-410) the request started 0.05/0.21/0.05 s (light), 0.15/0.39/0.19 s (medium) and 0.30/0.62/0.26 s
# (strong) earlier at 5-10/10-17/>17 m/s with every turn's apex at 100% of today's (p10 included) and no outward
# drift at the median. Unrestricted, the same averages cut the apex to 82-96% and let turns out early -- running
# wide. The cost is a turn-in that begins inside the lane: 0.5-1.1 / 1.0-2.2 / 1.1-3.1 m median open-loop,
# before the model's own re-planning. None makes highway builds lighter: the controller adds that abruptness.
MOONPILOT_LOOKAHEAD: tuple[tuple[float, float, float] | None, ...] = (None, (1.0, 0.3, 0.8), (1.0, 0.6, 1.5), (1.5, 0.6, 1.5))
# Below about 36 km/h, residential corners and intersections, an early turn-in is a cut toward the inside -- the curb
# on a right turn -- for little lead (0.05-0.30 s at 5-10 m/s). Faded out there: at 5-10 m/s the lead-in's median
# open-loop inside drift fell from 0.8-2.0 m to 0.07-0.15 m, and the 10-17 and >17 m/s leads were unchanged.
MOONPILOT_LOOKAHEAD_SPEEDS = ([8.0, 12.0], [0.0, 1.0])  # m/s: none below 8, full from 12


def _toward(desired_curvature: float, target: float, v_ego: float, gain: float, cap: float) -> float:
  """The request plus `gain` of the way to `target`, the step capped at `cap` m/s^2; the request if `target` is not finite."""
  if not math.isfinite(target):
    return desired_curvature
  limit = cap / max(v_ego, MIN_SPEED) ** 2
  return desired_curvature + float(np.clip(gain * (target - desired_curvature), -limit, limit))


def response_aligned_curvature(model, desired_curvature: float, *, v_ego: float, lat_delay: float, model_recv_time: float, now: float,
                               model_valid: bool, lookahead: tuple[float, float, float] | None = None) -> float:
  """Return the decoded request plus a bounded path correction, or the request unchanged. Called once per control frame."""
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
  if response_horizon <= 0.0 or response_horizon > t_idxs[-1]:
    return desired_curvature

  today = _toward(desired_curvature, float(get_curvature_from_plan(yaws, yaw_rates, t_idxs, v_ego, response_horizon)), v_ego,
                  MOONPILOT_PREVIEW_GAIN, MOONPILOT_PREVIEW_MAX_LAT_ACCEL)
  if lookahead is None or response_horizon + lookahead[0] > t_idxs[-1]:
    return today
  window, gain, cap = lookahead
  # The planned curvature (yaw rate / speed) averaged over the window ahead of the response: a turn coming up
  # pulls the request in early and gradually.
  samples = np.interp(np.arange(response_horizon, response_horizon + window + 1e-9, 0.05), t_idxs, yaw_rates)
  ahead = _toward(desired_curvature, float(np.mean(samples)) / max(v_ego, MIN_SPEED), v_ego, gain, cap)
  ahead = today + float(np.interp(v_ego, *MOONPILOT_LOOKAHEAD_SPEEDS)) * (ahead - today)
  # Never less turn than today's request: averaging in a turn's easing would cut its apex and let it out
  # early, and both run wide.
  return today if (ahead - today) * today < 0.0 else ahead


def moonpilot_curvature(params: Params | None = None) -> Callable[..., float] | None:
  """Return the response-aligned reference when enabled, with the look-ahead chosen at the same time;
  both are fixed until restart."""
  params = params or Params()
  if not enabled(PATH_PREVIEW, params):
    return None
  level = choice(PATH_LOOKAHEAD, params) if enabled(PATH_LOOKAHEAD, params) else 0
  lookahead = MOONPILOT_LOOKAHEAD[level]
  return functools.partial(response_aligned_curvature, lookahead=lookahead) if lookahead else response_aligned_curvature


class PathSmooth:
  """First-order lag on the curvature request: an IIR at the control rate, `x += alpha * (u - x)`.

  While not steering it tracks the input, so a re-engage starts from the path the car is already
  on — the same reset `clip_curvature`'s engage path makes, and what keeps the filter from
  re-injecting its pre-disengage state. A non-finite request passes through untouched with the
  state held: this stage must never be the thing that turns a good value bad.

  The lag the filter adds is `tau` seconds, and controlsd folds it into the `lat_delay` both the
  response-aligned reference and the controller time against, so enabling it does not silently
  desynchronize the fork from its own delay bookkeeping."""

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
  tau = MOONPILOT_SMOOTH_TAUS[choice(PATH_SMOOTH, params)] if enabled(PATH_SMOOTH, params) else 0.0
  return PathSmooth(tau) if tau > 0.0 else None
