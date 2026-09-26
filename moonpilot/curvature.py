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

import math
from collections.abc import Callable

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_curvature_from_plan

from moonpilot.features import PATH_PREVIEW, PATH_SMOOTH, enabled

MOONPILOT_PREVIEW_GAIN = 0.3  # share of (response-aligned path - model request) added to the command
MOONPILOT_PREVIEW_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the delta itself may ask for
MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the path curvature is noise
MOONPILOT_MAX_MODEL_AGE = 2 * DT_MDL  # s; two missed 20 Hz model periods disable the correction
MOONPILOT_SMOOTH_TAU = 0.15  # s; first-order time constant of the curvature request filter


def response_aligned_curvature(model, desired_curvature: float, *, v_ego: float, lat_delay: float, model_recv_time: float, now: float,
                               model_valid: bool) -> float:
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

  response_curvature = float(get_curvature_from_plan(yaws, yaw_rates, t_idxs, v_ego, response_horizon))
  if not math.isfinite(response_curvature):
    return desired_curvature
  limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2
  delta = float(np.clip(MOONPILOT_PREVIEW_GAIN * (response_curvature - desired_curvature), -limit, limit))
  return desired_curvature + delta


def moonpilot_curvature(params: Params | None = None) -> Callable[..., float] | None:
  """Return the response-aligned reference when enabled; the choice is fixed until restart."""
  if not enabled(PATH_PREVIEW, params or Params()):
    return None
  return response_aligned_curvature


class PathSmooth:
  """First-order lag on the curvature request: an IIR at the control rate, `x += alpha * (u - x)`.

  While not steering it tracks the input, so a re-engage starts from the path the car is already
  on — the same reset `clip_curvature`'s engage path makes, and what keeps the filter from
  re-injecting its pre-disengage state. A non-finite request passes through untouched with the
  state held: this stage must never be the thing that turns a good value bad.

  The lag the filter adds is `tau` seconds, and controlsd folds it into the `lat_delay` both the
  response-aligned reference and the controller time against, so enabling it does not silently
  desynchronize the fork from its own delay bookkeeping."""

  def __init__(self, tau: float = MOONPILOT_SMOOTH_TAU, dt: float = DT_CTRL):
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
  """Return the curvature request filter when enabled; the choice (and its tau) is fixed until restart."""
  if not enabled(PATH_SMOOTH, params or Params()):
    return None
  return PathSmooth()
