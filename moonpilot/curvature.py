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
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_curvature_from_plan

from moonpilot.features import PATH_PREVIEW, enabled

MOONPILOT_PREVIEW_GAIN = 0.3  # share of (response-aligned path - model request) added to the command
MOONPILOT_PREVIEW_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the delta itself may ask for
MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the path curvature is noise
MOONPILOT_MAX_MODEL_AGE = 2 * DT_MDL  # s; two missed 20 Hz model periods disable the correction


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
