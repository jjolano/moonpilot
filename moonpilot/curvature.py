"""The fork's path preview: a bounded share of the curvature the model's path is *about to* ask for.

Upstream's lateral command is `modelV2.action.desiredCurvature` sampled at the delay-matched action
time in modeld, so a curve is entered as late as that horizon and the car turns in once the model's
own plan has already started. This module reads the same plan a little further out and hands controlsd
a delta, which is all it can be: the model's own request stays the base, a gain of 0 is upstream
exactly, and `clip_curvature` downstream still applies the ISO lateral jerk and lateral accel limits,
so nothing here can widen the envelope the car is allowed to use.

The preview is upstream's own `get_curvature_from_plan` at a longer action time rather than new math —
the same function modeld uses for the request it publishes, so the two are the same quantity read at
two horizons and their difference is meaningful. It is chosen once at construction, so the toggle takes
a restart, like the fork's other control-path behaviors.

Two non-obvious points:

- **The smoothing runs at `DT_CTRL`.** `smooth_value`'s default `dt` is `DT_MDL`, and controlsd runs at
  100 Hz; with the default the filter would move five times too fast.
- **Every gate returns a raw delta of `0.0` rather than leaving the last one standing**, so the filter
  decays instead of stepping when the preview stops speaking — below `MOONPILOT_PREVIEW_MIN_SPEED`, on
  a path with no yaw arrays, on a non-finite one, and whenever lateral control is not active.

`LAT_SMOOTH_SECONDS` is deliberately not imported: the seam passes the already-summed delay, which
keeps this module off modeld's import graph (modeld imports its constants).
"""

import math

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED, get_curvature_from_plan, smooth_value
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.features import PATH_PREVIEW, enabled

MOONPILOT_PREVIEW_T = 0.45  # s past the delay-matched action time the preview curvature is read at
MOONPILOT_PREVIEW_GAIN = 0.3  # share of (preview - model request) added to the command
MOONPILOT_PREVIEW_MAX_LAT_ACCEL = 0.8  # m/s^2; ceiling on what the delta itself may ask for
MOONPILOT_PREVIEW_MIN_SPEED = 5.0  # m/s; below this the preview is noise
MOONPILOT_PREVIEW_SMOOTH_T = 0.25  # s time constant on the delta, at DT_CTRL


class MoonpilotCurvature:
  """The preview itself. Stateless apart from the smoothed delta, which is why nothing here needs a
  process of its own — controlsd calls `update` once per frame with the state it already has."""

  def __init__(self):
    self.delta = 0.0

  def _raw_delta(self, model, v_ego, lat_delay, desired_curvature) -> float:
    if v_ego <= MOONPILOT_PREVIEW_MIN_SPEED:
      return 0.0
    yaws, yaw_rates = model.orientation.z, model.orientationRate.z
    # `get_curvature_from_plan` interpolates over the model's own time grid, so a yaw array of any
    # other length is not the same quantity and is refused rather than read at the wrong times.
    if len(yaws) != len(ModelConstants.T_IDXS) or len(yaw_rates) == 0:
      return 0.0
    preview = get_curvature_from_plan(yaws, yaw_rates, ModelConstants.T_IDXS, v_ego, max(float(lat_delay), 0.0) + MOONPILOT_PREVIEW_T)
    limit = MOONPILOT_PREVIEW_MAX_LAT_ACCEL / max(v_ego, MIN_SPEED) ** 2
    delta = float(np.clip(MOONPILOT_PREVIEW_GAIN * (float(preview) - desired_curvature), -limit, limit))
    return delta if math.isfinite(delta) else 0.0

  def update(self, model, v_ego, lat_delay, lat_active, desired_curvature) -> float:
    """The curvature to command: the caller's own request plus a bounded delta, or that request
    unchanged while this is not ours to refine."""
    desired_curvature = float(desired_curvature)
    if not lat_active:
      # Not ours to refine: controlsd hands in the *measured* curvature while lateral is off, and a
      # delta carried across a re-engage would arrive as a step.
      self.delta = 0.0
      return desired_curvature
    self.delta = float(smooth_value(self._raw_delta(model, v_ego, lat_delay, desired_curvature), self.delta, MOONPILOT_PREVIEW_SMOOTH_T, DT_CTRL))
    out = desired_curvature + self.delta
    return out if math.isfinite(out) else desired_curvature


def moonpilot_curvature(params: Params | None = None) -> MoonpilotCurvature | None:
  """The seam's fork side: the preview when the driver wants it, None to leave the model's own
  request untouched. The param is read once, here, so the toggle takes a restart."""
  if not enabled(PATH_PREVIEW, params or Params()):
    return None
  return MoonpilotCurvature()
