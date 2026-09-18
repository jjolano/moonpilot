"""Normalization for the vision model's lead trajectories.

Pure functions only: the publisher (`moonpilot/leadd.py`), the longitudinal planner and the
onroad renderers all consume this module, so it must stay importable without a messaging loop.

Frame and sign: published `x` is meters forward of the front bumper (radarState.dRel convention)
and published `y` is LEFT POSITIVE (radarState.yRel convention). modelV2's device frame is
right-positive, so `y` is negated on the way out.
"""

import math
from collections import deque

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.features import LEAD_LATERAL, enabled

# Mirrors RADAR_TO_CAMERA in openpilot/selfdrive/controls/radard.py, and pinned to it by
# test_lead.test_matches_radard. Defined here rather than imported so this module stays free of
# radard's messaging/opendbc imports — the planner seam and the renderers import it.
MOONPILOT_RADAR_TO_CAMERA = 1.52  # m; radar is ~1.5m ahead of the camera mesh frame

# Starting points, all fork-owned. Tune against logs: see the lead path ribbon in the UI and
# LeadTrajectory.inPathProb in PlotJuggler.
MOONPILOT_PATH_HALF_WIDTH = 1.8  # m; half a typical lane
MOONPILOT_MIN_SPEED_FOR_YAW = 1.0  # m/s; below this, yawRel is meaningless -> 0.0
MOONPILOT_MIN_Y_STD = 0.05  # m; floor on yStd so the Gaussian never collapses
MOONPILOT_OUT_OF_PATH_DANGER = 0.4  # relaxed MPC danger factor for a fully out-of-path lead
MOONPILOT_INPATH_RC = 1.0  # s; decay time constant of the asymmetric inPath filter
# Dense where a cut-in matters, none of it past the ~4 s where a cut-in can still be avoided.
MOONPILOT_INPATH_GRID = np.arange(0.0, 4.0 + 1e-9, 0.25)
# The lead accel estimator's window. radard's aLeadK is a KF1D with fixed gains (radard.py:29-48),
# tau = 0.49 s at DT_MDL, so 90 % of a -3 m/s^2 step takes 1.20 s. A least-squares slope over this
# window reaches the same step in the window's own length and averages noise instead of lagging it.
MOONPILOT_LEAD_ACCEL_WINDOW = 7  # samples, 0.35 s at DT_MDL
MOONPILOT_LEAD_ACCEL_MIN_SAMPLES = 3  # below this the slope is noise, so radard's value stands
MOONPILOT_LEAD_SPEED_JUMP = 2.5  # m/s in one frame: re-association, not motion (50 m/s^3)

LEAD_T_IDXS = ModelConstants.LEAD_T_IDXS
LEAD_T_OFFSETS = ModelConstants.LEAD_T_OFFSETS

_LIST_FIELDS = ("t", "x", "y", "v", "a", "xStd", "yStd", "vStd", "aStd", "yawRel", "inPathProb")


def resample(t_src, values, t_dst) -> np.ndarray:
  """Linear resample onto t_dst. The one interpolation entry point, so consumers agree.

  Linear on purpose: the model's grid is 2 s apart, and a cubic through two points manufactures
  sub-2 s structure the network never predicted.
  """
  t_src = np.asarray(t_src, dtype=float)
  values = np.asarray(values, dtype=float)
  if t_src.size < 2 or values.size < 2:
    return np.empty((0,), dtype=float)
  return np.interp(np.asarray(t_dst, dtype=float), t_src, values)


class LeadAccelEstimator:
  """A lead's acceleration from the slope of `vLead` over a short window.

  radard reports `aLeadK` from a Kalman filter with hardcoded gains (`radard.py:29-48`): tau is
  0.49 s at DT_MDL, so 90 % of a -3 m/s^2 step arrives 1.20 s late, which is most of the margin a
  braking lead gives us. The least-squares slope over MOONPILOT_LEAD_ACCEL_WINDOW samples has no lag
  of its own — a clean ramp reads exactly as soon as MOONPILOT_LEAD_ACCEL_MIN_SAMPLES are on it,
  0.10 s — so the only latency it carries is the ramp onto a window that already holds samples from
  before the step: 0.30 s at the full window. It averages measurement noise rather than lagging it:
  0.19 m/s^2 of jitter over the window for 0.05 m/s of noise, against 1.41 m/s^2 for a one-frame
  difference. The lead's *speed* is left alone — `vLead` has no filter lag to remove, and this
  fit's endpoint value would add a transient bias exactly at the onset of braking.

  Returns radard's own `aLeadK` whenever the window cannot speak: a vision-only lead (there
  `aLeadK` is the model's own unfiltered accel), fewer than MOONPILOT_LEAD_ACCEL_MIN_SAMPLES
  samples, a new `radarTrackId`, a source flip, or a `vLead` jump past MOONPILOT_LEAD_SPEED_JUMP.

  One instance per radarState slot, updated every frame *including* the frames where the slot is
  absent — the reset is what keeps a new lead from inheriting the previous one's samples. Sample
  spacing is one model frame by construction: radard publishes `radarState` once per `modelV2`
  frame (`radard.py:263-272`), and plannerd is polled on the same message.
  """

  def __init__(self, dt: float = DT_MDL, window: int = MOONPILOT_LEAD_ACCEL_WINDOW):
    self._samples: deque[float] = deque(maxlen=window)
    self._track: tuple[bool, bool, int] | None = None
    # slope = sum(w_i * v_i) for a uniform grid: w_i = 12 (i - (n-1)/2) / (dt n (n^2 - 1)).
    self._weights = {n: (np.arange(n) - (n - 1) / 2.0) * (12.0 / (dt * n * (n * n - 1))) for n in range(MOONPILOT_LEAD_ACCEL_MIN_SAMPLES, window + 1)}

  def update(self, lead) -> float:
    track = (bool(lead.present), bool(lead.radar), int(lead.radarTrackId))
    v_lead = float(lead.vLead)
    if track != self._track or (self._samples and abs(v_lead - self._samples[-1]) > MOONPILOT_LEAD_SPEED_JUMP):
      self._samples.clear()
    self._track = track

    if not (lead.present and lead.radar):
      return float(lead.aLeadK)
    self._samples.append(v_lead)
    if len(self._samples) < MOONPILOT_LEAD_ACCEL_MIN_SAMPLES:
      return float(lead.aLeadK)
    n = len(self._samples)
    return float(self._weights[n] @ np.fromiter(self._samples, float, n))


def lead_yaw_rel(t, y, v) -> list[float]:
  """Lead heading relative to the ego x axis, left positive, one entry per sample.

  Central difference over the non-uniform t grid (forward at index 0, backward at the last).
  Coarse at the model's 2 s spacing, so treat it as a trend, not a measurement.
  """
  if len(t) < 2 or len(y) < 2 or len(v) < 2:
    return []

  t = np.asarray(t, dtype=float)
  y = np.asarray(y, dtype=float)
  v = np.asarray(v, dtype=float)

  dy_dt = np.empty_like(y)
  dy_dt[0] = (y[1] - y[0]) / max(t[1] - t[0], 1e-6)
  dy_dt[-1] = (y[-1] - y[-2]) / max(t[-1] - t[-2], 1e-6)
  dy_dt[1:-1] = (y[2:] - y[:-2]) / np.maximum(t[2:] - t[:-2], 1e-6)

  return [float(math.atan2(dydt, max(vi, MOONPILOT_MIN_SPEED_FOR_YAW))) for dydt, vi in zip(dy_dt, v, strict=True)]


def lead_in_path_prob(x, y, y_std, ego_path_x, ego_path_y) -> list[float]:
  """Per-sample probability the lead's lateral Gaussian lies inside the ego path corridor.

  A probability rather than a binary in/out count: 6 samples give 7 discrete levels otherwise,
  which would step the MPC's danger factor in visible jumps. Continuous in both offset and
  uncertainty, so it is also smoother frame to frame.
  """
  if len(x) == 0 or len(y) == 0 or len(ego_path_x) == 0 or len(ego_path_y) == 0:
    return []

  ego_path_x = np.asarray(ego_path_x, dtype=float)
  ego_path_y = np.asarray(ego_path_y, dtype=float)
  y_std = np.asarray(y_std, dtype=float)

  probs = []
  for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
    # Ego path negated into the left-positive convention the lead's y lives in.
    dy = yi - (-float(np.interp(xi, ego_path_x, ego_path_y)))
    s = max(float(y_std[i]), MOONPILOT_MIN_Y_STD) if i < y_std.size else MOONPILOT_MIN_Y_STD
    w = MOONPILOT_PATH_HALF_WIDTH
    probs.append(0.5 * (math.erf((w - dy) / (math.sqrt(2.0) * s)) + math.erf((w + dy) / (math.sqrt(2.0) * s))))
  return probs


def lead_in_path(t, x, y, y_std, ego_path_x, ego_path_y, filt) -> float:
  """One scalar in-path probability, inverse-variance weighted and asymmetrically filtered.

  `filt` is a caller-owned FirstOrderFilter whose state must persist across frames. Weighting by
  1/yStd^2 lets the samples the model is confident about dominate; if yStd is flat this degenerates
  to a uniform mean. Rising is instant (stock behavior returns immediately), decaying takes about
  MOONPILOT_INPATH_RC of consistent evidence. Both directions fail safe.
  """
  if len(t) < 2 or len(x) < 2 or len(y) < 2 or len(ego_path_x) == 0 or len(ego_path_y) == 0:
    filt.x = 1.0
    return 1.0

  x_d = resample(t, x, MOONPILOT_INPATH_GRID)
  y_d = resample(t, y, MOONPILOT_INPATH_GRID)
  s_d = resample(t, y_std, MOONPILOT_INPATH_GRID) if len(y_std) >= 2 else np.full(MOONPILOT_INPATH_GRID.shape, MOONPILOT_MIN_Y_STD)
  if x_d.size == 0 or y_d.size == 0:
    filt.x = 1.0
    return 1.0

  probs = np.array(lead_in_path_prob(x_d, y_d, s_d, ego_path_x, ego_path_y), dtype=float)
  s_clamped = np.maximum(s_d[: probs.size], MOONPILOT_MIN_Y_STD)
  weights = 1.0 / s_clamped**2
  raw = float(np.sum(probs * weights) / np.sum(weights))

  if raw > filt.x:
    filt.x = raw
  else:
    filt.update(raw)
  return float(filt.x)


def _empty(slot: int, filt) -> dict:
  filt.x = 1.0
  return {
    "present": False,
    "prob": 0.0,
    "probTime": float(LEAD_T_OFFSETS[slot]) if slot < len(LEAD_T_OFFSETS) else 0.0,
    "source": "none",
    **{field: [] for field in _LIST_FIELDS},
    "inPath": 1.0,
  }


def normalize_lead(slot, model_lead, fused_lead, ego_path_x, ego_path_y, in_path_filter) -> dict:
  """One leadsV3 slot as a LeadTrajectory dict, on the model's native 6-point t grid.

  `fused_lead` is radarState.leadOne/leadTwo, or None for slot 2 / when radard published none.
  """
  if model_lead is None or len(model_lead.x) == 0 or len(ego_path_x) == 0:
    return _empty(slot, in_path_filter)

  model_x = np.asarray(model_lead.x, dtype=float)
  model_y = np.asarray(model_lead.y, dtype=float)

  # The low-speed override substitutes a bare radar track that does not correspond to this slot,
  # so it has no predicted trajectory at all: one point, straight from the fused state.
  if fused_lead is not None and fused_lead.present and fused_lead.radar and fused_lead.modelProb == 0.0:
    return {
      "present": True,
      "prob": float(fused_lead.modelProb),
      "probTime": float(LEAD_T_OFFSETS[slot]) if slot < len(LEAD_T_OFFSETS) else 0.0,
      "source": "radar",
      "t": [0.0],
      "x": [float(fused_lead.dRel)],
      "y": [float(fused_lead.yRel)],
      "v": [float(fused_lead.vLead)],
      "a": [float(fused_lead.aLeadK)],
      "xStd": [],
      "yStd": [],
      "vStd": [],
      "aStd": [],
      "yawRel": [],
      "inPath": 1.0,
      "inPathProb": [],
    }

  if fused_lead is not None and fused_lead.present:
    # Anchor the predicted shape to the fused t=0 state, so index 0 equals radarState exactly.
    x = float(fused_lead.dRel) + (model_x - model_x[0])
    y = float(fused_lead.yRel) + (-model_y + model_y[0])
    prob = float(fused_lead.modelProb)
  else:
    # Vision-only: no fused state to anchor to, so publish in the raw radarState conventions.
    x = model_x - MOONPILOT_RADAR_TO_CAMERA
    y = -model_y
    prob = float(model_lead.prob)

  t = np.asarray(model_lead.t, dtype=float) if len(model_lead.t) else np.asarray(LEAD_T_IDXS[: model_x.size], dtype=float)
  y_std = np.asarray(model_lead.yStd, dtype=float)

  return {
    "present": True,
    "prob": prob,
    "probTime": float(model_lead.probTime),
    "source": "vision",
    "t": t.tolist(),
    "x": x.tolist(),
    "y": y.tolist(),
    "v": list(model_lead.v),
    "a": list(model_lead.a),
    "xStd": list(model_lead.xStd),
    "yStd": y_std.tolist(),
    "vStd": list(model_lead.vStd),
    "aStd": list(model_lead.aStd),
    "yawRel": lead_yaw_rel(t, y, model_lead.v),
    "inPath": lead_in_path(t, x, y, y_std, ego_path_x, ego_path_y, in_path_filter),
    "inPathProb": lead_in_path_prob(x, y, y_std, ego_path_x, ego_path_y),
  }


def _moonpilot_leads(sm):
  """The published leads, or None when there is no valid, alive moonpilotState.

  getattr, not sm.valid: the longitudinal maneuver harness
  (test/longitudinal_maneuvers/plant.py) passes a plain dict as `sm`.
  """
  if not (getattr(sm, "valid", {}).get("moonpilotState", False) and getattr(sm, "alive", {}).get("moonpilotState", False)):
    return None
  return sm["moonpilotState"].leads


def lead_danger_factor(sm, params: Params, default: float) -> float:
  """MPC lead_danger_factor for the nearest lead, scaled by how likely it is to stay in path.

  `default` is handed in by the seam so fork code never imports long_mpc (which pulls in the
  compiled acados solver). At inPath == 1.0 this returns `default` exactly, so with the feature
  off, or with no moonpilotState published, the MPC behaves bit-identically to upstream.
  """
  if not enabled(LEAD_LATERAL, params):
    return default

  leads = _moonpilot_leads(sm)
  if leads is None or len(leads) == 0 or not leads[0].present:
    return default

  # One scalar shared by both MPC leads, so only the nearest one's prediction can be acted on.
  return float(np.interp(leads[0].inPath, [0.0, 1.0], [MOONPILOT_OUT_OF_PATH_DANGER, default]))


def nearest_lead_in_path(sm) -> float:
  """inPath probability of the nearest published lead; 1.0 when there is nothing to say about it.

  lead_danger_factor's counterpart for the fork longitudinal policy, which has no MPC danger
  zone to scale and scales its time gap instead.
  """
  leads = _moonpilot_leads(sm)
  if leads is None or len(leads) == 0 or not leads[0].present:
    return 1.0
  return float(leads[0].inPath)
