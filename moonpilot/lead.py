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

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.features import LEAD_LATERAL, enabled

# Mirrors RADAR_TO_CAMERA in openpilot/selfdrive/controls/radard.py, and pinned to it by
# test_lead.test_matches_radard. Defined here rather than imported so this module stays free of
# radard's messaging/opendbc imports — the planner seam and the renderers import it.
MOONPILOT_RADAR_TO_CAMERA = 1.52  # m; radar is ~1.5m ahead of the camera mesh frame

# Starting points, all fork-owned. Tune against logs: see the lead path line in the UI and
# LeadTrajectory.inPathProb in PlotJuggler.
MOONPILOT_PATH_HALF_WIDTH = 1.8  # m; half a typical lane
MOONPILOT_MIN_SPEED_FOR_YAW = 1.0  # m/s; below this, yawRel is meaningless -> 0.0
MOONPILOT_MIN_Y_STD = 0.05  # m; floor on yStd so the Gaussian never collapses
MOONPILOT_OUT_OF_PATH_DANGER = 0.4  # relaxed MPC danger factor for a fully out-of-path lead
MOONPILOT_INPATH_RC = 1.0  # s; decay time constant of the asymmetric inPath filter
# Dense where a cut-in matters, none of it past the ~4 s where a cut-in can still be avoided.
MOONPILOT_INPATH_HORIZON = 4.0  # s; the last sample the grid is evaluated at, and so the last one
# the planner can act on. The renderers stop their lead line here too: past it the model's own
# yStd is ~1 m and the drawn 10 s tail landed beyond the white path's own 100 m cap.
MOONPILOT_INPATH_GRID = np.arange(0.0, MOONPILOT_INPATH_HORIZON + 1e-9, 0.25)
# The model's own confidence in a slot, below which normalize_lead publishes it as empty rather
# than as a lead. radard refuses a vision lead at this same threshold (radard.py:156-165) — and, as
# there, only behind an asymmetric filter (MOONPILOT_LEAD_PROB_RC), because slot 0 crosses it in
# runs whose median is 4 frames and 29 % of which are a single frame (corpus, 47k frames), so a raw
# gate would blink at 20 Hz. The fail direction is upstream's own — no lead, inPath 1.0, base
# t_follow — which is why nothing downstream has to know the slot was ever there.
MOONPILOT_LEAD_PROB_GATE = 0.5
MOONPILOT_LEAD_PROB_RC = 0.2  # s; radard's own gate-filter decay (radard.py:234-241)
# The lead accel estimator's window. radard's aLeadK is a KF1D with fixed gains (radard.py:29-48),
# tau = 0.49 s at DT_MDL, so 90 % of a -3 m/s^2 step takes 1.20 s. A least-squares slope over this
# window reaches the same step in the window's own length and averages noise instead of lagging it.
MOONPILOT_LEAD_ACCEL_WINDOW = 7  # samples, 0.35 s at DT_MDL
MOONPILOT_LEAD_ACCEL_MIN_SAMPLES = 3  # below this the slope is noise, so radard's value stands
MOONPILOT_LEAD_SPEED_JUMP = 2.5  # m/s in one frame: re-association, not motion (50 m/s^3)
# The onset window. The full window's bias is what a braking lead costs us: at a step onto -3.5 m/s^2
# the seven-sample slope needs 0.20 s to pass -2.0 and 0.25 s to pass -3.0, because that is how long
# the window is still half full of pre-onset samples — and a real brake ramps in over a few tenths, so
# the lead's own onset is what the estimate is chasing. The newest three samples read the step in
# 0.10 s. Reading both and believing the short one only when it is persistently deeper is one-sided by
# construction (the policy only ever uses a_lead to add braking — `min(a_lead, 0.0)` in
# `moonpilot/longitudinal.py`'s preview), and the streak is what keeps radar noise out of the command:
# a single noisy frame cannot flip it, while an onset stays deep for many. Measured: on 51k settled
# frames of the offline corpus (199 segments, the radar lead's own speed history) the estimate's error
# against a centered reference is 0.188 -> 0.190 m/s^2 sd, and a spurious read past -1 m/s^2 goes
# 0.19 % -> 0.21 % of frames — both inside a measurement whose radar vLead noise is 0.024-0.035 m/s,
# so the design's own 0.05 m/s assumption is conservative. Closed loop
# (`moonpilot/tests/test_longitudinal.py`), a lead braking at -3.5 m/s^2 from a settled follow: the
# command reaches -1.0 m/s^2 at 0.15 s instead of 0.25, -2.0 at 0.30 instead of 0.35 and -3.0 at
# 0.40 instead of 0.45 — and the arm with the lead's *true* accel instead of any estimate reaches
# -1.0 at the same 0.15 s, so this closes the sensing side of that onset rather than chipping at it.
MOONPILOT_LEAD_ACCEL_FAST = 3  # samples in the onset window, 0.10 s at DT_MDL
MOONPILOT_LEAD_ACCEL_FAST_MARGIN = 1.0  # m/s^2 the onset window must read deeper than the full one
MOONPILOT_LEAD_ACCEL_FAST_STREAK = 2  # consecutive frames the margin must hold before it is used
# radard's accel-decay model, mirrored rather than imported: `radard.py` pulls messaging and opendbc,
# and this module is on plannerd's and both renderers' import path. `MOONPILOT_LEAD_ACCEL_TAU` is
# pinned to radard's own constant by test_lead; the two below copy inline literals there
# (`radard.py:55,76`), so they have nothing to assert against and this comment is their only record.
MOONPILOT_LEAD_ACCEL_TAU = 1.5  # s; the decay a lead believed to be holding its accel gets
MOONPILOT_LEAD_ACCEL_TAU_RC = 0.45  # s; time constant of the filter that gives that belief up
MOONPILOT_LEAD_ACCEL_TAU_RESET = 0.5  # m/s^2; |a| above this is braking, not a transient

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
  before the step: 0.30 s at the full window. The onset window is what shortens that: the newest
  MOONPILOT_LEAD_ACCEL_FAST samples are read beside the full window and believed only when they are
  persistently deeper, which reaches -2.0 m/s^2 in 0.10 s on a step where the full window needs
  0.20 s, and -3.0 in 0.10 s where it needs 0.25. It averages measurement noise rather than lagging it:
  0.19 m/s^2 of jitter over the window for 0.05 m/s of noise, against 1.41 m/s^2 for a one-frame
  difference. The lead's *speed* is left alone — `vLead` has no filter lag to remove, and this
  fit's endpoint value would add a transient bias exactly at the onset of braking.

  Returns radard's own `aLeadK` and `aLeadTau` whenever the window cannot speak: a vision-only lead
  (there `aLeadK` is the model's own unfiltered accel, paired with its own 0.3 s decay), fewer than
  MOONPILOT_LEAD_ACCEL_MIN_SAMPLES samples, a new `radarTrackId`, a source flip, or a `vLead` jump
  past MOONPILOT_LEAD_SPEED_JUMP. Both are returned because they are one judgment: the decay says how
  long the accel is expected to hold, and it must describe the accel it travels with.

  The decay is radard's rule — MOONPILOT_LEAD_ACCEL_TAU until |a| passes
  MOONPILOT_LEAD_ACCEL_TAU_RESET, then a filter toward 0 — driven by *this* estimate rather than by
  radard's `aLeadK`. That matters because the two disagree exactly where it counts: at the onset of a
  brake the fork's estimate is already deep while radard's Kalman is still under the reset threshold,
  so pairing the fork's accel with radard's decay says "transient" about a sustained brake, and the
  published plan's 2.5 s tail decays the braking away — it contradicts the command for 0.25-0.40 s
  past a -3.5 m/s^2 onset, worst +0.24 m/s^2 against a -1.8 m/s^2 command. On this estimate's decay
  that window is 0.20-0.25 s. The command itself never sees either: the action horizon is 0.20 s,
  where the decay is 3 %. The residue is the decay *model* — exp(-aLeadTau t^2 / 2) assumes a lead's
  accel is a transient — and 2.5 s is where that assumption is wrong.

  One instance per radarState slot, updated every frame *including* the frames where the slot is
  absent — the reset is what keeps a new lead from inheriting the previous one's samples. Sample
  spacing is one model frame by construction: radard publishes `radarState` once per `modelV2`
  frame (`radard.py:263-272`), and plannerd is polled on the same message.
  """

  def __init__(self, dt: float = DT_MDL, window: int = MOONPILOT_LEAD_ACCEL_WINDOW):
    self._samples: deque[float] = deque(maxlen=window)
    self._track: tuple[bool, bool, int] | None = None
    self._fast_streak = 0
    self._tau = FirstOrderFilter(MOONPILOT_LEAD_ACCEL_TAU, MOONPILOT_LEAD_ACCEL_TAU_RC, dt)
    self.a_lead_tau = MOONPILOT_LEAD_ACCEL_TAU
    # slope = sum(w_i * v_i) for a uniform grid: w_i = 12 (i - (n-1)/2) / (dt n (n^2 - 1)).
    self._weights = {n: (np.arange(n) - (n - 1) / 2.0) * (12.0 / (dt * n * (n * n - 1))) for n in range(MOONPILOT_LEAD_ACCEL_MIN_SAMPLES, window + 1)}

  def update(self, lead) -> tuple[float, float]:
    track = (bool(lead.present), bool(lead.radar), int(lead.radarTrackId))
    v_lead = float(lead.vLead)
    if track != self._track or (self._samples and abs(v_lead - self._samples[-1]) > MOONPILOT_LEAD_SPEED_JUMP):
      self._samples.clear()
      self._fast_streak = 0
      # A new lead's decay is new too: radard builds a fresh `Track` — and a fresh filter — for a new
      # `radarTrackId`, so carrying the old track's value over would tell the rollout that this lead's
      # accel holds for a length the previous lead earned.
      self._tau.x = MOONPILOT_LEAD_ACCEL_TAU
    self._track = track

    # No window to read from: hand back radard's own pair, which is self-consistent by construction.
    if not (lead.present and lead.radar):
      return float(lead.aLeadK), float(lead.aLeadTau)
    self._samples.append(v_lead)
    if len(self._samples) < MOONPILOT_LEAD_ACCEL_MIN_SAMPLES:
      return float(lead.aLeadK), float(lead.aLeadTau)

    n = len(self._samples)
    samples = np.fromiter(self._samples, float, n)
    a_lead = float(self._weights[n] @ samples)
    # The onset window, one-sided: it may only deepen the estimate, never lift it, because deeper is
    # the direction the policy's preview and floor already treat as more braking. The streak is the
    # noise gate — a real onset holds it for many frames, a noisy one cannot hold it at all.
    if n >= MOONPILOT_LEAD_ACCEL_FAST:
      a_fast = float(self._weights[MOONPILOT_LEAD_ACCEL_FAST] @ samples[-MOONPILOT_LEAD_ACCEL_FAST:])
      self._fast_streak = self._fast_streak + 1 if a_fast < a_lead - MOONPILOT_LEAD_ACCEL_FAST_MARGIN else 0
      if self._fast_streak >= MOONPILOT_LEAD_ACCEL_FAST_STREAK:
        a_lead = min(a_lead, a_fast)
    if abs(a_lead) < MOONPILOT_LEAD_ACCEL_TAU_RESET:
      # `radard.py:77` re-arms the filter's own state here, not just the value it reports, and that
      # is what lets the *next* onset decay from the long tau. Assigning only `a_lead_tau` leaves
      # `_tau.x` ratcheting toward 0, so one hard brake would flatten the decay for the whole drive.
      self._tau.x = MOONPILOT_LEAD_ACCEL_TAU
      self.a_lead_tau = MOONPILOT_LEAD_ACCEL_TAU
    else:
      self.a_lead_tau = self._tau.update(0.0)
    return a_lead, self.a_lead_tau


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


def _filtered_prob(prob: float, filt) -> float:
  """radard's own gate filter: a rise is instant, a fall decays, so one low-prob frame cannot drop
  a lead. Same shape as the inPath filter, and the same reason."""
  if prob > filt.x:
    filt.x = prob
  else:
    filt.update(prob)
  return float(filt.x)


def normalize_lead(slot, model_lead, fused_lead, ego_path_x, ego_path_y, in_path_filter, prob_filter) -> dict:
  """One leadsV3 slot as a LeadTrajectory dict, on the model's native 6-point t grid.

  `fused_lead` is radarState.leadOne/leadTwo, or None for slot 2 / when radard published none.
  A vision-only slot whose filtered prob is below MOONPILOT_LEAD_PROB_GATE comes back as the empty
  trajectory. A fused one is never gated here: its prob is radard's own filtered number, and radard
  only publishes a lead it already believes.
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
    # Vision-only: no fused state to anchor to, so publish in the raw radarState conventions, and
    # this is the one branch that can be published as empty -- the model's own prob is the only
    # confidence anyone here has, and the UI drew it and the planner scaled its time gap by its
    # inPath. `present=False` is the upstream default everywhere downstream, so the gate only
    # removes the fork's own state. The low-speed override above is exempt for the mirror reason:
    # it is a track radard saw, not a slot the model rates.
    prob = _filtered_prob(float(model_lead.prob), prob_filter)
    if prob < MOONPILOT_LEAD_PROB_GATE:
      return _empty(slot, in_path_filter)
    x = model_x - MOONPILOT_RADAR_TO_CAMERA
    y = -model_y

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
