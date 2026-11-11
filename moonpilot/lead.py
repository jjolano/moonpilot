"""Normalization for the vision model's lead trajectories.

The module owns per-slot filters and acceleration-estimator state, plus normalization used by the
publisher, planner and onroad renderers.

Frame and sign: published `x` is meters forward of the front bumper (radarState.dRel convention)
and published `y` is LEFT POSITIVE (radarState.yRel convention). modelV2's device frame is
right-positive, so `y` is negated on the way out.
"""

import math
from collections import deque
from types import SimpleNamespace

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.features import LEAD_LATERAL, enabled
from moonpilot.state import moonpilot_state

# Mirrors RADAR_TO_CAMERA in radard without importing its messaging/opendbc dependencies.
MOONPILOT_RADAR_TO_CAMERA = 1.52  # m; radar is ~1.5m ahead of the camera mesh frame

# Starting points, all fork-owned. Tune against logs: see the lead path line in the UI and
# LeadTrajectory.inPathProb in PlotJuggler.
MOONPILOT_PATH_HALF_WIDTH = 1.8  # m; half a typical lane
MOONPILOT_MIN_SPEED_FOR_YAW = 1.0  # m/s; below this, yawRel is meaningless -> 0.0
MOONPILOT_MIN_Y_STD = 0.05  # m; floor on yStd so the Gaussian never collapses
MOONPILOT_OUT_OF_PATH_DANGER = 0.4  # relaxed MPC danger factor for a fully out-of-path lead
MOONPILOT_INPATH_RC = 1.0  # s; decay time constant of the asymmetric inPath filter
# Floor on the lateral std in the 1/yStd^2 weights of `lead_in_path` -- the weights only; each sample's own corridor
# probability still uses its true std. Measured on 14 local routes (98k frames with an in-lane radar lead, the model's own path
# as reference): the model's 4 s knot sits a median 4.2 m off an in-lane lead's position with a median yStd of 0.17 m, against
# 3.0 m at the 2 s knot, so inverse-variance weighting put 0.64 of the weight on that biased knot (0.92 at 60-90 m, where the
# first knot's std is wider). The unfiltered inPath fell below the latch's ENTER on 73 % of in-lane 60-90 m frames and the planner
# was latched out of path on 67 % of them (t_follow 0.81 of the personality's). With the floor at 1.0 m: 1.5 % and 0.7 %. An
# adjacent lead still reads out of path (unfiltered below ENTER on 59 % of 60-90 m frames, against 75 % unfloored), and a floor
# of 0.5 m leaves the in-lane figures at 2.4 % / 1.4 %. `python -m moonpilot.lab inpath --routes longitudinal manual-approach`
# prints the table.
MOONPILOT_INPATH_YSTD_FLOOR = 1.0  # m
# Dense where a cut-in matters, none of it past the ~4 s where a cut-in can still be avoided.
MOONPILOT_INPATH_HORIZON = 4.0  # s; the last sample the grid is evaluated at, and so the last one
# the planner can act on. The renderers stop their lead line here too: past it the model's own
# yStd is ~1 m and the drawn 10 s tail landed beyond the white path's own 100 m cap.
MOONPILOT_INPATH_GRID = np.arange(0.0, MOONPILOT_INPATH_HORIZON + 1e-9, 0.25)
# The renderers' lead path line. Width, not alpha, carries the model's uncertainty -- a faint band at a
# fixed 1 m read as a squiggle rather than as a measurement, and yStd is the model's own answer to
# "how much do I know here".
MOONPILOT_LEAD_PATH_PX_PER_M = 8.0  # px of line width per meter of yStd
MOONPILOT_LEAD_PATH_WIDTH = (2.0, 18.0)  # px clamp
# The model's own confidence in a slot, below which normalize_lead publishes it as empty rather
# than as a lead. radard uses the same threshold after its asymmetric lead-probability filter;
# slot 0 crosses it in a median 4 frames, 29 % of which are single-frame runs (47k-frame corpus).
MOONPILOT_LEAD_PROB_GATE = 0.5
MOONPILOT_LEAD_PROB_RC = 0.2  # s; matches radard's lead_prob_filters decay
# A 15-sample window rather than 7 reduced command jitter from 0.090 to 0.082 m/s^2 RMS over 340
# closing-approach segments; closed-loop braking thresholds moved by at most 0.05 s and gaps by
# 0.02 m. Track IDs changed 3.1/s, 98 % for the same vehicle; the window held >7 samples on 42 %
# of those frames, lowering jitter 0.096 -> 0.077 m/s^2. Keeping same-car history reduced stop
# jitter 0.090 -> 0.056 m/s^2 on the stopped-track route and 0.0430 -> 0.0410 over 5.5k frames.
MOONPILOT_LEAD_ACCEL_WINDOW = 15  # samples, 0.75 s at DT_MDL
MOONPILOT_LEAD_ACCEL_MIN_SAMPLES = 3  # below this the slope is noise, so radard's value stands
MOONPILOT_LEAD_SPEED_JUMP = 2.5  # m/s in one frame: re-association, not motion (50 m/s^3)
# One vehicle under two radar tracks: the 517/541 pair above sat within 0.1 m of range, 0.25 m/s of
# speed and 0.5 m of lateral offset. Lateral is what tells two cars side by side apart — equal range
# and speed is exactly what an adjacent-lane car pacing the lead has, and its accel is not the lead's.
# Over 3.1k same-range, same-speed track pairs on hand (slot pairs and swaps), |dyRel| ran median 0.1 m,
# p99 0.8 m; cars in adjacent lanes sit a lane apart (~3.5 m), and anything outside the bound stays
# two leads.
MOONPILOT_LEAD_SAME_CAR_GAP = 1.0  # m of dRel
MOONPILOT_LEAD_SAME_CAR_LATERAL = 1.0  # m of yRel
MOONPILOT_LEAD_SAME_CAR_SPEED = 1.0  # m/s of vLead
# The onset window. The full window's bias is what a braking lead costs us: at a step onto -3.5 m/s^2
# the fifteen-sample slope needs 0.40 s to pass -2.0 and 0.55 s to pass -3.0, because that is how long
# the window is still half full of pre-onset samples — and a real brake ramps in over a few tenths, so
# the lead's own onset is what the estimate is chasing. The newest three samples read the step in
# 0.10 s, and the four-frame streak below believes them at 0.20 s. Reading both and believing the short
# one only when it is persistently deeper is one-sided by
# construction (the policy only ever uses a_lead to add braking — `min(a_lead, 0.0)` against
# `moonpilot/longitudinal.py`'s `MOONPILOT_LEAD_PREVIEW_T`), and the streak is what keeps radar noise
# out of the command:
# a single noisy frame cannot flip it, while an onset stays deep for many. Measured on 18,350
# steady-lead frames of the 340 local segments (a +-1 s centered slope under 0.1 m/s^2 as the
# reference), the onset window (at a two-frame streak) moves the estimate's error from 0.284 to
# 0.307 m/s^2 sd and a spurious read past -1 m/s^2 from 0.68 % to 1.00 % of frames. Closed loop, a lead braking at
# -3.5 m/s^2 from a settled 20 m/s follow draws -0.5 / -1.0 / -2.0 m/s^2 at 0.30 / 0.55 / 0.90 s with
# or without it: the lead's falling speed reaches the command through the other terms as fast. What
# the window buys is the published plan's tail and the FCW, both of which read `a_lead` directly.
MOONPILOT_LEAD_ACCEL_FAST = 3  # samples in the onset window, 0.10 s at DT_MDL
MOONPILOT_LEAD_ACCEL_FAST_MARGIN = 1.0  # m/s^2 the onset window must read deeper than the full one
MOONPILOT_LEAD_ACCEL_FAST_STREAK = 4  # consecutive frames the margin must hold before it is used
# Four, not two: a single `vLead` blip-and-return (a +0.2 m/s sample that is gone the next frame) sits
# inside the three-sample window for three frames, and its falling edge read -1.0..-1.6 m/s^2 for two or
# three of them. Those reads do reach the command — through `v_lead_eff` and the lead's projected
# state — as single-frame brake taps while following. On route 000003f8 segments 42-45 (2.0 min of
# engaged steady follow) streak 2 gave 69 frames past -0.8 m/s^2 and the command's high-frequency sd
# 0.141; streak 4 gives 35 and 0.114, within 0.002 of having no onset window at all, for 0.10 s more
# onset latency on the plan tail and FCW.
# Radard's accel-decay model is mirrored to keep this module lightweight.
MOONPILOT_LEAD_ACCEL_TAU = 1.5  # s; the decay a lead believed to be holding its accel gets
MOONPILOT_LEAD_ACCEL_TAU_RC = 0.45  # s; time constant of the filter that gives that belief up
MOONPILOT_LEAD_ACCEL_TAU_RESET = 0.5  # m/s^2; |a| above this is braking, not a transient
# Low-passed acceleration is used only by the planner's comfort response.
MOONPILOT_LEAD_ACCEL_SLOW_RC = 1.0  # s

LEAD_T_IDXS = ModelConstants.LEAD_T_IDXS
LEAD_T_OFFSETS = ModelConstants.LEAD_T_OFFSETS

class LeadSlot:
  def __init__(self, slot: int):
    self.slot = slot
    self.in_path_filter = FirstOrderFilter(1.0, MOONPILOT_INPATH_RC, DT_MDL)
    self.prob_filter = FirstOrderFilter(0.0, MOONPILOT_LEAD_PROB_RC, DT_MDL)

  def normalize(self, model_lead, fused_lead, ego_path_x, ego_path_y) -> dict:
    return normalize_lead(self.slot, model_lead, fused_lead, ego_path_x, ego_path_y, self.in_path_filter, self.prob_filter)

RADAR_ONLY_MODEL_PROB = 0.0


def same_car(a, b) -> bool:
  """Whether two radar leads (anything with `dRel`, `yRel`, `vLead`) are one vehicle under two radar
  tracks: close in range, lateral offset and speed at once. Ambiguous is two vehicles."""
  return (
    abs(a.dRel - b.dRel) < MOONPILOT_LEAD_SAME_CAR_GAP
    and abs(a.yRel - b.yRel) < MOONPILOT_LEAD_SAME_CAR_LATERAL
    and abs(a.vLead - b.vLead) < MOONPILOT_LEAD_SAME_CAR_SPEED
  )


class LeadAccelEstimator:
  """A lead's acceleration from the slope of `vLead` over a short window.

  radard reports `aLeadK` from `KalmanParams`' KF1D gains: tau is 0.49 s at DT_MDL, so
  90 % of a -3 m/s^2 step arrives 1.20 s late, which is most of the margin a
  braking lead gives us. The least-squares slope over MOONPILOT_LEAD_ACCEL_WINDOW samples has no lag
  of its own — a clean ramp reads exactly as soon as MOONPILOT_LEAD_ACCEL_MIN_SAMPLES are on it,
  0.10 s — so the only latency it carries is the ramp onto a window that already holds samples from
  before the step: 0.70 s at the full window. The onset window is what shortens that: the newest
  MOONPILOT_LEAD_ACCEL_FAST samples are read beside the full window and believed only when they are
  persistently deeper, which reaches -2.0 m/s^2 in 0.20 s on a step where the full window needs
  0.40 s, and -3.0 in 0.20 s where it needs 0.55. It averages measurement noise rather than lagging it:
  0.06 m/s^2 of jitter over the full window for 0.05 m/s of noise, against 1.41 m/s^2 for a one-frame
  difference. The lead's *speed* is left alone — `vLead` has no filter lag to remove, and this
  fit's endpoint value would add a transient bias exactly at the onset of braking.

  Returns radard's own `aLeadK` and `aLeadTau` whenever the window cannot speak: a vision-only lead
  (there `aLeadK` is the model's own unfiltered accel, paired with its own 0.3 s decay), fewer than
  MOONPILOT_LEAD_ACCEL_MIN_SAMPLES samples, a new vehicle (a new `radarTrackId` that is not `same_car`), a source flip, or a `vLead` jump
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
  absent — the reset is what keeps a new lead from inheriting the previous one's samples, and
  `same_car` is what keeps a new track on the same vehicle from being a new lead. Sample
  spacing is one model frame by construction: radard publishes `radarState` once per `modelV2`
  frame (`radard.py:263-272`), and plannerd is polled on the same message.
  """

  def __init__(self, dt: float = DT_MDL, fast_streak: int = MOONPILOT_LEAD_ACCEL_FAST_STREAK):
    self._samples: deque[float] = deque(maxlen=MOONPILOT_LEAD_ACCEL_WINDOW)
    self._track: tuple[bool, bool, int] | None = None
    self._last = None
    self._fast_streak = 0
    self._fast_streak_required = fast_streak
    self._tau = FirstOrderFilter(MOONPILOT_LEAD_ACCEL_TAU, MOONPILOT_LEAD_ACCEL_TAU_RC, dt)
    self._slow = FirstOrderFilter(0.0, MOONPILOT_LEAD_ACCEL_SLOW_RC, dt)
    self.a_lead_tau = MOONPILOT_LEAD_ACCEL_TAU
    self.a_slow = 0.0
    self._weights = {n: (np.arange(n) - (n - 1) / 2.0) * (12.0 / (dt * n * (n * n - 1)))
                     for n in range(MOONPILOT_LEAD_ACCEL_MIN_SAMPLES, MOONPILOT_LEAD_ACCEL_WINDOW + 1)}

  def update(self, lead) -> tuple[float, float]:
    a_lead, a_lead_tau = self._estimate(lead)
    if lead.present and math.isfinite(a_lead):
      self.a_slow = self._slow.update(a_lead)
    else:  # an empty slot, or a vision lead's NaN `aLeadK`: start over rather than carry a NaN forward
      self._slow.x = self.a_slow = 0.0
    return a_lead, a_lead_tau

  def _estimate(self, lead) -> tuple[float, float]:
    track = (bool(lead.present), bool(lead.radar), int(lead.radarTrackId))
    v_lead = float(lead.vLead)
    same = track[:2] == (True, True) and self._track is not None and self._track[:2] == track[:2] and self._last is not None and same_car(self._last, lead)
    if (track != self._track and not same) or (self._samples and abs(v_lead - self._samples[-1]) > MOONPILOT_LEAD_SPEED_JUMP):
      self._samples.clear()
      self._fast_streak = 0
      # A new lead's decay is new too: radard builds a fresh `Track` — and a fresh filter — for a new
      # `radarTrackId`, so carrying the old track's value over would tell the rollout that this lead's
      # accel holds for a length the previous lead earned. A new track on the same vehicle is not a new
      # lead, so it keeps both.
      self._tau.x = MOONPILOT_LEAD_ACCEL_TAU
      self._slow.x = 0.0
    self._track = track
    self._last = SimpleNamespace(dRel=float(lead.dRel), yRel=float(lead.yRel), vLead=v_lead)

    # No window to read from: hand back radard's own pair, which is self-consistent by construction.
    if not (lead.present and lead.radar):
      return float(lead.aLeadK), float(lead.aLeadTau)
    self._samples.append(v_lead)
    if len(self._samples) < MOONPILOT_LEAD_ACCEL_MIN_SAMPLES:
      return float(lead.aLeadK), float(lead.aLeadTau)

    n = len(self._samples)
    samples = np.fromiter(self._samples, float, n)
    a_lead = float(self._weights[n] @ samples)

    if n >= MOONPILOT_LEAD_ACCEL_FAST:
      a_fast = float(self._weights[MOONPILOT_LEAD_ACCEL_FAST] @ samples[-MOONPILOT_LEAD_ACCEL_FAST:])
      self._fast_streak = self._fast_streak + 1 if a_fast < a_lead - MOONPILOT_LEAD_ACCEL_FAST_MARGIN else 0
      if self._fast_streak >= self._fast_streak_required:
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
    dy = yi - (-float(np.interp(xi, ego_path_x, ego_path_y)))
    s = max(float(y_std[i]), MOONPILOT_MIN_Y_STD) if i < y_std.size else MOONPILOT_MIN_Y_STD
    w = MOONPILOT_PATH_HALF_WIDTH
    probs.append(0.5 * (math.erf((w - dy) / (math.sqrt(2.0) * s)) + math.erf((w + dy) / (math.sqrt(2.0) * s))))
  return probs


def lead_in_path(t, x, y, y_std, ego_path_x, ego_path_y, filt) -> float:
  """One scalar in-path probability, inverse-variance weighted and asymmetrically filtered.

  `filt` is a caller-owned FirstOrderFilter whose state must persist across frames. Weighting by
  1/yStd^2 lets the samples the model is confident about dominate, but only down to
  MOONPILOT_INPATH_YSTD_FLOOR: below it the weights are flat, so one far knot with a small std and a
  biased position cannot outvote the near samples. Rising is instant (stock behavior returns
  immediately), decaying takes about MOONPILOT_INPATH_RC of consistent evidence. Both directions fail safe.
  """
  if len(t) < 2 or len(x) < 2 or len(y) < 2 or len(ego_path_x) == 0 or len(ego_path_y) == 0:
    filt.x = 1.0
    return 1.0

  # Linear on purpose: the model's grid is 2 s apart, and a cubic through two points manufactures sub-2 s structure
  # the network never predicted.
  x_d = np.interp(MOONPILOT_INPATH_GRID, t, x)
  y_d = np.interp(MOONPILOT_INPATH_GRID, t, y)
  s_d = np.interp(MOONPILOT_INPATH_GRID, t, y_std) if len(y_std) >= 2 else np.full(MOONPILOT_INPATH_GRID.shape, MOONPILOT_MIN_Y_STD)

  probs = np.array(lead_in_path_prob(x_d, y_d, s_d, ego_path_x, ego_path_y), dtype=float)
  s_weighted = np.maximum(s_d[: probs.size], MOONPILOT_INPATH_YSTD_FLOOR)
  weights = 1.0 / s_weighted**2
  raw = float(np.sum(probs * weights) / np.sum(weights))
  return rise_hold(raw, filt)


def _empty(slot: int, filt) -> dict:
  filt.x = 1.0
  return {
    "present": False,
    "prob": 0.0,
    "probTime": float(LEAD_T_OFFSETS[slot]),
    "source": "none",
    "t": [], "x": [], "y": [], "v": [], "a": [], "xStd": [], "yStd": [], "vStd": [], "aStd": [], "yawRel": [], "inPathProb": [],
    "inPath": 1.0,
  }

def rise_hold(raw: float, filt) -> float:
  if raw > filt.x:
    filt.x = raw
  else:
    filt.update(raw)
  return float(filt.x)



def normalize_lead(slot, model_lead, fused_lead, ego_path_x, ego_path_y, in_path_filter, prob_filter) -> dict:
  """One leadsV3 slot as a LeadTrajectory dict on the model's native 6-point t grid.

  A vision slot below `MOONPILOT_LEAD_PROB_GATE` is published as empty, the upstream fail direction.
  """
  if model_lead is None or len(model_lead.x) == 0 or len(ego_path_x) == 0:
    return _empty(slot, in_path_filter)

  model_x = np.asarray(model_lead.x, dtype=float)
  model_y = np.asarray(model_lead.y, dtype=float)

  # The low-speed override substitutes a bare radar track that does not correspond to this slot,
  # so it has no predicted trajectory at all: one point, straight from the fused state.
  if fused_lead is not None and fused_lead.present and fused_lead.radar and fused_lead.modelProb == RADAR_ONLY_MODEL_PROB:
    lead = _empty(slot, in_path_filter)
    lead.update({
      "present": True, "prob": float(fused_lead.modelProb), "source": "radar",
      "t": [0.0], "x": [float(fused_lead.dRel)], "y": [float(fused_lead.yRel)],
      "v": [float(fused_lead.vLead)], "a": [float(fused_lead.aLeadK)],
    })
    return lead

  if fused_lead is not None and fused_lead.present:
    # Anchor the predicted shape to the fused t=0 state, so index 0 equals radarState exactly.
    x = float(fused_lead.dRel) + (model_x - model_x[0])
    y = float(fused_lead.yRel) + (-model_y + model_y[0])
    prob = float(fused_lead.modelProb)
  else:
    prob = rise_hold(float(model_lead.prob), prob_filter)
    if prob < MOONPILOT_LEAD_PROB_GATE:
      return _empty(slot, in_path_filter)
    x = model_x - MOONPILOT_RADAR_TO_CAMERA
    y = -model_y

  t = np.asarray(model_lead.t, dtype=float)
  y_std = np.asarray(model_lead.yStd, dtype=float)

  lead = _empty(slot, in_path_filter)
  lead.update({
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
  })
  return lead


def _moonpilot_leads(sm):
  state = moonpilot_state(sm)
  return None if state is None else state.leads


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


