"""Corridor free-space strip from the model's road edges (Phase 2 of true SLAM).

`modelV2.roadEdges`: two polylines on `ModelConstants.X_IDXS` (0…192 m), device frame,
y right-positive (`moonpilot/lead.py` negates that on the way to radar's left-positive).
This module answers one question at each sample along the ego path: how much free lateral
space is there, and is the path inside it. The geometry helpers are pure numpy — no
messaging — so a replay or a planner seam can call them the same way `lead_in_path` is
called. `path_outside_alert` is the one messaging-facing wrapper (selfdrived's banner).

Three things that are not obvious:

- **Edge index is not a side.** `roadEdges[0]` pairs with `laneLines[1]` (the left lane)
  in the mici renderer, but a swapped model or a mirrored calib would flip it. Bounds are
  ordered by y at each x (`min`/`max`), never by index, so left/right is the frame's own
  answer.
- **Occupancy is a path test, not a grid.** There is no persistent map: each call answers
  only about the path samples it is given, over the horizon the model already predicted.
  A point is in-corridor when its y lies between the two edge y's at that x, with a small
  lateral margin so a path painted on the edge itself is still "inside".
- **A sample the edges do not span is unknown, not free.** Interp outside the edge x-range
  is refused (`left`/`right` stay NaN); callers must treat NaN as no answer rather than as
  an open road. The model's far end and a clipped edge both land here.
"""

from collections import deque

import numpy as np

from opendbc.car.interfaces import ACCEL_MAX

# Fork-owned. Tune against logs: plot corridor width vs. the path in PlotJuggler.
MOONPILOT_CORRIDOR_MARGIN = 0.15  # m; path this close to an edge still counts as inside
# Squeeze braking (`MoonpilotSqueeze`): free width under FULL over the next LOOKAHEAD meters of
# known edge adds a bounded cruise-slot candidate. FULL is ~2 car widths (CarParams has no width
# field, so this is a fixed stand-in for a typical ~2 m body); MIN is one body, where the floor
# already applies. Lookahead is the model's near horizon, not the full 192 m edge span — far
# samples are less certain and not worth slowing for. Floor matches the curve terms.
MOONPILOT_SQUEEZE_LOOKAHEAD = 40.0  # m ahead of the car (after the path re-reference)
MOONPILOT_SQUEEZE_FULL_WIDTH = 4.0  # m; below this the corridor is under ~2 car widths
MOONPILOT_SQUEEZE_MIN_WIDTH = 2.0  # m; at or under one car body the floor is already applied
MOONPILOT_SQUEEZE_ACCEL_MIN = -1.5  # m/s^2; same bound as the curve terms, well above ACCEL_MIN
# Path-outside banner (`MoonpilotPathOutside`): among known path samples in the near horizon,
# this fraction outside the free corridor is enough to warn. One sample is model noise; a quarter
# of the near path is a trajectory that actually leaves the road. Lookahead matches squeeze so
# the banner and the brake look at the same stretch of road.
MOONPILOT_PATH_OUTSIDE_LOOKAHEAD = 40.0  # m
MOONPILOT_PATH_OUTSIDE_MIN_FRAC = 0.25  # of known samples in the lookahead that must be outside
# Pose-stitched history (`egoPose` frame, Phase 3): how long past free-space strips are kept and
# warped into the current device frame. Short enough that a hard cut or a rebase cannot leave
# stale geometry in the intersection; long enough to ride out a single frame's edge dropout.
MOONPILOT_CORRIDOR_HISTORY_S = 1.5  # s
MOONPILOT_CORRIDOR_HISTORY_MAX = 40  # frames; 1.5 s at 20 Hz is 30, this is the hard cap


def _xy(edge):
  """`(x, y)` float arrays from either an `XYZTData`-like object or a `(x, y)` pair.

  The pair form is evaluated only when the object has no `.x` — a bare `getattr(edge, "x",
  edge[0])` would subscript a capnp struct eagerly and raise before the fallback ran.
  """
  if hasattr(edge, "x"):
    return np.asarray(edge.x, dtype=float), np.asarray(edge.y, dtype=float)
  return np.asarray(edge[0], dtype=float), np.asarray(edge[1], dtype=float)


def corridor_bounds(edge_a, edge_b, sample_x):
  """Free lateral bounds at each `sample_x`.

  `edge_*` are `(x, y)` sequences (one `XYZTData` road edge each). Returns `(left, right)`
  float arrays in the device frame's y (right-positive → left is the smaller y), NaN where
  `sample_x` is outside both edges' x-span or either edge is empty/too short to interp.
  """
  x_a, y_a = _xy(edge_a)
  x_b, y_b = _xy(edge_b)
  sample_x = np.asarray(sample_x, dtype=float)

  empty = x_a.size < 2 or y_a.size != x_a.size or x_b.size < 2 or y_b.size != x_b.size
  if empty or sample_x.size == 0:
    nan = np.full(sample_x.shape, np.nan)
    return nan, nan.copy()

  lo_x = max(float(x_a.min()), float(x_b.min()))
  hi_x = min(float(x_a.max()), float(x_b.max()))
  inside = (sample_x >= lo_x) & (sample_x <= hi_x)

  ya = np.interp(sample_x, x_a, y_a, left=np.nan, right=np.nan)
  yb = np.interp(sample_x, x_b, y_b, left=np.nan, right=np.nan)
  left = np.minimum(ya, yb)
  right = np.maximum(ya, yb)
  left = np.where(inside, left, np.nan)
  right = np.where(inside, right, np.nan)
  return left, right


def path_in_corridor(path_x, path_y, edge_a, edge_b, margin=MOONPILOT_CORRIDOR_MARGIN):
  """Per-sample in-corridor flag for the ego path: 1.0 inside (with margin), 0.0 outside, NaN unknown.

  `path_x`/`path_y` are the model's own path (or any polyline in the same frame). A sample
  whose edges are unknown stays NaN so a consumer can refuse to act rather than assume free.
  """
  path_x = np.asarray(path_x, dtype=float)
  path_y = np.asarray(path_y, dtype=float)
  if path_x.size == 0 or path_y.size != path_x.size:
    return np.full(path_x.shape, np.nan)

  left, right = corridor_bounds(edge_a, edge_b, path_x)
  known = np.isfinite(left) & np.isfinite(right) & np.isfinite(path_y)
  inside = (path_y >= left - margin) & (path_y <= right + margin)
  out = np.full(path_x.shape, np.nan)
  out[known] = np.where(inside[known], 1.0, 0.0)
  return out


def corridor_width(edge_a, edge_b, sample_x):
  """Free width (right − left) at each `sample_x`, NaN where unknown or crossed."""
  left, right = corridor_bounds(edge_a, edge_b, sample_x)
  width = right - left
  return np.where(np.isfinite(width) & (width >= 0.0), width, np.nan)


def squeeze_accel(edge_a, edge_b, sample_x, x_ego=0.0):
  """Bounded braking for a corridor that pinches over the next `MOONPILOT_SQUEEZE_LOOKAHEAD` meters.

  `sample_x` is the edges' own x frame (the model path's absolute distances); `x_ego` is how far
  the car has already covered of that frame — the same re-reference the curve targets take, so the
  window is meters *ahead of the car now*. Deepest known finite width in the window wins. Wide
  enough, unknown, or empty → `ACCEL_MAX` (the inactive sentinel `policy` leaves the cruise term
  alone for); otherwise a linear ramp from 0 at `FULL_WIDTH` down to `ACCEL_MIN` (this module's
  squeeze floor) at `MIN_WIDTH` and no deeper. Pure geometry: no toggle, no messaging."""
  sample_x = np.asarray(sample_x, dtype=float)
  d = sample_x - float(x_ego)
  window = (d >= 0.0) & (d <= MOONPILOT_SQUEEZE_LOOKAHEAD)
  if not window.any():
    return ACCEL_MAX
  w = corridor_width(edge_a, edge_b, sample_x[window])
  known = np.isfinite(w)
  if not known.any():
    return ACCEL_MAX
  w_min = float(np.min(w[known]))
  if w_min >= MOONPILOT_SQUEEZE_FULL_WIDTH:
    return ACCEL_MAX
  span = MOONPILOT_SQUEEZE_FULL_WIDTH - MOONPILOT_SQUEEZE_MIN_WIDTH
  frac = float(np.clip((MOONPILOT_SQUEEZE_FULL_WIDTH - w_min) / span, 0.0, 1.0))
  return MOONPILOT_SQUEEZE_ACCEL_MIN * frac


def path_outside_fraction(path_x, path_y, edge_a, edge_b, lookahead=MOONPILOT_PATH_OUTSIDE_LOOKAHEAD):
  """Fraction of *known* path samples in `[0, lookahead]` that sit outside the free corridor.

  Unknown samples (edges do not span that x) are excluded from both numerator and denominator:
  a path the model has not painted edges for is not evidence the path left the road. Returns
  0.0 when nothing in the window is known — no answer, not free.
  """
  path_x = np.asarray(path_x, dtype=float)
  path_y = np.asarray(path_y, dtype=float)
  if path_x.size == 0 or path_y.size != path_x.size:
    return 0.0
  flags = path_in_corridor(path_x, path_y, edge_a, edge_b)
  window = (path_x >= 0.0) & (path_x <= lookahead)
  known = window & np.isfinite(flags)
  n_known = int(np.count_nonzero(known))
  if n_known == 0:
    return 0.0
  return float(np.count_nonzero(flags[known] == 0.0)) / n_known


def path_outside_alert(sm, params=None) -> int | None:
  """selfdrived's banner: `EventName.pathOutside` when the model path leaves the free corridor.

  Same shape as `turn_desire_alert`: feature gate, then the one thing modeld does not check —
  `carControl.latActive`, so a disengaged car (or half-engagement latched off) does not get a
  banner for a path nobody is following. Distinct from LDW (lane lines / desire) and from the
  turn-desire banner (blinker): this is the model's own trajectory leaving `roadEdges`.
  """
  from openpilot.cereal import log
  from openpilot.common.params import Params
  from moonpilot.features import PATH_OUTSIDE, enabled

  if params is None:
    params = Params()
  if not enabled(PATH_OUTSIDE, params):
    return None
  if not sm["carControl"].latActive:
    return None
  model = sm["modelV2"]
  edges = model.roadEdges
  if len(edges) < 2:
    return None
  frac = path_outside_fraction(model.position.x, model.position.y, edges[0], edges[1])
  if frac >= MOONPILOT_PATH_OUTSIDE_MIN_FRAC:
    return log.OnroadEvent.EventName.pathOutside
  return None


def _warp_xy(x, y, from_pose, to_pose):
  """Device-frame `(x, y)` from `from_pose`'s frame into `to_pose`'s frame.

  Poses are `(x, y, yaw)` in the GPS-seeded ENU frame (`moonpilotState.egoPose`). Relative
  only: the absolute origin never enters, so a rebase that jumps the origin mid-history
  leaves every stored strip inconsistent with the new one and the caller must clear.
  """
  fx, fy, fyaw = from_pose
  tx, ty, tyaw = to_pose
  cf, sf = np.cos(fyaw), np.sin(fyaw)
  wx = fx + cf * x - sf * y
  wy = fy + sf * x + cf * y
  ct, st = np.cos(tyaw), np.sin(tyaw)
  dx, dy = wx - tx, wy - ty
  return ct * dx + st * dy, -st * dx + ct * dy


class RollingCorridor:
  """Free-space strips held in the `egoPose` frame and queried in the current one (Phase 3).

  The model repaints `roadEdges` every frame; a strip that is solid for half a second and
  missing for one frame should not release squeeze or clear the path-outside banner on that
  frame. Each push stores the strip with the pose that frame was captured in; a query warps
  every strip still inside the history window into the caller's pose and intersects the free
  regions (max of lefts, min of rights — the narrowest strip any frame observed). Unknown
  samples stay unknown: intersection only tightens where both frames had an answer.

  Requires a valid `egoPose` on every push and query. An invalid pose clears the history —
  a rebase or a slam-toggle-off mid-window would otherwise mix frames from two origins.
  """

  def __init__(self) -> None:
    # (mono_s, pose, edge_a, edge_b); pose is (x, y, yaw) in ENU.
    self._hist: deque = deque()

  def clear(self) -> None:
    self._hist.clear()

  def push(self, mono_s: float, pose, edge_a, edge_b, now: float | None = None) -> None:
    """Record one frame's free-space strip at `pose`. `now` defaults to `mono_s` (no expiry yet)."""
    if pose is None:
      self.clear()
      return
    t_now = mono_s if now is None else now
    self._hist.append((mono_s, tuple(pose), edge_a, edge_b))
    cutoff = t_now - MOONPILOT_CORRIDOR_HISTORY_S
    while self._hist and (self._hist[0][0] < cutoff or len(self._hist) > MOONPILOT_CORRIDOR_HISTORY_MAX):
      self._hist.popleft()

  def fused_bounds(self, pose, sample_x):
    """`(left, right)` after intersecting every strip in the window, warped to `pose`.

    `pose is None` or an empty history returns all-NaN (no answer). A single-frame history
    is exactly `corridor_bounds` on that frame's edges.
    """
    sample_x = np.asarray(sample_x, dtype=float)
    if pose is None or not self._hist:
      nan = np.full(sample_x.shape, np.nan)
      return nan, nan.copy()
    # Collect first so the accumulators are ndarrays from the start (ty cannot narrow
    # `left_acc is None` through the assignment in the loop body).
    strips = []
    for _, strip_pose, edge_a, edge_b in self._hist:
      ax, ay = _xy(edge_a)
      bxx, byy = _xy(edge_b)
      wx, wy = _warp_xy(ax, ay, strip_pose, pose)
      bx, by = _warp_xy(bxx, byy, strip_pose, pose)
      # np.interp needs increasing x; a hard yaw between frames can reverse a polyline.
      strips.append(corridor_bounds((wx[np.argsort(wx)], wy[np.argsort(wx)]),
                                    (bx[np.argsort(bx)], by[np.argsort(bx)]), sample_x))
    left_acc, right_acc = strips[0]
    # Free intersection: the strip can only get narrower. NaN (unknown) keeps the other
    # frame's answer rather than poisoning the intersection to unknown.
    for left, right in strips[1:]:
      left_acc = np.where(np.isfinite(left) & np.isfinite(left_acc), np.maximum(left_acc, left), left_acc)
      right_acc = np.where(np.isfinite(right) & np.isfinite(right_acc), np.minimum(right_acc, right), right_acc)
      left_acc = np.where(~np.isfinite(left_acc) & np.isfinite(left), left, left_acc)
      right_acc = np.where(~np.isfinite(right_acc) & np.isfinite(right), right, right_acc)
    return left_acc, right_acc

  def squeeze_accel(self, pose, sample_x, x_ego=0.0):
    """Same contract as the free function, over the pose-stitched intersection."""
    if pose is None or not self._hist:
      return ACCEL_MAX
    left, right = self.fused_bounds(pose, sample_x)
    sample_x = np.asarray(sample_x, dtype=float)
    d = sample_x - float(x_ego)
    window = (d >= 0.0) & (d <= MOONPILOT_SQUEEZE_LOOKAHEAD)
    if not window.any():
      return ACCEL_MAX
    w = right[window] - left[window]
    known = np.isfinite(w) & (w >= 0.0)
    if not known.any():
      return ACCEL_MAX
    w_min = float(np.min(w[known]))
    if w_min >= MOONPILOT_SQUEEZE_FULL_WIDTH:
      return ACCEL_MAX
    span = MOONPILOT_SQUEEZE_FULL_WIDTH - MOONPILOT_SQUEEZE_MIN_WIDTH
    frac = float(np.clip((MOONPILOT_SQUEEZE_FULL_WIDTH - w_min) / span, 0.0, 1.0))
    return MOONPILOT_SQUEEZE_ACCEL_MIN * frac

  def path_outside_fraction(self, pose, path_x, path_y, lookahead=MOONPILOT_PATH_OUTSIDE_LOOKAHEAD):
    """Same contract as the free function, over the pose-stitched intersection."""
    if pose is None or not self._hist:
      return 0.0
    path_x = np.asarray(path_x, dtype=float)
    path_y = np.asarray(path_y, dtype=float)
    if path_x.size == 0 or path_y.size != path_x.size:
      return 0.0
    left, right = self.fused_bounds(pose, path_x)
    known = (path_x >= 0.0) & (path_x <= lookahead) & np.isfinite(left) & np.isfinite(right) & np.isfinite(path_y)
    n_known = int(np.count_nonzero(known))
    if n_known == 0:
      return 0.0
    inside = (path_y >= left - MOONPILOT_CORRIDOR_MARGIN) & (path_y <= right + MOONPILOT_CORRIDOR_MARGIN)
    return float(np.count_nonzero(~inside[known])) / n_known
