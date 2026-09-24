"""Corridor free-space strip from the model's road edges (Phase 2 of true SLAM).

`modelV2.roadEdges`: two polylines on `ModelConstants.X_IDXS` (0…192 m), device frame,
y right-positive (`moonpilot/lead.py` negates that on the way to radar's left-positive).
This module answers one question at each sample along the ego path: how much free lateral
space is there, and is the path inside it. Pure numpy — no messaging, no toggle — so a
replay or a future planner seam can call it the same way `lead_in_path` is called.

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

import numpy as np

from opendbc.car.interfaces import ACCEL_MAX

# Fork-owned. Tune against logs: plot corridor width vs. the path in PlotJuggler.
MOONPILOT_CORRIDOR_MARGIN = 0.15  # m; path this close to an edge still counts as inside
# Squeeze braking (`MoonpilotSqueeze`): free width under FULL over the next LOOKAHEAD metres of
# known edge adds a bounded cruise-slot candidate. FULL is ~2 car widths (CarParams has no width
# field, so this is a fixed stand-in for a typical ~2 m body); MIN is one body, where the floor
# already applies. Lookahead is the model's near horizon, not the full 192 m edge span — far
# samples are less certain and not worth slowing for. Floor matches the curve terms.
MOONPILOT_SQUEEZE_LOOKAHEAD = 40.0  # m ahead of the car (after the path re-reference)
MOONPILOT_SQUEEZE_FULL_WIDTH = 4.0  # m; below this the corridor is under ~2 car widths
MOONPILOT_SQUEEZE_MIN_WIDTH = 2.0  # m; at or under one car body the floor is already applied
MOONPILOT_SQUEEZE_ACCEL_MIN = -1.5  # m/s^2; same bound as the curve terms, well above ACCEL_MIN


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
  """Bounded braking for a corridor that pinches over the next `MOONPILOT_SQUEEZE_LOOKAHEAD` metres.

  `sample_x` is the edges' own x frame (the model path's absolute distances); `x_ego` is how far
  the car has already covered of that frame — the same re-reference the curve targets take, so the
  window is metres *ahead of the car now*. Deepest known finite width in the window wins. Wide
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
