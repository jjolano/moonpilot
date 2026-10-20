"""Does leadd's `inPath` call an in-lane lead out of path? The planner's `MoonpilotLeadLateral` latch acts on it.

The in-lane reference never touches leadd: the radar lead's lateral offset from the model's own path at the lead's range,
`|yRel + path_y(dRel + RADAR_TO_CAMERA)|` (radarState yRel is left-positive, modelV2.position.y right-positive, so the sum
is the offset; the header line prints the check that the sum, not the difference, is the one that vanishes on followed
leads). The claim under test is the logged `moonpilotState.leads[0].inPath`; the latch and time gap are the planner's own
`_t_follow`, run on the drive's rows.

Clock: seconds since the route's first log message, the one the drive reports quote.
"""
import math
from contextlib import contextmanager
from itertools import islice

import numpy as np
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.tools.lib.logreader import LogReader
from opendbc.car.structs import car

from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, runs
from moonpilot.lab import planner as lab_planner
from moonpilot.lead import MOONPILOT_INPATH_RC, MOONPILOT_MIN_Y_STD, MOONPILOT_PATH_HALF_WIDTH, MOONPILOT_RADAR_TO_CAMERA
from moonpilot.longitudinal import (
  MOONPILOT_OUT_OF_PATH_ENTER,
  MOONPILOT_OUT_OF_PATH_LEAVE,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_T_FOLLOW,
  MOONPILOT_T_FOLLOW_SLEW_DOWN,
  MOONPILOT_T_FOLLOW_SLEW_UP,
  MoonpilotLongitudinalPlanner,
)

EXTRACT_VERSION = 2
LATCH_VERSION = 1
ROUTE_SETS = {
  # The engaged drives, plus the one the report is about.
  "longitudinal": (*lab_planner.ROUTE_SETS["longitudinal"], "0000044d"),
  "manual-approach": lab_planner.ROUTE_SETS["manual-approach"],
}
N_PATH, N_LEAD = 33, 6  # modelV2.position's grid; a leadd slot's native t grid
# The grid the logged drives' leadd evaluated (`MOONPILOT_INPATH_HORIZON` was 4 s on every one): pinned here rather than imported,
# so the counterfactual stays a replay of those drives after the shipped horizon changes. The cached latch is code-dependent, so
# bump LATCH_VERSION when `_t_follow` or its constants change.
DRIVE_GRID = np.arange(0.0, 4.0 + 1e-9, 0.25)
LANE, ADJACENT = 1.0, (2.5, 5.0)  # m: the reference's in-lane bound, and its adjacent-lane band
BINS = (0.0, 30.0, 60.0, 90.0, 130.0)  # m of dRel
SPEEDS = (3.0, 10.0, 20.0, 30.0, 60.0)  # m/s of vEgo
MIN_SPEED = 3.0  # below it the planner's own low-speed lead handling owns the frame, not the lateral claim
SOURCES = {"none": 0, "vision": 1, "radar": 2}
SCALARS = (
  "t", "v_ego", "long_active", "lead", "d_rel", "y_rel", "radar_only", "personality",
  "mp_present", "mp_in_path", "mp_source", "mp_age",
)


def _build(segs, root, route):
  first = min(m.logMonoTime for m in islice(LogReader(segs[0], sort_by_time=False), 500))
  data = lab_planner.extract(route, root)
  blob, offsets = data["blob"], data["offsets"]
  index = {k: 1 + i * 3 for i, k in enumerate(lab_planner.MESSAGES)}
  cols = {k: [] for k in (*SCALARS, "lx", "ly", "lystd", "lt", "my", "mystd", "px", "py")}

  def pad(values, n):
    out = np.full(n, np.nan)
    out[: min(len(values), n)] = list(values)[:n]
    return out

  for row in offsets:
    def read(kind, row=row):
      start, size = row[index[kind]], row[index[kind] + 1]
      return messaging.log_from_bytes(blob[start:start + size].tobytes())
    cs, cc, lead = read("carState").carState, read("carControl").carControl, read("radarState").radarState.leadOne
    mp, model, sd = read("moonpilotState").moonpilotState, read("modelV2").modelV2, read("selfdriveState").selfdriveState
    slot = mp.leads[0] if len(mp.leads) else None
    s = {
      "t": row[0] * 1e-9, "v_ego": cs.vEgo, "long_active": cc.longActive, "lead": lead.present, "d_rel": lead.dRel, "y_rel": lead.yRel,
      "radar_only": lead.radar and lead.modelProb == 0.0, "personality": sd.personality.raw,
      "mp_present": slot is not None and slot.present, "mp_in_path": slot.inPath if slot is not None else 1.0,
      "mp_source": SOURCES[str(slot.source)] if slot is not None else 0, "mp_age": (row[0] - row[index["moonpilotState"] + 2]) * 1e-9,
    }
    for k in SCALARS:
      cols[k].append(float(s[k]))
    lists = (("lx", "x"), ("ly", "y"), ("lystd", "yStd"), ("lt", "t"))
    for k, field in lists:
      cols[k].append(pad(getattr(slot, field) if slot is not None else [], N_LEAD))
    # modelV2.leadsV3[0] as the network wrote it (y right-positive), beside what leadd republished from it
    ml = model.leadsV3[0] if len(model.leadsV3) else None
    cols["my"].append(pad(ml.y if ml is not None else [], N_LEAD))
    cols["mystd"].append(pad(ml.yStd if ml is not None else [], N_LEAD))
    cols["px"].append(pad(model.position.x, N_PATH))
    cols["py"].append(pad(model.position.y, N_PATH))
  return {**{k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}, "t0": np.float64(first * 1e-9)}


def extract(route, root=DEFAULT_ROOT):
  """One row per planner tick of the route's extract: the radar lead, leadd's slot 0, the model's path."""
  return cached(route, "inpath", EXTRACT_VERSION, root, lambda segs: _build(segs, root, route))


@contextmanager
def _replay_env(route, root):
  """The drive's own params and the row's clock under the planner module, as `planner.replay` sets them."""
  import moonpilot.longitudinal as longitudinal
  params = lab_planner.ReplayParams({k: v.encode() for k, v in drive_config(route, root)[1].items()})
  clock = lab_planner.ReplayClock()
  original = longitudinal.Params, longitudinal.time
  longitudinal.Params, longitudinal.time = lambda: params, clock  # ty: ignore[invalid-assignment]
  try:
    yield clock
  finally:
    longitudinal.Params, longitudinal.time = original


def _latch_build(_segs, root, route):
  """The planner's own `_t_follow` on every row of the drive: returns (t_follow, lead_leaving). Only that method runs,
  which is every state it has, so the numbers equal a full replay's at a hundredth of the cost."""
  data = lab_planner.extract(route, root)
  cp = messaging.log_from_bytes(data["cp"].tobytes(), car.CarParams)
  blob, offsets = data["blob"], data["offsets"]
  index = {k: 1 + i * 3 for i, k in enumerate(lab_planner.MESSAGES)}
  out = []
  with _replay_env(route, root) as clock:
    planner = MoonpilotLongitudinalPlanner(cp)
    for row in offsets:
      messages, stamps = {}, {}
      for kind in ("radarState", "selfdriveState", "moonpilotState"):
        start, size, stamp = row[index[kind]:index[kind] + 3]
        messages[kind], stamps[kind] = messaging.log_from_bytes(blob[start:start + size].tobytes()), int(stamp)
      clock.now_ns = int(row[0])
      out.append((planner._t_follow(lab_planner.ReplaySubMaster(messages, stamps, clock.now_ns)), planner.lead_leaving))
  return {"t_follow": np.asarray([o[0] for o in out]), "leaving": np.asarray([o[1] for o in out], dtype=bool)}


def latch(route, root=DEFAULT_ROOT):
  return cached(route, "inpath-latch", LATCH_VERSION, root, lambda segs: _latch_build(segs, root, route))


def pooled(routes, root=DEFAULT_ROOT):
  """Every route's frames end to end: `route` numbers them (episodes never span two), `el` is the report clock, and
  `off`, `path_y` and `agree` are the reference. `agree` marks frames where leadd's slot 0 is the radar's leadOne
  (same range and lateral at t=0), because the two messages are stamped apart and a lead swap falls between them."""
  frames = [(extract(r, root), latch(r, root)) for r in routes]
  d = {k: np.concatenate([f[k] for f, _ in frames]) for k in frames[0][0] if k not in ("segments", "t0")}
  d.update({k: np.concatenate([lt[k] for _, lt in frames]) for k in ("t_follow", "leaving")})
  d["route"] = np.concatenate([np.full(len(f["t"]), i) for i, (f, _) in enumerate(frames)])
  d["el"] = np.concatenate([f["t"] - f["t0"] for f, _ in frames])
  d["off"], d["path_y"] = offset(d)
  d["base"] = np.array([MOONPILOT_T_FOLLOW.get(int(p), MOONPILOT_T_FOLLOW[1]) for p in d["personality"]])
  d["agree"] = (np.abs(d["lx"][:, 0] - d["d_rel"]) < 0.5) & (np.abs(d["ly"][:, 0] - d["y_rel"]) < 0.25)
  d["valid"] = (
    (d["lead"] > 0) & (d["mp_present"] > 0) & (d["mp_source"] == 1) & d["agree"] & (d["mp_age"] < 0.1) & (d["v_ego"] > MIN_SPEED) & np.isfinite(d["off"])
  )
  return d


def offset(d):
  """The radar lead's lateral offset from the model's path at its range, and that path's y there (right-positive)."""
  x = d["d_rel"] + MOONPILOT_RADAR_TO_CAMERA
  path_y = np.array([np.interp(xi, px, py) if np.isfinite(px).all() else np.nan for xi, px, py in zip(x, d["px"], d["py"], strict=True)])
  return d["y_rel"] + path_y, path_y


# --- leadd's inPath, recomputed on stored rows so each input can be changed on its own --------------------------------

_erf = np.frompyfunc(math.erf, 1, 1)


def grid(d, rows):
  """Leadd's 0.25 s grid samples for `rows`: lead x, lead y (left-positive), yStd, and the ego path's y at x (left-positive)."""
  n = len(DRIVE_GRID)
  x, y, s, p = (np.zeros((len(rows), n)) for _ in range(4))
  for k, i in enumerate(rows):
    m = np.isfinite(d["lx"][i])
    t = d["lt"][i][m]
    x[k] = np.interp(DRIVE_GRID, t, d["lx"][i][m])
    y[k] = np.interp(DRIVE_GRID, t, d["ly"][i][m])
    s[k] = np.interp(DRIVE_GRID, t, d["lystd"][i][m])
    p[k] = -np.interp(x[k], d["px"][i], d["py"][i])
  return x, y, s, p


def raw_in_path(y, s, path, half_width=MOONPILOT_PATH_HALF_WIDTH, weights="inverse variance"):
  """`lead_in_path` before its filter, on `grid`'s arrays: the Gaussian-in-corridor probability per sample, then the mean
  weighted by 1/yStd^2 (or uniform)."""
  s = np.maximum(s, MOONPILOT_MIN_Y_STD)
  dy, scale = y - path, math.sqrt(2.0) * s
  p = 0.5 * ((_erf((half_width - dy) / scale) + _erf((half_width + dy) / scale)).astype(float))
  w = 1.0 / s**2 if weights == "inverse variance" else np.ones_like(s)
  return (p * w).sum(1) / w.sum(1)


def emulate_t_follow(raw, d, vision, in_path_rc=MOONPILOT_INPATH_RC):
  """leadd's rise-hold filter and the planner's latch + slew over a pooled frame set, given each frame's raw in-path
  (`raw[i]` is read only where `vision[i]`). Returns (t_follow, leaving). Mirrors `lead.lead_in_path`'s filter and
  `MoonpilotLongitudinalPlanner._t_follow`; `test_lab_inpath` pins the second against the real method."""
  alpha = DT_MDL / (in_path_rc + DT_MDL)
  n = len(raw)
  tf, leaving = np.empty(n), np.zeros(n, dtype=bool)
  x, latched, cur, prev_route = 1.0, False, None, -1
  for i in range(n):
    if d["route"][i] != prev_route:
      x, latched, cur, prev_route = 1.0, False, None, d["route"][i]
    if not d["mp_present"][i]:
      x = 1.0
    elif vision[i]:
      x = raw[i] if raw[i] > x else x + alpha * (raw[i] - x)
    in_path = x if d["mp_present"][i] and d["mp_source"][i] == 1 else 1.0
    base = d["base"][i]
    latched = in_path < (MOONPILOT_OUT_OF_PATH_LEAVE if latched else MOONPILOT_OUT_OF_PATH_ENTER)
    if not d["lead"][i]:
      cur, latched = base, False
    else:
      target = base * MOONPILOT_OUT_OF_PATH_T_FOLLOW if latched else base
      cur = target if cur is None else cur
      cur = max(target, cur - MOONPILOT_T_FOLLOW_SLEW_DOWN * DT_MDL) if target < cur else min(target, cur + MOONPILOT_T_FOLLOW_SLEW_UP * DT_MDL)
    tf[i], leaving[i] = cur, latched
  return tf, leaving


# --- reports -----------------------------------------------------------------------------------------------------------


def _classes(d):
  a = np.abs(d["off"])
  return {f"in lane (<{LANE:.1f} m)": a < LANE, f"adjacent ({ADJACENT[0]:.1f}-{ADJACENT[1]:.1f} m)": (a >= ADJACENT[0]) & (a < ADJACENT[1])}


def _summary(d, mask, in_path, tf, leaving):
  """One table row: the logged inPath's spread, how often it crosses the latch thresholds, how often the planner was latched,
  and what that did to the time gap (mean t_follow over the personality's own, and the mean gap shortfall at the car's speed)."""
  if mask.sum() < 20:
    return f"n={int(mask.sum()):6d}"
  x = in_path[mask]
  ratio = tf[mask] / d["base"][mask]
  short = (d["base"][mask] - tf[mask]) * d["v_ego"][mask]
  p10, p25, p90 = (np.percentile(x, q) for q in (10, 25, 90))
  enter, leave = np.mean(x < MOONPILOT_OUT_OF_PATH_ENTER), np.mean(x < MOONPILOT_OUT_OF_PATH_LEAVE)
  spread = f"n={int(mask.sum()):6d} inPath med {np.median(x):.2f} p10 {p10:.2f} p25 {p25:.2f} p90 {p90:.2f}"
  crossing = f"<{MOONPILOT_OUT_OF_PATH_ENTER:.2f} {enter:.3f} <{MOONPILOT_OUT_OF_PATH_LEAVE:.2f} {leave:.3f}"
  cost = f"latched {np.mean(leaving[mask]):.3f} t_follow/base mean {np.mean(ratio):.3f} min {np.min(ratio):.2f}"
  return f"{spread} | {crossing} | {cost} shortfall mean {np.mean(short):5.1f} m p90 {np.percentile(short, 90):5.1f} m"


def _tables(d, mask, label):
  print(f"\n{label}: frames {int(mask.sum())} of {len(mask)}")
  for cname, cmask in _classes(d).items():
    print(f" {cname}, by distance")
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
      print(f"  {lo:3.0f}-{hi:3.0f} m  " + _summary(d, mask & cmask & (d["d_rel"] >= lo) & (d["d_rel"] < hi), d["mp_in_path"], d["t_follow"], d["leaving"]))
    print(f" {cname}, by speed")
    for lo, hi in zip(SPEEDS[:-1], SPEEDS[1:], strict=True):
      print(f"  {lo:3.0f}-{hi:3.0f} m/s " + _summary(d, mask & cmask & (d["v_ego"] >= lo) & (d["v_ego"] < hi), d["mp_in_path"], d["t_follow"], d["leaving"]))


def _window(d, routes, route, start, end):
  i0 = d["route"] == routes.index(route)
  print(f"\n{route} {start:.0f}-{end:.0f} s, one row per second (path_y right-positive, off = yRel + path_y; class by |off|)")
  print("   t     v  dRel  yRel path_y   off class     inPath  latched t_follow base  valid")
  for sec in range(int(start), int(end) + 1):
    idx = np.flatnonzero(i0 & (d["el"] >= sec))
    if not idx.size:
      continue
    i = idx[0]
    a = abs(d["off"][i])
    cls = "in-lane" if a < LANE else "adjacent" if ADJACENT[0] <= a < ADJACENT[1] else "between" if a < ADJACENT[0] else "far"
    row = f"{d['el'][i]:6.1f} {d['v_ego'][i]:5.1f} {d['d_rel'][i]:5.1f} {d['y_rel'][i]:5.2f} {d['path_y'][i]:6.2f} {d['off'][i]:5.2f} {cls:9s}"
    state = "LATCHED" if d["leaving"][i] else "-"
    print(f"{row} {d['mp_in_path'][i]:6.2f}  {state:8s} {d['t_follow'][i]:6.3f} {d['base'][i]:4.2f}  {'' if d['valid'][i] else 'no'}")
  w = i0 & (d["el"] >= start) & (d["el"] <= end) & d["valid"]
  inl = w & (np.abs(d["off"]) < LANE)
  first = d["el"][w & d["leaving"]]
  onset = first[0] if first.size else math.nan
  print(f" window frames {int(w.sum())}: in-lane {np.mean(inl[w]):.3f}, latched {np.mean(d['leaving'][w]):.3f} (first at {onset:.2f} s)")
  print(f" t_follow/base mean {np.mean(d['t_follow'][w] / d['base'][w]):.3f}, in-lane frames latched {np.mean(d['leaving'][inl]):.3f}")


def _below(x, thresholds=(MOONPILOT_OUT_OF_PATH_ENTER, MOONPILOT_OUT_OF_PATH_LEAVE), digits=3):
  """'a/b': the fractions of x under each latch threshold."""
  return "/".join(f"{np.mean(x < t):.{digits}f}" for t in thresholds)


def _causes(d):
  """Which input of the probability drives a low inPath on in-lane leads."""
  rows = np.flatnonzero(d["valid"] & (np.abs(d["off"]) < LANE))
  adj = np.flatnonzero(d["valid"] & (np.abs(d["off"]) >= ADJACENT[0]) & (np.abs(d["off"]) < ADJACENT[1]))
  print(f"\nroot cause, in-lane frames {len(rows)} (adjacent {len(adj)})")
  print(" modelV2.leadsV3[0] as the network wrote it, per knot (in-lane frames): y_k - y_0 (right-positive), yStd, |y_k - y_0| / yStd")
  for k in range(N_LEAD):
    dy, s = d["my"][rows, k] - d["my"][rows, 0], d["mystd"][rows, k]
    z = np.abs(dy) / np.maximum(s, MOONPILOT_MIN_Y_STD)
    spread = f"dy med {np.median(dy):6.2f} |dy| p90 {np.percentile(np.abs(dy), 90):5.2f}  yStd med {np.median(s):5.2f} p10 {np.percentile(s, 10):5.2f}"
    print(f"  t={2 * k:2d} s  {spread}  |dy|/yStd med {np.median(z):5.1f}")
  _, y, s, p = grid(d, rows)
  s_c = np.maximum(s, MOONPILOT_MIN_Y_STD)
  w = 1.0 / s_c**2
  w /= w.sum(1, keepdims=True)
  dy, scale = y - p, math.sqrt(2) * s_c
  per = 0.5 * (_erf((MOONPILOT_PATH_HALF_WIDTH - dy) / scale) + _erf((MOONPILOT_PATH_HALF_WIDTH + dy) / scale)).astype(float)
  print(" leadd's grid: weight share and per-sample probability by second of the 4 s horizon (mean over in-lane frames):")
  for lo in range(4):
    cols = (DRIVE_GRID >= lo) & ((DRIVE_GRID < lo + 1) if lo < 3 else (DRIVE_GRID <= lo + 1))
    share, inside, away = w[:, cols].sum(1).mean(), per[:, cols].mean(), np.median(np.abs(dy)[:, cols])
    print(f"  {lo}-{lo + 1} s  weight {share:.3f}  P(in corridor) {inside:.3f}  |y - path| med {away:.2f} m")
  late = w[:, DRIVE_GRID >= 3.0].sum(1)
  dist = d["d_rel"][rows]
  cells = []
  for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
    m = (dist >= lo) & (dist < hi)
    cells.append(f"{lo:.0f}-{hi:.0f} m {np.mean(late[m]):.3f} / {np.median(d['mystd'][rows[m], 0]):.2f} m")
  print(" weight on the 3-4 s samples, and the first knot's yStd, by distance: " + "; ".join(cells))
  drift = np.abs(p[:, -1] - p[:, 0])
  print(f" ego path y at the lead's 4 s position minus at its position now: |.| med {np.median(drift):.2f} p90 {np.percentile(drift, 90):.2f} m")
  print("\n counterfactual raw inPath (before leadd's filter): fraction of frames below the latch's ENTER/LEAVE, in-lane then adjacent")
  ya, sa, pa = grid(d, adj)[1:]
  hold_p = lambda p_: np.repeat(p_[:, :1], p_.shape[1], axis=1)  # noqa: E731
  hold_y = lambda y_: np.repeat(y_[:, :1], y_.shape[1], axis=1)  # noqa: E731
  variants = {
    "current": lambda y_, s_, p_: raw_in_path(y_, s_, p_),
    "uniform weights": lambda y_, s_, p_: raw_in_path(y_, s_, p_, weights="uniform"),
    "horizon 2 s": lambda y_, s_, p_: raw_in_path(y_[:, :9], s_[:, :9], p_[:, :9]),
    "lead y held at its radar value": lambda y_, s_, p_: raw_in_path(hold_y(y_), s_, p_),
    "path held at its value now": lambda y_, s_, p_: raw_in_path(y_, s_, hold_p(p_)),
    "both held (yStd alone)": lambda y_, s_, p_: raw_in_path(hold_y(y_), s_, hold_p(p_)),
    "corridor half width 2.5 m": lambda y_, s_, p_: raw_in_path(y_, s_, p_, half_width=2.5),
    "yStd floored at 1.0 m": lambda y_, s_, p_: raw_in_path(y_, np.maximum(s_, 1.0), p_),
  }
  in_dist, adj_dist = d["d_rel"][rows], d["d_rel"][adj]
  for name, fn in variants.items():
    a, b = fn(y, s, p), fn(ya, sa, pa)
    cells = []
    for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
      ma, mb = (in_dist >= lo) & (in_dist < hi), (adj_dist >= lo) & (adj_dist < hi)
      cells.append(f"{lo:.0f}-{hi:.0f}: in {_below(a[ma])} adj {_below(b[mb], digits=2)}")
    print(f"  {name:31s} " + " | ".join(cells))


def _fix(d):
  """The proposed fix through leadd's filter and the planner's latch: only the horizon (`MOONPILOT_INPATH_HORIZON`) changes."""
  vision = (d["mp_present"] > 0) & (d["mp_source"] == 1)
  rows = np.flatnonzero(vision)
  y, s, p = grid(d, rows)[1:]
  results = {}
  for horizon in (4.0, 3.0, 2.0, 1.0, 0.0):
    n = int(round(horizon / 0.25)) + 1
    raw = np.ones(len(d["t"]))
    raw[rows] = raw_in_path(y[:, :n], s[:, :n], p[:, :n])
    results[horizon] = emulate_t_follow(raw, d, vision)
  tf, leaving = results[4.0]
  print("\nthe fix: leadd's filter + the planner's latch and slew emulated over every frame, by the horizon leadd evaluates")
  differ, worst = int(np.sum(leaving != d["leaving"])), np.max(np.abs(tf - d["t_follow"]))
  print(f"  horizon 4 s is the shipped one; vs the planner's own `_t_follow`: latch differs on {differ} of {len(leaving)} frames,")
  print(f"  max t_follow diff {worst:.3f} s")
  for cname, cmask in _classes(d).items():
    print(f" {cname}: latched fraction, mean t_follow/base")
    for horizon, (tf, leaving) in results.items():
      cells = []
      for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
        m = d["valid"] & cmask & (d["d_rel"] >= lo) & (d["d_rel"] < hi)
        ratio = np.mean(tf[m] / d["base"][m])
        cells.append(f"{lo:.0f}-{hi:.0f}: {np.mean(leaving[m]):.3f} {ratio:.3f}" if m.sum() > 20 else f"{lo:.0f}-{hi:.0f}: n={int(m.sum())}")
      print(f"  horizon {horizon:.0f} s  " + " | ".join(cells))
  eng = d["valid"] & (d["long_active"] > 0) & (np.abs(d["off"]) < LANE)
  print(f" engaged in-lane frames {int(eng.sum())}, mean t_follow/base and mean gap shortfall at the car's speed:")
  for horizon, (tf, leaving) in results.items():
    short = np.mean((d["base"][eng] - tf[eng]) * d["v_ego"][eng])
    print(f"  horizon {horizon:.0f} s  latched {np.mean(leaving[eng]):.3f}  t_follow/base {np.mean(tf[eng] / d['base'][eng]):.3f}  shortfall {short:.2f} m")
  _adjacent_episodes(d, results)


def _adjacent_episodes(d, results):
  """The true-positive side. Leads are rarely adjacent for long and the filter only falls with ~1 s of consistent evidence, so
  a latch already asserted when an adjacent episode began is the in-lane one carried over, not a detection."""
  adj = d["valid"] & (np.abs(d["off"]) >= ADJACENT[0]) & (np.abs(d["off"]) < ADJACENT[1])
  episodes = runs(adj)
  seconds = np.array([(e.stop - e.start) * DT_MDL for e in episodes])
  from_lane = sum(np.mean(np.abs(d["off"][max(e.start - 20, 0):e.start]) < LANE) > 0.5 for e in episodes if e.start)
  span = f"duration median {np.median(seconds):.2f} s p90 {np.percentile(seconds, 90):.2f} s max {seconds.max():.1f} s"
  print(f"\nadjacent leads ({ADJACENT[0]}-{ADJACENT[1]} m): {int(adj.sum())} frames in {len(episodes)} episodes, {span};")
  print(f" in-lane for most of the second before: {from_lane} episodes")
  for horizon, (_, leaving) in results.items():
    before = np.array([bool(leaving[e.start - 1]) if e.start else False for e in episodes])
    during = np.array([bool(leaving[e].any()) for e in episodes])
    frames = sum(int(leaving[e].sum()) for e in episodes)
    carried = sum(int(leaving[e].sum()) for e, b in zip(episodes, before, strict=True) if b)
    newly = int((during & ~before).sum())
    latched = f"latched frames {frames:4d} (in episodes latched before they began: {carried:4d})"
    print(f"  horizon {horizon:.0f} s  {latched}; episodes latched {int(during.sum()):3d}, new {newly}")
  for e, sec in zip(episodes, seconds, strict=True):
    if sec >= 1.0:
      flags = " ".join(f"h{h:.0f}:{int(lv[e].sum())}/{e.stop - e.start}" for h, (_, lv) in results.items())
      print(f"  episode >= 1 s: {sec:.1f} s at dRel {d['d_rel'][e.start]:.0f} m, offset {np.median(d['off'][e]):.1f} m, latched frames {flags}")


def _inpath(args):
  routes = list(dict.fromkeys(expand(args.routes, ROUTE_SETS)))
  d = pooled(routes, args.root)
  print(f"routes {len(routes)}: " + ", ".join(f"{r}:{drive_config(r, args.root)[0][:9]}" for r in routes))
  followed = d["valid"] & (d["v_ego"] > 10) & (d["d_rel"] > 20) & (d["d_rel"] < 80)
  plus, minus = np.median(np.abs(d["off"][followed])), np.median(np.abs(d["y_rel"] - d["path_y"])[followed])
  print(f"convention check on {int(followed.sum())} followed-lead frames (10+ m/s, 20-80 m): median |yRel + path_y| {plus:.2f} m,")
  print(f"  median |yRel - path_y| {minus:.2f} m: the sum is the offset")
  _tables(d, d["valid"], f"all valid frames (radar leadOne, leadd vision slot 0 agreeing, v > {MIN_SPEED:.0f} m/s)")
  _tables(d, d["valid"] & (d["long_active"] > 0), "engaged frames (carControl.longActive)")
  for route in (r for r in routes if r in args.window_routes):
    _window(d, routes, route, args.start, args.end)
  _causes(d)
  _fix(d)


def register(sub):
  parser = sub.add_parser("inpath", help="leadd's inPath on in-lane vs adjacent radar leads, and what the lead-lateral latch cost")
  parser.add_argument("--routes", nargs="+", default=["longitudinal"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--window-routes", nargs="+", default=["0000044d"], help="routes to print row by row")
  parser.add_argument("--start", type=float, default=205.0)
  parser.add_argument("--end", type=float, default=245.0)
  parser.set_defaults(fn=_inpath)
