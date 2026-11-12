"""MoonpilotPathRecenter over recorded drives: how much of the model's request goes beyond the lane's own bend, against
the driver's own steering on the same roads, and what the cap does to it.

  python -m moonpilot.lab recenter   # AGENTS.md, Plan-timed path steering

One row per modelV2 (20 Hz), with the newest carState/carControl/controlsState it saw. The lane lines' first
MOONPILOT_RECENTER_FIT_X m are kept, which is all `lane_bend` reads.
"""
import argparse
from types import SimpleNamespace

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.lib.logreader import LogReader

from moonpilot.curvature import MOONPILOT_RECENTER_FIT_X, MOONPILOT_RECENTER_LAT_ACCEL, lane_bend, recentered
from moonpilot.lab import DEFAULT_ROOT, cached, expand
from moonpilot.lab.routes import LURCH_EVENT, add_routes_arg

VERSION = 3  # v3 keeps each frame's capture age and lateralDelay, the horizon the recovery-side guard reads
N_X = int(np.searchsorted(ModelConstants.X_IDXS, MOONPILOT_RECENTER_FIT_X, side="right"))



def _build(segs):
  rows, cs, lat, curv, delay = [], (0.0, False), False, 0.0, 0.0  # no lateralDelay yet: the guard reads no response horizon
  # One reader per segment, so each is freed once read: a LogReader over the whole route keeps every segment it has read.
  for m in (msg for seg in segs for msg in LogReader(seg, sort_by_time=True)):
    w = m.which()
    if w == "carState":
      cs = (m.carState.vEgo, m.carState.steeringPressed)
    elif w == "carControl":
      lat = m.carControl.latActive
    elif w == "controlsState":
      curv = m.controlsState.curvature
    elif w == "lateralDelay":
      delay = m.lateralDelay.lateralDelay
    elif w == "modelV2":
      mv = m.modelV2
      if len(mv.laneLines) != 4 or len(mv.laneLineProbs) != 4 or min(len(mv.laneLines[1].y), len(mv.laneLines[2].y)) < N_X:
        continue
      edges = [e.y[0] if len(e.y) else float("nan") for e in mv.roadEdges] if len(mv.roadEdges) == 2 else [float("nan")] * 2
      stds = list(mv.roadEdgeStds) if len(mv.roadEdgeStds) == 2 else [float("nan")] * 2
      rows.append((m.logMonoTime * 1e-9, cs[0], cs[1], lat, curv, mv.action.desiredCurvature, mv.meta.laneChangeState.raw,
                   mv.laneLineProbs[1], mv.laneLineProbs[2], *edges, *stds, (m.logMonoTime - mv.timestampEof) * 1e-9, delay,
                   *list(mv.laneLines[1].y)[:N_X], *list(mv.laneLines[2].y)[:N_X]))
  a = np.asarray(rows, dtype=float)
  keys = ("t", "v", "pressed", "lat", "curvature", "action", "lane_change", "prob_left", "prob_right", "edge_left", "edge_right",
          "edge_std_left", "edge_std_right", "capture_age", "lat_delay")
  d = {k: a[:, i] for i, k in enumerate(keys)}
  d["left"], d["right"] = a[:, len(keys):len(keys) + N_X], a[:, len(keys) + N_X:]
  return d


def extract(route, root=DEFAULT_ROOT):
  return cached(route, "recenter", VERSION, root, _build)


def lanes(d):
  """lane_bend per row, through the same reader controlsd's frames go through."""
  x = list(ModelConstants.X_IDXS[:N_X])
  return [lane_bend(SimpleNamespace(laneLines=[None, SimpleNamespace(x=x, y=d["left"][k]), SimpleNamespace(x=x, y=d["right"][k]), None],
                                    laneLineProbs=[0.0, d["prob_left"][k], d["prob_right"][k], 0.0],
                                    roadEdges=[SimpleNamespace(y=[d["edge_left"][k]]), SimpleNamespace(y=[d["edge_right"][k]])],
                                    roadEdgeStds=[d["edge_std_left"][k], d["edge_std_right"][k]])) for k in range(len(d["t"]))]


def _route(job):
  route, root, cap = job
  try:
    d = extract(route, root)
  except (FileNotFoundError, ValueError) as e:
    return route, None, str(e)
  lane = lanes(d)
  # controlsd's horizon at the model's arrival; ceiling: the drive's PathSmooth lead and a marketplace model's smoothing are not added
  horizon = d["capture_age"] + d["lat_delay"]

  def replay(lanes_):
    return np.asarray([recentered(r, b, s, cap, h) if c == 0 else r
                       for r, b, s, c, h in zip(d["action"], lanes_, d["v"], d["lane_change"], horizon, strict=True)])

  capped = replay(lane)
  position_only = replay([b._replace(left_slope=0.0, right_slope=0.0) for b in lane])  # the guard reading distance alone
  return route, {**d, "bend": np.asarray([b.bend for b in lane]), "weight": np.asarray([b.weight for b in lane]), "capped": capped,
                 "position_only": position_only}, None


def _q(x):
  x = x[np.isfinite(x)]
  return " / ".join(f"{np.percentile(x, p):.2f}" for p in (50, 90, 99, 99.9)) + f" (n {len(x)})" if len(x) else "n/a"


def _recenter(args):
  """The request's part beyond the lane's bend (|desiredCurvature - bend| * v^2) on well-seen lanes above 12 m/s, openpilot
  steering against the driver's own (his actual curvature, controlsState.curvature); then what the cap removes, in
  curves and out, and the lurch before and after."""
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(3) as pool:  # each worker holds a decoded segment
    results = list(pool.map(_route, [(r, args.root, args.cap) for r in expand(args.routes, args.root)]))
  done = {route: r for route, r, _ in results if r is not None}
  for route, _, err in results:
    if err is not None:
      print(f"{route}: skipped, {err}")
  cat = lambda f: np.concatenate([f(r) for r in done.values()])  # noqa: E731
  v, w, bend = cat(lambda r: r["v"]), cat(lambda r: r["weight"]), cat(lambda r: r["bend"])
  seen = (v > 12.0) & (w >= 1.0) & (cat(lambda r: r["lane_change"]) == 0)
  ours = seen & (cat(lambda r: r["lat"]) > 0) & (cat(lambda r: r["pressed"]) == 0)
  driver = seen & (cat(lambda r: r["lat"]) == 0)
  beyond_request = np.abs(cat(lambda r: r["action"]) - bend) * v**2
  beyond_driver = np.abs(cat(lambda r: r["curvature"]) - bend) * v**2
  print(f"{len(done)} drives; on well-seen lanes above 12 m/s, |curvature - lane bend| * v^2, p50 / p90 / p99 / p99.9 (m/s^2):")
  print(f"  openpilot's request ({ours.sum() * DT_MDL / 60:.0f} min): {_q(beyond_request[ours])}")
  print(f"  the driver's own    ({driver.sum() * DT_MDL / 60:.0f} min): {_q(beyond_driver[driver])}")

  removed = np.abs(cat(lambda r: r["action"]) - cat(lambda r: r["capped"])) * v**2
  curve = np.abs(bend) * v**2 >= 0.5
  steer = (v > 12.0) & (cat(lambda r: r["lat"]) > 0) & (cat(lambda r: r["pressed"]) == 0)
  print(f"\ncap {args.cap} m/s^2, openpilot steering above 12 m/s ({steer.sum() * DT_MDL / 60:.0f} min):")
  for name, m in (("straight-ish (bend < 0.5 m/s^2)", steer & ~curve), ("curves (bend >= 0.5 m/s^2)", steer & curve)):
    if not m.any():
      print(f"  {name}: no frames")
      continue
    r = removed[m]
    print(f"  {name}: the request lowered by > 0.05 m/s^2 on {np.mean(r > 0.05) * 100:.2f}% of {m.sum() * DT_MDL / 60:.0f} min, "
          + f"by p99 {np.percentile(r, 99):.2f}, max {r.max():.2f}")
  action, capped, position_only = cat(lambda r: r["action"]), cat(lambda r: r["capped"]), cat(lambda r: r["position_only"])
  by_position = np.abs(action - position_only) * v**2 > 0.05
  harder = steer & (np.abs(action - capped) > np.abs(action - position_only))
  print(f"  the approach term passes {np.sum(steer & by_position & (removed <= 0.05))} of the {np.sum(steer & by_position)} frames "
        + f"a guard reading the line's distance alone lowers by > 0.05 m/s^2; it changes {np.sum(steer & (capped != position_only))} "
        + f"and lowers {np.sum(harder)} harder")

  route, lo, hi = LURCH_EVENT
  if route in done:
    e = done[route]
    win = (e["t"] >= lo) & (e["t"] <= hi)
    print(f"\n{route}, the lurch (route 640-644 s): request, lane bend, capped (m/s^2), line probs")
    for k in np.flatnonzero(win)[::4]:
      a = e["v"][k] ** 2
      print(f"  {e['t'][k] - lo + 640.0:6.2f}  {e['action'][k] * a:+.2f}  {e['bend'][k] * a:+.2f}  {e['capped'][k] * a:+.2f}  "
            + f"{e['prob_left'][k]:.2f}/{e['prob_right'][k]:.2f}")
    print(f"  peak left {np.min(e['action'][win] * e['v'][win] ** 2):+.2f} -> {np.min(e['capped'][win] * e['v'][win] ** 2):+.2f}")


def _cap(text):
  cap = float(text)
  if not cap > 0.0:  # the guard's stopping distance divides by it; on the car, 0 is the feature off
    raise argparse.ArgumentTypeError("must be above 0 m/s^2")
  return cap


def register(sub):
  p = sub.add_parser("recenter", help="MoonpilotPathRecenter: the request beyond the lane's bend, openpilot against the driver, and the cap")
  add_routes_arg(p, ["recenter"])
  p.add_argument("--cap", type=_cap, default=MOONPILOT_RECENTER_LAT_ACCEL, help="m/s^2 beyond the lane's bend, above 0")
  p.set_defaults(fn=_recenter)
