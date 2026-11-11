"""The positive ramp (`MOONPILOT_JERK_RAMP_*` in `moonpilot/longitudinal.py`): how hard launches build, the driver's against
the car's, and today's ramp replayed against the up-jerk it replaced (4.0 m/s^3 at launch speeds, tapering to 1.5 by 5 m/s).
Command-path evidence: the car is held as recorded, so whether the car's own build softens with the command is a drive's
answer.
"""
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from openpilot.common.realtime import DT_CTRL, DT_MDL
from moonpilot.lab import burst_starts, expand, runs
from moonpilot.lab.bursts import extract_chain
from moonpilot.lab.planner import _launch_frames, extract, replay
from moonpilot.lab.routes import ROUTE_SETS

LAUNCH_RAMP_GAP = 25.0  # m; a lead nearer than this when the ego moves is the one launched behind (MOONPILOT_JERK_RAMP_GAP)


def _rest_launches(t, v, ok, group, dt, n):
  """The first moving frame (v >= 0.1) after every standstill held under `ok` for a second or more, where `ok`, one
  `group` (drive) and timestamp contiguity all hold through the `n` frames from it."""
  out = []
  for r in runs(ok & (v < 0.1)):
    j = r.stop
    if (r.stop - r.start >= round(1.0 / dt) and j + n <= len(v) and ok[j:j + n].all() and (group[j:j + n] == group[j]).all()
        and np.all(np.diff(t[j - 1:j + n]) < 1.5 * dt)):
      out.append(j)
  return out


def _steepest_rise(a, dt, stop_s, start_s=0.5, span_s=0.5):
  """The steepest rise of `a` over `span_s`, from `start_s` (past a standstill release's lurch) to `stop_s`."""
  k, w = round(span_s / dt), a[round(start_s / dt):round(stop_s / dt)]
  return float(np.max(w[k:] - w[:-k])) / span_s


def _replay(job):
  """One drive, replayed in its own configuration through today's planner or (`old`) with `jerk_limit`'s positive side
  patched back to the up-jerk the ramp replaced."""
  import moonpilot.longitudinal as longitudinal
  import moonpilot.jerk as jerk
  route, root, old = job
  saved = jerk.jerk_limit
  if old:
    def jerk_limit(a_cmd, a_prev, dt, v_ego, comfort_scale=1.0, lead_braking=False, ramp=None):
      return saved(a_cmd, a_prev, dt, v_ego, comfort_scale, lead_braking, float(np.interp(v_ego, [2.5, 5.0], [4.0, 1.5])))
    longitudinal.jerk_limit = jerk_limit  # ty: ignore[invalid-assignment]  # the replaced arm, same call shape
  try:
    replayed, _, _, details = replay(route, extract(route, root), root, details=True)
  finally:
    longitudinal.jerk_limit = saved
  return {"t": details["t"], "v": details["v_ego"], "active": details["long_active"], "logged": details["logged_a_target"], "lead": details["lead_present"],
          "cmd": np.asarray(replayed)}


def _accel_ramp(args):
  """The driver's manual launches out of a standstill of a second or more: aEgo's steepest 0.5 s rise past the release
  (0.5-4 s after moving) and its peak (0.5-3.5 s), with no lead within `LAUNCH_RAMP_GAP` and by how hard a lead there
  launched (its speed gained over 2 s); the first two bins only launches that reach 2 m/s by 3 s, not creeps. The
  `--engaged` drives' standstill launches, the same measures (to 3.5 s) from the 100 Hz chain. Then the drives with such a
  launch replayed in their own configuration, the car held as recorded, through today's planner and through it with the
  replaced up-jerk: per launch the command's steepest 0.5 s in the 5 s from its onset (the last logged target at or under
  0.02 before the move) and its ask 2 and 2.5 s in, and over all their engaged frames above 5 m/s the command's bursts
  (`bursts.BURST_RISE` in `BURST_T`) per minute. The brake exit's rebuild is today's in both arms."""
  routes, d = _launch_frames(args)
  n2, n3, n4 = (round(s / DT_MDL) for s in (2.0, 3.0, 4.0))
  rows = []
  for j in _rest_launches(d["t"], d["v"], d["manual"] > 0, d["route"], DT_MDL, n4):
    near = d["lead"][j] > 0 and d["gap"][j] < LAUNCH_RAMP_GAP
    a_lead = (d["v_lead"][j + n2] - d["v_lead"][j]) / 2.0 if near and (d["lead"][j:j + n2] > 0).all() else math.nan
    a = d["a"][j:j + n4]
    rows.append((near, a_lead, d["v"][j + n3], _steepest_rise(a, DT_MDL, 4.0), a[round(0.5 / DT_MDL):round(3.5 / DT_MDL)].max()))
  x = np.asarray(rows, dtype=float).reshape(-1, 5)
  near, a_lead, v3 = x[:, 0] > 0, x[:, 1], x[:, 2]
  print(f"driver: {len(x)} manual launches out of a standstill over {len(routes)} drives; aEgo steepest 0.5 s rise, peak (median, p75)")
  bins = ((f"no lead within {LAUNCH_RAMP_GAP:g} m, 2 m/s by 3 s", ~near & (v3 > 2.0)),
          ("lead launching < 1.0, 2 m/s by 3 s", near & (a_lead < 1.0) & (v3 > 2.0)),
          ("lead launching 1.0-1.5", near & (a_lead >= 1.0) & (a_lead < 1.5)), ("lead launching >= 1.5", near & (a_lead >= 1.5)))
  for name, m in bins:
    if m.any():
      print(f"  {name:36s} n {m.sum():4d}  rise {np.median(x[m, 3]):.2f} ({np.percentile(x[m, 3], 75):.2f}) m/s^3  "
            + f"peak {np.median(x[m, 4]):.2f} ({np.percentile(x[m, 4], 75):.2f}) m/s^2")

  engaged, n35 = [], round(3.5 / DT_CTRL)
  for route in expand(args.engaged, args.root):
    c = extract_chain(route, args.root)
    t, v, a = c["t"] * 1e-9, c["v_ego"], c["a_ego"]
    for j in _rest_launches(t, v, c["long_active"] > 0, np.zeros(len(t)), DT_CTRL, n35):
      engaged.append((route, _steepest_rise(a[j:j + n35], DT_CTRL, 3.5), a[j + round(0.5 / DT_CTRL):j + n35].max()))
  if not engaged:
    print("engaged: no standstill launches")
    return
  e = np.asarray([r[1:] for r in engaged])
  launch_routes = list(dict.fromkeys(r[0] for r in engaged))
  print(f"engaged: {len(e)} standstill launches over {len(launch_routes)} drives; aEgo steepest 0.5 s rise {np.median(e[:, 0]):.2f} "
        + f"({np.percentile(e[:, 0], 25):.2f}-{np.percentile(e[:, 0], 75):.2f}) m/s^3, peak {np.median(e[:, 1]):.2f} m/s^2")

  n5 = round(5.0 / DT_MDL)
  with ProcessPoolExecutor(4) as pool:
    arms = {name: list(pool.map(_replay, [(r, args.root, old) for r in launch_routes])) for name, old in (("replaced", True), ("today", False))}
  print("replayed on those drives, the car as recorded: launches from the command's onset (median steepest 0.5 s rise m/s^3 / ask "
        + "at 2.0 s / at 2.5 s), and command bursts per engaged minute above 5 m/s")
  for name, flights in arms.items():
    out, bursts, frames = [], 0, 0
    for f in flights:
      for j in _rest_launches(f["t"], f["v"], f["active"], np.zeros(len(f["t"])), DT_MDL, 0):
        k = j
        while k > j - n2 and f["logged"][k] > 0.02:
          k -= 1
        if k + n5 <= len(f["t"]) and f["active"][k:k + n5].all():
          cmd = f["cmd"][k:k + n5]
          out.append((f["lead"][k], _steepest_rise(cmd, DT_MDL, 5.0, start_s=0.0), cmd[n2], cmd[round(2.5 / DT_MDL)]))
      fast = f["active"] & (f["v"] > 5.0)
      bursts, frames = bursts + len(burst_starts(f["cmd"], fast)), frames + int(fast.sum())
    y = np.asarray(out, dtype=float).reshape(-1, 4)
    cells = [f"{label} (n {m.sum()}) {np.median(y[m, 1]):.2f} / {np.median(y[m, 2]):+.2f} / {np.median(y[m, 3]):+.2f}"
             for label, m in (("all", np.ones(len(y), bool)), ("no lead", y[:, 0] == 0), ("lead", y[:, 0] > 0)) if m.any()]
    print(f"  {name:9s} " + "  |  ".join(cells) + f"  |  bursts {bursts / max(frames * DT_MDL / 60, 1e-9):.2f}/min over {frames * DT_MDL / 60:.0f} min")


def register(sub):
  parser = sub.add_parser("accel-ramp", help="the positive ramp: launch build-up by lead launch, engaged aEgo, today's ramp against the replaced up-jerk")
  parser.add_argument("--routes", nargs="*", default=[], help=f"manual route names or sets: {', '.join(ROUTE_SETS)}; default every drive with rlogs")
  parser.add_argument("--engaged", nargs="+", default=["longitudinal-engaged"], help=f"engaged route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_accel_ramp)
