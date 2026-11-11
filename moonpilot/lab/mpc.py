"""Compare the shipped and stock upstream longitudinal MPC over recorded model ticks."""
import glob
import math
import os
from collections import Counter
from typing import Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, T_IDXS as MPC_T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX
from opendbc.car.structs import car
from moonpilot.features import LONGITUDINAL

from moonpilot.lab import drive_config
from moonpilot.lab.planner import MESSAGES, extract
from moonpilot.long_mpc import MOONPILOT_MPC_MAX_STEPS, MoonpilotSolve


class _OneStep(MoonpilotSolve):
  """Upstream's solve: one RTI step, and upstream's own reset in `run()` on any failure."""

  def __call__(self, solver, x0, x_sol, u_sol):
    return solver.solve(), float(solver.get_stats("time_tot")[0])


class _SolverProxy:
  def __init__(self, solver):
    self.solver = solver
    self.statuses = []
    self.times = []

  def solve(self):
    status = self.solver.solve()
    self.statuses.append(int(status))
    self.times.append(float(self.solver.get_stats("time_tot")[0]))
    return status

  def __getattr__(self, name):
    return getattr(self.solver, name)


def _pct(values):
  x = np.asarray(values, dtype=float)
  x = x[np.isfinite(x)]
  return tuple(float(np.percentile(x, p)) for p in (50, 99)) + (float(np.max(x)) if len(x) else math.nan,)


def _plan_target(mpc, cp):
  speeds = np.interp(CONTROL_N_T_IDX, MPC_T_IDXS, mpc.v_solution)
  accels = np.interp(CONTROL_N_T_IDX, MPC_T_IDXS, mpc.a_solution)
  return float(get_accel_from_plan(speeds, accels, CONTROL_N_T_IDX, action_t=cp.longitudinalActuatorDelay + DT_MDL))


def _route_job(job):
  route, root = job
  try:
    d = extract(route, root)
    cp = messaging.log_from_bytes(d["cp"].tobytes(), car.CarParams)
    _, params = drive_config(route, root)
    stock_error_allowed = not cp.openpilotLongitudinalControl or params.get(LONGITUDINAL.key) in ("0", "false", "False", "")

    fork, upstream = LongitudinalMpc(), LongitudinalMpc()
    fork_proxy, upstream_proxy = _SolverProxy(fork.solver), _SolverProxy(upstream.solver)
    fork.solver, upstream.solver = fork_proxy, upstream_proxy

    upstream.moonpilot_solve = _OneStep(MPC_T_IDXS, DT_MDL)
    blob, offsets = d["blob"], d["offsets"]
    idx = {kind: 1 + i * 3 for i, kind in enumerate(MESSAGES)}
    counters = {"fork": Counter(), "upstream": Counter()}
    hold_runs, held_run = [], 0
    fail_frames = held_frames = resets = fcw_fork = fcw_upstream = 0
    hold_frames = fork.moonpilot_solve.hold_frames
    steps_fork, steps_upstream, time_fork, time_upstream, targets = [], [], [], [], []
    logged_speeds, logged_accels = [], []
    for row in offsets:
      messages = {}
      for kind in MESSAGES:
        start, size = int(row[idx[kind]]), int(row[idx[kind] + 1])
        messages[kind] = messaging.log_from_bytes(blob[start:start + size].tobytes())
      cs = messages["carState"].carState
      radar = messages["radarState"].radarState
      personality = messages["selfdriveState"].selfdriveState.personality
      for proxy in (fork_proxy, upstream_proxy):
        proxy.statuses.clear()
        proxy.times.clear()
      held_before = fork.moonpilot_solve.held
      for mpc in (fork, upstream):
        mpc.set_weights(prev_accel_constraint=not cs.standstill, personality=personality)
        mpc.set_cur_state(cs.vEgo, cs.aEgo)
        mpc.update(radar, personality=personality)

      fs, us = fork_proxy.statuses, upstream_proxy.statuses
      counters["fork"].update(code for code in fs if code != 0)
      counters["upstream"].update(code for code in us if code != 0)
      steps_fork.append(len(fs))
      steps_upstream.append(len(us))
      time_fork.append(sum(fork_proxy.times))
      time_upstream.append(sum(upstream_proxy.times))
      fail_frames += any(code != 0 for code in fs)
      held_after = fork.moonpilot_solve.held
      if held_after > held_before:
        held_frames += 1
        held_run += 1
      else:
        if held_run:
          hold_runs.append(held_run)
          held_run = 0
        resets += held_before == held_after == hold_frames  # still failing past the hold: upstream's reset ran
      fcw_fork += int(fork.crash_cnt > 2 and not cs.standstill)
      fcw_upstream += int(upstream.crash_cnt > 2 and not cs.standstill)
      targets.append((_plan_target(fork, cp), _plan_target(upstream, cp)))

      pstart, psize = int(row[1 + len(MESSAGES) * 3]), int(row[2 + len(MESSAGES) * 3])
      if stock_error_allowed and psize:
        lp = messaging.log_from_bytes(blob[pstart:pstart + psize].tobytes()).longitudinalPlan
        lv, la = np.asarray(lp.speeds), np.asarray(lp.accels)
        replay_v = np.interp(CONTROL_N_T_IDX, MPC_T_IDXS, upstream.v_solution)
        replay_a = np.interp(CONTROL_N_T_IDX, MPC_T_IDXS, upstream.a_solution)
        n = min(len(lv), len(replay_v))
        logged_speeds.extend(np.abs(replay_v[:n] - lv[:n]))
        n = min(len(la), len(replay_a))
        logged_accels.extend(np.abs(replay_a[:n] - la[:n]))

    if held_run:
      hold_runs.append(held_run)
    diff = np.abs(np.asarray(targets)[:, 0] - np.asarray(targets)[:, 1])
    return route, {
      "frames": len(offsets), "failures": counters, "fail_frames": fail_frames, "held": held_frames, "hold_runs": hold_runs,
      "resets": resets, "steps_fork": steps_fork, "steps_upstream": steps_upstream,
      "time_fork": time_fork, "time_upstream": time_upstream, "diff": diff,
      "fcw_fork": fcw_fork, "fcw_upstream": fcw_upstream,
      "logged_speed": logged_speeds, "logged_accel": logged_accels,
      "stock_error_allowed": stock_error_allowed,
    }, None
  except Exception as e:
    return route, None, f"{type(e).__name__}: {e}"


def _report(args):
  routes = sorted({os.path.basename(os.path.dirname(p)).split("--", 1)[0]
                   for p in glob.glob(os.path.join(args.root, "*--*", "rlog.zst"))})
  if args.routes:
    routes = [r for r in routes if any(r == f or r.startswith(f) for f in args.routes)]
  if args.limit is not None:
    routes = routes[:args.limit]
  with ProcessPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(_route_job, ((r, args.root) for r in routes)))

  totals: dict[str, Any] = {"frames": 0, "failures": {k: Counter() for k in ("fork", "upstream")}, "fail_frames": 0,
            "held": 0, "hold_runs": [], "resets": 0, "steps_fork": [], "steps_upstream": [], "time_fork": [], "time_upstream": [],
            "diff": [], "fcw_fork": 0, "fcw_upstream": 0, "logged_speed": [], "logged_accel": []}
  failed_routes, errors, fork_only = [], [], 0
  for route, d, error in results:
    if error:
      errors.append((route, error))
      continue
    t = totals
    t["frames"] += d["frames"]
    for k in t["failures"]:
      t["failures"][k].update(d["failures"][k])
    for k in ("fail_frames", "held", "resets", "fcw_fork", "fcw_upstream"):
      t[k] += d[k]
    t["hold_runs"].extend(d["hold_runs"])
    for k in ("steps_fork", "steps_upstream", "time_fork", "time_upstream", "diff", "logged_speed", "logged_accel"):
      t[k].extend(d[k])
    fork_only += not d["stock_error_allowed"]
    if any(d["failures"][k] for k in d["failures"]):
      failed_routes.append((route, d))

  def summary(label, d):
    sf, su = _pct(d["steps_fork"]), _pct(d["steps_upstream"])
    tf, tu = _pct(d["time_fork"]), _pct(d["time_upstream"])
    de = np.asarray(d["diff"])
    shares = (float(np.mean(de > .05)) if len(de) else math.nan, float(np.mean(de > .2)) if len(de) else math.nan)
    fs = d["failures"]
    print(f"{label}: frames={d['frames']} raw failures a={dict(sorted(fs['fork'].items()))} b={dict(sorted(fs['upstream'].items()))}")
    hold_runs = f"max={max(d['hold_runs'], default=0)}, histogram={dict(sorted(Counter(d['hold_runs']).items()))}"
    print(f"  a: frames with a failure={d['fail_frames']} held={d['held']} hold-runs({hold_runs}) resets-after-hold={d['resets']}")
    at_cap = int(np.sum(np.asarray(d["steps_fork"]) >= MOONPILOT_MPC_MAX_STEPS))
    print(f"  steps/frame a mean={np.mean(d['steps_fork']):.3f} p99={sf[1]:.3f} max={sf[2]:.0f} at cap={at_cap}; " +
          f"b mean={np.mean(d['steps_upstream']):.3f} p99={su[1]:.3f} max={su[2]:.0f}")
    print(f"  acados ms/frame a p50/p99/max={tf[0]*1000:.3f}/{tf[1]*1000:.3f}/{tf[2]*1000:.3f}; b={tu[0]*1000:.3f}/{tu[1]*1000:.3f}/{tu[2]*1000:.3f}")
    print(f"  |a_target(a)-a_target(b)| p50/p99/max={_pct(de)[0]:.4f}/{_pct(de)[1]:.4f}/{_pct(de)[2]:.4f}; >0.05={shares[0]:.3%} >0.2={shares[1]:.3%}")
    print(f"  FCW frames a={d['fcw_fork']} b={d['fcw_upstream']}")
    if d["logged_speed"]:
      ls, la = _pct(d["logged_speed"]), _pct(d["logged_accel"])
      print(f"  upstream plan vs logged: |speed| p50/p99/max={ls[0]:.4f}/{ls[1]:.4f}/{ls[2]:.4f} m/s; " +
            f"|accel|={la[0]:.4f}/{la[1]:.4f}/{la[2]:.4f} m/s^2")
    else:
      print("  upstream plan vs logged: n/a (fork planner)")

  summary("TOTAL", totals)
  print(f"routes={len(routes)} completed={len(results)-len(errors)} errors={len(errors)} fork-planner routes={fork_only}")
  for route, d in failed_routes:
    summary(route, d)
  for route, error in errors:
    print(f"ERROR {route}: {error}")


def register(sub):
  parser = sub.add_parser("mpc-solve", help="replay shipped and upstream acados longitudinal MPC solves on recorded routes")
  parser.add_argument("--routes", nargs="+", help="route ID/name filters (prefix matches)")
  parser.add_argument("--limit", type=int, help="run at most the first N sorted routes")
  parser.set_defaults(fn=_report)
