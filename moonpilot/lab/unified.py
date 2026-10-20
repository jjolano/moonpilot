"""One approach requirement instead of TTC + arrival + stopping floor: `python -m moonpilot.lab unified-lead`.

`unified_lead_accel` is a prototype with `moonpilot.longitudinal.lead_accel`'s signature, so
`mock.patch.object(moonpilot.longitudinal, "lead_accel", unified_lead_accel)` swaps it into `policy` (which calls it by global
name; test modules that import it by name need the swap before they load, as `unified-tests` does). Nothing in the planner uses it.

  python -m moonpilot.lab unified-lead [--sections flights replay driver] [--modes engine default honda] [--only SCENARIO...]
  python -m moonpilot.lab unified-tests [--law unified|current] [--only TEST...]   # own process: the swap is permanent
"""
import contextlib
import sys
import unittest
from unittest import mock

import numpy as np

import moonpilot.longitudinal as L
from moonpilot.brake_band import MOONPILOT_BRAKE_BAND_FLOOR, MOONPILOT_BRAKE_BAND_MIN_SPEED
from moonpilot.lab import expand
from moonpilot.lab.planner import ROUTE_SETS as PLANNER_ROUTE_SETS

ROUTE_SETS = {"unified": (*PLANNER_ROUTE_SETS["longitudinal"], "0000044d")}

UNIFIED_COMFORT_V = [8.0, 15.0]  # m/s; the comfort map is the identity at and below the first, whole at the second
UNIFIED_HEADWAY_RATIO = 1.25 / 1.45  # the driver's settled headway (median 1.25 s, manual-decel) over the standard personality's
UNIFIED_COAST_FIRST = True  # the arrival's coast-first budget as part of the comfort map; False is the pure requirement
UNIFIED_LAG_T = 0.15  # s; travel at the closing speed the requirement reserves for the brake's onset (jerk limit, dead time)
UNIFIED_BOUND_ANCHOR = None  # m; the stop bound's gap anchor; None is the stop anchor (5.2 m at rest, 6.0 m by 4 m/s), CRASH_DISTANCE is the contact margin


def unified_lead_accel(v_ego, gap, v_lead, a_lead, t_follow, catch_up_accel=0.0, catch_up_coast=0.0, coast=MOONPILOT_BRAKE_BAND_FLOOR) -> float:
  """a = min(a_reg, ask, bound) where `bound` is admitted; the regulator, crawl/launch pieces and soft stop are `lead_accel`'s.

  target = max(rest(v_lead), anchor(v_ego), tau * v_lead): rest runs STOP_REST -> STOP_DISTANCE + MOVING_STOP_RESERVE as the lead
  moves, anchor is the stop's margin (STOP_REST -> STOP_DISTANCE over 4 m/s of ego speed), tau is t_follow, shrunk to the driver's
  1.25 s headway (x 1.25/1.45) with speed (8-15 m/s) behind a steady moving lead.
  ask = max(stopping_decel(v_ego, v_eff, a_lead, slack, sustain=0), -APPROACH_DECEL), slack = max(gap - target - T_LAG * closing,
  MIN_SLACK, 1.5 s * closing behind a moving lead); then coast first (the arrival's budget rule), the lead's positive accel credit,
  the catch-up credit and the engine coast's hold, in `lead_accel`'s order.
  bound = stopping_decel(v_ego, v_eff, a_lead, max(gap - anchor - T_LAG * closing, MIN_SLACK)) with the lead's braking held for
  `LEAD_BRAKE_SUSTAIN_T`, in force only past the driver's speed-scheduled plateau (`FLOOR_ADMISSION_*`).
  """
  gap = max(float(gap), 0.0)
  v_lead = max(float(v_lead), 0.0)
  # Regulator side, as `lead_accel` has it: braking-previewed speed for the gates, comfort-faded accel for the match.
  v_lead_eff = max(0.0, v_lead + min(float(a_lead), 0.0) * L.MOONPILOT_LEAD_PREVIEW_T)
  a_comfort = float(a_lead) * min(max(abs(float(a_lead)) / L.MOONPILOT_LEAD_ACCEL_COMFORT_NOISE - 1.0, 0.0), 1.0)
  v_lead_match = v_lead + max(a_comfort, 0.0) * L.MOONPILOT_LEAD_PREVIEW_T
  moving = min(v_lead / L.MOONPILOT_CREEP_SPEED, 1.0)
  stop_floor = L.MOONPILOT_STOP_DISTANCE + L.MOONPILOT_MOVING_STOP_RESERVE * min(v_ego / L.MOONPILOT_CREEP_SPEED, moving)
  gap_target = max(stop_floor, t_follow * v_ego)
  closing_match = max(v_ego - v_lead_match, 0.0)
  gap_cushion = min(L.MOONPILOT_FOLLOW_CUSHION, closing_match * L.MOONPILOT_K_V / L.MOONPILOT_K_GAP, max(t_follow * v_ego - stop_floor, 0.0))
  a_track = L.MOONPILOT_K_GAP * (gap - gap_target + gap_cushion) + L.MOONPILOT_K_V * (v_lead_match - v_ego)
  if a_track > 0.0:
    a_track *= L.pos_authority(v_ego)
  a_track = max(a_track, -L.MOONPILOT_APPROACH_DECEL)
  if a_track < 0.0 and v_ego > L.MOONPILOT_CREEP_SPEED and gap >= L.MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T * v_ego:
    w_v = (v_lead_match - v_ego) / L.MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V + 1.0
    w_a = (a_comfort - L.MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD) / L.MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_A + 1.0
    a_track += min(max(w_v, 0.0), 1.0) * min(max(w_a, 0.0), 1.0) * max(L.MOONPILOT_K_V * (v_lead_match - v_ego) - a_track, 0.0)
  if a_track < coast and v_ego > MOONPILOT_BRAKE_BAND_MIN_SPEED:
    w = min(max((gap / v_ego - L.MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T) / L.MOONPILOT_FOLLOW_COAST_HEADWAY_FADE_T, 0.0), 1.0)
    w *= float(np.interp(v_ego - v_lead_eff, L.MOONPILOT_FOLLOW_COAST_CLOSING, [1.0, 0.0]))
    w *= float(np.interp(a_lead, L.MOONPILOT_FOLLOW_COAST_LEAD_A, [0.0, 1.0]))
    a_track += w * (coast - a_track)
  a_track = L.crawl_accel(a_track, v_ego, v_lead, a_lead)
  a_track = L.departure_crawl(a_track, v_ego, v_lead, a_lead)
  a_track = L.soft_stop(a_track, v_ego, gap, v_lead_eff)
  if a_track > 0.0 and v_ego > v_lead_match and gap > gap_target:
    a_track *= min(1.0, (gap - gap_target) / L.MOONPILOT_FOLLOW_TAPER_M)

  # The comfort map, one speed weight: the identity below 8 m/s. Above, a steady moving lead is followed to the driver's own
  # tighter headway (the target moves in, so small closing finishes inside it instead of braking to it), and the engine
  # coast's hold keeps shallow asks at speed. A lead that is braking is a requirement, not a comfort question.
  w_v = float(np.interp(v_ego, UNIFIED_COMFORT_V, [0.0, 1.0]))
  w_steady = w_v * moving * float(np.interp(a_lead, L.MOONPILOT_FOLLOW_COAST_LEAD_A, [0.0, 1.0]))
  t_req = t_follow * (1.0 - w_steady * (1.0 - UNIFIED_HEADWAY_RATIO))
  rest = L.MOONPILOT_STOP_REST + (L.MOONPILOT_STOP_DISTANCE - L.MOONPILOT_STOP_REST + L.MOONPILOT_MOVING_STOP_RESERVE) * moving
  anchor = L.MOONPILOT_STOP_REST + min(L.MOONPILOT_STOP_TAPER_T * v_ego, L.MOONPILOT_STOP_DISTANCE - L.MOONPILOT_STOP_REST)  # the stop's margin
  target = max(rest, anchor, t_req * v_lead)
  # Behind a moving lead the final slack is floored at 1.5 s of closing, so the ask fades with the closing inside the target.
  # The ask reads the lead's braking only as the 0.5 s preview (`v_lead_eff`): holding it for seconds against a time-gap target
  # turns a half-second brake tap into a -1.8 m/s^2 command (`test_a_tap_does_not_buy_a_sustain_stop`), so the sustain goes to
  # the bound below. Capped at the approach decel: deeper is the bound's, and only where the driver's plateau says so.
  slack = max(gap - target - UNIFIED_LAG_T * (v_ego - v_lead), L.MOONPILOT_MIN_SLACK, moving * L.MOONPILOT_ARRIVAL_MIN_T * (v_ego - v_lead))
  ask = max(L.stopping_decel(v_ego, v_lead_eff, a_lead, slack, sustain=0.0), -L.MOONPILOT_APPROACH_DECEL)
  # The bound the map may not soften: the lead's braking held for its sustain, against the stop anchor, admitted where the
  # driver's plateau ends (`MOONPILOT_FLOOR_ADMISSION_*`) so it never blocks a catch-up.
  b_anchor = anchor if UNIFIED_BOUND_ANCHOR is None else UNIFIED_BOUND_ANCHOR
  bound = L.stopping_decel(v_ego, v_lead_eff, a_lead, max(gap - b_anchor - UNIFIED_LAG_T * (v_ego - v_lead), L.MOONPILOT_MIN_SLACK))
  admitted = bound < -float(np.interp(v_ego, L.MOONPILOT_FLOOR_ADMISSION_BP, L.MOONPILOT_FLOOR_ADMISSION_V))
  if ask >= 0.0 and not admitted:
    return a_track  # nothing closing, now or predicted
  a = a_track
  if ask < 0.0:
    # Coast first behind a moving lead: while the budget at COAST_DECEL leaves more than COAST_T of closing, the time to brake has
    # not come (the arrival's own rule). One weight, then the engine coast's hold keeps shallow asks at speed only while it is whole.
    closing = max(v_ego - v_lead_eff, 1e-6)
    room = max(gap - target, 0.0)
    coast_time = room * (1.0 + L.stopping_decel(v_ego, v_lead_eff, a_lead, max(room, 0.1)) / L.MOONPILOT_ARRIVAL_COAST_DECEL) / closing
    w_coast = min(max(coast_time / L.MOONPILOT_ARRIVAL_COAST_T, 0.0), 1.0) * moving
    w_coast *= min(max(closing / (L.MOONPILOT_ARRIVAL_COAST_DECEL * L.MOONPILOT_ARRIVAL_COAST_T) - 1.0, 0.0), 1.0)
    if UNIFIED_COAST_FIRST:
      ask = max(ask, -L.MOONPILOT_APPROACH_DECEL + (L.MOONPILOT_APPROACH_DECEL + coast) * w_coast)
    ask += max(a_comfort, 0.0)  # a lead pulling away is owed the credit, never a brake
    if catch_up_accel > 0.0 and catch_up_coast < 0.0 and a_lead > -L.MOONPILOT_LEAD_ACCEL_TAU_RESET:
      used = ask / catch_up_coast
      if used < 1.0:
        reserve = L.MOONPILOT_MOVING_STOP_RESERVE * moving
        weight = float(np.interp(t_follow * v_lead_eff, [L.MOONPILOT_STOP_REST, L.MOONPILOT_STOP_DISTANCE + reserve], [0.0, 1.0]))
        ask += catch_up_accel * weight * (1.0 - used)
    hold_full, hold_start = L.MOONPILOT_ARRIVAL_HOLD_FULL * coast, L.MOONPILOT_ARRIVAL_HOLD_START * coast
    if coast < MOONPILOT_BRAKE_BAND_FLOOR and hold_full < ask < 0.0:
      ask += w_v * w_coast * (float(np.interp(ask, [hold_full, hold_start], [hold_full, 0.0])) - ask)
    a = min(a_track, ask)
  if admitted:
    a = min(a, bound)
  # Crawl stop: spend the rest gap toward L.MOONPILOT_CRAWL_REST before braking past the driver's peak.
  cap = min(L.stopping_decel(v_ego, v_lead_eff, a_lead, max(gap - L.MOONPILOT_CRAWL_REST - L.MOONPILOT_MIN_SLACK, 0.05)), -L.MOONPILOT_CRAWL_STOP_DECEL)
  return a + float(np.interp(v_ego, L.MOONPILOT_SOFT_STOP_V, [1.0, 0.0])) * max(cap - a, 0.0)


# ---- (ii) closed-loop flights ------------------------------------------------------------------------------------------

def _scenarios():
  """(label, v0, set kph, gap0, v_lead0, lead accel(t), brake start, seconds). Leads brake to rest."""
  out = []
  for v in (10.0, 15.0, 20.0, 25.0):
    out.append((f"stopped lead, {v:.0f} m/s from 200 m", v, 3.6 * v, 200.0, 0.0, lambda t: 0.0, 0.0, 70.0))
  for v in (10.0, 15.0, 20.0, 25.0):
    out.append((f"{v / 2:.1f} m/s lead, {v:.0f} m/s from 120 m", v, 3.6 * v + 5.0, 120.0, v / 2, lambda t: 0.0, 0.0, 70.0))
  for a in (1.5, 3.5, 5.0):
    out.append((f"lead -{a} from a 1.45 s follow at 20", 20.0, 80.0, 29.0, 20.0, lambda t, a=a: -a if t > 6.0 else 0.0, 6.0, 45.0))
  out.append(("cut-in at 15 m, lead 17 vs 20 m/s", 20.0, 80.0, 15.0, 17.0, lambda t: 0.0, 0.0, 30.0))
  out.append(("steady 20 m/s follow, noisy", 20.0, 80.0, 29.0, 20.0, lambda t: 0.0, 0.0, 60.0))
  return out


@contextlib.contextmanager
def _default_rav4():
  """`_fly_lagged` builds its planner from test_engine_coast's `_planner(True)`; the same car without the flag is the default mode."""
  import moonpilot.tests.test_engine_coast as engine_tests
  real = engine_tests._planner
  with mock.patch.object(engine_tests, "_planner", lambda flag: real(False)):
    yield


def _flown(mode, scenario, seed=None):
  from moonpilot.lab.planner import _fly_lagged
  from moonpilot.tests.test_longitudinal import _fly
  _, v0, set_kph, gap0, v_lead0, a_fn, _, T = scenario
  if mode == "honda":
    r = _fly(v0, set_kph, gap0, v_lead0, a_fn, T)
    return {k: np.asarray(r[k]) for k in ("v", "gap", "cmd")}
  with _default_rav4() if mode == "default" else contextlib.nullcontext():
    return _fly_lagged(v0, set_kph, gap0, v_lead0, lambda t, vl: a_fn(t), T, seed=seed)


def _lamp_floor(mode, v):
  from opendbc.car.moonpilot.engine_coast import engine_coast_floor
  return engine_coast_floor(v) if mode == "engine" else MOONPILOT_BRAKE_BAND_FLOOR


def _flight_metrics(mode, scenario, r):
  from moonpilot.lab.planner import DT_MDL
  cmd, gap, v, t0 = r["cmd"], r["gap"], r["v"], scenario[6]
  after = np.arange(len(cmd)) * DT_MDL >= t0
  onset = np.flatnonzero((cmd < -0.05) & after)
  lamp = sum(c < _lamp_floor(mode, x) - 0.15 for c, x in zip(cmd, v, strict=True) if x > MOONPILOT_BRAKE_BAND_MIN_SPEED) * DT_MDL
  hf = float(np.std((cmd - np.convolve(cmd, np.ones(20) / 20, mode="same"))[10:-10]))
  return {"gap_on": float(gap[onset[0]]) if onset.size else float("nan"), "t_on": float(onset[0] * DT_MDL - t0) if onset.size else float("nan"),
          "peak": float(cmd.min()), "min_gap": float(gap.min()), "end_gap": float(gap[-1]), "lamp": float(lamp), "hf": hf}


def _flights(args, laws):
  seeds = range(args.seeds)
  for mode in args.modes:
    print(f"\nFlights, {mode} mode ({'Honda point mass, no lag, no noise' if mode == 'honda' else 'RAV4, 0.2 s dead time + 0.25 s lag'}). "
          + "gap_on/t_on: gap and seconds after the lead's brake onset at the first ask past -0.05; lamp: seconds below the coast floor - 0.15 above 5 m/s.")
    print(f"  {'scenario':40s} {'law':8s} {'gap_on':>6s} {'t_on':>5s} {'peak':>6s} {'min gap':>8s} {'end gap':>8s} {'lamp s':>6s} {'hf':>5s}")
    for scenario in (s for s in _scenarios() if not args.only or any(x in s[0] for x in args.only)):
      noisy = scenario[0].endswith("noisy")
      for name, law in laws:
        with law():
          runs = [_flown(mode, scenario, s) for s in (seeds if noisy and mode != "honda" else (None,))]
          ms = [_flight_metrics(mode, scenario, r) for r in runs]
        worst = {k: (max if k in ("lamp", "hf", "end_gap") else min)(m[k] for m in ms) for k in ms[0]} if noisy else ms[0]
        print(f"  {scenario[0]:40s} {name:8s} {worst['gap_on']:6.1f} {worst['t_on']:5.1f} {worst['peak']:6.2f} {worst['min_gap']:8.2f} {worst['end_gap']:8.2f} "
              + f"{worst['lamp']:6.1f} {worst['hf']:5.2f}")


def _current():
  return contextlib.nullcontext()


def _unified():
  return mock.patch.object(L, "lead_accel", unified_lead_accel)


# ---- (iii) open-loop replay of engaged drives --------------------------------------------------------------------------

WINDOWS = {"0000044d": ((204.5, 217.5, "arrival, lead 119 m"), (218.0, 222.0, "model braking at the intersection"), (211.0, 212.0, "lead latched out of path"))}


def _replay_route(route, root):
  """Both laws over one drive; a process-pool task, so the swap stays inside the worker."""
  from moonpilot.lab import segments
  from moonpilot.lab.planner import extract, replay
  from openpilot.tools.lib.logreader import LogReader
  data = extract(route, root)
  cur, logged, active, _ = replay(route, data, root)
  with _unified():
    uni, _, _, _ = replay(route, data, root)
  t0 = next(iter(LogReader(segments(route, root)[0], sort_by_time=True))).logMonoTime * 1e-9
  return cur, uni, logged, active, data["offsets"][:, 0] * 1e-9 - t0


def _replays(args):
  """The drive's own configuration replayed with the current law and with the unified one: how far apart the two commands are."""
  from concurrent.futures import ProcessPoolExecutor
  from moonpilot.lab import runs
  pooled = []
  print("\nOpen-loop replay on engaged frames (the drive's own params; the ego's recorded speed, so no closed loop):")
  routes = expand(args.routes, ROUTE_SETS)
  with ProcessPoolExecutor() as pool:
    results = list(pool.map(_replay_route, routes, [args.root] * len(routes)))
  for route, (cur, uni, logged, active, t) in zip(routes, results, strict=True):
    engaged = active & np.isfinite(logged)
    if not engaged.any():
      print(f"  {route}: no engaged frames")
      continue
    diff = uni - cur
    pooled.append(np.abs(diff[engaged]))
    p95, p99 = (np.percentile(np.abs(diff[engaged]), q) for q in (95, 99))
    print(f"  {route}  {engaged.sum() * 0.05 / 60:5.1f} min  p95 |unified - current| {p95:.3f}  p99 {p99:.3f}  "
          + f"deeper by > 0.2: {np.mean(diff[engaged] < -0.2) * 100:.1f}%  shallower by > 0.2: {np.mean(diff[engaged] > 0.2) * 100:.1f}%  "
          + f"| p95 vs logged: current {np.percentile(np.abs(cur - logged)[engaged], 95):.3f}, "
          + f"unified {np.percentile(np.abs(uni - logged)[engaged], 95):.3f}")
    windows = sorted(runs(engaged & (np.abs(diff) > 0.2)), key=lambda s: -np.abs(diff[s]).max())
    for s in windows[:3]:
      print(f"      {t[s.start]:7.1f}-{t[s.stop - 1]:7.1f} s  logged {logged[s].min():6.2f}..{logged[s].max():6.2f}  "
            + f"current {cur[s].min():6.2f}..{cur[s].max():6.2f}  unified {uni[s].min():6.2f}..{uni[s].max():6.2f}")
    for lo, hi, label in WINDOWS.get(route, ()):
      w = engaged & (t >= lo) & (t <= hi)
      if w.any():
        print(f"      {lo:.1f}-{hi:.1f} s ({label}): logged min {logged[w].min():6.2f}  "
              + f"current {cur[w].min():6.2f} (median {np.median(cur[w]):5.2f})  unified {uni[w].min():6.2f} (median {np.median(uni[w]):5.2f})")
  if pooled:
    p = np.concatenate(pooled)
    print(f"  pooled {len(p) * 0.05 / 60:.1f} min: p95 {np.percentile(p, 95):.3f}, p99 {np.percentile(p, 99):.3f}, max {p.max():.2f}")


# ---- (iv) against the driver's own longitudinal ------------------------------------------------------------------------

def _driver(args):
  """Both laws on the manual frames behind a radar lead that `manual-decel` bins, against the driver's aEgo."""
  from opendbc.car.moonpilot.engine_coast import engine_coast_floor
  from moonpilot.engine_coast import coast_budget
  from moonpilot.lab.planner import DT_MDL, _pooled_manual
  from moonpilot.longitudinal import lead_accel, lead_accel_estimate
  t_follow = 1.45
  d = _pooled_manual(expand(args.driver_routes, PLANNER_ROUTE_SETS), args.root)
  v, v_lead, gap = d["v"], d["v_lead"], d["gap"]
  lead = (d["manual"] > 0) & (d["lead"] > 0)
  a_lead = np.array([lead_accel_estimate(x) for x in d["a_lead"]])
  moving = np.clip(v_lead / L.MOONPILOT_CREEP_SPEED, 0.0, 1.0)
  target = np.maximum(L.MOONPILOT_STOP_REST + (L.MOONPILOT_STOP_DISTANCE - L.MOONPILOT_STOP_REST + L.MOONPILOT_MOVING_STOP_RESERVE) * moving, t_follow * v_lead)
  need = np.array([L.stopping_decel(*x) if m else 0.0 for m, x in zip(lead, zip(v, v_lead, a_lead, np.maximum(gap - target, 0.1), strict=True), strict=True)])
  approach = lead & (v > 2.0) & (gap - target > 1.0) & (need < 0.0)  # `manual-decel`'s frames
  out = {}
  for mode in ("default", "engine"):
    for name, law in (("current", lead_accel), ("unified", unified_lead_accel)):
      a = np.zeros(len(v))
      for i in np.flatnonzero(approach):
        floor = engine_coast_floor(v[i]) if mode == "engine" and v[i] > 5.0 else MOONPILOT_BRAKE_BAND_FLOOR
        a[i] = law(v[i], gap[i], v_lead[i], a_lead[i], t_follow, 0.0, 0.0, floor)
        if mode == "engine" and v[i] > 5.0:
          a[i] = coast_budget(a[i], v[i], gap[i], v_lead[i], a_lead[i], floor, L.MOONPILOT_APPROACH_DECEL,
                              L.MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T, L.MOONPILOT_STOP_DISTANCE)
      out[mode, name] = a
  leaves = np.zeros(len(v), dtype=bool)
  for i in np.flatnonzero(approach & (np.minimum.reduce(list(out.values())) <= -0.2)):
    ahead = slice(i, min(i + round(5.0 / DT_MDL), len(v)))
    same = d["route"][ahead] == d["route"][i]
    leaves[i] = not (d["lead"][ahead][same] > 0).all() or bool((np.abs(np.diff(gap[ahead][same])) > 5.0).any())
  print(f"\nAgainst the driver's aEgo: {approach.sum() * DT_MDL / 60:.1f} min of manual approach frames "
        + "(the leads that leave within 5 s are dropped from the ratio rows).")
  print("  ratio = median aEgo / ask over frames where the ask is -0.2 or deeper; "
        + "deep/shallow = share of ALL band frames where the ask is deeper / shallower than aEgo by > 0.3.")
  print(f"  {'band':9s} {'mode':8s} {'law':8s} {'n ask<=-0.2':>11s} {'ratio':>6s} {'p25':>5s} {'p75':>5s} {'deep':>6s} {'shallow':>7s} {'rms':>5s}")
  for lo, hi in ((2.0, 8.0), (8.0, 15.0), (15.0, 45.0)):
    band = approach & (v >= lo) & (v < hi)
    for mode in ("default", "engine"):
      for name in ("current", "unified"):
        a = out[mode, name]
        stays = band & ~leaves & (a <= -0.2)
        ratio = d["a"][stays] / a[stays]
        gap_a = a - d["a"]
        print(f"  {lo:.0f}-{hi:.0f} m/s {mode:8s} {name:8s} {stays.sum():11d} {np.median(ratio):6.2f} "
              + f"{np.percentile(ratio, 25):5.2f} {np.percentile(ratio, 75):5.2f} "
              + f"{np.mean(gap_a[band] < -0.3) * 100:5.1f}% {np.mean(gap_a[band] > 0.3) * 100:6.1f}% {np.sqrt(np.mean(gap_a[band] ** 2)):5.2f}")


def _stop(kph):
  """`test_the_approach_stops_behind_a_stopped_lead_at_every_reachable_speed`'s loop: rest gap and peak decel from 400 m."""
  from openpilot.common.constants import CV
  from moonpilot.tests.test_longitudinal import DT_MDL, V_CRUISE_MAX, _inputs, _lead, _planner
  planner, v, gap, peak = _planner(), kph * CV.KPH_TO_MS, 400.0, 0.0
  for _ in range(4000):
    planner.update(_inputs(v_ego=v, v_cruise_kph=V_CRUISE_MAX, lead=_lead(gap, 0.0), standstill=v < 0.1))
    peak = max(peak, abs(planner.output_a_target))
    v = max(0.0, v + planner.output_a_target * DT_MDL)
    gap = max(0.0, gap - v * DT_MDL)
    if v < 0.05 and abs(planner.output_a_target) < 0.05:
      break
  return gap, peak


@contextlib.contextmanager
def _variant(**constants):
  with _unified(), contextlib.ExitStack() as stack:
    for name, value in constants.items():
      stack.enter_context(mock.patch.object(sys.modules[__name__], name, value))
    yield


def _stops(args):
  """Rest gap behind a stopped lead (the pinned 4.2 m) for the law and for the variants that drop what carries its margin."""
  variants = (("current", _current), ("unified", _unified), ("no lag margin", lambda: _variant(UNIFIED_LAG_T=0.0)),
              ("contact bound", lambda: _variant(UNIFIED_BOUND_ANCHOR=L.MOONPILOT_CRASH_DISTANCE)),
              ("contact, no lag", lambda: _variant(UNIFIED_BOUND_ANCHOR=L.MOONPILOT_CRASH_DISTANCE, UNIFIED_LAG_T=0.0)))
  print("\nStopped lead from 400 m, Honda point mass, rest gap m (peak decel): the test pins > 4.2")
  print(f"  {'law':16s}" + "".join(f"{k:>14.1f} kph" for k in args.stop_kph))
  for name, law in variants:
    with law():
      print(f"  {name:16s}" + "".join(f"{g:9.2f} ({p:4.2f})" for g, p in (_stop(k) for k in args.stop_kph)))


def _run(args):
  laws = (("current", _current), ("unified", _unified))
  if "flights" in args.sections:
    _flights(args, laws)
  if "replay" in args.sections:
    _replays(args)
  if "driver" in args.sections:
    _driver(args)
  if "stops" in args.sections:
    _stops(args)


def register(sub):
  p = sub.add_parser("unified-lead", help="the one-requirement lead law against lead_accel: flights, replay, driver comparison")
  p.add_argument("--routes", nargs="+", default=["unified"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  p.add_argument("--driver-routes", nargs="+", default=["manual-approach"], help="the manual drives the law is compared with")
  p.add_argument("--seeds", type=int, default=5, help="noisy flights per steady-follow scenario")
  p.add_argument("--modes", nargs="+", default=["engine", "default", "honda"], choices=["engine", "default", "honda"])
  p.add_argument("--sections", nargs="+", default=["flights", "replay", "driver", "stops"], choices=["flights", "replay", "driver", "stops"])
  p.add_argument("--stop-kph", nargs="+", type=float, default=[90.0, 118.8, 129.6, 144.0])
  p.add_argument("--only", nargs="+", default=[], help="flight scenarios whose label contains any of these")
  p.set_defaults(fn=_run)
  p = sub.add_parser("unified-tests", help="the fork's longitudinal tests with lead_accel swapped (own process: the swap is permanent)")
  p.add_argument("--law", choices=("unified", "current"), default="unified")
  p.add_argument("--modules", nargs="+", default=list(TEST_MODULES))
  p.add_argument("--only", nargs="+", default=[], help="tests whose id contains any of these")
  p.set_defaults(fn=_tests)


# ---- (i) the fork's longitudinal tests with the law swapped in ---------------------------------------------------------

TEST_MODULES = ("test_longitudinal", "test_longitudinal_follow", "test_longitudinal_features", "test_engine_coast")


def _flatten(suite):
  for t in suite:
    yield from _flatten(t) if isinstance(t, unittest.TestSuite) else (t,)


def _tests(args):
  """The test modules with `lead_accel` replaced before they import it by name; a failure prints its test id and assertion.
  Process-wide, so nothing else in this command runs after the swap."""
  if args.law == "unified":
    L.lead_accel = unified_lead_accel  # ty: ignore[invalid-assignment]  # the swap under test, same call slot
  suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(f"moonpilot.tests.{m}") for m in args.modules)
  if args.only:
    suite = unittest.TestSuite(t for group in suite for t in _flatten(group) if any(x in t.id() for x in args.only))
  with open("/dev/null", "w") as null:
    result = unittest.TextTestRunner(stream=null, verbosity=0).run(suite)
  print(f"{args.law}: ran {result.testsRun}, {len(result.failures)} failures, {len(result.errors)} errors")
  for test, trace in (*result.failures, *result.errors):
    print(f"\n{test.id()}\n  {' '.join(trace.strip().splitlines()[-1].split())[:600]}")
