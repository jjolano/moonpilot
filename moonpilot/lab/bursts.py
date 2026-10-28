"""Positive accel bursts while openpilot has the pedals: in the logged command, in the car's measured acceleration, and
down the chain between them (plan, the fork controller, what the car's controller sent the PCM).

`accel-bursts` replays today's planner (each drive's own params) to attribute the logged bursts it reproduces;
`burst-chain` reads the 100 Hz chain at `accel-bursts`' unmatched events and at every measured burst it finds itself.
Command-path evidence: the car is held as recorded.
"""
import math

import numpy as np
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.tools.lib.logreader import LogReader
from moonpilot.longitudinal import LongitudinalPlanSource
from moonpilot.lab import DEFAULT_ROOT, cached, expand, lag_s, runs
from moonpilot.lab.planner import MESSAGES, ROUTE_SETS, _details_arrays, extract, replay


CHAIN_VERSION = 1


def _chain_build(segs):
  """One row per carControl frame (100 Hz): the plan's aTarget, the fork controller's accel, what the car's own controller
  sent to the PCM after its PID, pitch compensation and clip (`carOutput.actuatorsOutput.accel`), and the measured aEgo."""
  latest, rows = {}, []
  for msg in (m for seg in segs for m in LogReader(seg, sort_by_time=True)):
    kind = msg.which()
    if kind == "carState":
      cs = msg.carState
      latest[kind] = (cs.vEgo, cs.aEgo, cs.gasPressed, cs.brakePressed)
    elif kind == "longitudinalPlan":
      latest[kind] = (msg.longitudinalPlan.aTarget,)
    elif kind == "carOutput":
      latest[kind] = (msg.carOutput.actuatorsOutput.accel,)
    elif kind == "carControl" and all(k in latest for k in ("carState", "longitudinalPlan", "carOutput")):
      cc = msg.carControl
      rows.append((int(msg.logMonoTime), float(cc.longActive), float(cc.actuators.accel), *latest["longitudinalPlan"], *latest["carOutput"],
                   *latest["carState"]))
  if not rows:
    raise ValueError("no complete chain frames")
  cols = np.asarray(rows, dtype=float).T
  return dict(zip(("t", "long_active", "accel", "a_target", "output", "v_ego", "a_ego", "gas", "brake"), cols, strict=True))


def extract_chain(route: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
  return cached(route, "chain", CHAIN_VERSION, root, _chain_build)


def _rise(x):
  """The largest rise inside a window: the most any sample sits above the lowest before it."""
  return float(np.max(x - np.minimum.accumulate(x))) if len(x) else math.nan


CHAIN_RISE = 0.2  # m/s^2; half a burst — a stage that rose this much behind a burst is a stage that asked for it


def _transition(c, k, lead_in, n):
  """How control changed around a burst starting at chain frame `k`: 2, a hand-back from the driver's pedal — openpilot's
  longitudinal goes inactive to active between `lead_in` frames before `k` and `k`, with the gas or brake pressed in the
  burst second before that edge; 1, any other frame without openpilot's longitudinal in the whole window (an engagement
  with no pedal, a disengagement during or after the burst); 0, active throughout."""
  active = c["long_active"] > 0
  lo, hi = max(k - lead_in - n, 0), min(k + n + 1, len(active))
  if active[lo:hi].all():
    return 0
  for j in range(max(k - lead_in, 1), k + 1):
    if active[j] and not active[j - 1]:
      before = slice(max(j - n, 0), j)
      if (c["gas"][before] > 0).any() or (c["brake"][before] > 0).any():
        return 2
  return 1


def _chain_window(c, a, k, lp, lo, n):
  """(plan, fork controller, sent to the PCM, measured) rises over one burst second starting at chain frame `k`, each
  command shifted back by its own delay to aEgo, and `_transition` over the earliest shifted start: on a hand-back the
  controller steps from its inactive zero to the plan and the car's measured accel fills in the dip the release left."""
  w_plan, w_out = slice(k - lp, k - lp + n + 1), slice(k - lo, k - lo + n + 1)
  return (_rise(c["a_target"][w_plan]), _rise(c["accel"][w_plan]), _rise(c["output"][w_out]), a[k + n] - a[k],
          float(_transition(c, k, max(lp, lo), n)))


def _chain_bursts(job):
  """One drive's chain: its delays, its own 100 Hz measured bursts, and the chain at each of `accel-bursts`' unmatched
  events (`_unmatched_starts`), or None per event the chain cannot place — no delay estimate (under 30 s of engaged frames
  above 5 m/s), or no chain frame within a control tick of the event's start or its shifted windows."""
  route, root = job
  c = extract_chain(route, root)
  lf = _logged_frames(route, root)
  event_t = lf["t"][_unmatched_starts(lf["logged"], lf["a_ego"], lf["v_ego"], lf["active"])]
  n, smooth = round(BURST_T / DT_CTRL), round(0.5 / DT_CTRL)
  mask = (c["long_active"] > 0) & (c["gas"] == 0) & (c["brake"] == 0) & (c["v_ego"] > 5.0) & np.isfinite(c["a_ego"])
  if mask.sum() < round(30.0 / DT_CTRL):
    return route, None, [(float(t), None) for t in event_t]
  a = np.convolve(c["a_ego"], np.ones(smooth) / smooth, mode="same")
  max_lag = 1.0
  lag_plan = lag_s(c["a_target"], a, mask, DT_CTRL, max_lag)
  lag_out = lag_s(c["output"], a, mask, DT_CTRL, max_lag)
  lp, lo = round(lag_plan / DT_CTRL), round(lag_out / DT_CTRL)
  rows = []
  for r in runs(mask):
    i = r.start + max(lp, lo)
    while i < r.stop - n:
      if a[i + n] > BURST_END and a[i + n] - a[i] >= BURST_RISE:
        rows.append(_chain_window(c, a, i, lp, lo, n))
        i += n
      else:
        i += 1
  events, t_ns = [], c["t"]
  for t in event_t:
    k = int(np.searchsorted(t_ns, t * 1e9))
    placed = (max(lp, lo) <= k < len(t_ns) - n and abs(t_ns[k] - t * 1e9) <= 1.5 * DT_CTRL * 1e9
              and np.all(np.diff(t_ns[k - max(lp, lo):k + n + 1]) <= 1.5 * DT_CTRL * 1e9))
    events.append((float(t), _chain_window(c, a, k, lp, lo, n) if placed else None))
  return route, (lag_plan, lag_out, mask.sum() * DT_CTRL / 60, np.asarray(rows, dtype=float).reshape(-1, 5)), events


def _class_masks(x):
  """Each burst on the first stage that rose `CHAIN_RISE` behind it — control transitions first — as (name, mask) per
  class, in chain order."""
  plan, ctrl, out, _, transition = x.T
  own = transition == 0
  return (
    ("a hand-back from the driver's pedal", transition == 2),
    ("another control transition in the window", transition == 1),
    ("the plan rose a full burst", own & (plan >= BURST_RISE)),
    ("the plan rose, under a burst", own & (plan >= CHAIN_RISE) & (plan < BURST_RISE)),
    ("the fork controller rose without the plan", own & (plan < CHAIN_RISE) & (ctrl >= CHAIN_RISE)),
    ("the car's controller rose without either", own & (plan < CHAIN_RISE) & (ctrl < CHAIN_RISE) & (out >= CHAIN_RISE)),
    ("no stage rose: the measurement's alone", own & (plan < CHAIN_RISE) & (ctrl < CHAIN_RISE) & (out < CHAIN_RISE)),
  )


def _chain_classes(x):
  """The classes, printed with each stage's median rise."""
  print(f"  {'':44s} {'n':>4s}  median rise behind it, m/s^2: plan / fork controller / sent to PCM / measured")
  for name, m in _class_masks(x):
    cells = " / ".join(f"{np.median(v[m]):+.2f}" if m.any() else "  -  " for v in x.T[:4])
    print(f"  {name:44s} {int(m.sum()):4d}  {cells}")


def _burst_chain(args):
  """Measured accel bursts while openpilot has the pedals above 5 m/s (aEgo over 0.5 s; `driver._bursts`' rise of 0.4 m/s^2
  in 1 s ending above +0.3) against what each stage of the chain did over the same second, at 100 Hz, shifted by that
  drive's own delay to aEgo (`lag_s`, up to 1 s): the plan's aTarget, the fork controller's accel, and what the car's
  controller sent the PCM. First `accel-bursts`' unmatched events — measured bursts no logged-command burst starts
  within 1 s of, the same `_unmatched_starts` — each by its drive and time, with any the chain cannot place unresolved;
  then, separately, every burst the chain itself finds. A burst no stage rose behind is the measurement's alone — the
  PCM, the road's grade or the aEgo estimate, which this cannot tell apart."""
  from concurrent.futures import ProcessPoolExecutor
  routes = list(dict.fromkeys(expand(args.routes, ROUTE_SETS)))
  with ProcessPoolExecutor(3, max_tasks_per_child=1) as pool:
    results = list(pool.map(_chain_bursts, [(route, args.root) for route in routes]))
  events = [(route, t, row) for route, _, ev in results for t, row in ev]
  placed = np.asarray([row for _, _, row in events if row is not None], dtype=float).reshape(-1, 5)
  unresolved = [(route, t) for route, t, row in events if row is None]
  print(f"accel-bursts' {len(events)} unmatched events, by drive and time: {len(placed)} placed on the chain, {len(unresolved)} unresolved")
  if len(placed):
    _chain_classes(placed)
  for route, t in unresolved:
    print(f"  unresolved: {route} at {t:.2f} s (mono)")
  if args.list:
    placed_events = [(route, t) for route, t, row in events if row is not None]
    for name, m in _class_masks(placed):
      for (route, t), row in zip(np.asarray(placed_events, dtype=object)[m], placed[m], strict=True):
        print(f"  {name}: {route} at {t:.2f} s (mono), rises {' / '.join(f'{v:+.2f}' for v in row[:4])}")
  summaries = [s for _, s, _ in results if s is not None]
  if not summaries:
    print(f"none of {len(routes)} drives has 30 s of engaged frames above 5 m/s to estimate its delay from")
    return
  lags = np.asarray([(lp, lo) for lp, lo, _, _ in summaries])
  minutes = sum(m for _, _, m, _ in summaries)
  x = np.concatenate([rows for *_, rows in summaries])
  print(f"\nseparately, every burst the chain finds: {len(summaries)} drives, {minutes:.1f} engaged min above 5 m/s (gas and brake off), "
        + f"{len(x)} measured bursts ({len(x) / max(minutes, 1e-9):.2f}/min); delay to aEgo median: plan {np.median(lags[:, 0]):.2f} s, "
        + f"sent to the PCM {np.median(lags[:, 1]):.2f} s")
  _chain_classes(x)


BURST_RISE, BURST_END, BURST_T = 0.4, 0.3, 1.0  # `driver._bursts`' definition: a rise of 0.4 m/s^2 in 1 s that ends above +0.3
BURST_FAR_MIN_GAP = 50.0  # m; `--far-headway` drops a lead only past this as well, so a slow-speed lead is never far


def _burst_replay(job):
  """One engaged drive replayed through today's planner with each live `policy` call recorded — what won, the cruise
  candidate with no lead cap, each lead's own ask, the model's ask — optionally ignoring leads past `far_headway`."""
  import moonpilot.longitudinal as longitudinal
  route, root, far_headway = job
  policy0, update0 = longitudinal.policy, longitudinal.MoonpilotLongitudinalPlanner.update
  calls, frames = [], []

  def policy(v_ego, leads, *a, **k):
    if far_headway is not None:
      leads = [lead for lead in leads if lead[1] <= max(far_headway * v_ego, BURST_FAR_MIN_GAP)]
    accel, source = policy0(v_ego, leads, *a, **k)
    if not calls:  # the live call; the plan rollout's follow
      cruise = policy0(v_ego, [], *a[:3], None, *a[4:], **k)[0]  # ty: ignore[too-many-positional-arguments]  # the same call, no leads, no model
      # the set-speed ladder alone: the same, with curve speed, the in-curve hold, the squeeze and the curve-exit cap all lifted
      ladder = policy0(v_ego, [], *a[:3], None, *a[4:], **{**k, "curve": None, "v_hold": math.inf,  # ty: ignore[too-many-positional-arguments]
                                                             "squeeze": longitudinal.ACCEL_MAX, "exit_cap": longitudinal.ACCEL_MAX})[0]
      asks = [(float(gap), float(v_lead), float(extra[-1]) if extra else 0.0, longitudinal.lead_accel(v_ego, gap, v_lead, a_lead, a[1], *extra))
              for _, gap, v_lead, a_lead, *extra in leads]
      calls.append((accel, int(source), cruise, asks, a[3], v_ego, ladder))
    return accel, source

  def update(self, sm):
    calls.clear()
    update0(self, sm)
    frames.append(calls[0] if calls else None)

  longitudinal.policy, longitudinal.MoonpilotLongitudinalPlanner.update = policy, update  # ty: ignore[invalid-assignment]
  try:
    details = []
    replayed, logged, active, _ = replay(route, extract(route, root), root, details_out=details)
  finally:
    longitudinal.policy, longitudinal.MoonpilotLongitudinalPlanner.update = policy0, update0
  return replayed, logged, active, _details_arrays(details), frames


def _burst_starts(a, mask):
  """Where each burst starts, inside `mask` (engaged, above 5 m/s) only and never across its edge."""
  n, out = round(BURST_T / DT_MDL), []
  for r in runs(mask):
    i = r.start
    while i < r.stop - n:
      if a[i + n] > BURST_END and a[i + n] - a[i] >= BURST_RISE:
        out.append(i)
        i += n
      else:
        i += 1
  return out


def _unmatched_starts(logged, a_ego, v_ego, active):
  """Measured bursts (aEgo over 0.5 s) no logged-command burst starts within `BURST_T` of, engaged above 5 m/s: unmatched,
  not a cause — a smaller or slower command ramp, a response delayed past the window, or aEgo noise reads the same."""
  mask = active & np.isfinite(logged) & (v_ego > 5.0)
  car = np.convolve(a_ego, np.ones(10) / 10, mode="same")
  commanded, n = _burst_starts(logged, mask), round(BURST_T / DT_MDL)
  return [c for c in _burst_starts(car, mask) if not any(abs(c - k) <= n for k in commanded)]


def _logged_frames(route, root=DEFAULT_ROOT):
  """The logged command and the car, one row per planner tick of the route's extract, read without replaying: the same
  values `replay`'s details carry for them."""
  data = extract(route, root)
  blob, offsets = data["blob"], data["offsets"]
  index = {k: 1 + i * 3 for i, k in enumerate(MESSAGES)}
  plan_col = 1 + len(MESSAGES) * 3
  rows = []
  for row in offsets:
    def read(kind, row=row):
      start, size = row[index[kind]], row[index[kind] + 1]
      return messaging.log_from_bytes(blob[start:start + size].tobytes())
    cs, cc = read("carState").carState, read("carControl").carControl
    start, size = row[plan_col], row[plan_col + 1]
    a_target = float(messaging.log_from_bytes(blob[start:start + size].tobytes()).longitudinalPlan.aTarget) if size else np.nan
    rows.append((int(row[0]) * 1e-9, a_target, cs.aEgo, cs.vEgo, cc.longActive))
  t, logged, a_ego, v_ego, active = (np.asarray(c, dtype=float) for c in zip(*rows, strict=True))
  return {"t": t, "logged": logged, "a_ego": a_ego, "v_ego": v_ego, "active": active > 0}


BURST_REPRO_TOL = 0.05  # m/s^2; a logged burst is attributed only where today's replay matches it this closely throughout


def _burst_kind(f0, f1, m, i, n, far_headway):
  """What lifted the command over one burst, from the recorded `policy` frames at its start and end: the winning candidate,
  and for a lead the headway and whether its own ask rose or it left the race; for cruise, a lead's runway cap lifting,
  the throttle permission returning, or the set-speed ladder (with its deficit)."""
  names = {int(LongitudinalPlanSource.cruise): "cruise", int(LongitudinalPlanSource.lead0): "lead0",
           int(LongitudinalPlanSource.lead1): "lead1", int(LongitudinalPlanSource.e2e): "model"}
  kind, far, deficit = f"{names.get(f0[1], f0[1])} -> {names.get(f1[1], f1[1])}", False, None
  if f0[1] in (int(LongitudinalPlanSource.lead0), int(LongitudinalPlanSource.lead1)) and f0[3]:
    gap0 = min(f0[3], key=lambda x: x[3])[0]
    far = gap0 / max(f0[5], 0.1) > far_headway and gap0 > BURST_FAR_MIN_GAP
    lead_left = not f1[3] or min(x[3] for x in f1[3]) > f1[2] - 1e-6
    kind += f" ({'far' if far else 'near'} lead, {'lead left the race' if lead_left else 'lead ask rose'})"
  elif f0[1] == f1[1] == int(LongitudinalPlanSource.cruise):
    # which part of the cruise slot let go: a lead's runway cap, the curve terms (curve speed, in-curve hold, squeeze,
    # curve-exit cap), the throttle permission, or the set-speed ladder itself; anything else stays unresolved
    capped_by_lead = f0[0] < f0[2] - 0.05 and f1[0] >= f1[2] - 0.05
    capped_by_curve = f0[2] < f0[6] - 0.05 and f1[2] >= f1[6] - 0.05
    if capped_by_lead:
      kind += " (a lead's runway cap lifted)"
    elif capped_by_curve:
      kind += " (a curve term lifted: curve speed, in-curve hold, squeeze or curve-exit cap)"
    elif not m["plan_allow_throttle"][i] and m["plan_allow_throttle"][i + n]:
      kind += " (throttle permission returned)"
    elif f0[0] >= f0[6] - 0.05 and f1[0] >= f1[6] - 0.05:
      kind += " (the set-speed ladder)"
      deficit = m["v_cruise"][i] / 3.6 - m["v_ego"][i]
    else:
      kind += " (unresolved)"
  return kind, far, deficit


def _accel_bursts(args):
  """Positive accel bursts while openpilot has the pedals above 5 m/s — `driver._bursts`' rise of 0.4 m/s^2 in 1 s ending
  above +0.3 — in the logged command and in the car's own aEgo (over 0.5 s), and what the logged ones came from.

  The drives ran older code, and this replays today's (with each drive's own params): a logged burst is attributed only
  where today's replay reproduces the logged command within `BURST_REPRO_TOL` over the whole burst, so the recorded
  candidates are the ones that produced it; the rest are counted, unexplained. Today's own bursts, and today's with leads
  past `--far-headway` (and `BURST_FAR_MIN_GAP`) ignored, are counterfactual counts. Command path only."""
  from concurrent.futures import ProcessPoolExecutor
  routes = list(dict.fromkeys(expand(args.routes, ROUTE_SETS)))
  n = round(BURST_T / DT_MDL)
  totals, kinds = {"logged": 0, "car": 0, "today (counterfactual)": 0, "today, far leads ignored": 0}, {}
  minutes, errs, deficits, car_only, reproduced, far_count = 0.0, [], [], 0, 0, 0
  with ProcessPoolExecutor(3, max_tasks_per_child=1) as pool:
    base = list(pool.map(_burst_replay, [(r, args.root, None) for r in routes]))
    arm = list(pool.map(_burst_replay, [(r, args.root, args.far_headway) for r in routes]))
  for (replayed, logged, active, m, frames), (dropped, *_) in zip(base, arm, strict=True):
    mask = active & np.isfinite(logged) & (m["v_ego"] > 5.0)
    if not mask.any() or len(frames) != len(replayed):
      continue
    errs.append(np.abs(replayed[mask] - logged[mask]))
    minutes += mask.sum() * DT_MDL / 60
    car = np.convolve(m["a_ego"], np.ones(10) / 10, mode="same")
    commanded = _burst_starts(logged, mask)
    for key, series in (("logged", logged), ("car", car), ("today (counterfactual)", replayed), ("today, far leads ignored", dropped)):
      totals[key] += len(_burst_starts(series, mask))
    car_only += len(_unmatched_starts(logged, m["a_ego"], m["v_ego"], active))  # the same selection `burst-chain` places
    for i in commanded:
      if np.max(np.abs(replayed[i:i + n + 1] - logged[i:i + n + 1])) > BURST_REPRO_TOL or frames[i] is None or frames[i + n] is None:
        continue
      reproduced += 1
      kind, far, deficit = _burst_kind(frames[i], frames[i + n], m, i, n, args.far_headway)
      far_count += far
      if deficit is not None:
        deficits.append(deficit)
      kinds[kind] = kinds.get(kind, 0) + 1
  err = np.concatenate(errs) if errs else np.zeros(1)
  print(f"{len(routes)} drives, {minutes:.1f} engaged min above 5 m/s; today's replay vs the logged command |aTarget error| median "
        + f"{np.median(err):.4f} p95 {np.percentile(err, 95):.4f} m/s^2 — the drives' older code")
  print("bursts per engaged minute: " + ", ".join(f"{k} {v / max(minutes, 1e-9):.2f} ({v})" for k, v in totals.items()))
  print(f"  of the car's {totals['car']}, {car_only} are unmatched: no logged-command burst starts within {BURST_T:g} s "
        + "(a count, not a cause: it needs delay-aligned plan, actuator output and measured accel to say whose they are)")
  print(f"{reproduced} of the {totals['logged']} logged bursts reproduced within {BURST_REPRO_TOL:g} m/s^2 and attributed "
        + f"({far_count} start on a far lead); the other {totals['logged'] - reproduced} are unexplained:")
  for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
    print(f"  {count:3d}  {kind}")
  if deficits:
    print(f"  the ladder's bursts start a median {np.median(deficits):.1f} m/s under the set speed (p25 {np.percentile(deficits, 25):.1f}, "
          + f"p75 {np.percentile(deficits, 75):.1f})")


HANDBACK_WINDOW_S, HANDBACK_GAS_S = 4.0, 0.5


def _handback_replay(job):
  route, root, on = job
  details = []
  replayed, logged, active, _ = replay(route, extract(route, root), root, overrides={"MoonpilotHandbackEase": int(on)}, details_out=details)
  return replayed, logged, active, _details_arrays(details)


def _handback_edges(active, gas, v_ego, t):
  """Gas-led hand-backs: openpilot's longitudinal inactive to active, the gas pressed in the `HANDBACK_GAS_S` before, above
  5 m/s, and active, timestamp-contiguous, for the `HANDBACK_WINDOW_S` after."""
  n, pre = round(HANDBACK_WINDOW_S / DT_MDL), round(HANDBACK_GAS_S / DT_MDL)
  out = []
  for e in np.flatnonzero(active[1:] & ~active[:-1]) + 1:
    w = slice(e, e + n + 1)
    if e < pre or e + n >= len(active) or v_ego[e] <= 5.0 or not gas[e - pre:e].any() or not active[w].all():
      continue
    if np.max(np.diff(t[w])) > 1.5 * DT_MDL:
      continue
    out.append(e)
  return out


def _handback(args):
  """Every gas-led hand-back on the engaged drives, today's planner replayed with `MoonpilotHandbackEase` off and on: the
  command's largest rise in any 1 s of the `HANDBACK_WINDOW_S` after, the bursts in it (`driver._bursts`' definition), the
  command at +1 s, its peak and its mean (the speed it buys). The off arm against the logged command is the gate: those
  drives had no ease. Command path only: the car is held as recorded, so the ease's own effect on speed is not flown."""
  from concurrent.futures import ProcessPoolExecutor
  routes = list(dict.fromkeys(expand(args.routes, ROUTE_SETS)))
  with ProcessPoolExecutor(3, max_tasks_per_child=1) as pool:
    off = list(pool.map(_handback_replay, [(r, args.root, False) for r in routes]))
    on = list(pool.map(_handback_replay, [(r, args.root, True) for r in routes]))
  n, one = round(HANDBACK_WINDOW_S / DT_MDL), round(1.0 / DT_MDL)
  rows, errs = {"off": [], "on": []}, []
  for (cmd_off, logged, active, m), (cmd_on, *_) in zip(off, on, strict=True):
    for e in _handback_edges(active, m["gas_pressed"], m["v_ego"], m["t"]):
      w = slice(e, e + n + 1)
      errs.append(np.abs(cmd_off[w] - logged[w]))
      inside = np.zeros(len(active), dtype=bool)
      inside[w] = True
      for arm, cmd in (("off", cmd_off), ("on", cmd_on)):
        c = cmd[w]
        rows[arm].append((float(np.max(c[one:] - c[:-one])), len(_burst_starts(cmd, inside)), c[one], c.max(), c.mean()))
  if not rows["off"]:
    print(f"no gas-led hand-backs on {len(routes)} drives")
    return
  err = np.concatenate(errs)
  print(f"{len(rows['off'])} gas-led hand-backs on {len(routes)} drives; the off arm against the logged command over their "
        + f"{HANDBACK_WINDOW_S:g} s: |error| median {np.median(err):.4f} p95 {np.percentile(err, 95):.4f} m/s^2")
  print(f"  {'':4s} {'largest 1 s rise':>17s} {'bursts':>7s} {'at +1 s':>8s} {'peak':>7s} {'mean':>7s}   (medians; bursts summed)")
  for arm, x in rows.items():
    x = np.asarray(x)
    print(f"  {arm:4s} {np.median(x[:, 0]):+17.2f} {int(x[:, 1].sum()):7d} {np.median(x[:, 2]):+8.2f} {np.median(x[:, 3]):+7.2f} {np.median(x[:, 4]):+7.2f}")


def register(sub):
  parser = sub.add_parser("accel-bursts", help="engaged positive accel bursts and what they come from; far leads ignored as an arm")
  parser.add_argument("--routes", nargs="+", default=["engaged", "00000468"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--far-headway", type=float, default=4.0, help="s of headway past which a lead counts as far (and is ignored in the arm)")
  parser.set_defaults(fn=_accel_bursts)
  parser = sub.add_parser("burst-chain", help="each measured engaged accel burst against the delay-aligned plan, controller and PCM command")
  parser.add_argument("--routes", nargs="+", default=["engaged", "00000468"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_burst_chain)
  parser.add_argument("--list", action="store_true", help="every placed unmatched event, by class, drive and time")
  parser = sub.add_parser("handback", help="gas-led hand-backs replayed with MoonpilotHandbackEase off and on")
  parser.add_argument("--routes", nargs="+", default=["engaged", "00000468"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_handback)
