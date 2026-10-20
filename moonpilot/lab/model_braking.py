"""model-braking: how often the model's own braking asks for more than the lead and the geometry justify.

`MoonpilotModelBraking` admits `modelV2.action.desiredAcceleration` into the planner once it asks for
`MOONPILOT_MODEL_BRAKE_THRESHOLD` or deeper. This measures what that admits, on drives the fork ran
and on the driver's own driving:

- engaged episodes (longActive, logged plan source e2e, ask at or below the threshold above 10 m/s): the
  lead, what the lead terms alone would have asked (the fork planner replayed with the model off), the model's
  own plan, and what the driver did next;
- manual episodes (no cruise, in drive, model ask at or below the threshold above 10 m/s): whether the driver
  slowed too, per hour of manual driving, with and without a slower radar lead inside 50 m;
- both, as the admission threshold is raised above 15 m/s.

An episode is a run of qualifying frames, runs closer than MERGE_S joined. Frames are the drive's modelV2
ticks (20 Hz). Logs hold no traffic-light state and no map: a context like an intersection cannot be told
from the signals here, only from the video.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.tools.lib.logreader import LogReader
from moonpilot.lab import DEFAULT_ROOT, cached, expand, runs
from moonpilot.lab import planner
from moonpilot.lab.planner import MESSAGES, extract, replay

ROUTE_SETS = {
  "engaged": (*planner.ROUTE_SETS["longitudinal"], "0000044d"),
  "manual": planner.ROUTE_SETS["manual-approach"],
}
VERSION = 1
WARMUP_S = 15.0  # replay lead-in before each engaged episode
MERGE_S = 1.0  # runs of qualifying frames closer than this are one episode
ASK = -0.5  # m/s^2; the shipped admission threshold, and the episode definition
V_MIN = 10.0  # m/s
V_SENS = 15.0  # m/s; the speed above which the threshold sweep applies
THRESHOLDS = (-0.5, -0.8, -1.0, -1.5)
REACT_ENGAGED_S = 5.0  # driver response window past an engaged episode's end
REACT_MANUAL_S = 3.0  # driver response window from a manual episode's start
LEAD_NEAR_M = 50.0
SLOWED_A = -0.3  # m/s^2; mean aEgo over the window that counts as the driver slowing
UNJUSTIFIED_A = -0.25  # m/s^2; the lead terms alone ask less braking than this
KEYS = ("t", "v", "a", "gas", "brake", "cruise", "active", "drive", "lead", "gap", "v_lead", "a_lead", "m_acc", "m_stop", "m_vmin",
        "m_lead_a0", "m_lead_prob", "e2e", "plan_a")


def episodes(mask: np.ndarray, t: np.ndarray, group: np.ndarray, merge_s: float = MERGE_S) -> list[slice]:
  """Runs of `mask` as episodes: runs in the same group (drive) no more than `merge_s` apart in time are joined."""
  pieces = []
  for r in runs(mask):  # a run is cut where the group changes, then pieces are joined within their group
    cuts = (np.flatnonzero(group[r.start + 1:r.stop] != group[r.start:r.stop - 1]) + r.start + 1).tolist()
    pieces += [slice(a, b) for a, b in zip([r.start, *cuts], [*cuts, r.stop], strict=True)]
  merged: list[slice] = []
  for r in pieces:
    last = merged[-1] if merged else None
    if last is not None and group[r.start] == group[last.stop - 1] and t[r.start] - t[last.stop - 1] <= merge_s:
      merged[-1] = slice(last.start, r.stop)
    else:
      merged.append(r)
  return merged


def _build(route, root, segs):
  t0 = min(m.logMonoTime for m in islice(LogReader(segs[0]), 200))
  data = extract(route, root)
  blob, offsets = data["blob"], data["offsets"]
  index = {k: 1 + i * 3 for i, k in enumerate(MESSAGES)}
  rows = []
  for row in offsets:
    def read(kind, row=row):
      start, size = row[index[kind]], row[index[kind] + 1]
      return getattr(messaging.log_from_bytes(blob[start:start + size].tobytes()), kind)
    cs, cc, lead, mv = read("carState"), read("carControl"), read("radarState").leadOne, read("modelV2")
    plan_start, plan_size = row[1 + len(MESSAGES) * 3:3 + len(MESSAGES) * 3]
    plan = messaging.log_from_bytes(blob[plan_start:plan_start + plan_size].tobytes()).longitudinalPlan if plan_size else None
    one = mv.leadsV3[0] if len(mv.leadsV3) else None
    rows.append((
      (row[0] - t0) * 1e-9, cs.vEgo, cs.aEgo, cs.gasPressed, cs.brakePressed, cs.cruiseState.enabled, cc.longActive,
      cs.gearShifter == car.CarState.GearShifter.drive, lead.present and lead.radar, lead.dRel, lead.vLead, lead.aLeadK,
      mv.action.desiredAcceleration, mv.action.shouldStop, min(mv.velocity.x) if len(mv.velocity.x) else np.nan,
      one.a[0] if one is not None and len(one.a) else np.nan, one.prob if one is not None else np.nan,
      plan is not None and str(plan.longitudinalPlanSource) == "e2e", plan.aTarget if plan is not None else np.nan,
    ))
  return {k: np.asarray(col, dtype=float) for k, col in zip(KEYS, zip(*rows, strict=True), strict=True)}


def frames(route, root=DEFAULT_ROOT):
  """One row per modelV2 tick of the route's planner extract, `KEYS`; `t` is seconds since the route's first log message."""
  return cached(route, "model-braking", VERSION, root, lambda segs: _build(route, root, segs))


def _route(job):
  """One route's frames and, for engaged routes, the planner replayed with the drive's configuration and with the model
  off over each episode (WARMUP_S before it, so the planner's own state has settled) — NaN elsewhere."""
  route, root, with_replay = job
  d = dict(frames(route, root))
  d.pop("segments")
  if with_replay:
    data = extract(route, root)
    d["replayed"], d["lead_only"] = np.full(len(d["t"]), np.nan), np.full(len(d["t"]), np.nan)
    n, lead_in = len(d["t"]), round(WARMUP_S / 0.05)
    windows = [slice(max(sl.start - lead_in, 0), min(sl.stop + 40, n)) for sl in episodes(_mask_engaged(d), d["t"], np.zeros(n))]
    merged = []
    for w in windows:
      if merged and w.start <= merged[-1].stop:
        merged[-1] = slice(merged[-1].start, max(w.stop, merged[-1].stop))
      else:
        merged.append(w)
    for w in merged:
      part = {"cp": data["cp"], "blob": data["blob"], "offsets": data["offsets"][w]}
      d["replayed"][w] = replay(route, part, root)[0]
      d["lead_only"][w] = replay(route, part, root, overrides={"MoonpilotModelBraking": 0})[0]
  return d


def _pool(routes, root, with_replay):
  """Every route's frames end to end, `route` numbering them so episodes never span two drives."""
  with ProcessPoolExecutor(max_workers=4) as pool:
    parts = list(pool.map(_route, [(r, root, with_replay) for r in routes]))
  for i, p in enumerate(parts):
    p["route"] = np.full(len(p["t"]), i)
  return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _dt(d):
  return np.clip(np.diff(d["t"], append=d["t"][-1] + 0.05), 0.0, 0.1)


def _mask_engaged(d, thr=None, v_min=V_MIN):
  """Engaged e2e frames asking ASK or deeper; with `thr`, only those whose model ask is also at or below it."""
  mask = (d["active"] > 0) & (d["e2e"] > 0) & (d["plan_a"] <= ASK) & (d["v"] > v_min)
  return mask if thr is None else mask & (d["m_acc"] <= thr)


def _mask_manual(d, thr=ASK, v_min=V_MIN):
  return (d["active"] == 0) & (d["cruise"] == 0) & (d["drive"] > 0) & (d["m_acc"] <= thr) & (d["v"] > v_min)


def _response_engaged(d, sl):
  """First of brake press, cruise cancel (cruise falls with no pedal down), gas press, from the episode's start to
  REACT_ENGAGED_S past its end; 'nothing' when none."""
  t = d["t"]
  ix = np.arange(sl.start, min(len(t), sl.stop + 120))  # 120 ticks > REACT_ENGAGED_S at 20 Hz
  ix = ix[(d["route"][ix] == d["route"][sl.start]) & (t[ix] <= t[sl.stop - 1] + REACT_ENGAGED_S)]
  cruise = d["cruise"][ix] > 0
  hits = {
    "brake": d["brake"][ix] > 0,
    "cancel": np.r_[False, cruise[:-1] & ~cruise[1:]] & (d["brake"][ix] == 0) & (d["gas"][ix] == 0),
    "gas": d["gas"][ix] > 0,
  }
  first = {k: int(np.argmax(h)) for k, h in hits.items() if h.any()}
  if not first:
    return "nothing", np.nan
  name = min(first, key=lambda k: (first[k], tuple(hits).index(k)))
  return name, float(t[ix[first[name]]] - t[sl.start])


def _response_manual(d, sl):
  """'slowed' (a brake press or mean aEgo <= SLOWED_A in the first REACT_MANUAL_S), 'did not' (gas held half the window
  or mean aEgo > -0.1), else 'between'."""
  t, i0 = d["t"], sl.start
  ix = np.arange(i0, min(len(t), i0 + 80))  # 80 ticks > REACT_MANUAL_S at 20 Hz
  ix = ix[(d["route"][ix] == d["route"][i0]) & (t[ix] <= t[i0] + REACT_MANUAL_S)]
  mean_a = float(np.mean(d["a"][ix]))
  if d["brake"][ix].any() or mean_a <= SLOWED_A:
    return "slowed"
  if d["gas"][ix].mean() >= 0.5 or mean_a > -0.1:
    return "did not"
  return "between"


def _lead_text(d, i):
  if not d["lead"][i]:
    return "none"
  return f"{d['gap'][i]:5.1f}m cl{d['v'][i] - d['v_lead'][i]:+4.1f} a{d['a_lead'][i]:+5.2f}"


def _engaged_rows(d):
  mask, dt = _mask_engaged(d), _dt(d)
  out = []
  for sl in episodes(mask, d["t"], d["route"]):
    m = mask[sl]
    seg = np.arange(sl.start, sl.stop)[m]
    peak = seg[int(np.argmin(d["plan_a"][seg]))]
    out.append((sl, peak, float(dt[sl][m].sum()), seg))
  return out


def _engaged(args, routes):
  d = _pool(routes, args.root, with_replay=True)
  dt = _dt(d)
  active, base = d["active"] > 0, _mask_engaged(d)
  fast = active & (d["v"] > V_MIN)
  print(f"engaged: {len(routes)} routes, {dt[active].sum() / 60:.1f} min longActive, {dt[fast].sum() / 60:.1f} min above {V_MIN:g} m/s")
  print("\nper route: engaged s | e2e-source s | e2e s with plan > +0.2 (only experimental mode can) | episodes")
  print("  | replay-vs-logged |dA| mean, p95 over the episode frames")
  rows = _engaged_rows(d)
  for i, route in enumerate(routes):
    r = d["route"] == i
    a = r & active & (d["e2e"] > 0)
    err = np.abs(d["replayed"][r & base] - d["plan_a"][r & base])
    mean, p95 = (err.mean(), np.percentile(err, 95)) if len(err) else (np.nan, np.nan)
    n = sum(d["route"][sl.start] == i for sl, *_ in rows)
    print(f"  {route}  {dt[r & active].sum():7.0f}  {dt[a].sum():6.0f}  {dt[a & (d['plan_a'] > 0.2)].sum():6.0f}  {n:3d}  {mean:.3f} {p95:.3f}")
  print(f"\nengaged episodes: e2e source, plan <= {ASK:g}, v > {V_MIN:g} m/s, {len(rows)} total")
  print("  cl = closing speed v - vLead, a = radar aLeadK, at the plan's peak frame")
  print("  lead-only = replay with MoonpilotModelBraking=0, at the peak and mean over the episode; extra = peak plan - lead-only at the peak")
  print("  model v = min of modelV2.velocity.x over 10 s | stop = action.shouldStop | a0 = leadsV3[0].a[0] (prob)")
  print(f"  lead-in = seconds the source was already e2e before the first frame (the slot opening before the plan reached {ASK:g})")
  print(f"  driver = first of brake/cancel/gas from start to end + {REACT_ENGAGED_S:g} s")
  cols = f"{'route':8} {'t':>7} {'dur':>5} {'peak':>6} {'v':>5}  {'lead':26} {'lead-only@pk':>12} {'mean':>5} {'extra':>6}"
  print(f"  {cols}  {'model v':>7} {'stop':>4} {'a0(prob)':>12} {'lead-in':>7}  driver")
  tally: dict[str, int] = {}
  unjustified, extra = 0, []
  for sl, peak, _secs, seg in rows:
    name, when = _response_engaged(d, sl)
    tally[name] = tally.get(name, 0) + 1
    lead_only, plan_peak = float(d["lead_only"][peak]), float(d["plan_a"][peak])
    extra.append(plan_peak - lead_only)
    unjustified += lead_only > UNJUSTIFIED_A
    i = sl.start
    while i > 0 and d["e2e"][i - 1] > 0 and d["route"][i - 1] == d["route"][sl.start]:
      i -= 1
    head = f"{routes[int(d['route'][peak])]:8} {d['t'][peak]:7.1f} {d['t'][sl.stop - 1] - d['t'][sl.start] + 0.05:5.1f} {plan_peak:6.2f}"
    lead = f"{d['v'][peak]:5.1f}  {_lead_text(d, peak):26} {lead_only:12.2f} {np.mean(d['lead_only'][seg]):5.2f} {extra[-1]:+6.2f}"
    model = f"{d['m_vmin'][peak]:7.1f} {int(d['m_stop'][peak]):4d} {d['m_lead_a0'][peak]:+6.2f}({d['m_lead_prob'][peak]:.2f})"
    print(f"  {head} {lead}  {model} {d['t'][sl.start] - d['t'][i]:6.1f}s  {name}{f' +{when:.1f}s' if name != 'nothing' else ''}")
  print(f"\n  driver in the window: {tally}")
  print(f"  lead-only at the peak above {UNJUSTIFIED_A:g} (lead terms alone not braking): {unjustified} of {len(rows)}")
  more = [sum(e <= -x for e in extra) for x in (0.3, 0.5, 1.0)]
  print(f"  model asked >= 0.3 / 0.5 / 1.0 m/s^2 more than lead-only at the peak: {more[0]} / {more[1]} / {more[2]} of {len(rows)}")
  print(f"  no radar lead at the peak: {sum(1 for _, p, *_ in rows if not d['lead'][p])} of {len(rows)}")
  print(f"\nengaged threshold sweep above {V_SENS:g} m/s (model ask <= threshold; plan e2e and <= {ASK:g}):")
  print(f"  {'thr':>5} {'episodes':>8} {'seconds':>8} {'s lead-only > ' + format(UNJUSTIFIED_A, 'g'):>18}")
  for thr in THRESHOLDS:
    mask = _mask_engaged(d, thr, V_SENS)
    unjust = mask & (d["lead_only"] > UNJUSTIFIED_A)
    print(f"  {thr:5.1f} {len(episodes(mask, d['t'], d['route'])):8d} {dt[mask].sum():8.1f} {dt[unjust].sum():18.1f}")


def _responses(resp):
  return {k: resp.count(k) for k in ("slowed", "between", "did not")}


def _manual(args, routes):
  d = _pool(routes, args.root, with_replay=False)
  dt = _dt(d)
  manual = (d["active"] == 0) & (d["cruise"] == 0) & (d["drive"] > 0)
  near = (d["lead"] > 0) & (d["gap"] < LEAD_NEAR_M) & (d["v_lead"] < d["v"])
  contexts = {"slower lead < 50 m": near, "farther / none": ~near}
  fast_h = dt[manual & (d["v"] > V_MIN)].sum() / 3600
  print(f"\nmanual: {len(routes)} routes, {dt[manual].sum() / 3600:.2f} h manual in drive, {fast_h:.2f} h above {V_MIN:g} m/s")
  mask = _mask_manual(d, ASK, V_MIN)
  eps = episodes(mask, d["t"], d["route"])
  peaks = [int(np.arange(sl.start, sl.stop)[mask[sl]][np.argmin(d["m_acc"][sl][mask[sl]])]) for sl in eps]
  resp = [_response_manual(d, sl) for sl in eps]
  print(f"\nmanual episodes above {V_MIN:g} m/s, model ask <= {ASK:g}: {len(eps)}; response {_responses(resp)}")
  print(f"  slowed = brake press or mean aEgo <= {SLOWED_A:g} in the first {REACT_MANUAL_S:g} s; did not = gas on half the window or mean aEgo > -0.1")
  print(f"  {'context (at peak)':20} {'hours':>6} {'episodes':>8} {'per hour':>8}   {'slowed':>6} {'between':>7} {'did not':>7}  {'did-not per hour':>16}")
  for name, ctx in contexts.items():
    hours = max(dt[manual & (d["v"] > V_MIN) & ctx].sum() / 3600, 1e-9)
    mine = [r for p, r in zip(peaks, resp, strict=True) if ctx[p]]
    n = _responses(mine)
    counts = f"{n['slowed']:6d} {n['between']:7d} {n['did not']:7d}"
    print(f"  {name:20} {hours:6.2f} {len(mine):8d} {len(mine) / hours:8.1f}   {counts}  {n['did not'] / hours:16.1f}")
  hours = max(dt[manual & (d["v"] > V_SENS)].sum() / 3600, 1e-9)
  print(f"\nmanual threshold sweep above {V_SENS:g} m/s (model ask <= threshold; per hour of manual driving above {V_SENS:g} m/s, {hours:.2f} h):")
  print(f"  {'thr':>5} {'episodes':>8} {'seconds':>8} {'per hour':>8}   {'slowed':>6} {'between':>7} {'did not':>7}  {'did-not %':>9}")
  for thr in THRESHOLDS:
    mask = _mask_manual(d, thr, V_SENS)
    eps = episodes(mask, d["t"], d["route"])
    n = _responses([_response_manual(d, sl) for sl in eps])
    did_not = 100 * n["did not"] / max(len(eps), 1)
    print(f"  {thr:5.1f} {len(eps):8d} {dt[mask].sum():8.1f} {len(eps) / hours:8.1f}   {n['slowed']:6d} {n['between']:7d} {n['did not']:7d}  {did_not:9.1f}")


def _run(args):
  _engaged(args, expand(args.engaged, ROUTE_SETS))
  _manual(args, expand(args.manual, ROUTE_SETS))


def register(sub):
  parser = sub.add_parser("model-braking", help="how often the model's own braking asks for more than the lead and geometry justify")
  parser.add_argument("--engaged", nargs="+", default=["engaged"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--manual", nargs="+", default=["manual"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_run)
