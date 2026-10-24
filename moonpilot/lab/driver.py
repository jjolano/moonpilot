"""Measure the driver's manual longitudinal policy from every cached route."""
import glob
import math
import os
import sys
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from opendbc.car.structs import car
from openpilot.tools.lib.logreader import LogReader
from moonpilot.lab import DEFAULT_ROOT, cached, runs
from moonpilot.lead import LeadAccelEstimator

VERSION = 3
DT = 0.05
FIELDS = (
  "t", "manual", "long_active", "v", "a", "gas", "brake", "gas_analog", "brake_analog", "standstill", "cruise",
  "steer", "steer_rate", "yaw_rate", "left_blinker", "right_blinker", "pitch", "lead", "lead_radar", "gap", "v_lead",
  "a_lead_k", "lead_y", "lead_prob", "track_id", "lead2", "gap2", "v_lead2", "a_lead", "a_slow", "desired_a",
  "should_stop", "v_2s", "v_4s", "brake_prob", "gas_prob", "leads_prob", "curvature_1s", "curvature_2s", "left_line_prob",
  "right_line_prob",
)


def _value(obj, name, default=math.nan):
  try:
    return getattr(obj, name)
  except (AttributeError, TypeError):
    return default


def _sequence(obj, name):
  try:
    return getattr(obj, name)
  except (AttributeError, TypeError):
    return ()


def _number(obj, name, default=math.nan):
  try:
    value = getattr(obj, name)
    return float(value) if value is not None else default
  except (AttributeError, TypeError, ValueError):
    return default


def _at_time(obj, field, seconds):
  values, times = _sequence(obj, field), _sequence(obj, "t")
  n = min(len(values), len(times))
  if not n:
    return math.nan
  index = min(range(n), key=lambda i: abs(float(times[i]) - seconds))
  return float(values[index]) if abs(float(times[index]) - seconds) <= 0.3 else math.nan


def _curvature(orient_rate, seconds, speed):
  return _at_time(orient_rate, "z", seconds) / speed


def _first_probability(obj, field):
  values = _sequence(obj, field)
  return float(values[0]) if values else math.nan


def _build(segs):
  latest, rows, estimator = {}, [], LeadAccelEstimator()
  for seg in segs:
    segment_rows, frame_error = len(rows), None
    latest.clear()
    try:
      messages = LogReader([seg], sort_by_time=True)
      for msg in messages:
        kind = msg.which()
        if kind in ("carState", "carControl", "radarState", "modelV2"):
          latest[kind] = msg
        if kind != "modelV2" or not all(k in latest for k in ("carState", "carControl", "radarState")):
          continue
        try:
          cs, cc = latest["carState"].carState, latest["carControl"].carControl
          rs, model = latest["radarState"].radarState, msg.modelV2
          lead, lead2 = rs.leadOne, rs.leadTwo
          a_lead, _ = estimator.update(lead)
          lead_present = bool(lead.present)
          manual = not cc.longActive and not cs.cruiseState.enabled and cs.gearShifter == car.CarState.GearShifter.drive
          action = model.action
          desired = _number(action, "desiredAcceleration")
          should_stop = float(bool(_value(action, "shouldStop", False)))
          velocity = _value(model, "velocity", None)
          v2 = _at_time(velocity, "x", 2.0)
          v4 = _at_time(velocity, "x", 4.0)
          orient_rate = _sequence(model, "orientationRate")
          speed = max(_number(cs, "vEgo", 0.0), 0.1)
          meta = model.meta
          disengage = _value(meta, "disengagePredictions", None)
          lines = _sequence(model, "laneLineProbs")
          leads = _sequence(model, "leadsV3")
          pitch = math.nan
          try:
            pitch = float(cc.orientationNED[1])
          except (AttributeError, IndexError, TypeError):
            pass
          row = (
            int(msg.logMonoTime) * 1e-9, manual, bool(cc.longActive), _number(cs, "vEgo"), _number(cs, "aEgo"),
            float(bool(_value(cs, "gasPressed", False))), float(bool(_value(cs, "brakePressed", False))), _number(cs, "gas"), _number(cs, "brake"),
            float(bool(_value(cs, "standstill", False))), float(bool(cs.cruiseState.enabled)), _number(cs, "steeringAngleDeg"),
            _number(cs, "steeringRateDeg"), _number(cs, "yawRate"), float(bool(_value(cs, "leftBlinker", False))),
            float(bool(_value(cs, "rightBlinker", False))), pitch, float(lead_present), float(bool(lead.radar)) if lead_present else 0.0,
            _number(lead, "dRel"), _number(lead, "vLead"), _number(lead, "aLeadK"), _number(lead, "yRel"),
            _number(lead, "modelProb"), float(_value(lead, "radarTrackId", math.nan)), float(bool(lead2.present)),
            _number(lead2, "dRel"), _number(lead2, "vLead"), a_lead if lead_present else math.nan,
            estimator.a_slow if lead_present else math.nan, desired, should_stop, v2, v4,
            _first_probability(disengage, "brakePressProbs") if disengage is not None else math.nan,
            _first_probability(disengage, "gasPressProbs") if disengage is not None else math.nan,
            _number(leads[0], "prob") if leads else math.nan,
            _curvature(orient_rate, 1.0, speed), _curvature(orient_rate, 2.0, speed),
            float(lines[0]) if len(lines) else math.nan, float(lines[-1]) if len(lines) > 1 else math.nan,
          )
          rows.append(row)
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError) as error:
          if frame_error is None:
            frame_error = str(error)
          continue
      if len(rows) == segment_rows and frame_error is not None:
        print(f"SKIP frames in {seg}: {frame_error}", file=sys.stderr)
    except Exception as error:
      print(f"SKIP segment {seg}: {error}", file=sys.stderr)
  if not rows:
    raise ValueError("no modelV2-aligned frames")
  return {name: np.asarray([row[i] for row in rows], dtype=float) for i, name in enumerate(FIELDS)}

def extract(route, root=DEFAULT_ROOT):
  def build(segs):
    data = _build(segs)
    data["route"] = np.full(len(data["t"]), route)
    return data
  return cached(route, "driver", VERSION, root, build)


def _route_job(job):
  route, root = job
  try:
    return route, extract(route, root), None
  except Exception as error:
    return route, None, str(error)


def _routes(root):
  return sorted({os.path.basename(os.path.dirname(p)).rsplit("--", 1)[0] for p in glob.glob(os.path.join(root, "*--*", "rlog.zst"))})


def _rank(x):
  order = np.argsort(x, kind="mergesort")
  sorted_x = x[order]
  _, first, counts = np.unique(sorted_x, return_index=True, return_counts=True)
  ranks = np.repeat(first + (counts - 1) / 2, counts)
  out = np.empty(len(x), dtype=float)
  out[order] = ranks
  return out


def _corr(x, y):
  mask = np.isfinite(x) & np.isfinite(y)
  if mask.sum() < 30 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
    return math.nan
  return float(np.corrcoef(_rank(x[mask]), _rank(y[mask]))[0, 1])


def _report(args):
  routes = _routes(args.root)
  if not routes:
    raise ValueError(f"no rlog routes under {args.root}")
  failed, all_data = [], []
  with ProcessPoolExecutor(max_workers=6) as pool:
    for route, data, error in pool.map(_route_job, [(r, args.root) for r in routes]):
      if error is not None:
        failed.append((route, error))
      else:
        all_data.append((route, data))
  print(f"routes={len(all_data)} ok, {len(failed)} failed; tick=~20 Hz; segments with no rlog excluded")
  for route, error in failed:
    print(f"SKIP {route}: {error}")
  if not all_data:
    raise ValueError("no routes extracted successfully")
  rows = []
  for route, d in all_data:
    n = len(d["t"])
    manual = d["manual"] > 0
    lead = manual & (d["lead"] > 0) & (d["lead_radar"] > 0)
    rows.append((route, n, int(manual.sum()), int(lead.sum())))
  print("route,frames,manual_min,radar_lead_min")
  for route, n, manual_n, lead_n in rows:
    print(f"{route},{n},{manual_n * DT / 60:.2f},{lead_n * DT / 60:.2f}")
  data = {key: np.concatenate([d[key] for _, d in all_data]) for key in FIELDS}
  data["route"] = np.concatenate([d["route"] for _, d in all_data])
  data["route_id"] = np.concatenate([np.full(len(d["t"]), i) for i, (_, d) in enumerate(all_data)])
  manual = (data["manual"] > 0) & np.isfinite(data["a"])
  lead_manual_minutes = (manual & (data["lead"] > 0) & (data["lead_radar"] > 0)).sum() * DT / 60
  print(f"TOTAL frames={sum(x[1] for x in rows)} manual={manual.sum() * DT / 60:.2f} min; radar lead={lead_manual_minutes:.2f} min")
  print("field,non_null_share")
  for key in FIELDS:
    print(f"{key},{np.isfinite(data[key]).mean():.4f}")
  data["closing"] = data["v"] - data["v_lead"]
  data["headway"] = data["gap"] / np.maximum(data["v"], 0.1)
  data["ttc"] = np.divide(data["gap"], data["closing"], out=np.full(len(manual), np.nan), where=data["closing"] > 0)
  target_gap = np.maximum(2.0, 1.5 * data["v_lead"])
  data["required_decel"] = np.divide(
    data["closing"] ** 2, 2 * (data["gap"] - target_gap),
    out=np.full(len(manual), np.nan), where=data["gap"] > target_gap,
  )
  data["lateral_accel"] = data["curvature_1s"] * data["v"] ** 2
  specs = (("free road", manual & (data["lead"] == 0)),
           ("steady following", manual & (data["lead"] > 0) & (np.abs(data["closing"]) < 1.0)),
           ("approaching slower lead", manual & (data["lead"] > 0) & (data["closing"] > 1.0)),
           ("stopping behind lead", manual & (data["lead_radar"] > 0) & (data["v"] < 2.0) & (data["v_lead"] < 1.0)),
           ("launching behind lead", manual & (data["lead_radar"] > 0) & (data["v"] < 5.0) & (data["v_lead"] > data["v"] + 1.0)),
           ("curves", manual & (np.abs(data["lateral_accel"]) > 1.0)))
  a_half = np.full(len(manual), np.nan)
  for route_id in np.unique(data["route_id"]):
    indices = np.flatnonzero(data["route_id"] == route_id)
    if len(indices) >= 10:
      values = data["a"][indices]
      sums = np.convolve(values, np.ones(10) / 10, mode="valid")
      a_half[indices[:len(sums)]] = sums
  data["a_half"] = a_half
  jerk = np.full(len(manual), np.nan)
  for route_id in np.unique(data["route_id"]):
    indices = np.flatnonzero(data["route_id"] == route_id)
    if len(indices) > 1:
      jerk[indices[1:]] = np.diff(data["a"][indices]) / DT
  data["jerk"] = jerk
  print("\nManual regime smoothness: |jerk| p50/p90/p99, pedal transitions/min, accel sign reversals/min")
  for label, mask in (("all manual", manual), *specs):
    j = np.abs(data["jerk"][mask & np.isfinite(data["jerk"])])
    edges = mask[1:] & mask[:-1] & (data["route_id"][1:] == data["route_id"][:-1])
    transition = (data["gas"][1:] != data["gas"][:-1]) | (data["brake"][1:] != data["brake"][:-1])
    reversals = data["a"][1:] * data["a"][:-1] < 0
    minutes = mask.sum() * DT / 60
    jerk_p = np.percentile(j, [50, 90, 99]) if len(j) else [math.nan] * 3
    pedal_rate = np.sum(transition & edges) / max(minutes, DT / 60)
    reversal_rate = np.sum(reversals & edges) / max(minutes, DT / 60)
    print(f"{label:24s} n={mask.sum():8d} jerk={jerk_p} pedal/min={pedal_rate:.2f} reversals/min={reversal_rate:.2f}")
  route_edges = data["route_id"][1:] == data["route_id"][:-1]
  gas_on = (data["gas"][1:] > 0) & (data["gas"][:-1] == 0)
  pulls_away = (data["lead_radar"][1:] > 0) & (data["v_lead"][1:] > data["v"][1:] + 0.5)
  launch_from_stop = (data["v"][:-1] < 0.3) & (data["v"][1:] > 0.5)
  lead_minutes = (manual & (data["lead_radar"] > 0)).sum() * DT / 60
  gas_launches = np.sum(gas_on & pulls_away & route_edges)
  print(f"launch events: gas-on after lead pull-away={gas_launches} "
        + f"({gas_launches/max(lead_minutes,DT/60):.2f}/lead-min); "
        + f"launch from standstill={np.sum(launch_from_stop & route_edges)}")
  print("\nInterpretable least-squares accel model by regime (speed, closing, gap error to 1.5 s target, lead accel)")
  for label, mask in specs:
    lead_target_error = data["gap"] - np.maximum(2.0, 1.5 * data["v_lead"])
    x = np.column_stack((np.ones(len(manual)), data["v"], data["closing"], lead_target_error, data["a_lead"]))
    good = mask & np.isfinite(x).all(axis=1) & np.isfinite(data["a"])
    if good.sum() > 100:
      coeff, *_ = np.linalg.lstsq(x[good], data["a"][good], rcond=None)
      residual = data["a"][good] - x[good] @ coeff
      r2 = 1 - np.sum(residual**2) / np.sum((data["a"][good] - np.mean(data["a"][good]))**2)
      print(f"{label:24s} n={good.sum()} R2={r2:.3f} residual_sd={np.std(residual):.3f} coefficients={np.round(coeff,3).tolist()}")
  variables = ("v", "v_lead", "gap", "headway", "closing", "ttc", "required_decel", "a_lead", "a_slow", "pitch",
               "lateral_accel", "left_blinker", "right_blinker", "desired_a", "should_stop", "brake_prob", "gas_prob",
               "gap2", "v_lead2", "lead2", "lead_y", "a_lead_k", "lead_prob", "leads_prob", "gas", "brake", "yaw_rate", "steer")
  print("\nSpearman(driver mean aEgo over next 0.5 s and jerk, variable) in manual regimes; ranked by accel rho")
  for label, mask in specs:
    results = sorted(((_corr(data["a_half"][mask], data[v][mask]), v) for v in variables),
                     key=lambda x: abs(x[0]) if np.isfinite(x[0]) else -1, reverse=True)
    print(f"[{label}] n={int(mask.sum())}")
    for corr, variable in results:
      jerk_rho = _corr(data["jerk"][mask], data[variable][mask])
      print(f"{variable:18s} rho_a={corr:+.3f} rho_jerk={jerk_rho:+.3f}")
    print("  binned partial response: controls=(speed, closing, gap, lead accel), quintiles, median residual accel")
    for _, variable in results[:3]:
      controls = [name for name in ("v", "closing", "gap", "a_lead") if name != variable]
      good = mask & np.isfinite(data[variable]) & np.isfinite(data["a_half"])
      good &= np.isfinite(np.column_stack([data[name] for name in controls])).all(axis=1)
      if good.sum() < 40:
        continue
      x = np.column_stack((np.ones(good.sum()), *(data[name][good] for name in controls)))
      y = data["a_half"][good]
      feature = data[variable][good]
      y_residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
      x_residual = feature - x @ np.linalg.lstsq(x, feature, rcond=None)[0]
      cuts = np.unique(np.percentile(x_residual, [20, 40, 60, 80]))
      bins = np.digitize(x_residual, cuts)
      for band in range(len(cuts) + 1):
        rows = bins == band
        if rows.sum() >= 20:
          print(f"  {variable:16s} bin{band+1} n={rows.sum():6d} median_partial_a={np.median(y_residual[rows]):+.3f}")
  print("\nManual speed-band acceleration")
  for lo, hi in zip((0, 5, 15, 25), (5, 15, 25, 45), strict=True):
    m = manual & (data["v"] >= lo) & (data["v"] < hi)
    a = data["a"][m]
    if m.any():
      print(f"{lo:2d}-{hi:2d} m/s n={m.sum():7d} accel median={np.nanmedian(a):+.3f} "
            + f"p10/p90={np.nanpercentile(a,10):+.3f}/{np.nanpercentile(a,90):+.3f} "
            + f"|a|p90={np.nanpercentile(np.abs(a),90):.3f}")
  episodes = []
  for route_id in np.unique(data["route_id"]):
    mroute = data["route_id"] == route_id
    mask = mroute & manual & (data["lead_radar"] > 0) & (data["closing"] > 1.0) & (data["gap"] > 1.5 * data["v_lead"])
    for window in runs(mask):
      start, end = window.start, window.stop
      if end - start >= 40:
        route_end = np.flatnonzero(mroute)[-1] + 1
        settled = np.flatnonzero((data["closing"][start:route_end] < 0.3) & (data["lead_radar"][start:route_end] > 0) &
                                 manual[start:route_end])
        arrival = start + settled[0] if settled.size else end - 1
        decel = np.flatnonzero(data["a"][start:end] < -0.15)
        release = np.flatnonzero((data["gas"][start + 1:end] == 0) & (data["gas"][start:end - 1] > 0))
        onset = start + decel[0] if decel.size else start + release[0] + 1 if release.size else start
        episodes.append((onset, arrival, max(2.0, 1.5 * data["v_lead"][arrival])))
  print(f"\nApproach episodes n={len(episodes)} "
        + "(radar lead, closing>1 m/s, gap>1.5*vLead for >=2 s; target=max(2 m,1.5s*vLead))")
  if episodes:
    onset_gaps = np.asarray([data["gap"][s] for s, _, _ in episodes])
    onset_closing = np.asarray([data["closing"][s] for s, _, _ in episodes])
    onset_v = np.asarray([data["v"][s] for s, _, _ in episodes])
    onset_vlead = np.asarray([data["v_lead"][s] for s, _, _ in episodes])
    onset_alead = np.asarray([data["a_lead"][s] for s, _, _ in episodes])
    print(f"onset (first a<-0.15 m/s2, else gas release, else context): ego={np.median(onset_v):.2f} m/s "
          + f"lead={np.median(onset_vlead):.2f} closing={np.median(onset_closing):.2f} gap={np.median(onset_gaps):.2f} m "
          + f"lead_accel={np.nanmedian(onset_alead):.2f} m/s2")
    gaps = np.asarray([data["gap"][e] for _, e, _ in episodes])
    leads = np.asarray([data["v_lead"][e] for _, e, _ in episodes])
    times = np.asarray([(e - s) * DT for s, e, _ in episodes])
    closure_time = np.asarray([(data["gap"][s] - target) / max(data["closing"][s], .1) for s, _, target in episodes])
    ratios = times / np.maximum(closure_time, .1)
    print(f"arrival gap median={np.median(gaps):.2f} m "
          + f"p10/p90={np.percentile(gaps,10):.2f}/{np.percentile(gaps,90):.2f}; "
          + f"headway={np.median(gaps/np.maximum(leads,.1)):.2f} s")
    print(f"arrival lead speed median={np.median(leads):.2f} m/s; "
          + f"T median={np.median(times):.2f} s; closure-law median={np.median(closure_time):.2f} s; "
          + f"T/law median={np.median(ratios):.2f}")
    print("T by onset closing (m/s): band n median_T_s")
    for lo, hi in ((1, 2), (2, 4), (4, 8), (8, 30)):
      selected = np.asarray([lo <= data["closing"][s] < hi for s, _, _ in episodes], dtype=bool)
      if selected.any():
        print(f"{lo}-{hi} {selected.sum()} {np.median(times[selected]):.2f}")
    profiles = []
    for start, arrival, target in episodes:
      a = data["a"][start:arrival + 1]
      if len(a) > 2:
        reversals = np.sum(a[1:] * a[:-1] < 0)
        pedal = np.sum((data["gas"][start + 1:arrival + 1] != data["gas"][start:arrival]) |
                       (data["brake"][start + 1:arrival + 1] != data["brake"][start:arrival]))
        required = data["closing"][start] ** 2 / max(2 * (data["gap"][start] - target), 0.1)
        required_profile = data["closing"][start:arrival + 1] ** 2 / (
          2 * np.maximum(data["gap"][start:arrival + 1] - target, 0.1))
        third = max(1, len(a) // 3)
        profiles.append((np.std(a), np.min(a), np.mean(a), reversals, pedal, required,
                         np.nanmedian(required_profile), np.mean(a[:third]), np.mean(a[-third:])))
    if profiles:
      p = np.asarray(profiles)
      print(f"approach accel_sd={np.median(p[:,0]):.3f} m/s2 "
            + f"peak_decel={np.median(p[:,1]):.3f} mean_accel={np.median(p[:,2]):.3f} "
            + f"reversals={np.median(p[:,3]):.0f} pedal_transitions={np.median(p[:,4]):.0f}")
      print(f"constant-decel required at onset median={np.median(p[:,5]):.3f}; "
            + f"episode median requirement={np.median(p[:,6]):.3f} m/s2")
      print(f"approach mean accel first/last third={np.median(p[:,7]):+.3f}/{np.median(p[:,8]):+.3f} m/s2")
    print("arrival gap/headway by arrival lead speed (m/s):")
    for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 40)):
      selected = (leads >= lo) & (leads < hi)
      if selected.any():
        print(f"{lo}-{hi}: n={selected.sum()} gap={np.median(gaps[selected]):.2f} m headway={np.median(gaps[selected]/np.maximum(leads[selected],.1)):.2f} s")
  stops = []
  for route_id in np.unique(data["route_id"]):
    ix = np.flatnonzero(data["route_id"] == route_id)
    slow_lead = manual[ix] & (data["lead_radar"][ix] > 0) & (data["v_lead"][ix] < 1.0)
    for window in runs(slow_lead):
      begin, end = window.start, window.stop
      fast = np.flatnonzero(data["v"][ix[begin:end]] > 2.0)
      rest = np.flatnonzero(data["v"][ix[begin:end]] < 0.3)
      if fast.size and rest.size and rest[0] > fast[0] and rest[0] - fast[0] >= 20:
        start, arrival = ix[begin + fast[0]], ix[begin + rest[0]]
        stops.append((start, arrival))
  print(f"\nStops behind a stopped/stopping radar lead n={len(stops)}")
  if stops:
    gaps = np.asarray([data["gap"][e] for _, e in stops])
    durations = np.asarray([(e-s)*DT for s, e in stops])
    final_a = np.asarray([data["a"][max(s, e-20):e+1].mean() for s, e in stops])
    print(f"rest gap median={np.median(gaps):.2f} m "
          + f"p10/p90={np.percentile(gaps,10):.2f}/{np.percentile(gaps,90):.2f}; "
          + f"elapsed={np.median(durations):.2f} s; final-1s mean accel={np.median(final_a):+.3f} m/s2")
  print("\nSteady-following headway by ego speed (radar lead, |closing|<1 m/s)")
  steady = manual & (data["lead_radar"] > 0) & (np.abs(data["closing"]) < 1.0) & (data["v"] > 1.0)
  for lo, hi in ((0, 5), (5, 15), (15, 25), (25, 40)):
    selected = steady & (data["v"] >= lo) & (data["v"] < hi) & np.isfinite(data["headway"])
    if selected.any():
      print(f"{lo}-{hi} m/s n={selected.sum()} headway p10/median/p90={np.percentile(data['headway'][selected],[10,50,90]).round(2).tolist()} s")
  ahead = np.full(len(manual), np.nan)
  for route_id in np.unique(data["route_id"]):
    ix = np.flatnonzero(data["route_id"] == route_id)
    if len(ix) > 20:
      ahead[ix[:-20]] = data["headway"][ix[20:]]
  error = data["headway"] - 1.5
  correction = steady & np.isfinite(ahead) & np.isfinite(error)
  delta = ahead - data["headway"]
  print(f"headway error correction over 1 s: n={correction.sum()} rho(error, delta_headway)={_corr(error[correction], delta[correction]):+.3f}")
  print("\nNo per-frame windows bridge route boundaries; episode estimates are descriptive.")


def _load(root, keys):
  with ProcessPoolExecutor(max_workers=6) as pool:
    parts = [d for _, d, error in pool.map(_route_job, [(r, root) for r in _routes(root)]) if error is None]
  out = {k: np.concatenate([np.asarray(p[k], dtype=float) for p in parts]) for k in keys}
  out["route_id"] = np.concatenate([np.full(len(p["t"]), i) for i, p in enumerate(parts)])
  # the driver's accel over the next 0.5 s, never across a route boundary
  out["a_next"] = np.concatenate([np.convolve(np.asarray(p["a"], dtype=float), np.ones(10) / 10, mode="full")[9:9 + len(p["a"])] for p in parts])
  return out


def _pinball_fit(y, dv, a, q):
  best = (math.inf, 0.0, 0.0, 0.0)
  for c0 in np.arange(-0.1, 0.41, 0.025):
    for kdv in np.arange(0.0, 0.31, 0.02):
      base = y - c0 - kdv * dv
      for ka in np.arange(0.0, 1.21, 0.05):
        r = base - ka * a
        loss = float(np.mean(np.maximum(q * r, (q - 1) * r)))
        if loss < best[0]:
          best = (loss, c0, kdv, ka)
  return best[1:]


def _bursts(a, dt=DT):
  """Rises of at least 0.4 m/s^2 within 1 s that end above +0.3."""
  n, i, count = round(1.0 / dt), 0, 0
  while i < len(a) - n:
    if a[i + n] > 0.3 and a[i + n] - a[i] >= 0.4:
      count, i = count + 1, i + n
    else:
      i += 1
  return count


def _pumps(cmd, dt=DT):
  """While braking past -0.3: an ease of at least 0.15 m/s^2 that re-deepens by at least 0.15 within 2 s."""
  pumps, low, high, t_high = 0, math.inf, math.nan, 0
  for i, c in enumerate(cmd):
    if c > -0.3:
      low, high = math.inf, math.nan
    elif math.isnan(high):
      low = min(low, c)
      if c > low + 0.15:
        high, t_high = c, i
    elif c > high:
      high, t_high = c, i
    elif (i - t_high) * dt > 2.0:
      low, high = c, math.nan
    elif c < high - 0.15:
      pumps, low, high = pumps + 1, c, math.nan
  return pumps


def _fly_stages(job):
  """`_fly_logged_lead` with the first lead ask and engine-coast budget output of each planner tick recorded."""
  import moonpilot.longitudinal as longitudinal
  from moonpilot.lab.planner import _fly_logged_lead
  frame, stages = {}, {"lead": [], "budget": []}
  lead_accel, coast_budget, update = longitudinal.lead_accel, longitudinal.coast_budget, longitudinal.MoonpilotLongitudinalPlanner.update

  def record(name, fn):
    def wrapped(*a, **k):
      out = fn(*a, **k)
      frame.setdefault(name, out)
      return out
    return wrapped

  def tick(self, sm):
    frame.clear()
    update(self, sm)
    for k in stages:
      stages[k].append(frame.get(k, frame.get("lead", math.nan)))
  longitudinal.lead_accel, longitudinal.coast_budget = record("lead", lead_accel), record("budget", coast_budget)
  longitudinal.MoonpilotLongitudinalPlanner.update = tick
  try:
    out = _fly_logged_lead(job)
  finally:
    longitudinal.lead_accel, longitudinal.coast_budget, longitudinal.MoonpilotLongitudinalPlanner.update = lead_accel, coast_budget, update
  out.update({k: np.nan_to_num(np.asarray(x, dtype=float)) for k, x in stages.items()})
  return out


def _decel(args):
  """The planner flown through every manual decel a radar lead caused (`decel-peaks`' episodes): brake pumps — an ease
  of 0.15 m/s^2 re-deepened by 0.15 within 2 s — in the lead ask, after the engine-coast budget and in the command,
  the command's jitter about its 1 s mean, brake-band presses, the 10-16 m/s peak and the closest gap, with each of
  this law's comfort terms off in turn; the hard brakes alongside."""
  from unittest import mock
  import moonpilot.longitudinal as longitudinal
  from moonpilot.lab.planner import ROUTE_SETS, _calibration, _decel_jobs, _hard_brake_run, _pooled_manual
  from moonpilot.lab import expand
  d = _pooled_manual(expand(["manual-approach", "0000044e", "0000044f", "00000450", "00000451"], ROUTE_SETS), args.root)
  a_s = np.concatenate([np.convolve(d["a"][d["route"] == r], np.ones(10) / 10, mode="same") for r in np.unique(d["route"])])
  jobs, drv = _decel_jobs(d, a_s)
  calibration = _calibration(args.calibration, args.root)
  jobs = [(*job, calibration) for job in jobs]
  band = (drv[:, 0] >= 10) & (drv[:, 0] < 16)
  print(f"{len(jobs)} manual decels a radar lead caused, calibration {args.calibration}; the driver's 10-16 m/s peak {np.median(drv[band, 1]):.2f}")
  variants = (
    ("current", nullcontext()),
    ("engine-coast budget off", mock.patch.object(longitudinal, "coast_budget", lambda ask, *_: ask)),
    ("mirror off", mock.patch.object(longitudinal, "MOONPILOT_MIRROR_FADE_T", [-2.0, -1.0])),
    ("follow envelope off", mock.patch.object(longitudinal, "MOONPILOT_FOLLOW_ACCEL_T", [-2.0, -1.0])),
  )
  for name, ctx in variants:
    with ctx, ProcessPoolExecutor(6) as pool:
      flown = list(pool.map(_fly_stages, jobs))
    with ctx:
      hard = [_hard_brake_run("fork+engine", v0, a, lag)[0] for lag in (False, True) for v0 in (20.0, 25.0, 30.0) for a in (-3.5, -5.0)]
    peak = np.asarray([np.convolve(f["a"], np.ones(10) / 10, mode="same")[f["v"] > 2.0].min() for f in flown])
    jitter = [np.std(f["cmd"] - np.convolve(f["cmd"], np.ones(20) / 20, mode="same")) for f in flown]
    presses = sum(int(np.sum(np.diff(np.asarray(f["pressing"], dtype=int)) > 0)) for f in flown)
    print(f"  {name:24s} pumps lead {sum(_pumps(f['lead']) for f in flown):3d} / budget {sum(_pumps(f['budget']) for f in flown):3d} / "
          + f"command {sum(_pumps(f['cmd']) for f in flown):3d}  jitter {np.median(jitter):.3f}  presses {presses:3d}  "
          + f"10-16 peak {np.median(peak[band]):5.2f}  closest {min(f['gap'].min() for f in flown):4.1f} m")
    print(f"    hard-brake min gap, default calibration (20/25/30 m/s x -3.5/-5, then lagged): {' '.join(f'{g:5.1f}' for g in hard)}")


def _follow(args):
  """The driver's accel behind a lead, by headway, lead-minus-ego speed and lead accel; its p50/p75 envelope on the
  lead's speed excess and slow accel; then the planner flown through every 30 s of manual following behind one radar
  track (set speed 3 m/s over the driver's highest), the follow envelope off and on, against the driver's own stretch."""
  from unittest import mock
  import moonpilot.longitudinal as longitudinal
  from moonpilot.lab.planner import _calibration, _fly_logged_lead
  d = _load(args.root, ("t", "manual", "v", "a", "lead", "lead_radar", "gap", "v_lead", "a_lead", "a_slow"))
  v, gap, v_lead, a_next = d["v"], d["gap"], d["v_lead"], d["a_next"]
  hw, rel = gap / np.maximum(v, 0.1), v_lead - v
  ok = (d["manual"] > 0) & (d["lead"] > 0) & (d["lead_radar"] > 0) & (v > 4.0) & np.isfinite(d["a_lead"])
  print(f"Driver accel over the next 0.5 s behind a radar lead (median / p75 / p90, n), {ok.sum() * DT / 60:.0f} manual lead-minutes:")
  for alo, ahi in ((-0.3, 0.3), (0.3, 0.8), (0.8, 3.0)):
    for lo, hi in ((1.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 9.0)):
      cells = []
      for rlo, rhi in ((-0.5, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 8.0)):
        m = ok & (hw >= lo) & (hw < hi) & (rel >= rlo) & (rel < rhi) & (d["a_lead"] >= alo) & (d["a_lead"] < ahi)
        q = np.percentile(a_next[m], [50, 75, 90]) if m.sum() >= 150 else None
        cells.append(f"dv {rlo:+.1f}..{rhi:+.1f} " + (f"{q[0]:5.2f}/{q[1]:4.2f}/{q[2]:4.2f} ({m.sum():5d})" if q is not None else "       -           "))
      print(f"  a_lead {alo:+.1f}..{ahi:+.1f} hw {lo:3.1f}-{hi:3.1f} s: " + "  ".join(cells))
  for q in (0.5, 0.75):
    for vlo, vhi in ((5, 18), (18, 40)):
      m = ok & (hw < 6.0) & (rel > -0.5) & (v >= vlo) & (v < vhi) & np.isfinite(d["a_slow"])
      idx = np.flatnonzero(m)[::4]
      c0, kdv, ka = _pinball_fit(a_next[idx], np.clip(rel[idx], 0, 6), np.clip(d["a_slow"][idx], 0, 3), q)
      print(f"  q{q:.2f} at {vlo}-{vhi} m/s, hw < 6 s: a = {c0:+.3f} + {kdv:.2f} * dv+ + {ka:.2f} * a_slow+  (n {m.sum()})")

  win, jobs, drv = round(30.0 / DT), [], []
  cut = np.r_[True, (np.diff(d["route_id"]) != 0) | (np.abs(np.diff(v_lead)) > 2.5)]
  calibration = _calibration(args.calibration, args.root)
  for r in runs((d["manual"] > 0) & (d["lead"] > 0) & (d["lead_radar"] > 0) & (v > 4.0) & ~cut):
    for s0 in range(r.start, r.stop - win + 1, win):
      w = slice(s0, s0 + win)
      if np.max(np.abs(np.diff(gap[w]) - (v_lead[w][:-1] - v[w][:-1]) * DT)) > 1.0 or gap[w].max() > 9.0 * max(v[w].min(), 1.0):
        continue
      jobs.append((v[w], gap[w], v_lead[w], d["a"][w][0], calibration, (v[w].max() + 3.0) * 3.6))
      a_s = np.convolve(d["a"][w], np.ones(10) / 10, mode="same")
      drv.append((np.percentile(a_s, 90), np.percentile(a_s, 99), _bursts(a_s), np.median(gap[w] / np.maximum(v[w], 0.1)), gap[w].min(), np.percentile(a_s, 1)))
  drv = np.asarray(drv)
  minutes = len(jobs) * 30.0 / 60
  print(f"\nFlown through {len(jobs)} stretches ({minutes:.0f} min) of manual following, calibration {args.calibration}: accel p90 / p99 / p1 (median "
        + "over stretches), positive bursts per minute (a rise of 0.4 m/s^2 in 1 s ending over +0.3), median headway, closest gap")
  print(f"  driver:        {np.median(drv[:, 0]):5.2f} / {np.median(drv[:, 1]):5.2f} / {np.median(drv[:, 5]):5.2f}  "
        + f"bursts {drv[:, 2].sum() / minutes:5.2f}/min  headway {np.median(drv[:, 3]):4.2f} s  closest {drv[:, 4].min():4.1f} m")
  off = mock.patch.object(longitudinal, "MOONPILOT_FOLLOW_ACCEL_T", [-2.0, -1.0])  # gone at every headway
  for name, ctx in (("envelope off", off), ("envelope on", nullcontext())):
    with ctx, ProcessPoolExecutor(6) as pool:
      flown = list(pool.map(_fly_logged_lead, jobs))
    pl = np.asarray([(np.percentile(f["a"], 90), np.percentile(f["a"], 99), _bursts(f["a"]), np.median(f["gap"] / np.maximum(f["v"], 0.1)), f["gap"].min(),
                      np.percentile(f["a"], 1)) for f in flown])
    print(f"  {name:13s}: {np.median(pl[:, 0]):5.2f} / {np.median(pl[:, 1]):5.2f} / {np.median(pl[:, 5]):5.2f}  bursts {pl[:, 2].sum() / minutes:5.2f}/min  "
          + f"headway {np.median(pl[:, 3]):4.2f} s  closest {pl[:, 4].min():4.1f} m")


def register(sub):
  parser = sub.add_parser("driver", help="extract cached model-tick driver variables and print policy summaries")
  parser.set_defaults(fn=_report)
  parser = sub.add_parser("driver-follow", help="the driver's accel behind a lead, its envelope, and the planner flown through his following")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  parser.set_defaults(fn=_follow)
  parser = sub.add_parser("driver-decel", help="brake pumps, jitter and presses of the planner flown through the driver's lead-caused decels")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  parser.set_defaults(fn=_decel)
