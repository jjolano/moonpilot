"""Online longitudinal estimator replays on locally cached drives."""
import glob
import json
import os
from types import SimpleNamespace

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.tools.lib.logreader import LogReader

from moonpilot.lab import DEFAULT_ROOT, drive_config, expand, segments
from moonpilot.lead import LeadAccelEstimator
from moonpilot.latency import MOONPILOT_LAG_BLOCK_COUNT, MOONPILOT_LAG_ESTIMATE_EVERY, MOONPILOT_LAG_HISTORY_KEY, MOONPILOT_LAG_KEY, LongLagEstimator
from moonpilot.pitch import MOONPILOT_PITCH_MIN_SAMPLES, MOONPILOT_PITCH_OFFSET_KEY, PitchOffsetEstimator
from moonpilot.radar_latency import MOONPILOT_RADAR_LATENCY_KEY, MOONPILOT_RADAR_LATENCY_SAMPLES_KEY, RadarLatencyEstimator

# Default scope: the engaged local drives the estimator claims were re-measured on. "all" is every cached route.
ROUTE_SETS = {
  "engaged": ("000003f8", "000003fe", "00000400", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  "all": tuple(sorted({os.path.basename(p).split("--", 1)[0] for p in glob.glob(os.path.join(DEFAULT_ROOT, "*--*", "rlog.zst"))})),
}


def _frames(route, root):
  """Yield planner ticks, retaining each input message until its next log update."""
  latest = {}
  for path in segments(route, root):
    for m in LogReader(path, sort_by_time=True):
      kind = m.which()
      if kind in ("carParams", "carState", "carControl", "controlsState", "radarState", "longitudinalPlan"):
        latest[kind] = m
      if kind != "longitudinalPlan" or not all(k in latest for k in ("carParams", "carState", "carControl", "controlsState", "radarState")):
        continue
      cp, cs, cc, controls, radar = (getattr(latest[k], k) for k in ("carParams", "carState", "carControl", "controlsState", "radarState"))
      yield m.logMonoTime * 1e-9, cp, cs, cc, controls, radar, m.longitudinalPlan


def _lead(lead):
  return SimpleNamespace(present=lead.present, radar=lead.radar, radarTrackId=lead.radarTrackId, dRel=lead.dRel, yRel=lead.yRel,
                         vLead=lead.vLead, aLeadK=lead.aLeadK, aLeadTau=lead.aLeadTau)


def _params(route, root):
  return drive_config(route, root)[1]


def _seed_radar(est, params):
  try:
    latency = float(params[MOONPILOT_RADAR_LATENCY_KEY])
    samples = int(params.get(MOONPILOT_RADAR_LATENCY_SAMPLES_KEY, "0"))
  except (KeyError, TypeError, ValueError):
    return
  if latency > 0.0 and samples > 0:
    est.seed(latency, samples)


def _lag_blocks(params):
  history = params.get(MOONPILOT_LAG_HISTORY_KEY)
  if history:
    try:
      blocks = json.loads(history)
    except (TypeError, ValueError):
      blocks = []
    if isinstance(blocks, list) and blocks:
      return [float(b) for b in blocks[:MOONPILOT_LAG_BLOCK_COUNT]]
  try:
    lag = float(params[MOONPILOT_LAG_KEY])
    count = int(params.get("MoonpilotLongLagBlocks", "0"))
  except (KeyError, TypeError, ValueError):
    return []
  return [lag] * min(max(count, 0), MOONPILOT_LAG_BLOCK_COUNT - 1)


def _pitch(args):
  for route in expand(args.routes, ROUTE_SETS):
    est = PitchOffsetEstimator()
    params = _params(route, args.root)
    try:
      est.seed(float(params[MOONPILOT_PITCH_OFFSET_KEY]), MOONPILOT_PITCH_MIN_SAMPLES)
    except (KeyError, TypeError, ValueError):
      pass
    frames = valid = 0
    for _, _, cs, cc, _, _, _ in _frames(route, args.root):
      pose = cc.orientationNED
      ok = len(pose) == 3 and cs.vEgo > 5.0 and not cs.standstill
      est.update(float(pose[1]) if len(pose) == 3 else 0.0, ok)
      frames += 1
      valid += int(ok)
    print(f"{route}: ticks={frames} valid={valid} status={est.status} estimate={np.degrees(est.estimate):.3f}° applied={np.degrees(est.applied()):.3f}°")


def _radar(args):
  for route in expand(args.routes, ROUTE_SETS):
    est = RadarLatencyEstimator()
    _seed_radar(est, _params(route, args.root))
    frames = 0
    for _, _, cs, _, _, radar, _ in _frames(route, args.root):
      for lead in (radar.leadOne, radar.leadTwo):
        est.update(_lead(lead), cs.aEgo)
      frames += 1
    applied_ms = est.applied() * 1000
    estimate_ms = est.estimate * 1000
    print(f"{route}: ticks={frames} gated_samples={est.samples} status={est.status} estimate={estimate_ms:.1f} ms applied={applied_ms:.1f} ms")


def _lead_replay(args):
  for route in expand(args.routes, ROUTE_SETS):
    estimators = (LeadAccelEstimator(), LeadAccelEstimator())
    estimates = [[], []]
    speeds = [[], []]
    kalman = [[], []]
    present = [[], []]
    for _, _, _, _, _, radar, _ in _frames(route, args.root):
      for i, lead in enumerate((radar.leadOne, radar.leadTwo)):
        lead = _lead(lead)
        estimate, _ = estimators[i].update(lead)
        estimates[i].append(estimate)
        speeds[i].append(lead.vLead)
        kalman[i].append(lead.aLeadK)
        present[i].append(bool(lead.present and lead.radar))
    errors, base_errors, under = [], [], 0
    count = 0
    for est, v, base, mask in zip(estimates, speeds, kalman, present, strict=True):
      v, est, base, mask = np.asarray(v), np.asarray(est), np.asarray(base), np.asarray(mask)
      n = len(v)
      if n < 41:
        continue
      # Centered +/-1 s slope; only score intervals continuously belonging to one radar track.
      ref = (v[40:] - v[:-40]) / (40 * DT_MDL)
      valid = mask[40:] & mask[:-40]
      middle = mask[20:-20]
      valid &= middle
      e, b = est[20:-20][valid], base[20:-20][valid]
      r = ref[valid]
      if len(r):
        errors.extend((e - r).tolist())
        base_errors.extend((b - r).tolist())
        under += int(np.count_nonzero(e < -1.0))
        count += len(e)
    if count:
      kalman_sd = np.std(base_errors)
      current_sd = np.std(errors)
      print(f"{route}: radar_frames={count} aLeadK_error_sd={kalman_sd:.3f} current_error_sd={current_sd:.3f} current_below_-1={under / count:.2%}")
    else:
      print(f"{route}: no scored radar-lead intervals")


def _lag(args):
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState

  for route in expand(args.routes, ROUTE_SETS):
    estimator = None
    frames = attempts = 0
    for _, cp, cs, cc, controls, _, plan in _frames(route, args.root):
      if estimator is None:
        estimator = LongLagEstimator(cp, DT_MDL)
        estimator.restore(_lag_blocks(_params(route, args.root)))
      valid = (cc.longActive and controls.longControlState == LongCtrlState.pid and cs.vEgo > 5.0 and not cs.standstill
               and not cs.brakePressed and not cs.gasPressed and not controls.forceDecel)
      if estimator.frame % MOONPILOT_LAG_ESTIMATE_EVERY == 0:
        attempts += 1
      estimator.update(float(plan.aTarget), float(np.clip(cs.aEgo, -3.5, 2.0)), valid)
      frames += 1
    if estimator is None:
      print(f"{route}: no planner ticks")
    else:
      estimate_ms = estimator.estimate * 1000
      applied_ms = estimator.applied_delay() * 1000
      summary = f"{route}: ticks={frames} attempts={attempts} status={estimator.status} blocks={estimator.valid_blocks}"
      print(f"{summary} estimate={estimate_ms:.1f} ms applied={applied_ms:.1f} ms")


def register(sub):
  for name, fn in (("pitch", _pitch), ("radar-latency", _radar), ("lead-accel", _lead_replay), ("long-lag", _lag)):
    parser = sub.add_parser(name, help=f"Replay the {name} estimator per drive")
    parser.add_argument("--routes", nargs="+", default=["engaged"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
    parser.set_defaults(fn=fn)
