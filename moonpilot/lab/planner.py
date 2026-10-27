"""Recorded-drive replay of the fork longitudinal planner.

Every route is replayed in the configuration its own initData recorded, one planner tick per logged `modelV2`, with
each message's real `logMonoTime` and the module's wall clock shimmed to the row's time — so the planner's own age
math (lead age, curve-path age, the pose's staleness check) sees what it saw on the drive. Two ceilings are named
rather than hidden: window overrides the drive did not carry (a marketplace model's smoothing, a car-side constant)
and the car's own brake arbitration, which no replay models — `brake-lamps` reports the drive's real BRKON, not what
the band would have produced.
"""
import math
from collections import deque
from contextlib import contextmanager
from functools import cache
from types import MethodType, SimpleNamespace

import numpy as np
from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.tools.lib.logreader import LogReader
from moonpilot.latency import MOONPILOT_LAG_MAX
from moonpilot.longitudinal import LongitudinalPlanSource, MoonpilotLongitudinalPlanner
from moonpilot.longcontrol import MoonpilotLongControl
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, lag_s, param_index, runs

EXTRACT_VERSION = 2
LAMP_VERSION = 1
LAMP_ADDRESS = 0x320  # VSC1S07: the skid-control ECU's own frame, where BRKON is the brake-lamp proxy
CONTROL_VERSION = 1  # one row per carControl frame: the rate the acceleration controller is written for
MANUAL_VERSION = 2
CUES_VERSION = 1
ROUTE_SETS = {
  # Engaged local drives. 0000042c/42d/42e have zero longActive frames and are skipped by the commands.
  "longitudinal": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  "lead-approach": ("000003f8", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  # The driver's own longitudinal behind radar leads; routes without such frames are counted as empty.
  "manual-approach": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414",
                      "0000042b", "0000042c", "0000043e", "0000043f", "0000044d"),
  # Every cached drive with engaged longitudinal frames.
  "engaged": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "0000040f", "00000410", "00000414",
              "00000417", "0000041d", "00000424", "00000429", "0000042a", "0000042b", "00000439", "0000043a", "0000043e", "0000043f",
              "0000044d", "00000450", "00000451", "0000045a", "0000045d", "0000045e"),
}
MESSAGES = ("carState", "controlsState", "carControl", "vehicleParameters", "radarState", "selfdriveState", "moonpilotState", "modelV2")
ON_BY_DEFAULT = {"MoonpilotLongitudinal", "MoonpilotModelBraking", "MoonpilotBrakeBand", "MoonpilotCurveSpeed", "MoonpilotLeadLateral", "MoonpilotCoastGrade"}


class ReplayParams:
  def __init__(self, values):
    self.values = dict(values)

  def get(self, key, block=False, return_default=False):
    value = self.values.get(key)
    if value is None:
      return key in ON_BY_DEFAULT if return_default else None
    text = value.decode() if isinstance(value, bytes) else value
    try:
      return int(text)
    except (TypeError, ValueError):
      try:
        return float(text)
      except (TypeError, ValueError):
        return text

  def put(self, key, value):
    self.values[key] = value


class ReplayClock:
  def __init__(self):
    self.now_ns = 0

  def monotonic(self):
    return self.now_ns * 1e-9

  def monotonic_ns(self):
    return self.now_ns


class ReplaySubMaster:
  """The plannerd SubMaster subset, with row-level message timestamps.

  `valid` is snapshot presence, `alive` is stamp age against SERVICE_LIST's frequency, and `freq_ok` is an
  approximation: replay has no process receive history. `recv_time` is likewise publish time, not a process
  read time; the planner's path-age calculation needs the model's publish stamp there.
  """

  def __init__(self, messages, stamps, now_ns):
    self.messages = messages
    self.logMonoTime = dict(stamps)
    self.recv_time = {k: v * 1e-9 for k, v in stamps.items()}
    self.valid = {k: k in stamps for k in MESSAGES}
    self.alive = {}
    self.freq_ok = dict.fromkeys(MESSAGES, True)
    now = now_ns * 1e-9
    for kind in MESSAGES:
      frequency = SERVICE_LIST[kind].frequency
      stamp = stamps.get(kind)
      self.alive[kind] = stamp is not None and (frequency <= 0.0 or now - stamp * 1e-9 < 10.0 / frequency)
    self.frame = 0

  def __getitem__(self, key):
    return getattr(self.messages[key], key)

  def all_checks(self):
    return all(self.valid.values()) and all(self.alive.values()) and all(self.freq_ok.values())


def _build(segs):
  latest, rows, plans, cp = {}, [], {}, None
  for msg in LogReader(segs, sort_by_time=True):
    kind = msg.which()
    if kind == "carParams":
      cp = msg.carParams.as_builder().to_bytes()
    elif kind == "longitudinalPlan":
      plans[msg.longitudinalPlan.modelMonoTime] = msg.as_builder().to_bytes()
    elif kind in MESSAGES:
      latest[kind] = (msg.as_builder().to_bytes(), int(msg.logMonoTime))
      if kind == "modelV2" and cp is not None and all(k in latest for k in MESSAGES[:-1]):
        rows.append((int(msg.logMonoTime), dict(latest)))
  if not cp or not rows:
    raise ValueError("no planner frames or carParams")
  blob, offsets = bytearray(), []
  for mono, snapshots in rows:
    row = [mono]
    for kind in MESSAGES:
      data, stamp = snapshots.get(kind, (b"", 0))
      row.extend((len(blob), len(data), stamp))
      blob.extend(data)
    plan = plans.get(mono, b"")
    row.extend((len(blob), len(plan)))
    blob.extend(plan)
    offsets.append(row)
  return {"cp": np.frombuffer(cp, dtype=np.uint8), "blob": np.frombuffer(blob, dtype=np.uint8), "offsets": np.asarray(offsets, dtype=np.int64)}


def extract(route, root=DEFAULT_ROOT):
  return cached(route, "planner", EXTRACT_VERSION, root, _build)


def replay(route, data, root=DEFAULT_ROOT, overrides=None, details_out=None):
  cp = messaging.log_from_bytes(data["cp"].tobytes(), car.CarParams)
  params = ReplayParams({k: v.encode() for k, v in drive_config(route, root)[1].items()})
  if overrides:
    params.values.update({k: str(v).encode() for k, v in overrides.items()})
  import moonpilot.longitudinal as longitudinal
  original_params, original_time = longitudinal.Params, longitudinal.time
  clock = ReplayClock()
  longitudinal.Params = lambda: params  # ty: ignore[invalid-assignment]  # the replay's params, for the planner's own constructor
  longitudinal.time = clock  # ty: ignore[invalid-assignment]  # wall clock == the row's logMonoTime
  try:
    planner = MoonpilotLongitudinalPlanner(cp)
    blob, offsets = data["blob"], data["offsets"]
    replayed, logged, active, lead_age = [], [], [], []
    for row in offsets:
      mono, messages, stamps = int(row[0]), {}, {}
      for i, kind in enumerate(MESSAGES):
        start, size, stamp = row[1 + i * 3:4 + i * 3]
        if size:
          messages[kind] = messaging.log_from_bytes(blob[start:start + size].tobytes())
          stamps[kind] = int(stamp)
      clock.now_ns = mono
      sm = ReplaySubMaster(messages, stamps, mono)
      planner.update(sm)
      replayed.append(planner.output_a_target)
      plan_start, plan_size = row[1 + len(MESSAGES) * 3:3 + len(MESSAGES) * 3]
      plan = messaging.log_from_bytes(blob[plan_start:plan_start + plan_size].tobytes()) if plan_size else None
      logged.append(float(plan.longitudinalPlan.aTarget) if plan is not None else np.nan)
      active.append(bool(messages["carControl"].carControl.longActive))
      radar_stamp = stamps.get("radarState", 0)
      lead_age.append((mono - radar_stamp) * 1e-9 if messages.get("radarState") and messages["radarState"].radarState.leadOne.present else np.nan)
      if details_out is not None:
        cs = messages["carState"].carState
        cc = messages["carControl"].carControl
        lead = messages["radarState"].radarState.leadOne
        mv = messages["modelV2"].modelV2
        plan_reader = plan.longitudinalPlan if plan is not None else None
        details_out.append({
          "t": mono * 1e-9,
          "manual_authority": not cc.longActive and not cs.cruiseState.enabled and cs.gearShifter == car.CarState.GearShifter.drive,
          "radar_present": bool(lead.radar),
          "v_ego": float(cs.vEgo),
          "a_ego": float(cs.aEgo),
          "v_cruise": float(cs.vCruise),
          "brake_pressed": bool(cs.brakePressed),
          "gas_pressed": bool(cs.gasPressed),
          "standstill": bool(cs.standstill),
          "cruise_standstill": bool(cs.cruiseState.standstill),
          "long_active": bool(cc.longActive),
          "curvature": float(mv.action.desiredCurvature),
          "model_accel": float(mv.action.desiredAcceleration),
          "lead_present": bool(lead.present),
          "d_rel": float(lead.dRel),
          "v_lead": float(lead.vLead),
          "plan_allow_throttle": bool(planner.allow_throttle),
          "replay_should_stop": bool(planner.output_should_stop),
          "logged_a_target": float(plan_reader.aTarget) if plan_reader is not None else np.nan,
          "logged_should_stop": bool(plan_reader.shouldStop) if plan_reader is not None else False,
          "logged_allow_throttle": bool(plan_reader.allowThrottle) if plan_reader is not None else False,
          "logged_accel": float(cc.actuators.accel),
          "band_pressing": bool(planner.brake_band.pressing),
          "model_slot": bool(planner.model_braking),
          "source": int(planner.source),
        })
  finally:
    longitudinal.Params, longitudinal.time = original_params, original_time
  return np.asarray(replayed), np.asarray(logged), np.asarray(active), np.asarray(lead_age)


def _details_arrays(details):
  return {k: np.asarray([row[k] for row in details]) for k in details[0]}


def _curve_mask(d):
  """Engaged curve frames: the logged model path's requested lateral accel reaches 1 m/s^2."""
  return d["long_active"] & (np.abs(d["curvature"]) * np.maximum(d["v_ego"], 0.0) ** 2 >= 1.0)


def _rms_about_mean(x, mask):
  kernel = np.ones(20) / 20
  mean = np.convolve(x, kernel, mode="same")
  return float(np.sqrt(np.mean((x - mean)[mask] ** 2))) if mask.any() else float("nan")


def _skip_no_engaged(route, active):
  if not active.any():
    print(f"{route}: no longActive frames, skipping")
    return True
  return False


def _fidelity(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    replayed, logged, active, _ = replay(route, extract(route, args.root), args.root)
    if _skip_no_engaged(route, active):
      continue
    mask = active & np.isfinite(logged)
    err = float(np.percentile(np.abs(replayed[mask] - logged[mask]), 95)) if mask.any() else float("nan")
    print(f"{route}: commit {commit}, frames {mask.sum()}, p95 |aTarget error| {err:.4f} m/s^2")


def _band(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, cfg = drive_config(route, args.root)
    d = extract(route, args.root)
    driven, logged, active, _ = replay(route, d, args.root)
    if _skip_no_engaged(route, active):
      continue
    off, _, _, _ = replay(route, d, args.root, {"MoonpilotBrakeBand": 0})
    mask = active & np.isfinite(logged)
    err = float(np.percentile(np.abs(driven[mask] - logged[mask]), 95)) if mask.any() else float("nan")
    off_err = float(np.percentile(np.abs(off[mask] - logged[mask]), 95)) if mask.any() else float("nan")
    print(f"{route}: commit {commit}, as-driven MoonpilotBrakeBand={cfg.get('MoonpilotBrakeBand', '0')};")
    print(f"  p95 logged as-driven {err:.4f}, band-off {off_err:.4f};")
    print(f"  engaged command sd band on/off {np.std(driven[mask]):.4f}/{np.std(off[mask]):.4f}; see `brake-lamps` for the car's own BRKON")


def _hf_sd(x, mask):
  """Standard deviation of the command residual after a centered one-second mean."""
  kernel = np.ones(20) / 20
  mean = np.convolve(x, kernel, mode="same")
  return float(np.std((x - mean)[mask]))


def _smooth(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    d = extract(route, args.root)
    driven, logged, active, _ = replay(route, d, args.root)
    if _skip_no_engaged(route, active):
      continue
    off, _, _, _ = replay(route, d, args.root, {"MoonpilotBrakeBand": 0})
    mask = active & np.isfinite(logged)
    err = float(np.percentile(np.abs(driven[mask] - logged[mask]), 95)) if mask.any() else float("nan")
    print(f"{route}: commit {commit}, engaged aTarget sd band on/off {np.std(driven[mask]):.4f}/{np.std(off[mask]):.4f} m/s^2,",
          f"HF sd band on/off {_hf_sd(driven, mask):.4f}/{_hf_sd(off, mask):.4f}, logged p95 {err:.4f}")


def _events(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    d = extract(route, args.root)
    driven, logged, active, _ = replay(route, d, args.root)
    if _skip_no_engaged(route, active):
      continue
    mask = active & np.isfinite(logged)
    duration = mask.sum() / 20.0
    press_episodes = runs(mask & (driven < -0.4))
    represses = sum(start.start - previous.stop <= 2 * 20 for previous, start in zip(press_episodes, press_episodes[1:], strict=False))
    per_minute = len(press_episodes) / max(duration, 1) * 60
    print(
      f"{route}: commit {commit}, engaged {duration:.1f} s; presses {len(press_episodes)} ({per_minute:.2f}/min),",
      f"re-presses within 2 s {represses} ({represses / max(duration, 1) * 60:.2f}/min);",
    )
    print(f"  peak min/max {np.nanmin(driven[mask]):.3f}/{np.nanmax(driven[mask]):.3f} m/s^2")


def _lamps_build(segs):
  """BRKON from the car's own VSC1S07 (0x320), decoded from the pt-bus frames the log carries."""
  from opendbc.can.parser import CANParser
  from opendbc.car.toyota.values import Bus, DBC

  parser, times, on = None, [], []
  for msg in LogReader(segs, sort_by_time=True):
    if msg.which() == "carParams" and parser is None:
      cp = messaging.log_from_bytes(msg.carParams.as_builder().to_bytes(), car.CarParams)
      parser = CANParser(DBC[cp.carFingerprint][Bus.pt], [("VSC1S07", 50)], 0)
    elif msg.which() == "can" and parser is not None:
      frames = [[c.address, bytes(c.dat), c.src] for c in msg.can if c.address == LAMP_ADDRESS and c.src == 0]
      if frames:
        parser.update([(int(msg.logMonoTime), frames)])
        times.append(int(msg.logMonoTime))
        on.append(bool(parser.vl["VSC1S07"]["BRKON"]))
  return {"lamp_t": np.asarray(times, dtype=np.int64), "lamp": np.asarray(on, dtype=bool)}


def extract_lamps(route, root=DEFAULT_ROOT):
  return cached(route, "lamps", LAMP_VERSION, root, _lamps_build)


def _lamps(args):
  """The car's own brake lamp, beside the planner's press episodes, over engaged speed-control frames above 5 m/s."""
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    driven, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    if _skip_no_engaged(route, active):
      continue
    d = _details_arrays(details)
    lamp = extract_lamps(route, args.root)
    if not len(lamp["lamp"]):
      print(f"{route}: commit {commit}, no VSC1S07 frames in the logs")
      continue
    # Sample-and-hold the lamp at each planner tick, then keep the engaged speed-control frames above 5 m/s.
    held = lamp["lamp"][np.clip(np.searchsorted(lamp["lamp_t"], (d["t"] * 1e9).astype(np.int64), side="right") - 1, 0, len(lamp["lamp"]) - 1)]
    mask = active & (d["v_ego"] > 5.0)
    duration = mask.sum() / 20.0
    press = runs(mask & (driven < -0.4))
    reps = sum(a.start - b.stop <= 2 * 20 for b, a in zip(press, press[1:], strict=False))
    lit = runs(mask & held)
    relights = sum(a.start - b.stop <= 2 * 20 for b, a in zip(lit, lit[1:], strict=False))
    print(f"{route}: commit {commit}, engaged>5 {duration:.1f} s; brake-lamp episodes {len(lit)} ({len(lit) / max(duration, 1) * 60:.2f}/min),"
          + f" re-lights within 2 s {relights} ({relights / max(duration, 1) * 60:.2f}/min), lit {100 * held[mask].mean():.0f}% of frames")
    print(f"  planner presses {len(press)} ({len(press) / max(duration, 1) * 60:.2f}/min), re-presses within 2 s {reps}"
          + " — the lamp is the drive's own; its counterfactual under the band still needs the switching-rate model")


def _lead_ages(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    _, _, active, lead_age = replay(route, extract(route, args.root), args.root)
    if _skip_no_engaged(route, active):
      continue
    ages = lead_age[active & np.isfinite(lead_age)]
    if not len(ages):
      print(f"{route}: commit {commit}, no engaged lead-present frames")
      continue
    print(f"{route}: commit {commit}, frames {len(ages)}, lead age median {np.median(ages) * 1e3:.1f} ms, max {np.max(ages) * 1e3:.1f} ms")


def _curve_metrics(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    driven, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    if _skip_no_engaged(route, active):
      continue
    d = _details_arrays(details)
    mask = active & np.isfinite(logged)
    curve = _curve_mask(d)
    curve &= np.isfinite(logged)
    err = float(np.percentile(np.abs(driven[mask] - logged[mask]), 95)) if mask.any() else float("nan")
    flips = int(np.sum(d["plan_allow_throttle"][mask][1:] != d["plan_allow_throttle"][mask][:-1]))
    duration = mask.sum() / 20.0
    curve_hf = _hf_sd(driven, curve) if curve.any() else float("nan")
    print(f"{route}: commit {commit}, p95 logged {err:.4f}; curve mask |a_y|>=1 m/s^2, frames {curve.sum()}")
    print(f"  curve HF sd current {curve_hf:.4f}, pre-brake frames driven<-1 {int(np.sum(curve & (driven < -1.0)))};",
          f"allowThrottle flips {flips} ({flips / max(duration, 1):.3f}/s)")
    closing = mask & d["lead_present"] & (d["v_ego"] > d["v_lead"])
    print(f"  closing-approach RMS about 1s mean current/logged {_rms_about_mean(driven, closing):.4f}/{_rms_about_mean(logged, closing):.4f}")


def _window_metrics(args, start, end, label):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    driven, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    if _skip_no_engaged(route, active):
      continue
    d = _details_arrays(details)
    elapsed = d["t"] - d["t"][0]
    window = (elapsed >= start) & (elapsed <= end) & active & np.isfinite(logged)
    if not window.any():
      print(f"{route}: commit {commit}, no {label} frames in {start}-{end} s")
      continue
    err = float(np.percentile(np.abs(driven[window] - logged[window]), 95))
    idx = np.flatnonzero(window)
    near = idx[np.argmin(np.abs(d["v_ego"][idx] - 0.30))] if len(idx) else None
    peak_i = idx[np.argmin(driven[idx])] if len(idx) else None
    print(f"{route}: commit {commit}, {label} {start}-{end} s, frames {len(idx)}, p95 logged {err:.4f}")
    if near is not None:
      print(f"  at v=0.30 nearest {d['v_ego'][near]:.3f} m/s: ask {driven[near]:.3f}, logged {logged[near]:.3f}")
    if peak_i is not None:
      print(f"  window peak ask {driven[peak_i]:.3f}, logged {logged[peak_i]:.3f} at v={d['v_ego'][peak_i]:.3f}")
    for i in idx[::20]:
      print(f"  t={elapsed[i]:.1f}s v={d['v_ego'][i]:.2f} ask={driven[i]:.3f} logged={logged[i]:.3f} gap={d['d_rel'][i]:.1f}")


def _queue_stop(args):
  _window_metrics(args, 126.0, 138.0, "queue stop")


def _crawl_stop(args):
  _window_metrics(args, 440.0, 456.0, "crawl stop")


def _rolling_exits(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    driven, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    if _skip_no_engaged(route, active):
      continue
    d = _details_arrays(details)
    mask = active & np.isfinite(logged)
    engaged = np.flatnonzero(mask)
    releases = engaged[1:][(driven[engaged[1:]] > 0.0) & (driven[engaged[:-1]] <= 0.0)]
    if not len(releases):
      print(f"{route}: commit {commit}, no release events")
      continue
    err = float(np.percentile(np.abs(driven[mask] - logged[mask]), 95))
    best = []
    for i in releases:
      j = min(i + 100, len(driven))
      window = np.arange(i + 1, j)
      window = window[mask[window]]
      if not len(window):
        continue
      peak_i = int(window[np.nanargmax(driven[window])])
      peak = float(driven[peak_i])
      rise = float((peak - driven[i]) / max((peak_i - i) / 20.0, 0.05))
      best.append((peak, rise, d["t"][i] - d["t"][0]))
    best.sort(reverse=True)
    print(f"{route}: commit {commit}, p95 logged {err:.4f}, release events {len(best)}")
    for peak, rise, at in best[:3]:
      print(f"  release t={at:.1f}s peak={peak:.3f} rise={rise:.3f} m/s^3")


def _control_build(segs):
  """One row per carControl frame: the tick rate the acceleration controller is written for (`DT_CTRL`).

  Everything else is the newest logged message controlsd had at that tick, so a rate-based controller
  (`MoonpilotLongControl`'s jerk ramps and PID) sees the inputs and the cadence it was tuned at."""
  cp, latest = b"", {}
  rows = []
  # One reader per segment, so each is freed once read: a LogReader over the whole route keeps every segment it has read.
  for msg in (m for seg in segs for m in LogReader(seg, sort_by_time=True)):
    kind = msg.which()
    if kind == "carParams":
      cp = msg.carParams.as_builder().to_bytes()
    elif kind == "carState":
      cs = msg.carState
      latest[kind] = (cs.vEgo, cs.aEgo, cs.standstill, cs.cruiseState.standstill, cs.brakePressed, cs.gasPressed, cs.vCruise)
    elif kind == "longitudinalPlan":
      latest[kind] = (msg.longitudinalPlan.aTarget, msg.longitudinalPlan.shouldStop)
    elif kind == "carControl":
      cc = msg.carControl
      if cp and all(k in latest for k in ("carState", "longitudinalPlan")):
        rows.append((int(msg.logMonoTime), float(cc.longActive), float(cc.actuators.accel), *latest["longitudinalPlan"], *latest["carState"]))
  if not rows:
    raise ValueError("no complete control frames or carParams")
  cols = np.asarray(rows, dtype=float).T
  keys = ("t", "long_active", "accel", "a_target", "should_stop", "v_ego", "a_ego", "standstill", "cruise_standstill",
          "brake_pressed", "gas_pressed", "v_cruise")
  d = dict(zip(keys, cols, strict=True))
  for k in ("long_active", "should_stop", "standstill", "cruise_standstill", "brake_pressed", "gas_pressed"):
    d[k] = d[k].astype(bool)
  d["cp"] = np.frombuffer(cp, dtype=np.uint8)
  return d


def extract_control(route: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
  return cached(route, "control", CONTROL_VERSION, root, _control_build)


def control_rate_dt(t_ns: np.ndarray) -> float:
  """The median spacing of control frames, refusing anything that is not the rate the controller is written for.

  A rate-based controller replayed at the wrong tick rate is a different controller: its jerk ramps and its
  integrator advance per call, so a 20 Hz feed is one fifth the accel change per second. Fail loudly instead."""
  if len(t_ns) < 10:
    raise ValueError(f"only {len(t_ns)} control frames: too few to replay a rate-based controller")
  dt = float(np.median(np.diff(t_ns)) * 1e-9)
  if abs(dt - DT_CTRL) > 0.25 * DT_CTRL:
    raise ValueError(f"control frames are {1e3 * dt:.1f} ms apart, not {1e3 * DT_CTRL:.0f} ms: this is not the control rate")
  return dt


def _longcontrol(args):
  """The acceleration controller on the drive's own control ticks: the same `CI.get_pid_accel_limits` call controlsd
  makes (nothing in cereal logs the limits), the logged plan and `carState`, and the logged accel as the reference."""
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    c = extract_control(route, args.root)
    dt = control_rate_dt(c["t"])
    d = extract(route, args.root)
    details = []
    replayed, _, active, _ = replay(route, d, args.root, details_out=details)
    if not active.any():
      print(f"{route}: no longActive frames, skipping")
      continue
    plan = _details_arrays(details)
    # The replayed plan, held at the value the planner last published, on the control ticks.
    held = np.clip(np.searchsorted(plan["t"], c["t"] * 1e-9, side="right") - 1, 0, len(replayed) - 1)
    cp = messaging.log_from_bytes(d["cp"].tobytes(), car.CarParams)
    ci = interfaces[cp.carFingerprint](cp)
    out_logged, out_replayed = np.zeros(len(c["t"])), np.zeros(len(c["t"]))
    for target, out in ((c["a_target"], out_logged), (replayed[held], out_replayed)):
      loc = MoonpilotLongControl(cp)
      for i in range(len(out)):
        cs = SimpleNamespace(vEgo=float(c["v_ego"][i]), aEgo=float(c["a_ego"][i]), vCruise=float(c["v_cruise"][i]),
                             standstill=bool(c["standstill"][i]), brakePressed=bool(c["brake_pressed"][i]),
                             gasPressed=bool(c["gas_pressed"][i]), cruiseState=SimpleNamespace(standstill=bool(c["cruise_standstill"][i])))
        limits = ci.get_pid_accel_limits(cp, cs.vEgo, cs.vCruise * CV.KPH_TO_MS)
        if not c["long_active"][i]:
          loc.reset()  # controlsd's own line, so the integrator does not carry across a disengagement
        out[i] = loc.update(bool(c["long_active"][i]), cs, float(target[i]), bool(c["should_stop"][i]), limits)
    mask = c["long_active"] & np.isfinite(c["accel"])
    err_logged = float(np.percentile(np.abs(out_logged[mask] - c["accel"][mask]), 95))
    err_replayed = float(np.percentile(np.abs(out_replayed[mask] - c["accel"][mask]), 95))
    plan_err = float(np.percentile(np.abs(replayed[active] - plan["logged_a_target"][active]), 95))
    print(f"{route}: commit {commit}, control frames {mask.sum()} at {1e3 * dt:.1f} ms;")
    print(f"  controller p95 vs logged accel: on the logged plan {err_logged:.4f} (isolates the controller),"
          + f" on the replayed plan {err_replayed:.4f} m/s^2 (that plus the planner's own {plan_err:.4f})")


BRAKE_GAIN_DEPTHS = ((-0.7, -0.3), (-1.2, -0.7), (-2.0, -1.2), (-10.0, -2.0))  # m/s^2 of target
BRAKE_GAIN_SPEEDS = ((2.0, 8.0), (8.0, 15.0), (15.0, 40.0))  # m/s


def _brake_gain_route(job):
  """One drive's target and lag-aligned aEgo over engaged frames above 2 m/s with no pedal and no stop, and whether it
  ran engine coast, which holds the friction brakes off near the coast: a different plant for a shallow ask."""
  route, root = job
  c = extract_control(route, root)
  dt = control_rate_dt(c["t"])
  ok = c["long_active"] & (c["v_ego"] > 2.0) & ~c["brake_pressed"] & ~c["gas_pressed"] & ~c["should_stop"]
  if np.sum(ok & (c["a_target"] < -0.3)) * dt < 5.0:
    return route, None
  lag = lag_s(c["a_target"], c["a_ego"], ok, dt=dt, max_lag=1.0)
  k, half = max(int(round(lag / dt)), 1), int(round(0.5 / dt))
  change = np.full(len(c["t"]), np.inf)  # the target's change over the surrounding second
  change[half:-half] = np.abs(c["a_target"][2 * half:] - c["a_target"][:-2 * half])
  mean = np.ones(half) / half  # the 0.5 s mean every lab peak is read on, applied to the ask and the response alike
  felt, asked = np.convolve(c["a_ego"], mean, mode="same"), np.convolve(c["a_target"], mean, mode="same")
  return route, {"lag": lag, "dt": dt, "cmd": c["a_target"][:-k], "asked": asked[:-k], "a": c["a_ego"][k:], "felt": felt[k:], "v": c["v_ego"][:-k],
                 "ok": ok[:-k] & ok[k:], "steady": change[:-k] < 0.3, "ec": param_index(drive_config(route, root)[1], "MoonpilotEngineCoast")}


def _brake_gain(args):
  """How hard the car brakes for what the planner asks: `carState.aEgo` against the published `aTarget`, aligned by the
  whole chain's own lag (controlsd, Toyota's carcontroller PID, the PCM). Steady frames — the target moving less than
  0.3 m/s^2 over the surrounding second — give the gain the chain settles at, by depth and speed; each brake episode
  (target under -0.7 for 1 s) gives its deepest aEgo, up to 0.5 s past its end, against its deepest target, both 0.5 s means.
  Engine-coast drives and drives whose lag sits on the search bound are listed but not pooled."""
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(4) as pool:
    results = list(pool.map(_brake_gain_route, [(r, args.root) for r in expand(args.routes, ROUTE_SETS)]))
  steady, episodes = [], []
  print("drive: braking s (target < -0.3), lag, steady aEgo / target median, episodes, peak aEgo / peak target median")
  for route, r in results:
    if r is None:
      print(f"  {route}: under 5 s of engaged braking, skipped")
      continue
    braking = r["ok"] & (r["cmd"] < -0.3)
    s = braking & r["steady"]
    tail = int(round(0.5 / r["dt"]))
    mine = []
    for run in runs(r["ok"] & (r["cmd"] < -0.7)):
      if (run.stop - run.start) * r["dt"] >= 1.0:
        i = run.start + int(np.argmin(r["asked"][run]))
        window = np.arange(run.start, min(run.stop + tail, len(r["a"])))
        mine.append((r["asked"][i], r["felt"][window[r["ok"][window]]].min(), r["v"][i]))
    pooled = not r["ec"] and r["lag"] < 0.95
    if pooled:
      steady.append(np.column_stack((r["cmd"][s], r["a"][s], r["v"][s])))
      episodes += mine
    ratio = np.median(r["a"][s] / r["cmd"][s]) if s.any() else np.nan
    peak = np.median([a / c for c, a, _ in mine]) if mine else np.nan
    note = "" if pooled else "  (engine coast on: not pooled)" if r["ec"] else "  (lag on the search bound: not pooled)"
    print(f"  {route}: {braking.sum() * r['dt']:6.1f} s  lag {r['lag']:.3f} s  steady {ratio:.2f} (n {s.sum():5d})  "
          + f"episodes {len(mine):3d}  peak {peak:.2f}{note}")
  st, ep = np.concatenate(steady), np.asarray(episodes)
  print(f"\nsteady braking, aEgo / target median (n): {len(st)} frames of {len(steady)} drives")
  print("  target \\ speed    " + "".join(f"{lo:4.0f}-{hi:<4.0f} m/s     " for lo, hi in BRAKE_GAIN_SPEEDS) + "all")
  for lo, hi in BRAKE_GAIN_DEPTHS:
    cells = []
    for vlo, vhi in (*BRAKE_GAIN_SPEEDS, (0.0, 99.0)):
      m = (st[:, 0] >= lo) & (st[:, 0] < hi) & (st[:, 2] >= vlo) & (st[:, 2] < vhi)
      cells.append(f"{np.median(st[m, 1] / st[m, 0]):5.2f} ({m.sum():6d})" if m.sum() >= 50 else f"{'-':>5} ({m.sum():6d})")
    print(f"  {lo:5.1f}..{hi:5.1f}     " + "   ".join(cells))
  print(f"\nbrake episodes: deepest aEgo against the deepest target, both 0.5 s means, {len(ep)} episodes")
  for lo, hi in BRAKE_GAIN_DEPTHS[1:]:
    m = (ep[:, 0] >= lo) & (ep[:, 0] < hi)
    if m.any():
      ratio, over = ep[m, 1] / ep[m, 0], ep[m, 1] - ep[m, 0]
      print(f"  target {lo:5.1f}..{hi:5.1f} (n {m.sum():3d}): ratio median {np.median(ratio):.2f} p75 {np.percentile(ratio, 75):.2f} "
            + f"p90 {np.percentile(ratio, 90):.2f}; aEgo past the target median {np.median(over):+.2f} p10 {np.percentile(over, 10):+.2f} m/s^2")


def _approach_windows(mask, times, gaps, minimum_duration):
  windows, start = [], None
  for i in range(len(mask) + 1):
    joined = i < len(mask) and mask[i] and (start is None or 0 < times[i] - times[i - 1] <= 0.15)
    if joined:
      if start is None:
        start = i
    elif start is not None:
      end = i - 1
      duration = times[end] - times[start]
      if duration >= minimum_duration:
        windows.append((float(np.median(gaps[start:end + 1])), float(duration)))
      start = i if i < len(mask) and mask[i] else None
  return windows


def _window_summary(label, windows):
  if not windows:
    print(f"  {label}: 0 windows meeting duration threshold")
    return
  gaps = np.asarray([w[0] for w in windows])
  durations = np.asarray([w[1] for w in windows])
  print(f"  {label}: {len(windows)} unpaired lead-slot windows; median gap {np.median(gaps):.2f} m "
        + f"(range {np.min(gaps):.2f}-{np.max(gaps):.2f}); duration median {np.median(durations):.1f} s "
        + f"(range {np.min(durations):.1f}-{np.max(durations):.1f} s)")


def _approach_scenarios():
  from moonpilot.tests.test_longitudinal import _fly

  scenarios = (
    ("steady lead", 12.0, 43.2, 80.0, 3.0, lambda t: 0.0),
    ("stopped lead", 12.0, 43.2, 80.0, 0.0, lambda t: 0.0),
    ("lead brakes after 4s", 12.0, 43.2, 80.0, 3.0, lambda t: -1.5 if t >= 4.0 else 0.0),
    ("ordinary 16->11.7m/s", 16.0, 58.0, 119.0, 11.7, lambda t: 0.0),
  )

  def report(label, result):
    cmd, times, gaps = np.asarray(result["cmd"]), np.arange(len(result["cmd"])) * 0.05, np.asarray(result["gap"])
    for threshold in (-0.05, -0.25):
      crossed = np.flatnonzero(cmd < threshold)
      print(f"  first cmd < {threshold:.2f}: " +
            (f"{times[crossed[0]]:.2f}s, gap {gaps[crossed[0]]:.2f}m" if len(crossed) else "not reached"))
    print(f"  {label}: peak {result['peak']:.3f} m/s^2, end gap {result['end_gap']:.2f} m, "
          + f"minimum gap {result['min_gap']:.2f} m, contact={result['contact']}")

  print("Current full planner, existing _fly point-mass harness (no actuator delay):")
  for label, v0, setpoint, gap, v_lead, accel in scenarios:
    report(label, _fly(v0, setpoint, gap, v_lead, accel, 55.0))


def _lead_approach(args):
  if not args.smoke_only:
    pooled = {"moving": [], "stationary": []}
    for route in expand(args.routes, ROUTE_SETS):
      commit, config = drive_config(route, args.root)
      details = []
      replayed, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
      d = _details_arrays(details)
      mask = active & np.isfinite(logged)
      p95 = float(np.percentile(np.abs(replayed[mask] - logged[mask]), 95)) if mask.any() else float("nan")
      print(f"{route}: commit {commit}, config {config}")
      print(f"  replay-vs-logged p95 |aTarget error| {p95:.4f} m/s^2 on {mask.sum()} engaged frames "
            + "(current-code mismatch is expected on pre-change logs)")
      times, gaps = d["t"], d["d_rel"]
      manual = d["manual_authority"] & d["radar_present"] & d["lead_present"]
      manual_seconds = float(np.sum(np.minimum(np.diff(times, prepend=times[0]), 0.1) * manual))
      moving = manual & (d["v_lead"] >= 1.0) & (d["v_lead"] < 5.0) & (np.abs(d["v_ego"] - d["v_lead"]) < 0.5) & (gaps > 2.0) & (gaps < 30.0)
      stationary = manual & d["brake_pressed"] & (d["v_ego"] < 0.05) & (np.abs(d["v_lead"]) < 0.2) & (gaps > 2.0) & (gaps < 20.0)
      windows = {"moving": _approach_windows(moving, times, gaps, 1.0), "stationary": _approach_windows(stationary, times, gaps, 2.0)}
      _window_summary("moving lead (1<=vLead<5, speed match <0.5 m/s)", windows["moving"])
      _window_summary("braked stationary lead", windows["stationary"])
      for kind, found in windows.items():
        pooled[kind] += found
      print(f"  manual-authority lead data: {manual_seconds / 60:.1f} minutes")
    print("All routes:")
    _window_summary("moving lead (1<=vLead<5, speed match <0.5 m/s)", pooled["moving"])
    _window_summary("braked stationary lead", pooled["stationary"])
    print("  Descriptive unpaired lead-slot windows only; radar track IDs may swap. Not a fixed driver target or safety proof.")
  _approach_scenarios()


def _register_lead_approach(sub):
  parser = sub.add_parser("lead-approach", help="manual lead windows and full-planner approach diagnostic")
  parser.add_argument("--routes", nargs="+", default=["lead-approach"], help=f"route names or IDs: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--smoke-only", action="store_true", help="skip route replay and run only planner scenarios")
  parser.set_defaults(fn=_lead_approach)


def _manual_frames(route, root=DEFAULT_ROOT):
  """The driver's own longitudinal and the radar lead, one row per planner tick of the route's extract, with the lead's
  accel from the planner's own `LeadAccelEstimator` run over the same rows."""
  from moonpilot.lead import LeadAccelEstimator

  def build(_segs):
    data = extract(route, root)
    blob, offsets = data["blob"], data["offsets"]
    index = {k: 1 + i * 3 for i, k in enumerate(MESSAGES)}
    estimator, rows = LeadAccelEstimator(), []
    for row in offsets:
      def read(kind, row=row):
        start, size = row[index[kind]], row[index[kind] + 1]
        return messaging.log_from_bytes(blob[start:start + size].tobytes())
      cs, cc, lead = read("carState").carState, read("carControl").carControl, read("radarState").radarState.leadOne
      a_lead, _ = estimator.update(lead)
      manual = not cc.longActive and not cs.cruiseState.enabled and cs.gearShifter == car.CarState.GearShifter.drive
      rows.append((row[0] * 1e-9, manual, cs.vEgo, cs.aEgo, cs.gasPressed, cs.brakePressed, lead.present and lead.radar, lead.dRel, lead.vLead,
                   a_lead, estimator.a_slow))
    keys = ("t", "manual", "v", "a", "gas", "brake", "lead", "gap", "v_lead", "a_lead", "a_slow")
    return {k: np.asarray(col, dtype=float) for k, col in zip(keys, zip(*rows, strict=True), strict=True)}

  return cached(route, "manual", MANUAL_VERSION, root, build)


def _pooled_manual(routes, root):
  """Every route's manual frames end to end, with `route` numbering them so episodes never span two drives."""
  frames = [_manual_frames(r, root) for r in routes]
  d = {k: np.concatenate([f[k] for f in frames]) for k in frames[0] if k != "segments"}
  d["route"] = np.concatenate([np.full(len(f["t"]), i) for i, f in enumerate(frames)])
  return d


def _bin_rows(d, mask, need, bins):
  for lo, hi in zip(bins[:-1], bins[1:], strict=True):
    m = mask & (need >= lo) & (need < hi)
    if m.any():
      print(f"    [{lo:5.2f}, {hi:5.2f})  {m.sum():6d}  gas {d['gas'][m].mean() * 100:3.0f}%  brake {d['brake'][m].mean() * 100:3.0f}%  "
            + f"median aEgo {np.median(d['a'][m]):6.2f} (p25 {np.percentile(d['a'][m], 25):6.2f}, p75 {np.percentile(d['a'][m], 75):6.2f})")


HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S = 0.2, 0.25  # the plant: dead time, then a first-order lag on the command


class _LaggedPlant:
  """Acceleration following the command after `dead` s of dead time, then a first-order lag of `tau` s (by default
  `HOLD_FLY_DEAD_S` and `HOLD_FLY_TAU_S`), exactly discretized at the tick: the command fed on frame i first moves `a` on
  frame i + dead/dt."""

  def __init__(self, dead=HOLD_FLY_DEAD_S, tau=HOLD_FLY_TAU_S, dt=DT_MDL):
    self.queue = deque([0.0] * round(dead / dt))
    self.alpha = 1.0 - math.exp(-dt / tau)
    self.a = 0.0

  def step(self, cmd):
    delayed = self.queue.popleft()  # the oldest leaves before this frame's command joins
    self.queue.append(cmd)
    self.a += (delayed - self.a) * self.alpha
    return self.a



def _fly_lagged(v0, v_set_kph, gap0, v_lead0, a_lead_fn, T, seed=None):
  """The engine-coast RAV4 planner against a scripted lead through a lagged point-mass plant; with a seed, white radar
  noise on `dRel` (0.3 m + 0.4 %) and `vLead` (0.05 m/s)."""
  from moonpilot.tests.test_engine_coast import _planner
  from moonpilot.tests.test_longitudinal import _inputs, _lead
  rng = np.random.default_rng(seed) if seed is not None else None
  planner, plant, v, gap, v_lead = _planner(True), _LaggedPlant(), v0, gap0, v_lead0
  out = {k: [] for k in ("v", "gap", "v_lead", "cmd")}
  for i in range(round(T / DT_MDL)):
    d_seen, v_seen = gap, v_lead
    if rng is not None:
      d_seen, v_seen = gap + rng.normal(0.0, 0.3 + 0.004 * gap), v_lead + rng.normal(0.0, 0.05)
    planner.update(_inputs(v_ego=max(v, 0.0), v_cruise_kph=v_set_kph, a_ego=plant.a, lead=_lead(max(d_seen, 0.0), max(v_seen, 0.0)), personality=1))
    for k, x in zip(out, (v, gap, v_lead, float(planner.output_a_target)), strict=True):
      out[k].append(x)
    a = plant.step(float(planner.output_a_target))
    v_next = max(0.0, v_lead + float(a_lead_fn(i * DT_MDL, v_lead)) * DT_MDL)
    v = max(0.0, v + a * DT_MDL)
    gap += (0.5 * (v_lead + v_next) - v) * DT_MDL
    v_lead = v_next
  return {k: np.asarray(x) for k, x in out.items()}


def _rest_gaps(route, v, lead, v_lead, gap, settle_s=1.0):
  """The gap behind a stopped radar lead, once per ego stop: the ego arms above 2 m/s and the stop fires on its first
  standstill frame (below 0.1 m/s) in the same route; the lead is read on that frame or up to `settle_s` into the
  standstill (a radar dropout at the stop frame). Only motion above 2 m/s re-arms, so radar reacquisition or `vLead`
  jitter during one standstill, and a creep and re-stop, are the same stop."""
  settle = round(settle_s / DT_MDL)
  gaps, armed = [], False
  for i in range(len(v)):
    if i and route[i] != route[i - 1]:
      armed = False
    if v[i] > 2.0:
      armed = True
    elif armed and v[i] < 0.1:
      armed = False
      for j in range(i, min(i + settle, len(v))):
        if route[j] != route[i] or v[j] >= 0.1:
          break
        if lead[j] and v_lead[j] < 0.2 and gap[j] < 20.0:
          gaps.append(float(gap[j]))
          break
  return np.asarray(gaps)


def _manual_decel(args):
  """The lead law's approach requirement against the driver's own longitudinal: `stopping_decel` (the lead's braking held
  for up to its sustain, positive lead accel ignored) to the arrival target, max(moving rest, t_follow * v_lead), on
  manual frames behind a radar lead; then where the driver actually ends up — rest gaps and settled headways."""
  from moonpilot.longitudinal import MOONPILOT_STOP_REST, lead_accel_estimate, stopping_decel
  t_follow = args.t_follow
  routes = expand(args.routes, ROUTE_SETS)
  d = _pooled_manual(routes, args.root)
  v, v_lead, gap = d["v"], d["v_lead"], d["gap"]
  target = _arrival_target(v_lead, t_follow)
  lead = (d["manual"] > 0) & (d["lead"] > 0)
  a_lead = np.array([lead_accel_estimate(x) for x in d["a_lead"]])
  need = np.array([stopping_decel(*x) if m else 0.0 for m, x in zip(lead, zip(v, v_lead, a_lead, np.maximum(gap - target, 0.1), strict=True), strict=True)])
  approach = lead & (v > 2.0) & (gap - target > 1.0) & (need < 0.0)
  # A lead that leaves within 5 s (radar lead gone, or its gap jumping 5 m in a frame: another object) is one the
  # driver may already be passing or watching turn off; the ratios below are on the leads that stay.
  leaves = np.zeros(len(v), dtype=bool)
  for i in np.flatnonzero(approach):
    ahead = slice(i, min(i + round(5.0 / DT_MDL), len(v)))
    same = d["route"][ahead] == d["route"][i]
    leaves[i] = not (d["lead"][ahead][same] > 0).all() or bool((np.abs(np.diff(gap[ahead][same])) > 5.0).any())
  to_target = (gap - target) / np.maximum(v - v_lead, 0.05)
  print(f"{approach.sum() * DT_MDL / 60:.1f} min of manual approach frames on {len(set(d['route'][approach]))} of {len(routes)} routes; "
        + "the requirement, binned, against what the driver did:")
  for lo, hi in ((2.0, 8.0), (8.0, 15.0), (15.0, 45.0)):
    band = approach & (v >= lo) & (v < hi)
    deep = band & (need <= -0.2)
    stays = deep & ~leaves
    ratio = d["a"][stays] / need[stays]
    print(f"  {lo:.0f}-{hi:.0f} m/s: {band.sum() * DT_MDL / 60:.1f} min. Asking -0.2 or deeper, the lead leaves within 5 s on "
          + f"{leaves[deep].mean() * 100:.0f}% of frames; where it stays the driver's aEgo / requirement is "
          + (f"median {np.median(ratio):.2f} (p25 {np.percentile(ratio, 25):.2f}, p75 {np.percentile(ratio, 75):.2f})" if ratio.size else "n/a"))
    by_time = [(a, b, stays & (to_target >= a) & (to_target < b)) for a, b in ((0, 3), (3, 6), (6, 10), (10, np.inf))]
    print("    by time to the target at the current closing: " + ", ".join(
      f"{a}-{b} s {np.median(d['a'][m] / need[m]):.2f} (n {m.sum()})" for a, b, m in by_time if m.sum() >= 50))
    _bin_rows(d, band, need, (-10.0, -1.5, -1.0, -0.5, -0.33, -0.2, -0.1, 0.0))

  rests = _rest_gaps(d["route"], v, lead, v_lead, gap)
  if rests.size:
    print(f"\nRest gap behind a stopped lead ({rests.size} stops from above 2 m/s): median {np.median(rests):.2f} m, p10 {np.percentile(rests, 10):.2f}, "
          + f"p90 {np.percentile(rests, 90):.2f}; the law's rest point is {MOONPILOT_STOP_REST} m")
  follow = lead & (v > 8.0) & (np.abs(v - v_lead) < 0.3) & (np.abs(a_lead) < 0.3)
  headway = gap[follow] / v[follow]
  if headway.size:
    print(f"Settled following above 8 m/s ({follow.sum() * DT_MDL / 60:.1f} min, speeds within 0.3 m/s, lead steady): headway median "
          + f"{np.median(headway):.2f} s, p10 {np.percentile(headway, 10):.2f}, p90 {np.percentile(headway, 90):.2f}; the law's is {t_follow} s")


def _arrival_target(v_lead, t_follow):
  """The lead law's near gap behind a lead at constant speed: max(the moving rest, t_follow * v_lead)."""
  from moonpilot.longitudinal import MOONPILOT_CREEP_SPEED, MOONPILOT_MOVING_STOP_RESERVE, MOONPILOT_STOP_DISTANCE, MOONPILOT_STOP_REST
  reserve = MOONPILOT_STOP_DISTANCE - MOONPILOT_STOP_REST + MOONPILOT_MOVING_STOP_RESERVE
  return np.maximum(MOONPILOT_STOP_REST + reserve * np.clip(v_lead / MOONPILOT_CREEP_SPEED, 0.0, 1.0), t_follow * v_lead)


def _lead_ask(t_follow):
  """(v, gap, v_lead, a_lead, a_slow) -> the lead law's ask as `policy` gives it for the engine-coast RAV4 with cruise at the
  ego's speed, before the brake band."""
  import moonpilot.longitudinal as longitudinal
  from opendbc.car.toyota.interface import CarInterface
  from opendbc.car.toyota.values import CAR, ToyotaFlags
  CP = CarInterface.get_non_essential_params(CAR.TOYOTA_RAV4_TSS2)
  CP.openpilotLongitudinalControl = True
  CP.flags |= ToyotaFlags.MOONPILOT_ENGINE_COAST.value
  src = longitudinal.LongitudinalPlanSource.lead0

  def ask(v, gap, v_lead, a_lead, a_slow):
    lead = (src, gap, v_lead, a_lead, longitudinal.brake_sustain(a_lead), a_slow)
    return longitudinal.policy(v, [lead], v, t_follow, False, None, 0.0, CP, -0.3, True)[0]
  return ask


def _approach_runs(t, route, ok, gap, target, lead_in=10.0, min_s=3.0):
  """Runs of `ok` frames (one route, the gap moving under 5 m a frame) lasting `min_s`, starting `lead_in` m past the target."""
  runs, start = [], None
  for i in range(len(ok) + 1):
    if i < len(ok) and ok[i] and start is not None and route[i] == route[i - 1] and abs(gap[i] - gap[i - 1]) < 5.0:
      continue
    if start is not None and t[i - 1] - t[start] >= min_s and gap[start] - target[start] >= lead_in:
      runs.append((start, i))
    start = i if i < len(ok) and ok[i] else None
  return runs


def _onset(event, s, e, hold=1):
  """First index in [s, e) where boolean `event` has held `hold` frames; -1 if it already holds at s (it began before the
  window, onset unknown: left-censored); None if it never does."""
  held = np.convolve(event[s:e], np.ones(hold), mode="valid") >= hold
  if not held.any():
    return None
  k = int(np.argmax(held))
  return -1 if k == 0 else s + k


def _paired(t, runs, ref, cand, shift):
  """`cand`'s onset (+ `shift` s) against `ref`'s, on runs where both are observed."""
  first, after, censored = [], [], 0
  for s, e in runs:
    a, b = ref(s, e), cand(s, e)
    if a == -1 or b == -1:
      censored += 1
    elif a is not None and b is not None:
      dt = t[a] - (t[b] + shift)
      (first if dt > 0.0 else after).append(abs(dt))
  n = len(first) + len(after)

  def med(x):
    return f"{np.median(x):4.1f} s" if x else "   - "
  return f"first {len(first):3d}/{n:<3d} by {med(first)}, after {len(after):3d} by {med(after)}, censored {censored}"


def _onset_report(label, runs, t, v, gap, closing, rows):
  """`rows`: (name, reference onset, candidate onset, shift s), each onset `(s, e) -> index | -1 | None`."""
  print(f"\n{label}: {len(runs)} approaches")
  for lo, hi in ((5.0, 12.0), (12.0, 20.0), (20.0, 45.0)):
    band = [(s, e) for s, e in runs if lo <= v[s] < hi]
    if band:
      print(f"  {lo:.0f}-{hi:.0f} m/s at the start: {len(band)}, gap {np.median([gap[s] for s, _ in band]):.0f} m, "
            + f"closing {np.median([closing[s] for s, _ in band]):.1f} m/s (medians)")
      for name, ref, cand, shift in rows:
        print(f"    {name:40s} {_paired(t, band, ref, cand, shift)}")


def _approach_onsets(args):
  """Where approaches to a slower radar lead start slowing — descriptive. Manual drives: the lead law is asked at
  the states the driver drove (cruise at the ego's speed, so it holds where the driver may still be accelerating) and
  its ask onset compared with the driver's decisions (gas release, brake press) and, shifted by `--lag`, with the
  driver's physical slowing. A closed-loop law would leave the driver's states once its command differs, so 'first' and
  'after' do not predict closed-loop onset. Windows that open already slowing are left-censored, not counted. Engaged
  drives (`--engaged`): the logged command against today's planner replayed on them."""
  t_follow = args.t_follow
  d = _pooled_manual(expand(args.routes, ROUTE_SETS), args.root)
  t, v, gap, v_lead = d["t"], d["v"], d["gap"], d["v_lead"]
  target, closing = _arrival_target(v_lead, t_follow), v - v_lead
  ok = (d["manual"] > 0) & (d["lead"] > 0) & (v > 5.0) & (closing > 1.0) & (gap > target)
  runs = _approach_runs(t, d["route"], ok, gap, target)
  idx = np.unique(np.concatenate([np.arange(s, e) for s, e in runs])) if runs else np.array([], dtype=int)
  gas, brake, a = d["gas"] > 0, d["brake"] > 0, d["a"]
  hold = round(0.5 / DT_MDL)
  ask_now = _lead_ask(t_follow)
  ask = np.zeros(len(v))
  ask[idx] = [ask_now(v[i], gap[i], v_lead[i], d["a_lead"][i], d["a_slow"][i]) for i in idx]
  rows = [
    ("ask -0.05 vs gas release", lambda s, e: _onset(~gas, s, e), lambda s, e: _onset(ask <= -0.05, s, e), 0.0),
    ("ask -0.5 vs brake press", lambda s, e: _onset(brake, s, e), lambda s, e: _onset(ask <= -0.5, s, e), 0.0),
    ("ask -0.05 + lag vs aEgo -0.05", lambda s, e: _onset(a <= -0.05, s, e, hold), lambda s, e: _onset(ask <= -0.05, s, e), args.lag),
  ]
  _onset_report("Manual approaches: law asks on the driver's states; 'first' = the law's onset precedes the driver's", runs, t, v, gap, closing, rows)

  for route in expand(args.engaged, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    replayed, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    e = _details_arrays(details)
    target_e, closing_e = _arrival_target(e["v_lead"], t_follow), e["v_ego"] - e["v_lead"]
    ok_e = active & e["lead_present"] & e["radar_present"] & (e["v_ego"] > 5.0) & (closing_e > 1.0) & (e["d_rel"] > target_e)
    runs_e = _approach_runs(e["t"], np.zeros(len(ok_e)), ok_e, e["d_rel"], target_e)
    m = active & np.isfinite(logged)
    p95 = float(np.percentile(np.abs(replayed - logged)[m], 95)) if m.any() else math.nan
    logged = np.nan_to_num(logged)
    rows_e = [(f"now vs logged at {thr}", lambda s, e, x=logged <= thr: _onset(x, s, e), lambda s, e, x=replayed <= thr: _onset(x, s, e), 0.0)
              for thr in (-0.05, -0.3)]
    _onset_report(f"Engaged {route} (ran {commit[:9]}; replay vs logged p95 {p95:.3f})", runs_e, e["t"], e["v_ego"], e["d_rel"], closing_e, rows_e)


def _plant_inputs(v, a, gap, v_lead, a_lead, v_cruise):
  """Upstream's maneuver plant's messages (`plant.py` step): radar lead in path, a model that does not see the lead."""
  from openpilot.cereal import log
  from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
  from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
  from openpilot.selfdrive.modeld.constants import ModelConstants
  radar, control, ss, cs, model = (messaging.new_message(s) for s in ("radarState", "controlsState", "selfdriveState", "carState", "modelV2"))
  lead = log.RadarState.LeadData.new_message(dRel=float(gap), vRel=float(v_lead - v), vLead=float(v_lead), vLeadK=float(v_lead), aLeadK=float(a_lead),
                                             aLeadTau=float(_LEAD_ACCEL_TAU), present=True, modelProb=1.0, radar=True)
  radar.radarState.leadOne = radar.radarState.leadTwo = lead
  model.modelV2.position = log.XYZTData.new_message(x=[float(x) for x in (v + 0.5) * np.array(ModelConstants.T_IDXS)])
  model.modelV2.action.desiredAcceleration = float(a + 0.5)
  model.modelV2.velocity = log.XYZTData.new_message(x=[float(v)] + [float(v + 0.5)] * (len(ModelConstants.T_IDXS) - 1))
  model.modelV2.acceleration = log.XYZTData.new_message(x=[0.0] * len(ModelConstants.T_IDXS))
  control.controlsState.longControlState = LongCtrlState.pid
  ss.selfdriveState.personality = log.LongitudinalPersonality.standard
  cs.carState.vEgo, cs.carState.aEgo, cs.carState.vCruise = float(v), float(a), float(v_cruise * 3.6)
  return {"radarState": radar.radarState, "carState": cs.carState, "controlsState": control.controlsState,
          "selfdriveState": ss.selfdriveState, "modelV2": model.modelV2}


def _hard_brake_run(kind, v0, a_brake, lag, onset=5.0, T=30.0):
  """A 1.45 s follow at v0 whose lead brakes to rest at `onset`: (min gap, peak, s to an ask <= -3.0, s to FCW)."""
  from openpilot.selfdrive.controls.lib.longitudinal_planner import ACCEL_MIN, LongitudinalPlanner
  from opendbc.car.toyota.interface import CarInterface
  from opendbc.car.toyota.values import CAR
  from moonpilot.tests.test_engine_coast import _planner
  from moonpilot.tests.test_longitudinal import _plant_sm
  if kind == "upstream":
    CP = CarInterface.get_non_essential_params(CAR.TOYOTA_RAV4_TSS2)
    CP.openpilotLongitudinalControl = True
    planner = LongitudinalPlanner(CP, init_v=v0)
    update = planner.update
  elif kind == "ideal":
    planner, update = None, None
  else:
    planner = _planner(kind == "fork+engine")
    update = lambda sm: planner.update(_plant_sm(sm))  # noqa: E731
  plant = _LaggedPlant() if lag else None
  v, v_lead, gap, a, v_prev = v0, v0, 1.45 * v0, 0.0, v0
  gaps, cmds, fcw = [], [], []
  for i in range(round(T / DT_MDL)):
    t = i * DT_MDL
    if planner is None or update is None:
      cmd, warned = (ACCEL_MIN if t >= onset else 0.0), False
    else:
      update(_plant_inputs(v, a, gap, v_lead, (v_lead - v_prev) / DT_MDL, v0 + 1.0))
      cmd, warned = float(planner.output_a_target), bool(planner.fcw)
    v_prev = v_lead
    cmds.append(cmd)
    fcw.append(warned)
    a = plant.step(cmd) if plant else cmd
    v_next = max(0.0, v_lead + (a_brake * DT_MDL if t >= onset else 0.0))
    v_ego_next = max(0.0, v + a * DT_MDL)
    gap += (0.5 * (v_lead + v_next) - 0.5 * (v + v_ego_next)) * DT_MDL  # both cars on the trapezoid
    v, v_lead = v_ego_next, v_next
    gaps.append(gap)
  deep, warn = np.flatnonzero(np.asarray(cmds) <= -3.0), np.flatnonzero(fcw)
  return min(gaps), min(cmds), (deep[0] * DT_MDL - onset) if deep.size else math.nan, (warn[0] * DT_MDL - onset) if warn.size else math.nan


def _hard_brake(args):
  """A lead braking hard to rest from a 1.45 s follow: upstream's planner, the fork's (default, engine coast) and an ideal
  response (ACCEL_MIN from the lead's first braking frame), on upstream's plant inputs, through no plant and the lagged
  one. Contact is a min gap under 0.4 m, upstream's maneuver bar."""
  kinds = ("upstream", "fork", "fork+engine", "ideal")
  print("min gap m / peak / s to ask <= -3.0 / s to FCW, per planner")
  print(f"{'plant':5s} {'v0':>4s} {'lead a':>6s}  " + "  ".join(f"{k:>30s}" for k in kinds))
  for lag in (False, True):
    for v0 in (20.0, 25.0, 30.0):
      for a_brake in (-3.5, -5.0):
        cells = (_hard_brake_run(k, v0, a_brake, lag) for k in kinds)
        print(f"{'lag' if lag else 'none':5s} {v0:4.0f} {a_brake:6.1f}  "
              + "  ".join(f"{g:7.2f} {pk:5.2f} {d:5.2f} {w:5.2f}{' X' if g < 0.4 else '  '}" for g, pk, d, w in cells))


SUSTAIN_RULES = {  # sustain(v_lead, a_lead, braking_s, shipped): braking_s is how long the planner's lead accel has read <= -1.0
  "2.0 s": lambda v, a, b, s: 2.0,
  "3.0 s": lambda v, a, b, s: 3.0,
  "to rest": lambda v, a, b, s: math.inf,
  "|a| s": lambda v, a, b, s: s,  # shipped: brake_sustain of the measured accel (None: of the one given)
  "2 s+braking": lambda v, a, b, s: 2.0 + b,
  "2 s+2x braking": lambda v, a, b, s: 2.0 + 2.0 * b,
}


@contextmanager
def _sustain_rule(rule):
  """Every comfort-horizon `stopping_decel` call takes `rule`'s sustain (FCW's `inf` passes through). The braking time is
  counted per planner update from the deepest lead accel the previous update's calls saw, so it reads only what the
  planner itself observed."""
  from unittest import mock
  import moonpilot.longitudinal as longitudinal
  original, update = longitudinal.stopping_decel, MoonpilotLongitudinalPlanner.update
  state = {"deepest": 0.0, "braking": 0.0}

  def decel(v_ego, v_lead, a_lead, slack, sustain=None):
    if sustain is not None and math.isinf(sustain):
      return original(v_ego, v_lead, a_lead, slack, sustain)
    state["deepest"] = min(state["deepest"], float(a_lead))
    return original(v_ego, v_lead, a_lead, slack, rule(v_lead, float(a_lead), state["braking"], sustain))

  def tick(self, sm):
    state["braking"] = state["braking"] + DT_MDL if state["deepest"] <= -1.0 else 0.0
    state["deepest"] = 0.0
    return update(self, sm)

  with mock.patch.object(longitudinal, "stopping_decel", decel), mock.patch.object(MoonpilotLongitudinalPlanner, "update", tick):
    yield


def _sustain_flights(names):
  from moonpilot.tests.test_longitudinal import _fly
  print("hard brake to rest from a 1.45 s follow (fork, RAV4): min gap m, contact < 0.4")
  print(f"{'plant':5s} {'v0':>4s} {'lead a':>6s}  " + "  ".join(f"{n:>14s}" for n in names))
  for lag in (False, True):
    for v0 in (20.0, 25.0, 30.0):
      for a_brake in (-3.5, -5.0):
        gaps = []
        for n in names:
          with _sustain_rule(SUSTAIN_RULES[n]):
            gaps.append(_hard_brake_run("fork", v0, a_brake, lag)[0])
        print(f"{'lag' if lag else 'none':5s} {v0:4.0f} {a_brake:6.1f}  " + "  ".join(f"{g:12.2f}{' X' if g < 0.4 else '  '}" for g in gaps))
  taps = (  # (label, v0, gap0, lead brake, tap length s)
    ("30 m/s, 60 m, -5 x 0.5 s", 30.0, 60.0, -5.0, 0.5),
    ("25 m/s follow, -3.5 x 1 s", 25.0, 36.25, -3.5, 1.0),
    ("20 m/s follow, -2 x 1.5 s", 20.0, 29.0, -2.0, 1.5),
    ("15 m/s follow, -3 x 1 s", 15.0, 21.75, -3.0, 1.0),
  )
  print("\nlead taps then holds speed: peak command m/s^2 (end gap m); the brake band presses below -0.40")
  print(f"{'scenario':28s}  " + "  ".join(f"{n:>14s}" for n in names))
  for label, v0, gap0, brake, length in taps:
    cells = []
    for n in names:
      with _sustain_rule(SUSTAIN_RULES[n]):
        s = _fly(v0, v0 * 3.6 + 10.0, gap0, v0, lambda t, b=brake, n_=length: b if 1.0 < t < 1.0 + n_ else 0.0, 20.0)
      cells.append(f"{s['peak']:6.2f} ({s['end_gap']:5.1f})")
    print(f"{label:28s}  " + "  ".join(f"{c:>14s}" for c in cells))


def _sustain_drives(args, names):
  print("engaged drives replayed: replay vs logged p95 (current code), then frames deeper than 2.0 s by > 0.05 m/s^2, deepest delta")
  for route in expand(args.routes, ROUTE_SETS):
    data = extract(route, args.root)
    shipped, logged, active, _ = replay(route, data, args.root)
    if _skip_no_engaged(route, active):
      continue
    m = active & np.isfinite(logged)
    with _sustain_rule(SUSTAIN_RULES["2.0 s"]):
      base = replay(route, data, args.root)[0]
    cells = []
    for n in names[1:]:
      with _sustain_rule(SUSTAIN_RULES[n]):
        cmd = replay(route, data, args.root)[0]
      delta = (cmd - base)[active]
      cells.append(f"{n} {int((delta < -0.05).sum()):4d} {delta.min():6.2f}")
    err = float(np.percentile(np.abs(shipped - logged)[m], 95)) if m.any() else math.nan
    print(f"{route} ({int(active.sum())} engaged, p95 {err:.4f}): " + "  ".join(cells), flush=True)


def _sustain(args):
  """How long `stopping_decel` assumes a braking lead keeps braking: the fixed 2.0 s against longer and scaled horizons,
  on the hard brakes it must survive, the taps it must not over-answer, and the engaged drives' own commands."""
  names = ("2.0 s", *(n for n in args.rules if n != "2.0 s"))
  if args.part != "drives":
    _sustain_flights(names)
  if args.part != "flights":
    _sustain_drives(args, names)


PEAK_DECEL_A, PEAK_LEAD_IN_S, PEAK_TAIL_S = -0.3, 4.0, 4.0  # a decel: aEgo over 0.5 s below this; flown this much either side


def _advance(s, v, v_new):
  """One tick of travel on the trapezoid: the logged lead path and the flown ego both step with this, so flying the
  logged speeds recovers the logged `dRel` exactly."""
  return s + 0.5 * (v + v_new) * DT_MDL


def _calibration(route, root):
  """The learned horizons a drive's initData recorded — longitudinal lag and radar latency — as planner params."""
  import json
  cfg, out = drive_config(route, root)[1], {}
  for key, kind in (("MoonpilotLongLag", float), ("MoonpilotLongLagBlocks", int), ("MoonpilotLongLagHistory", json.loads),
                    ("MoonpilotRadarLatency", float), ("MoonpilotRadarLatencySamples", int)):
    if cfg.get(key):
      out[key] = kind(cfg[key])
  return out


def _in_flight_speed(planner, v_ego, a_prev, standstill):
  """The speed the car reaches over `action_t` on the commands still in flight — the last `action_t - dt` issued, oldest
  first, one tick each — then `a_prev` for what is left: exact on a pure dead-time plant with the planner's delay."""
  if not hasattr(planner, "issued"):
    planner.issued = deque(maxlen=round(MOONPILOT_LAG_MAX / planner.dt) + 1)  # the first frame: nothing issued yet
  else:
    planner.issued.append(planner.output_a_target)
  issued = [max(a, 0.0) for a in planner.issued] if standstill else list(planner.issued)
  n = min(int(round((planner.action_t - planner.dt) / planner.dt, 6)), len(issued))
  return max(0.0, v_ego + planner.dt * sum(issued[len(issued) - n:]) + (planner.action_t - n * planner.dt) * a_prev)


@cache
def _in_flight_update():
  """`MoonpilotLongitudinalPlanner.update` with its ego projection — the latest ask spent over the whole delay — replaced
  by `_in_flight_speed`. The variant `delay-cost` measures and rejects, kept here so the claim stays reproducible."""
  import inspect
  import textwrap
  import moonpilot.longitudinal as longitudinal
  line = "v_pred = max(0.0, v_ego + a_prev * self.action_t)"
  source = textwrap.dedent(inspect.getsource(longitudinal.MoonpilotLongitudinalPlanner.update))
  if source.count(line) != 1:
    raise ValueError(f"the planner no longer projects with `{line}`: rebuild the in-flight variant against what it does now")
  scope = dict(vars(longitudinal), _in_flight_speed=_in_flight_speed)
  exec(compile(source.replace(line, "v_pred = _in_flight_speed(self, v_ego, a_prev, CS.standstill)"), longitudinal.__file__, "exec"), scope)
  return scope["update"]


def _fly_logged_lead(job, plant=(HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S), engine_coast=True, delay=None, in_flight=False):
  """The RAV4 planner (engine coast on unless `engine_coast` is False), seeded with `calibration`, on the lagged plant
  (`plant`: dead time, first-order lag) from a logged start behind the logged lead's own path. The set speed is the
  start's unless a sixth element gives one, so by default only the lead terms brake; a seventh is the drive's model ask,
  frame by frame. `delay` replaces the delay the planner projects through, `in_flight` its projection (`delay-cost`)."""
  from moonpilot.tests.test_engine_coast import _planner
  from moonpilot.tests.test_longitudinal import _inputs, _lead
  v_log, gap_log, v_lead, a0, calibration, *extra = job
  v_cruise_kph = extra[0] if extra else v_log[0] * 3.6
  model = extra[1] if len(extra) > 1 else np.zeros(len(v_log))
  s_ego = [0.0]
  for v0, v1 in zip(v_log[:-1], v_log[1:], strict=True):
    s_ego.append(_advance(s_ego[-1], v0, v1))
  s_lead = np.asarray(s_ego) + gap_log
  planner, plant = _planner(engine_coast, calibration), _LaggedPlant(*plant)
  if delay is not None:
    planner.long_lag.applied_delay = lambda: delay  # ty: ignore[invalid-assignment]  # this flight's delay, same signature
  if in_flight:
    planner.update = MethodType(_in_flight_update(), planner)  # ty: ignore[invalid-assignment]  # the rejected variant, same signature
  plant.a, plant.queue = a0, deque([a0] * len(plant.queue))
  v, s, out = v_log[0], 0.0, {"v": [], "a": [], "gap": [], "pressing": [], "cmd": []}
  for i in range(len(v_log)):
    planner.update(_inputs(v_ego=v, v_cruise_kph=v_cruise_kph, a_ego=plant.a, lead=_lead(max(s_lead[i] - s, 0.0), max(v_lead[i], 0.0)),
                           standstill=v < 0.05, personality=1, model_accel=float(model[i])))
    for k, x in zip(out, (v, plant.a, s_lead[i] - s, planner.brake_band.pressing, float(planner.output_a_target)), strict=True):
      out[k].append(x)
    v_new = max(0.0, v + plant.step(float(planner.output_a_target)) * DT_MDL)
    s, v = _advance(s, v, v_new), v_new
  return {k: np.asarray(x) for k, x in out.items()}


def _centered_slope(v, half):
  """The least-squares slope of `v` over `half` samples either side, NaN where the window does not fit."""
  n = 2 * half + 1
  out = np.full(len(v), np.nan)
  if len(v) >= n:
    out[half:-half] = np.convolve(v, ((np.arange(n) - half) * (12.0 / (DT_MDL * n * (n * n - 1))))[::-1], mode="valid")
  return out


def _decel_jobs(d, a_s):
  """Every manual decel a radar lead caused (aEgo over 0.5 s, `a_s`, past PEAK_DECEL_A for 2 s, shedding 2 m/s from 4 m/s,
  one radar track throughout), as flight jobs from PEAK_LEAD_IN_S before to PEAK_TAIL_S after, and the driver's (onset
  speed, peak decel above 2 m/s, minimum gap) on each."""
  from moonpilot.lab.model_braking import episodes
  v, gap, v_lead = d["v"], d["gap"], d["v_lead"]
  n_in, n_tail = round(PEAK_LEAD_IN_S / DT_MDL), round(PEAK_TAIL_S / DT_MDL)
  jobs, drv = [], []
  for sl in episodes((a_s < PEAK_DECEL_A) & (d["manual"] > 0), d["t"], d["route"], 0.5):
    w = slice(sl.start - n_in, min(sl.stop + n_tail, len(v)))
    if d["t"][sl.stop - 1] - d["t"][sl.start] < 2.0 or v[sl.start] < 4.0 or v[sl.start] - v[sl.stop - 1] < 2.0 or w.start < 0:
      continue
    vw, gw, lw = v[w], gap[w], v_lead[w]
    if (d["route"][w] != d["route"][sl.start]).any() or not (d["manual"][w] > 0).all() or not (d["lead"][w] > 0).all() or gw[n_in] > 80.0:
      continue
    if np.max(np.abs(np.diff(gw) - (lw[:-1] - vw[:-1]) * DT_MDL)) > 1.0:  # one radar track throughout
      continue
    if lw[n_in] < vw[n_in] - 0.5 or gw[-1] < 15.0:  # the lead's doing: slower at the onset, or close at the end
      jobs.append((vw, gw, lw, d["a"][w][0]))
      drv.append((vw[n_in], a_s[w][vw > 2.0].min(), gw.min()))
  return jobs, np.asarray(drv)


def _decel_peaks(args):
  """The driver's share of a slowing lead's accel by headway, then every manual decel a radar lead caused, flown from the
  same start behind the same logged lead, seeded with the calibration `--calibration`'s drive recorded: peak decel (aEgo
  over 0.5 s, above 2 m/s) and the minimum gap against the driver's; the hard brakes (`hard-brake`'s default
  calibration); engaged drives' press time."""
  from concurrent.futures import ProcessPoolExecutor
  from moonpilot.tests.test_engine_coast import _planner
  d = _pooled_manual(expand(args.routes, ROUTE_SETS), args.root)
  v, gap, v_lead = d["v"], d["gap"], d["v_lead"]
  a_s = np.concatenate([np.convolve(d["a"][d["route"] == r], np.ones(10) / 10, mode="same") for r in np.unique(d["route"])])
  near = (d["manual"] > 0) & (d["lead"] > 0) & (v > 5.0) & (np.abs(v - v_lead) < 1.5)
  print("Driver, lead slowing, |closing| < 1.5 m/s, above 5 m/s: median aEgo / aLead by headway (n)")
  for lo_a, hi_a in ((-0.6, -0.2), (-1.0, -0.6), (-1.6, -1.0)):
    cells = []
    for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)):
      m = near & (d["a_lead"] >= lo_a) & (d["a_lead"] < hi_a) & (gap >= lo * v) & (gap < hi * v)
      cells.append(f"{lo:.1f}-{hi:.1f} s {np.median(d['a'][m] / d['a_lead'][m]):5.2f} ({m.sum()})" if m.sum() >= 40 else f"{lo:.1f}-{hi:.1f} s   -")
    print(f"  aLead [{lo_a}, {hi_a}): " + "  ".join(cells))

  # The estimate on leads steady over +-3 s (centered slope under 0.1, speed within 0.6 m/s) and on a -0.3..-0.8 slowing
  # (centered +-1 s slope), raw and low-passed (`LeadAccelEstimator.a_slow`).
  ref1, ref3, spread = np.full(len(v), np.nan), np.full(len(v), np.nan), np.full(len(v), np.nan)
  cut = np.r_[True, (np.abs(np.diff(v_lead)) > 2.5) | (np.diff(d["route"]) != 0)]
  for r in runs((d["lead"] > 0) & ~cut):
    if r.stop - r.start > 121:
      vl = v_lead[r]
      ref1[r], ref3[r] = _centered_slope(vl, 20), _centered_slope(vl, 60)
      spread[r.start + 60:r.stop - 60] = np.ptp(np.lib.stride_tricks.sliding_window_view(vl, 121), axis=1)
  moving = (d["lead"] > 0) & (v_lead > 3.0)
  steady, slowing = moving & (np.abs(ref3) < 0.1) & (spread < 0.6), moving & (ref1 > -0.8) & (ref1 < -0.3)
  print(f"\nLead accel on {steady.sum() * DT_MDL / 60:.1f} min of leads steady over +-3 s, and {slowing.sum() * DT_MDL / 60:.1f} min slowing at -0.3..-0.8:")
  for name, x in (("estimate", d["a_lead"]), ("slow", d["a_slow"])):
    print(f"  {name:8s} steady: below -0.2 on {np.mean(x[steady] < -0.2) * 100:5.2f} %, below -0.3 on {np.mean(x[steady] < -0.3) * 100:5.2f} %, "
          + f"min {x[steady].min():5.2f}; slowing reads {np.median(x[slowing] / ref1[slowing]):.2f} of itself")

  jobs, drv = _decel_jobs(d, a_s)
  calibration = _calibration(args.calibration, args.root) if args.calibration != "none" else {}
  seeded = _planner(True, calibration)
  jobs = [(*job, calibration) for job in jobs]
  print(f"\n{len(jobs)} manual decels a radar lead caused, flown (engine-coast RAV4, {HOLD_FLY_DEAD_S} s dead time + {HOLD_FLY_TAU_S} s lag; "
        + f"calibration {args.calibration}: action_t {seeded.action_t:.3f} s, radar latency {seeded.radar_latency.applied():.3f} s). "
        + "Median peak decel and minimum gap; 'deeper' = planner peak past the driver's by more than 0.5:")
  with ProcessPoolExecutor(4) as pool:
    flown = list(pool.map(_fly_logged_lead, jobs))
  peak = np.asarray([np.convolve(f["a"], np.ones(10) / 10, mode="same")[f["v"] > 2.0].min() for f in flown])
  min_gap = np.asarray([f["gap"].min() for f in flown])
  print(f"  closest {min_gap.min():.1f} m")
  for lo, hi in ((4, 10), (10, 16), (16, 40)):
    b = (drv[:, 0] >= lo) & (drv[:, 0] < hi)
    print(f"    from {lo:2d}-{hi:2d} m/s (n {b.sum():2d}): peak driver {np.median(drv[b, 1]):5.2f} planner {np.median(peak[b]):5.2f}, "
          + f"deeper in {np.mean(peak[b] < drv[b, 1] - 0.5) * 100:3.0f}%; min gap driver {np.median(drv[b, 2]):4.1f} planner {np.median(min_gap[b]):4.1f} m")
  hard = [_hard_brake_run("fork+engine", v0, a, lag)[0] for lag in (False, True) for v0 in (20.0, 25.0, 30.0) for a in (-3.5, -5.0)]
  print("  hard-brake min gap, default calibration (20/25/30 m/s x -3.5/-5, then lagged): " + " ".join(f"{g:5.1f}" for g in hard))
  # Brake-band press edges (`BrakeBand.pressing` rising) flown through 30 s stretches of every manual follow behind one
  # radar track, the lead's real (correlated) estimate noise included, and those starting while the lead is steady
  # over +-3 s: what the law adds where nothing is slowing.
  win, follows = round(30.0 / DT_MDL), []
  for r in runs((d["manual"] > 0) & (d["lead"] > 0) & (v > 5.0) & ~cut):
    for s0 in range(r.start, r.stop - win + 1, win):
      w = slice(s0, s0 + win)
      if np.max(np.abs(np.diff(gap[w]) - (v_lead[w][:-1] - v[w][:-1]) * DT_MDL)) <= 1.0:
        follows.append(w)
  print(f"\nBrake-band press edges flown through {len(follows) * 30.0 / 60:.0f} min of manual follows (30 s stretches):")
  with ProcessPoolExecutor(4) as pool:
    flown = list(pool.map(_fly_logged_lead, [(v[w], gap[w], v_lead[w], d["a"][w][0], calibration) for w in follows]))
  edges = [np.flatnonzero(np.diff(np.asarray(f["pressing"], dtype=int)) > 0) + 1 for f in flown]
  quiet = sum(int(steady[w][e].sum()) for w, e in zip(follows, edges, strict=True))
  print(f"  {sum(len(e) for e in edges)} presses, {quiet} of them while the lead was steady")

  for route in expand(args.engaged, ROUTE_SETS):
    now_d = []
    now, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=now_d)
    m = active & np.isfinite(logged)
    presses = int(np.sum(m[1:] & (np.diff(np.asarray([r["band_pressing"] for r in now_d], dtype=int)) > 0)))
    print(f"{route}: {m.sum()} engaged frames; ask < -0.4 for {np.sum(m & (now < -0.4)) * DT_MDL:.1f} s; band presses {presses}; peak {now[m].min():.2f}")


def _engaged_fly(args):
  """An engaged window flown closed-loop: today's planner on the lagged plant from the car's state at `--start`, behind
  the logged lead's own path, fed the drive's model ask and set speed — against what the car did there. The model's ask
  is open-loop (what it asked from where the car really was), so a flight that strays far from the car reads it less."""
  details = []
  _, logged, _, _ = replay(args.route, extract(args.route, args.root), args.root, details_out=details)
  m = _details_arrays(details)
  t = m["t"] - m["t"][0]
  w = (t >= args.start) & (t < args.end)
  v_cruise = float(np.nanmedian(m["v_cruise"][w]))
  flown = _fly_logged_lead((m["v_ego"][w], m["d_rel"][w], m["v_lead"][w], float(m["a_ego"][w][0]), _calibration(args.route, args.root),
                            v_cruise, m["model_accel"][w]))

  def line(name, a, cmd, gap, v):
    s = np.sign(np.where(np.abs(cmd) < 0.05, 0.0, cmd))
    s = s[s != 0]
    return (f"  {name:8s} peak aEgo {np.convolve(a, np.ones(10) / 10, mode='same').min():+.2f}  command {cmd.min():+.2f}  "
            + f"sign flips {int(np.sum(s[1:] != s[:-1])):2d}  closest {gap.min():5.2f} m  ends {v[-1]:.2f} m/s at {gap[-1]:5.2f} m")

  print(f"{args.route} {args.start:.1f}-{args.end:.1f} s, set speed {v_cruise:.0f} kph:")
  print(line("car", m["a_ego"][w], logged[w], m["d_rel"][w], m["v_ego"][w]))
  print(line("planner", flown["a"], flown["cmd"], flown["gap"], flown["v"]))


# (name, plant dead time s, plant first-order lag s, the delay the planner projects or None for its learned one, in flight)
DELAY_ARMS = (
  ("instant car", DT_MDL, 1e-6, 0.0, False),
  ("car model (today)", HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, None, False),
  ("car model, not projected", HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, 0.0, False),
  ("car model, asks in flight", HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, None, True),
  ("slower car, 0.3 s dead", 0.3, HOLD_FLY_TAU_S, None, False),
  ("dead time 0.35 s", 0.35, 1e-6, None, False),
  ("dead time 0.35 s, in flight", 0.35, 1e-6, None, True),
)


def _delay_fly(job):
  """One `DELAY_ARMS` flight, engine coast off as the device runs: the peak ask, the peak aEgo (0.5 s mean, above 2 m/s),
  brake pumps and the closest gap."""
  from moonpilot.lab.driver import _pumps
  arm, flight = job
  _, dead, tau, delay, in_flight = DELAY_ARMS[arm]
  f = _fly_logged_lead(flight, plant=(dead, tau), engine_coast=False, delay=delay, in_flight=in_flight)
  felt = np.convolve(f["a"], np.ones(10) / 10, mode="same")
  return float(f["cmd"].min()), float(felt[f["v"] > 2.0].min()), _pumps(f["cmd"]), float(f["gap"].min())


def _delay_cost(args):
  """What the car's actuation delay costs at a stop's peak, and that projecting the ego better does not recover it: the
  57 lead-caused manual decels (`intent` (b)) and one engaged stop flown on an instant car, on the lab's car model with
  and without the planner's projection through its delay, with the commands still in flight projected instead of the
  latest one (`_in_flight_update`), on a car slower than the delay it projects, and on a pure dead time, where that
  in-flight projection is exact. The extra ask is each flight's peak against the same flight on the instant car."""
  from concurrent.futures import ProcessPoolExecutor
  d = _pooled_manual(expand(["manual-approach", "0000044e", "0000044f", "00000450", "00000451"], ROUTE_SETS), args.root)
  a_s = np.concatenate([np.convolve(d["a"][d["route"] == r], np.ones(10) / 10, mode="same") for r in np.unique(d["route"])])
  jobs, _ = _decel_jobs(d, a_s)
  calibration = _calibration(args.calibration, args.root)
  details = []
  replay(args.route, extract(args.route, args.root), args.root, details_out=details)
  m = _details_arrays(details)
  t = m["t"] - m["t"][0]
  w = (t >= args.start) & (t < args.end)
  stop = (m["v_ego"][w], m["d_rel"][w], m["v_lead"][w], float(m["a_ego"][w][0]), _calibration(args.route, args.root),
          float(np.nanmedian(m["v_cruise"][w])), m["model_accel"][w])
  arms = range(len(DELAY_ARMS))
  with ProcessPoolExecutor(4) as pool:
    flown = [list(pool.map(_delay_fly, [(arm, (*job, calibration)) for job in jobs])) for arm in arms]
    stops = list(pool.map(_delay_fly, [(arm, stop) for arm in arms]))
  print(f"{len(jobs)} manual lead-caused decels, calibration {args.calibration}: peak ask median / p10, peak aEgo median / p10, "
        + f"brake pumps, closest gap, extra peak ask over the instant car median / p90; then {args.route} {args.start:g}-{args.end:g} s")
  instant = np.asarray([row[0] for row in flown[0]])
  for (name, *_), rows, (s_ask, s_felt, _, s_gap) in zip(DELAY_ARMS, flown, stops, strict=True):
    ask, felt, pumps, gap = (np.asarray([row[i] for row in rows]) for i in range(4))
    extra = instant - ask
    print(f"  {name:27s} ask {np.median(ask):+.2f} / {np.percentile(ask, 10):+.2f}  aEgo {np.median(felt):+.2f} / {np.percentile(felt, 10):+.2f}  "
          + f"pumps {int(pumps.sum()):3d}  closest {gap.min():4.1f} m  extra {np.median(extra):.3f} / {np.percentile(extra, 90):.3f}  "
          + f"| stop: ask {s_ask:+.2f} aEgo {s_felt:+.2f} closest {s_gap:.2f} m")


# stop-shape's arms: (braking-match shares, the ease rate while a lead brakes); None keeps today's value, and an infinite ease
# rate is no ease limit at all.
SHAPE_ARMS: dict[str, tuple[list[float] | None, float | None]] = {
  "flat 0.85": ([0.85, 0.85, 0.85], math.inf),
  "scheduled": (None, math.inf),
  "today": (None, None),
  "ease 0.3": (None, 0.3),
}


@contextmanager
def _shape_arm(arm):
  """The planner's constants under one `SHAPE_ARMS` arm, restored on exit."""
  import moonpilot.longitudinal as longitudinal
  shares, ease = SHAPE_ARMS[arm]
  saved = longitudinal.MOONPILOT_TRACK_BRAKE_V, longitudinal.MOONPILOT_JERK_EASE_LEAD_BRAKING
  if shares is not None:
    longitudinal.MOONPILOT_TRACK_BRAKE_V = shares
  if ease is not None:
    longitudinal.MOONPILOT_JERK_EASE_LEAD_BRAKING = ease  # ty: ignore[invalid-assignment]  # the arm's rate, a float like the constant
  try:
    yield
  finally:
    longitudinal.MOONPILOT_TRACK_BRAKE_V, longitudinal.MOONPILOT_JERK_EASE_LEAD_BRAKING = saved


def _shape_fly(job):
  """One `stop-shape` flight under one `SHAPE_ARMS` arm, engine coast off as the device runs."""
  arm, flight = job
  with _shape_arm(arm):
    return _fly_logged_lead(flight, engine_coast=False)


def _shape(v, a, gap, cmd=None):
  """A decel's shape over its braking (aEgo over 0.5 s past -0.3): peak, where in the braking it lands, the first and last
  thirds' mean, closest gap, brake pumps."""
  from moonpilot.lab.driver import _pumps
  felt = np.convolve(a, np.ones(10) / 10, mode="same")
  braking = np.flatnonzero(felt < -0.3)
  seg = felt[braking[0]:braking[-1] + 1]
  n = len(seg)
  return (seg.min(), np.argmin(seg) / max(n - 1, 1), seg[:n // 3].mean(), seg[2 * n // 3:].mean(), gap.min(), _pumps(cmd) if cmd is not None else 0)


def _stop_shape(args):
  """Where a lead-caused decel's braking peaks: the 57 manual decels `decel-peaks` flies, split into stops (the driver
  ends at rest) and slowdowns, the driver against `SHAPE_ARMS` — the flat 0.85 braking match, the headway schedule alone,
  today's schedule with its ease limit while a lead brakes, and a slower ease; then one engaged stop flown with its own
  model ask (`engaged-fly`'s window)."""
  from concurrent.futures import ProcessPoolExecutor
  d = _pooled_manual(expand(args.routes, ROUTE_SETS), args.root)
  a_s = np.concatenate([np.convolve(d["a"][d["route"] == r], np.ones(10) / 10, mode="same") for r in np.unique(d["route"])])
  jobs, _ = _decel_jobs(d, a_s)
  calibration = _calibration(args.calibration, args.root)
  jobs = [(*job, calibration) for job in jobs]
  stops = np.asarray([job[0][-1] < 0.5 for job in jobs], dtype=bool)
  shapes = {"driver": [_shape(job[0], np.r_[np.diff(job[0]) / DT_MDL, 0.0], job[1]) for job in jobs]}
  with ProcessPoolExecutor(4) as pool:
    for name in SHAPE_ARMS:
      shapes[name] = [_shape(f["v"], f["a"], f["gap"], f["cmd"]) for f in pool.map(_shape_fly, [(name, job) for job in jobs])]
  print(f"{len(jobs)} manual lead-caused decels (calibration {args.calibration}): median peak, where in the braking it lands, "
        + "first / last third, closest gap, brake pumps")
  for kind, mask in (("stops", stops), ("slowdowns", ~stops)):
    print(f"  {kind} (n {mask.sum()}):")
    for name, rows in shapes.items():
      x = np.asarray(rows)[mask]
      print(f"    {name:10s} peak {np.median(x[:, 0]):+.2f} at {np.median(x[:, 1]):4.0%}  thirds {np.median(x[:, 2]):+.2f} / {np.median(x[:, 3]):+.2f}  "
            + f"closest {x[:, 4].min():4.1f} m  pumps {int(x[:, 5].sum()):3d}")

  # The gate before the counterfactual: the drive replayed under the arm it drove (`--route-match`), against the aTarget
  # it logged. Everything below is flown on the lab's plant; this is the only line about the car's own code.
  details = []
  with _shape_arm(args.route_match):
    replayed, logged, active, _ = replay(args.route, extract(args.route, args.root), args.root, details_out=details)
  m = _details_arrays(details)
  t = m["t"] - m["t"][0]
  w = (t >= args.start) & (t < args.end)
  commit, _ = drive_config(args.route, args.root)
  for name, mask in (("drive", active & np.isfinite(logged)), ("window", active & np.isfinite(logged) & w)):
    err = np.abs(replayed[mask] - logged[mask])
    print(f"{args.route} (commit {commit}, replayed as '{args.route_match}') replay-vs-logged |aTarget error| over the {name}, "
          + f"{mask.sum()} engaged frames: median {np.median(err):.4f} p95 {np.percentile(err, 95):.4f} max {err.max():.4f} m/s^2")
  flight = (m["v_ego"][w], m["d_rel"][w], m["v_lead"][w], float(m["a_ego"][w][0]), _calibration(args.route, args.root),
            float(np.nanmedian(m["v_cruise"][w])), m["model_accel"][w])
  print(f"{args.route} {args.start:g}-{args.end:g} s: peak, where, thirds, s to rest, rest gap, pumps — the car as recorded, "
        + "then every arm flown on the lab's plant (simulation against simulation)")
  for name, f in (("car (log)", {"v": m["v_ego"][w], "a": m["a_ego"][w], "gap": m["d_rel"][w], "cmd": logged[w]}),
                  *((arm, _shape_fly((arm, flight))) for arm in SHAPE_ARMS)):
    peak, where, first, last, _, pumps = _shape(f["v"], f["a"], f["gap"], f["cmd"])
    rest = np.flatnonzero(f["v"] < 0.3)
    print(f"  {name:10s} peak {peak:+.2f} at {where:4.0%}  thirds {first:+.2f} / {last:+.2f}  "
          + f"under 0.3 m/s at {rest[0] * DT_MDL if len(rest) else float('nan'):4.1f} s, {f['gap'][-1]:.1f} m  pumps {pumps}")


def _cue_frames(route, root=DEFAULT_ROOT):
  """leadOne, leadTwo and the driving model's stop intent, one row per planner tick of the route's extract, with leadOne's
  slow accel from the planner's own `LeadAccelEstimator` over the same rows (as `_manual_frames`)."""
  from moonpilot.lead import LeadAccelEstimator

  def build(_segs):
    data = extract(route, root)
    blob, offsets = data["blob"], data["offsets"]
    index = {k: 1 + i * 3 for i, k in enumerate(MESSAGES)}
    estimator, rows = LeadAccelEstimator(), []
    for row in offsets:
      def read(kind, row=row):
        start, size = row[index[kind]], row[index[kind] + 1]
        return messaging.log_from_bytes(blob[start:start + size].tobytes()) if size else None
      radar, model = read("radarState").radarState, read("modelV2")
      one, two = radar.leadOne, radar.leadTwo
      estimator.update(one)
      m = model.modelV2 if model is not None else None
      vel = m.velocity if m is not None and len(m.velocity.x) else None
      rows.append((row[0] * 1e-9, one.radarTrackId if one.present and one.radar else -1, one.dRel, one.vLead, estimator.a_slow,
                   two.present, two.dRel, two.vLead, bool(m.action.shouldStop) if m is not None else False,
                   float(m.action.desiredAcceleration) if m is not None else np.nan,
                   float(np.interp(6.0, list(vel.t), list(vel.x))) if vel is not None else np.nan, float(vel.x[-1]) if vel is not None else np.nan))
    keys = ("t", "track", "gap", "v_lead", "a_slow", "two", "gap_two", "v_two", "should_stop", "model_accel", "model_v6", "model_v10")
    return {k: np.asarray(col, dtype=float) for k, col in zip(keys, zip(*rows, strict=True), strict=True)}

  return cached(route, "cues", CUES_VERSION, root, build)


STOP_CUE_HOLD_S = 4.0  # s after a braking read ends in which the same track must reach rest, or stay visible to count as released


def _stop_cues(args):
  """Whether a braking lead ends at rest, from what the planner can see while it brakes: how long it has braked, its time to
  rest, the driving model's stop intent and leadTwo. One lead is, by `--identity`, a gap and speed that move continuously
  (at most 1 m and 1 m/s a tick) under any `radarTrackId` — the radar re-issues ids to one car — or one id with no vLead
  jump past 2.5 m/s; a braking read starting within 1 s of its lead's first frame is skipped, the slow accel being the
  previous lead's. Each read is stopped (the same lead under 0.5 m/s within `STOP_CUE_HOLD_S` of its end), released (the
  lead then stays visible that long without stopping) or censored (the lead or the drive ends first) — censored reads are
  counted and kept out of every share."""
  hold, settle = round(STOP_CUE_HOLD_S / DT_MDL), round(1.0 / DT_MDL)
  rows, censored = [], 0
  for route in dict.fromkeys(expand(args.routes, ROUTE_SETS)):  # the default sets overlap
    c = _cue_frames(route, args.root)
    track, v1, a_slow = c["track"], c["v_lead"], c["a_slow"]
    if args.identity == "track":
      same = np.r_[False, (track[1:] == track[:-1]) & (np.abs(np.diff(v1)) <= 2.5)]
    else:  # the radar re-issues ids to one car: the same lead is a gap and speed that move continuously, under any id
      same = np.r_[False, (track[1:] >= 0) & (track[:-1] >= 0) & (np.abs(np.diff(v1)) <= 1.0) & (np.abs(np.diff(c["gap"])) <= 1.0)]
    starts = np.flatnonzero((track >= 0) & ~same)
    for r in runs(track >= 0):
      bounds = [r.start, *(p for p in starts if r.start < p < r.stop), r.stop]  # a new track id, or a vLead jump, starts a new lead
      for s, e in zip(bounds[:-1], bounds[1:], strict=True):
        for b in runs(a_slow[s:e] < args.threshold):
          if b.start < settle:
            continue
          b0, b1 = s + b.start, s + b.stop
          if (v1[b0:min(b1 + hold, e)] < 0.5).any():
            outcome = 1
          elif b1 + hold <= e:
            outcome = 0
          else:
            censored += 1
            continue
          for into in (1.0, 2.0, 3.0, 4.0):
            k = b0 + round(into / DT_MDL)
            if k >= b1:
              break
            stopped_two = c["two"][k] > 0 and c["v_two"][k] < 1.0 and c["gap_two"][k] > c["gap"][k] + 2.0
            rows.append((outcome, (b1 - b0) * DT_MDL, into, v1[k] / -a_slow[k], c["model_v10"][k] < 1.0, c["model_v6"][k] < 1.0,
                         c["should_stop"][k] > 0, c["model_accel"][k] <= -1.0, stopped_two))
  x = np.asarray(rows, dtype=float)
  first = x[x[:, 2] == 1.0]
  stop = x[:, 0] > 0
  print(f"leadOne braking reads (slow accel < {args.threshold}) that last past 1 s, one lead each ({args.identity} identity): {len(first)} decided "
        + f"({np.mean(first[:, 0]) * 100:.0f}% stop), {censored} censored (all lengths) and left out")
  print("  braked tau s so far: n, share that stop")
  for tau in (1.0, 2.0, 3.0, 4.0, 6.0):
    m = first[:, 1] >= tau
    if m.sum() >= 10:
      print(f"    {tau:3.1f} s: n {m.sum():4d}  {np.mean(first[m, 0]) * 100:3.0f}%")
  print("  1 s in, by time to rest at the current slow accel: n, share that stop")
  for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, math.inf)):
    m = (first[:, 3] >= lo) & (first[:, 3] < hi)
    if m.any():
      print(f"    {lo:2d}-{hi:<3g} s: n {m.sum():4d}  {np.mean(first[m, 0]) * 100:3.0f}%")
  print("  gentle (time to rest > 3 s at that point): share that stop with / without each cue, and the stops it catches")
  for when in (1.0, 2.0, 3.0, 4.0):
    g = (x[:, 2] == when) & (x[:, 3] > 3.0)
    cells = []
    for name, col in (("model v10<1", 4), ("model v6<1", 5), ("shouldStop", 6), ("model<=-1", 7), ("leadTwo at rest", 8)):
      cue = x[:, col] > 0
      hit = cue & g
      share = f"{np.mean(stop[hit]) * 100:3.0f}%" if hit.any() else "  -"
      caught = f"{np.mean(cue[stop & g]) * 100:3.0f}%" if (stop & g).any() else "  -"
      cells.append(f"{name} n {int(hit.sum()):3d} {share} (else {np.mean(stop[~cue & g]) * 100:3.0f}%) catches {caught}")
    print(f"    {when:.0f} s in, n {int(g.sum()):3d} ({np.mean(stop[g]) * 100:3.0f}% stop): " + " | ".join(cells))


def _launch_episodes(d, windows_out=None):
  """Each manual standstill behind a stopped radar lead that then launches: (lead launch accel over its first 2 s, gap at
  rest, brake held when the lead moved, then brake release, gas and ego-moving times from the lead's onset, the ego's own
  accel over its first 2 s moving, the lead's own travel (integrated vLead) and its speed at the release).

  Every marker is read only inside the launch's window: from the onset to the first frame that is not the same drive, the
  same `radarTrackId`, timestamp-contiguous, a continuous gap and speed, a radar lead and manual. A marker seen there is
  its time; one not seen in a window that lasted the full 6 s is `inf`; one not seen in a window cut short is NaN,
  censored. The ego's accel needs its whole 2 s inside the window, and a launch whose window is shorter than the lead's
  own 2 s is dropped. `windows_out`, when given, gets each kept launch's (rest start, onset, window end) indices, row for row."""
  n2, n6, settle = round(2.0 / DT_MDL), round(6.0 / DT_MDL), round(1.0 / DT_MDL)
  v, v_lead, gap, brake, gas = d["v"], d["v_lead"], d["gap"], d["brake"] > 0, d["gas"] > 0
  lead = (d["manual"] > 0) & (d["lead"] > 0) & (d["lead_radar"] > 0)
  joined = np.r_[False, (np.abs(np.diff(gap)) <= 1.0) & (np.abs(np.diff(v_lead)) <= 1.0) & (np.diff(d["route"]) == 0)
                 & (np.diff(d["track_id"]) == 0) & (np.diff(d["t"]) <= 1.5 * DT_MDL)] & lead
  rows = []
  for r in runs(lead & (v < 0.1)):
    fast = np.flatnonzero(v_lead[r.start:min(r.stop + n6, len(v))] > 1.0)
    if not len(fast):
      continue
    k1 = r.start + fast[0]
    slow = np.flatnonzero(v_lead[r.start:k1] < 0.2)
    if not len(slow):
      continue
    t0 = r.start + slow[-1]  # the lead's onset: its last frame under 0.2 m/s before it passes 1 m/s (0.2/a after it starts)
    # the lead judged stopped, and the gap at rest, only before it launched and only on the lead that launches: the rest
    # starts at the last break (a new track or drive, a gap) before the onset, so an old lead's wait is not credited to it
    breaks = np.flatnonzero(~joined[r.start + 1:t0 + 1])
    rest = slice(r.start + 1 + breaks[-1] if len(breaks) else r.start, t0)
    if t0 - rest.start < settle or t0 >= r.stop or np.median(v_lead[rest]) > 0.3 or np.median(gap[rest]) > 20.0:
      continue
    end = t0 + 1
    while end < min(t0 + n6 + n2, len(v)) and joined[end]:
      end += 1
    if end - t0 <= n2:
      continue
    full = end - t0 >= n6

    def first(mask, t0=t0, end=end, full=full):
      hit = np.flatnonzero(mask[t0:min(end, t0 + n6)])
      return hit[0] * DT_MDL if len(hit) else (math.inf if full else math.nan)

    moving = first(v > 0.3)
    held = bool(brake[t0])
    release = first(~brake) if held else math.nan  # a brake already off when the lead moved is not a reaction
    k_rel = t0 + round(release / DT_MDL) if math.isfinite(release) else None
    k_move = t0 + round(moving / DT_MDL) if math.isfinite(moving) else None
    ego_a = (v[k_move + n2] - v[k_move]) / 2.0 if k_move is not None and k_move + n2 < end else math.nan
    lead_travel = float(np.sum(v_lead[t0:k_rel]) * DT_MDL) if k_rel is not None else math.nan  # its own, not the gap: the ego creeps
    v_rel = v_lead[k_rel] if k_rel is not None else math.nan
    rows.append(((v_lead[t0 + n2] - v_lead[t0]) / 2.0, float(np.median(gap[rest])), held, release, first(gas), moving, ego_a, lead_travel, v_rel))
    if windows_out is not None:
      windows_out.append((rest.start, t0, end))
  return np.asarray(rows, dtype=float).reshape(-1, 9)


def _launch_frames(args):
  """The driver extract pooled over `--routes` (every drive with rlogs by default), with `route` numbering the drives."""
  from moonpilot.lab import driver
  routes = expand(args.routes, ROUTE_SETS) if args.routes else driver._routes(args.root)
  frames = [driver.extract(r, args.root) for r in routes]
  keys = ("t", "manual", "v", "a", "brake", "gas", "lead", "lead_radar", "gap", "v_lead", "track_id", "desired_a")
  d = {k: np.concatenate([f[k] for f in frames]) for k in keys}
  d["route"] = np.concatenate([np.full(len(f["t"]), i) for i, f in enumerate(frames)])
  return routes, d


LAUNCH_FLY_PRE_S, LAUNCH_FLY_POST_S, LAUNCH_FLY_REST_GAP = 2.0, 8.0, 4.3
LAUNCH_ARMS = {"as flown": (None, None), f"own rest gap {LAUNCH_FLY_REST_GAP}": (LAUNCH_FLY_REST_GAP, None),
               f"{LAUNCH_FLY_REST_GAP}, reserve 0": (LAUNCH_FLY_REST_GAP, 0.0)}


def _launch_fly_one(job):
  """One `launch-fly` flight: today's planner from rest behind the logged lead (engine coast off, the drive's model ask),
  under one `LAUNCH_ARMS` arm — the rest gap moved to the planner's own (the same lead path, the ego resting further back),
  and the moving-lead reserve changed."""
  import moonpilot.longitudinal as longitudinal
  arm, (v_log, gap, v_lead, model, calibration) = job
  rest_gap, reserve = LAUNCH_ARMS[arm]
  n_pre = round(LAUNCH_FLY_PRE_S / DT_MDL)
  if rest_gap is not None:
    gap = gap + (rest_gap - float(np.mean(gap[:n_pre])))
  saved = longitudinal.MOONPILOT_MOVING_STOP_RESERVE
  if reserve is not None:
    longitudinal.MOONPILOT_MOVING_STOP_RESERVE = reserve  # ty: ignore[invalid-assignment]  # the arm's reserve, a float like the constant
  try:
    f = _fly_logged_lead((v_log, gap, v_lead, 0.0, calibration, 50.0, model), engine_coast=False)
  finally:
    longitudinal.MOONPILOT_MOVING_STOP_RESERVE = saved
  return {k: f[k][n_pre:] for k in ("cmd", "a", "v", "gap")}


def _launch_fly(args):
  """Today's planner flown from rest behind the leads of `launch-reaction`'s launches, the same selection, kept only where
  that launch's own window — one drive, one radar track, contiguous — covers the whole flight (`LAUNCH_FLY_PRE_S` before
  the onset to `LAUNCH_FLY_POST_S` after). The ego's logged path is the lead's frame; the planner flies its own on the lab's
  plant (`HOLD_FLY_DEAD_S` dead time, `HOLD_FLY_TAU_S` lag) at a 50 kph set speed with the drive's own model ask. The car's
  own resume out of standstill is not simulated: command-path evidence, against the driver's recorded launch."""
  from concurrent.futures import ProcessPoolExecutor
  routes, d = _launch_frames(args)
  windows = []
  x = _launch_episodes(d, windows)
  n_pre, n_post = round(LAUNCH_FLY_PRE_S / DT_MDL), round(LAUNCH_FLY_POST_S / DT_MDL)
  calibration = _calibration(args.calibration, args.root)
  keep = [i for i, (rest, t0, end) in enumerate(windows) if rest <= t0 - n_pre and end >= t0 + n_post]
  jobs = []
  for i in keep:
    w = slice(windows[i][1] - n_pre, windows[i][1] + n_post)
    jobs.append((d["v"][w], d["gap"][w], d["v_lead"][w], np.nan_to_num(d["desired_a"][w], nan=0.0), calibration))
  driver_a = [d["a"][windows[i][1]:windows[i][1] + n_post] for i in keep]
  driver_gap = [float(np.median(d["gap"][windows[i][1] + n_post - round(2.0 / DT_MDL):windows[i][1] + n_post])) for i in keep]
  driver_v = [float(d["v"][windows[i][1] + n_post - 1]) for i in keep]
  print(f"{len(routes)} drives, {len(x)} launches, {len(keep)} whose own window covers the flight; the driver releases after "
        + f"{np.nanmedian(x[keep, 7]):.2f} m of lead travel and is moving at {np.nanmedian(x[keep, 5][np.isfinite(x[keep, 5])]):.2f} s")
  times = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
  with ProcessPoolExecutor(4) as pool:
    for arm in LAUNCH_ARMS:
      flown = list(pool.map(_launch_fly_one, [(arm, job) for job in jobs]))
      ask, mv, travel = [], [], []
      for f, job in zip(flown, jobs, strict=True):
        pos, moving = np.flatnonzero(f["cmd"] > 0.05), np.flatnonzero(f["v"] > 0.3)
        ask.append(pos[0] * DT_MDL if len(pos) else math.inf)
        mv.append(moving[0] * DT_MDL if len(moving) else math.inf)
        travel.append(float(np.sum(job[2][n_pre:n_pre + pos[0]]) * DT_MDL) if len(pos) else math.nan)
      ask, mv = np.asarray(ask), np.asarray(mv)
      print(f"  {arm:18s} ask > 0.05 at {np.median(ask[np.isfinite(ask)]):4.2f} s after {np.nanmedian(travel):4.2f} m of lead travel, "
            + f"moving {np.median(mv[np.isfinite(mv)]):4.2f} s (never {int(np.sum(mv == math.inf))}), "
            + f"gap {LAUNCH_FLY_POST_S - 2:g}-{LAUNCH_FLY_POST_S:g} s in {np.median([np.median(f['gap'][-round(2.0 / DT_MDL):]) for f in flown]):4.1f} m")
      if arm == f"own rest gap {LAUNCH_FLY_REST_GAP}":
        traced = flown
  print("  s after onset             " + " ".join(f"{s:5.1f}" for s in times))
  for name, series in (("planner ask (own rest)", [f["cmd"] for f in traced]), ("plant aEgo (own rest)", [f["a"] for f in traced]),
                       ("model desired", [job[3][n_pre:] for job in jobs]), ("driver aEgo", driver_a)):
    print(f"  {name:24s}  " + " ".join(f"{np.median([s[round(t / DT_MDL)] for s in series]):+5.2f}" for t in times))
  print(f"  driver gap {LAUNCH_FLY_POST_S - 2:g}-{LAUNCH_FLY_POST_S:g} s in {np.median(driver_gap):.1f} m at {np.median(driver_v):.1f} m/s")


def _launch_reaction(args):
  """The driver's reaction from a standstill behind a stopped radar lead when it launches, against how hard it launches:
  from the lead's onset (its last frame under 0.2 m/s before it passes 1 m/s), how long to the brake release, the gas and
  the ego moving. Pedal booleans at the driver extract's 20 Hz: switch timing, not pressure, and a switch can lag the pedal
  (this car leaves `carState.brake`/`gas` unpopulated). Every drive with rlogs under --root by default; one lead is one
  `radarTrackId` whose gap and speed move continuously through the launch."""
  routes, d = _launch_frames(args)
  x = _launch_episodes(d)
  print(f"{len(routes)} drives")
  a_lead, gap, held, release, gas, moving, ego_a, opened, v_rel = x.T  # opened: the lead's own travel to the release
  held = held > 0
  print(f"{len(x)} manual launches behind a stopped radar lead (median gap at rest {np.median(gap):.1f} m); brake held at the lead's "
        + f"onset on {held.sum()} ({int((held & (release == math.inf)).sum())} still held 6 s later, {int((held & np.isnan(release)).sum())} "
        + f"censored before a release), already released on {(~held).sum()} (no reaction to time)")

  def cell(y):
    y = y[np.isfinite(y)]
    return f"{np.median(y):4.2f} ({np.percentile(y, 25):4.2f}-{np.percentile(y, 75):4.2f}) n {len(y):3d}" if len(y) >= 3 else f"{'-':>21s}"

  print(f"  {'lead launch, m/s^2':20s} {'brake release s':>24s} {'gas s':>24s} {'ego moving s':>24s}  {'lead travel to release m':>24s}"
        + f"  {'lead speed at release':>24s}  ego accel, first 2 s")
  for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, math.inf)):
    m = (a_lead >= lo) & (a_lead < hi)
    print(f"  {lo:3.1f}-{hi:<3g} (n {m.sum():3d})       {cell(release[m & held]):>24s} {cell(gas[m & held]):>24s} {cell(moving[m & held]):>24s}  "
          + f"{cell(opened[m & held]):>24s}  {cell(v_rel[m & held]):>24s}  " + (f"{np.nanmedian(ego_a[m & held]):+.2f}" if (m & held).sum() >= 3 else "-"))

  def spearman(a, b):
    """`driver._corr`'s tie-averaged Spearman (NaN under 30 pairs or a constant side), with the pairs it saw."""
    from moonpilot.lab.driver import _corr
    return _corr(a, b), int((np.isfinite(a) & np.isfinite(b)).sum())

  for name, y in (("brake release", release), ("gas", gas), ("ego moving", moving), ("ego accel", ego_a), ("lead travel", opened)):
    rho, n = spearman(a_lead[held], y[held])
    print(f"  Spearman rho, lead launch accel vs {name:13s}: {rho:+.2f} (n {n})")


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
    # a measured burst with no logged-command burst start within BURST_T: unmatched, not a cause — a smaller or slower
    # command ramp, a response delayed past the window, or aEgo noise reads the same
    car_only += sum(not any(abs(c - k) <= n for k in commanded) for c in _burst_starts(car, mask))
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


def _register_manual(sub):
  parser = sub.add_parser("manual-decel", help="the lead law's approach requirement against the driver's own longitudinal")
  parser.add_argument("--routes", nargs="+", default=["manual-approach"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--t-follow", type=float, default=1.45, help="the target's time gap, s (standard personality's 1.45)")
  parser.set_defaults(fn=_manual_decel)
  parser = sub.add_parser("approach-onsets", help="where approaches to a slower lead start slowing: driver vs the lead law, logged vs now")
  parser.add_argument("--routes", nargs="+", default=["manual-approach"], help=f"manual route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--engaged", nargs="+", default=[], help="engaged drives to replay: logged command against today's planner")
  parser.add_argument("--t-follow", type=float, default=1.45, help="the target's time gap, s")
  parser.add_argument("--lag", type=float, default=0.36, help="command to aEgo lag, s (MoonpilotLongLag on this device)")
  parser.set_defaults(fn=_approach_onsets)
  sub.add_parser("hard-brake", help="lead braking hard from a follow: upstream vs fork vs ideal, with and without lag").set_defaults(fn=_hard_brake)
  parser = sub.add_parser("sustain", help="the lead-brake sustain horizon: fixed vs scaled, hard brakes, taps, drives")
  parser.add_argument("--routes", nargs="+", default=["longitudinal"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--part", choices=("all", "flights", "drives"), default="all", help="flown scenarios, replayed drives or both")
  parser.add_argument("--rules", nargs="+", choices=tuple(SUSTAIN_RULES), default=list(SUSTAIN_RULES), help="horizons beside the fixed 2.0 s")
  parser.set_defaults(fn=_sustain)
  parser = sub.add_parser("decel-peaks", help="manual lead-caused decels flown: peaks, gaps, hard brakes")
  parser.add_argument("--routes", nargs="+", default=["manual-approach", "0000044e", "0000044f", "00000450", "00000451"],
                      help=f"manual route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--engaged", nargs="+", default=[], help="engaged drives to replay")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights, or none")
  parser.set_defaults(fn=_decel_peaks)
  parser = sub.add_parser("brake-gain", help="how hard the car brakes for what the planner asks, over every engaged drive")
  parser.add_argument("--routes", nargs="+", default=["engaged"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_brake_gain)
  parser = sub.add_parser("engaged-fly", help="an engaged window flown closed-loop through today's planner, with the drive's model ask")
  parser.add_argument("route", help="full route name, e.g. 0000045a--8c1d34e64d")
  parser.add_argument("--start", type=float, required=True, help="s from the drive's first plan")
  parser.add_argument("--end", type=float, required=True, help="s from the drive's first plan")
  parser.set_defaults(fn=_engaged_fly)
  parser = sub.add_parser("delay-cost", help="what the actuation delay costs at a stop's peak, and the in-flight projection that did not recover it")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the decel flights")
  parser.add_argument("--route", default="0000045a--8c1d34e64d", help="engaged drive whose stop is flown with its own model ask")
  parser.add_argument("--start", type=float, default=183.0, help="s from the drive's first plan")
  parser.add_argument("--end", type=float, default=216.0, help="s from the drive's first plan")
  parser.set_defaults(fn=_delay_cost)
  parser = sub.add_parser("stop-shape", help="where a lead-caused decel's braking peaks: driver vs the braking match, flat and headway-scheduled")
  parser.add_argument("--routes", nargs="+", default=["manual-approach", "0000044e", "0000044f", "00000450", "00000451"],
                      help=f"manual route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  parser.add_argument("--route", default="00000468--34a8e259a0", help="engaged drive whose stop is flown with its own model ask")
  parser.add_argument("--start", type=float, default=597.0, help="s from the drive's first plan")
  parser.add_argument("--end", type=float, default=623.0, help="s from the drive's first plan")
  parser.add_argument("--route-match", choices=tuple(SHAPE_ARMS), default="flat 0.85",
                      help="the arm the route drove: 'flat 0.85' before a48c66e63, 'scheduled' to the ease limit, 'today' after")
  parser.set_defaults(fn=_stop_shape)
  parser = sub.add_parser("stop-cues", help="whether a braking lead ends at rest: braking time, time to rest, model stop intent, leadTwo")
  parser.add_argument("--routes", nargs="+", default=["manual-approach", "0000044e", "0000044f", "00000450", "00000451", "engaged", "00000468"],
                      help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--threshold", type=float, default=-0.3, help="slow accel that reads as braking, m/s^2")
  parser.add_argument("--identity", choices=("track", "kinematic"), default="kinematic",
                      help="one lead = one radarTrackId, or a gap and speed that move continuously under any id")
  parser.set_defaults(fn=_stop_cues)
  parser = sub.add_parser("launch-reaction", help="the driver's brake release, gas and move times behind a launching lead, by its launch accel")
  parser.add_argument("--routes", nargs="*", default=[], help=f"route names or sets: {', '.join(ROUTE_SETS)}; default every drive with rlogs")
  parser.set_defaults(fn=_launch_reaction)
  parser = sub.add_parser("launch-fly", help="today's planner flown from rest behind launch-reaction's launching leads, against the driver")
  parser.add_argument("--routes", nargs="*", default=[], help=f"route names or sets: {', '.join(ROUTE_SETS)}; default every drive with rlogs")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  parser.set_defaults(fn=_launch_fly)
  parser = sub.add_parser("accel-bursts", help="engaged positive accel bursts and what they come from; far leads ignored as an arm")
  parser.add_argument("--routes", nargs="+", default=["engaged", "00000468"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--far-headway", type=float, default=4.0, help="s of headway past which a lead counts as far (and is ignored in the arm)")
  parser.set_defaults(fn=_accel_bursts)


def register(sub):
  _register_lead_approach(sub)
  _register_manual(sub)
  commands = (
    ("planner-fidelity", _fidelity),
    ("brake-band", _band),
    ("long-smoothness", _smooth),
    ("long-events", _events),
    ("lead-ages", _lead_ages),
    ("brake-lamps", _lamps),
    ("curve-metrics", _curve_metrics),
    ("queue-stop", _queue_stop),
    ("crawl-stop", _crawl_stop),
    ("rolling-exits", _rolling_exits),
    ("longcontrol", _longcontrol),
  )
  for name, fn in commands:
    parser = sub.add_parser(name, help=fn.__name__)
    parser.add_argument("--routes", nargs="+", default=["longitudinal"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
    parser.set_defaults(fn=fn)
