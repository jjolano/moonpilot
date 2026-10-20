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
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.tools.lib.logreader import LogReader
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.longcontrol import MoonpilotLongControl
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, runs, segments

EXTRACT_VERSION = 2
LAMP_VERSION = 1
LAMP_ADDRESS = 0x320  # VSC1S07: the skid-control ECU's own frame, where BRKON is the brake-lamp proxy
CONTROL_VERSION = 1  # one row per carControl frame: the rate the acceleration controller is written for
MANUAL_VERSION = 1
ROUTE_SETS = {
  # Engaged local drives. 0000042c/42d/42e have zero longActive frames and are skipped by the commands.
  "longitudinal": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  "lead-approach": ("000003f8", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  # The driver's own longitudinal behind radar leads; routes without such frames are counted as empty.
  "manual-approach": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414",
                      "0000042b", "0000042c", "0000043e", "0000043f", "0000044d"),
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
  for msg in LogReader(segs, sort_by_time=True):
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
  from unittest import mock
  from moonpilot.tests.test_longitudinal import _fly
  import moonpilot.longitudinal as longitudinal

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
  label, v0, setpoint, gap, v_lead, accel = scenarios[0]
  with mock.patch.object(longitudinal, "MOONPILOT_MOVING_STOP_RESERVE", 0.0), \
       mock.patch.object(longitudinal, "MOONPILOT_ARRIVAL_COAST_T", float("inf")):
    former = _fly(v0, setpoint, gap, v_lead, accel, 55.0)
  print("Reserve and coast-first off in current code: arrival then aims at STOP_DISTANCE, not the pre-change STOP_REST,")
  print("behind a moving lead, so this is near the pre-change law, not identical; check out the pre-change commit for that.")
  report("steady lead, reserve and coast off", former)


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
  parser = sub.add_parser("lead-approach", help="manual lead windows and full-planner coast-first diagnostic")
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
      rows.append((row[0] * 1e-9, manual, cs.vEgo, cs.aEgo, cs.gasPressed, cs.brakePressed, lead.present and lead.radar, lead.dRel, lead.vLead, a_lead))
    keys = ("t", "manual", "v", "a", "gas", "brake", "lead", "gap", "v_lead", "a_lead")
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
HOLD_FLY_SCENARIOS = (  # (label, v0, set kph, gap0, v_lead0, lead accel (t, v_lead), seconds)
  ("20 -> 15 m/s from 119 m", 20.0, 72.0, 119.0, 15.0, lambda t, vl: 0.0, 80.0),
  ("same, lead +0.18 for 10 s", 20.0, 72.0, 119.0, 15.0, lambda t, vl: 0.18 if t < 10.0 else 0.0, 80.0),
  ("000003e4: 16 -> 11.7 from 119 m", 16.0, 58.0, 119.0, 11.7, lambda t, vl: 0.0, 80.0),
  ("lead -1.5 to 5 m/s at 6 s", 20.0, 72.0, 119.0, 15.0, lambda t, vl: -1.5 if t > 6.0 and vl > 5.0 else 0.0, 60.0),
  ("lead -3.5 to rest at 6 s", 20.0, 72.0, 119.0, 15.0, lambda t, vl: -3.5 if t > 6.0 else 0.0, 60.0),
  ("lead -3.5 to rest at 12 s", 20.0, 72.0, 119.0, 15.0, lambda t, vl: -3.5 if t > 12.0 else 0.0, 60.0),
)


class _LaggedPlant:
  """Acceleration following the command after `HOLD_FLY_DEAD_S` of dead time, then a first-order lag of
  `HOLD_FLY_TAU_S`, exactly discretized at the tick: the command fed on frame i first moves `a` on frame i + dead/dt."""

  def __init__(self, dt=DT_MDL):
    self.queue = deque([0.0] * round(HOLD_FLY_DEAD_S / dt))
    self.alpha = 1.0 - math.exp(-dt / HOLD_FLY_TAU_S)
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


def _fly_metrics(r):
  """First ask past -0.05 (gap there), peak, minimum gap, the command's noise about its 1 s mean, and how many frames
  after that first ask the command came back above -0.01 while still closing faster than 0.5 m/s — a return to the hold."""
  cmd, gap = r["cmd"], r["gap"]
  onset = np.flatnonzero(cmd < -0.05)
  closing = r["v"] - r["v_lead"]
  back = int(np.sum((cmd[onset[0]:] > -0.01) & (closing[onset[0]:] > 0.5))) if onset.size else 0
  hf = float(np.std((cmd - np.convolve(cmd, np.ones(20) / 20, mode="same"))[10:-10]))
  return {"onset": float(gap[onset[0]]) if onset.size else float("nan"), "peak": float(cmd.min()), "min_gap": float(gap.min()), "hf": hf, "back": back}


def _lead_hold(args):
  """The engine-coast arrival hold (`MOONPILOT_ARRIVAL_HOLD_*`): the driver data it is shaped on, where it changes a
  recorded drive's command, and lagged, noisy closed-loop flights with it on and off."""
  from unittest import mock
  import moonpilot.longitudinal as longitudinal
  off = mock.patch.multiple(longitudinal, MOONPILOT_ARRIVAL_HOLD_START=0.0, MOONPILOT_ARRIVAL_HOLD_FULL=0.0)

  routes = expand(args.routes, ROUTE_SETS)
  d = _pooled_manual(routes, args.root)
  closing, slack = d["v"] - d["v_lead"], d["gap"] - 1.45 * d["v_lead"]
  mask = (d["manual"] > 0) & (d["lead"] > 0) & (d["v"] > 12.0) & (closing > 1.0) & (slack > 5.0)
  need = -(closing**2) / (2.0 * np.maximum(slack, 1e-3))
  print(f"Driver: {mask.sum() * DT_MDL / 60:.1f} min of manual approaches above 12 m/s on {len(set(d['route'][mask]))} of {len(routes)} routes "
        + "(radar lead slower by > 1 m/s, gap > 5 m past 1.45 s); the arrival term's ask, binned:")
  _bin_rows(d, mask, need, (-0.5, -0.33, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0))

  route = args.replay_route
  data = extract(route, args.root)
  details = []
  on, logged, active, _ = replay(route, data, args.root, details_out=details)
  with off:
    held_off, _, _, _ = replay(route, data, args.root)
  t0 = next(iter(LogReader(segments(route, args.root)[0], sort_by_time=True))).logMonoTime * 1e-9
  t = np.asarray([row["t"] for row in details]) - t0
  engaged = active & np.isfinite(logged)
  print(f"\nReplay {route} (commit {drive_config(route, args.root)[0][:9]}): hold off reproduces the log to p95 "
        + f"{np.percentile(np.abs(held_off - logged)[engaged], 95):.4f} m/s^2 on {engaged.sum()} engaged frames. Where the hold changes the ask:")
  for s in runs(engaged & (np.abs(on - held_off) > 0.005)):
    print(f"  {t[s.start]:7.1f}-{t[s.stop - 1]:7.1f} s  logged {logged[s].min():6.3f}..{logged[s].max():6.3f}  "
          + f"hold off {held_off[s].min():6.3f}..{held_off[s].max():6.3f}  hold on {on[s].min():6.3f}..{on[s].max():6.3f}")

  print(f"\nFlown: engine-coast RAV4, {HOLD_FLY_DEAD_S} s dead time + {HOLD_FLY_TAU_S} s lag, clean and {args.seeds} noisy seeds "
        + "(worst over seeds after the slash). onset = gap at the first ask past -0.05; back = frames returned to the hold.")
  print(f"  {'scenario':32s} {'hold':4s} {'onset':>6s} {'peak':>13s} {'min gap':>13s} {'hf':>5s} {'back':>4s}")
  for label, *scenario in HOLD_FLY_SCENARIOS:
    for name, ctx in (("off", off), ("on", nullcontext())):
      with ctx:
        clean = _fly_metrics(_fly_lagged(*scenario))
        noisy = [_fly_metrics(_fly_lagged(*scenario, seed=s)) for s in range(args.seeds)] or [clean]
      print(f"  {label:32s} {name:4s} {clean['onset']:6.1f} {clean['peak']:6.2f}/{min(n['peak'] for n in noisy):6.2f} "
            + f"{clean['min_gap']:6.2f}/{min(n['min_gap'] for n in noisy):6.2f} {max(n['hf'] for n in noisy):5.2f} "
            + f"{max([clean['back']] + [n['back'] for n in noisy]):4d}")


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
  """The unified approach requirement against the driver's own longitudinal: `stopping_decel` (the lead's braking held
  for up to its sustain, positive lead accel ignored) to the arrival target, max(moving rest, t_follow * v_lead), on
  manual frames behind a radar lead; then where the driver actually ends up — rest gaps and settled headways."""
  from moonpilot.longitudinal import (MOONPILOT_CREEP_SPEED, MOONPILOT_MOVING_STOP_RESERVE, MOONPILOT_STOP_DISTANCE, MOONPILOT_STOP_REST,
                                      lead_accel_estimate, stopping_decel)
  t_follow = args.t_follow
  routes = expand(args.routes, ROUTE_SETS)
  d = _pooled_manual(routes, args.root)
  v, v_lead, gap = d["v"], d["v_lead"], d["gap"]
  reserve = MOONPILOT_STOP_DISTANCE - MOONPILOT_STOP_REST + MOONPILOT_MOVING_STOP_RESERVE
  rest = MOONPILOT_STOP_REST + reserve * np.clip(v_lead / MOONPILOT_CREEP_SPEED, 0.0, 1.0)
  target = np.maximum(rest, t_follow * v_lead)
  lead = (d["manual"] > 0) & (d["lead"] > 0)
  a_lead = np.array([lead_accel_estimate(x) for x in d["a_lead"]])
  need = np.array([stopping_decel(*x) if m else 0.0 for m, x in zip(lead, zip(v, v_lead, a_lead, np.maximum(gap - target, 0.1), strict=True), strict=True)])
  approach = lead & (v > 2.0) & (gap - target > 1.0) & (need < 0.0)
  # A lead that leaves within 5 s (radar lead gone, or its gap jumping 5 m in a frame: another object) is one the
  # driver may already be passing or watching turn off; the ratios below are on the leads that stay.
  leaves = np.zeros(len(v), dtype=bool)
  for i in np.flatnonzero(approach & (need <= -0.2)):
    ahead = slice(i, min(i + round(5.0 / DT_MDL), len(v)))
    same = d["route"][ahead] == d["route"][i]
    leaves[i] = not (d["lead"][ahead][same] > 0).all() or bool((np.abs(np.diff(gap[ahead][same])) > 5.0).any())
  to_target = (gap - target) / np.maximum(v - v_lead, 0.05)
  print(f"{approach.sum() * DT_MDL / 60:.1f} min of manual approach frames on {len(set(d['route'][approach]))} of {len(routes)} routes; "
        + "the unified requirement, binned, against what the driver did:")
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


def _register_manual(sub):
  parser = sub.add_parser("lead-hold", help="engine-coast arrival hold: driver bins, replayed drive, lagged/noisy flights")
  parser.add_argument("--routes", nargs="+", default=["manual-approach"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--replay-route", default="0000044d", help="engaged drive to replay with the hold on and off")
  parser.add_argument("--seeds", type=int, default=5, help="noisy flights per scenario")
  parser.set_defaults(fn=_lead_hold)
  parser = sub.add_parser("manual-decel", help="the unified approach requirement against the driver's own longitudinal")
  parser.add_argument("--routes", nargs="+", default=["manual-approach"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--t-follow", type=float, default=1.45, help="the target's time gap, s (standard personality's 1.45)")
  parser.set_defaults(fn=_manual_decel)
  sub.add_parser("hard-brake", help="lead braking hard from a follow: upstream vs fork vs ideal, with and without lag").set_defaults(fn=_hard_brake)
  parser = sub.add_parser("sustain", help="the lead-brake sustain horizon: fixed vs scaled, hard brakes, taps, drives")
  parser.add_argument("--routes", nargs="+", default=["longitudinal"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.add_argument("--part", choices=("all", "flights", "drives"), default="all", help="flown scenarios, replayed drives or both")
  parser.add_argument("--rules", nargs="+", choices=tuple(SUSTAIN_RULES), default=list(SUSTAIN_RULES), help="horizons beside the fixed 2.0 s")
  parser.set_defaults(fn=_sustain)


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
