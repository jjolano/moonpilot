"""Recorded-drive replay of the fork longitudinal planner.

Every route is replayed in the configuration its own initData recorded, one planner tick per logged `modelV2`, with
each message's real `logMonoTime` and the module's wall clock shimmed to the row's time — so the planner's own age
math (lead age, curve-path age, the pose's staleness check) sees what it saw on the drive. Two ceilings are named
rather than hidden: window overrides the drive did not carry (a marketplace model's smoothing, a car-side constant)
and the car's own brake arbitration, which no replay models — `brake-lamps` reports the drive's real BRKON, not what
the band would have produced.
"""
from types import SimpleNamespace

import numpy as np
from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from openpilot.common.constants import CV
from openpilot.tools.lib.logreader import LogReader
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.longcontrol import MoonpilotLongControl
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, runs

EXTRACT_VERSION = 2
LAMP_VERSION = 1
LAMP_ADDRESS = 0x320  # VSC1S07: the skid-control ECU's own frame, where BRKON is the brake-lamp proxy
ROUTE_SETS = {
  # Engaged local drives. 0000042c/42d/42e have zero longActive frames and are skipped by the commands.
  "longitudinal": ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
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


def _longcontrol(args):
  for route in expand(args.routes, ROUTE_SETS):
    commit, _ = drive_config(route, args.root)
    details = []
    driven, logged, active, _ = replay(route, extract(route, args.root), args.root, details_out=details)
    if _skip_no_engaged(route, active):
      continue
    d = _details_arrays(details)
    cp = messaging.log_from_bytes(extract(route, args.root)["cp"].tobytes(), car.CarParams)
    ci = interfaces[cp.carFingerprint](cp)
    loc = MoonpilotLongControl(cp)
    out = np.zeros(len(driven))
    for i in range(len(out)):
      cs = SimpleNamespace(vEgo=float(d["v_ego"][i]), aEgo=float(d["a_ego"][i]), vCruise=float(d["v_cruise"][i]),
                           brakePressed=bool(d["brake_pressed"][i]), gasPressed=bool(d["gas_pressed"][i]),
                           cruiseState=SimpleNamespace(standstill=bool(d["cruise_standstill"][i])))
      limits = ci.get_pid_accel_limits(cp, cs.vEgo, cs.vCruise * CV.KPH_TO_MS)
      out[i] = loc.update(bool(d["long_active"][i]), cs, float(driven[i]), bool(d["replay_should_stop"][i]), limits)
    mask = active & np.isfinite(d["logged_accel"])
    err = float(np.percentile(np.abs(out[mask] - d["logged_accel"][mask]), 95))
    plan_err = float(np.percentile(np.abs(driven[mask] - d["logged_a_target"][mask]), 95))
    print(f"{route}: commit {commit}, longcontrol replay-vs-logged p95 {err:.4f}, plan p95 {plan_err:.4f}, frames {mask.sum()}")


def register(sub):
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
