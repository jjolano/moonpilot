"""lagd replays: the fork estimator per route, the stock estimator per speed band, and the closed loop."""
import glob
import math

import numpy as np
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.selfdrive.locationd.lagd import LateralLagEstimator, MAX_LAG, MIN_LAG
from openpilot.tools.lib.logreader import LogReader

from moonpilot.lagd import MoonpilotLagEstimator
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, lag_s
from moonpilot.lab.lateral import action, extract as lateral_extract

CLAIMS = tuple(f"{i:08x}" for i in range(0x3F3, 0x40B))
ROUTE_SETS = {"claims": tuple(r for r in CLAIMS if glob.glob(f"{DEFAULT_ROOT}/{r}--*/rlog.zst")),  # 000003fb has no rlogs
              "closed-loop": ("00000408", "0000040e", "0000040d")}
INPUTS = ("deviceMotion", "extrinsicsCalibration", "carState", "controlsState", "carControl", "lateralDelay")
BANDS = ((10.0, 17.0), (17.0, 22.0), (22.0, math.inf), (0.0, 10.0))
EXTRACT_VERSION = 3  # v2 stores the CarParams payload (as lagd reads it); v3 adds the carState speed beside each event


def _build(segs: list[str]) -> dict[str, np.ndarray]:
  events, cp, speed = [], b"", 0.0
  for m in LogReader(segs, sort_by_time=True):
    which = m.which()
    if which == "carParams":
      cp = m.carParams.as_builder().to_bytes()
    elif which in INPUTS:
      if which == "carState":
        speed = float(m.carState.vEgo)
      events.append((m.logMonoTime, which, m.as_builder().to_bytes(), speed))
  blobs, offsets, times, kinds, speeds = [], [0], [], [], []
  for t, kind, blob, v in events:
    times.append(t)
    kinds.append(INPUTS.index(kind))
    speeds.append(v)
    blobs.extend(blob)
    offsets.append(len(blobs))
  if not cp:
    raise ValueError("no carParams")
  return {"time": np.array(times, dtype=np.int64), "kind": np.array(kinds, dtype=np.uint8),
          "offset": np.array(offsets, dtype=np.int64), "blob": np.frombuffer(bytes(blobs), dtype=np.uint8),
          "speed": np.array(speeds, dtype=np.float32), "cp": np.frombuffer(cp, dtype=np.uint8)}


def extract(route: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
  """The estimator's inputs in rlog order, raw, so every replay feeds them on their own logMonoTime."""
  return cached(route, "lagd", EXTRACT_VERSION, root, _build)


def _blocks(learner) -> list[float]:
  """The completed block values a fresh device would restore: the current partial slot excluded."""
  b = learner.block_avg
  return [float(b.values[i].mean()) for i in range(b.valid_blocks) if i != b.block_idx]


def _replay(route: str, root: str, stock: bool = False, band: tuple[float, float] | None = None) -> dict:
  """One route through `main()`'s loop from an empty cache: messages sorted by logMonoTime, update_estimate every 5th deviceMotion."""
  d = extract(route, root)
  CP = messaging.log_from_bytes(d["cp"].tobytes(), car.CarParams)
  dt = 1. / SERVICE_LIST["deviceMotion"].frequency
  learner = LateralLagEstimator(CP, dt) if stock else MoonpilotLagEstimator(CP, dt)
  if band is not None:
    learner.min_vego = band[0]
  speed, logged, updates = d["speed"], [], 0
  for i, (t, kind) in enumerate(zip(d["time"], d["kind"], strict=True)):
    which = INPUTS[int(kind)]
    if band is not None and which != "lateralDelay" and not band[0] < speed[i] <= band[1]:
      continue
    msg = getattr(messaging.log_from_bytes(d["blob"][d["offset"][i]:d["offset"][i + 1]].tobytes()), which)
    if which == "lateralDelay":
      logged.append((float(msg.lateralDelay), float(msg.lateralDelayEstimate), int(msg.validBlocks), int(msg.version)))
      continue
    learner.handle_log(t * 1e-9, which, msg)
    if which == "deviceMotion":
      learner.update_points()
      updates += 1
      if updates % 5 == 0:
        learner.update_estimate()
  out = learner.get_msg(True).lateralDelay
  # The same learner through upstream's get_msg: what the fork would apply without MOONPILOT_LAT_LAG_MAX_APPLIED.
  uncapped = LateralLagEstimator.get_msg(learner, True).lateralDelay.lateralDelay
  return {"est": float(out.lateralDelayEstimate), "applied": float(out.lateralDelay), "uncapped": float(uncapped),
          "valid": int(out.validBlocks), "blocks": _blocks(learner), "logged": logged, "commit": drive_config(route, root)[0]}


def _command(args):
  for route in expand(args.routes, ROUTE_SETS):
    r = _replay(route, args.root)
    last = r["logged"][-1] if r["logged"] else (float("nan"), float("nan"), 0, 0)
    err = f"{abs(r['applied'] - last[0]):.3f}" if last[3] == 1000 else "n/a"
    nb, est, applied, unc = len(r["blocks"]), r["est"], r["applied"], r["uncapped"]
    lg = f"{last[0]:.3f}({last[2]})"
    print(f"{route} {r['commit'][:9]}: blocks={nb} est={est:.3f} applied={applied:.3f} uncapped={unc:.3f} logged={lg} v{last[3]} error={err}")
    # The estimator the drive itself ran (version 1 is stock's), from an empty cache, against its logged value.
    s = _replay(route, args.root, stock=last[3] == 1)
    who = "fork" if last[3] == 1000 else "stock"
    print(f"  as-driven {who}: blocks={len(s['blocks'])} est={s['est']:.3f} applied={s['applied']:.3f} error={abs(s['applied'] - last[0]):.3f}")


def _bands(args):
  routes = expand(args.routes, ROUTE_SETS)
  for low, high in BANDS:
    pooled, estimates, rows = [], [], 0
    for route in routes:
      r = _replay(route, args.root, stock=True, band=(low, high))
      pooled += r["blocks"]
      if r["valid"] >= 5:
        estimates.append(r["est"])
        rows += 1
      print(f"{route} {low:g}-{high:g} m/s: blocks={len(r['blocks'])} valid={r['valid']} est={r['est']:.3f}")
    mean = float(np.mean(pooled)) if pooled else float("nan")
    route_mean = float(np.mean(estimates)) if estimates else float("nan")
    print(f"band {low:g}-{high:g} m/s: estimated={rows}/{len(routes)} routes pooled_blocks={len(pooled)} mean={mean:.3f} route_mean={route_mean:.3f}")


def _closed(args):
  for route in expand(args.routes, ROUTE_SETS):
    r = _replay(route, args.root)
    last = r["logged"][-1] if r["logged"] else (float("nan"), float("nan"), 0, 0)
    err = f"{abs(r['applied'] - last[0]):.3f}" if last[3] == 1000 else "n/a"
    first = next((row for row in r["logged"] if row[2] > 0), None)
    peak = max((row for row in r["logged"] if row[2] < 5), key=lambda row: row[1], default=None)
    d = lateral_extract(route, args.root)
    CP = messaging.log_from_bytes(d["cp"].tobytes(), car.CarParams)
    vm = VehicleModel(CP)
    vm.update_params(float(np.median(d["stiffness"])), float(np.median(d["steer_ratio"])))
    measured = -vm.calc_curvature(np.radians(d["angle"] - d["angle_offset"]), d["v_ego"], d["roll"])
    mask = d["lat_active"] & (d["v_ego"] > 10.0)
    wheel, model = lag_s(d["curvature"], measured, mask), lag_s(action(d), measured, mask)
    # The doc's "cross-correlation" reading of the same pair: upstream's own masked-NCC peak, no smoothing.
    xr, xm = (LateralLagEstimator.actuator_delay(np.asarray(ref, float), np.asarray(measured, float), mask, DT_CTRL, MIN_LAG, MAX_LAG)[0]
              for ref in (d["curvature"], action(d)))
    # The same pair against the car's calibrated yaw (what lagd itself identifies), not the steer angle.
    calib, y_t, y_v = PoseCalibrator(), [], []
    ld = extract(route, args.root)
    for i, (t, kind) in enumerate(zip(ld["time"], ld["kind"], strict=True)):
      which = INPUTS[int(kind)]
      if which not in ("deviceMotion", "extrinsicsCalibration"):
        continue
      msg = getattr(messaging.log_from_bytes(ld["blob"][ld["offset"][i]:ld["offset"][i + 1]].tobytes()), which)
      if which == "extrinsicsCalibration":
        calib.feed_extrinsics_calibration(msg)
      else:
        y_t.append(t * 1e-9)
        y_v.append(calib.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity.yaw)
    vehicle = np.interp(d["t"], np.asarray(y_t), np.asarray(y_v)) * d["v_ego"]
    yr, ym = (LateralLagEstimator.actuator_delay(np.asarray(ref, float), np.asarray(vehicle, float), mask, DT_CTRL, MIN_LAG, MAX_LAG)[0]
              for ref in (d["curvature"], action(d)))
    nb, est, applied, unc = len(r["blocks"]), r["est"], r["applied"], r["uncapped"]
    lg = f"{last[0]:.3f}({last[2]})"
    print(f"{route} {r['commit'][:9]}: blocks={nb} est={est:.3f} applied={applied:.3f} uncapped={unc:.3f} logged={lg} v{last[3]} error={err}")
    print(f"  wheel lag: rms request={wheel:.3f}s model={model:.3f}s xcorr request={xr:.3f}s model={xm:.3f}s frames={int(mask.sum())}")
    print(f"  yaw ncc: request={yr:.3f}s model={ym:.3f}s")
    print(f"  logged: first_block={first} unestimated_peak={peak}")


def register(sub):
  for name, fn, default in (("lagd", _command, "claims"), ("lagd-bands", _bands, "claims"), ("lagd-loop", _closed, "closed-loop")):
    p = sub.add_parser(name)
    p.add_argument("--routes", nargs="+", default=[default])
    p.set_defaults(fn=fn)
