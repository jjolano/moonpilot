"""Replays for the fork's rolling ego-speed correction and chained ego pose."""
import numpy as np

from openpilot.tools.lib.logreader import LogReader

from moonpilot.lab import DEFAULT_ROOT, cached, expand
from moonpilot.leadd import _gps_push, _pose_push, _slam_node
from moonpilot.pose import PoseTracker, latlon_to_enu
from moonpilot.slam import PriorChannel, RotatingPoseWindow

EXTRACT_VERSION = 2
ROUTE_SETS: dict[str, tuple[str, ...]] = {"latest": ("0000043f", "0000043e")}


class ReplaySM:
  def __init__(self):
    self.data, self.updated, self.valid, self.logMonoTime = {}, {}, {}, {}

  def __getitem__(self, key):
    return self.data[key]


def _build(segs: list[str]) -> dict[str, np.ndarray]:
  sm = ReplaySM()
  prior, window, tracker = PriorChannel(), RotatingPoseWindow(), PoseTracker()
  latest_a_ego, latest_long_active, latest_onroad = float("nan"), False, True
  truth_service, gps_origin = None, None
  truth, logged, poses, slam, lags = [], [], [], [], []
  gps_services = ("gpsLocationExternal", "gpsLocation")
  for msg in LogReader(segs, sort_by_time=True):
    service = msg.which()
    t = msg.logMonoTime * 1e-9
    if service == "moonpilotState":
      state = msg.moonpilotState
      logged.append((t, bool(state.egoCorrection.valid), float(state.egoCorrection.dVel),
                     bool(state.egoPose.valid), float(state.egoPose.x), float(state.egoPose.y)))
      continue
    if service == "modelV2":
      if sm.updated.get("cameraOdometry"):
        node = _slam_node(sm, prior)
        if node is not None:
          window.push(node)
          corr = window.update()
          publish_t = sm.logMonoTime["cameraOdometry"] * 1e-9
          v_publish = prior.at(publish_t)
          prior_diff = node.v_ego - v_publish if v_publish is not None else np.nan
          slam.append((node.mono_time, node.v_ego, node.v_ego + corr["dVel"], corr["dVel"],
                       corr["valid"], prior_diff, latest_a_ego, latest_long_active, latest_onroad))
      for key in sm.updated:
        sm.updated[key] = False
      continue
    if service == "deviceState":
      latest_onroad = bool(msg.deviceState.started)
      continue
    if service == "carControl":
      latest_long_active = bool(msg.carControl.longActive)
      continue
    if service not in ("carState", "cameraOdometry", "extrinsicsCalibration", *gps_services, "liveLocationKalman"):
      continue
    if service == "carState":
      latest_a_ego = float(msg.carState.aEgo)
      prior.push(t, float(msg.carState.vEgo))
    if service == "cameraOdometry":
      lags.append(t - float(msg.cameraOdometry.timestampEof) * 1e-9)
    if service in ("carState", "cameraOdometry", "extrinsicsCalibration", *gps_services):
      sm.data[service] = getattr(msg, service)
      sm.valid[service] = bool(getattr(msg, "valid", True))
      sm.logMonoTime[service] = msg.logMonoTime
      sm.updated[service] = True
    if service in gps_services:
      if service == "gpsLocationExternal":
        if truth_service not in (None, "gpsLocationExternal"):
          truth, gps_origin = [], None
        truth_service = service
      elif truth_service is None:
        truth_service = service
      if service == truth_service:
        g = getattr(msg, service)
        if bool(getattr(g, "hasFix", True)):
          lat, lon = float(g.latitude), float(g.longitude)
          if gps_origin is None:
            gps_origin = (lat, lon)
          x, y = latlon_to_enu(lat, lon, *gps_origin)
          truth.append((t, x, y, float(getattr(g, "speed", np.nan))))
        _gps_push(tracker, sm, service)
      continue
    if service == "liveLocationKalman" and truth_service is None:
      g = msg.liveLocationKalman
      if g.status and g.positionGeodetic.valid:
        truth_service = service
        lat, lon = map(float, g.positionGeodetic.value[:2])
        gps_origin = (lat, lon)
        x, y = latlon_to_enu(lat, lon, *gps_origin)
        truth.append((t, x, y, np.nan))
      continue
    if service == "cameraOdometry":
      _pose_push(tracker, sm, prior)
      if tracker.has_origin:
        poses.append((tracker.mono_time, tracker.x, tracker.y))
  slam_arr = np.asarray(slam, dtype=float).reshape((-1, 9))
  truth_arr = np.asarray(truth, dtype=float).reshape((-1, 4))
  gps_v = np.full(len(slam_arr), np.nan)
  if len(slam_arr) and len(truth_arr):
    if len(truth_arr) == 1:
      gps_v[:] = truth_arr[0, 3]
    else:
      right = np.clip(np.searchsorted(truth_arr[:, 0], slam_arr[:, 0]), 1, len(truth_arr) - 1)
      left = right - 1
      nearest = left + (np.abs(slam_arr[:, 0] - truth_arr[left, 0]) > np.abs(truth_arr[right, 0] - slam_arr[:, 0]))
      gps_v = truth_arr[nearest, 3]
  return {"slam": slam_arr, "truth": truth_arr, "gps_v": gps_v,
          "pose": np.asarray(poses, dtype=float).reshape((-1, 3)),
          "logged": np.asarray(logged, dtype=float).reshape((-1, 6)),
          "lags": np.asarray(lags, dtype=float),
          "truth_service": np.asarray(truth_service or "none")}


def extract(route: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
  return cached(route, "slam", EXTRACT_VERSION, root, _build)


def _rms(x):
  x = np.asarray(x, dtype=float)
  x = x[np.isfinite(x)]
  return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")


def _nearest(t, logged):
  if not len(t) or not len(logged):
    return np.empty((0, 6))
  if len(logged) == 1:
    return np.repeat(logged, len(t), axis=0)
  right = np.searchsorted(logged[:, 0], t)
  right = np.clip(right, 1, len(logged) - 1)
  left = right - 1
  choose_left = np.abs(t - logged[left, 0]) <= np.abs(logged[right, 0] - t)
  return logged[np.where(choose_left, left, right)]


def _report(route: str, args):
  d = extract(route, args.root)
  logged, truth_service = d["logged"], str(d["truth_service"])
  if args.replay == "slam":
    a, gps_v, lags = d["slam"], d["gps_v"], d["lags"] * 1000.0
    valid = a[:, 4].astype(bool)
    onroad = a[:, 8].astype(bool)
    duty = 100 * valid[onroad].mean() if onroad.any() else 100 * valid.mean() if len(valid) else 0.0
    duty_counts = f"{int(valid[onroad].sum())}/{int(onroad.sum())}" if onroad.any() else f"{int(valid.sum())}/{len(valid)}"
    raw_rmse, corrected_rmse = _rms(a[:, 1] - gps_v), _rms(a[:, 2] - gps_v)
    print(f"{route}: truth={truth_service}; speed RMSE raw {raw_rmse:.3f} m/s, corrected {corrected_rmse:.3f} m/s; valid duty {duty:.2f}% ({duty_counts})")
    parked = a[:, 1] < 0.3
    exc = a[parked, 3] > 0.1
    engaged = a[parked, 7].astype(bool) & exc
    worst = float(np.max(a[parked, 3])) if parked.any() else float("nan")
    rate = 100 * exc.mean() if len(exc) else 0.0
    counts = f"{int(exc.sum())}/{int(parked.sum())}; engaged-long {int(engaged.sum())}/{int(exc.sum())}"
    print(f"  parked correction >0.1 m/s: {rate:.2f}% ({counts}), worst {worst:.3f}")
    finite = np.isfinite(lags)
    if finite.any():
      p50, p95, mx = np.median(lags[finite]), np.percentile(lags[finite], 95), np.max(lags[finite])
      print(f"  cameraOdometry publish latency: median {p50:.1f} ms, p95 {p95:.1f} ms, max {mx:.1f} ms ({int(finite.sum())} frames)")
    arm = np.isfinite(a[:, 5]) & np.isfinite(a[:, 6])
    braking = arm & (a[:, 6] < 0.0)
    if arm.any():
      corr_aego = float(np.corrcoef(a[arm, 5], a[arm, 6])[0, 1])
      diff_brake = float(np.mean(a[braking, 5])) if braking.any() else float("nan")
      corr_brake = float(np.mean(np.abs(a[braking, 3]))) if braking.any() else float("nan")
      ratio = 100 * diff_brake / corr_brake if np.isfinite(corr_brake) and corr_brake else float("nan")
      print(f"  prior pose-time minus publish-time: corr(aEgo) {corr_aego:.3f}; braking mean {diff_brake:+.3f} m/s ({ratio:.1f}% of |correction|)")
    match = _nearest(a[:, 0], logged)
    mask = match[:, 1].astype(bool) if len(match) else np.zeros(0, dtype=bool)
    print(f"  egoCorrection replay/logged RMSE {_rms(a[mask, 3] - match[mask, 2]):.3f} m/s ({int(mask.sum())} valid matches)")
  else:
    truth, pose = d["truth"], d["pose"]
    if len(truth) and len(pose):
      xy = np.column_stack([np.interp(pose[:, 0], truth[:, 0], truth[:, j]) for j in (1, 2)])
      pose_rmse = _rms(np.linalg.norm(pose[:, 1:3] - xy, axis=1))
      print(f"{route}: truth={truth_service}; chained pose RMSE {pose_rmse:.3f} m ({len(pose)} poses, {len(truth)} truth fixes)")
    else:
      print(f"{route}: truth={truth_service}; chained pose RMSE unavailable (no pose origin or truth)")
    match = _nearest(pose[:, 0], logged)
    mask = match[:, 3].astype(bool) if len(match) else np.zeros(0, dtype=bool)
    err = np.linalg.norm(pose[mask, 1:3] - match[mask, 4:6], axis=1) if len(match) else []
    print(f"  egoPose replay/logged RMSE {_rms(err):.3f} m ({int(mask.sum())} valid matches)")


def _run(args):
  for route in expand(args.routes, ROUTE_SETS):
    _report(route, args)


def register(sub):
  for name in ("slam", "pose"):
    p = sub.add_parser(name, help=f"replay {name} against recorded drives")
    p.add_argument("--routes", nargs="+", default=["latest"])
    p.set_defaults(fn=_run, replay=name)
