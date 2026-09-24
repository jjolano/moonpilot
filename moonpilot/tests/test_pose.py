"""The observation-only pose tracker (`moonpilot/pose.py`) and its chain-RMSE gate.

`TestPoseTracker` drives the tracker on synthetic paths where the answer is known: a
rebase on the first fix, a soft pull that beats pure dead-reckoning under VO scale error,
the accuracy ceiling, the outlier radius, and a parked fix that must not steal the heading.

`TestChainRmse` is the Phase 0 exit gate from `moonpilot/docs/slam-spike.md`: chain
several route-corpus segments with no mid-route re-align, score fused pose and raw-VO
dead-reckon the same way against LLK truth, and require fused to win. The corpus lives
outside the checkout, so the whole class skips when it is absent — the synthetic half
still pins the mechanism.
"""

import math
import unittest
from pathlib import Path

from moonpilot.pose import (
  MOONPILOT_POSE_GATE_RADIUS,
  MOONPILOT_POSE_GPS_ALPHA,
  MOONPILOT_POSE_MAX_HACC,
  MOONPILOT_POSE_REBASE_STREAK,
  PoseTracker,
  latlon_to_enu,
)

CORPUS = Path("/home/coder/route-corpus/logs")
ROUTE = "000001b7--1e83052604"
# Segments 1–5: the same chain the spike measured at RMSE 341 m without a gate.
CHAIN_SEGS = (1, 2, 3, 4, 5)
SETTLE_S = 2.0  # s; drop after stream start before the alignment window
ALIGN_MIN_PATH_M = 40.0  # m of reference path inside the fit; 4 s at rest fits noise
ALIGN_MAX_S = 30.0  # s; cap on growing the fit window


class TestPoseTracker(unittest.TestCase):
  def test_latlon_to_enu_is_local_and_linear(self):
    x, y = latlon_to_enu(43.7, -79.8, 43.7, -79.8)
    self.assertEqual((x, y), (0.0, 0.0))
    # One degree of latitude is ~111 km; one of longitude at this latitude is shorter.
    _, y = latlon_to_enu(44.7, -79.8, 43.7, -79.8)
    self.assertAlmostEqual(y, 111_000, delta=1_000)
    x, _ = latlon_to_enu(43.7, -78.8, 43.7, -79.8)
    self.assertAlmostEqual(x, 81_000, delta=2_000)

  def test_first_fix_rebases_and_later_fixes_soft_correct(self):
    t = PoseTracker()
    # Dead-reckon 10 m east with no origin yet — still allowed, just not absolute.
    for i in range(21):
      t.push_odom(i * 0.05, v_forward=10.0, yaw_rate=0.0)
    self.assertFalse(t.has_origin)
    self.assertAlmostEqual(t.x, 10.0, delta=1e-6)

    # First fix: origin set, position snapped to 0, history dropped on purpose.
    self.assertTrue(t.push_gps(1.1, 43.7, -79.8, horizontal_accuracy=2.0))
    self.assertTrue(t.has_origin)
    self.assertEqual(t.n_gps, 1)
    self.assertEqual((t.x, t.y), (0.0, 0.0))

    # A fix 5 m north of where VO put us pulls the estimate partway there.
    lat = 43.7 + math.degrees(5.0 / 6_378_137.0)
    self.assertTrue(t.push_gps(2.1, lat, -79.8, horizontal_accuracy=2.0))
    self.assertEqual(t.n_gps, 2)
    self.assertAlmostEqual(t.y, 5.0 * MOONPILOT_POSE_GPS_ALPHA, delta=1e-9)
    self.assertAlmostEqual(t.x, 0.0, delta=1e-9)

  def test_gated_fixes_do_not_move_the_estimate(self):
    t = PoseTracker()
    self.assertFalse(t.push_gps(0.0, 43.7, -79.8, has_fix=False))
    self.assertFalse(t.has_origin)
    # Above the accuracy ceiling: unknown-but-huge is not a gate we trust either.
    self.assertFalse(t.push_gps(0.0, 43.7, -79.8, horizontal_accuracy=MOONPILOT_POSE_MAX_HACC + 1.0))
    self.assertFalse(t.has_origin)
    # Negative accuracy is nonsense.
    self.assertFalse(t.push_gps(0.0, 43.7, -79.8, horizontal_accuracy=-1.0))
    self.assertFalse(t.has_origin)

    # hacc == 0 is "unknown" on qcom (every corpus sample), and passes as hasFix-only.
    self.assertTrue(t.push_gps(0.0, 43.7, -79.8, horizontal_accuracy=0.0))
    self.assertTrue(t.has_origin)

  def test_an_outlier_fix_is_rejected_by_radius(self):
    t = PoseTracker()
    self.assertTrue(t.push_gps(0.0, 43.7, -79.8, horizontal_accuracy=2.0))
    # 100 m north: past the radius, so not a correction.
    lat = 43.7 + math.degrees(100.0 / 6_378_137.0)
    self.assertFalse(t.push_gps(1.0, lat, -79.8, horizontal_accuracy=2.0))
    self.assertEqual(t.n_gps, 1)
    self.assertEqual((t.x, t.y), (0.0, 0.0))
    # Just inside the radius still corrects.
    lat = 43.7 + math.degrees((MOONPILOT_POSE_GATE_RADIUS - 1.0) / 6_378_137.0)
    self.assertTrue(t.push_gps(2.0, lat, -79.8, horizontal_accuracy=2.0))
    self.assertEqual(t.n_gps, 2)

  def test_a_rejected_streak_rebases_instead_of_dying_in_the_gate(self):
    """Sustained >radius disagreement is a lost dead-reckon, not multipath forever."""
    t = PoseTracker()
    self.assertTrue(t.push_gps(0.0, 43.7, -79.8, horizontal_accuracy=2.0))
    lat_far = 43.7 + math.degrees(100.0 / 6_378_137.0)
    for i in range(MOONPILOT_POSE_REBASE_STREAK - 1):
      self.assertFalse(t.push_gps(1.0 + i, lat_far, -79.8, horizontal_accuracy=2.0))
    self.assertEqual(t.n_rebase, 0)
    self.assertTrue(t.push_gps(1.0 + MOONPILOT_POSE_REBASE_STREAK, lat_far, -79.8, horizontal_accuracy=2.0))
    self.assertEqual(t.n_rebase, 1)
    self.assertAlmostEqual(t.y, 100.0, delta=1e-6)
    # The streak resets: the next fix, now near the rebased position, soft-corrects.
    # A fix still near the original origin would be 95 m out and correctly rejected again.
    lat_ok = 43.7 + math.degrees(105.0 / 6_378_137.0)
    self.assertTrue(t.push_gps(10.0, lat_ok, -79.8, horizontal_accuracy=2.0))
    self.assertAlmostEqual(t.y, 100.0 + MOONPILOT_POSE_GPS_ALPHA * 5.0, delta=1e-6)

  def test_a_parked_bearing_does_not_steal_the_heading(self):
    t = PoseTracker()
    t.push_odom(0.0, v_forward=0.0, yaw_rate=0.0)
    t.yaw = 0.5  # already moving / turning before the fix
    self.assertTrue(t.push_gps(1.0, 43.7, -79.8, bearing_deg=180.0, speed=0.1))
    self.assertAlmostEqual(t.yaw, 0.5)

  def test_a_moving_fix_seeds_yaw_from_course_over_ground(self):
    t = PoseTracker()
    # Course 90° = due east; local yaw is CCW from east, so 0.
    self.assertTrue(t.push_gps(1.0, 43.7, -79.8, bearing_deg=90.0, speed=10.0))
    self.assertAlmostEqual(t.yaw, 0.0, delta=1e-9)
    # Course 0° = due north = +90° local.
    t2 = PoseTracker()
    self.assertTrue(t2.push_gps(1.0, 43.7, -79.8, bearing_deg=0.0, speed=10.0))
    self.assertAlmostEqual(t2.yaw, math.pi / 2.0, delta=1e-9)

  def test_a_slow_first_fix_leaves_yaw_for_later_course_blends(self):
    """The corpus case: first fix under the speed gate, so yaw is not seeded."""
    t = PoseTracker()
    self.assertTrue(t.push_gps(1.0, 43.7, -79.8, bearing_deg=214.0, speed=1.2))
    self.assertEqual(t.yaw, 0.0)  # not seeded — below MOONPILOT_POSE_BEARING_MIN_SPEED
    # Once moving, each fix blends toward course (bearing 0° = north = +π/2).
    self.assertTrue(t.push_gps(2.0, 43.7, -79.8, bearing_deg=0.0, speed=10.0))
    self.assertGreater(t.yaw, 0.0)
    self.assertLess(t.yaw, math.pi / 2.0)

  def test_odom_gaps_are_skipped_not_leapt(self):
    t = PoseTracker()
    t.push_odom(0.0, v_forward=10.0, yaw_rate=0.0)
    # A 2 s hole: do not integrate 20 m through the gap.
    t.push_odom(2.0, v_forward=10.0, yaw_rate=0.0)
    self.assertAlmostEqual(t.x, 0.0, delta=1e-9)

  def test_a_fused_chain_beats_raw_dead_reckoning_under_scale_error(self):
    """Synthetic stand-in for the corpus gate: VO forward runs 10 % fast, GPS every second."""
    dt = 0.05
    v_true = 10.0
    n = 400  # 20 s, 200 m east (yaw stays 0, matching the odom integrate)
    fused = PoseTracker()
    raw_x = 0.0
    lat0, lon0 = 43.7, -79.8
    m_per_deg_lon = 6_378_137.0 * math.cos(math.radians(lat0))

    for i in range(n + 1):
      t = i * dt
      if i % int(round(1.0 / dt)) == 0:
        # Eastward truth as a noiseless GPS fix. The t=0 fix sets the origin before any
        # motion, so fused.x and truth_x share a frame (a later rebase would drop the
        # first leg and leave a constant offset against absolute truth).
        lon = lon0 + math.degrees(v_true * t / m_per_deg_lon)
        fused.push_gps(t, lat0, lon, horizontal_accuracy=2.0)
      # Fused dead-reckons on the true (wheel) forward; raw integrates the bad VO scale.
      fused.push_odom(t, v_forward=v_true, yaw_rate=0.0)
      if i > 0:
        raw_x += v_true * 1.1 * dt

    truth_x = v_true * n * dt
    fused_err = abs(fused.x - truth_x)
    raw_err = abs(raw_x - truth_x)
    self.assertLess(fused_err, raw_err)
    # With a 1 Hz gate the residual is small, not merely "better than terrible".
    self.assertLess(fused_err, 5.0)


def _load_chain(segments):
  """Merged cam/car/gps/llk/extrinsics from the chain, or None when the corpus is absent."""
  if not CORPUS.is_dir():
    return None
  try:
    from openpilot.tools.lib.logreader import LogReader
    import numpy as np
  except ImportError:
    return None

  cam, car, gps, llk = [], [], [], []
  extr = None
  any_seg = False
  for i in segments:
    path = CORPUS / f"{ROUTE}--{i}" / "rlog.zst"
    if not path.exists():
      continue
    any_seg = True
    for m in LogReader(str(path)):
      w = m.which()
      if w == "extrinsicsCalibration" and extr is None:
        try:
          rpy = np.asarray(m.extrinsicsCalibration.rpyCalib, dtype=float)
          if rpy.shape == (3,) and np.all(np.isfinite(rpy)):
            extr = rpy
        except Exception:
          pass
      elif w == "cameraOdometry":
        od = m.cameraOdometry
        if not m.valid or len(od.trans) < 3 or len(od.rot) < 3:
          continue
        # timestampEof is already nanoseconds; pose time is exposure, not publish.
        t = od.timestampEof * 1e-9 - 0.1
        cam.append((t, float(od.trans[0]), float(od.trans[1]), float(od.rot[2])))
      elif w == "carState":
        car.append((m.logMonoTime * 1e-9, float(m.carState.vEgo)))
      elif w == "gpsLocation":
        g = m.gpsLocation
        gps.append((m.logMonoTime * 1e-9, float(g.latitude), float(g.longitude),
                    float(g.horizontalAccuracy), bool(g.hasFix), float(g.bearingDeg), float(g.speed)))
      elif w == "liveLocationKalmanDEPRECATED":
        ll = m.liveLocationKalmanDEPRECATED
        try:
          pos = ll.positionGeodetic
          if pos.valid:
            llk.append((m.logMonoTime * 1e-9, float(pos.value[0]), float(pos.value[1])))
        except Exception:
          pass
  if not any_seg or not cam or not llk:
    return None
  return {"cam": cam, "car": car, "gps": gps, "llk": llk, "extr": extr}


def _rpy_to_R(rpy):
  import numpy as np
  roll, pitch, yaw = rpy
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)
  return np.array([
    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
    [-sp, cp * sr, cp * cr],
  ])


def _integrate_vo_raw(cam, extr):
  """Raw dead-reckon: full VO body velocity, no wheel, no GPS. Returns (t[], xy Nx2)."""
  import numpy as np
  R_calib = _rpy_to_R(extr) if extr is not None else np.eye(3)
  yaw = x = y = 0.0
  ts, xs, ys = [], [], []
  prev_t = None
  for t, vx, vy, wz in cam:
    if prev_t is not None and t > prev_t:
      dt = min(t - prev_t, 0.2)
      v_dev = R_calib @ np.array([vx, vy, 0.0])
      yaw_mid = yaw + 0.5 * wz * dt
      c, s = math.cos(yaw_mid), math.sin(yaw_mid)
      x += (c * v_dev[0] - s * v_dev[1]) * dt
      y += (s * v_dev[0] + c * v_dev[1]) * dt
      yaw += wz * dt
    prev_t = t
    ts.append(t)
    xs.append(x)
    ys.append(y)
  return np.asarray(ts), np.column_stack([xs, ys])


def _run_fused(data):
  """Wheel+VO+GPS through PoseTracker, streams merged by time. Returns (t[], xy Nx2).

  Wheel speed is read with full-history ``np.interp``, not ``PriorChannel``: production
  feeds carState continuously and reads a pose time 0.1 s in the past, but a time-merged
  offline replay only has car samples with stamp ≤ the cam pose time when the cam event
  fires, so ``PriorChannel.at``'s no-extrapolate rule rejects every frame.
  """
  import numpy as np

  tracker = PoseTracker()
  R_calib = _rpy_to_R(data["extr"]) if data["extr"] is not None else np.eye(3)
  car_t = np.asarray([r[0] for r in data["car"]], dtype=float)
  car_v = np.asarray([r[1] for r in data["car"]], dtype=float)

  events = []
  for t, v in data["car"]:
    events.append((t, "car", v))
  for row in data["gps"]:
    events.append((row[0], "gps", row))
  for t, vx, vy, wz in data["cam"]:
    events.append((t, "cam", (t, vx, vy, wz)))
  events.sort(key=lambda e: e[0])

  ts, xs, ys = [], [], []
  for t, kind, payload in events:
    if kind == "car":
      continue  # history already in car_t/car_v
    if kind == "gps":
      row = payload
      assert isinstance(row, tuple) and len(row) == 7
      _, lat, lon, hacc, fix, bearing, speed = row
      tracker.push_gps(t, lat, lon, horizontal_accuracy=hacc, has_fix=fix,
                       bearing_deg=bearing, speed=speed)
      continue
    cam_row = payload
    assert isinstance(cam_row, tuple) and len(cam_row) == 4
    _, vx, vy, wz = cam_row
    v_fwd = float(np.interp(t, car_t, car_v)) if car_t[0] <= t <= car_t[-1] else None
    v_dev = R_calib @ np.array([vx, vy, 0.0])
    if v_fwd is None:
      v_fwd = float(v_dev[0])
    tracker.push_odom(t, v_forward=v_fwd, yaw_rate=wz, v_lateral=float(v_dev[1]))
    ts.append(t)
    xs.append(tracker.x)
    ys.append(tracker.y)
  return np.asarray(ts), np.column_stack([xs, ys])


def _llk_enu(llk):
  import numpy as np
  lat0, lon0 = llk[0][1], llk[0][2]
  ts = np.asarray([r[0] for r in llk])
  # asarray, not column_stack: latlon_to_enu returns a 2-tuple per row, and column_stack
  # would treat those tuples as columns (shape 2×N), breaking every np.interp below.
  xy = np.asarray([latlon_to_enu(r[1], r[2], lat0, lon0) for r in llk])
  return ts, xy


def _align_end(t_lo, ref_t, ref_xy):
  """Grow the fit window until it carries ALIGN_MIN_PATH_M of reference path (or the cap)."""
  import numpy as np
  i0 = int(np.searchsorted(ref_t, t_lo))
  acc = 0.0
  for i in range(i0 + 1, len(ref_t)):
    acc += float(np.hypot(ref_xy[i, 0] - ref_xy[i - 1, 0], ref_xy[i, 1] - ref_xy[i - 1, 1]))
    if acc >= ALIGN_MIN_PATH_M or ref_t[i] - t_lo >= ALIGN_MAX_S:
      return float(ref_t[i])
  return float(min(ref_t[-1], t_lo + ALIGN_MAX_S))


def _align_yaw_xy(est_t, est_xy, ref_t, ref_xy, t_lo, t_hi):
  """One yaw+translation fit (no scale) over [t_lo, t_hi]; apply to the whole chain."""
  import numpy as np
  mask = (est_t >= t_lo) & (est_t <= t_hi)
  if int(mask.sum()) < 20:
    return None
  idx = np.flatnonzero(mask)
  vt = est_t[idx]
  if len(ref_t) == 0 or vt[0] < ref_t[0] or vt[-1] > ref_t[-1]:
    return None
  origin = est_xy[idx[0]].copy()
  V = est_xy[idx] - origin
  R_abs = np.column_stack([np.interp(vt, ref_t, ref_xy[:, 0]), np.interp(vt, ref_t, ref_xy[:, 1])])
  R0 = R_abs[0].copy()
  R = R_abs - R0
  path = float(np.sum(np.linalg.norm(np.diff(R, axis=0), axis=1)))
  if path < ALIGN_MIN_PATH_M * 0.5:
    return None  # too little motion to identify a yaw — refuse rather than fit noise
  dV = np.diff(V, axis=0)
  dR = np.diff(R, axis=0)
  cross = float(np.sum(dV[:, 0] * dR[:, 1] - dV[:, 1] * dR[:, 0]))
  dot = float(np.sum(dV[:, 0] * dR[:, 0] + dV[:, 1] * dR[:, 1]))
  yaw = math.atan2(cross, dot)
  c, s = math.cos(yaw), math.sin(yaw)
  Rot = np.array([[c, -s], [s, c]])
  return (est_xy - origin) @ Rot.T + R0


def _score(est_t, est_xy, ref_t, ref_xy, t_lo, t_hi):
  import numpy as np
  mask = (est_t >= t_lo) & (est_t <= t_hi)
  if int(mask.sum()) < 50:
    return None
  vt = est_t[mask]
  if len(ref_t) == 0 or vt[0] < ref_t[0] or vt[-1] > ref_t[-1]:
    return None
  R = np.column_stack([np.interp(vt, ref_t, ref_xy[:, 0]), np.interp(vt, ref_t, ref_xy[:, 1])])
  err = np.linalg.norm(est_xy[mask] - R, axis=1)
  dist = float(np.sum(np.linalg.norm(np.diff(R, axis=0), axis=1)))
  return {"rmse": float(np.sqrt(np.mean(err**2))), "final": float(err[-1]), "dist": dist, "n": int(mask.sum())}


class TestChainRmse(unittest.TestCase):
  def test_fused_chain_rmse_beats_raw_vo_against_llk(self):
    data = _load_chain(CHAIN_SEGS)
    if data is None:
      self.skipTest("route-corpus not present")

    raw_t, raw_xy = _integrate_vo_raw(data["cam"], data["extr"])
    fused_t, fused_xy = _run_fused(data)
    ref_t, ref_xy = _llk_enu(data["llk"])

    # One alignment window near the start (after both streams exist + settle, grown until
    # it carries real path), then score the rest of the chain with no re-align — the
    # spike's chained protocol, minus the free scale the per-segment Procrustes gave raw VO.
    t_lo = max(float(raw_t[0]), float(ref_t[0])) + SETTLE_S
    t_hi = _align_end(t_lo, ref_t, ref_xy)
    t_end = min(float(raw_t[-1]), float(fused_t[-1]), float(ref_t[-1]))

    # Raw VO starts in a free frame (origin + yaw from the first sample) and needs the
    # one-shot fit. Fused is already GPS-seeded ENU — origin = first fix ≈ first LLK
    # within ~5 m — so rotating it with a short yaw fit only destroys that; score as-is.
    raw_aligned = _align_yaw_xy(raw_t, raw_xy, ref_t, ref_xy, t_lo, t_hi)
    self.assertIsNotNone(raw_aligned, "raw VO did not align")
    assert raw_aligned is not None

    raw = _score(raw_t, raw_aligned, ref_t, ref_xy, t_hi, t_end)
    fused = _score(fused_t, fused_xy, ref_t, ref_xy, t_hi, t_end)
    if raw is None or fused is None:
      self.fail(f"score returned None (raw={raw}, fused={fused})")

    # The gate: fused must beat raw VO. The spike's ungated chain was 341 m / 13 % —
    # anything in that neighbourhood means the GPS gate is not doing its job.
    self.assertLess(
      fused["rmse"], raw["rmse"],
      msg=f"fused {fused['rmse']:.1f} m not below raw {raw['rmse']:.1f} m (dist {fused['dist']:.0f} m, n={fused['n']})",
    )
    # And fused should be in the metres, not tens of metres, on a 1 Hz gate.
    self.assertLess(fused["rmse"], 20.0, msg=f"fused RMSE {fused['rmse']:.1f} m too large")


if __name__ == "__main__":
  unittest.main()
