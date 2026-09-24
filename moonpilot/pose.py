"""moonpilot's observation-only 2D ego pose: dead-reckon, gated by GPS.

Phase 1 of the true-SLAM path (`moonpilot/docs/slam-spike.md`). Short-horizon pose is
`cameraOdometry` alone — excellent over ~60 s and useless over minutes — so this tracker
dead-reckons the body at 20 Hz (wheel speed forward, VO lateral and yaw) and soft-corrects
position against a GPS fix whenever one clears the gate. Nothing here is in the control path:
the consumer that does not exist yet would read `moonpilotState.egoPose`; until it does, the
tracker exists so a replay can score chained-pose RMSE against LLK the way `test_slam.py`
scores speed.

Four things that are not obvious:

- **Forward comes from the wheel, not the VO.** Measured scale is 0.99–1.02 either way
  (`slam-spike.md`), but the wheel is the prior every other consumer already trusts, and the
  VO's job here is lateral slip and yaw rate — what the wheels cannot see. When there is no
  wheel sample at the pose time the VO's forward is used, so a bare replay still integrates.
- **The first accepted fix rebases to the origin.** Local ENU is meaningless without one, and
  dead-reckoning from an unknown absolute start is exactly the drift this gate exists to stop.
  The rebase drops position history on purpose: absolute continuity before the first fix is
  not something this module can honestly claim.
- **Yaw is steered by GPS course-over-ground while moving, not only seeded once.** `wz` from
  the VO is a rate: it cannot discover a wrong absolute heading, and a first fix too slow to
  trust bearing leaves yaw at 0 forever — the route-corpus chain headed south-west while yaw
  sat on east, position soft-corrects held for ~130 s, then the 50 m radius gate rejected
  every remaining fix with no way back. So every fix at or above
  `MOONPILOT_POSE_BEARING_MIN_SPEED` blends yaw toward course (qcom's `bearingDeg` matches
  displacement-derived course to ~1° median when moving), and a streak of radius rejects
  rebases position instead of dying in the gate.
- **The gate is `hasFix` plus a horizontal-accuracy ceiling, with a radius reject on top.**
  `horizontalAccuracy == 0` is treated as unknown rather than perfect — qcom's `gpsLocation`
  reports 0 on every corpus sample — so it passes the ceiling and leans on `hasFix` alone.
  A fix farther than `MOONPILOT_POSE_GATE_RADIUS` from the dead-reckon is an outlier (a
  multipath jump or a clock step), not a correction: pulling toward it would be the bug —
  unless it keeps happening, which is the streak rebase above.
"""

import math

R_EARTH = 6378137.0  # m; WGS84 equatorial radius, same constant the spike used

# Fork-owned tuning. Tune by replaying a route and watching chained RMSE against LLK.
MOONPILOT_POSE_MAX_HACC = 15.0  # m; worse than this is not a gate we trust. 0 passes as unknown.
MOONPILOT_POSE_GATE_RADIUS = 50.0  # m; reject a fix this far from the dead-reckon as an outlier
MOONPILOT_POSE_GPS_ALPHA = 0.25  # per-fix blend toward GPS ENU; 1 Hz ≈ 4 s time constant
MOONPILOT_POSE_MIN_DT = 1e-3  # s; below this the odom step is a repeat, not an interval
MOONPILOT_POSE_MAX_DT = 0.5  # s; a longer gap is a hole — skip rather than leap
MOONPILOT_POSE_BEARING_MIN_SPEED = 3.0  # m/s; below this GPS bearing is noise, not heading
MOONPILOT_POSE_YAW_ALPHA = 0.25  # per-fix blend of yaw toward course-over-ground; same τ as position
# Consecutive radius rejects before position is rebased hard. Multipath does not sit >50 m off
# for five seconds while the dead-reckon is healthy; a lost heading does.
MOONPILOT_POSE_REBASE_STREAK = 5


def latlon_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
  """Geodetic degrees -> local ENU meters relative to (lat0, lon0). x east, y north."""
  x = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
  y = math.radians(lat - lat0) * R_EARTH
  return x, y


def _wrap_pi(a: float) -> float:
  """Wrap to (-π, π]."""
  return (a + math.pi) % (2.0 * math.pi) - math.pi


class PoseTracker:
  """Dead-reckoned body pose in a GPS-seeded local ENU frame, soft-gated by fixes."""

  def __init__(self) -> None:
    self.x = 0.0  # m east of the origin
    self.y = 0.0  # m north of the origin
    self.yaw = 0.0  # rad, CCW from +x (east), right-handed
    self.lat0: float | None = None
    self.lon0: float | None = None
    self.t: float | None = None  # s; last odom stamp
    self.n_gps = 0  # accepted fixes (first one rebases)
    self.n_rebase = 0  # streak rebases after the first (diagnostic for replays)
    self._reject_streak = 0

  @property
  def has_origin(self) -> bool:
    return self.lat0 is not None

  def push_odom(self, mono_time: float, v_forward: float, yaw_rate: float, v_lateral: float = 0.0) -> None:
    """One body-frame step at the odometry's pose time (seconds, monotonic)."""
    if not all(math.isfinite(v) for v in (mono_time, v_forward, yaw_rate, v_lateral)):
      return
    if self.t is None:
      self.t = mono_time
      return
    dt = mono_time - self.t
    self.t = mono_time
    if not (MOONPILOT_POSE_MIN_DT < dt < MOONPILOT_POSE_MAX_DT):
      return
    # Mid-point yaw so a constant turn integrates without the end-of-step bias.
    yaw_mid = self.yaw + 0.5 * yaw_rate * dt
    c, s = math.cos(yaw_mid), math.sin(yaw_mid)
    self.x += (c * v_forward - s * v_lateral) * dt
    self.y += (s * v_forward + c * v_lateral) * dt
    self.yaw += yaw_rate * dt

  def push_gps(self, mono_time: float, lat: float, lon: float, horizontal_accuracy: float | None = None,
               has_fix: bool = True, bearing_deg: float | None = None, speed: float | None = None) -> bool:
    """Gate and apply one GPS fix. True when the fix moved the estimate (rebase, blend, or streak rebase)."""
    if not has_fix:
      return False
    if not (math.isfinite(lat) and math.isfinite(lon)):
      return False
    h_acc = MOONPILOT_POSE_MAX_HACC if horizontal_accuracy is None else float(horizontal_accuracy)
    if not math.isfinite(h_acc) or h_acc < 0.0 or h_acc > MOONPILOT_POSE_MAX_HACC:
      # Negative is nonsense; above the ceiling is untrusted. Zero falls through as unknown.
      return False

    # Course-over-ground is only steering-grade at speed; below that it is receiver noise.
    steer = (
      bearing_deg is not None and speed is not None
      and math.isfinite(bearing_deg) and math.isfinite(speed)
      and speed >= MOONPILOT_POSE_BEARING_MIN_SPEED
    )

    if self.lat0 is None:
      self.lat0, self.lon0 = lat, lon
      self.x = 0.0
      self.y = 0.0
      if steer and bearing_deg is not None:
        # GPS bearing is course-over-ground, degrees clockwise from true north; local yaw is
        # CCW from east.
        self.yaw = math.pi / 2.0 - math.radians(bearing_deg)
      self.n_gps = 1
      return True

    # Yaw is steered even when the position gate below rejects: a multipath jump moves the
    # reported position, not the course, and a wrong heading is what makes the radius trip.
    if steer and bearing_deg is not None:
      target = math.pi / 2.0 - math.radians(bearing_deg)
      self.yaw += MOONPILOT_POSE_YAW_ALPHA * _wrap_pi(target - self.yaw)

    lat0, lon0 = self.lat0, self.lon0
    if lat0 is None or lon0 is None:
      return False  # unreachable: the rebase above is the only path that leaves origin unset
    gx, gy = latlon_to_enu(lat, lon, lat0, lon0)
    if math.hypot(gx - self.x, gy - self.y) > MOONPILOT_POSE_GATE_RADIUS:
      self._reject_streak += 1
      if self._reject_streak < MOONPILOT_POSE_REBASE_STREAK:
        return False
      # Sustained disagreement: the dead-reckon lost it (wrong absolute yaw was the corpus
      # case). Trust the fix for position; yaw has been blending from course above.
      self.x, self.y = gx, gy
      self._reject_streak = 0
      self.n_gps += 1
      self.n_rebase += 1
      return True

    self._reject_streak = 0
    self.x += MOONPILOT_POSE_GPS_ALPHA * (gx - self.x)
    self.y += MOONPILOT_POSE_GPS_ALPHA * (gy - self.y)
    self.n_gps += 1
    return True
