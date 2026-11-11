"""Synthetic behavior checks for the GPS-gated pose tracker."""

import math
import unittest

from moonpilot.pose import (
  MOONPILOT_POSE_GATE_RADIUS,
  MOONPILOT_POSE_GPS_ALPHA,
  MOONPILOT_POSE_MAX_HACC,
  MOONPILOT_POSE_REBASE_STREAK,
  PoseTracker,
  fill_ego_pose,
  latlon_to_enu,
  read_ego_pose,
)

class TestPoseTracker(unittest.TestCase):
  def test_fill_ego_pose_is_invalid_until_the_first_fix_rebases(self):
    from openpilot.cereal import messaging

    t = PoseTracker()
    t.push_odom(1.0, v_forward=10.0, yaw_rate=0.0)
    msg = messaging.new_message("moonpilotState")
    fill_ego_pose(msg, t, 1.0)
    self.assertFalse(msg.moonpilotState.egoPose.valid)

    self.assertTrue(t.push_gps(43.7, -79.8, horizontal_accuracy=2.0))
    t.push_odom(1.05, v_forward=10.0, yaw_rate=0.0)
    fill_ego_pose(msg, t, 1.05)
    pose = msg.moonpilotState.egoPose
    self.assertTrue(pose.valid)
    self.assertEqual(pose.monoTime, int(1.05 * 1e9))
    self.assertAlmostEqual(pose.x, 0.5, delta=1e-6)
    self.assertAlmostEqual(pose.y, 0.0, delta=1e-6)
    self.assertAlmostEqual(pose.yaw, 0.0, delta=1e-6)
    self.assertEqual(read_ego_pose(msg.moonpilotState, int(1.05e9)), (pose.x, pose.y, pose.yaw))
    fill_ego_pose(msg, t, 1.6)
    self.assertFalse(pose.valid)
    self.assertIsNone(read_ego_pose(msg.moonpilotState, int(1.6e9)))

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
    self.assertTrue(t.push_gps(43.7, -79.8, horizontal_accuracy=2.0))
    self.assertTrue(t.has_origin)
    self.assertEqual((t.x, t.y), (0.0, 0.0))

    # A fix 5 m north of where VO put us pulls the estimate partway there.
    lat = 43.7 + math.degrees(5.0 / 6_378_137.0)
    self.assertTrue(t.push_gps(lat, -79.8, horizontal_accuracy=2.0))
    self.assertAlmostEqual(t.y, 5.0 * MOONPILOT_POSE_GPS_ALPHA, delta=1e-9)
    self.assertAlmostEqual(t.x, 0.0, delta=1e-9)

  def test_gated_fixes_do_not_move_the_estimate(self):
    t = PoseTracker()
    self.assertFalse(t.push_gps(43.7, -79.8, has_fix=False))
    self.assertFalse(t.has_origin)
    # Above the accuracy ceiling: unknown-but-huge is not a gate we trust either.
    self.assertFalse(t.push_gps(43.7, -79.8, horizontal_accuracy=MOONPILOT_POSE_MAX_HACC + 1.0))
    self.assertFalse(t.has_origin)
    # Negative accuracy is nonsense.
    self.assertFalse(t.push_gps(43.7, -79.8, horizontal_accuracy=-1.0))
    self.assertFalse(t.has_origin)

    # hacc == 0 is "unknown" on qcom (every corpus sample), and passes as hasFix-only.
    self.assertTrue(t.push_gps(43.7, -79.8, horizontal_accuracy=0.0))
    self.assertTrue(t.has_origin)

  def test_an_outlier_fix_is_rejected_by_radius(self):
    t = PoseTracker()
    self.assertTrue(t.push_gps(43.7, -79.8, horizontal_accuracy=2.0))
    # 100 m north: past the radius, so not a correction.
    lat = 43.7 + math.degrees(100.0 / 6_378_137.0)
    self.assertFalse(t.push_gps(lat, -79.8, horizontal_accuracy=2.0))
    self.assertEqual((t.x, t.y), (0.0, 0.0))
    # Just inside the radius still corrects.
    lat = 43.7 + math.degrees((MOONPILOT_POSE_GATE_RADIUS - 1.0) / 6_378_137.0)
    self.assertTrue(t.push_gps(lat, -79.8, horizontal_accuracy=2.0))

  def test_a_rejected_streak_rebases_instead_of_dying_in_the_gate(self):
    """Sustained >radius disagreement is a lost dead-reckon, not multipath forever."""
    t = PoseTracker()
    self.assertTrue(t.push_gps(43.7, -79.8, horizontal_accuracy=2.0))
    lat_far = 43.7 + math.degrees(100.0 / 6_378_137.0)
    for _ in range(MOONPILOT_POSE_REBASE_STREAK - 1):
      self.assertFalse(t.push_gps(lat_far, -79.8, horizontal_accuracy=2.0))
    self.assertTrue(t.push_gps(lat_far, -79.8, horizontal_accuracy=2.0))
    self.assertAlmostEqual(t.y, 100.0, delta=1e-6)
    # The streak resets: the next fix, now near the rebased position, soft-corrects.
    # A fix still near the original origin would be 95 m out and correctly rejected again.
    lat_ok = 43.7 + math.degrees(105.0 / 6_378_137.0)
    self.assertTrue(t.push_gps(lat_ok, -79.8, horizontal_accuracy=2.0))
    self.assertAlmostEqual(t.y, 100.0 + MOONPILOT_POSE_GPS_ALPHA * 5.0, delta=1e-6)

  def test_a_parked_bearing_does_not_steal_the_heading(self):
    t = PoseTracker()
    t.push_odom(0.0, v_forward=0.0, yaw_rate=0.0)
    t.yaw = 0.5  # already moving / turning before the fix
    self.assertTrue(t.push_gps(43.7, -79.8, bearing_deg=180.0, speed=0.1))
    self.assertAlmostEqual(t.yaw, 0.5)

  def test_a_moving_fix_seeds_yaw_from_course_over_ground(self):
    t = PoseTracker()
    # Course 90° = due east; local yaw is CCW from east, so 0.
    self.assertTrue(t.push_gps(43.7, -79.8, bearing_deg=90.0, speed=10.0))
    self.assertAlmostEqual(t.yaw, 0.0, delta=1e-9)
    # Course 0° = due north = +90° local.
    t2 = PoseTracker()
    self.assertTrue(t2.push_gps(43.7, -79.8, bearing_deg=0.0, speed=10.0))
    self.assertAlmostEqual(t2.yaw, math.pi / 2.0, delta=1e-9)

  def test_a_slow_first_fix_leaves_yaw_for_later_course_blends(self):
    """The corpus case: first fix under the speed gate, so yaw is not seeded."""
    t = PoseTracker()
    self.assertTrue(t.push_gps(43.7, -79.8, bearing_deg=214.0, speed=1.2))
    self.assertEqual(t.yaw, 0.0)  # not seeded — below MOONPILOT_POSE_BEARING_MIN_SPEED
    # Once moving, each fix blends toward course (bearing 0° = north = +π/2).
    self.assertTrue(t.push_gps(43.7, -79.8, bearing_deg=0.0, speed=10.0))
    self.assertGreater(t.yaw, 0.0)
    self.assertLess(t.yaw, math.pi / 2.0)

  def test_odom_gaps_are_skipped_not_leapt(self):
    t = PoseTracker()
    t.push_odom(0.0, v_forward=10.0, yaw_rate=0.0)
    # A 2 s hole: do not integrate 20 m through the gap.
    t.push_odom(2.0, v_forward=10.0, yaw_rate=0.0)
    self.assertAlmostEqual(t.x, 0.0, delta=1e-9)

  def test_a_fused_chain_beats_raw_dead_reckoning_under_scale_error(self):
    """Periodic GPS fixes correct a VO forward speed scale that runs 10 % fast."""
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
        fused.push_gps(lat0, lon, horizontal_accuracy=2.0)
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


if __name__ == "__main__":
  unittest.main()
