import math
import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL

from moonpilot.lab.planner import HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, _LaggedPlant, _launch_episodes, _rest_gaps, control_rate_dt


class TestControlRateGuard(unittest.TestCase):
  """The acceleration controller advances per call, so a replay has to tick it at its own rate.

  A 20 Hz feed is one fifth the ramp per second of a 100 Hz one: the harness must refuse to print a
  controller number rather than quietly reporting the wrong controller.
  """

  def test_accepts_the_control_rate(self):
    t = np.arange(0, 3_000_000_000, int(DT_CTRL * 1e9))
    self.assertAlmostEqual(control_rate_dt(t), DT_CTRL, places=6)

  def test_refuses_a_planner_rate_feed(self):
    t = np.arange(0, 3_000_000_000, int(0.05 * 1e9))  # 20 Hz, the planner's tick rate
    with self.assertRaises(ValueError):
      control_rate_dt(t)

  def test_refuses_too_few_frames(self):
    with self.assertRaises(ValueError):
      control_rate_dt(np.arange(0, int(0.05 * 1e9), int(DT_CTRL * 1e9)))


class TestLaggedPlant(unittest.TestCase):
  def test_a_step_waits_the_dead_time_then_rises_on_the_lag(self):
    """The flights' `0.2 s dead time + 0.25 s lag` is the plant's, to the frame: a queue appended before it is read
    delivered the step one frame early (0.15 s)."""
    plant = _LaggedPlant()
    a = [plant.step(1.0) for _ in range(20)]
    dead = round(HOLD_FLY_DEAD_S / DT_MDL)
    self.assertEqual(a[:dead], [0.0] * dead)
    self.assertGreater(a[dead], 0.0)
    self.assertAlmostEqual(a[dead - 1 + round(HOLD_FLY_TAU_S / DT_MDL)], 1.0 - math.exp(-1.0), places=9)


def _frames(*pieces):
  """Concatenated (frames, route, v, lead present, gap) pieces, the lead stopped throughout."""
  route, v, lead, gap = (np.concatenate([np.full(p[0], p[i]) for p in pieces]) for i in range(1, 5))
  return route, v, lead.astype(bool), np.zeros(len(v)), gap


class TestRestGaps(unittest.TestCase):
  def test_a_radar_dropout_during_one_standstill_is_one_stop(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 5.0), (5, 0, 0.0, False, 5.0), (40, 0, 0.0, True, 5.5)))
    self.assertEqual(gaps.tolist(), [5.0])

  def test_the_lead_may_be_read_just_after_the_stop_frame(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (10, 0, 0.0, False, 0.0), (40, 0, 0.0, True, 4.0)))
    self.assertEqual(gaps.tolist(), [4.0])

  def test_a_creep_restop_is_the_same_stop_and_only_real_motion_rearms(self):
    creep = _frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 6.0), (20, 0, 1.0, True, 5.0), (20, 0, 0.0, True, 4.0))
    self.assertEqual(_rest_gaps(*creep).tolist(), [6.0])
    again = _frames((40, 0, 8.0, True, 30.0), (20, 0, 0.0, True, 6.0), (20, 0, 3.0, True, 9.0), (20, 0, 0.0, True, 4.0))
    self.assertEqual(_rest_gaps(*again).tolist(), [6.0, 4.0])

  def test_motion_in_another_route_does_not_arm_a_stop(self):
    gaps = _rest_gaps(*_frames((40, 0, 8.0, True, 30.0), (40, 1, 0.0, True, 5.0)))
    self.assertEqual(gaps.tolist(), [])


def _launch(release_s=1.0, break_at_s=None, break_kind="track", total_s=14.0):
  """3 s at rest 4 m behind a stopped lead, which then launches at 1 m/s^2; the driver releases the brake `release_s` in
  (None: never), presses the gas 0.5 s later and the car pulls away 0.3 s after that at 1 m/s^2. `break_at_s` after the
  onset the lead's track id or the drive changes."""
  n, onset = round(total_s / DT_MDL), round(3.0 / DT_MDL)
  t = np.arange(n) * DT_MDL
  after = np.maximum(t - t[onset], 0.0)
  v_lead = after * 1.0
  rel = math.inf if release_s is None else release_s
  v = np.maximum(after - rel - 0.8, 0.0) * 1.0
  gap = 4.0 + np.cumsum(v_lead - v) * DT_MDL
  d = {"t": t, "manual": np.ones(n), "v": v, "brake": (after < rel).astype(float), "gas": (after >= rel + 0.5).astype(float),
       "lead": np.ones(n), "lead_radar": np.ones(n), "gap": gap, "v_lead": v_lead, "track_id": np.full(n, 7.0), "route": np.zeros(n)}
  if break_at_s is not None:
    d["track_id" if break_kind == "track" else "route"][onset + round(break_at_s / DT_MDL):] = 8.0
  return _launch_episodes(d)


class TestLaunchWindow(unittest.TestCase):
  """`launch-reaction` reads each launch only while it is one drive and one lead: a marker past a lost track or a route
  boundary is censored, never read off the next lead, and only a full 6 s without one reads as 'never'. Times run from the
  lead's onset as the lab defines it, its last frame under 0.2 m/s: here 0.15 s after it starts at 1 m/s^2 (`ONSET`)."""
  ONSET = 0.15

  def test_a_clean_launch_times_every_marker(self):
    (row,) = _launch()
    _, _, held, release, gas, moving, ego_a, lead_travel, _ = row
    self.assertTrue(held)
    self.assertAlmostEqual(release, 1.0 - self.ONSET, delta=DT_MDL)
    self.assertAlmostEqual(gas, 1.5 - self.ONSET, delta=DT_MDL)
    self.assertAlmostEqual(lead_travel, 0.5 * (1.0**2 - self.ONSET**2), delta=0.06)  # the lead's own, not the gap's
    self.assertTrue(math.isfinite(moving) and math.isfinite(ego_a))

  def test_a_lost_track_or_drive_censors_what_comes_after_it(self):
    for kind in ("track", "route"):
      with self.subTest(kind=kind):
        (row,) = _launch(break_at_s=2.5, break_kind=kind)
        _, _, _, release, _, moving, ego_a, _, _ = row
        self.assertAlmostEqual(release, 1.0 - self.ONSET, delta=DT_MDL)  # inside the window
        self.assertTrue(math.isnan(ego_a))  # its 2 s would cross the break
        (row,) = _launch(release_s=None, break_at_s=3.0, break_kind=kind)
        self.assertTrue(math.isnan(row[3]))  # never released before the break: unknown, not 'never'

  def test_never_needs_the_whole_six_seconds(self):
    (row,) = _launch(release_s=None)
    self.assertEqual(row[3], math.inf)

  def test_a_window_shorter_than_the_lead_launch_is_dropped(self):
    self.assertEqual(len(_launch(break_at_s=1.0)), 0)


if __name__ == "__main__":
  unittest.main()
