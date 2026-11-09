import math
import pickle
import unittest

import numpy as np

from openpilot.common.realtime import DT_CTRL, DT_MDL

from moonpilot.lab.plant import PLANT_MIN_FRAMES, _PlantGain, _plant_table
from moonpilot.lab.probes import _scorecard_checks
from moonpilot.lab.driver import _pumps
from moonpilot.lab.planner import HOLD_FLY_DEAD_S, HOLD_FLY_TAU_S, _LaggedPlant, _launch_episodes, _rest_gaps, _shape_line, _split_routes, control_rate_dt


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

  def test_a_gain_scales_the_delayed_command(self):
    plant = _LaggedPlant(gain=2.0)
    a = [plant.step(1.0) for _ in range(20)]
    dead = round(HOLD_FLY_DEAD_S / DT_MDL)
    n = 20 - dead
    self.assertAlmostEqual(a[-1], 2.0 * (1.0 - math.exp(-n * DT_MDL / HOLD_FLY_TAU_S)), places=6)

  def test_the_identified_gain_is_picklable_and_selects_by_sign(self):
    table = {"accel": np.full(5, 1.25), "brake": np.full(5, 0.90)}
    gain = pickle.loads(pickle.dumps(_PlantGain(table)))
    self.assertAlmostEqual(float(gain(1.0, 1.0)), 1.25, places=6)
    self.assertAlmostEqual(float(gain(-1.0, 1.0)), 0.90, places=6)
    np.testing.assert_allclose(gain(np.array([1.0, -1.0, 0.0]), 1.0), [1.25, 0.90, 1.0])


class TestPlantTable(unittest.TestCase):
  """The gain table is built from steady frames only, by sign and speed band, and leaves engine-coast braking out."""

  @staticmethod
  def _result(ratio, sign, ec=0, frames=PLANT_MIN_FRAMES):
    v = np.concatenate([np.full(frames, (lo + hi) / 2) for lo, hi in ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 40.0))])
    cmd = np.full(len(v), 1.0 if sign > 0 else -1.0)
    return "r", {"lag": 0.3, "cmd": cmd, "a": cmd * ratio, "v": v, "steady": np.ones(len(v), dtype=bool), "ec": ec}

  def test_it_medians_by_sign_and_band_and_excludes_engine_coast_braking(self):
    table, counts = _plant_table([self._result(1.25, 1), self._result(1.10, -1), self._result(2.0, -1, ec=1)])
    self.assertAlmostEqual(table["accel"][0], 1.25, places=6)
    self.assertAlmostEqual(table["brake"][0], 1.10, places=6)
    self.assertEqual(counts[("accel", 0.0, 3.0)], PLANT_MIN_FRAMES)
    self.assertEqual(counts[("brake", 0.0, 3.0)], PLANT_MIN_FRAMES)

  def test_a_thin_band_falls_back_to_the_unity_gain(self):
    table, counts = _plant_table([self._result(1.5, 1, frames=PLANT_MIN_FRAMES - 1)])
    self.assertAlmostEqual(table["accel"][0], 1.0, places=6)
    self.assertEqual(counts[("accel", 0.0, 3.0)], PLANT_MIN_FRAMES - 1)


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

  def test_a_lead_that_appeared_under_a_second_before_its_launch_has_not_settled(self):
    for kind in ("track", "route"):
      with self.subTest(kind=kind):
        self.assertEqual(len(_launch(break_at_s=-0.5, break_kind=kind)), 0)
        self.assertEqual(len(_launch(break_at_s=-1.5, break_kind=kind)), 1)  # a second and more on it is a launch


class TestScorecardSplit(unittest.TestCase):
  """The scorecard's held-out split: deterministic, disjoint, complete, and never one-sided."""

  ROUTES = [f"0000{i:04x}--deadbeef" for i in range(86)]

  def test_it_is_deterministic_disjoint_and_complete(self):
    train, held = _split_routes(self.ROUTES, 0.25, seed=7)
    self.assertEqual(sorted(train + held), sorted(self.ROUTES))
    self.assertFalse(set(train) & set(held))
    self.assertEqual((train, held), _split_routes(self.ROUTES, 0.25, seed=7))
    self.assertGreater(len(train), 0)
    self.assertGreater(len(held), 0)

  def test_a_different_seed_moves_routes(self):
    self.assertNotEqual(_split_routes(self.ROUTES, 0.25, seed=1)[1], _split_routes(self.ROUTES, 0.25, seed=2)[1])

  def test_an_invalid_fraction_or_a_one_sided_split_is_refused(self):
    for holdout in (0.0, 1.0, -0.1, 1.1):
      with self.subTest(holdout=holdout), self.assertRaises(ValueError):
        _split_routes(self.ROUTES, holdout)
    with self.assertRaises(ValueError):
      _split_routes(["000003f8--deadbeef"], 0.5)

  def test_the_shape_line_ignores_and_counts_a_no_brake_flight(self):
    line = _shape_line([(math.nan, math.nan, math.nan, math.nan, 4.0, 2), (-1.5, 0.5, -0.6, -0.8, 3.0, 1)])
    self.assertTrue("peak -1.50" in line)
    self.assertTrue("thirds -0.60/-0.80" in line)
    self.assertTrue("1 no brake" in line)

  def test_stop_pumps_are_counted_on_stop_flights_only(self):
    def flight(cmd):
      return {"a": np.zeros(len(cmd)), "v": np.full(len(cmd), 10.0), "gap": np.full(len(cmd), 5.0), "cmd": np.asarray(cmd, dtype=float)}
    stop = flight([-1.0, -0.8, -1.0] * 4)  # long enough for the 0.5 s mean, two pumps
    slow = flight([-1.0, -0.8] * 10)  # many pumps, but not a stop
    results = {"train": {"decels": {"drv": np.asarray([[10.0, -1.0, 5.0]]), "flown": [stop] * 10 + [slow],
                                      "stops": np.asarray([True] * 10 + [False])}}}
    detail = next(d for n, _, d in _scorecard_checks(results, []) if n == "train stop pumps")
    self.assertTrue(f"{float(_pumps(stop['cmd'])):.2f} per stop ({_pumps(stop['cmd']) * 10}/10)" in detail)


if __name__ == "__main__":
  unittest.main()
