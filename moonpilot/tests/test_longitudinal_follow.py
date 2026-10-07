"""The lead-following policy: the positive side's authority, and the time gap it is given.

Split out of `test_longitudinal.py`, which is at the 120 KB lint gate and where these two subjects
were one file too many. The helpers are that file's, by the same convention `test_slam.py` and
`test_longitudinal_features.py` already use.
"""

import itertools
import math
import unittest
from unittest import mock

import numpy as np
from openpilot.cereal import custom, log

import moonpilot.longitudinal as longitudinal
from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_ARRIVAL_MIN_T,
  MOONPILOT_FAST_ACCEL_BP,
  MOONPILOT_FAST_ACCEL_V,
  MOONPILOT_K_GAP,
  MOONPILOT_K_V,
  MOONPILOT_LADDER_V_BP,
  MOONPILOT_OUT_OF_PATH_ENTER,
  MOONPILOT_OUT_OF_PATH_LEAVE,
  MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V,
  MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T,
  MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_CRAWL_REST,
  MOONPILOT_SHOULD_STOP_SPEED,
  MOONPILOT_SLOW_ACCEL_V,
  MOONPILOT_SOFT_STOP_HANDOFF_V,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_STOP_REST,
  MOONPILOT_T_FOLLOW,
  cruise_accel,
  lead_accel,
  policy,
  pos_authority,
)
from moonpilot.tests.test_longitudinal import DT_MDL, _cp, _fly, _gap_target, _inputs, _lead, _planner

Personality = log.LongitudinalPersonality


class TestFollowPolicy(unittest.TestCase):
  def test_fast_cruise_ask_needs_room_to_accelerate_then_match_a_lead(self):
    v_ego, v_cruise = 20.0, 25.0
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    cp = _cp()
    source = log.LongitudinalPlan.LongitudinalPlanSource
    slow = float(np.interp(v_ego, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V))
    fast = cruise_accel(v_ego, v_cruise, False, 0.0, cp, -0.3, True)

    def ask(leads):
      return policy(v_ego, leads, v_cruise, t_follow, False, None, 0.0, cp, -0.3, True)[0]

    self.assertGreater(fast, slow)
    self.assertEqual(ask([]), fast)
    self.assertEqual(ask([(source.lead0, 200.0, v_ego, 0.0)]), fast)
    self.assertEqual(ask([(source.lead0, 40.0, v_ego, 0.0)]), slow)
    self.assertEqual(ask([(source.lead0, 40.0, v_ego, 1.0)]), fast)  # opening runway
    self.assertTrue(slow < ask([(source.lead0, 65.0, v_ego, 0.0)]) < fast)
    # Same runway, more closing: below set speed, the arrival term may take a gentle catch-up while
    # coasting still has room to arrive. It remains below the slow rung, not the cruise fast rung.
    closing = ask([(source.lead0, 65.0, v_ego - 2.0, 0.0)])
    self.assertGreater(closing, lead_accel(v_ego, 65.0, v_ego - 2.0, 0.0, t_follow))
    self.assertLess(closing, slow)
    self.assertEqual(ask([(source.lead0, 200.0, v_ego, 0.0), (source.lead1, 40.0, v_ego, 0.0)]), slow)

    braking = (source.lead0, 20.0, 15.0, 0.0)
    self.assertEqual(ask([braking]), lead_accel(v_ego, 20.0, 15.0, 0.0, t_follow))

  def test_a_closing_approach_coasts_to_the_target_instead_of_holding_set_speed(self):
    """Route 000003e4 segments 7-8, flown closed loop: 16.1 m/s held at a 58 kph set speed for
    18.8 s while the gap to a 42 kph lead closed from 119 m to 34 m, the plan asking 0.0 m/s² the
    whole way, then -1.09 m/s² actual with the car 1.1 m inside the 17 m target while still closing,
    and 2.5 s of braking at the settled gap — the brake lights while following. The regulator's
    negative window does not open until 31.5 m there, TTC and the stopping floor are both past their
    admission gates by less than they ask, so cruise held set speed and nothing else spoke.

    The arrival term is what replaces it, and this is the before/after on the same harness: it coasts
    from the first frame, never asks deeper than the driver's own no-pedal coast, and lands on the
    target at a speed match instead of braking into it. The old numbers are the same law with the
    arrival term made inert (`arrival_accel` patched to ask nothing), which is how they were
    produced, so this test fails if the term stops being what does the work.
    """
    s = _fly(16.0, 58.0, 119.0, 11.7, lambda t: 0.0, 60.0)
    cmds, vs = s["cmd"], s["v"]
    target = 1.45 * 11.7

    self.assertLess(cmds[0], 0.0)  # coasting from the first frame, not from 32 m out
    self.assertGreater(s["peak"], -0.15)  # a coast, not a brake: -0.091 here, -0.8602 without the term
    self.assertGreater(min(cmds), -0.1)  # no frame of the 60 s approach asks for a real brake
    self.assertAlmostEqual(s["min_gap"], target, delta=0.5)  # lands on the target, 0.19 m inside at worst
    self.assertAlmostEqual(vs[-1], 11.7, delta=0.02)  # arrives at a speed match
    self.assertAlmostEqual(s["end_gap"], target, delta=0.5)
    self.assertFalse(s["contact"])
    with mock.patch.object(longitudinal, "arrival_accel", lambda *_: math.inf):
      old = _fly(16.0, 58.0, 119.0, 11.7, lambda t: 0.0, 60.0)
    first_coast = next(i for i, c in enumerate(old["cmd"]) if c < -0.05)
    self.assertGreater(first_coast * DT_MDL, 18.0)  # the old law waited 19.9 s to say anything
    self.assertLess(old["gap"][first_coast], 35.0)  # ...and said it at 32 m, i.e. late
    self.assertLess(old["peak"], -0.8)
    self.assertGreater(sum(1 for c in old["cmd"] if c < -0.1), 100)  # 174 frames of real braking

  def test_below_set_catch_up_closes_a_far_moving_gap_without_overspeed(self):
    """A slower moving lead far ahead may be approached on the slow rung, then coasted onto the
    follow target. The set speed and braking-lead guard remain hard boundaries."""
    s = _fly(12.0, 64.8, 120.0, 10.0, lambda t: 0.0, 70.0)
    with mock.patch.object(longitudinal, "MOONPILOT_CATCH_UP_COAST_FRACTION", 0.0):
      old = _fly(12.0, 64.8, 120.0, 10.0, lambda t: 0.0, 70.0)
    target = 1.45 * 10.0
    self.assertGreater(s["cmd"][0], 0.0)
    self.assertLess(old["cmd"][0], 0.0)
    self.assertLess(max(s["v"]), 64.8 / 3.6)
    self.assertGreater(s["peak"], -0.15)
    self.assertAlmostEqual(s["end_gap"], target, delta=0.5)
    self.assertGreater(old["end_gap"], target + 10.0)
    self.assertFalse(s["contact"])

    cp = _cp()
    source = log.LongitudinalPlan.LongitudinalPlanSource
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    for v_ego, v_cruise, a_lead in ((18.0, 18.0, 0.0), (12.0, 18.0, -0.6)):
      lead = [(source.lead0, 120.0, 10.0, a_lead)]
      with self.subTest(v_ego=v_ego, a_lead=a_lead):
        with mock.patch.object(longitudinal, "MOONPILOT_CATCH_UP_COAST_FRACTION", 0.0):
          before = policy(v_ego, lead, v_cruise, t_follow, False, None, 0.0, cp, -0.3, True)
        self.assertEqual(policy(v_ego, lead, v_cruise, t_follow, False, None, 0.0, cp, -0.3, True), before)

  def test_stopped_and_slow_lead_approaches_do_not_switch_at_speed_thresholds(self):
    cp = _cp()
    source = log.LongitudinalPlan.LongitudinalPlanSource.lead0
    for t_follow, v_set, a_lead in itertools.product(MOONPILOT_T_FOLLOW.values(), (8.0, 12.0), (-0.6, 0.0, 0.2)):
      for threshold in (MOONPILOT_SHOULD_STOP_SPEED, MOONPILOT_STOP_DISTANCE / t_follow, MOONPILOT_STOP_REST / t_follow):
        with self.subTest(t_follow=t_follow, v_set=v_set, a_lead=a_lead, threshold=threshold):
          asks = [
            policy(8.0, [(source, 80.0, threshold + delta, a_lead)], v_set, t_follow, False, None, 0.0, cp, -0.3, True)[0]
            for delta in (-0.001, 0.001)
          ]
          self.assertLess(abs(asks[1] - asks[0]), 0.005)

  def test_crossing_the_preferred_rest_gap_does_not_demand_a_hard_crawl_stop(self):
    # The route spent its preferred rest gap while still crawling. This is not an emergency:
    # the lead is over five meters away, but the old tenth-meter denominator demanded -2.31.
    asks = [lead_accel(0.68, gap, 0.0, 0.0, 1.45) for gap in (5.3, 5.2, 5.1)]
    self.assertTrue(all(-0.8 < ask < 0.0 for ask in asks))
    self.assertLess(max(abs(a - b) for a, b in zip(asks, asks[1:], strict=False)), 0.05)

  def test_a_stop_behind_a_stopped_lead_lands_softly_near_the_rest_gap(self):
    """Route 0000040e (126-138 s): a stopped lead first seen ~83 m out from 17 m/s, and the regulator's
    gap-deficit brake held -0.54..-0.77 m/s^2 to 0.3 m/s, so the car came to rest from -0.7. The soft
    landing ends the stop gently, near the rest gap, and once the brake eases it never deepens again —
    the handover from the approach plateau is not a second brake."""
    for v0, gap0 in ((12.0, 70.0), (17.0, 85.0)):
      with self.subTest(v0=v0, gap0=gap0):
        s = _fly(v0, v0 * 3.6 + 5, gap0, 0.0, lambda t: 0.0, 40.0)
        v, c = np.array(s["v"]), np.array(s["cmd"])
        last = int(np.flatnonzero(v > 0.1)[-1])
        self.assertGreater(c[last], -0.2)
        self.assertFalse(s["contact"])
        # 4.36 m from 17 m/s: the crawl cap spends the rest gap toward MOONPILOT_CRAWL_REST at the end.
        self.assertTrue(MOONPILOT_CRAWL_REST - 0.2 < s["end_gap"] < MOONPILOT_STOP_DISTANCE, s["end_gap"])
        # The approach, not the standstill handoff (`MOONPILOT_SOFT_STOP_HANDOFF_V`), where the regulator's
        # hold returns 0.05-0.07 m/s^2 deeper and `stopping` owns the brake: nothing re-deepens above it.
        braking = c[: int(np.flatnonzero(v > 2 * MOONPILOT_SOFT_STOP_HANDOFF_V)[-1]) + 1]
        eased = np.flatnonzero(braking - np.minimum.accumulate(braking) > 0.05)
        tail = braking[eased[0] :]
        self.assertLess(float(np.max(np.maximum.accumulate(tail) - tail)), 0.03)

  def test_crawl_softening_does_not_spend_a_close_stopped_leads_remaining_gap(self):
    # The time-only comfort floor drove a 0.68 m/s crawl through a lead initially 0.5 m away,
    # even on this instantaneous point-mass plant. Short-gap braking must retain its authority.
    for gap0, min_gap in ((0.5, 0.1), (0.8, 0.4)):
      with self.subTest(gap0=gap0):
        s = _fly(0.68, 108.0, gap0, 0.0, lambda t: 0.0, 10.0)
        self.assertGreater(s["min_gap"], min_gap)
        self.assertEqual(s["v"][-1], 0.0)

  def test_a_crawl_stop_spends_the_rest_gap_instead_of_braking_past_the_driver(self):
    """Route 00000414 at 448 s: crawling at 3.1 m/s 8.2 m behind a lead that stopped, the floor, soft
    stop and TTC all aimed 5.2-6 m back and the car pulled -2.77 m/s^2. The driver's own 13 such stops
    peaked at -1.31..-1.90 and rested 2.7-5.9 m back. Flown here the old law peaked -1.93 / -2.02 for a
    lead braking at -1.5 / -2.5; the crawl cap stays under the driver's ceiling and lands near 4 m."""
    for a_lead in (-1.0, -1.5, -2.5):
      with self.subTest(a_lead=a_lead):
        s = _fly(3.1, 40.0, 8.2, 2.5, lambda t, a=a_lead: a if t > 1.0 else 0.0, 15.0)
        self.assertGreater(s["peak"], -1.9)
        self.assertGreater(s["end_gap"], MOONPILOT_CRAWL_REST - 0.2)
        self.assertFalse(s["contact"])

  def test_a_launch_asks_for_what_the_car_delivers_not_what_the_feet_did(self):
    """Sweep of 434 cached segments: below 3 m/s the car lands 1.27x the fast-rung ask, and four
    separate routes show cmd 1.97 -> a 2.43..2.51. The tracking factor is what turns the measured
    rung into a setpoint the plant holds, and it must be the ask that shrinks — never the ladder.
    Exact factors only at the factor's own breakpoints; in between it interpolates."""
    for v0, expected_factor in ((0.0, 0.786), (8.0, 1.0), (20.0, 1.0)):
      with self.subTest(v0=v0):
        measured = float(np.interp(v0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
        asked = cruise_accel(v0, v0 + 5.0, False, 0.0, _cp(), -0.3, True)
        self.assertAlmostEqual(asked, measured * expected_factor, places=6)
    for v0 in (1.0, 3.0, 4.5, 12.0):
      measured = float(np.interp(v0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
      self.assertLessEqual(cruise_accel(v0, v0 + 5.0, False, 0.0, _cp(), -0.3, True), measured)
    # Above the band the ask *is* the measured rung, so a merge at speed keeps the authority the driver
    # measured; the taper after 3.5 m/s is the ladder's own, not the factor's.
    for v0 in (6.0, 9.0, 13.0):
      measured = float(np.interp(v0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
      self.assertAlmostEqual(cruise_accel(v0, v0 + 5.0, False, 0.0, _cp(), -0.3, True), measured, places=6)

  def test_an_accelerating_lead_retires_the_approach_brake(self):
    self.assertLess(lead_accel(18.0, 90.0, 12.5, 0.0, 1.45), 0.0)
    self.assertGreater(lead_accel(18.0, 90.0, 12.5, 1.0, 1.45), 0.0)

  def test_a_stopped_lead_uses_a_soft_time_shaped_approach(self):
    v_ego = 8.0
    for gap in (80.0, 100.0, 112.0):
      with self.subTest(gap=gap):
        s = _fly(v_ego, 108.0, gap, 0.0, lambda t: 0.0, 45.0)

        self.assertLess(s["cmd"][0], 0.0)
        self.assertGreater(s["peak"], -0.6)
        self.assertFalse(s["contact"])
        self.assertGreaterEqual(s["min_gap"], MOONPILOT_STOP_REST - 0.2)
        self.assertAlmostEqual(s["end_gap"], MOONPILOT_STOP_REST, delta=0.25)

  def test_the_positive_side_cannot_reach_the_cruise_rung_from_a_speed_deficit(self):
    """The reported fault, as an invariant rather than a number that will drift. The measurement and
    the A/B against upstream's MPC are on `MOONPILOT_POS_AUTH_V`; the shape is the claim: gentle
    where the old law was not, and still under the cruise ladder's rung. *Reaching* the rung is what
    turned a proportional response into a step - "slightly behind" and "far behind" were the same
    command."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    rung = float(np.interp(30.0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
    raw = MOONPILOT_K_GAP * 1.5 + MOONPILOT_K_V * 0.9  # the episode's spacing and speed error
    self.assertAlmostEqual(raw, 0.99, delta=0.01)
    self.assertGreater(raw, rung)  # the pre-gain law did reach it

    for v_ego, gap_err, v_err in ((24.2, 1.5, 0.9), (25.0, 0.0, 1.0), (30.0, 1.5, 0.9)):
      with self.subTest(v_ego=v_ego):
        a = lead_accel(v_ego, _gap_target(v_ego, t_follow) + gap_err, v_ego + v_err, 0.0, t_follow)
        self.assertAlmostEqual(a, pos_authority(v_ego) * (MOONPILOT_K_GAP * gap_err + MOONPILOT_K_V * v_err), delta=1e-9)
        self.assertLess(a, rung, "the positive side reached the cruise ladder's rung")
        if v_ego == 24.2:  # the measured state: upstream's own number, not merely a smaller one
          self.assertTrue(0.27 <= a <= 0.41, f"{a} is outside the MPC's reading there")

    # the saturation point moved out: 1.45 m/s of lag used to be exactly the rung
    self.assertLess(pos_authority(24.2) * raw, rung)
    self.assertLess(pos_authority(24.2) * MOONPILOT_K_V * 1.45, rung - 0.5)
    self.assertGreater(pos_authority(24.2) * MOONPILOT_K_V * 4.4, rung)

    # the cost, measured rather than left in a comment: one authority scales the positive region's
    # damping ratio by sqrt(f) and its period by 1/sqrt(f) - zeta 0.55 -> 0.33, period 11 s -> 19 s.
    # The closed-loop suite is what says the slower loop still settles.
    for v_ego, expected in ((10.0, 1.0), (20.0, 1.0), (30.0, 0.28)):
      f = pos_authority(v_ego)
      with self.subTest(v_ego=v_ego):
        self.assertAlmostEqual(f, expected, delta=1e-9)
        self.assertGreater(math.sqrt(f) * MOONPILOT_K_V / (2 * math.sqrt(MOONPILOT_K_GAP)), 0.25)
        self.assertLess(2 * math.pi / math.sqrt(f * MOONPILOT_K_GAP), 25.0)

  def test_the_multiplier_moves_the_positive_side_and_nothing_else(self):
    """A differential, not a duplicated copy of the arithmetic: the pre-gain law is the shipped one
    with the authority pinned at 1.0, so what the multiplier must not have touched is decided by the
    code. Below 20 m/s the output is bit-identical; above it the output is never higher anywhere and
    is exactly equal wherever the old law was braking, so the multiplier can only soften an
    acceleration and cannot have softened a brake.

    That half is not free, and the reason is on the constant: scaling the regulator's two *terms*
    moves states where they carry opposite signs toward zero, and a state that braked at the -1.0
    approach clamp read -0.84 under a 0.35 authority. An ego both too far back and too fast is
    exactly the case that has to brake."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    counts = [0, 0, 0]
    for v_ego, gap, v_lead, a_lead in itertools.product(
      (0.5, 12.0, 20.0, 24.0, 33.0),
      (5.0, 15.0, 45.0, 200.0),
      (0.0, None, 1.5, 6.0),
      (0.0, -2.0),
    ):
      v_lead = v_ego + v_lead if v_lead else 0.0
      with mock.patch("moonpilot.longitudinal.MOONPILOT_POS_AUTH_V_BP", [0.0]), mock.patch("moonpilot.longitudinal.MOONPILOT_POS_AUTH_V", [1.0]):
        old = lead_accel(v_ego, gap, v_lead, a_lead, t_follow)
      new = lead_accel(v_ego, gap, v_lead, a_lead, t_follow)
      state = (v_ego, gap, v_lead, a_lead)
      if v_ego <= 20.0:
        counts[0] += 1
        self.assertEqual(new, old, f"the pre-20 m/s law moved at {state}")
      else:
        counts[1] += 1
        self.assertLessEqual(new, old + 1e-12, f"the law got *stronger* at {state}")
        if old <= 0.0:
          counts[2] += 1
          self.assertEqual(new, old, f"a braking state changed at {state}")
    self.assertGreater(min(counts), 40)

  def test_the_out_of_path_headway_is_a_latched_claim_not_a_scale(self):
    """`inPath` at highway speed is a weak signal, and the measured table is on
    `MOONPILOT_OUT_OF_PATH_ENTER`. Mapping it linearly onto a time gap gave an in-lane lead a
    permanent 1.10-1.30 s headway where the personality asks 1.45, breathing with the estimate: a
    moving target under a proportional regulator, and the second half of the reported fault. So the
    reduction is a claim about the lead *leaving*, it takes strong evidence, and it latches. The
    failure direction is the safe one - inPath that says nothing leaves the car at its own headway."""
    base = MOONPILOT_T_FOLLOW[Personality.standard]
    reduced = base * MOONPILOT_OUT_OF_PATH_T_FOLLOW
    lead = _lead(35.0, 20.0)

    def at(in_path):
      traj = custom.MoonpilotState.LeadTrajectory.new_message()
      traj.present = True
      traj.inPath = float(in_path)
      return traj

    def reads(in_path, planner=None):
      return (planner or _planner())._t_follow(_inputs(lead=lead, moonpilot_leads=[at(in_path)]))

    # an ordinary in-lane lead - its measured median, and everything down to its p10 and through the
    # hysteresis band - is not evidence of leaving
    for in_path in (0.54, 0.30, 0.29, 0.25, 0.23, 0.15, 0.12, 0.11):
      self.assertAlmostEqual(reads(in_path), base, delta=1e-9)

    # positive evidence latches, and the latch then holds across the whole band
    planner = _planner()
    self.assertAlmostEqual(reads(0.0, planner), reduced, delta=1e-9)
    for in_path in (MOONPILOT_OUT_OF_PATH_ENTER, 0.20, MOONPILOT_OUT_OF_PATH_LEAVE - 0.01):
      self.assertAlmostEqual(reads(in_path, planner), reduced, delta=1e-9)
    released = [reads(MOONPILOT_OUT_OF_PATH_LEAVE + 0.05, planner) for _ in range(20)]
    self.assertGreater(released[0], reduced)
    self.assertAlmostEqual(released[-1], base, delta=1e-9)

    # the thresholds are ordered and sit inside the measured distribution, or the band is not a band
    self.assertLess(MOONPILOT_OUT_OF_PATH_ENTER, MOONPILOT_OUT_OF_PATH_LEAVE)
    self.assertLessEqual(MOONPILOT_OUT_OF_PATH_ENTER, 0.13)  # the adjacent-lane median
    self.assertGreaterEqual(MOONPILOT_OUT_OF_PATH_LEAVE, 0.23)  # the in-lane p10

  def test_the_gap_target_is_not_a_reason_to_brake_below_the_leads_speed(self):
    """The reported second half: at a set speed above the lead's, the ego's speed decayed and stayed
    decayed. Measured cause: `gap_target` is 1.45 s of headway, unreachable at whatever speed the
    lead chooses, so the gap term alone asked for the full approach brake — -1.00 at 24, 25 *and* 26
    m/s behind a lead holding 24 m/s, i.e. at exactly the lead's speed too, while `stopping_decel`
    read -0.00 to -0.08 in the same states. Nothing needed stopping for; the command was a headway
    the lead itself set, bought with the throttle shut.

    The floor keeps pace instead. Each of its four gates is a way it could be wrong, and each was
    found by a test rather than by argument: inert when the ego is the faster car (otherwise a fly
    crossed 7.5 m inside a 1 m cushion), off for a lead that is slowing (holding speed into a
    -3.5 m/s^2 brake makes the ego relatively faster and cost 2.5 m of end gap), off below 1.0 s of
    headway, and off at rest (where the headway test is vacuous and `v_lead_eff` clamps to zero, so
    the car would pull away from a stopped lead inside the standstill gap).
    """
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_lead = 24.0

    # the decay state, and the three gates that must not let it brake
    for v_ego, gap, expected in ((26.0, 30.0, -1.00), (25.0, 30.0, -1.00), (24.0, 30.0, 0.00), (23.5, 30.0, 0.30), (23.0, 28.0, 0.60)):
      with self.subTest(v_ego=v_ego, gap=gap):
        self.assertAlmostEqual(lead_accel(v_ego, gap, v_lead, 0.0, t_follow), expected, delta=1e-9)
    # a lead that is slowing keeps the brake, however the speeds compare
    for a_lead in (-0.1, -0.5, -3.5):
      with self.subTest(a_lead=a_lead):
        self.assertLess(lead_accel(24.0, 30.0, v_lead, a_lead, t_follow), 0.0)
    # a short headway keeps the brake, and so does a standstill
    for v_ego, gap in ((23.0, 10.0), (20.0, 8.0)):
      with self.subTest(v_ego=v_ego, gap=gap):
        self.assertLess(gap / v_ego, MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T)
        self.assertLess(lead_accel(v_ego, gap, v_lead + 1.0, 0.0, t_follow), 0.0)
    self.assertLess(lead_accel(0.0, 4.5, 0.4, -1.0, t_follow), 0.0)
    # the closing approach is untouched, and so is the surge the authority exists for
    self.assertLess(lead_accel(25.0, _gap_target(25.0, t_follow) - 1.0, v_lead, 0.0, t_follow), 0.0)
    rung = float(np.interp(30.0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
    self.assertLess(lead_accel(24.2, _gap_target(24.2, t_follow) + 1.5, 25.1, 0.0, t_follow), rung)

  def test_a_settled_follow_does_not_step_on_radar_noise(self):
    """Route 000003f8 segment 44: at a settled follow 0.6 m inside the target the lead's speed and
    accel straddle the floor's two gates by radar noise alone, and as switches they swapped the
    floor's ~0 for the gap term's -0.18 frame to frame. As fades the ask is continuous across both.
    The speed fade is also narrow: once the ego is `MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V` faster the
    floor is entirely gone — the same output a hard gate gives there — so a car that is really a
    little faster is still braked back out of the target rather than drifting inside it."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_ego = 17.4
    gap = t_follow * v_ego - 0.6
    for name, asks in (
      ("v_lead", [lead_accel(v_ego, gap, v_ego + dv, 0.0, t_follow) for dv in np.arange(-0.05, 0.0501, 0.005)]),
      ("a_lead", [lead_accel(v_ego, gap, v_ego, MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD + da, t_follow) for da in np.arange(-0.05, 0.0501, 0.005)]),
    ):
      with self.subTest(sweep=name):
        self.assertLess(max(abs(np.diff(asks))), 0.03)  # a hard gate steps 0.18 here
    for excess in (MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V, 2 * MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V):
      with mock.patch.object(longitudinal, "MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V", 1e-9):
        gated = lead_accel(v_ego, gap, v_ego - excess, 0.0, t_follow)
      with self.subTest(excess=excess):
        self.assertLess(gated, 0.0)
        self.assertEqual(lead_accel(v_ego, gap, v_ego - excess, 0.0, t_follow), gated)

  def test_the_arrival_term_neither_taps_at_the_target_nor_switches_on_a_far_leads_accel(self):
    """Two route 000003f8 faults of the arrival term, one per end of the approach.

    At the target (segment 44): the last meter at 0.2-0.45 m/s of closing read -1.00, -0.10, -0.34,
    -1.00 on consecutive frames, `-closing**2 / (2 * slack)` with both going to zero on dRel noise.
    With the slack floored at `MOONPILOT_ARRIVAL_MIN_T` of closing, the arrival ask anywhere inside
    `T * closing` of the target is `-closing / (2 T)` (inside the target the regulator's gap term is
    the deeper one), and jitter of the gap moves the output no faster than the regulator's gap gain.

    Far out (segment 42): a lead 60-120 m ahead read +0.1..+0.35 m/s^2 of slope noise, and a switch
    at +0.1 stepped the command between the term's ~-0.1 and the cruise rung. `arrival_accel` credits
    the lead's positive accel instead, so the ask rises with it without a step."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_lead = 17.0
    slacks = np.linspace(-0.3, 0.3, 13)
    for closing in (0.2, 0.3, 0.45):
      asks = [lead_accel(v_lead + closing, t_follow * v_lead + s, v_lead, 0.0, t_follow) for s in slacks]
      with self.subTest(closing=closing):
        self.assertGreater(min(asks), -0.5 * MOONPILOT_APPROACH_DECEL)
        self.assertLessEqual(max(abs(np.diff(asks))), MOONPILOT_K_GAP * (slacks[1] - slacks[0]) + 1e-9)
        self.assertLess(slacks[-1], closing * MOONPILOT_ARRIVAL_MIN_T + 1e-9)  # still on the floored slack
        self.assertAlmostEqual(asks[-1], -closing / (2.0 * MOONPILOT_ARRIVAL_MIN_T), delta=1e-6)
    asks = [lead_accel(25.0, 90.0, 22.0, a_lead, t_follow) for a_lead in np.arange(0.0, 0.401, 0.01)]
    self.assertLess(asks[0], 0.0)  # the arrival term is braking this approach
    steps = np.diff(asks)
    self.assertGreaterEqual(min(steps), 0.0)
    self.assertLess(max(steps), 0.03)  # a +0.1 switch steps ~4.8 here

  def test_a_far_radar_lead_outlives_a_vision_swap_and_a_dropout(self):
    """Route 000003f8 segment 42: a 14 m/s radar track 90-125 m out kept losing radard's association,
    and leadOne alternated with the model's vision-only lead (~21 m/s, ~20 m nearer) or with nothing,
    stepping the arrival command -0.5 <-> +0.1. The last radar lead stays a candidate for
    `MOONPILOT_RADAR_HOLD_T`, so neither a swap nor a dropout lifts the brake — and it is released
    once the hold runs out."""
    vision = _lead(80.0, 21.0)
    vision.radar = False
    absent = _lead(0.0, 0.0, present=False)

    def fly(swap, hold_t):
      with mock.patch.object(longitudinal, "MOONPILOT_RADAR_HOLD_T", hold_t):
        planner = _planner()
        for _ in range(40):
          planner.update(_inputs(v_ego=22.5, lead=_lead(100.0, 14.0)))
        braking = planner.output_a_target
        for _ in range(20):  # 1 s inside the hold
          planner.update(_inputs(v_ego=22.5, lead=swap))
        return braking, planner

    for name, swap in (("vision", vision), ("dropout", absent)):
      with self.subTest(swap=name):
        braking, held = fly(swap, longitudinal.MOONPILOT_RADAR_HOLD_T)
        _, released = fly(swap, 0.0)
        self.assertLess(braking, -0.2)  # the arrival term is braking this approach
        self.assertLess(held.output_a_target, braking + 0.05)
        self.assertGreater(released.output_a_target, braking + 0.5)  # what the swap did without the hold
    planner = fly(absent, longitudinal.MOONPILOT_RADAR_HOLD_T)[1]
    for _ in range(round(longitudinal.MOONPILOT_RADAR_HOLD_T / DT_MDL)):
      planner.update(_inputs(v_ego=22.5, lead=absent))
    self.assertEqual(planner.source, longitudinal.LongitudinalPlanSource.cruise)

  def test_throttle_returns_only_past_the_release_probability(self):
    """Route 000003f8 segment 43: the gas-press probability hovered on 0.4 and the coast cap flipped
    on and off a frame at a time. Throttle is lost under 0.4 at once and returns only once the
    probability has held over 0.5 for `MOONPILOT_ALLOW_THROTTLE_RETURN_T` — route 0000040e's swings
    (0.31 -> 0.53 -> 0.38 -> 0.66 inside 0.2 s) crossed both thresholds a frame at a time."""
    planner = _planner()
    hold = round(longitudinal.MOONPILOT_ALLOW_THROTTLE_RETURN_T / DT_MDL)
    for prob, allowed in ((0.45, True), (0.35, False), (0.45, False)):
      planner.update(_inputs(v_ego=20.0, throttle_prob=prob))
      self.assertEqual(planner.allow_throttle, allowed, prob)
    for _ in range(hold - 1):
      planner.update(_inputs(v_ego=20.0, throttle_prob=0.55))
      self.assertFalse(planner.allow_throttle)
    planner.update(_inputs(v_ego=20.0, throttle_prob=0.38))  # one dip restarts the hold
    for _ in range(hold - 1):
      planner.update(_inputs(v_ego=20.0, throttle_prob=0.66))
      self.assertFalse(planner.allow_throttle)
    planner.update(_inputs(v_ego=20.0, throttle_prob=0.66))
    self.assertTrue(planner.allow_throttle)
    planner.update(_inputs(v_ego=20.0, throttle_prob=0.45))
    self.assertTrue(planner.allow_throttle)
