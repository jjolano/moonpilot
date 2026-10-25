"""The lead-following policy: the cruise slot's runway cap, the lead law's closed-loop outcomes, and the time gap it is given.

Split out of `test_longitudinal.py`, which is at the 120 KB lint gate and where these two subjects
were one file too many. The helpers are that file's, by the same convention `test_slam.py` and
`test_longitudinal_features.py` already use.
"""

import math
import unittest
from unittest import mock

import numpy as np
from openpilot.cereal import custom, log

import moonpilot.longitudinal as longitudinal
from moonpilot.brake_band import MOONPILOT_BRAKE_BAND_ENTER_LEAD
from moonpilot.longitudinal import (
  MOONPILOT_BRAKE_EXIT_JERK,
  MOONPILOT_FAST_ACCEL_BP,
  MOONPILOT_FAST_ACCEL_V,
  MOONPILOT_FAST_CRAWL,
  MOONPILOT_LADDER_V_BP,
  MOONPILOT_MIN_SLACK,
  MOONPILOT_OUT_OF_PATH_ENTER,
  MOONPILOT_OUT_OF_PATH_LEAVE,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_SLOW_ACCEL_V,
  MOONPILOT_T_FOLLOW,
  cruise_accel,
  lead_accel,
  policy,
)
from moonpilot.tests.test_longitudinal import DT_MDL, _cp, _fly, _inputs, _lead, _planner

Personality = log.LongitudinalPersonality


class TestFollowPolicy(unittest.TestCase):
  def test_fast_cruise_ask_needs_room_to_accelerate_then_match_a_lead(self):
    """The cruise slot's runway cap, alone: with the lead law opened up, so what is measured is the cap and not
    the lead's own candidate."""
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
    with mock.patch.object(longitudinal, "lead_accel", lambda *_, **__: math.inf):
      self.assertEqual(ask([(source.lead0, 400.0, v_ego, 0.0)]), fast)  # runway to spare: the fast rung stands
      self.assertLessEqual(ask([(source.lead0, 40.0, v_ego, 0.0)]), slow)  # no runway to reach set speed and match
      gaps = np.linspace(40.0, 400.0, 73)
      asks = [ask([(source.lead0, g, v_ego, 0.0)]) for g in gaps]
      self.assertGreaterEqual(min(np.diff(asks)), 0.0)  # more runway never asks for less
      self.assertLess(max(np.diff(asks)), 0.1)  # and opens as a fade, not a step
      # a lead that is not a reason to hold back leaves the rung alone, near or not
      self.assertEqual(ask([(source.lead0, 40.0, v_ego, MOONPILOT_FAST_CRAWL)]), fast)  # pulling away
      self.assertEqual(ask([(source.lead0, 40.0, v_cruise + 1.0, 0.0)]), fast)  # already above set speed
      self.assertEqual(ask([(source.lead0, 400.0, v_ego, 0.0), (source.lead1, 40.0, v_ego, 0.0)]), ask([(source.lead0, 40.0, v_ego, 0.0)]))

    # a braking lead's own candidate is what binds, whatever the cruise rung would have asked
    braking = (source.lead0, 20.0, 15.0, 0.0)
    self.assertEqual(ask([braking]), lead_accel(v_ego, 20.0, 15.0, 0.0, t_follow))
    self.assertLess(ask([braking]), slow)

  def test_a_slow_lead_that_stops_later_finishes_without_contact(self):
    """A lead that slows to 3 m/s and then stops is followed to rest with a gap left."""
    s = _fly(12.0, 43.2, 80.0, 3.0, lambda t: -1.5 if t > 4.0 else 0.0, 50.0)
    self.assertFalse(s["contact"])
    self.assertGreater(s["min_gap"], MOONPILOT_MIN_SLACK)
    self.assertLess(s["v"][-1], 0.05)

  def test_crawl_softening_does_not_spend_a_close_stopped_leads_remaining_gap(self):
    # A 0.68 m/s crawl toward a lead 0.5 m away, even on this instantaneous point-mass plant: short-gap
    # braking must retain its authority inside the last meter.
    for gap0, min_gap in ((0.5, 0.1), (0.8, 0.4)):
      with self.subTest(gap0=gap0):
        s = _fly(0.68, 108.0, gap0, 0.0, lambda t: 0.0, 10.0)
        self.assertGreater(s["min_gap"], min_gap)
        self.assertEqual(s["v"][-1], 0.0)

  def test_a_departing_lead_lifts_the_models_stop_hold_once_the_model_asks_to_go(self):
    """Route 00000414 at 472 s: the model's stop hold, latched on the approach, kept the plan at
    +0.04..+0.09 — under `should_stop`'s 0.1 — for 1.8 s while the lead pulled away, until the driver
    pressed the gas. A positive model ask with leadOne moving now hands the slot back to the lead; a
    model still asking to hold (a light the lead drove through) keeps it. The lead's 0.7 m/s is the
    departure the lead law answers at the standstill gap (0.14 m/s^2): slower than `MOONPILOT_LEAVE_DEADBAND`
    plus the band's own room and the car is right to sit, and it moves once the gap opens."""
    for model_go, held in ((0.05, False), (0.0, True)):
      with self.subTest(model_go=model_go):
        planner = _planner()
        planner.update(_inputs(v_ego=8.0, v_cruise_kph=62.0, model_accel=-1.5, lead=_lead(30.0, 0.0)))
        for _ in range(40):
          planner.update(_inputs(v_ego=0.0, v_cruise_kph=62.0, model_accel=-0.05, lead=_lead(5.2, 0.0), standstill=True))
        self.assertTrue(planner.output_should_stop)
        for _ in range(4):
          planner.update(_inputs(v_ego=0.0, v_cruise_kph=62.0, model_accel=model_go, lead=_lead(5.4, 0.7), standstill=True))
        self.assertEqual(planner.output_should_stop, held)
        self.assertEqual(planner.output_a_target > 0.1, not held)

  def test_speeding_up_after_a_brake_rebuilds_at_the_drivers_pace(self):
    """Rolling out of a brake (routes 00000410 at 7.6 m/s, 00000414 at 11.5 m/s) the cruise rung came back
    at the 1.5 m/s^3 comfort jerk and peaked 1.5 / 1.21 m/s^2; over 210 manual brake-then-gas episodes
    on 31 routes the driver rose at a median 0.50 at 5-10 m/s and 0.30 at 10-15. Above the launch band
    the positive ask now rebuilds at `MOONPILOT_BRAKE_EXIT_JERK`; inside it, a launch keeps its jerk."""
    for v_ego, exit_ramp in ((10.0, True), (2.0, False)):
      with self.subTest(v_ego=v_ego):
        planner = _planner()
        for _ in range(20):
          planner.update(_inputs(v_ego=v_ego, v_cruise_kph=62.0, model_accel=-1.5))
        cmds = []
        for _ in range(60):
          planner.update(_inputs(v_ego=v_ego, v_cruise_kph=62.0, model_accel=0.3))  # the model asks to go
          cmds.append(planner.output_a_target)
        release = next(i for i, c in enumerate(cmds) if c >= -0.5)
        ramp = [MOONPILOT_BRAKE_EXIT_JERK * (k - release) * DT_MDL for k in range(len(cmds))]
        if exit_ramp:
          self.assertTrue(all(c <= r + 1e-9 for c, r in zip(cmds[release:], ramp[release:], strict=True)))
          self.assertGreater(cmds[-1], 0.5)  # it does come back
        else:
          self.assertGreater(max(c - r for c, r in zip(cmds[release:], ramp[release:], strict=True)), 0.2)

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

  def test_a_far_radar_lead_outlives_a_vision_swap_and_a_dropout(self):
    """Route 000003f8 segment 42: a 14 m/s radar track kept losing radard's association, and
    leadOne alternated with the model's vision-only lead (~21 m/s, nearer) or with nothing, stepping
    the closing command -0.5 <-> +0.1. The last radar lead stays a candidate for
    `MOONPILOT_RADAR_HOLD_T`, so neither a swap nor a dropout lifts the brake — and it is released
    once the hold runs out. Flown at 70 m behind a 14 m/s lead at 22.5 m/s, so the lead law brakes."""
    vision = _lead(80.0, 21.0)
    vision.radar = False
    absent = _lead(0.0, 0.0, present=False)

    def fly(swap, hold_t):
      with mock.patch.object(longitudinal, "MOONPILOT_RADAR_HOLD_T", hold_t):
        planner = _planner()
        for _ in range(40):
          planner.update(_inputs(v_ego=22.5, lead=_lead(70.0, 14.0)))
        braking = planner.output_a_target
        for _ in range(20):  # 1 s inside the hold
          planner.update(_inputs(v_ego=22.5, lead=swap))
        return braking, planner

    for name, swap in (("vision", vision), ("dropout", absent)):
      with self.subTest(swap=name):
        braking, held = fly(swap, longitudinal.MOONPILOT_RADAR_HOLD_T)
        _, released = fly(swap, 0.0)
        self.assertLess(braking, -0.2)  # the lead law is braking this approach
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

  def test_a_settled_follow_does_not_step_on_radar_noise(self):
    """Closed loop, on the near edge of the band where the gap noise straddles it: the lead's gap and speed
    arrive as white radar noise, the lead's accel estimate wanders with it, and the command stays a steady
    coast that never asks for a press (the brake band's shallowest entry) — the settled follow does not
    swing between a brake and a release."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    for seed in (0, 1):
      with self.subTest(seed=seed):
        rng = np.random.default_rng(seed)
        planner = _planner()
        v, gap, a, cmds = 20.0, t_follow * 20.0, 0.0, []
        for _ in range(round(40.0 / DT_MDL)):
          lead = _lead(gap + rng.normal(0.0, 0.3), 20.0 + rng.normal(0.0, 0.05))
          planner.update(_inputs(v_ego=v, v_cruise_kph=108.0, a_ego=a, lead=lead))
          a = float(planner.output_a_target)
          cmds.append(a)
          v += a * DT_MDL
          gap += (20.0 - v) * DT_MDL
        cmds = np.array(cmds)
        window = round(1.0 / DT_MDL)
        about_mean = (cmds - np.convolve(cmds, np.ones(window) / window, mode="same"))[window:-window]
        self.assertLess(float(np.std(about_mean)), 0.02)
        self.assertGreater(float(cmds.min()), MOONPILOT_BRAKE_BAND_ENTER_LEAD)

  def test_a_slowing_lead_is_matched_without_contact_and_without_pumping(self):
    """A lead that slows from 20 to 15 m/s at -1.0 m/s^2 while followed at the time gap is matched, and once the
    command eases it does not deepen again: the brake is one shape, not a pulse train."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    for extra_gap in (0.0, 10.0):
      with self.subTest(extra_gap=extra_gap):
        s = _fly(20.0, 108.0, t_follow * 20.0 + extra_gap, 20.0, lambda t: -1.0 if 1.0 <= t < 6.0 else 0.0, 40.0)
        self.assertFalse(s["contact"])
        self.assertAlmostEqual(s["v"][-1], 15.0, delta=0.2)
        tail = np.array(s["cmd"][int(np.argmin(s["cmd"])) :])
        redeepen = max(tail[i] - tail[i + 1 : i + 1 + round(2.0 / DT_MDL)].min() for i in range(len(tail) - round(2.0 / DT_MDL) - 1))
        self.assertLess(redeepen, 0.15)
