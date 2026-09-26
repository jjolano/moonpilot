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
from openpilot.common.constants import CV

from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_FAST_ACCEL_BP,
  MOONPILOT_FAST_ACCEL_V,
  MOONPILOT_K_GAP,
  MOONPILOT_K_V,
  MOONPILOT_OUT_OF_PATH_ENTER,
  MOONPILOT_OUT_OF_PATH_LEAVE,
  MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_T_FOLLOW,
  lead_accel,
  pos_authority,
)
from moonpilot.tests.test_longitudinal import _gap_target, _inputs, _lead, _planner

Personality = log.LongitudinalPersonality


class TestFollowPolicy(unittest.TestCase):
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
      (0.5, 12.0, 20.0, 24.0, 33.0), (5.0, 15.0, 45.0, 200.0), (0.0, None, 1.5, 6.0), (0.0, -2.0),
    ):
      v_lead = v_ego + v_lead if v_lead else 0.0
      with mock.patch("moonpilot.longitudinal.MOONPILOT_POS_AUTH_V_BP", [0.0]), \
           mock.patch("moonpilot.longitudinal.MOONPILOT_POS_AUTH_V", [1.0]):
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
    for v_ego, gap, expected in ((26.0, 30.0, -1.00), (25.0, 30.0, -1.00), (24.0, 30.0, 0.00),
                                 (23.5, 30.0, 0.30), (23.0, 28.0, 0.60)):
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
