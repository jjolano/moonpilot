"""The fork longitudinal planner: the policy's numbers, and the properties the seam depends on.

The last test is the load-bearing one — it runs upstream's own maneuver suite against this planner,
which is the end-to-end statement that the fork strategy drives the 15 stock scenarios without a
crash, without stalling at a stop, and while still decelerating under forceDecel.
"""

import itertools
import math
import unittest
from typing import cast
from unittest import mock

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.common.params import Params
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
import openpilot.selfdrive.test.longitudinal_maneuvers.plant as plant_mod
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers

from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_JERK_EMERGENCY,
  MOONPILOT_JERK_UP,
  MOONPILOT_K_GAP,
  MOONPILOT_K_V,
  MOONPILOT_MIN_SLACK,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_T_FOLLOW,
  MoonpilotLongitudinalPlanner,
  cruise_accel,
  lead_accel,
  moonpilot_longitudinal_planner,
  policy,
)

Personality = log.LongitudinalPersonality
Source = log.LongitudinalPlan.LongitudinalPlanSource


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store."""

  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


def _cp(car=CAR.HONDA_CIVIC, openpilot_longitudinal=True):
  CP = CarInterface.get_non_essential_params(car)
  CP.openpilotLongitudinalControl = openpilot_longitudinal
  return CP


def _lead(d_rel, v_lead, a_lead=0.0, a_lead_tau=1.5, model_prob=1.0, present=True):
  lead = log.RadarState.LeadData.new_message()
  lead.present = present
  lead.dRel = float(d_rel)
  lead.vRel = float(v_lead)
  lead.vLead = float(v_lead)
  lead.vLeadK = float(v_lead)
  lead.aLeadK = float(a_lead)
  lead.aLeadTau = float(a_lead_tau)
  lead.modelProb = float(model_prob)
  lead.radar = True
  return lead


class SubMaster(dict):
  """A duck-typed SubMaster: the planner only ever indexes with sm['key'], which is also why the
  maneuver harness can hand it a plain dict. valid/alive carry the moonpilotState probe that
  moonpilot/lead.py makes, logMonoTime and all_checks() what publish() reads."""

  def __init__(self, data, moonpilot_leads=None):
    super().__init__(data)
    moonpilot = custom.MoonpilotState.new_message()
    if moonpilot_leads is not None:
      moonpilot.leads = moonpilot_leads
    self["moonpilotState"] = moonpilot
    self.valid = {"moonpilotState": moonpilot_leads is not None}
    self.alive = {"moonpilotState": moonpilot_leads is not None}
    self.logMonoTime = {"modelV2": 1e9}

  def all_checks(self, *services):
    return True


def _inputs(
  v_ego=20.0,
  v_cruise_kph=108.0,
  a_ego=0.0,
  lead=None,
  lead_two=None,
  standstill=None,
  personality=Personality.standard,
  experimental=False,
  enabled=True,
  force_decel=False,
  throttle_prob=1.0,
  pitch=0.0,
  steer_angle_deg=0.0,
  model_accel=0.0,
  model_should_stop=False,
  moonpilot_leads=None,
):
  car_state = messaging.new_message("carState")
  car_state.carState.vEgo = float(v_ego)
  car_state.carState.aEgo = float(a_ego)
  car_state.carState.vCruise = float(v_cruise_kph)
  car_state.carState.standstill = bool(v_ego < 0.01) if standstill is None else bool(standstill)
  car_state.carState.steeringAngleDeg = float(steer_angle_deg)

  car_control = messaging.new_message("carControl")
  car_control.carControl.orientationNED = [0.0, float(pitch), 0.0]

  controls_state = messaging.new_message("controlsState")
  controls_state.controlsState.forceDecel = bool(force_decel)
  controls_state.controlsState.longControlState = car.CarControl.Actuators.LongControlState.pid

  selfdrive_state = messaging.new_message("selfdriveState")
  selfdrive_state.selfdriveState.experimentalMode = bool(experimental)
  selfdrive_state.selfdriveState.enabled = bool(enabled)
  selfdrive_state.selfdriveState.personality = personality

  vehicle_parameters = messaging.new_message("vehicleParameters")
  vehicle_parameters.vehicleParameters.angleOffsetDeg = 0.0

  model = messaging.new_message("modelV2")
  model.modelV2.meta.disengagePredictions.gasPressProbs = [float(throttle_prob)] * 6
  model.modelV2.action.desiredAcceleration = float(model_accel)
  model.modelV2.action.shouldStop = bool(model_should_stop)

  radar = messaging.new_message("radarState")
  radar.radarState.leadOne = lead if lead is not None else _lead(200.0, float(v_ego), present=False)
  radar.radarState.leadTwo = lead_two if lead_two is not None else _lead(200.0, float(v_ego), present=False)

  return SubMaster(
    {
      "carState": car_state.carState,
      "carControl": car_control.carControl,
      "controlsState": controls_state.controlsState,
      "selfdriveState": selfdrive_state.selfdriveState,
      "vehicleParameters": vehicle_parameters.vehicleParameters,
      "modelV2": model.modelV2,
      "radarState": radar.radarState,
    },
    moonpilot_leads=moonpilot_leads,
  )


def _planner(car=CAR.HONDA_CIVIC, params_on=True, **kwargs):
  planner = MoonpilotLongitudinalPlanner(_cp(car), **kwargs)
  planner.params = _params(on=params_on)
  return planner


class TestPolicyFunctions(unittest.TestCase):
  def test_setpoint_is_stop_distance_plus_time_gap(self):
    """Steady following holds gap == STOP_DISTANCE + t_follow * v_ego, for every personality."""
    for t_follow in MOONPILOT_T_FOLLOW.values():
      for v in (0.0, 5.0, 20.0, 33.0):
        self.assertAlmostEqual(lead_accel(v, MOONPILOT_STOP_DISTANCE + t_follow * v, v, 0.0, t_follow), 0.0, delta=1e-9)

  def test_approach_to_a_stopped_lead_is_the_constant_decel_profile(self):
    # (25^2 - 0) / (2 * (120 - 6)) — the number the approach profile holds all the way in.
    self.assertAlmostEqual(lead_accel(25.0, 120.0, 0.0, 0.0, 1.45), -2.74, delta=0.02)

  def test_a_far_lead_closing_slowly_does_not_block_acceleration(self):
    # The failure mode of a naive "must be able to stop before the lead" limit: it would command
    # braking here, where 200 m of gap means there is nothing to brake for.
    self.assertGreater(lead_accel(25.0, 200.0, 24.0, 0.0, 1.45), 0.0)

  def test_regulator_never_brakes_harder_than_the_approach_decel(self):
    """Where the kinematic term does not bind, the spacing gain cannot command more than the
    approach decel, however large the speed error is.

    Filtered on the same kinematics lead_accel uses (STOP_DISTANCE and the slack floor), not on
    required_decel: that helper keeps only a 0.25 m contact margin, so it is less conservative than
    the fork's own term, and filtering on it would assert a bound the design gives way on exactly
    where braking harder is genuinely needed.
    """
    checked = 0
    for v_ego in (0.0, 5.0, 10.0, 20.0, 30.0):
      for gap in (6.0, 10.0, 20.0, 50.0, 200.0):
        for v_lead in (0.0, 5.0, 15.0, 25.0):
          a_req = -(v_ego**2 - v_lead**2) / (2 * max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK))
          for t_follow in MOONPILOT_T_FOLLOW.values():
            a = lead_accel(v_ego, gap, v_lead, 0.0, t_follow)
            if a_req < -MOONPILOT_APPROACH_DECEL:
              self.assertLess(a, -MOONPILOT_APPROACH_DECEL)  # the kinematic term governs there
            else:
              self.assertGreaterEqual(a, -MOONPILOT_APPROACH_DECEL)
              checked += 1
    self.assertGreater(checked, 0, "the grid never exercised the regulator's regime")

  def test_the_kinematic_term_is_exactly_what_governs_once_it_binds(self):
    """Once the required decel is past the approach decel, the output is that decel exactly — an
    approach to a stopped lead is planned as a constant-deceleration profile, and the value
    (25^2 - 0) / (2 * (gap - STOP_DISTANCE)) is what it holds all the way in."""
    for gap in (300.0, 250.0, 200.0, 150.0, 120.0):
      a_req = -(25.0**2) / (2 * (gap - MOONPILOT_STOP_DISTANCE))
      self.assertLess(a_req, -MOONPILOT_APPROACH_DECEL)
      self.assertAlmostEqual(lead_accel(25.0, gap, 0.0, 0.0, 1.45), a_req, delta=1e-9)

  def test_the_approach_profile_moves_no_faster_than_the_kinematics(self):
    """Inside the regime where the kinematic term governs, the output is that term, so it changes
    only as fast as the required decel does — the property that makes the profile smooth without a
    solver. (The handover into this regime is a step, bounded by the regulator's own output at that
    gap; the planner's jerk limit is what absorbs it, which is what the maneuver suite exercises.)
    """
    gaps = np.linspace(300.0, 100.0, 400)
    outputs = [lead_accel(25.0, float(gap), 0.0, 0.0, 1.45) for gap in gaps]
    self.assertLess(max(outputs), -MOONPILOT_APPROACH_DECEL)  # the whole sweep is in that regime
    self.assertLess(float(np.abs(np.diff(outputs)).max()), 0.05)

  def test_arbitration_absorbs_the_candidate_step_a_slow_closing_lead_makes(self):
    """The step that reaches the output, not the one the candidate makes.

    At 25 m/s closing on a 20 m/s lead the crossing is at 118.5 m, where the regulator still wants
    +19.9 m/s^2 — a 20.9 m/s^2 step in the lead candidate, and a case upstream's maneuver suite has
    no maneuver for. It cannot reach the output: while the lead asks for more than the cruise
    candidate, `min` picks cruise, so what the output steps by is cruise's cap minus the approach
    decel. Two invariants, and nothing else pins them: arbitration never amplifies a candidate step,
    and the step the output does take is bounded by the cruise cap plus the approach decel.
    """
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    worst_shrink = 1.0
    for v_ego, v_lead, v_cruise_kph in ((25.0, 20.0, 108.0), (20.0, 18.0, 108.0), (15.0, 14.0, 108.0)):
      v_cruise = v_cruise_kph * CV.KPH_TO_MS
      gap_star = MOONPILOT_STOP_DISTANCE + (v_ego**2 - v_lead**2) / (2 * MOONPILOT_APPROACH_DECEL)
      def run(gap, v_ego=v_ego, v_lead=v_lead, v_cruise=v_cruise):
        return policy(v_ego, [(Source.lead0, gap, v_lead, 0.0)], v_cruise, t_follow, False, 0.0, 0.0, CP, -0.3, True)[0]
      candidate_step = abs(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow) - lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow))
      output_step = abs(run(gap_star + 1e-3) - run(gap_star - 1e-3))
      self.assertLessEqual(output_step, candidate_step + 1e-9)  # arbitration never amplifies
      # the slack is the probe offset: inside the crossing the kinematic term is a hair past the approach decel
      self.assertLessEqual(output_step, cruise_accel(v_ego, v_cruise, False, 0.0, CP, -0.3, True) + MOONPILOT_APPROACH_DECEL + 1e-3)
      worst_shrink = min(worst_shrink, output_step / candidate_step)

    # the 25/20 case is the one where the candidate step is large and the output's is not
    self.assertLess(worst_shrink, 0.2)

  def test_the_regulator_really_is_the_candidate_just_outside_the_handover(self):
    """Both sides of the crossing, stated exactly: just inside the regime the kinematic term is the
    output and it equals the approach decel; just outside it, the spacing regulator is the output,
    and it is positive while the gap is still wide. The step between them is what the planner's jerk
    limit has to absorb, which test_the_handover_is_absorbed_by_the_jerk_limit_closed_loop pins.
    """
    for v_ego, v_lead in ((25.0, 20.0), (25.0, 0.0), (15.0, 10.0), (10.0, 0.0)):
      gap_star = MOONPILOT_STOP_DISTANCE + (v_ego**2 - v_lead**2) / (2 * MOONPILOT_APPROACH_DECEL)
      t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
      a_track = max(MOONPILOT_K_GAP * (gap_star - MOONPILOT_STOP_DISTANCE - t_follow * v_ego) + MOONPILOT_K_V * (v_lead - v_ego), -MOONPILOT_APPROACH_DECEL)
      self.assertAlmostEqual(lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow), -MOONPILOT_APPROACH_DECEL, delta=1e-3)
      self.assertAlmostEqual(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow), a_track, delta=1e-3)


class TestPlanner(unittest.TestCase):
  def test_resume_from_a_standstill_commands_movement(self):
    """Upstream's stock-ACC resume spam releases only when the planner commands a_target >= 0.1,
    which is should_stop's threshold — so an opening gap at a standstill has to produce it."""
    planner = _planner()
    sm = _inputs(v_ego=0.0, v_cruise_kph=108.0, lead=_lead(6.2, 0.7), standstill=True)
    for _ in range(10):
      planner.update(sm)
    self.assertGreaterEqual(planner.output_a_target, 0.1)

  def test_standstill_behind_a_stopped_lead_stays_stopped(self):
    planner = _planner()
    sm = _inputs(v_ego=0.0, lead=_lead(MOONPILOT_STOP_DISTANCE, 0.0), standstill=True)
    for _ in range(50):
      planner.update(sm)
    self.assertTrue(planner.output_should_stop)

  def test_a_nan_set_speed_holds_the_current_speed(self):
    planner = _planner()
    sm = _inputs(v_ego=15.0, v_cruise_kph=float("nan"))
    for _ in range(20):
      planner.update(sm)
    self.assertTrue(math.isfinite(planner.output_a_target))
    self.assertLessEqual(abs(planner.output_a_target), 0.2)

  def test_an_unset_set_speed_resets_the_command_from_measured_accel(self):
    """V_CRUISE_UNSET is a reset, as upstream has it: the command restarts from measured accel
    instead of from the previous frame's command."""
    planner = _planner()
    braking = _inputs(v_ego=15.0, v_cruise_kph=108.0, lead=_lead(12.0, 0.0))
    for _ in range(50):
      planner.update(braking)
    self.assertLess(planner.output_a_target, -1.0)

    # no lead, so the policy itself wants to accelerate: the only thing holding the command back is
    # the jerk limit measured from the reset baseline of aEgo
    planner.update(_inputs(v_ego=15.0, v_cruise_kph=V_CRUISE_UNSET, a_ego=0.4))
    self.assertAlmostEqual(planner.output_a_target, 0.4 + MOONPILOT_JERK_UP * DT_MDL, delta=1e-6)

  def test_jerk_is_bounded_at_every_frame(self):
    """A set speed step and a cut-in 10 m ahead, which is the worst case the rate limit exists for:
    the cruise term sits at its cap, then a stopped lead 10 m out demands ACCEL_MIN."""
    planner = _planner()
    bound = MOONPILOT_JERK_EMERGENCY * DT_MDL + 1e-6
    previous = planner.output_a_target
    for frame in range(300):
      cut_in = frame >= 100
      sm = _inputs(v_ego=10.0, v_cruise_kph=108.0 if cut_in else 0.0, lead=_lead(10.0, 0.0, model_prob=1.0) if cut_in else None, standstill=False)
      planner.update(sm)
      self.assertLessEqual(abs(planner.output_a_target - previous), bound)
      previous = planner.output_a_target
    self.assertAlmostEqual(planner.output_a_target, ACCEL_MIN, delta=1e-6)

  def test_fcw_latches_on_an_avoidable_collision(self):
    planner = _planner()
    # 20 m/s against a stopped lead 30 m out: -6.7 m/s^2 required, past the -4.0 m/s^2 threshold.
    sm = _inputs(v_ego=20.0, lead=_lead(30.0, 0.0, model_prob=0.95), standstill=False)
    for _ in range(4):
      planner.update(sm)
    self.assertTrue(planner.fcw)

  def test_fcw_stays_off_while_following(self):
    planner = _planner()
    sm = _inputs(v_ego=20.0, lead=_lead(35.0, 20.0, model_prob=0.95), standstill=False)
    for _ in range(10):
      planner.update(sm)
    self.assertFalse(planner.fcw)

  def test_fcw_ignores_a_lead_the_model_does_not_believe_in(self):
    planner = _planner()
    sm = _inputs(v_ego=20.0, lead=_lead(30.0, 0.0, model_prob=0.5), standstill=False)
    for _ in range(10):
      planner.update(sm)
    self.assertFalse(planner.fcw)

  def test_fcw_does_not_latch_at_a_standstill(self):
    # Creeping at 1.5 m/s with the gap nearly closed is the case the clause guards: the required
    # decel is far past the threshold, but a car that is stopped is not about to be warned.
    planner = _planner()
    sm = _inputs(v_ego=1.5, lead=_lead(0.3, 0.0, model_prob=0.95), standstill=True)
    for _ in range(10):
      planner.update(sm)
    self.assertFalse(planner.fcw)

  def test_source_names_the_winning_candidate(self):
    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(35.0, 20.0)))
    self.assertEqual(planner.source, Source.lead0)

    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=30.0, lead=_lead(35.0, 30.0)))
    self.assertEqual(planner.source, Source.cruise)

    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(35.0, 30.0), experimental=True, model_accel=-2.0))
    self.assertEqual(planner.source, Source.e2e)

  def test_second_lead_only_binds_when_it_is_the_slower_candidate(self):
    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(200.0, 20.0, present=False), lead_two=_lead(12.0, 5.0)))
    self.assertEqual(planner.source, Source.lead1)

  def test_allow_throttle_follows_the_model_and_the_speed(self):
    planner = _planner()
    planner.update(_inputs(v_ego=20.0, throttle_prob=0.0))
    self.assertFalse(planner.allow_throttle)

    planner = _planner()
    planner.update(_inputs(v_ego=1.0, throttle_prob=0.0))
    self.assertTrue(planner.allow_throttle)

  def test_time_gap_scales_with_the_lead_lateral_prediction(self):
    """MoonpilotLeadLateral reaches the longitudinal policy through the time gap, since there is no
    MPC danger factor to scale here."""
    out_of_path = custom.MoonpilotState.LeadTrajectory.new_message()
    out_of_path.present = True
    out_of_path.inPath = 0.0
    in_path = custom.MoonpilotState.LeadTrajectory.new_message()
    in_path.present = True
    in_path.inPath = 1.0

    base = MOONPILOT_T_FOLLOW[Personality.standard]
    self.assertAlmostEqual(
      _planner()._t_follow(_inputs(lead=_lead(35.0, 20.0), moonpilot_leads=[out_of_path])), base * MOONPILOT_OUT_OF_PATH_T_FOLLOW, delta=1e-9
    )
    self.assertAlmostEqual(_planner()._t_follow(_inputs(lead=_lead(35.0, 20.0), moonpilot_leads=[in_path])), base, delta=1e-9)

    # the toggle off, and no moonpilotState published, are both the upstream time gap
    for sm in (_inputs(lead=_lead(35.0, 20.0), moonpilot_leads=[out_of_path]), _inputs(lead=_lead(35.0, 20.0))):
      self.assertAlmostEqual(_planner(params_on=False)._t_follow(sm), base, delta=1e-9)
    self.assertAlmostEqual(_planner()._t_follow(_inputs(lead=_lead(35.0, 20.0))), base, delta=1e-9)

  def test_time_gap_reads_the_personality_off_the_message(self):
    """The table is keyed by the enum's raw value because the enum read off a message is a capnp
    _DynamicEnum whose hash is not that value — keyed by the schema members this silently returned
    the standard gap for every personality."""
    for personality, t_follow in MOONPILOT_T_FOLLOW.items():
      self.assertAlmostEqual(_planner()._t_follow(_inputs(personality=personality)), t_follow, delta=1e-9)

  def test_the_handover_is_absorbed_by_the_jerk_limit_closed_loop(self):
    """Fly the planner at a stopped lead from outside the handover distance and integrate: the step
    the policy makes when the kinematic term takes over has to come out of the jerk limit as a rate,
    and the approach has to stay inside the declared deceleration budget while it stops.
    """
    planner = _planner()
    v_ego, gap = 25.0, 340.0
    previous = planner.output_a_target
    peak = 0.0
    for _ in range(1000):
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, 0.0), standstill=v_ego < 0.1))
      self.assertLessEqual(abs(planner.output_a_target - previous), MOONPILOT_JERK_EMERGENCY * DT_MDL + 1e-6)
      previous = planner.output_a_target
      peak = max(peak, abs(planner.output_a_target))
      v_ego = max(0.0, v_ego + previous * DT_MDL)
      gap = max(0.0, gap - v_ego * DT_MDL)
    self.assertLess(peak, 3.0)
    self.assertLess(v_ego, 0.5)  # it stopped
    self.assertGreater(gap, 5.0)  # and it did not run into the lead

  def test_publish_fills_what_consumers_read(self):
    planner = _planner()
    sm = _inputs(v_ego=20.0, lead=_lead(35.0, 20.0))
    planner.update(sm)

    sent = {}

    class PubMaster:
      def send(self, service, msg):
        sent[service] = msg

    planner.publish(sm, PubMaster())
    plan = sent["longitudinalPlan"].longitudinalPlan
    self.assertEqual(len(plan.speeds), CONTROL_N)
    self.assertEqual(len(plan.accels), CONTROL_N)
    self.assertEqual(len(plan.jerks), CONTROL_N)
    self.assertAlmostEqual(plan.aTarget, planner.output_a_target, delta=1e-9)
    self.assertEqual(plan.shouldStop, planner.output_should_stop)
    self.assertEqual(plan.allowThrottle, planner.allow_throttle)
    self.assertTrue(plan.hasLead)
    self.assertTrue(plan.longitudinalPlanSource in (Source.cruise, Source.lead0, Source.lead1, Source.e2e))
    self.assertGreaterEqual(plan.solverExecutionTime, 0.0)

  def test_trajectory_is_the_same_policy_rolled_forward(self):
    """The published plan has to agree with the command, or controlsd's plan and its accel disagree."""
    planner = _planner()
    sm = _inputs(v_ego=25.0, v_cruise_kph=108.0, lead=_lead(60.0, 0.0))
    for _ in range(20):
      planner.update(sm)
    self.assertAlmostEqual(planner.v_desired_trajectory[0], 25.0, delta=1e-6)
    self.assertAlmostEqual(planner.a_desired_trajectory[0], planner.output_a_target, delta=1e-6)
    # a decaying lead closes the gap, so the plan must not be asking for acceleration by the end
    self.assertLess(planner.v_desired_trajectory[-1], planner.v_desired_trajectory[0])


class TestPlannerSeam(unittest.TestCase):
  def test_factory_returns_the_fork_planner_only_when_wanted_and_applicable(self):
    self.assertIsInstance(moonpilot_longitudinal_planner(_cp(), _params(on=True)), MoonpilotLongitudinalPlanner)
    self.assertIsNone(moonpilot_longitudinal_planner(_cp(), _params(on=False)))
    self.assertIsNone(moonpilot_longitudinal_planner(_cp(openpilot_longitudinal=False), _params(on=True)))


class TestUpstreamManeuvers(unittest.TestCase):
  """Upstream's maneuver suite, run through the seam: the plant builds whatever the seam returns,
  so patching the symbol it binds is exactly how plannerd picks the fork planner."""

  @staticmethod
  def _fork_planner(CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    return MoonpilotLongitudinalPlanner(CP, init_v=init_v, init_a=init_a, dt=dt)

  def test_upstream_maneuvers_pass(self):
    with mock.patch.object(plant_mod, "LongitudinalPlanner", self._fork_planner):
      for e2e, force_decel in itertools.product([True, False], repeat=2):
        for maneuver in create_maneuvers({"e2e": e2e, "force_decel": force_decel}):
          with self.subTest(title=maneuver.title, e2e=e2e, force_decel=force_decel):
            valid, _ = maneuver.evaluate()
            assert valid


if __name__ == "__main__":
  unittest.main()
