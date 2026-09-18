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

from moonpilot.lead import MOONPILOT_LEAD_ACCEL_TAU, LeadAccelEstimator
from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_CONTROL_T_IDX,
  MOONPILOT_FCW_DECEL,
  MOONPILOT_JERK_EMERGENCY,
  MOONPILOT_JERK_UP,
  MOONPILOT_K_GAP,
  MOONPILOT_K_TTC,
  MOONPILOT_K_V,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_T_FOLLOW,
  MOONPILOT_TTC_TARGET,
  MoonpilotLongitudinalPlanner,
  cruise_accel,
  lead_accel,
  lead_state_at,
  moonpilot_longitudinal_planner,
  policy,
  required_decel,
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
  """One radarState lead.

  `a_lead` and `a_lead_tau` are the *message's* fields, and the planner no longer reads either for a
  radar lead: `LeadAccelEstimator` fits the accel from `vLead` across frames and carries its own
  decay. They still matter where the estimator falls back to radard's pair — the first two frames of
  a track, and a vision-only lead — and for the `lead_state_at` calls below that pass an accel in
  directly. A test that wants the planner to see a lead decelerating has to rebuild `vLead` every
  frame, as the FCW tests do.
  """
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

  def test_approach_to_a_stopped_lead_follows_the_ttc_law(self):
    # 114 m of slack at a 5 s headway affords 22.8 m/s of closing, so 25 m/s is 2.2 m/s^2 of excess.
    self.assertAlmostEqual(lead_accel(25.0, 120.0, 0.0, 0.0, 1.45), -2.2, delta=0.02)

  def test_a_far_lead_closing_slowly_does_not_block_acceleration(self):
    # The failure mode of a naive "must be able to stop before the lead" limit: it would command
    # braking here, where 200 m of gap means there is nothing to brake for.
    self.assertGreater(lead_accel(25.0, 200.0, 24.0, 0.0, 1.45), 0.0)

  def test_regulator_never_brakes_harder_than_the_approach_decel(self):
    """Where the TTC term does not bind, the spacing gain cannot command more than the approach
    decel, however large the speed error is.

    Filtered on the same bind condition lead_accel uses — closing, and the slack the headway affords
    at TTC_TARGET — not on required_decel: that helper keeps only a 0.25 m contact margin, so it is
    less conservative than the fork's own term, and filtering on it would assert a bound the design
    gives way on exactly where braking harder is genuinely needed.
    """
    checked = 0
    for v_ego in (0.0, 5.0, 10.0, 20.0, 30.0):
      for gap in (6.0, 10.0, 20.0, 50.0, 200.0):
        for v_lead in (0.0, 5.0, 15.0, 25.0):
          closing = v_ego - v_lead
          binds = closing > 0.0 and (gap - MOONPILOT_STOP_DISTANCE) < MOONPILOT_TTC_TARGET * (closing - MOONPILOT_APPROACH_DECEL / MOONPILOT_K_TTC)
          for t_follow in MOONPILOT_T_FOLLOW.values():
            a = lead_accel(v_ego, gap, v_lead, 0.0, t_follow)
            if binds:
              self.assertLess(a, -MOONPILOT_APPROACH_DECEL)  # the TTC term governs there
            else:
              self.assertGreaterEqual(a, -MOONPILOT_APPROACH_DECEL)
              checked += 1
    self.assertGreater(checked, 0, "the grid never exercised the regulator's regime")

  def test_the_ttc_term_is_exactly_what_governs_once_it_binds(self):
    """Once the TTC term is past the approach decel the output is that term exactly — the closing
    rate this gap's slack affords at TTC_TARGET, and nothing else."""
    for gap in (120.0, 100.0, 60.0):
      a_ttc = -MOONPILOT_K_TTC * (25.0 - (gap - MOONPILOT_STOP_DISTANCE) / MOONPILOT_TTC_TARGET)
      self.assertLess(a_ttc, -MOONPILOT_APPROACH_DECEL)
      self.assertAlmostEqual(lead_accel(25.0, gap, 0.0, 0.0, 1.45), a_ttc, delta=1e-9)

  def test_the_ttc_profile_moves_no_faster_than_its_slope(self):
    """Inside the regime where the TTC term governs, the output is that term, so it changes only as
    fast as the slack does — the property that makes the profile smooth without a solver. (The
    handover into this regime is a step, bounded by the regulator's own output at that gap; the
    planner's jerk limit is what absorbs it, which is what the maneuver suite exercises.)
    """
    gaps = np.linspace(120.0, 30.0, 400)
    outputs = [lead_accel(25.0, float(gap), 0.0, 0.0, 1.45) for gap in gaps]
    self.assertLess(max(outputs), -MOONPILOT_APPROACH_DECEL)  # the whole sweep is in that regime
    self.assertLess(float(np.abs(np.diff(outputs)).max()), 0.05)

  def test_arbitration_absorbs_the_candidate_step_the_handover_makes(self):
    """The step that reaches the output, not the one the candidate makes.

    The crossing is where the TTC term reaches the approach decel, and at 25 m/s against a stopped
    lead that is 126 m, where the regulator still wants +10.1 m/s^2 — an 11.1 m/s^2 step in the lead
    candidate, and a case upstream's maneuver suite has no maneuver for. It cannot reach the output:
    while the lead asks for more than the cruise candidate, `min` picks cruise, so what the output
    steps by is cruise's cap minus the approach decel. Two invariants, and nothing else pins them:
    arbitration never amplifies a candidate step, and the step the output does take is bounded by the
    cruise cap plus the approach decel.
    """
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    worst_shrink = 1.0
    for v_ego, v_lead, v_cruise_kph in ((25.0, 0.0, 108.0), (30.0, 0.0, 108.0), (20.0, 0.0, 108.0)):
      v_cruise = v_cruise_kph * CV.KPH_TO_MS
      gap_star = MOONPILOT_STOP_DISTANCE + MOONPILOT_TTC_TARGET * ((v_ego - v_lead) - MOONPILOT_APPROACH_DECEL / MOONPILOT_K_TTC)

      def run(gap, v_ego=v_ego, v_lead=v_lead, v_cruise=v_cruise):
        return policy(v_ego, [(Source.lead0, gap, v_lead, 0.0)], v_cruise, t_follow, False, 0.0, 0.0, CP, -0.3, True)[0]

      candidate_step = abs(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow) - lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow))
      output_step = abs(run(gap_star + 1e-3) - run(gap_star - 1e-3))
      self.assertLessEqual(output_step, candidate_step + 1e-9)  # arbitration never amplifies
      # the slack is the probe offset: inside the crossing the TTC term is a hair past the approach decel
      self.assertLessEqual(output_step, cruise_accel(v_ego, v_cruise, False, 0.0, CP, -0.3, True) + MOONPILOT_APPROACH_DECEL + 1e-3)
      worst_shrink = min(worst_shrink, output_step / candidate_step)

    # the 30/0 case is the one where the candidate step is large and the output's is not
    self.assertLess(worst_shrink, 0.2)

  def test_the_regulator_really_is_the_candidate_just_outside_the_handover(self):
    """Both sides of the crossing, stated exactly: just inside the regime the TTC term is the output
    and it equals the approach decel; just outside it, the spacing regulator is the output, and it is
    positive while the gap is still wide. The step between them is what the planner's jerk limit has
    to absorb, which test_the_handover_is_absorbed_by_the_jerk_limit_closed_loop pins.
    """
    for v_ego, v_lead in ((25.0, 0.0), (30.0, 0.0), (10.0, 0.0)):
      gap_star = MOONPILOT_STOP_DISTANCE + MOONPILOT_TTC_TARGET * ((v_ego - v_lead) - MOONPILOT_APPROACH_DECEL / MOONPILOT_K_TTC)
      t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
      a_track = max(MOONPILOT_K_GAP * (gap_star - MOONPILOT_STOP_DISTANCE - t_follow * v_ego) + MOONPILOT_K_V * (v_lead - v_ego), -MOONPILOT_APPROACH_DECEL)
      self.assertAlmostEqual(lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow), -MOONPILOT_APPROACH_DECEL, delta=1e-3)
      self.assertAlmostEqual(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow), a_track, delta=1e-3)

  def test_a_lead_that_stops_inside_the_horizon_keeps_its_stopping_distance(self):
    """The predicted travel is the integral of the lead's clamped speed: monotone in t, and exactly
    the stopping distance once it has stopped. The unclamped integral matched until the lead's own
    stop time and then walked it backwards, clamping the 12.5 m it had covered to 0.5 m by the end
    of this sweep."""
    lead = _lead(50.0, 10.0, a_lead=-4.0, a_lead_tau=0.0)
    travel = [lead_state_at(lead, float(t), 0.0, -4.0, 0.0)[0] - 50.0 for t in np.arange(0.05, 5.0, 0.05)]
    self.assertTrue(all(later >= earlier - 1e-9 for earlier, later in zip(travel, travel[1:], strict=False)))
    self.assertAlmostEqual(travel[-1], 10.0**2 / (2 * 4.0), delta=1e-6)
    self.assertAlmostEqual(lead_state_at(lead, 1.0, 0.0, -4.0, 0.0)[1], 6.0, delta=1e-6)
    self.assertEqual(lead_state_at(lead, 5.0, 0.0, -4.0, 0.0)[1], 0.0)

  def test_the_lead_accel_estimate_is_bounded(self):
    """A stepped lead speed differentiates into hundreds of m/s^2; upstream clips it, so does this."""
    self.assertAlmostEqual(lead_state_at(_lead(50.0, 20.0, a_lead_tau=0.0), 1.0, 0.0, -400.0, 0.0)[1], 10.0, delta=1e-6)

  def test_required_decel_counts_a_braking_lead(self):
    """FCW's input: a lead braking hard at matched speed is a threat even at zero relative speed,
    and routine lead braking is not."""
    self.assertLess(required_decel(20.0, 20.0, 20.0, -8.0), MOONPILOT_FCW_DECEL)
    self.assertGreater(required_decel(20.0, 35.0, 20.0, -5.0), MOONPILOT_FCW_DECEL)
    self.assertGreater(required_decel(20.0, 25.0, 20.0, -3.0), MOONPILOT_FCW_DECEL)
    # a lead holding its speed is unchanged: the stopped-car case the threshold was set on
    self.assertAlmostEqual(required_decel(20.0, 30.0, 0.0, 0.0), -(20.0**2) / (2 * (30.0 - 0.25)), delta=1e-9)


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

  def test_fcw_fires_for_a_lead_braking_hard_at_matched_speed(self):
    """The lead's speed history is the only accel input (`a_lead=0.0` on every frame), so a planner
    that went back to reading `aLeadK` cannot pass this: at matched speed with a steady lead the
    required decel is zero for every frame."""
    planner = _planner()
    for i in range(10):
      lead = _lead(20.0, 20.0 - 8.0 * i * DT_MDL, a_lead=0.0, model_prob=0.95)
      planner.update(_inputs(v_ego=20.0, lead=lead, standstill=False))
    self.assertTrue(planner.fcw)

  def test_fcw_stays_off_for_routine_lead_braking(self):
    """Same shape, and the bounds that separate a warning from a nuisance. Not extended past 20
    frames: by vLead ~ 10 m/s the required decel legitimately passes -4.0."""
    planner = _planner()
    for i in range(20):
      lead = _lead(35.0, 20.0 - 5.0 * i * DT_MDL, a_lead=0.0, model_prob=0.95)
      planner.update(_inputs(v_ego=20.0, lead=lead, standstill=False))
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
    the policy makes when the TTC term takes over has to come out of the jerk limit as a rate, and
    the approach has to stop inside the deceleration budget while it stops.

    That budget is now the actuator's own limit. The TTC term is proportional rather than a profile,
    so on a parked lead it ramps until the command saturates at ACCEL_MIN and holds there — the
    kinematic term it replaced held -1.0 m/s^2 for the whole approach, peak 1.01 on this same
    flight. Asserted as an equality with a tight delta so it catches both directions: a law that
    stopped using the budget (the old 1.01), and one that overshot it.
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
    self.assertAlmostEqual(peak, abs(ACCEL_MIN), delta=1e-3)
    self.assertLess(v_ego, 0.5)  # it stopped
    self.assertGreater(gap, 5.0)  # and it did not run into the lead

  def test_a_braking_lead_reaches_the_command_from_its_speed_history(self):
    """The observable statement of the estimator, closed-loop: the lead's speed history alone — every
    frame carries `a_lead=0.0` — has to bring the braking into the command.

    Measured 0.45 s to -1.0 m/s^2 and 1.40 s to -2.0 m/s^2 on this loop's clock, against 0.60 s and
    2.50 s for the same planner reading `aLeadK` directly (which is a revert's shape here, since
    every `_lead` carries `a_lead=0.0`); the bounds sit between the two, not on the measurement.

    Both arms are later than they were under the kinematic term (estimator 0.25 s and 0.40 s, revert
    0.55 s and 1.50 s): the TTC term keys off the closing rate, which only grows once the lead's
    *speed* has fallen, where the kinematic term keyed off the lead's preview-corrected speed and so
    moved on the estimator's accel directly. That costs 0.20 s at -1.0 m/s^2 (4 frames) and a full
    second at -2.0 m/s^2 (20 frames). What survives is the gap between the two arms, which is what
    this test is for: 0.15 s at -1.0 and 1.10 s at -2.0, the latter the same 1.10 s it was under the
    kinematic term.
    """
    planner = _planner()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_ego, v_lead = 20.0, 20.0
    gap = MOONPILOT_STOP_DISTANCE + t_follow * v_ego
    reached = {}
    since_onset = None
    for frame in range(120):  # 2 s of settled following, then the lead brakes at -3.5 m/s^2
      if frame * DT_MDL >= 2.0:
        v_lead = max(0.0, v_lead - 3.5 * DT_MDL)
        since_onset = DT_MDL if since_onset is None else since_onset + DT_MDL
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead, model_prob=0.95), standstill=v_ego < 0.1))
      if since_onset is not None:
        for threshold in (-1.0, -2.0):
          if threshold not in reached and planner.output_a_target <= threshold:
            reached[threshold] = since_onset
      v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
      gap = max(0.0, gap - (v_ego - v_lead) * DT_MDL)
    self.assertTrue(-1.0 in reached)
    self.assertTrue(-2.0 in reached)
    self.assertLessEqual(reached[-1.0], 0.50)
    self.assertLessEqual(reached[-2.0], 1.50)

  def test_the_rollout_runs_on_the_estimators_own_decay(self):
    """The applied plan is rolled forward over 2.5 s, and the decay it applies has to be the
    estimator's own rather than the one the message carries.

    Sampled 0.80 s past the onset, not 0.30 s, and that is the whole of what the TTC approach term
    changed here. At 0.30 s the horizon's end sits on the spacing regulator's -1.0 m/s^2 floor
    whichever decay is used, so the sample the old test took had stopped answering the question —
    0.002 of separation between the two decays. At 0.80 s the horizon reaches into the TTC term's
    active regime, where the fork's own decay (0.278 here) puts the plan at -2.31 m/s^2 and
    radard's 1.5 s decay of the *same* accel floors it at -1.00: 1.309 of separation.

    Both arms run the estimator's accel and differ only in `aLeadTau`, so what moves the plan is the
    decay and nothing else. The defect this pins is the pairing — `aLeadTau` comes from radard's
    slower `aLeadK`, so at a brake onset it calls "transient" (1.5 s) an accel the fork already reads
    as sustained — which is exactly the case below.
    """

    def fly(tau_override=None):
      """Flights 2 s of settled following then a -3.5 m/s^2 onset; returns the published accel
      trajectory, the command and the estimator's decay at 0.80 s past the onset. `tau_override`
      replaces the estimator's own decay while keeping its accel."""

      class ForcedTau(LeadAccelEstimator):
        def update(self, lead):
          a_lead, _ = super().update(lead)
          return a_lead, tau_override

      patcher = mock.patch("moonpilot.longitudinal.LeadAccelEstimator", ForcedTau) if tau_override is not None else None
      if patcher:
        patcher.start()
      try:
        planner = _planner()
        t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
        v_ego, v_lead = 20.0, 20.0
        gap = MOONPILOT_STOP_DISTANCE + t_follow * v_ego
        for frame in range(200):
          onset = frame * DT_MDL - 2.0
          if onset >= 0.0:
            v_lead = max(0.0, v_lead - 3.5 * DT_MDL)
          planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead, model_prob=0.95), standstill=v_ego < 0.1))
          if abs(onset - 0.80) < 1e-9:
            return np.array(planner.a_desired_trajectory), float(planner.output_a_target), planner.lead_accel[0].a_lead_tau
          v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
          gap = max(0.0, gap - (v_ego - v_lead) * DT_MDL)
        raise AssertionError("the sample never arrived")
      finally:
        if patcher:
          patcher.stop()

    own, command, estimator_tau = fly()
    self.assertLessEqual(command, -MOONPILOT_APPROACH_DECEL)  # the onset really is being braked for
    self.assertLess(estimator_tau, MOONPILOT_LEAD_ACCEL_TAU)  # the estimate went deep, so it decayed

    # `_lead`'s default is what the message carries, and what a revert to `lead.aLeadTau` would read
    msg, _, _ = fly(tau_override=MOONPILOT_LEAD_ACCEL_TAU)
    self.assertGreater(float(np.abs(own - msg).max()), 0.8, msg="the rollout is not running on the estimator's decay")

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
    # a stopped lead 60 m ahead closes as the ego approaches, so the plan must not be asking for
    # acceleration by the end of the horizon
    self.assertLess(planner.v_desired_trajectory[-1], planner.v_desired_trajectory[0])

  def test_the_published_speeds_are_the_integral_of_the_published_accels(self):
    """The plan has to be self-consistent: its speeds integrate its own accels on the published
    grid. They were one grid step apart, which also meant the gap the rollout used was not the gap
    this plan produces.

    The lead decelerates and its message is rebuilt every frame, so this runs against a real ramp
    rather than a constant speed the estimator would read as zero accel. It is 25 m out, not further:
    the TTC term binds once the slack is inside `TTC_TARGET * (closing - APPROACH_DECEL / K_TTC)`, so
    against a 20 m/s lead at 25 m/s that is 20 m of slack — a lead 60 m out leaves the term dormant
    and the plan accelerating, which is not the regime this test needs to be in.
    """
    planner = _planner()
    for i in range(20):
      lead = _lead(25.0, 20.0 - 2.0 * i * DT_MDL, a_lead_tau=0.0)
      planner.update(_inputs(v_ego=25.0, v_cruise_kph=108.0, lead=lead))
    self.assertLess(planner.output_a_target, 0.0)  # the ramp really is being braked for
    for i in range(CONTROL_N - 1):
      dt = float(MOONPILOT_CONTROL_T_IDX[i + 1] - MOONPILOT_CONTROL_T_IDX[i])
      expected = max(0.0, float(planner.v_desired_trajectory[i] + planner.a_desired_trajectory[i] * dt))
      self.assertAlmostEqual(float(planner.v_desired_trajectory[i + 1]), expected, delta=1e-6)


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
