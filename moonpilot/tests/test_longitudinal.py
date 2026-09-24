"""The fork longitudinal planner: the policy's numbers, and the properties the seam depends on.

The last test is the load-bearing one — it runs upstream's own maneuver suite against this planner,
which is the end-to-end statement that the fork strategy drives the 15 stock scenarios without a
crash, without stalling at a stop, and while still decelerating under forceDecel.
"""

import itertools
import math
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import custom, log, messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
import openpilot.selfdrive.test.longitudinal_maneuvers.plant as plant_mod
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers

from moonpilot.curve import (
  MOONPILOT_CURVE_A_LAT,
  MOONPILOT_CURVE_ACCEL_MIN,
  MOONPILOT_CURVE_BIAS_KEY,
  MOONPILOT_CURVE_BIAS_MIN_SAMPLES,
  MOONPILOT_CURVE_BIAS_PERSIST_EVERY,
  MOONPILOT_CURVE_HOLD_MARGIN,
  MOONPILOT_CURVE_K_HOLD,
  MOONPILOT_CURVE_PATH_MAX_AGE,
  curve_targets,
  lat_accel_hold,
)
from moonpilot.jerk import (
  MOONPILOT_LONG_JERK_MIN_SAMPLES,
  MOONPILOT_LONG_JERK_PERSIST_EVERY,
  MOONPILOT_LONG_JERK_SCALE_KEY,
)

from moonpilot.features import FEATURES
from moonpilot.lead import MOONPILOT_LEAD_ACCEL_TAU, LeadAccelEstimator
from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_BRAKE_LIGHT,
  MOONPILOT_BRAKE_MEDIUM,
  MOONPILOT_COAST_BAND,
  MOONPILOT_COAST_FLAT_ACCEL,
  MOONPILOT_COAST_GRADE_MIN,
  MOONPILOT_COAST_RECOVERY_ACCEL,
  MOONPILOT_COASTING_BP,
  MOONPILOT_COASTING_V,
  MOONPILOT_CONTROL_T_IDX,
  MOONPILOT_CREEP_SPEED,
  MOONPILOT_FAST_ACCEL_BP,
  MOONPILOT_FAST_ACCEL_V,
  MOONPILOT_FAST_CRAWL,
  MOONPILOT_FCW_DECEL,
  MOONPILOT_FLOOR_ADMISSION_BP,
  MOONPILOT_FLOOR_ADMISSION_V,
  MOONPILOT_FOLLOW_CUSHION,
  MOONPILOT_JERK_EMERGENCY,
  MOONPILOT_JERK_DOWN,
  MOONPILOT_JERK_LAUNCH,
  MOONPILOT_JERK_LAUNCH_SPEED,
  MOONPILOT_JERK_UP,
  MOONPILOT_K_GAP,
  MOONPILOT_K_TTC,
  MOONPILOT_K_V,
  MOONPILOT_LADDER_V_BP,
  MOONPILOT_LEAD_COAST_BRAKE_A,
  MOONPILOT_LEAD_COAST_MAX_GAP,
  MOONPILOT_LEAD_COAST_MAX_V,
  MOONPILOT_LEAD_PREVIEW_T,
  MOONPILOT_MIN_SLACK,
  MOONPILOT_MODEL_BRAKE_THRESHOLD,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_SLOW_ACCEL_V,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_STOP_REST,
  MOONPILOT_T_FOLLOW,
  MOONPILOT_TTC_TARGET,
  MOONPILOT_TTC_TARGET_BP,
  MOONPILOT_TTC_TARGET_V,
  MoonpilotLongitudinalPlanner,
  coast_accel,
  crawl_accel,
  cruise_accel,
  jerk_limit,
  lead_accel,
  lead_state_at,
  model_candidate,
  moonpilot_longitudinal_planner,
  policy,
  required_decel,
  stopping_decel,
)
from moonpilot.tests.fakes import _params

Personality = log.LongitudinalPersonality
Source = log.LongitudinalPlan.LongitudinalPlanSource
ROOT = Path(__file__).resolve().parents[2]
COAST_DESCENT_PITCH = -0.1
COAST_CLIMB_PITCH = 0.1
COAST_SET_SPEED = 30.0
COAST_SUBTHRESHOLD_PITCH = 0.06  # ~6 %: the latest route's moderate grade, below the high-grade gate


def _admission(v_ego):
  return float(np.interp(v_ego, MOONPILOT_FLOOR_ADMISSION_BP, MOONPILOT_FLOOR_ADMISSION_V))


def _ttc_target(v_ego):
  return float(np.interp(v_ego, MOONPILOT_TTC_TARGET_BP, MOONPILOT_TTC_TARGET_V))


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
  moonpilot/lead.py makes, logMonoTime and all_checks() what publish() reads.

  `logMonoTime` carries both stamps, since the planner ages the lead by the gap between them. Equal
  by default, which is the zero correction the planner made before that existed — a test that wants
  to exercise the staleness path shifts the radar stamp, as `_inputs(radar_age_s=...)` does."""

  def __init__(self, data, moonpilot_leads=None, radar_age_s=0.0, recv_age_s=0.0):
    super().__init__(data)
    moonpilot = custom.MoonpilotState.new_message()
    if moonpilot_leads is not None:
      moonpilot.leads = moonpilot_leads
    self["moonpilotState"] = moonpilot
    self.valid = {"moonpilotState": moonpilot_leads is not None}
    self.alive = {"moonpilotState": moonpilot_leads is not None}
    self.logMonoTime = {"modelV2": 1e9, "radarState": 1e9 - radar_age_s * 1e9}
    # The tick the planner consumes the message at, in seconds, as `messaging.SubMaster` stamps it.
    # Equal to the model stamp by default, so `model_age_s` stays the age the planner reads; a test
    # that wants the two anchors apart shifts this one with `recv_age_s`.
    self.recv_time = {"modelV2": 1.0 + recv_age_s}

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
  radar_age_s=0.0,
  model_age_s=0.0,
  recv_age_s=0.0,
  path=None,
  lat_active=True,
  steering_pressed=False,
  vp_valid=False,
  steer_ratio=0.0,
  stiffness_factor=0.0,
  roll=0.0,
):
  car_state = messaging.new_message("carState")
  car_state.carState.vEgo = float(v_ego)
  car_state.carState.aEgo = float(a_ego)
  car_state.carState.vCruise = float(v_cruise_kph)
  car_state.carState.standstill = bool(v_ego < 0.01) if standstill is None else bool(standstill)
  car_state.carState.steeringAngleDeg = float(steer_angle_deg)
  car_state.carState.steeringPressed = bool(steering_pressed)

  car_control = messaging.new_message("carControl")
  car_control.carControl.orientationNED = [0.0, float(pitch), 0.0]
  car_control.carControl.latActive = bool(lat_active)

  controls_state = messaging.new_message("controlsState")
  controls_state.controlsState.forceDecel = bool(force_decel)
  controls_state.controlsState.longControlState = car.CarControl.Actuators.LongControlState.pid

  selfdrive_state = messaging.new_message("selfdriveState")
  selfdrive_state.selfdriveState.experimentalMode = bool(experimental)
  selfdrive_state.selfdriveState.enabled = bool(enabled)
  selfdrive_state.selfdriveState.personality = personality

  vehicle_parameters = messaging.new_message("vehicleParameters")
  vehicle_parameters.vehicleParameters.angleOffsetDeg = 0.0
  vehicle_parameters.vehicleParameters.valid = bool(vp_valid)
  vehicle_parameters.vehicleParameters.steerRatio = float(steer_ratio)
  vehicle_parameters.vehicleParameters.stiffnessFactor = float(stiffness_factor)
  vehicle_parameters.vehicleParameters.roll = float(roll)

  model = messaging.new_message("modelV2")
  model.modelV2.timestampEof = int(round(1e9 - model_age_s * 1e9))  # the publish lag, against SubMaster's modelV2 stamp
  model.modelV2.meta.disengagePredictions.gasPressProbs = [float(throttle_prob)] * 6
  model.modelV2.action.desiredAcceleration = float(model_accel)
  model.modelV2.action.shouldStop = bool(model_should_stop)
  if path is not None:
    x, v_path, psi_rate = path
    model.modelV2.position = log.XYZTData.new_message(x=[float(v) for v in x], t=ModelConstants.T_IDXS)
    model.modelV2.velocity = log.XYZTData.new_message(x=[float(v) for v in v_path], t=ModelConstants.T_IDXS)
    model.modelV2.orientationRate = log.XYZTData.new_message(z=[float(v) for v in psi_rate], t=ModelConstants.T_IDXS)

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
    radar_age_s=radar_age_s,
    recv_age_s=recv_age_s,
  )


def _path(v_path, curvature, x=None):
  """The three path arrays the curve terms read, for a car travelling at `v_path`: the distance ahead,
  the path speed there, and `orientationRate.z = curvature * v_path` — which is how modeld fills the
  two, and why `curve_targets` divides one by the other to get the curvature back.

  `x` defaults to the constant-speed grid `v_path * T_IDXS`, i.e. the model's own time grid turned
  into distances. `curvature` is a scalar or one value per sample, so a test can ramp a curve in over
  a distance rather than stepping it — a step is not a shape a path has, and `np.gradient` reads its
  edge as a lateral-jerk spike.
  """
  t = np.array(ModelConstants.T_IDXS)
  v = np.full(len(t), float(v_path))
  x = float(v_path) * t if x is None else np.asarray(x, dtype=float)
  return x, v, np.broadcast_to(np.asarray(curvature, dtype=float), x.shape) * v


def _planner(car=CAR.HONDA_CIVIC, params_on=True, params_overrides=None, **kwargs):
  # The fake has to be in place *before* construction: `__init__` seeds the learned values
  # (`MoonpilotLongLag`, `MoonpilotLongJerkScale`, `MoonpilotCurveLatScale`) from `Params`, so a dev
  # box whose real store holds a learned lag, a softened jerk scale or a lateral scale above 1.0 would
  # build a different planner than CI does and move every number pinned on this harness — the onset
  # bounds of `test_a_braking_lead_reaches_the_command_from_its_speed_history` included.
  fake = _params(on=params_on, overrides=params_overrides)
  with mock.patch("moonpilot.longitudinal.Params", lambda: fake):
    planner = MoonpilotLongitudinalPlanner(_cp(car), **kwargs)
  planner.params = fake
  return planner


def _fly(v0, v_set_kph, gap0, v_lead0, a_lead_fn, T, dt=DT_MDL):
  """Fly the shipped planner against a scripted lead, point-mass plant, and summarise the run.

  The lead is rebuilt every frame from its own integrated speed rather than handed an `a_lead`, so
  `LeadAccelEstimator` reads a real speed history — the same thing it reads on the road. Scenarios
  give the lead a second of steady following before any brake onset, which is what the estimator
  needs before it can speak. `contact` is upstream's maneuver bar, a gap under 0.4 m.
  """
  planner = _planner()
  v, gap, v_lead, a_prev = v0, gap0, v_lead0, 0.0
  vs, gaps, v_leads, cmds = [], [], [], []
  for frame in range(round(T / dt)):
    a_lead = float(a_lead_fn(frame * dt))
    planner.update(_inputs(v_ego=max(v, 0.0), v_cruise_kph=v_set_kph, a_ego=a_prev, lead=_lead(max(gap, 0.0), max(v_lead, 0.0)), personality=1))
    a_prev = float(planner.output_a_target)
    vs.append(v)
    gaps.append(gap)
    v_leads.append(v_lead)
    cmds.append(a_prev)
    v = max(0.0, v + a_prev * dt)
    v_lead_next = max(0.0, v_lead + a_lead * dt)
    gap = max(0.0, gap + (0.5 * (v_lead + v_lead_next) - v) * dt)
    v_lead = v_lead_next
  return {
    "v": vs,
    "vl": v_leads,
    "gap": gaps,
    "cmd": cmds,
    "peak": min(cmds),
    "min_gap": min(gaps),
    "end_gap": gaps[-1],
    "contact": min(gaps) < 0.4,
  }


def _gap_target(v_ego, t_follow):
  """The spacing regulator's setpoint: the time gap, floored at the standstill distance.

  A floor and not an offset — `STOP_DISTANCE + t_follow * v_ego` made the effective headway
  `t_follow + STOP_DISTANCE / v_ego`, so the car hung back further the slower it went."""
  return max(MOONPILOT_STOP_DISTANCE, t_follow * v_ego)


class TestPolicyFunctions(unittest.TestCase):
  def test_setpoint_is_the_time_gap_floored_at_the_standstill_distance(self):
    """Steady following holds gap == t_follow * v_ego, for every personality, and the standstill
    distance only takes over below `STOP_DISTANCE / t_follow`. The old setpoint — the offset form —
    is a *larger* gap above that speed, i.e. a positive accel ask: the leftover hang-back."""
    for t_follow in MOONPILOT_T_FOLLOW.values():
      for v in (0.0, 3.0, 5.0, 20.0, 33.0):
        self.assertAlmostEqual(lead_accel(v, _gap_target(v, t_follow), v, 0.0, t_follow), 0.0, delta=1e-9)
      self.assertEqual(_gap_target(0.0, t_follow), MOONPILOT_STOP_DISTANCE)
      for v in (5.0, 20.0, 33.0):  # above the floor the setpoint is the pure time gap
        self.assertAlmostEqual(_gap_target(v, t_follow), t_follow * v, delta=1e-9)
        self.assertGreater(lead_accel(v, MOONPILOT_STOP_DISTANCE + t_follow * v, v, 0.0, t_follow), 0.0)

  def test_the_follow_target_is_a_cushion_not_a_wall(self):
    """A low closing speed may cross the nominal target, then the same regulator holds a small
    opening speed until the exact target is recovered. The target remains the only equilibrium."""
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_lead = 20.0
    v_ego = v_lead + MOONPILOT_FOLLOW_CUSHION * MOONPILOT_K_GAP / (2 * MOONPILOT_K_V)
    gap = _gap_target(v_ego, t_follow)
    accel = 0.0
    gap_errors = []

    for _ in range(round(30.0 / DT_MDL)):
      accel = jerk_limit(lead_accel(v_ego, gap, v_lead, 0.0, t_follow), accel, DT_MDL, v_ego)
      v_ego = max(0.0, v_ego + accel * DT_MDL)
      gap += (v_lead - v_ego) * DT_MDL
      gap_errors.append(gap - _gap_target(v_ego, t_follow))

    self.assertLess(min(gap_errors), -0.1 * MOONPILOT_FOLLOW_CUSHION)
    self.assertGreater(min(gap_errors), -MOONPILOT_FOLLOW_CUSHION)
    self.assertAlmostEqual(v_ego, v_lead, delta=1e-3)
    self.assertAlmostEqual(gap_errors[-1], 0.0, delta=1e-3)

  def test_low_speed_lead_braking_coasts_then_yields_to_safety_and_recovery(self):
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    coast = float(np.interp(3.0, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V))

    def ask(v_ego, gap, v_lead, a_lead):
      return policy(v_ego, [(Source.lead0, gap, v_lead, a_lead)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True)

    coasted, source = ask(3.0, 10.0, 1.5, -MOONPILOT_LEAD_COAST_BRAKE_A)
    self.assertAlmostEqual(coasted, coast)
    self.assertEqual(source, Source.lead0)
    self.assertLess(ask(3.0, 6.0, 1.0, -0.5)[0], coast)
    self.assertGreater(ask(3.0, 6.0, 3.0, 1.0)[0], 0.0)
    self.assertGreater(ask(MOONPILOT_LEAD_COAST_MAX_V + 0.1, 10.0, 4.0, -MOONPILOT_LEAD_COAST_BRAKE_A)[0], coast)
    self.assertGreater(ask(3.0, MOONPILOT_LEAD_COAST_MAX_GAP + 0.1, 1.5, -MOONPILOT_LEAD_COAST_BRAKE_A)[0], coast)

  def test_the_approach_handover_has_the_stopping_geometry(self):
    """The regulator hands over to the approach term at gap == STOP_DISTANCE + (v_ego - v_lead)^2 /
    (2 * admission(v_ego)), which is the point where stopping needs more than the floor's scheduled
    admission (`MOONPILOT_FLOOR_ADMISSION_V`, 1.0 at 8 m/s to 1.6 from 14 m/s). The moving (25, 20)
    probe is gone: at its 13.8 m crossing on the scheduled admission the TTC term is the deeper one. That is the relative-frame
    crossing the shipped floor measures: the gap closes at the relative rate while the lead keeps
    travelling, so the absolute-frame form it replaced (`(v^2 - v_lead^2) / 2`) was exact only for a
    stopped lead and sat further out by `(v + v_lead) / (v - v_lead)` everywhere else — 118.5 m
    against 18.5 m at 25/20 m/s. For a stopped lead the two crossings coincide, so those probes are
    unchanged; the (25, 20) probe is at 18.5 m, where the regulator's own output is already capped at
    the approach decel on both sides of the crossing (measured +69.98/+106.05/+41.40/+6.75 m/s^2 above
    it on the stopped cases, -1.0 on the moving one).
    """
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    for v_ego, v_lead in ((25.0, 0.0), (30.0, 0.0), (20.0, 0.0), (10.0, 0.0)):
      gap_star = MOONPILOT_STOP_DISTANCE + (v_ego - v_lead) ** 2 / (2 * _admission(v_ego))
      cushion = min(MOONPILOT_FOLLOW_CUSHION, (v_ego - v_lead) * MOONPILOT_K_V / MOONPILOT_K_GAP)
      a_track = max(
        MOONPILOT_K_GAP * (gap_star - _gap_target(v_ego, t_follow) + cushion) + MOONPILOT_K_V * (v_lead - v_ego),
        -MOONPILOT_APPROACH_DECEL,
      )
      self.assertAlmostEqual(lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow), -_admission(v_ego), delta=1e-3)
      self.assertAlmostEqual(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow), a_track, delta=1e-3)

  def test_the_ttc_term_governs_where_it_asks_for_more_than_stopping(self):
    """Past the crossing the two approach terms are live together, and the TTC term wins wherever it
    asks for more braking than stopping requires — the fork's own response, which is the reason it is
    here rather than the old term alone. Exactly its value, and the floor is provably not what is
    being read (it is shallower at every one of these points).

    The window moved with the floor: at 25 m/s against a stopped lead the floor/TTC crossover is at
    65.1 m (floor -5.25, TTC -5.17 at 65.5 m; TTC -5.50, floor -5.34 at 64.5 m), so TTC governs
    65.1 m down to 21.9 m, where the floor takes over again at the end of the approach. The old
    window's outer probe (64 m) still governs; the inner one (30 m) is unchanged in kind.
    """
    for v_ego, gap in ((25.0, 60.0), (20.0, 50.0), (25.0, 30.0)):
      a_ttc = -MOONPILOT_K_TTC * (v_ego - (gap - MOONPILOT_STOP_DISTANCE) / MOONPILOT_TTC_TARGET)
      a_stop = -(v_ego**2) / (2 * (gap - MOONPILOT_STOP_DISTANCE))
      self.assertLess(a_ttc, a_stop)  # the premise: TTC is the deeper of the two here
      self.assertAlmostEqual(lead_accel(v_ego, gap, 0.0, 0.0, 1.45), a_ttc, delta=1e-9)
    # the crossover itself, and the floor retaking the end of the approach
    floor_out = -(25.0**2) / (2 * (65.5 - MOONPILOT_STOP_DISTANCE))
    ttc_out = -MOONPILOT_K_TTC * (25.0 - (65.5 - MOONPILOT_STOP_DISTANCE) / MOONPILOT_TTC_TARGET)
    self.assertGreater(ttc_out, floor_out)  # outside the window the floor is deeper
    self.assertAlmostEqual(lead_accel(25.0, 65.5, 0.0, 0.0, 1.45), floor_out, delta=1e-9)
    floor_in = -(25.0**2) / (2 * (20.0 - MOONPILOT_STOP_DISTANCE))
    self.assertAlmostEqual(lead_accel(25.0, 20.0, 0.0, 0.0, 1.45), floor_in, delta=1e-9)

  def test_the_stopping_floor_governs_where_the_ttc_target_is_not_enough(self):
    """And the floor wins where it is the deeper one: just inside its own crossing, at the very end
    of an approach, and for a lead that is already moving — the last is the case the TTC term is
    dormant for, where the floor is the only thing braking at all. Each value is the relative-frame
    match `-(v_ego - v_lead)^2 / (2 * slack)`: the old absolute-frame helper read -3.31 at the moving
    case below where the geometry asks -0.37, and at that slack the regulator's own cap (-1.0) is the
    deeper term, so the case it pins moved to a gap where the floor genuinely governs.
    """

    def a_stop(v_ego, gap, v_lead):
      return -((v_ego - v_lead) ** 2) / (2 * (gap - MOONPILOT_STOP_DISTANCE))

    for v_ego, gap, v_lead in ((25.0, 120.0, 0.0), (25.0, 8.0, 0.0), (25.0, 36.0, 15.0), (20.0, 35.0, 10.0)):
      self.assertAlmostEqual(lead_accel(v_ego, gap, v_lead, 0.0, 1.45), a_stop(v_ego, gap, v_lead), delta=1e-6)

  def test_the_last_meter_is_a_soft_stop_target(self):
    """At the latest route's crawl stop, the constant-decel finish to `MOONPILOT_STOP_REST` sets the
    strength: the TTC term's 1.5 s low-speed headway is not past its gate, and the floor (-0.94) is
    under its 1.0 admission. The 3 s headway this state used to pin asked -1.18."""
    v_ego, gap, v_lead, a_lead = 1.37, 6.58, 0.32, -0.8
    v_lead_eff = max(0.0, v_lead + a_lead * MOONPILOT_LEAD_PREVIEW_T)
    a_ttc = -MOONPILOT_K_TTC * (v_ego - v_lead_eff - (gap - MOONPILOT_STOP_DISTANCE) / _ttc_target(v_ego))
    rel_floor = stopping_decel(v_ego, v_lead_eff, a_lead, max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK))
    finish = -(v_ego**2) / (2 * (gap - MOONPILOT_STOP_REST))

    actual = lead_accel(v_ego, gap, v_lead, a_lead, MOONPILOT_T_FOLLOW[int(Personality.standard)])
    self.assertGreater(a_ttc, -MOONPILOT_APPROACH_DECEL)
    self.assertGreater(rel_floor, -_admission(v_ego))
    self.assertAlmostEqual(actual, finish, places=9)
    self.assertGreater(actual, -1.18)  # softer than the 3 s headway's crawl stop

  def test_the_lead_accel_credit_splits_by_direction_and_by_term(self):
    """The lead's own accel is credited forward into two different speeds, one direction each, and
    where each one lands is the whole contract. Braking goes into the speed the safety terms are
    measured against (`MOONPILOT_LEAD_PREVIEW_T`) — the cautious direction, and the only prediction
    either term makes; the constant carries why it stops at half a second. Acceleration goes into the
    speed the *regulator* matches (`MOONPILOT_LEAD_PREVIEW_T`), and must not reach the safety
    terms at all: releasing braking on a predicted launch is the fork braking for a prediction where
    the geometry had not moved. Pinned by arithmetic on both halves, at states where each is the term
    that governs — every other `lead_accel` call in this file passes `a_lead=0.0`, so this is the only
    coverage either has.

    The braking arm moved with the floor: at (20, 40, 10, -2) the old absolute-frame helper read
    -4.41 where the sustain law reads -2.9605, and the uncredited arms -1.4706 — which the floor's
    scheduled admission (1.6 at 20 m/s) no longer admits, so the state moved to a 35 m gap: -3.4091
    credited (predicted speed 9.0 held 2 s against a 29 m slack) and -1.7241 uncredited.
    """
    v_ego, gap, v_lead = 20.0, 35.0, 10.0
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]

    # the floor reads the braking credit only, and it governs this state (past its 1.6 admission at 20 m/s)
    for a_lead, expected in ((0.0, -1.7241379310344827), (2.0, -1.7241379310344827), (-2.0, -3.409090909090909)):
      v_lead_eff = v_lead + min(a_lead, 0.0) * MOONPILOT_LEAD_PREVIEW_T
      a_stop = stopping_decel(v_ego, v_lead_eff, a_lead, gap - MOONPILOT_STOP_DISTANCE)
      self.assertAlmostEqual(a_stop, expected, places=9)
      self.assertLess(a_stop, -_admission(v_ego))
      self.assertAlmostEqual(lead_accel(v_ego, gap, v_lead, a_lead, t_follow), a_stop, places=9)

    # the regulator reads the acceleration credit, one-sided, and this state is the early return's —
    # the lead is faster than the ego, so there is nothing to brake for and `a_track` is the output
    gap_target = max(MOONPILOT_STOP_DISTANCE, t_follow * 10.0)
    a_track_free = MOONPILOT_K_GAP * (20.0 - gap_target) + MOONPILOT_K_V * 2.0
    for a_lead, expected in (
      (0.0, a_track_free),
      (2.0, a_track_free + MOONPILOT_K_V * MOONPILOT_LEAD_PREVIEW_T * 2.0),
      (-2.0, a_track_free),  # a braking lead adds nothing to what the car matches
    ):
      self.assertAlmostEqual(lead_accel(10.0, 20.0, 12.0, a_lead, t_follow), expected, places=9)

    # And the ordering is the safety direction at every state: the acceleration credit can only ever
    # raise the command, the braking credit can only ever lower it, and neither can cross the other.
    for v_ego in (5.0, 15.0, 30.0):
      for gap in (20.0, 60.0, 200.0):
        with self.subTest(v_ego=v_ego, gap=gap):
          neutral = lead_accel(v_ego, gap, 10.0, 0.0, 1.45)
          self.assertLessEqual(neutral, lead_accel(v_ego, gap, 10.0, 2.0, 1.45))
          self.assertGreaterEqual(neutral, lead_accel(v_ego, gap, 10.0, -2.0, 1.45))

  def test_the_up_jerk_is_scheduled_on_speed_and_the_down_jerk_is_not(self):
    """The up limit is `MOONPILOT_JERK_LAUNCH` at and below `MOONPILOT_JERK_LAUNCH_SPEED`, tapering
    back to `MOONPILOT_JERK_UP` by twice it — a launch is the one place the plan's own ask outruns the
    comfort limit, and the car's answer to a lead pulling away is the most visible thing this planner
    does. The down side reads the *command* and never the speed: every braking number in the file was
    measured against it, and the emergency ramp it interpolates into is untouched.
    """
    for v_ego, expected in (
      (0.0, MOONPILOT_JERK_LAUNCH),
      (MOONPILOT_JERK_LAUNCH_SPEED, MOONPILOT_JERK_LAUNCH),
      (2 * MOONPILOT_JERK_LAUNCH_SPEED, MOONPILOT_JERK_UP),
      (25.0, MOONPILOT_JERK_UP),
    ):
      # measured from a command already off zero: the first step of a ramp is the comfort step at
      # every speed, which is what keeps `should_stop`'s 0.1 release threshold out of a single frame's
      # reach — a parked car cannot lose its brake hold to a lead whose reported speed jitters, and
      # `test_the_stop_gate_follows_the_governing_candidate_not_every_lead` pins that from the other end
      self.assertAlmostEqual(jerk_limit(10.0, 0.05, DT_MDL, v_ego), 0.05 + expected * DT_MDL, places=9)
    for v_ego in (0.0, MOONPILOT_JERK_LAUNCH_SPEED, 25.0):
      self.assertAlmostEqual(jerk_limit(10.0, 0.0, DT_MDL, v_ego), MOONPILOT_JERK_UP * DT_MDL, places=9)
    for v_ego in (0.0, MOONPILOT_JERK_LAUNCH_SPEED, 25.0):
      self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, v_ego), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)

  def test_the_approach_stops_behind_a_stopped_lead_at_every_reachable_speed(self):
    """The safety property, and the one the TTC term alone does not have: the fork's car can reach
    V_CRUISE_MAX on openpilot longitudinal, and a proportional approach term binds *after* the
    distance it needs to stop. TTC_TARGET = 5 binds at a slack of 5 * (v - 1), which is inside the
    v^2 / 7 that stopping at ACCEL_MIN needs above ~34 m/s: on a bare TTC term this loop reaches a
    stopped lead at 14 m/s from 36 m/s. The stopping floor is what closes that, so it is pinned
    closed-loop at the speeds that separate the two — and the maneuver suite cannot see any of it,
    its fastest stopped-lead approach being 25 m/s.
    """
    for v0_kph in (90.0, 118.8, 129.6, 144.0):
      planner = _planner()
      v_ego, gap = v0_kph * CV.KPH_TO_MS, 400.0
      peak = 0.0
      for _ in range(4000):
        planner.update(_inputs(v_ego=v_ego, v_cruise_kph=V_CRUISE_MAX, lead=_lead(gap, 0.0), standstill=v_ego < 0.1))
        peak = max(peak, abs(planner.output_a_target))
        v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
        gap = max(0.0, gap - v_ego * DT_MDL)
        if v_ego < 0.05 and abs(planner.output_a_target) < 0.05:
          break
      # Strictly inside the actuator's limit, which is also what separates the two laws: with the
      # floor this loop peaks at 2.02 / 2.32 / 2.43 / 2.51 m/s^2 for the four speeds (end gaps
      # 5.18/5.18/5.17/5.17, bit-identical to the old law — closing at those speeds is the stopped
      # lead the frame fix does not move), where the bare TTC term sits at exactly ACCEL_MIN at all
      # four. `<=` would be vacuous, since `update` clips the command to ACCEL_MIN before it ever
      # reaches `output_a_target`.
      self.assertLess(peak, abs(ACCEL_MIN), f"{v0_kph} kph used the whole actuator limit")
      self.assertLess(v_ego, 0.5)  # it stopped
      self.assertGreater(gap, 5.0)  # without reaching the lead, from a speed it can really be at

  def test_regulator_never_brakes_harder_than_the_approach_decel(self):
    """Where neither approach term binds, the spacing gain cannot command more than the approach
    decel, however large the speed error is.

    Filtered on the same bind condition lead_accel uses — closing, and whichever of the two approach
    terms is past the approach decel — not on required_decel: that helper keeps only a 0.25 m contact
    margin, so it is less conservative than the fork's own terms, and filtering on it would assert a
    bound the design gives way on exactly where braking harder is genuinely needed. The `a_stop` arm
    is the shipped sustain law, which is why the old absolute-frame grid cells at (10, 20, 5),
    (20, 20, 15), (20, 50, 15), (30, 20, 25), (30, 50, 25) and (30, 200, 15) no longer bind: the
    relative match reads -0.89/-0.89/-0.28/-0.89/-0.28/-0.58 there, inside the approach decel, and the
    regulator (or its cap) is what the output reads.
    """
    checked = boarded = 0
    for v_ego in (0.0, 5.0, 10.0, 20.0, 30.0):
      for gap in (6.0, 10.0, 20.0, 50.0, 200.0):
        for v_lead in (0.0, 5.0, 15.0, 25.0):
          closing = v_ego - v_lead
          a_ttc = a_stop = 0.0
          if closing > 0.0:
            a_ttc = -MOONPILOT_K_TTC * (closing - (gap - MOONPILOT_STOP_DISTANCE) / _ttc_target(v_ego))
            a_stop = stopping_decel(v_ego, v_lead, 0.0, max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK))
          binds = a_ttc < -MOONPILOT_APPROACH_DECEL or a_stop < -_admission(v_ego)
          for t_follow in MOONPILOT_T_FOLLOW.values():
            a = lead_accel(v_ego, gap, v_lead, 0.0, t_follow)
            if binds:
              self.assertLess(a, -MOONPILOT_APPROACH_DECEL)  # an approach term governs there
              boarded += 1
            else:
              self.assertGreaterEqual(a, -MOONPILOT_APPROACH_DECEL)
              checked += 1
    self.assertGreater(checked, 0, "the grid never exercised the regulator's regime")
    self.assertGreater(boarded, 0, "the grid never exercised an approach term's regime")

  def test_the_ttc_profile_moves_no_faster_than_its_slope(self):
    """Where the TTC term is the one governing, the output is that term, so it changes only as fast as
    the slack does — the property that makes the profile smooth without a solver. (The handover into
    the regime is a step, bounded by the regulator's own output at that gap; the planner's jerk limit
    is what absorbs it, which is what the maneuver suite exercises.)

    The sweep is the window where the TTC term is the deeper of the two: from the floor/TTC crossover
    (65.1 m at 25 m/s against a stopped lead) inward to 21.9 m, since below that the floor takes over
    again.
    """
    gaps = np.linspace(64.0, 23.0, 400)
    outputs = [lead_accel(25.0, float(gap), 0.0, 0.0, 1.45) for gap in gaps]
    self.assertLess(max(outputs), -MOONPILOT_APPROACH_DECEL)  # the whole sweep is in that regime
    self.assertLess(float(np.abs(np.diff(outputs)).max()), 0.05)

  def test_arbitration_absorbs_the_candidate_step_the_handover_makes(self):
    """The step that reaches the output, not the one the candidate makes.

    The crossing is where stopping first needs more than the approach decel, and at 25 m/s against a
    stopped lead that is 318.5 m, where the regulator still wants +69.7 m/s^2 — a 70.7 m/s^2 step in
    the lead candidate, and a case upstream's maneuver suite has no maneuver for. It cannot reach the
    output: while the lead asks for more than the cruise candidate, `min` picks cruise, so what the
    output steps by is the active cruise slot minus the approach decel. Two invariants pin it:
    arbitration never amplifies a candidate step, and the output step is bounded by that cruise
    slot plus the approach decel. At the set speed the slot is zero even when the gap is oversized.
    """
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    worst_shrink = 1.0
    for v_ego, v_lead, v_cruise_kph in ((25.0, 0.0, 108.0), (30.0, 0.0, 108.0), (20.0, 0.0, 108.0)):
      v_cruise = v_cruise_kph * CV.KPH_TO_MS
      gap_star = MOONPILOT_STOP_DISTANCE + (v_ego**2 - v_lead**2) / (2 * _admission(v_ego))

      def run(gap, v_ego=v_ego, v_lead=v_lead, v_cruise=v_cruise):
        return policy(v_ego, [(Source.lead0, gap, v_lead, 0.0)], v_cruise, t_follow, False, None, 0.0, CP, -0.3, True)[0]

      candidate_step = abs(lead_accel(v_ego, gap_star + 1e-3, v_lead, 0.0, t_follow) - lead_accel(v_ego, gap_star - 1e-3, v_lead, 0.0, t_follow))
      output_step = abs(run(gap_star + 1e-3) - run(gap_star - 1e-3))
      self.assertLessEqual(output_step, candidate_step + 1e-9)  # arbitration never amplifies
      # the slack is the probe offset: inside the crossing the approach term is a hair past the approach decel
      self.assertLessEqual(output_step, cruise_accel(v_ego, v_cruise, False, 0.0, CP, -0.3, True) + _admission(v_ego) + 1e-3)
      worst_shrink = min(worst_shrink, output_step / candidate_step)

    # the 25/0 case is the one where the candidate step is large and the output's is not
    self.assertLess(worst_shrink, 0.2)

  def test_an_oversized_gap_never_accelerates_at_or_above_set_speed(self):
    """A larger gap may remain; recovering it never buys acceleration beyond the set speed."""
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_cruise = 30.0

    for v_ego in (v_cruise, v_cruise + 1.0):
      setpoint = max(MOONPILOT_STOP_DISTANCE, t_follow * v_ego)
      actual = policy(
        v_ego,
        [(Source.lead0, setpoint + 20.0, v_cruise, 0.0)],
        v_cruise,
        t_follow,
        False,
        None,
        0.0,
        CP,
        -0.3,
        True,
      )[0]
      expected = cruise_accel(v_ego, v_cruise, False, 0.0, CP, -0.3, True)
      self.assertLessEqual(actual, 0.0)
      self.assertAlmostEqual(actual, expected, places=9)

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
    and routine lead braking is not. The shipped `stopping_decel` has no `closing > 0` gate for
    exactly this reason — a draft of it did, and `required_decel(20, 20, 20, -8)` read -0.0 there,
    the "reads as no threat until the speed measurement catches up" the helper exists to close —
    while an ego slower than a braking lead still reads through the prediction branch, and a lead
    holding its speed is the plain relative match.

    FCW predicts the lead all the way to rest (`sustain=inf`), so these are the to-rest values, not
    the planner's 2 s horizon: (20, 20, -8) reads -4.47 (the old second constraint read -4.47 there
    too), the routine cases stay silent, and the overtake probe (30 behind 28 holding speed at 12 m)
    reads -0.17 where the absolute-frame form fired at -4.94.
    """
    self.assertAlmostEqual(required_decel(20.0, 20.0, 20.0, -8.0), -4.4692737430167595, delta=1e-9)
    self.assertLess(required_decel(20.0, 20.0, 20.0, -8.0), MOONPILOT_FCW_DECEL)
    self.assertAlmostEqual(required_decel(15.0, 26.0, 20.0, -8.0), -2.2167487684729066, delta=1e-9)
    self.assertGreater(required_decel(20.0, 35.0, 20.0, -5.0), MOONPILOT_FCW_DECEL)
    self.assertGreater(required_decel(20.0, 25.0, 20.0, -3.0), MOONPILOT_FCW_DECEL)
    # a lead holding its speed is unchanged: the stopped-car case the threshold was set on
    self.assertAlmostEqual(required_decel(20.0, 30.0, 0.0, 0.0), -(20.0**2) / (2 * (30.0 - 0.25)), delta=1e-9)
    # and the overtake that must stay silent: 30 behind 28 holding speed at 12 m
    self.assertAlmostEqual(required_decel(30.0, 12.0, 28.0, 0.0), -0.1702127659574468, delta=1e-9)
    self.assertGreater(required_decel(30.0, 12.0, 28.0, 0.0), MOONPILOT_FCW_DECEL)
    # the hard-brake probe that must fire: 30 behind 28 braking at -6 at 25 m
    self.assertAlmostEqual(required_decel(30.0, 25.0, 28.0, -6.0), -4.995374653098982, delta=1e-9)
    self.assertLess(required_decel(30.0, 25.0, 28.0, -6.0), MOONPILOT_FCW_DECEL)

  def test_a_stopped_lead_is_bit_identical_to_the_old_law(self):
    """The frame fix must not move any pinned stopping-distance number: with `v_lead == 0` both the
    sustain arm (skipped — there is no lead speed to predict from) and the `a_lead < 0` arm collapse
    to `-v_ego^2 / (2 * slack)`, exactly. Bit-identical (`==`, not almost), over the speeds and slacks
    the suite pins, so the four reachable-speed approaches and the reference approach are the old
    law's numbers by construction rather than by re-measurement.
    """
    for v_ego in (5.0, 10.0, 20.0, 25.0, 36.0):
      for slack in (2.0, 20.0, 100.0, 312.5):
        old = -(v_ego**2) / (2 * slack)
        self.assertEqual(stopping_decel(v_ego, 0.0, 0.0, slack), old)
        self.assertEqual(stopping_decel(v_ego, 0.0, -5.0, slack), old)
    # and through the floor's own call shape, with the soft-meter clamp in place
    for v_ego, gap in ((25.0, 120.0), (25.0, 8.0), (10.0, 114.0)):
      slack = max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK)
      self.assertEqual(
        stopping_decel(v_ego, 0.0, 0.0, slack),
        -(v_ego**2) / (2 * (gap - MOONPILOT_STOP_DISTANCE))
        if gap - MOONPILOT_STOP_DISTANCE >= MOONPILOT_MIN_SLACK
        else -(v_ego**2) / (2 * MOONPILOT_MIN_SLACK),
      )

  def test_a_far_steady_lead_draws_no_floor_governed_brake(self):
    """The complaint case: route 000003c8's engaged block (70-115 m, 4-7 m/s of closing, ego 17-19
    m/s behind a lead doing 11-14) sat flat at -1.0..-1.17 for ~8 s because the absolute-frame floor
    was the only admitted braking term. The relative frame asks ~5x less there, inside the approach
    decel, so the floor is not admitted at all and the regulator — positive at those gaps — governs.
    Each case computes the old value inline, so the test documents the delta it removes.
    """
    for v_ego, gap, v_lead in ((18.0, 90.0, 12.5), (19.0, 70.0, 14.0), (17.0, 115.0, 11.0)):
      slack = gap - MOONPILOT_STOP_DISTANCE
      new = stopping_decel(v_ego, v_lead, 0.0, slack)
      old = -(v_ego**2 - v_lead**2) / (2 * slack)
      self.assertGreater(new, -MOONPILOT_APPROACH_DECEL)  # not admitted: the floor cannot govern here
      self.assertGreater(old / new, 4.0)  # ~4.7-6.6x across these cases (-0.77..-1.29 vs -0.17..-0.20)
      self.assertGreater(lead_accel(v_ego, gap, v_lead, 0.0, 1.45), 0.0)  # the regulator governs, positive
    # the first case sits exactly on the old admission edge (old -0.9985), which is the point: the old
    # floor governed by default at 70-115 m while the geometry asks for a ramp, and the route's 45/45
    # engaged braking ticks sat on it at -1.0..-1.17.
    # the analysed scene itself: (19.09, 12.25, -1.28, slack 106.9) reads -0.404, not -1.003
    self.assertAlmostEqual(stopping_decel(19.09, 12.25, -1.28, 106.9), -0.40361775991229676, delta=1e-9)
    self.assertAlmostEqual(-(19.09**2 - 12.25**2) / (2 * 106.9), -1.002645463049579, delta=1e-9)

  def test_a_braking_lead_matched_at_speed_still_stops_without_contact(self):
    """The safety direction that forbids the naive relative form: from the 1.45 s follow gap at
    20 m/s behind a lead holding -5 m/s^2 to rest, the plain match reads -0.0 at matched speed and
    contacts it. The sustain law holds 5.70 m of end gap (5.51 before; 5.00 at -3.5 against 5.09),
    flown through the shared closed-loop harness with 1 s of steady following before the onset.
    """
    for brake, end_gap in ((-3.5, 5.06), (-5.0, 5.70)):
      with self.subTest(brake=brake):
        s = _fly(20.0, 72.0, 29.0, 20.0, lambda t, b=brake: b if t > 1 else 0.0, 25.0)
        self.assertFalse(s["contact"])
        self.assertAlmostEqual(s["end_gap"], end_gap, delta=0.05)
        self.assertAlmostEqual(s["min_gap"], end_gap, delta=0.05)  # the gap never dips below its rest
    # and the direct assertion: at matched speed the prediction branch, not the match, is what binds
    self.assertAlmostEqual(stopping_decel(20.0, 20.0, -5.0, 23.0), -1.5151515151515151, delta=1e-9)
    self.assertLess(stopping_decel(20.0, 20.0, -5.0, 23.0), -MOONPILOT_APPROACH_DECEL)

  def test_a_tap_does_not_buy_a_sustain_stop(self):
    """The bound's other direction: a lead tapping -5 m/s^2 for 0.5 s at 30 m/s from 60 m costs
    -1.00 m/s^2 of peak command, below the absolute-frame law's -1.89 and the unbounded sustain's
    -3.50, for an end gap of ~40 m either way. That tap peak is the same trade that capped
    `MOONPILOT_LEAD_PREVIEW_T` at 0.5 s (3.36 there on the same scenario), so this is the number the
    2.0 s horizon was sized to hold under.
    """
    s = _fly(30.0, 108.0, 60.0, 30.0, lambda t: -5.0 if 1 < t < 1.5 else 0.0, 20.0)
    # -0.43 once the floor's admission is scheduled (1.6 at 30 m/s): the tap's -1.00 no longer admits it.
    self.assertAlmostEqual(s["peak"], -0.43, delta=0.05)
    self.assertFalse(s["contact"])

  def test_the_early_crossing_branch_is_exact_and_continuous(self):
    """Where the ego matches the lead's speed while the lead is still braking, the late form credits
    travel the lead has not made yet — at (25, 10, -0.5, slack 25, sustain=inf) it reads -2.50 where
    the true constant decel is -5.00 — so that case has its own closed form, `|a_lead| +
    closing^2 / (2 * slack)`. It applies exactly when `2 * slack <= closing * t`, agrees with the
    late form at that boundary, and binds mostly on the FCW arm where `t` runs to tens of seconds.
    """
    import math

    self.assertAlmostEqual(stopping_decel(25.0, 10.0, -0.5, 25.0, sustain=math.inf), -5.0, delta=1e-9)
    self.assertAlmostEqual(required_decel(25.0, 25.25, 10.0, -0.5), -5.0, delta=1e-9)
    # continuity at the boundary: (20, 10, -2) has t = 2, closing = 10, so 2*slack == 20 at slack 10
    self.assertAlmostEqual(stopping_decel(20.0, 10.0, -2.0, 10.0), -7.0, delta=1e-9)
    self.assertLess(abs(stopping_decel(20.0, 10.0, -2.0, 9.99) - -7.005005005005005), 1e-9)
    self.assertLess(abs(stopping_decel(20.0, 10.0, -2.0, 10.01) - -6.995003568879372), 1e-9)
    # and it is what fires FCW where the late form stays silent: (30, 28, -6, 25 m) reads -5.00
    self.assertLess(required_decel(30.0, 25.0, 28.0, -6.0), MOONPILOT_FCW_DECEL)


class TestDriverLadder(unittest.TestCase):
  """The cruise candidate lands on the driver's measured rungs at each speed-error threshold and
  interpolates between them; forceDecel keeps the pre-ladder law; the crawl cap only holds a creep."""

  def test_the_cruise_rungs(self):
    CP = _cp()
    for v in (0.0, 10.0, 20.0):
      rung = lambda err, v=v: cruise_accel(v, v + err, False, 0.0, CP, -0.3, True)  # noqa: E731
      self.assertAlmostEqual(rung(5.0), min(float(np.interp(v, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V)), 1.7))  # the combined budget
      self.assertAlmostEqual(rung(2.0), float(np.interp(v, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V)))
      self.assertAlmostEqual(rung(1.0), 0.5 * rung(2.0))  # interpolated, not stepped
      self.assertEqual(rung(0.0), 0.0)
    self.assertAlmostEqual(cruise_accel(20.0, 18.5, False, 0.0, CP, -0.3, True), -0.33)  # coasting
    self.assertAlmostEqual(cruise_accel(20.0, 17.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_LIGHT)
    self.assertAlmostEqual(cruise_accel(20.0, 10.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_MEDIUM)

  def test_force_decel_is_not_softened(self):
    CP = _cp()
    self.assertEqual(cruise_accel(1.0, 0.0, False, 0.0, CP, -0.3, True), -1.0)
    self.assertEqual(cruise_accel(20.0, 0.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_MEDIUM)

  def test_the_crawl_cap_holds_a_creep_not_a_launch(self):
    self.assertEqual(crawl_accel(0.8, 0.5, 0.5, 0.0), MOONPILOT_FAST_CRAWL)  # creeping lead
    self.assertEqual(crawl_accel(0.8, 0.5, 0.5, 1.0), 0.8)  # a lead pulling away is a launch
    self.assertEqual(crawl_accel(-0.8, 0.5, 0.5, 0.0), -0.8)  # braking is never shaped
    self.assertEqual(crawl_accel(0.8, 2 * MOONPILOT_CREEP_SPEED, 0.0, 0.0), 0.8)  # faded out


class TestCoastGrade(unittest.TestCase):
  """The grade-aware coast band changes only the cruise candidate, and only inside its bounded
  overspeed/underspeed window."""

  ERROR_INSIDE = 0.75
  ERROR_PAST_EDGE = 1.6
  LOOP_FRAMES = 250

  @staticmethod
  def _cruise(pitch, v_ego, v_cruise=COAST_SET_SPEED, e2e=False, coast_band=MOONPILOT_COAST_BAND, allow_throttle=True):
    sm = _inputs(v_ego=v_ego, v_cruise_kph=v_cruise * 3.6, pitch=pitch, experimental=e2e)
    return cruise_accel(
      v_ego,
      v_cruise,
      e2e,
      0.0,
      _cp(),
      coast_accel(sm['carControl'].orientationNED[1]),
      allow_throttle,
      coast_band,
    )

  def test_descent_inside_band_tapers_to_positive_coast_and_edge_restores_braking(self):
    """Above the set speed but inside the descent band, the command is positive tapered coast rather
    than braking; past 1.5 m/s over, the plain speed-error law is back and brakes."""
    coast = coast_accel(float(np.float32(COAST_DESCENT_PITCH)))
    actual = self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED + self.ERROR_INSIDE)
    expected = coast * (MOONPILOT_COAST_BAND - self.ERROR_INSIDE) / MOONPILOT_COAST_BAND
    self.assertGreater(actual, 0.0)
    self.assertAlmostEqual(actual, expected, delta=1e-12)
    self.assertAlmostEqual(
      self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED + self.ERROR_INSIDE, allow_throttle=False),
      expected,
      delta=1e-12,
    )

    past = self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED + self.ERROR_PAST_EDGE)
    plain = self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED + self.ERROR_PAST_EDGE, coast_band=0.0)
    self.assertLess(past, 0.0)
    self.assertAlmostEqual(past, plain, delta=1e-12)

  def test_climb_inside_band_tapers_to_negative_coast_and_edge_restores_acceleration(self):
    """Below the set speed but inside the climb band, the command is negative tapered coast rather
    than acceleration; past 1.5 m/s under, recovery resumes as a gentle capped pull, not the ladder."""
    coast = coast_accel(float(np.float32(COAST_CLIMB_PITCH)))
    actual = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_INSIDE)
    expected = coast * (MOONPILOT_COAST_BAND - self.ERROR_INSIDE) / MOONPILOT_COAST_BAND
    self.assertLess(actual, 0.0)
    self.assertAlmostEqual(actual, expected, delta=1e-12)
    self.assertAlmostEqual(
      self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_INSIDE, allow_throttle=False),
      self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_INSIDE, allow_throttle=False, coast_band=0.0),
      delta=1e-12,
    )

    past = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE)
    plain = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE, coast_band=0.0)
    self.assertAlmostEqual(past, MOONPILOT_COAST_RECOVERY_ACCEL, delta=1e-12)
    self.assertGreater(plain, past)

  def test_climb_recovery_is_capped_only_near_the_edge(self):
    """The gentle pull binds in [band, 2*band) of underspeed on a steep climb; deeper deficits and
    every non-climb grade keep the plain law, so a launch or a descent is not held to it."""
    near = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - 2.9)
    self.assertAlmostEqual(near, MOONPILOT_COAST_RECOVERY_ACCEL, delta=1e-12)
    deep = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - 3.0)
    deep_plain = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - 3.0, coast_band=0.0)
    self.assertAlmostEqual(deep, deep_plain, delta=1e-12)
    self.assertGreater(deep, MOONPILOT_COAST_RECOVERY_ACCEL)
    # a descent above the set speed by the same amount still brakes on the plain law
    descent = self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE)
    self.assertAlmostEqual(
      descent,
      self._cruise(COAST_DESCENT_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE, coast_band=0.0),
      delta=1e-12,
    )
    # feature off / e2e: no cap
    off = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE, coast_band=0.0)
    e2e = self._cruise(COAST_CLIMB_PITCH, COAST_SET_SPEED - self.ERROR_PAST_EDGE, e2e=True)
    self.assertAlmostEqual(e2e, off, delta=1e-12)

  def test_band_never_fires_against_gravity(self):
    """A descent below the set speed and a climb above it have the band disabled, exactly matching
    the feature-off cruise candidate."""
    for pitch, v_ego in (
      (COAST_DESCENT_PITCH, COAST_SET_SPEED - self.ERROR_INSIDE),
      (COAST_CLIMB_PITCH, COAST_SET_SPEED + self.ERROR_INSIDE),
    ):
      with self.subTest(pitch=pitch):
        self.assertAlmostEqual(
          self._cruise(pitch, v_ego),
          self._cruise(pitch, v_ego, coast_band=0.0),
          delta=1e-12,
        )

  def test_level_subthreshold_e2e_and_off_are_inert(self):
    """Level road, a grade below the threshold, e2e mode and coast_band zero each exactly preserve
    the unbanded cruise candidate."""
    cases = (
      (0.0, False, MOONPILOT_COAST_BAND),
      (COAST_SUBTHRESHOLD_PITCH, False, MOONPILOT_COAST_BAND),
      (COAST_DESCENT_PITCH, True, MOONPILOT_COAST_BAND),
      (COAST_DESCENT_PITCH, False, 0.0),
    )
    self.assertLess(abs(coast_accel(COAST_SUBTHRESHOLD_PITCH) - MOONPILOT_COAST_FLAT_ACCEL), MOONPILOT_COAST_GRADE_MIN)
    for pitch, e2e, band in cases:
      with self.subTest(pitch=pitch, e2e=e2e, band=band):
        self.assertAlmostEqual(
          self._cruise(pitch, COAST_SET_SPEED + self.ERROR_INSIDE, e2e=e2e, coast_band=band),
          self._cruise(pitch, COAST_SET_SPEED + self.ERROR_INSIDE, e2e=e2e, coast_band=0.0),
          delta=1e-12,
        )

  def test_closed_loop_grade_drift_stays_inside_the_band(self):
    """Starting at 30 m/s with the grade's initial coast acceleration, integrating planner output for
    250 frames rises on a descent and sags on a climb, then remains inside the respective 1.5 m/s edge."""
    settled = {}
    for pitch in (COAST_DESCENT_PITCH, COAST_CLIMB_PITCH):
      initial_a = coast_accel(pitch)
      planner = _planner(init_v=COAST_SET_SPEED, init_a=initial_a)
      v = COAST_SET_SPEED
      speeds = []
      for _ in range(self.LOOP_FRAMES):
        planner.update(_inputs(v_ego=v, v_cruise_kph=COAST_SET_SPEED * 3.6, pitch=pitch, a_ego=initial_a))
        v = max(0.0, v + planner.output_a_target * DT_MDL)
        speeds.append(v)
      settled[pitch] = v
      if pitch == COAST_DESCENT_PITCH:
        self.assertGreater(v, COAST_SET_SPEED)
        self.assertLessEqual(max(speeds), COAST_SET_SPEED + MOONPILOT_COAST_BAND + 1e-9)
      else:
        self.assertLess(v, COAST_SET_SPEED)
        self.assertGreaterEqual(min(speeds), COAST_SET_SPEED - MOONPILOT_COAST_BAND - 1e-9)
    self.assertLessEqual(settled[COAST_DESCENT_PITCH], COAST_SET_SPEED + MOONPILOT_COAST_BAND)
    self.assertGreaterEqual(settled[COAST_CLIMB_PITCH], COAST_SET_SPEED - MOONPILOT_COAST_BAND)

  def test_published_plan_rolls_forward_with_the_band(self):
    """A graded frame without a lead publishes the banded policy at trajectory index zero, and its
    remaining accels differ from a planner with the feature disabled."""
    on = _planner()
    off = _planner(params_overrides={"MoonpilotCoastGrade": False})
    sm = _inputs(v_ego=COAST_SET_SPEED + self.ERROR_INSIDE, v_cruise_kph=COAST_SET_SPEED * 3.6, pitch=COAST_DESCENT_PITCH)
    self.assertFalse(sm['radarState'].leadOne.present)
    on.update(sm)
    off.update(sm)
    self.assertAlmostEqual(float(on.a_desired_trajectory[0]), on.output_a_target, delta=1e-12)
    self.assertFalse(np.allclose(on.a_desired_trajectory, off.a_desired_trajectory))

  def test_feature_row_is_on_and_registered(self):
    """The row is offroad-only, applies immediately, and its persistent param default is on."""
    feature = next(f for f in FEATURES if f.key == "MoonpilotCoastGrade")
    self.assertTrue(feature.offroad_only)
    self.assertFalse(feature.requires)
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('{"MoonpilotCoastGrade", {PERSISTENT, BOOL, "1"}}' in text)


class TestStalenessScope(unittest.TestCase):
  """The scope of `lead_age`'s safety claim, which the AGENTS.md bullet states.

  Partitioned, and stated as such. Where the floor does not read the lead's accel (`a_lead == 0`
  here), ageing the lead shrinks the gap by `(v_lead - v_ego) * age` and lowers the lead's predicted
  speed under the decay, so the closing rate rises and `a_track`, `a_ttc` and `a_stop` are each
  non-increasing — and `min`, with the clip the command passes through, preserves that. That half
  is strict: 0 violations on all 170 in-scope cells at 46 ms and 0.2 s, and 0 on the 78,260-state
  full sweep at 46 ms, the measured median lead age.

  Where the floor does read it (`a_lead < 0`), the aged lead carries a decayed `a_traj`
  (`exp(-a_lead_tau * t^2 / 2)`), i.e. weaker predicted braking and a longer predicted lead travel,
  so the requirement can read genuinely shallower — the physically correct direction, and the reason
  the old absolute-frame form was monotone there (it ignored `a_lead` entirely). That half is
  bounded, not strict: worst rise on this grid is +0.010 m/s^2 at (0.2 s, v_ego 20, gap 90, v_lead 20,
  a_lead -6.0), fresh -1.2966 against aged -1.2866, both sustain-law bound; the full sweep goes
  non-zero only at ages this pipeline does not produce (1,218 states at 0.2 s, 7,563 at 0.5 s), all
  of it -1.0-gate admission flicker at 160-300 m. The `differ` counter below is over the strict half
  only, so the bound means something where it is claimed.
  """

  AGES = (0.046, 0.2)
  V_EGO = (5.0, 12.0, 20.0, 30.0, 40.0)
  GAPS = (8.0, 20.0, 45.0, 90.0, 200.0)
  V_LEAD = (0.0, 5.0, 12.0, 20.0)
  A_LEAD = (-6.0, -3.0, -1.0, 0.0)
  ACTION_T = 0.2  # the test CP's longitudinalActuatorDelay + DT_MDL, as the planner computes it
  RISE_BOUND = 0.02  # twice the worst measured rise on this grid (+0.010), for the a_lead < 0 half

  def test_the_correction_never_reads_optimistic_where_the_ego_is_closing(self):
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    differ = 0
    for age in self.AGES:
      for v_ego in self.V_EGO:
        for gap in self.GAPS:
          for v_lead in self.V_LEAD:
            if v_lead > v_ego:
              continue  # a lead faster than the ego is pulling away, and is not the case this claims
            for a_lead in self.A_LEAD:
              lead = _lead(gap, v_lead, a_lead=a_lead)
              fresh = lead_state_at(lead, self.ACTION_T, v_ego * self.ACTION_T, a_lead, MOONPILOT_LEAD_ACCEL_TAU)
              aged = lead_state_at(lead, self.ACTION_T + age, v_ego * (self.ACTION_T + age), a_lead, MOONPILOT_LEAD_ACCEL_TAU)
              a_fresh = lead_accel(v_ego, *fresh, t_follow)
              a_aged = lead_accel(v_ego, *aged, t_follow)
              p_fresh = policy(v_ego, [(Source.lead0, *fresh)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True)[0]
              p_aged = policy(v_ego, [(Source.lead0, *aged)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True)[0]
              if a_lead == 0.0:
                self.assertLessEqual(
                  a_aged, a_fresh + 1e-9, f"the lead candidate rose with the age at v_ego {v_ego}, gap {gap}, v_lead {v_lead}, a_lead {a_lead}"
                )
                self.assertLessEqual(p_aged, p_fresh + 1e-9)
                differ += a_aged < a_fresh - 1e-9
              else:
                self.assertLessEqual(
                  a_aged, a_fresh + self.RISE_BOUND, f"the lead candidate rose past the bound at v_ego {v_ego}, gap {gap}, v_lead {v_lead}, a_lead {a_lead}"
                )
                self.assertLessEqual(p_aged, p_fresh + self.RISE_BOUND)
    self.assertGreater(differ, 0)  # the arms are not identical over this grid, so the bound means something


class TestPlanner(unittest.TestCase):
  def test_resume_from_a_standstill_commands_movement(self):
    """Upstream's stock-ACC resume spam releases only when the planner commands a_target >= 0.1,
    which is should_stop's threshold — so an opening gap at a standstill has to produce it."""
    planner = _planner()
    sm = _inputs(v_ego=0.0, v_cruise_kph=108.0, lead=_lead(6.2, 0.7), standstill=True)
    for _ in range(10):
      planner.update(sm)
    self.assertGreaterEqual(planner.output_a_target, 0.1)

  def test_a_launch_is_not_held_back_by_the_comfort_jerk(self):
    """From a standstill behind a lead pulling away at 2 m/s^2, the ramp is the comfort step once and
    then the launch jerk — 0.2 m/s^2 per frame against the comfort 0.075, which is what gets the
    delivered command to 0.5 m/s^2 three frames sooner than the comfort limit alone reaches it (it
    needs seven). The first step stays the comfort one on purpose: `should_stop` releases a standstill
    hold when `a_target` crosses 0.1, so a step of 0.2 would let one frame of a lead's reported speed
    lift a parked car's brake hold, which `test_the_stop_gate_follows_the_governing_candidate_not_every_lead`
    pins from the other side.
    """
    planner = _planner()
    a_lead, steps = 2.0, []
    for frame in range(6):
      t = frame * DT_MDL
      lead = _lead(MOONPILOT_STOP_DISTANCE + 0.5 * a_lead * t**2, a_lead * t, a_lead=a_lead)
      previous = planner.output_a_target
      planner.update(_inputs(v_ego=0.0, v_cruise_kph=108.0, lead=lead, standstill=True))
      steps.append(planner.output_a_target - previous)
    self.assertAlmostEqual(steps[0], MOONPILOT_JERK_UP * DT_MDL, delta=1e-9)
    self.assertAlmostEqual(steps[1], MOONPILOT_JERK_LAUNCH * DT_MDL, delta=1e-9)
    self.assertGreater(planner.output_a_target, MOONPILOT_JERK_UP * DT_MDL * 6)  # past what six comfort frames could deliver

  def test_accelerating_lead_launch_stays_near_the_live_target(self):
    """Measured stop-and-go launches from the 6 m standstill target do not need a positive-side
    taper: over 5 s at 1/2/3 m/s^2 the minimum target error is 0/0/0 m, and the commands at 0.25 s
    are 0.478/0.83/0.83 m/s^2 — the two faster leads capped by the driver's fast-accel rung at rest
    (0.998/1.075 before it). The launch jerk answers the lead without crossing inward."""
    for a_lead, expected_cmd in ((1.0, 0.478), (2.0, 0.83), (3.0, 0.83)):
      planner = _planner()
      v_ego = v_lead = gap = 0.0
      gap = MOONPILOT_STOP_DISTANCE
      errors = []
      for frame in range(round(5.0 / DT_MDL)):
        t = frame * DT_MDL
        lead = _lead(gap, v_lead, a_lead=a_lead if t < 5.0 else 0.0)
        planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=lead, standstill=v_ego < 0.01))
        if frame == round(0.25 / DT_MDL):
          self.assertAlmostEqual(planner.output_a_target, expected_cmd, delta=0.02)
        errors.append(gap - max(MOONPILOT_STOP_DISTANCE, MOONPILOT_T_FOLLOW[int(Personality.standard)] * v_ego))
        v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
        v_lead = max(0.0, v_lead + a_lead * DT_MDL)
        gap = max(0.0, gap + (v_lead - v_ego) * DT_MDL)
      self.assertGreaterEqual(min(errors), -0.35)

  def test_mid_approach_lead_brake_stays_safe(self):
    """A lead braking at -3.5 m/s^2 mid-approach while TTC is 4–7 s exercises the floor admission
    with the regulator capped at -1.0. Flown closed-loop with 1 s of steady following before the
    onset so the estimator has a real speed history: the new law peaks at -1.50 m/s^2 with a 6.96 m
    minimum gap and a 9.26 m
    end gap, against the old law's -1.41 / 11.72 / 11.72 — shallower braking held longer, no contact
    either way. The old pin (32.55 m / -1.79) was the absolute-frame floor braking from 90 m out;
    the relative frame does not.
    """
    s = _fly(17.0, 61.2, 90.0, 17.0, lambda t: -3.5 if 2 <= t < 5 else 0.0, 25.0)
    self.assertTrue(any((v - vl) > 0.0 and 4.0 <= (g - MOONPILOT_STOP_DISTANCE) / (v - vl) <= 7.0 for v, vl, g in zip(s["v"], s["vl"], s["gap"], strict=True)))
    # 6.74 / -1.78 / 9.31 on the driver's ladder with the floor's scheduled admission (7.09 / -1.50 /
    # 9.26 with the flat -1.0 gate, 6.96 on the old 1/s cruise gain): at 17 m/s the floor now waits for
    # 1.5 m/s^2 of need, so it brakes later and firmer and the minimum lands 0.35 m closer.
    self.assertAlmostEqual(s["min_gap"], 6.74, delta=0.05)
    self.assertAlmostEqual(s["peak"], -1.78, delta=0.05)
    self.assertAlmostEqual(s["end_gap"], 9.31, delta=0.05)
    self.assertFalse(s["contact"])
    self.assertGreater(s["peak"], ACCEL_MIN)

  def test_a_radar_dropout_releases_the_lead_candidate(self):
    """A missing slot is not a held stop: once the radar lead leaves, policy sees cruise rather than
    a latched lead candidate."""
    planner = _planner()
    planner.update(_inputs(v_ego=10.0, v_cruise_kph=108.0, lead=_lead(20.0, 0.0)))
    planner.update(_inputs(v_ego=10.0, v_cruise_kph=108.0, lead=_lead(200.0, 10.0, present=False)))
    self.assertEqual(planner.source, Source.cruise)

  def test_a_launch_is_answered_through_the_projection_and_the_regulator(self):
    """Both positive channels, and nothing else. `lead_state_at` projects the lead's own accel over
    `action_t + lead_age`, so the gap the policy sees is the lead's future *position*; `lead_accel`'s
    regulator then matches `v_lead + MOONPILOT_LEAD_PREVIEW_T * a_lead`, its future *speed*. The
    safety terms stay on the raw speed (`v_ego - v_lead_eff` is still positive here — the braking
    credit is one-sided, and this lead is accelerating — so no approach term is even consulted); that
    split is the point, because a launch is the one moment there is nothing to brake for and a credit
    that reached the floor would be braking for a prediction anyway.
    """
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    gap, v_lead = 6.5, 0.3
    projected = lead_state_at(_lead(gap, v_lead), 0.25, 0.0, 2.0, 0.0)
    unprojected = lead_state_at(_lead(gap, v_lead), 0.25, 0.0, 0.0, 0.0)
    self.assertGreater(projected[0], unprojected[0])  # further ahead by the time it counts
    self.assertGreater(projected[1], unprojected[1])  # and faster

    a_track = MOONPILOT_K_GAP * (gap - MOONPILOT_STOP_DISTANCE) + MOONPILOT_K_V * v_lead
    for a_lead, expected in ((0.0, min(a_track, MOONPILOT_FAST_CRAWL)), (2.0, a_track + MOONPILOT_K_V * MOONPILOT_LEAD_PREVIEW_T * 2.0)):
      self.assertAlmostEqual(lead_accel(0.0, gap, v_lead, a_lead, t_follow), expected, places=9)
    self.assertGreater(lead_accel(0.0, gap, v_lead, 2.0, t_follow), 0.0)

  def test_standstill_behind_a_stopped_lead_stays_stopped(self):
    planner = _planner()
    sm = _inputs(v_ego=0.0, lead=_lead(MOONPILOT_STOP_DISTANCE, 0.0), standstill=True)
    for _ in range(50):
      planner.update(sm)
    self.assertTrue(planner.output_should_stop)

  def test_trailing_a_moving_lead_is_never_a_stop(self):
    """`should_stop`'s `v_ego < 0.3` gate is upstream's, and upstream's planner only reaches those
    speeds on the way to rest — it decides the car should stop. The fork's regulator trails a lead at
    the lead's own speed, where the command sits on zero inside the predicate's 0.1 threshold, so the
    flag used to toggle ~13 times a second and `LongControlState.stopping` spent the creep fighting the
    plan: measured against a 0.30 m/s lead the car parked 0.37 m behind its setpoint with 202 mm/s of
    speed ripple.

    Trailed at 0.2 m/s rather than 0.3: `should_stop`'s own boundary is `v_ego < 0.3`, so a car sitting
    exactly on 0.3 would make the ungated result depend on how that comparison lands rather than on this
    gate. 0.2 is inside the band and still above the gate's 0.1.
    """
    planner = _planner()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_lead = 0.2

    # closed loop: trailing the creep, the flag must not toggle at all
    v_ego, gap = v_lead, _gap_target(v_lead, t_follow)
    flips, previous = 0, None
    for _ in range(500):
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead)))
      if previous is not None and planner.output_should_stop != previous:
        flips += 1
      previous = planner.output_should_stop
      v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
      gap = max(0.0, gap - (v_ego - v_lead) * DT_MDL)
    self.assertEqual(flips, 0)
    self.assertFalse(planner.output_should_stop)

    # and the cases the upstream predicate was written for are untouched
    stopped = _planner()
    for _ in range(50):
      stopped.update(_inputs(v_ego=0.0, lead=_lead(MOONPILOT_STOP_DISTANCE, 0.0), standstill=True))
    self.assertTrue(stopped.output_should_stop)

    no_lead = _planner()
    for _ in range(50):
      no_lead.update(_inputs(v_ego=0.0, v_cruise_kph=0.0, force_decel=True, standstill=True))
    self.assertTrue(no_lead.output_should_stop)

  def test_the_stop_gate_follows_the_governing_candidate_not_every_lead(self):
    """Two ways a blanket "no moving lead" test would suppress a stop the plan really wants — both
    narrowed by keying the gate on the candidate that won `policy`'s minimum, not on every present lead.

    A moving `leadTwo` in the next lane while parked behind a stopped `leadOne` used to lift the flag,
    and with it `LongControlState.stopping`'s brake hold: measured closed-loop, the delivered command
    went from `CP.stopAccel` to 0.00 and the car rolled. And a force-decel stop is the *cruise*
    candidate's — `forceDecel` zeroes `v_cruise` — so any lead at all moving faster than the deadband
    used to erase it, which is the way `maneuver.py`'s validity check reads this flag.
    """
    # leadTwo pulling away in the next lane: leadOne still governs, and the stop stands
    two_leads = _planner()
    for _ in range(50):
      two_leads.update(_inputs(v_ego=0.0, lead=_lead(MOONPILOT_STOP_DISTANCE, 0.0), lead_two=_lead(30.0, 1.0), standstill=True))
    self.assertTrue(two_leads.output_should_stop)

    # forceDecel with a moving lead: the stop is cruise's, so it survives
    for gap, v_lead in ((6.2, 1.0), (30.0, 2.0)):
      decel = _planner()
      decel.update(_inputs(v_ego=0.2, v_cruise_kph=0.0, force_decel=True, lead=_lead(gap, v_lead)))
      self.assertEqual(decel.source, Source.cruise)
      self.assertTrue(decel.output_should_stop)

    # and a parked car cannot lose its hold to a lead whose reported speed jitters over the deadband.
    # Not force-decel: at a standstill with the set speed live the *lead* candidate governs, which is the
    # case the ego-speed half of the gate exists for. Under force-decel the cruise candidate governs and
    # `s == source` alone excludes it, so the assertion would hold either way. The jitter is 0.2 m/s —
    # twice the deadband — because at 0.4 the estimator reads ~+1.6 m/s^2 of lead accel, the credit
    # lifts the lead's ask over the ladder's 0.83 fast-accel rung at rest, and cruise governs instead.
    jitter = _planner()
    for frame in range(200):
      jitter.update(_inputs(v_ego=0.0, v_cruise_kph=108.0, lead=_lead(MOONPILOT_STOP_DISTANCE, 0.2 if frame % 2 else 0.0), standstill=True))
      self.assertEqual(jitter.source, Source.lead0)
      self.assertTrue(jitter.output_should_stop)

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
    # 29.0 m is the setpoint at 20 m/s (1.45 s, standard): the regulator's own ask ties at zero there
    # and is the minimum against cruise's +1.2, so the lead is the candidate that governs.
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(29.0, 20.0)))
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
    the policy makes when the approach term takes over has to come out of the jerk limit as a rate,
    and the approach has to stay inside the deceleration budget while it stops.

    The budget here is the floor's own: with the stopping floor governing the early approach this
    flight peaks at 1.43 m/s^2, close to the 1.01 the old kinematic term held, and the actuator's
    ACCEL_MIN is left to the speeds where the floor has to use it (the reachable-speed pin above).

    The edge is also what `MOONPILOT_JERK_DOWN` is set to: cruise to the floor's -1.0 m/s^2 takes
    0.50 s here at 2.0, against 0.25 s at 4.0, and that ramp is the whole of what the constant is for
    (see the longitudinal strategy in `AGENTS.md`). Asserted as a floor on the edge duration, since a
    shorter one is the abrupt assignment this was softened away from and costs nothing else — the peak
    moves 1.39 -> 1.43 m/s^2 across the range, and the stopping behavior is the floor's.
    """
    planner = _planner()
    v_ego, gap = 25.0, 340.0
    previous = planner.output_a_target
    peak, t_onset, t_arrived, now = 0.0, -1.0, -1.0, 0.0
    for _ in range(1000):
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, 0.0), standstill=v_ego < 0.1))
      self.assertLessEqual(abs(planner.output_a_target - previous), MOONPILOT_JERK_EMERGENCY * DT_MDL + 1e-6)
      # time from leaving the cruise term to arriving at the floor, latched at the first arrival: the
      # command rises back above the decel again near the stop, so an unlatched counter measures the
      # whole flight instead of the edge.
      if t_onset < 0.0 and planner.output_a_target < 0.0:
        t_onset = now
      elif t_onset >= 0.0 and t_arrived < 0.0 and planner.output_a_target <= -MOONPILOT_APPROACH_DECEL:
        t_arrived = now
      previous = planner.output_a_target
      peak = max(peak, abs(planner.output_a_target))
      v_ego = max(0.0, v_ego + previous * DT_MDL)
      gap = max(0.0, gap - v_ego * DT_MDL)
      now += DT_MDL
    self.assertGreaterEqual(t_arrived, 0.0, "the flight never reached the approach decel")
    self.assertGreaterEqual(t_arrived - t_onset, 0.45)  # the softened edge, not the 0.25 s it replaced
    self.assertLess(peak, 3.0)
    self.assertLess(v_ego, 0.5)  # it stopped
    self.assertGreater(gap, 5.0)  # and it did not run into the lead

  def test_the_reference_approach_holds_its_average_decel(self):
    """The developer's stated objective as an observable property: on the 10 m/s reference approach
    (stopped lead appearing at the 114 m first-present maximum, cruise at entry speed), average decel
    from the -0.5 crossing to rest is 0.979 m/s^2 with peak -1.20 and rest 5.18 m. A raised admission
    threshold shortens the distance used and raises the average (1.15 measured 1.137 with peak -1.41),
    so this fails if anyone re-raises the gate without measuring. Pinned on the command, not the
    constant — the law, not the dial.
    """
    planner = _planner()
    v_ego, gap = 10.0, 114.0
    commands, speeds, gaps = [], [], []
    for _ in range(round(80.0 / DT_MDL)):
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=10.0 * 3.6, lead=_lead(gap, 0.0), standstill=False))
      commands.append(float(planner.output_a_target))
      speeds.append(v_ego)
      gaps.append(gap)
      v_ego = max(0.0, v_ego + commands[-1] * DT_MDL)
      gap = max(0.0, gap - v_ego * DT_MDL)
      if v_ego < 0.05 and abs(commands[-1]) < 0.05:
        break
    onset = next(i for i, c in enumerate(commands) if c < -0.5)
    rest = next((i for i in range(onset, len(speeds)) if speeds[i] < 0.05), len(speeds) - 1)
    distance = gaps[onset] - gaps[rest]
    self.assertGreater(distance, 0.0, "the approach never reached the -0.5 command")
    # Re-pinned on the driver's data: the floor's admission is scheduled (1.2 at 10 m/s) and the TTC
    # headway shortens below 8 m/s, so the approach starts at 47 m instead of 55.5 m and averages
    # 1.176 against 0.979 — the driver's own 6-12 m/s stops plateau near -1.0 with a -1.75 median peak.
    self.assertLessEqual(speeds[onset] ** 2 / (2 * distance), 1.176 + 0.02)
    self.assertGreaterEqual(min(commands[onset:]), -1.23 - 0.05)
    self.assertAlmostEqual(gaps[-1], 5.12, delta=0.05)

  def test_a_fresh_matched_speed_radar_lead_stays_within_approach_decel(self):
    """A new radar slot with no reported lead acceleration must not turn matched-speed following into
    harder-than-approach braking on its first planner update."""
    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(15.0, 20.0)))
    self.assertEqual(planner.source, Source.lead0)
    self.assertAlmostEqual(planner.output_a_target, -MOONPILOT_JERK_DOWN * DT_MDL, delta=1e-9)
    self.assertGreaterEqual(planner.output_a_target, -MOONPILOT_APPROACH_DECEL)

  def test_a_vision_lead_accel_uses_the_half_second_preview(self):
    """A vision-only matched-speed lead gets the half-second preview while jerk limiting shapes the
    delivered braking command. The first-frame policy ask is -2.145 m/s^2 (sustain law against the
    credited speed), which the jerk limit delivers as -0.1388; by frame 10 the command is -1.2604.
    The old pins (-0.2972 / -2.2863) were the absolute-frame floor answering the same preview."""
    gap, v_ego = 25.0, 20.0
    zero_accel = _planner()
    zero_accel.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_ego)))

    vision_lead = _lead(gap, v_ego, a_lead=-4.0, a_lead_tau=0.3)
    vision_lead.radar = False
    vision = _planner()
    vision_input = _inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=vision_lead)
    vision.update(vision_input)

    self.assertEqual(zero_accel.source, Source.lead0)
    self.assertEqual(vision.source, Source.lead0)
    self.assertLess(vision.output_a_target, zero_accel.output_a_target)
    self.assertAlmostEqual(vision.output_a_target, -0.13875371166584527, delta=1e-7)

    for _ in range(9):
      vision.update(vision_input)
    self.assertAlmostEqual(vision.output_a_target, -1.2603791557257866, delta=1e-4)
    self.assertGreater(vision.output_a_target, ACCEL_MIN)

  def test_a_braking_lead_reaches_the_command_from_its_speed_history(self):
    """The observable statement of the estimator, closed-loop: the lead's speed history alone — every
    frame carries `a_lead=0.0` — has to bring the braking into the command.

    Measured on this loop's clock, a lead braking at -3.5 m/s^2 from a settled follow: -0.5 m/s^2 at
    0.30 s, -1.0 at 0.55 and -2.0 at 0.90 through the fork's slope estimate, against 0.40 / 0.70 / 2.65
    for the same planner with the estimator forced to zero (which is a revert's shape here, since
    every `_lead` carries `a_lead=0.0` — forcing zero is also the no-credit law, as the preview
    credit derives from that same estimate); each bound sits strictly between the arms, not on a
    measurement, so the test still fails on a revert. The -0.5 bound moved 0.25 -> 0.35 with the
    sustain floor (estimator 0.30, forced-zero 0.40); the -1.0/-2.0 bounds moved 0.40/0.80 -> 0.60/1.00
    (estimator 0.55/0.90, forced-zero 0.70/2.65 — the old bounds do not hold: 0.55 > 0.40, 0.90 > 0.80).
    The onset window in `moonpilot/lead.py` is worth one frame at -0.5 s and
    none at -2.0 with the credit at half a second, so what separates the arms below is the estimator
    itself, not the window.

    The floor is what makes any of it that early: without a credit the same flight reaches -1.0 at
    0.55 s and -2.0 at 0.95, and a *bare* TTC approach term is later still — it keys off the closing
    rate, which only grows once the lead's *speed* has fallen. The credit is what buys the onset back,
    and its length is the whole trade (see `MOONPILOT_LEAD_PREVIEW_T`): half a second here against the
    full second's 0.15 / 0.30, which is what a lead tapping its brakes was paying for.
    """
    planner = _planner()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_ego, v_lead = 20.0, 20.0
    gap = _gap_target(v_ego, t_follow)
    reached = {}
    since_onset = None
    for frame in range(120):  # 2 s of settled following, then the lead brakes at -3.5 m/s^2
      if frame * DT_MDL >= 2.0:
        v_lead = max(0.0, v_lead - 3.5 * DT_MDL)
        since_onset = DT_MDL if since_onset is None else since_onset + DT_MDL
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_lead, model_prob=0.95), standstill=v_ego < 0.1))
      if since_onset is not None:
        for threshold in (-0.5, -1.0, -2.0):
          if threshold not in reached and planner.output_a_target <= threshold:
            reached[threshold] = since_onset
      v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
      gap = max(0.0, gap - (v_ego - v_lead) * DT_MDL)
    # Between the arms with margin on both sides: the forced-zero estimator needs 0.70 s to -1.0 and
    # 2.65 to -2.0, so these separate it, and the no-credit law needs 0.55 / 0.95.
    self.assertLessEqual(reached[-0.5], 0.35)
    self.assertLessEqual(reached[-1.0], 0.60)
    self.assertLessEqual(reached[-2.0], 1.00)

  def test_a_stale_lead_is_planned_as_if_it_were_fresh(self):
    """radarState lands a full cycle behind the modelV2 tick that polls it, and both inputs get
    projected forward by the same `action_t`. Measured over 22.8k planner ticks on 25 routes: median
    age 46.2 ms for the lead against 4.1 ms for `carState`, radar the staler input on 100 % of them.

    The invariant is that the staleness stops mattering, and the harness has to be physically
    consistent for that to mean anything: the message's *content* is delayed by `radar_age_s` and the
    stamp says so. Stamping a current gap as old instead would double-count, because the correction
    would then subtract closing distance that was never lost.

    Measured stop gap against a fresh lead (5.2583 m), stopping at a stationary lead from 25 m/s:
    with the correction 0.046 s keeps it to -0.003, 0.1 s -0.008, 0.2 s -0.026; without it the same
    cases lose 0.035, 0.096 and 0.181 m. The invariant that holds all the way to the 0.5 s ceiling is
    weaker but load-bearing — corrected is never worse than uncorrected — which is what the second
    loop below pins, and what fails if `lead_age` is dropped.
    """

    def fly(content_age, stamp_age):
      """Message content delayed by `content_age`, stamped as `stamp_age`. The two differ only in the
      uncorrected arm, where the content is stale but the planner is told it is not."""
      planner = _planner()
      v_ego, v_lead, gap = 25.0, 0.0, 400.0
      history = []
      for _ in range(3000):
        history.append(gap)
        reported = history[max(0, len(history) - 1 - int(round(content_age / DT_MDL)))]
        planner.update(_inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(reported, v_lead), standstill=v_ego < 0.05, radar_age_s=stamp_age))
        v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
        gap = max(0.0, gap - v_ego * DT_MDL)
        if v_ego < 0.05 and abs(planner.output_a_target) < 0.05:
          break
      return gap

    def corrected(age):
      return fly(age, age)

    def uncorrected(age):
      return fly(age, 0.0)

    fresh = corrected(0.0)
    self.assertGreater(fresh, 5.0)  # it stops
    # realistic staleness: the stop is the fresh one to within 3 cm, up to 0.2 s — four times the
    # 46 ms measured on the corpus. The tight bound is calibrated to that range on purpose: past it
    # the correction loses accuracy because the lead's own acceleration over the stale interval is not
    # modeled (0.3 s -0.015, 0.4 s -0.026, 0.5 s -0.040), which is graceful rather than broken.
    for age in (0.05, 0.1, 0.2):
      with self.subTest(radar_age_s=age):
        self.assertAlmostEqual(corrected(age), fresh, delta=0.03, msg=f"{age} s of lead staleness moved the stop")
    # and the invariant that holds across the whole reachable range: corrected is never worse than
    # uncorrected. 0.5 s is the ceiling the correction is designed against, since upstream's own
    # `commIssue` disengages at ten radar periods; beyond it both degrade together and the car is
    # already disengaging, so nothing is claimed there.
    for age in (0.046, 0.1, 0.2, 0.3, 0.5):
      with self.subTest(radar_age_s=age, invariant="never worse"):
        c, u = corrected(age), uncorrected(age)
        self.assertGreaterEqual(c, u, f"{age} s of staleness made the correction stop closer than no correction")
        if age <= 0.2:
          self.assertGreater(c, 5.0)  # still a real stop, not a degraded one

  def test_a_lead_with_no_stamp_gets_no_correction(self):
    """The zero case, which is what keeps this out of the maneuver plant's way: a SubMaster with no
    radar stamp at all — the plant hands a bare dict with no `logMonoTime`, and the test harness
    defaults the two stamps equal — must plan exactly as it did before the correction existed."""
    planner = _planner()
    planner.update(_inputs(v_ego=25.0, lead=_lead(60.0, 0.0), radar_age_s=0.0))
    with_stamps = planner.output_a_target

    planner = _planner()
    sm = _inputs(v_ego=25.0, lead=_lead(60.0, 0.0), radar_age_s=0.0)
    sm.logMonoTime = {"modelV2": 1e9}  # no radarState stamp, as the maneuver plant's dict has none
    planner.update(sm)
    self.assertAlmostEqual(planner.output_a_target, with_stamps, delta=1e-12)

    planner = _planner()
    sm = _inputs(v_ego=25.0, lead=_lead(60.0, 0.0), radar_age_s=0.0)
    del sm.logMonoTime  # a plain dict, as the plant hands it
    planner.update(sm)
    self.assertAlmostEqual(planner.output_a_target, with_stamps, delta=1e-12)

  def test_the_rollout_runs_on_the_estimators_own_decay(self):
    """The applied plan is rolled forward over 2.5 s, and the decay it applies has to be the
    estimator's own rather than the one the message carries.

    Sampled 0.80 s past the onset rather than 0.30 s, for margin: at 0.80 s the fork's own decay
    (0.278 in this scenario) puts the deepest published point at -1.68 m/s^2 where radard's 1.5 s
    decay of the *same* accel has already decayed it to -0.01, a separation of 1.674 against the 0.8
    asserted; at 0.30 s the same comparison is 0.629, which passes but with a quarter of the room.
    Forcing the decay off entirely reads -2.62 there, so the three sit in the order the decay model
    implies.

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
        gap = _gap_target(v_ego, t_follow)
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
    self.assertAlmostEqual(plan.aTarget, planner.output_a_target, delta=1e-6)  # the field is Float32
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


class TestModelBraking(unittest.TestCase):
  """`MoonpilotModelBraking`: the model's own accel as a braking-only candidate outside experimental
  mode, behind a deadband and otherwise bounded only by the actuator.

  Every case is open-loop — the same `_inputs` for 40 frames at 20 m/s against a 30 m/s set speed and
  no lead — so the cruise candidate is firmly positive and the model is the only thing that can
  brake. 40 frames is past the jerk limit's reach in every one of them: the ramp to `ACCEL_MIN` is
  `MOONPILOT_JERK_EMERGENCY` at 10.0 m/s^3, i.e. 0.5 m/s^2 per 0.05 s frame, 7 frames.
  """

  @staticmethod
  def _run(params_on=True, frames=40, **inputs):
    planner = _planner(params_on=params_on)
    sm = _inputs(**inputs)
    for _ in range(frames):
      planner.update(sm)
    return planner

  def test_the_deadband_keeps_a_shallow_model_ask_out_of_the_command(self):
    """-0.4 m/s^2 is below the threshold, so the model is not in the comparison at all — not merely
    losing it. Without that, a model dithering either side of zero would ratchet the cruise term
    down one-way and the car would settle below the set speed on a clear road."""
    self.assertLess(-0.4, 0.0)
    self.assertGreater(-0.4, MOONPILOT_MODEL_BRAKE_THRESHOLD)
    asked, quiet = self._run(model_accel=-0.4), self._run(model_accel=0.0)
    self.assertAlmostEqual(asked.output_a_target, quiet.output_a_target, delta=1e-12)
    self.assertEqual(asked.source, quiet.source)
    self.assertEqual(asked.source, Source.cruise)

  def test_a_real_model_brake_reaches_the_command(self):
    planner = self._run(model_accel=-1.5)
    self.assertAlmostEqual(planner.output_a_target, -1.5)
    self.assertEqual(planner.source, Source.e2e)

  def test_a_deep_ask_reaches_the_actuator_floor_not_a_fork_floor(self):
    """Past the deadband the ask goes in whole. What bounds it is `ACCEL_MIN`, and only that — a
    fork-owned floor above it was measured and removed, because the corpus showed it withholding
    braking the driver had already started."""
    planner = self._run(model_accel=-6.0)
    self.assertAlmostEqual(planner.output_a_target, ACCEL_MIN)
    self.assertEqual(planner.source, Source.e2e)

  def test_an_emergency_ask_gets_the_emergency_jerk(self):
    """The bound is the actuator's own, and reaching it is not rate-limited by the comfort jerk: a
    deep ask moves the command at `MOONPILOT_JERK_EMERGENCY`, which is what makes full authority
    usable in the time an emergency leaves."""
    planner = self._run(model_accel=-6.0, frames=2)
    # one frame from a standing positive command: 0.5 m/s^2 of the emergency ramp, not 0.1 of it
    self.assertLess(planner.output_a_target, -0.4)

  def test_the_toggle_off_is_the_planner_without_the_model(self):
    planner = self._run(params_on=False, model_accel=-6.0)
    self.assertGreater(planner.output_a_target, 0.0)
    self.assertEqual(planner.source, Source.cruise)

  def test_experimental_mode_is_unchanged(self):
    """The accel goes in raw there — no deadband, and the toggle is not read at all, since
    experimental mode means the model's own accel is the candidate."""
    for params_on in (True, False):
      with self.subTest(params_on=params_on):
        planner = self._run(params_on=params_on, experimental=True, model_accel=-0.3)
        self.assertAlmostEqual(planner.output_a_target, -0.3)
        self.assertEqual(planner.source, Source.e2e)

  def test_the_model_shouldstop_stays_experimental_only(self):
    """`should_stop(v_ego, desiredAcceleration)` from modeld is the creep state the should-stop
    trailing gate exists to exclude, so unlike the accel it is admitted in experimental mode only."""
    self.assertFalse(self._run(model_should_stop=True).output_should_stop)
    self.assertTrue(self._run(experimental=True, model_should_stop=True).output_should_stop)

  def test_the_model_candidate_never_reduces_braking(self):
    """`policy` takes the minimum, so the candidate can only ever add braking — the invariant that
    lets a model signal sit in front of the verified deterministic law. Swept over the deterministic
    terms' regimes; the deepest ask is well past both the deadband and `ACCEL_MIN`, so the pass-
    through cannot hide an ordering change."""
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    v_cruise = 108.0 * CV.KPH_TO_MS
    model = messaging.new_message("modelV2").modelV2
    for v_ego in (0.0, 5.0, 20.0, 33.0):
      for gap in (8.0, 20.0, 60.0, 200.0):
        for v_lead in (0.0, 10.0, 20.0):
          leads = [(Source.lead0, gap, v_lead, 0.0)]
          deterministic, _ = policy(v_ego, leads, v_cruise, t_follow, False, None, 0.0, CP, -0.3, True)
          for a in (-6.0, -2.0, -0.6, 0.0, 2.0):
            model.action.desiredAcceleration = a
            candidate = model_candidate(model, False, True)
            braked, _ = policy(v_ego, leads, v_cruise, t_follow, False, candidate, 0.0, CP, -0.3, True)
            self.assertLessEqual(braked, deterministic + 1e-12)

  def test_the_feature_row_matches_the_params_default(self):
    feature = next(f for f in FEATURES if f.key == "MoonpilotModelBraking")
    # It changes driving behavior, so the row is offroad-only. It ships on: the candidate can only
    # add braking, and the corpus behind that is in `model_candidate`.
    self.assertTrue(feature.offroad_only)
    self.assertFalse(feature.requires)
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('{"MoonpilotModelBraking", {PERSISTENT, BOOL, "1"}}' in text)


class TestCurveSpeed(unittest.TestCase):
  """`MoonpilotCurveSpeed`, through the planner's own seam (`moonpilot/tests/test_curve.py` holds the
  pure math). The two things worth pinning here are the inertness — no path arrays, or a paramsd that
  has not validated its calibration, is exactly the planner this fork had before this existed — and
  that both terms reach the command *and* the published rollout.

  `_inputs` leaves `vehicleParameters.valid` false and the path arrays empty by default, which is also
  why the maneuver suite below is untouched: its plant fills `position` and `velocity` but never
  `orientationRate`, and its `vehicleParameters` message is bare.
  """

  V_EGO = 20.0
  # A curve the car can hold at the budget: 1 / 0.006 is a 167 m radius, and at 20 m/s the in-curve
  # setpoint lands below the ego speed by ~1.34 m/s, i.e. inside the term's proportional range rather
  # than pinned on its floor — so the test reads the gain, not the clamp.
  KAPPA = 0.006

  @staticmethod
  def _steer_angle_for(curvature: float, v_ego: float) -> float:
    """The steering angle whose *measured* curvature is `curvature`: `controlsd` and this planner both
    read `-VM.calc_curvature(radians(angle), v, roll)`, and `get_steer_from_curvature` is that model's
    own inverse, so the angle is exact rather than fitted."""
    vm = VehicleModel(_cp())
    vm.update_params(1.0, 15.0)
    return math.degrees(vm.get_steer_from_curvature(-curvature, v_ego, 0.0))

  def _ramp_curve(self, v_path, onset, ramp=40.0, curvature=0.004):
    """A curve that ramps in over `ramp` meters from `onset` and then holds — the shape a path has, and
    the shape the jerk ceiling is meant for. A step would be read as a jerk spike by `np.gradient`."""
    x, _, _ = _path(v_path, 0.0)
    return _path(v_path, np.clip((x - onset) / ramp, 0.0, 1.0) * curvature)

  # The curve the closed loop flies at, and the road layout it sits on: the car starts `START_ONSET`
  # meters short of a curve that ramps in over `RAMP` and then holds. The set speed is the starting
  # speed, so nothing accelerates on the way in and the approach is the feature's alone.
  CLOSED_LOOP_ONSET = 150.0
  CLOSED_LOOP_V0 = 25.0
  CLOSED_LOOP_RAMP = 40.0
  CLOSED_LOOP_KAPPA = 0.004

  def _fly_to_the_curve(self, scale=None, curvature=None):
    """The closed loop: the path is rebuilt each frame from the car's own travel, so the curve is a
    fixed piece of road, and the speed is integrated from the command. Returns the speed where the
    curve reaches full strength (the end of its ramp — measuring at the ramp's *start* would measure a
    target speed that is still rising), the deepest command, and the speeds along the way."""
    curvature = self.CLOSED_LOOP_KAPPA if curvature is None else curvature
    v_ego, travel = self.CLOSED_LOOP_V0, 0.0
    planner = _planner()
    if scale is not None:
      planner.lat_bias.seed(scale, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    speeds, deepest = [v_ego], 0.0
    for _ in range(4000):
      x, _, _ = _path(v_ego, 0.0)
      path = _path(v_ego, np.clip((x + travel - self.CLOSED_LOOP_ONSET) / self.CLOSED_LOOP_RAMP, 0.0, 1.0) * curvature)
      planner.update(_inputs(v_ego=v_ego, v_cruise_kph=self.CLOSED_LOOP_V0 * 3.6, path=path, standstill=v_ego < 0.05))
      self.assertGreaterEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
      deepest = min(deepest, planner.output_a_target)
      v_ego = max(0.0, v_ego + planner.output_a_target * DT_MDL)
      travel += v_ego * DT_MDL
      speeds.append(v_ego)
      if travel >= self.CLOSED_LOOP_ONSET + self.CLOSED_LOOP_RAMP:
        return v_ego, deepest, np.array(speeds)
    self.fail("the car never reached the curve")

  def test_no_path_arrays_is_the_planner_without_the_feature(self):
    """The inertness statement the maneuver suite rests on: a model message with no path gives no
    candidate at all, so the command is identical with the feature on and off."""
    self.assertIsNone(curve_targets(_inputs()['modelV2'], True))
    on, off = _planner(), _planner(params_on=False)
    for _ in range(20):
      on.update(_inputs())
      off.update(_inputs())
    self.assertAlmostEqual(on.output_a_target, off.output_a_target, delta=1e-12)
    self.assertEqual(on.source, off.source)

  def test_the_toggle_off_is_the_cruise_planner_with_a_path_present(self):
    """The other half of the inertness: the same curved path with the feature off is the planner
    without the feature, still cruising to its set speed."""
    path = self._ramp_curve(30.0, 60.0)
    off = _planner(params_on=False)
    for _ in range(20):
      off.update(_inputs(v_ego=30.0, v_cruise_kph=130.0, path=path))
    self.assertGreater(off.output_a_target, 0.0)
    self.assertEqual(off.source, Source.cruise)

  def test_it_pre_brakes_before_the_curve(self):
    """The pre-brake is the stopping floor's own kinematics against the curve's distance: from 30 m/s
    towards a target of `sqrt(A_LAT / 0.004)` it is negative at any distance the preview admits, and
    bounded by the term's own floor rather than by `ACCEL_MIN`."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertLess(planner.output_a_target, 0.0)
    self.assertGreaterEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
    self.assertEqual(planner.source, Source.cruise)

  def test_the_path_age_is_the_consuming_tick_not_the_publish_stamp(self):
    """`recv_time` is stamped when the planner reads the message — msgq buffers, so it is the age the
    shift must use — and the publish lag is only the fallback for a clock that does not line up with
    the stamps. The two are not the same number: measured over the corpus, the interval from the
    model publish to the plan's own send is ~16 ms median on top of the ~31 ms publish lag."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(**ages):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, **ages))
      return planner.output_a_target

    plain = command(model_age_s=0.05)
    tick = command(model_age_s=0.05, recv_age_s=0.03)  # the tick sees 0.08 s
    self.assertLess(tick, plain, "the tick anchor did not deepen the pre-brake")
    self.assertAlmostEqual(tick, command(model_age_s=0.08), delta=1e-12)

  def test_an_unusable_clock_falls_back_to_the_publish_lag(self):
    """A replay reads wall time against a log's stamps, so its tick age is absurd; the publish lag is
    computable from the stamps alone and is the fallback. A tick that is merely *late* — past
    `MOONPILOT_CURVE_PATH_MAX_AGE` but inside the sanity bound — is a real staleness and drops the
    shift instead, the same direction the bound always had."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(**ages):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, **ages))
      return planner.output_a_target

    plain = command(model_age_s=0.05)
    self.assertAlmostEqual(command(model_age_s=0.05, recv_age_s=1e6), plain, delta=1e-12)
    self.assertAlmostEqual(command(model_age_s=0.05, recv_age_s=0.5), command(model_age_s=0.0), delta=1e-12)

  def test_the_pre_brake_is_the_terms_own_floor_at_the_deepest_point(self):
    """At 30 m/s even a 60 m approach to a 21.79 m/s target needs more than the term's authority, so
    what bounds this is `MOONPILOT_CURVE_ACCEL_MIN` and not the geometry — the actuator's own
    `ACCEL_MIN` is never reached, which is the point of the term's floor."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertAlmostEqual(planner.output_a_target, MOONPILOT_CURVE_ACCEL_MIN, delta=1e-6)
    self.assertGreater(MOONPILOT_CURVE_ACCEL_MIN, ACCEL_MIN)

  def test_the_closed_loop_arrives_at_the_target(self):
    """Flown, not sampled: the arrival bound is the target plus 1 m/s of slack for the jerk-limited
    onset, and the floor is well below it — the term is a braking authority that converges on the
    target, not a speed limiter with its own dynamics.

    `MOONPILOT_CURVE_PREVIEW_T` admits 4 s of path, ~100 m at 25 m/s, and slowing 25 to 21.79 at the
    term's -1.5 m/s^2 floor needs 50 m, so the approach is inside the term's authority. At 30 m/s the
    same arithmetic does not close — 142 m of braking against a 117 m preview — which is why this
    flies at 25.
    """
    v_target = math.sqrt(MOONPILOT_CURVE_A_LAT / self.CLOSED_LOOP_KAPPA)
    arrival, deepest, speeds = self._fly_to_the_curve()
    self.assertLessEqual(arrival, v_target + 1.0)
    self.assertGreaterEqual(arrival, 14.0)
    self.assertGreaterEqual(deepest, MOONPILOT_CURVE_ACCEL_MIN - 1e-9)
    # it braked on the way in rather than arriving at the curve at its starting speed
    self.assertLess(arrival, self.CLOSED_LOOP_V0 - 1.0)
    self.assertTrue(np.all(np.diff(speeds) <= 1e-9), "the speed rose on the way in")

  def test_a_learned_scale_slows_the_arrival_and_below_one_is_ignored(self):
    """The one-sided clamp, end to end: seeded above 1.0 the car plans for a tighter curve and arrives
    slower, and seeded below it the scale is not a scale — `applied()` floors at 1.0, because the
    opposite direction would hand the feature less braking than the geometry justifies."""
    plain, _, _ = self._fly_to_the_curve()
    biased, _, _ = self._fly_to_the_curve(scale=1.4)
    ignored, _, _ = self._fly_to_the_curve(scale=0.7)
    self.assertLess(biased, plain)
    self.assertAlmostEqual(ignored, plain, delta=1e-9)

  def test_the_in_curve_hold_brakes_with_no_path_at_all(self):
    """The in-curve regulator needs no preview: the curvature is the one the car is *pulling*, from the
    steer angle and the vehicle model. This input carries no path arrays, so nothing but this term can
    be braking.

    The command is the term evaluated where the car will be at the actuator delay — every candidate in
    this planner is, and here that is visible in the magnitude: a proportional gain on a speed that the
    command itself moves solves to `K * (v_hold - v_ego) / (1 + K * action_t)`, i.e. 0.536 of the error
    rather than 0.6. Pinned rather than tolerated, because it is the composition the whole planner
    rests on.
    """
    planner = _planner()
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    sm = _inputs(v_ego=self.V_EGO, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0)
    self.assertEqual(len(sm['modelV2'].position.x), 0, "the input was meant to carry no path at all")
    for _ in range(60):
      planner.update(sm)
    v_hold = math.sqrt(MOONPILOT_CURVE_HOLD_MARGIN * MOONPILOT_CURVE_A_LAT / self.KAPPA)
    gain, action_t = MOONPILOT_CURVE_K_HOLD, _cp().longitudinalActuatorDelay + DT_MDL
    expected = max(gain * (v_hold - self.V_EGO) / (1.0 + gain * action_t), MOONPILOT_CURVE_ACCEL_MIN)
    self.assertAlmostEqual(planner.output_a_target, expected, delta=0.01)
    # the unprojected value the term itself computes, for the reader: deeper, and not what is delivered
    self.assertLess(lat_accel_hold(self.V_EGO, v_hold), planner.output_a_target)

  def test_the_in_curve_hold_needs_a_validated_paramsd(self):
    """`vehicleParameters.valid` is paramsd's own composition of its sensor, angle-offset and roll
    validity, and it is false in a bare capnp message — which is what keeps this term out of the
    maneuver plant and out of every existing test that passes a steer angle. Unvalidated, this input
    is the planner with the feature off: the steer angle still reaches `cruise_accel`'s cornering
    budget, and nothing else.
    """
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    inputs = {"v_ego": self.V_EGO, "steer_angle_deg": angle, "steer_ratio": 15.0, "stiffness_factor": 1.0}
    held = _planner()
    for _ in range(60):
      held.update(_inputs(vp_valid=True, **inputs))
    self.assertLess(held.output_a_target, 0.0)

    unvalidated = _planner()
    for _ in range(60):
      unvalidated.update(_inputs(vp_valid=False, **inputs))
    off = _planner(params_overrides={"MoonpilotCurveSpeed": False})
    for _ in range(60):
      off.update(_inputs(vp_valid=True, **inputs))
    self.assertAlmostEqual(unvalidated.output_a_target, off.output_a_target, delta=1e-12)

  def test_the_in_curve_hold_is_inert_at_low_speed(self):
    """Below `MOONPILOT_CURVE_HOLD_MIN_SPEED` the steer angle implies curvatures no plan should chase,
    so the term is off and the cruise term governs."""
    planner = _planner()
    angle = self._steer_angle_for(self.KAPPA, 3.0)
    for _ in range(60):
      planner.update(_inputs(v_ego=3.0, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0))
    self.assertGreater(planner.output_a_target, 0.0)

  def test_the_bench_correction_reaches_the_budget(self):
    """A bank adds budget to a left turn and takes it from a right one, so the same measured curvature
    on a banked road asks for a different speed. One angle, two rolls, two commands."""
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    commands = []
    for roll in (-0.05, 0.05):
      planner = _planner()
      for _ in range(60):
        planner.update(_inputs(v_ego=self.V_EGO, steer_angle_deg=angle, vp_valid=True, steer_ratio=15.0, stiffness_factor=1.0, roll=roll))
      commands.append(planner.output_a_target)
    self.assertNotAlmostEqual(commands[0], commands[1], delta=0.01)

  def test_the_bias_only_learns_while_lateral_is_ours(self):
    """What makes the learned scale a measurement of this car rather than of whoever is steering: a
    disengaged or overridden frame pairs a prediction about the model's path with a measurement of the
    driver's own steering, so it teaches nothing."""
    path = _path(self.V_EGO, self.KAPPA)
    angle = self._steer_angle_for(self.KAPPA, self.V_EGO)
    inputs = {"v_ego": self.V_EGO, "path": path, "steer_angle_deg": angle, "vp_valid": True, "steer_ratio": 15.0, "stiffness_factor": 1.0}

    learned = _planner()
    for _ in range(100):
      learned.update(_inputs(**inputs))
    self.assertGreater(learned.lat_bias.samples, 0)

    for disengaged_kwargs in ({"lat_active": False}, {"steering_pressed": True}):
      with self.subTest(**disengaged_kwargs):
        planner = _planner()
        for _ in range(100):
          planner.update(_inputs(**inputs, **disengaged_kwargs))
        self.assertEqual(planner.lat_bias.samples, 0)

  def test_the_path_is_referenced_from_the_frame_the_model_saw(self):
    """`modelV2.position.x` is measured from the pose of the frame the model saw — `T_IDXS` is
    published unshifted (`fill_model_msg.py:84`) — while every candidate in the `min` is evaluated at
    model publish + `action_t`, which is where `lead_age` puts the lead pair. So the pre-brake has to
    read the path `v_ego * (logMonoTime - timestampEof)` of the way in, or it brakes as if the car
    were still where that frame was taken. Measured publish lag over seven corpus segments: 29 ms
    median, 34 p95, 38 max — ~0.9 m at 30 m/s.

    The geometry is chosen so the answer is the term's own arithmetic rather than its floor: a 40 m
    ramp to a 250 m radius starting 20 m out, at 30 m/s, where one sample binds at ~66 m and the ask
    is ~-0.25 m/s². A 3 m shift (the bound, 0.1 s) is then `|a| * 3 / 66` ≈ 0.013 deeper.
    """
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def settle(age):
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_ego=30.0, path=path, model_age_s=age))
      return planner

    plain, shifted = settle(0.0), settle(0.1)
    delta = plain.output_a_target - shifted.output_a_target
    self.assertEqual(shifted.source, plain.source)
    self.assertGreater(delta, 0.005, "the shift did not reach the command")
    self.assertLess(delta, 0.05, "the shift moved the command by more than the term's own arithmetic")
    self.assertLess(float(shifted.a_desired_trajectory.min()), float(plain.a_desired_trajectory.min()))

  def test_the_path_age_is_bounded_and_absent_stamps_are_the_old_planner(self):
    """Two model periods, the same bound `moonpilot/curvature.py` puts on the age of this same
    message: past it the shift is dropped rather than extrapolated, and a missing, zero or impossible
    stamp is the zero shift this planner had before it existed. That direction matters — a shift that
    is too large empties the pre-brake's binding set, which stops it asking at all — so the guard is a
    window rather than `max(0.0, ...)`."""
    path = self._ramp_curve(30.0, 20.0, curvature=0.0022)

    def command(sm):
      planner = _planner()
      for _ in range(40):
        planner.update(sm)
      return planner.output_a_target

    plain = command(_inputs(v_ego=30.0, path=path))
    for age in (MOONPILOT_CURVE_PATH_MAX_AGE + 1e-9, 0.5):
      with self.subTest(model_age_s=age):
        self.assertAlmostEqual(command(_inputs(v_ego=30.0, path=path, model_age_s=age)), plain, delta=1e-9)

    unset = _inputs(v_ego=30.0, path=path, model_age_s=0.1)
    unset['modelV2'].timestampEof = 0  # a message that never carried one
    self.assertAlmostEqual(command(unset), plain, delta=1e-9)

    no_stamp = _inputs(v_ego=30.0, path=path, model_age_s=0.1)
    no_stamp.logMonoTime.pop('modelV2')  # a bare dict, as the maneuver plant hands the planner
    no_stamp.recv_time.pop('modelV2')  # and no tick stamp: the tick anchor is the primary one now
    self.assertAlmostEqual(command(no_stamp), plain, delta=1e-9)

  def test_the_rollout_carries_the_curve(self):
    """The published plan is the policy the command came from, so the rollout has to see the same
    terms — otherwise the plan reads as if the car were still cruising."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_ego=30.0, path=self._ramp_curve(30.0, 60.0)))
    self.assertLess(float(planner.a_desired_trajectory.min()), 0.0)
    self.assertTrue(np.all(np.diff(planner.v_desired_trajectory) <= 1e-9))

    without = _planner()
    for _ in range(40):
      without.update(_inputs(v_ego=30.0))
    self.assertTrue(np.all(without.a_desired_trajectory >= 0.0))
    self.assertTrue(np.all(np.diff(without.v_desired_trajectory) >= -1e-9))

  def test_the_learned_scale_persists(self):
    """The value handed to the next drive, on the estimator's own cadence and gate: a trusted
    estimate only, so an unestimated one never becomes the next boot's scale."""
    planner = _planner()
    planner.lat_bias.seed(1.2, MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    for _ in range(MOONPILOT_CURVE_BIAS_PERSIST_EVERY):
      planner.update(_inputs())
    self.assertEqual(planner.params.puts.get(MOONPILOT_CURVE_BIAS_KEY), 1.2)

    measuring = _planner()
    for _ in range(MOONPILOT_CURVE_BIAS_PERSIST_EVERY):
      measuring.update(_inputs())
    self.assertNotIn(MOONPILOT_CURVE_BIAS_KEY, measuring.params.puts)

  def test_the_longitudinal_jerk_scale_persists_only_after_estimation(self):
    planner = _planner()
    planner.long_jerk.seed(0.75, MOONPILOT_LONG_JERK_MIN_SAMPLES)
    for _ in range(MOONPILOT_LONG_JERK_PERSIST_EVERY):
      planner.update(_inputs())
    self.assertEqual(planner.params.puts.get(MOONPILOT_LONG_JERK_SCALE_KEY), 0.75)

    measuring = _planner()
    for _ in range(MOONPILOT_LONG_JERK_PERSIST_EVERY):
      measuring.update(_inputs())
    self.assertNotIn(MOONPILOT_LONG_JERK_SCALE_KEY, measuring.params.puts)

  def test_the_feature_rows_match_the_params_defaults(self):
    """Both behaviors ship off: each one is unvalidated on a car, and both are read with
    `enabled()`, so the row and the param default have to agree."""
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    for key in ("MoonpilotCurveSpeed", "MoonpilotPathPreview"):
      with self.subTest(key=key):
        feature = next(f for f in FEATURES if f.key == key)
        self.assertTrue(feature.offroad_only)
        self.assertFalse(feature.requires)
        self.assertTrue(f'{{"{key}", {{PERSISTENT, BOOL, "0"}}}}' in text)
    # The learned scale is a value, not a toggle: nothing may gate on it, and its neutral default has
    # to be the neutral ratio rather than the "unset" the lag param uses.
    self.assertTrue('{"MoonpilotCurveLatScale", {PERSISTENT, FLOAT, "1.0"}}' in text)
    self.assertTrue('{"MoonpilotLongJerkScale", {PERSISTENT, FLOAT, "1.0"}}' in text)


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
