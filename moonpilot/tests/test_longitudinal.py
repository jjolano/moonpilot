"""The fork longitudinal planner: the policy's numbers, and the properties the seam depends on.

The last test is the load-bearing one — it runs upstream's own maneuver suite against this planner,
which is the end-to-end statement that the fork strategy drives the 15 stock scenarios without a
crash, without stalling at a stop, and while still decelerating under forceDecel.
"""

import itertools
import math
import unittest
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.toyota.interface import CarInterface as ToyotaInterface
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
import openpilot.selfdrive.test.longitudinal_maneuvers.plant as plant_mod
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers


from moonpilot.features import FEATURES
from moonpilot.lead import MOONPILOT_LEAD_ACCEL_TAU, LeadAccelEstimator
from moonpilot.longitudinal import (
  MOONPILOT_APPROACH_DECEL,
  MOONPILOT_ARRIVE_MIN_T,
  MOONPILOT_ARRIVE_RAMP_T,
  MOONPILOT_ARRIVE_T,
  MOONPILOT_BRAKE_LIGHT,
  MOONPILOT_BRAKE_MEDIUM,
  MOONPILOT_CATCH_UP_CLOSING,
  MOONPILOT_CATCH_UP_SLACK,
  MOONPILOT_COAST_BAND,
  MOONPILOT_COAST_FLAT_ACCEL,
  MOONPILOT_COAST_GRADE_MIN,
  MOONPILOT_COAST_RECOVERY_ACCEL,
  MOONPILOT_CONTROL_T_IDX,
  MOONPILOT_CREEP_SPEED,
  MOONPILOT_CRUISE_ERR_BP,
  MOONPILOT_ERR_BP_SCALE,
  MOONPILOT_FAST_ACCEL_BP,
  MOONPILOT_FAST_ACCEL_V,
  MOONPILOT_FAST_ACCEL_TRACK_BP,
  MOONPILOT_FAST_ACCEL_TRACK_V,
  MOONPILOT_FCW_DECEL,
  MOONPILOT_FLOOR_ADMISSION_BP,
  MOONPILOT_FLOOR_ADMISSION_BAND,
  MOONPILOT_FLOOR_ADMISSION_V,
  MOONPILOT_FOLLOW_BAND_T,
  MOONPILOT_JERK_EMERGENCY,
  MOONPILOT_JERK_DOWN,
  MOONPILOT_JERK_ONSET,
  MOONPILOT_JERK_LAUNCH,
  MOONPILOT_JERK_LAUNCH_SPEED,
  MOONPILOT_JERK_UP,
  MOONPILOT_LADDER_V_BP,
  MOONPILOT_LEAD_PREVIEW_T,
  MOONPILOT_MIN_SLACK,
  MOONPILOT_MODEL_BRAKE_THRESHOLD,
  MOONPILOT_MODEL_STOP_RELEASE,
  MOONPILOT_MODEL_STOP_SPEED,
  MOONPILOT_MOVING_STOP_RESERVE,
  MOONPILOT_OUT_OF_PATH_T_FOLLOW,
  MOONPILOT_PREDICT_T,
  MOONPILOT_SLOW_ACCEL_V,
  MOONPILOT_STOP_DISTANCE,
  MOONPILOT_STOP_LAG_T,
  MOONPILOT_STOP_REST,
  MOONPILOT_STOP_TAPER_T,
  MOONPILOT_TRACK_BRAKE,
  MOONPILOT_T_FOLLOW,
  MOONPILOT_T_FOLLOW_SLEW_DOWN,
  MOONPILOT_T_FOLLOW_SLEW_UP,
  MoonpilotLongitudinalPlanner,
  brake_sustain,
  coast_accel,
  cruise_accel,
  jerk_limit,
  lead_accel,
  lead_state_at,
  model_candidate,
  model_release,
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
# A stop behind a stopped lead aims at the floor's rest anchor and lands within its slack allowance of it, twice over for the
# jerk-limited onset and the lag: the nearest a safe stop may rest, well clear of the 0.4 m contact bar.
MIN_REST_GAP = MOONPILOT_STOP_REST - 2 * MOONPILOT_MIN_SLACK


def _admission(v_ego):
  return float(np.interp(v_ego, MOONPILOT_FLOOR_ADMISSION_BP, MOONPILOT_FLOOR_ADMISSION_V))


def _floor(v_ego, gap, v_lead, a_lead=0.0):
  """The stopping floor's ask and its admission weight (0 out of its band, 1 at and past its plateau)."""
  v_lead_eff = v_lead + min(a_lead, 0.0) * MOONPILOT_LEAD_PREVIEW_T
  anchor = MOONPILOT_STOP_REST + min(MOONPILOT_STOP_TAPER_T * v_ego, MOONPILOT_STOP_DISTANCE - MOONPILOT_STOP_REST)
  slack = max(gap - anchor - MOONPILOT_STOP_LAG_T * (v_ego - v_lead_eff), MOONPILOT_MIN_SLACK)
  a_stop = stopping_decel(v_ego, v_lead_eff, a_lead, slack)
  return a_stop, min(max((-a_stop - _admission(v_ego)) / MOONPILOT_FLOOR_ADMISSION_BAND + 1.0, 0.0), 1.0)


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
  road_edges=None,
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
  if road_edges is not None:
    # `road_edges` is a pair of `(x, y)` sequences, one per road edge, on the model's distance grid.
    model.modelV2.roadEdges = [log.XYZTData.new_message(x=[float(v) for v in ex], y=[float(v) for v in ey]) for ex, ey in road_edges]

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
  """Fly the shipped planner against a scripted lead, point-mass plant, and summarize the run.

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


def _gap_target(v_ego, t_follow, v_lead=None):
  """The time gap over the moving stop floor: the near edge of the lead law's band, where a settled follow starts a test.

  A floor and not an offset — `STOP_DISTANCE + t_follow * v_ego` made the effective headway
  `t_follow + STOP_DISTANCE / v_ego`, so the car hung back further the slower it went. The floor's
  reserve grows with the slower car's speed up to creep speed, so launch at rest still sees the
  standstill distance."""
  v_lead = v_ego if v_lead is None else v_lead
  floor = MOONPILOT_STOP_DISTANCE + MOONPILOT_MOVING_STOP_RESERVE * min(v_ego / MOONPILOT_CREEP_SPEED, v_lead / MOONPILOT_CREEP_SPEED, 1.0)
  return max(floor, t_follow * v_ego)


def _lead_band(v_lead, t_follow=MOONPILOT_T_FOLLOW[int(log.LongitudinalPersonality.standard)], a_slow=0.0):
  """The lead law's band, `(near, far)`, from its own definition: the time gap over the moving rest floor, out to
  `MOONPILOT_FOLLOW_BAND_T` of the lead's predicted speed past it."""
  v_pred = max(v_lead + a_slow * MOONPILOT_PREDICT_T, 0.0)
  rest = MOONPILOT_STOP_REST + (MOONPILOT_STOP_DISTANCE + MOONPILOT_MOVING_STOP_RESERVE - MOONPILOT_STOP_REST) * min(v_pred / MOONPILOT_CREEP_SPEED, 1.0)
  near = max(rest, t_follow * v_pred)
  return near, near + MOONPILOT_FOLLOW_BAND_T * v_pred


def _fast_rung(v_ego):
  """The ladder's fast rung as asked for: the measured driver value times the plant's tracking factor."""
  return float(np.interp(v_ego, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V)) * float(
    np.interp(v_ego, MOONPILOT_FAST_ACCEL_TRACK_BP, MOONPILOT_FAST_ACCEL_TRACK_V))


class TestPolicyFunctions(unittest.TestCase):
  def test_the_stopping_floor_bounds_the_lead_ask_where_it_is_admitted(self):
    """Where the floor is fully admitted it is a bound: the lead law never asks softer than it, far out
    against a stopped lead, at the very end of an approach, and behind a moving lead. Its slack is the
    gap past the rest anchor less `MOONPILOT_STOP_LAG_T` of closing.
    """
    for v_ego, gap, v_lead in ((25.0, 120.0, 0.0), (25.0, 8.0, 0.0), (25.0, 36.0, 15.0), (20.0, 35.0, 10.0)):
      a_stop, weight = _floor(v_ego, gap, v_lead)
      self.assertEqual(weight, 1.0)
      self.assertLessEqual(lead_accel(v_ego, gap, v_lead, 0.0, 1.45), a_stop + 1e-9)

  def test_the_braking_credit_reaches_the_floor_and_a_launch_credit_does_not(self):
    """The lead's own accel is credited into the stopping floor's lead speed in one direction only,
    braking (`MOONPILOT_LEAD_PREVIEW_T`): the cautious one. Releasing the floor on a predicted launch
    is the fork braking for a prediction where the geometry had not moved, so an accelerating lead
    must ask exactly what a steady one does and a braking one must ask more. Pinned where the floor is
    that floor is fully admitted — (20, 35, 10) at 20 m/s — and as an ordering at every state.
    """
    v_ego, gap, v_lead = 20.0, 35.0, 10.0
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]

    for a_lead in (0.0, 2.0, -2.0):
      a_stop, weight = _floor(v_ego, gap, v_lead, a_lead)
      self.assertEqual(weight, 1.0)
      self.assertLessEqual(lead_accel(v_ego, gap, v_lead, a_lead, t_follow), a_stop + 1e-9)
    self.assertEqual(_floor(v_ego, gap, v_lead, 2.0)[0], _floor(v_ego, gap, v_lead, 0.0)[0])  # a launch is not credited
    self.assertLess(_floor(v_ego, gap, v_lead, -2.0)[0], _floor(v_ego, gap, v_lead, 0.0)[0])  # a brake is

    # the instantaneous lead accel can only ever lower the command: braking deeper, a launch not at all
    for v_ego in (5.0, 15.0, 30.0):
      for gap in (20.0, 60.0, 200.0):
        with self.subTest(v_ego=v_ego, gap=gap):
          neutral = lead_accel(v_ego, gap, 10.0, 0.0, 1.45)
          self.assertEqual(neutral, lead_accel(v_ego, gap, 10.0, 2.0, 1.45))
          self.assertGreaterEqual(neutral, lead_accel(v_ego, gap, 10.0, -2.0, 1.45))

  def test_the_up_jerk_is_scheduled_on_speed_and_the_down_jerk_is_not(self):
    """The up limit is `MOONPILOT_JERK_LAUNCH` at and below `MOONPILOT_JERK_LAUNCH_SPEED`, tapering
    back to `MOONPILOT_JERK_UP` by twice it — a launch is the one place the plan's own ask outruns the
    comfort limit, and the car's answer to a lead pulling away is the most visible thing this planner
    does. The down side reads the *command*: a shallow ask onsets at `MOONPILOT_JERK_ONSET` above the
    creep band, and keeps `MOONPILOT_JERK_DOWN` at a crawl, while a lead brakes, and from -2.0 on,
    where it interpolates into the untouched emergency ramp.
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
    for v_ego in (0.0, MOONPILOT_JERK_LAUNCH_SPEED):
      self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, v_ego), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, 25.0), -MOONPILOT_JERK_ONSET * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-1.0, 0.0, DT_MDL, 25.0, lead_braking=True), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)
    self.assertAlmostEqual(jerk_limit(-2.0, 0.0, DT_MDL, 25.0), -MOONPILOT_JERK_DOWN * DT_MDL, places=9)

  def test_the_approach_stops_behind_a_stopped_lead_at_every_reachable_speed(self):
    """The safety property: the fork's car can reach V_CRUISE_MAX on openpilot longitudinal, and it
    must stop behind a stopped lead from every reachable speed. The maneuver suite's fastest
    stopped-lead approach is 25 m/s, so it is pinned closed-loop here. A constant-decel stop aimed at
    a fixed point owes any unseen closing at its end (`MOONPILOT_STOP_LAG_T`), and the stop may land
    anywhere in the floor's rest region, no nearer than its rest anchor less `MOONPILOT_MIN_SLACK`.
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
      # Safety braking remains inside the actuator budget, and the rest gap stays inside the floor's rest region.
      self.assertLess(peak, abs(ACCEL_MIN), f"{v0_kph} kph used the whole actuator limit")
      self.assertLess(v_ego, 0.5)  # it stopped
      self.assertGreater(gap, MIN_REST_GAP)

  def test_a_crawl_stop_tail_does_not_redeepen_as_the_floor_admits(self):
    """A hard admission switch flipped the floor in and out as a 12 m/s stop crossed its plateau: from a
    75 m start the crawl tail re-deepened 0.069 m/s^2 in one frame, 0.1 from 80 m. Blended over
    `MOONPILOT_FLOOR_ADMISSION_BAND` it does not."""
    for gap0 in (75.0, 80.0):
      s = _fly(12.0, 43.2, gap0, 0.0, lambda t: 0.0, 30.0)
      tail = [a for v, a in zip(s["v"], s["cmd"], strict=True) if 0.3 < v < 4.0]
      self.assertLess(max(earlier - later for earlier, later in zip(tail, tail[1:], strict=False)), 0.02)

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
    decel, so the floor is not admitted at all, and the lead law's own ask is not a floor-governed brake.
    Each case computes the old value inline, so the test documents the delta it removes.
    """
    for v_ego, gap, v_lead in ((18.0, 90.0, 12.5), (19.0, 70.0, 14.0), (17.0, 115.0, 11.0)):
      slack = gap - MOONPILOT_STOP_DISTANCE
      new = stopping_decel(v_ego, v_lead, 0.0, slack)
      old = -(v_ego**2 - v_lead**2) / (2 * slack)
      self.assertGreater(new, -MOONPILOT_APPROACH_DECEL)  # not admitted: the floor cannot govern here
      self.assertGreater(old / new, 4.0)  # ~4.7-6.6x across these cases (-0.77..-1.29 vs -0.17..-0.20)
      self.assertEqual(_floor(v_ego, gap, v_lead)[1], 0.0)
      self.assertGreater(lead_accel(v_ego, gap, v_lead, 0.0, 1.45), -MOONPILOT_APPROACH_DECEL)
    # the first case sits exactly on the old admission edge (old -0.9985), which is the point: the old
    # floor governed by default at 70-115 m while the geometry asks for a ramp, and the route's 45/45
    # engaged braking ticks sat on it at -1.0..-1.17.
    # the analyzed scene itself: (19.09, 12.25, -1.28, slack 106.9) reads -0.404, not -1.003
    self.assertAlmostEqual(stopping_decel(19.09, 12.25, -1.28, 106.9), -0.40361775991229676, delta=1e-9)
    self.assertAlmostEqual(-(19.09**2 - 12.25**2) / (2 * 106.9), -1.002645463049579, delta=1e-9)

  def test_a_braking_lead_matched_at_speed_still_stops_without_contact(self):
    """Matched speed must still anticipate a lead braking to rest, not read zero closing and coast
    into it. Exercise both strong lead brakes through the real speed-history estimator."""
    for brake in (-3.5, -5.0):
      with self.subTest(brake=brake):
        # long enough for the creep a stopped lead beyond the band is closed at to arrive and stop
        s = _fly(20.0, 72.0, 29.0, 20.0, lambda t, b=brake: b if t > 1 else 0.0, 70.0)
        self.assertFalse(s["contact"])
        # the stop rests inside the floor's rest region, keeping its one-meter allowance
        self.assertGreater(s["min_gap"], MOONPILOT_STOP_REST - MOONPILOT_MIN_SLACK)
        self.assertLess(s["v"][-1], 0.05)
    # and the direct assertion: at matched speed the prediction branch, not the match, is what binds
    self.assertLess(stopping_decel(20.0, 20.0, -5.0, 23.0), -MOONPILOT_APPROACH_DECEL)

  def test_a_moderate_tap_does_not_buy_a_sustain_stop(self):
    """The bound's other direction: a lead tapping -2 m/s^2 for 1.5 s in front of a 20 m/s follow is
    never answered with more braking than the tap itself, where predicting it to rest asks several times
    the floor's own sustain-bounded ask. Up to -2 the horizon is the fixed `MOONPILOT_LEAD_BRAKE_SUSTAIN_T`;
    a harder tap is believed longer, which is the cost the next test buys (`python -m moonpilot.lab sustain`)."""
    s = _fly(20.0, 80.0, 29.0, 20.0, lambda t: -2.0 if 1 < t < 2.5 else 0.0, 20.0)
    self.assertGreater(s["peak"], -2.0)
    self.assertFalse(s["contact"])
    self.assertGreater(stopping_decel(20.0, 20.0, -2.0, 23.0), stopping_decel(20.0, 20.0, -2.0, 23.0, sustain=math.inf))

  def test_a_hard_brake_is_believed_for_longer(self):
    """`MOONPILOT_LEAD_BRAKE_SUSTAIN_PER_A`: from a 1.45 s follow, a lead braking -3.5 to rest at 30 m/s
    left 0.45 m on a fixed 2 s horizon and -5 at 25 m/s made contact; scaled with the braking, both
    stop clear. This plant integrates ego travel on the updated speed (~0.75 m optimistic from 30 m/s);
    on the trapezoid the fixed horizon contacts in both (`python -m moonpilot.lab sustain`)."""
    s = _fly(30.0, 118.0, 43.5, 30.0, lambda t: -3.5 if t > 1 else 0.0, 25.0)
    self.assertGreater(s["min_gap"], MOONPILOT_STOP_REST - MOONPILOT_MIN_SLACK)
    s = _fly(25.0, 100.0, 36.25, 25.0, lambda t: -5.0 if t > 1 else 0.0, 25.0)
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

  def test_the_error_axis_reaches_cruise_but_never_the_lead_asks(self):
    """The dial is cruise's own axis: with a governing lead the output is one number across all
    three axes — the regulator and stopping floor never read it — while with no lead the
    same three axes order the cruise ask."""
    CP = _cp()
    t_follow = MOONPILOT_T_FOLLOW[int(Personality.standard)]
    axes = [[e * MOONPILOT_ERR_BP_SCALE[p] for e in MOONPILOT_CRUISE_ERR_BP] for p in (0, 1, 2)]
    governed = [policy(2.0, [(Source.lead0, 4.0, 0.0, 0.0)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True, err_bp=axis)[0] for axis in axes]
    self.assertEqual(governed, [governed[0]] * 3)
    self.assertLess(governed[0], 0.0)  # it is the lead asking, not cruise
    cruise = [policy(20.0, [], 22.0, t_follow, False, None, 0.0, CP, -0.3, True, err_bp=axis)[0] for axis in axes]
    self.assertEqual(cruise, sorted(cruise, reverse=True))
    self.assertGreater(cruise[0], cruise[2])


class TestLeadLaw(unittest.TestCase):
  """The properties the lead law has to keep: where to be and by when, continuous through every edge.

  Pinned as behavior, not arithmetic: continuity across each edge it has, silence inside the band, the deadline
  the arrival is defined on, and two closed-loop flights through the shipped planner.
  """

  T_FOLLOW = MOONPILOT_T_FOLLOW[int(Personality.standard)]

  @staticmethod
  def _max_step(fn, lo, hi, n):
    """The largest step of `fn` over `n` evenly spaced samples in `[lo, hi]`."""
    ys = [fn(lo + (hi - lo) * i / (n - 1)) for i in range(n)]
    return max(abs(later - earlier) for earlier, later in zip(ys, ys[1:], strict=False))

  def _assert_continuous(self, fn, lo, hi, msg):
    """A continuous law's largest step halves when the sampling does. A jump does not: it stays the jump's size."""
    coarse, mid, fine = (self._max_step(fn, lo, hi, n) for n in (201, 401, 801))
    self.assertGreater(coarse, 1e-9, f"{msg}: the sweep never moved")
    self.assertLess(mid, 0.7 * coarse, f"{msg}: halving the step did not halve the largest step")
    self.assertLess(fine, 0.7 * mid, f"{msg}: halving the step again did not halve it again")

  def test_the_ask_is_continuous_through_every_edge_of_the_band(self):
    """Zero closing, both band edges, and the `MOONPILOT_ARRIVE_T` onset are all edges of one continuous function.

    The old regulator flipped from +0.975 to 0 across `v_lead` 15.001 -> 14.999 at (15, 60, t_follow 1.45); that
    exact pair is asserted here as its regression. The sweeps cover matched speed (with and without a predicted
    lead accel), the near and far edges, the onset of the arrival's ramp, and the last meters where the arrival's
    room is capped by what is physically left.
    """
    # the regression pair: 0.001 m/s of lead speed across the match, 60 m back, and a predicted launch
    fast, slow = (lead_accel(15.0, 60.0, v, 0.0, 1.45, a_slow=1.0) for v in (15.001, 14.999))
    self.assertLess(abs(fast - slow), 0.01)

    near15, far15 = _lead_band(15.0)
    for gap in (near15, 0.5 * (near15 + far15), far15, far15 + 5.0):
      with self.subTest(gap=gap, edge="matched speed"):
        self._assert_continuous(lambda v, g=gap: lead_accel(15.0, g, v, 0.0, self.T_FOLLOW), 14.5, 15.5, f"closing at {gap} m")
        self._assert_continuous(lambda v, g=gap: lead_accel(15.0, g, v, 0.0, self.T_FOLLOW, a_slow=1.0), 14.5, 15.5, f"closing at {gap} m, launch")
    with self.subTest(edge="near"):
      near = _lead_band(12.0)[0]
      self._assert_continuous(lambda g: lead_accel(16.0, g, 12.0, 0.0, self.T_FOLLOW), near - 3.0, near + 3.0, "near edge")
    with self.subTest(edge="far"):
      far = _lead_band(16.0)[1]
      self._assert_continuous(lambda g: lead_accel(12.0, g, 16.0, 0.0, self.T_FOLLOW), far - 3.0, far + 3.0, "far edge")
    with self.subTest(edge="onset"):
      # closing 4 m/s: the ramp is whole at 24 m of room from the near edge and silent past 36
      near = _lead_band(12.0)[0]
      self._assert_continuous(
        lambda g: lead_accel(16.0, g, 12.0, 0.0, self.T_FOLLOW),
        near + (MOONPILOT_ARRIVE_T - MOONPILOT_ARRIVE_RAMP_T) * 4.0 - 3.0,
        near + MOONPILOT_ARRIVE_T * 4.0 + 3.0,
        "arrival onset",
      )
    with self.subTest(edge="reach"):
      # inside the last meters the arrival's room is the physical gap, less the lag a close stop spends
      self._assert_continuous(lambda g: lead_accel(16.0, g, 12.0, 0.0, self.T_FOLLOW), 4.0, 10.0, "reach floor")

  def test_inside_the_band_at_matched_speed_the_law_asks_nothing(self):
    """Inside the band at matched speed the plan holds: the ask is exactly 0 at every gap from the near edge to
    the far one, and noise that keeps the state inside the band and non-closing moves nothing at all — the whole
    reason the band exists rather than a setpoint."""
    for v in (0.5, MOONPILOT_CREEP_SPEED, 5.0, 15.0, 30.0):
      near, far = _lead_band(v)
      for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        with self.subTest(v=v, frac=frac):
          self.assertLess(abs(lead_accel(v, near + frac * (far - near), v, 0.0, self.T_FOLLOW)), 1e-9)

    # noise: a lead reporting up to 0.1 m/s faster — inside MOONPILOT_LEAVE_DEADBAND — at any gap from the near
    # edge to the far one, moves nothing at all
    v_ego = 20.0
    for v_lead in (20.0, 20.02, 20.1):
      near, far = _lead_band(v_lead)
      for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        with self.subTest(v_lead=v_lead, frac=frac):
          self.assertEqual(lead_accel(v_ego, near + frac * (far - near), v_lead, 0.0, self.T_FOLLOW), 0.0)

  def test_the_arrival_waits_for_its_deadline_and_then_is_the_constant_decel(self):
    """The deadline is the law's `when`: a closing lead whose near edge is more than `MOONPILOT_ARRIVE_T` of
    closing time away is not braked for at all, and once the edge is inside `MOONPILOT_ARRIVE_T -
    MOONPILOT_ARRIVE_RAMP_T` the ask is the constant decel that arrives there speed-matched, plus
    `MOONPILOT_TRACK_BRAKE` of a slowing lead's accel.

    The states are chosen where the stopping floor is not admitted — the closing is small against a moving lead —
    so the arrival is the whole of the ask.
    """
    for v_ego, v_lead, a_slow in ((25.0, 20.0, 0.0), (20.0, 15.0, -2.0)):
      near, far = _lead_band(v_lead, a_slow=a_slow)
      closing = v_ego - max(v_lead + a_slow * MOONPILOT_PREDICT_T, 0.0)
      self.assertGreater(closing, 0.0)
      with self.subTest(v_ego=v_ego, v_lead=v_lead, a_slow=a_slow, phase="waiting"):
        gap = near + (MOONPILOT_ARRIVE_T + 2.0) * closing
        self.assertGreater(gap, far)  # beyond the band, and the catch-up bound is satisfied by the closing
        self.assertEqual(lead_accel(v_ego, gap, v_lead, 0.0, self.T_FOLLOW, a_slow=a_slow), 0.0)
      whole = (MOONPILOT_ARRIVE_T - MOONPILOT_ARRIVE_RAMP_T) * closing
      for room in (MOONPILOT_ARRIVE_MIN_T * closing, 0.5 * whole, whole):
        with self.subTest(v_ego=v_ego, v_lead=v_lead, a_slow=a_slow, room=room):
          expected = -(closing**2 / (2.0 * room) + MOONPILOT_TRACK_BRAKE * max(-a_slow, 0.0))
          self.assertAlmostEqual(lead_accel(v_ego, near + room, v_lead, 0.0, self.T_FOLLOW, a_slow=a_slow), expected, delta=1e-9)

  def test_beyond_the_band_the_closing_it_accelerates_to_is_bounded(self):
    """Far behind a lead, the law catches up at no more than `MOONPILOT_CATCH_UP_CLOSING`: at the bound and past
    it the ask is exactly 0, and below it the ask is positive — the band is approached by holding a small closing,
    not by running at the lead. The first `MOONPILOT_CATCH_UP_SLACK` past the far edge is left alone, so a car
    parked a little long behind a stopped lead stays put."""
    v_ego = 20.0
    _, far = _lead_band(v_ego - MOONPILOT_CATCH_UP_CLOSING)
    self.assertEqual(lead_accel(v_ego, far + 50.0, v_ego - MOONPILOT_CATCH_UP_CLOSING, 0.0, self.T_FOLLOW), 0.0)
    self.assertEqual(lead_accel(v_ego, far + 50.0, v_ego - 2 * MOONPILOT_CATCH_UP_CLOSING, 0.0, self.T_FOLLOW), 0.0)
    self.assertGreater(lead_accel(v_ego, far + 50.0, v_ego - 0.5 * MOONPILOT_CATCH_UP_CLOSING, 0.0, self.T_FOLLOW), 0.0)
    self.assertGreater(lead_accel(v_ego, far + 50.0, v_ego + 1.0, 0.0, self.T_FOLLOW), 0.0)
    # and the deadband past the edge: silent inside it, creeping outside it
    self.assertEqual(lead_accel(v_ego, far + 0.5 * MOONPILOT_CATCH_UP_SLACK, v_ego, 0.0, self.T_FOLLOW), 0.0)
    self.assertGreater(lead_accel(v_ego, far + 1.5 * MOONPILOT_CATCH_UP_SLACK, v_ego, 0.0, self.T_FOLLOW), 0.0)

  def test_a_slower_lead_is_arrived_at_speed_matched_without_a_pump(self):
    """16 m/s onto a steady 11.7 m/s lead from 119 m: it arrives speed-matched, inside the band, without contact,
    and the command never re-deepens after it has eased — the deadline is planned once, not pumped as the gap
    crosses an edge."""
    s = _fly(16.0, 108.0, 119.0, 11.7, lambda t: 0.0, 120.0)
    near, far = _lead_band(s["vl"][-1])
    self.assertFalse(s["contact"])
    self.assertAlmostEqual(s["v"][-1], s["vl"][-1], delta=0.1)
    self.assertGreaterEqual(s["end_gap"], near - 0.5)
    self.assertLessEqual(s["end_gap"], far + 1.0)
    # no pump: after easing off the deepest ask, the command never deepens more than the easing itself
    cmds, window = s["cmd"], round(2.0 / DT_MDL)
    worst = 0.0
    for i, command in enumerate(cmds):
      if command >= min(cmds[max(0, i - window) : i + 1]) + 0.15:
        worst = max(worst, command - min(cmds[i : i + window]))
    self.assertLess(worst, 0.15)

  def test_a_launching_lead_is_followed_inside_the_band(self):
    """A lead pulling away from rest at 1.5 m/s^2 to 15 m/s is followed to inside the band, and the plan never asks
    for positive accel while it is closing inside the near edge — a launch is answered by the far edge, not by
    pressing into the lead."""
    s = _fly(0.0, 108.0, MOONPILOT_STOP_DISTANCE, 0.0, lambda t: 1.5 if t < 10.0 else 0.0, 30.0)
    near, far = _lead_band(s["vl"][-1])
    self.assertAlmostEqual(s["vl"][-1], 15.0, delta=0.05)
    self.assertGreaterEqual(s["end_gap"], near - 0.5)
    self.assertLessEqual(s["end_gap"], far + 3.0)
    self.assertFalse(s["contact"])
    for gap, v, v_lead, command in zip(s["gap"], s["v"], s["vl"], s["cmd"], strict=True):
      if v > v_lead and gap <= _lead_band(v_lead)[0]:
        self.assertLessEqual(command, 0.0)
    # and the property the flight is a sample of, over the states around it: inside the near edge and closing, the
    # ask is never positive
    exercised = 0
    for v_ego in (2.0, 10.0, 20.0, 30.0):
      for v_lead, a_slow in ((0.0, 0.0), (3.0, 0.0), (19.0, -2.0)):
        near, _ = _lead_band(v_lead, a_slow=a_slow)
        if v_ego <= v_lead + a_slow * MOONPILOT_PREDICT_T:
          continue
        for frac in (0.0, 0.5, 1.0):
          with self.subTest(v_ego=v_ego, v_lead=v_lead, a_slow=a_slow, frac=frac):
            self.assertLessEqual(lead_accel(v_ego, near * frac, v_lead, 0.0, self.T_FOLLOW, a_slow=a_slow), 1e-9)
            exercised += 1
    self.assertGreater(exercised, 0)


class TestDriverLadder(unittest.TestCase):
  """The cruise candidate lands on the driver's measured rungs at each speed-error threshold and
  interpolates between them; forceDecel keeps the pre-ladder law; the crawl cap only holds a creep."""

  def test_the_cruise_rungs(self):
    CP = _cp()
    for v in (0.0, 10.0, 20.0):
      rung = lambda err, v=v: cruise_accel(v, v + err, False, 0.0, CP, -0.3, True)  # noqa: E731
      self.assertAlmostEqual(rung(5.0), min(_fast_rung(v), 1.7))  # the combined budget
      slow_err = MOONPILOT_CRUISE_ERR_BP[4]
      self.assertAlmostEqual(rung(slow_err), float(np.interp(v, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V)))
      self.assertAlmostEqual(rung(0.5 * slow_err), 0.5 * rung(slow_err))  # interpolated, not stepped
      self.assertEqual(rung(0.0), 0.0)
    self.assertAlmostEqual(cruise_accel(20.0, 18.5, False, 0.0, CP, -0.3, True), -0.33)  # coasting
    self.assertAlmostEqual(cruise_accel(20.0, 17.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_LIGHT)
    self.assertAlmostEqual(cruise_accel(20.0, 10.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_MEDIUM)
    # The launch band's budget lets the measured launch peak through instead of clipping it at 1.7, and
    # the plant's tracking factor takes it back down to what the car actually delivers.
    launch_rung = _fast_rung(3.0)
    self.assertGreater(launch_rung, 1.7)
    self.assertLess(launch_rung, float(np.interp(3.0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V)))
    self.assertAlmostEqual(cruise_accel(3.0, 8.0, False, 0.0, CP, -0.3, True), launch_rung)
    self.assertAlmostEqual(_fast_rung(6.0), float(np.interp(6.0, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V)))

  def test_force_decel_is_not_softened(self):
    CP = _cp()
    self.assertEqual(cruise_accel(1.0, 0.0, False, 0.0, CP, -0.3, True), -1.0)
    self.assertEqual(cruise_accel(20.0, 0.0, False, 0.0, CP, -0.3, True), MOONPILOT_BRAKE_MEDIUM)

  def test_the_error_axis_scales_with_personality(self):
    """The personality dial moves the speed-error breakpoints, not the rung magnitudes: one factor
    per breakpoint (0 stays 0, so the ladder keeps its shape), keyed by the enum's raw value like
    MOONPILOT_T_FOLLOW — a _DynamicEnum off a message does not hash as its int."""
    CP = _cp()
    axis = lambda personality: _planner()._err_bp(_inputs(personality=personality))  # noqa: E731
    for personality in (int(Personality.aggressive), int(Personality.standard), int(Personality.relaxed)):
      scaled = axis(personality)
      self.assertEqual(scaled, [e * MOONPILOT_ERR_BP_SCALE[personality] for e in MOONPILOT_CRUISE_ERR_BP])
      self.assertEqual(scaled[3], 0.0)  # the hold rung never moves
    agg, rel = axis(int(Personality.aggressive)), axis(int(Personality.relaxed))
    # aggressive reaches the measured rungs at less speed error: err -3.5 lands the medium brake
    self.assertAlmostEqual(cruise_accel(20.0, 16.5, False, 0.0, CP, -0.3, True, err_bp=agg), MOONPILOT_BRAKE_MEDIUM)
    # relaxed at the same error sits between the light brake and coasting — softer on the pedal
    relaxed = cruise_accel(20.0, 16.5, False, 0.0, CP, -0.3, True, err_bp=rel)
    self.assertGreater(relaxed, MOONPILOT_BRAKE_LIGHT)
    self.assertLess(relaxed, -0.33)
    # ...and binds the fast rung at err +3.5 instead of +5
    fast = min(_fast_rung(20.0), 1.7)  # the combined budget; the tracking factor is 1.0 out here
    self.assertAlmostEqual(cruise_accel(20.0, 23.5, False, 0.0, CP, -0.3, True, err_bp=agg), fast)
    # at a shared positive error the ordering holds everywhere on the axis
    for err in (1.0, 2.0, 3.5, 5.0):
      asks = [cruise_accel(20.0, 20.0 + err, False, 0.0, CP, -0.3, True, err_bp=axis(p)) for p in (0, 1, 2)]
      self.assertEqual(asks, sorted(asks, reverse=True))
      self.assertGreater(asks[0], asks[2])
    # forceDecel never sees the axis: v_cruise <= 0 short-circuits before the interpolation
    for personality in (0, 1, 2):
      self.assertEqual(cruise_accel(1.0, 0.0, False, 0.0, CP, -0.3, True, err_bp=axis(personality)), -1.0)
      self.assertEqual(cruise_accel(20.0, 0.0, False, 0.0, CP, -0.3, True, err_bp=axis(personality)), MOONPILOT_BRAKE_MEDIUM)


class TestCoastGrade(unittest.TestCase):
  """The grade-aware coast band changes only the cruise candidate, and only inside its bounded
  overspeed/underspeed window."""

  ERROR_INSIDE = 0.75
  ERROR_PAST_EDGE = 1.6
  LOOP_FRAMES = 250

  @staticmethod
  def _cruise(pitch, v_ego, v_cruise=COAST_SET_SPEED, e2e=False, coast_band=MOONPILOT_COAST_BAND, allow_throttle=True):
    sm = _inputs(v_ego=v_ego, v_cruise_kph=v_cruise * 3.6, pitch=pitch)
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

  Ageing a lead shrinks the gap by `(v_lead - v_ego) * age` and adds the lead's travel over the age,
  but not decay: `update` decays the lead's accel over `action_t` only (`lead_state_at`'s `t_decay`)
  and takes the sustain horizon from the measured accel (`brake_sustain`). So for a lead not faster
  than the ego and not accelerating the closing rate only rises with age and the candidate never
  reads shallower. Strict at every age here: 46 ms is the median message age, 0.176 s that plus the
  learned radar latency `_lead_age` adds on this car (~0.13 s), 0.5 s past anything `commIssue`
  allows. Decaying over the age as well read up to +0.056 m/s^2 shallower at 0.176 s.
  """

  AGES = (0.046, 0.176, 0.5)
  V_EGO = (5.0, 12.0, 20.0, 30.0, 40.0)
  GAPS = (8.0, 20.0, 45.0, 90.0, 200.0)
  V_LEAD = (0.0, 5.0, 12.0, 20.0)
  A_LEAD = (-6.0, -3.0, -1.0, 0.0)
  ACTION_T = 0.2  # the test CP's longitudinalActuatorDelay + DT_MDL, as the planner computes it

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
              # as `update` builds them: decay over action_t only, the measured braking's sustain
              fresh = lead_state_at(lead, self.ACTION_T, v_ego * self.ACTION_T, a_lead, MOONPILOT_LEAD_ACCEL_TAU, self.ACTION_T)
              aged = lead_state_at(lead, self.ACTION_T + age, v_ego * (self.ACTION_T + age), a_lead, MOONPILOT_LEAD_ACCEL_TAU, self.ACTION_T)
              sustain = brake_sustain(a_lead)
              a_fresh = lead_accel(v_ego, *fresh, t_follow, sustain=sustain)
              a_aged = lead_accel(v_ego, *aged, t_follow, sustain=sustain)
              p_fresh = policy(v_ego, [(Source.lead0, *fresh, sustain)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True)[0]
              p_aged = policy(v_ego, [(Source.lead0, *aged, sustain)], 30.0, t_follow, False, None, 0.0, CP, -0.3, True)[0]
              self.assertLessEqual(a_aged, a_fresh + 1e-9, f"rose with the age {age} at v_ego {v_ego}, gap {gap}, v_lead {v_lead}, a_lead {a_lead}")
              self.assertLessEqual(p_aged, p_fresh + 1e-9)
              differ += a_aged < a_fresh - 1e-9
    self.assertGreater(differ, 0)  # the arms are not identical over this grid, so the claim means something

  def test_an_older_radar_lead_is_not_planned_as_braking_less(self):
    """Through `update`: a braking lead's accel decays over `action_t` only, so the applied radar age (message age plus
    the learned radar latency, `_lead_age`) adds travel, not decay. Held at a 12 m/s matched-speed follow 30 m behind a
    lead the estimator reads at -3 m/s^2 with radard's 1.5 s decay (an onset's first frames), the aged command is no
    shallower than the fresh one; decaying over the age too read -1.465 against -1.493 at 0.176 s, -0.695 at 0.5 s."""

    class Onset(LeadAccelEstimator):
      def update(self, lead):
        super().update(lead)
        return -3.0, MOONPILOT_LEAD_ACCEL_TAU

    def command(age):
      aged = mock.patch.object(MoonpilotLongitudinalPlanner, "_lead_age", lambda self, lead, lead_age: age)
      with mock.patch("moonpilot.longitudinal.LeadAccelEstimator", Onset), aged:
        planner = _planner()
        sm = _inputs(v_ego=12.0, v_cruise_kph=63.0, lead=_lead(30.0, 12.0))
        for _ in range(60):
          planner.update(sm)
        return planner.output_a_target

    fresh = command(0.0)
    self.assertLess(fresh, -1.0)  # a real brake, not a coast the brake band would flatten
    for age in (0.176, 0.5):
      self.assertLessEqual(command(age), fresh + 1e-6)


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
    """From a standstill behind a lead pulling away at 2 m/s^2, the ramp starts at the comfort step and is
    not held there: the command is at 0.54 m/s^2 seven frames in, past what six comfort frames deliver,
    so the launch jerk is available where the plan wants it. The first step stays the comfort one on
    purpose: `should_stop` releases a standstill hold when `a_target` crosses 0.1, so a step of 0.2 would
    let one frame of a lead's reported speed lift a parked car's brake hold, which
    `test_the_stop_gate_follows_the_governing_candidate_not_every_lead` pins from the other side.

    The ramp is ask-limited here, not jerk-limited — the lead law takes only `MOONPILOT_TRACK_ACCEL` of a
    launching lead's accel, which is what keeps the answer this smooth — so what is pinned is the shape,
    not which of the two limits produced each step.
    """
    planner = _planner()
    a_lead, steps = 2.0, []
    for frame in range(8):
      t = frame * DT_MDL
      lead = _lead(MOONPILOT_STOP_DISTANCE + 0.5 * a_lead * t**2, a_lead * t, a_lead=a_lead)
      previous = planner.output_a_target
      planner.update(_inputs(v_ego=0.0, v_cruise_kph=108.0, lead=lead, standstill=True))
      steps.append(planner.output_a_target - previous)
    self.assertAlmostEqual(steps[0], MOONPILOT_JERK_UP * DT_MDL, delta=1e-9)
    self.assertTrue(all(step <= MOONPILOT_JERK_LAUNCH * DT_MDL + 1e-9 for step in steps))
    self.assertGreater(planner.output_a_target, MOONPILOT_JERK_UP * DT_MDL * 6)  # past what six comfort frames could deliver

  def test_a_stop_behind_a_stopped_lead_eases_in(self):
    s = _fly(12.0, 43.2, 70.0, 0.0, lambda t: 0.0, 30.0)
    tail = [a for v, a in zip(s["v"], s["cmd"], strict=True) if 0.3 < v < 4.0]
    self.assertLess(max(earlier - later for earlier, later in zip(tail, tail[1:], strict=False)), 0.02)
    # A soft rest target, not a requirement to recover the last few centimeters by braking harder. The
    # gentle onset of the shallow arrival ask leaves it a meter and a half short of the rest anchor.
    self.assertTrue(MIN_REST_GAP < s["end_gap"] < MOONPILOT_STOP_DISTANCE)
    self.assertLess(s["v"][-1], 0.01)
    self.assertFalse(s["contact"])

  def test_a_radar_lead_is_planned_at_its_measured_sensor_latency(self):
    """On a car where the radar latency is measured, it is part of the lead's age: a radar lead
    reported at 30 m is planned exactly like a lead with no sensor latency 0.15 s of travel nearer.
    Seeded directly — the estimator's gates are unit-tested in `test_radar_latency.py` — so this
    pins only the `_lead_age` wiring."""
    rav4 = ToyotaInterface.get_non_essential_params(TOYOTA.TOYOTA_RAV4_TSS2)
    rav4.openpilotLongitudinalControl = True
    latency = 0.15
    fake = _params(on=True)
    commands = []
    for gap, radar in ((30.0, True), (30.0 - 10.0 * latency, False)):
      with mock.patch("moonpilot.longitudinal.Params", lambda: fake):
        planner = MoonpilotLongitudinalPlanner(rav4)
      planner.params = fake
      planner.radar_latency.seed(latency, 50)
      lead = _lead(gap, 0.0)
      lead.radar = radar
      planner.update(_inputs(v_ego=10.0, v_cruise_kph=72.0, lead=lead))
      commands.append(planner.output_a_target)
    self.assertAlmostEqual(commands[0], commands[1], delta=1e-9)

  def test_mid_approach_lead_brake_stays_safe(self):
    """A lead braking at -3.5 m/s^2 mid-approach while time-to-collision is 4-7 s exercises the floor admission.
    Flown closed-loop with 2 s of steady following before the onset so the estimator has a real speed history.

    The lead law sheds the closing before the lead's brake — it is arriving at the far edge of the band — so
    the flight peaks at -2.95 m/s^2 with 8.16 m of gap at its closest and no contact. Once the lead has slowed
    to a crawl the car settles inside the band rather than hanging back beyond it, matching its speed to within
    0.05 m/s: the headway is not bought by shedding speed.
    """
    s = _fly(17.0, 61.2, 90.0, 17.0, lambda t: -3.5 if 2 <= t < 5 else 0.0, 140.0)
    self.assertTrue(any((v - vl) > 0.0 and 4.0 <= (g - MOONPILOT_STOP_DISTANCE) / (v - vl) <= 7.0 for v, vl, g in zip(s["v"], s["vl"], s["gap"], strict=False)))
    self.assertGreater(s["min_gap"], MOONPILOT_STOP_DISTANCE + 1.0)  # clear of the standstill floor
    near = max(MOONPILOT_STOP_REST, MOONPILOT_T_FOLLOW[int(Personality.standard)] * s["vl"][-1])
    # The catch-up closes at MOONPILOT_CATCH_UP_CLOSING, so the settle takes ~2 min from the slack; the flight
    # runs long enough to land inside the band, which is the claim.
    self.assertGreaterEqual(s["end_gap"], near)  # never inside the band's near edge
    self.assertLessEqual(s["end_gap"], near + MOONPILOT_FOLLOW_BAND_T * s["vl"][-1])  # and inside its far edge
    # the resting speed is the lead's, not a little under it: that is the invariant this floor holds
    self.assertAlmostEqual(s["v"][-1], s["vl"][-1], delta=0.1)
    self.assertFalse(s["contact"])
    self.assertGreater(s["peak"], ACCEL_MIN)

  def test_a_radar_dropout_releases_the_lead_candidate(self):
    """A missing slot is not a held stop: once the radar lead leaves, policy sees cruise rather than
    a latched lead candidate."""
    planner = _planner()
    planner.update(_inputs(v_ego=10.0, v_cruise_kph=108.0, lead=_lead(20.0, 0.0)))
    planner.update(_inputs(v_ego=10.0, v_cruise_kph=108.0, lead=_lead(200.0, 10.0, present=False)))
    self.assertEqual(planner.source, Source.cruise)

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

  def test_one_car_in_both_slots_is_one_candidate(self):
    """radard can match both model leads to one stopped car, on two of its radar tracks whose speeds
    disagree by a few tenths (route 00000400 seg 4). Two readings of one car under `min` handed the
    win back and forth and pulsed the command; one car is one candidate, so the plan is exactly the
    plan for `leadOne` alone and `lead1` never governs."""
    both, alone = _planner(), _planner()
    for frame in range(60):
      v_ego, gap = max(4.0 - 0.07 * frame, 0.0), 12.0 - 0.15 * frame
      one, two = _lead(gap, 0.3 if frame % 3 else 0.0), _lead(gap + 0.1, 0.0 if frame % 3 else 0.3)
      one.radarTrackId, two.radarTrackId = (517, 541) if frame % 2 else (541, 517)
      both.update(_inputs(v_ego=v_ego, lead=one, lead_two=two))
      alone.update(_inputs(v_ego=v_ego, lead=one))
      self.assertNotEqual(both.source, Source.lead1)
      self.assertEqual(both.output_a_target, alone.output_a_target)

  def test_fcw_sees_both_slots_of_one_car(self):
    """The dedupe is the plan's, not FCW's: radard's two matches of one car can carry different
    `modelProb`, and dropping `leadTwo` would drop the only one over the FCW bar."""
    planner = _planner()
    for _ in range(4):
      planner.update(_inputs(v_ego=20.0, lead=_lead(30.0, 0.0, model_prob=0.6), lead_two=_lead(30.0, 0.0, model_prob=0.95)))
    self.assertTrue(planner.fcw)

  def test_a_car_alongside_stays_a_separate_candidate(self):
    """Equal range and speed is what an adjacent-lane car pacing the lead has. `leadOne` here is that
    car, a lane over and holding speed; `leadTwo` is the in-path car beside it, starting to brake. They
    are two vehicles, so the braking one stays a candidate and governs exactly as it would alone."""
    both, alone = _planner(), _planner()
    for frame in range(30):
      v_lead = 15.0 - 3.0 * max(frame - 24, 0) * DT_MDL
      beside, ahead = _lead(30.0, 15.0), _lead(30.0, v_lead)
      beside.yRel, beside.radarTrackId, ahead.radarTrackId = 3.5, 1, 2
      both.update(_inputs(v_ego=15.0, v_cruise_kph=60.0, lead=beside, lead_two=ahead))
      alone.update(_inputs(v_ego=15.0, v_cruise_kph=60.0, lead=ahead))
    self.assertEqual(both.source, Source.lead1)
    self.assertEqual(both.output_a_target, alone.output_a_target)

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

    planner = _planner(params_overrides={"MoonpilotExperimentalMode": True})
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, lead=_lead(35.0, 30.0), model_accel=-2.0))
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

  def test_time_gap_handoff_is_smooth_and_resets_without_a_lead(self):
    out_of_path = custom.MoonpilotState.LeadTrajectory.new_message()
    out_of_path.present = True
    out_of_path.inPath = 0.0
    in_path = custom.MoonpilotState.LeadTrajectory.new_message()
    in_path.present = True
    in_path.inPath = 1.0
    lead = _lead(35.0, 20.0)
    planner = _planner()
    base = MOONPILOT_T_FOLLOW[Personality.standard]
    target = base * MOONPILOT_OUT_OF_PATH_T_FOLLOW

    self.assertAlmostEqual(planner._t_follow(_inputs(lead=lead, moonpilot_leads=[in_path])), base, delta=1e-9)
    leaving = [planner._t_follow(_inputs(lead=lead, moonpilot_leads=[out_of_path])) for _ in range(100)]
    self.assertGreater(leaving[0], target)
    self.assertAlmostEqual(leaving[-1], target, delta=1e-9)
    self.assertLessEqual(max(a - b for a, b in zip(leaving, leaving[1:], strict=False)), MOONPILOT_T_FOLLOW_SLEW_DOWN * DT_MDL + 1e-9)

    returning = [planner._t_follow(_inputs(lead=lead, moonpilot_leads=[in_path])) for _ in range(100)]
    self.assertLess(returning[0], base)
    self.assertAlmostEqual(returning[-1], base, delta=1e-9)
    self.assertLessEqual(max(b - a for a, b in zip(returning, returning[1:], strict=False)), MOONPILOT_T_FOLLOW_SLEW_UP * DT_MDL + 1e-9)

    self.assertAlmostEqual(planner._t_follow(_inputs(moonpilot_leads=[in_path])), base, delta=1e-9)

  def test_time_gap_reads_the_personality_off_the_message(self):
    """The table is keyed by the enum's raw value because the enum read off a message is a capnp
    _DynamicEnum whose hash is not that value — keyed by the schema members this silently returned
    the standard gap for every personality."""
    for personality, t_follow in MOONPILOT_T_FOLLOW.items():
      self.assertAlmostEqual(_planner()._t_follow(_inputs(personality=personality)), t_follow, delta=1e-9)

  def test_the_personality_dial_reaches_the_output(self):
    """The dial end to end: update reads the personality off the message and threads the scaled
    error axis into both the live policy and the trajectory solver. Without that threading the
    axis tests still pass while every personality drives identically, so this is the plumbing
    pin; at a shared positive error the steady output orders aggressive > standard > relaxed."""
    steady = {}
    for name, personality in (("aggressive", 0), ("standard", 1), ("relaxed", 2)):
      planner = _planner()
      a_ego = 0.0
      for _ in range(60):  # the jerk limit needs frames to let the asks through
        planner.update(_inputs(v_ego=20.0, v_cruise_kph=22.0 * 3.6, a_ego=a_ego, personality=personality))
        a_ego = float(planner.output_a_target)
      steady[name] = a_ego
    self.assertGreater(steady["aggressive"], steady["standard"])
    self.assertGreater(steady["standard"], steady["relaxed"])

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
    self.assertGreater(gap, MIN_REST_GAP)

  def test_a_vision_lead_accel_uses_the_half_second_preview(self):
    """A vision-only matched-speed lead's braking estimate reaches the command through the half-second
    preview: from the first frame and still ten frames on it brakes deeper than the same lead at zero
    accel, inside the actuator range. The values are the jerk limit's ramp, so they are not pinned."""
    gap, v_ego = 25.0, 20.0
    zero_accel = _planner()
    zero_input = _inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=_lead(gap, v_ego))
    zero_accel.update(zero_input)

    vision_lead = _lead(gap, v_ego, a_lead=-4.0, a_lead_tau=0.3)
    vision_lead.radar = False
    vision = _planner()
    vision_input = _inputs(v_ego=v_ego, v_cruise_kph=108.0, lead=vision_lead)
    vision.update(vision_input)

    self.assertEqual(zero_accel.source, Source.lead0)
    self.assertEqual(vision.source, Source.lead0)
    self.assertLess(vision.output_a_target, zero_accel.output_a_target)

    for _ in range(9):
      zero_accel.update(zero_input)
      vision.update(vision_input)
    self.assertLess(vision.output_a_target, zero_accel.output_a_target - 0.5)
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
    0.55 s and -2.0 at 0.95, and a *bare* time-to-collision term (since removed) is later still — it keys off the closing
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
    self.assertGreater(fresh, MIN_REST_GAP)  # it stops, inside the floor's rest region
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
          self.assertGreater(c, MIN_REST_GAP)  # still a real stop, not a degraded one

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

    Sampled 0.80 s past the onset: the two arms agree on the command at that frame (-2.94 against
    -2.87) and diverge across the 2.5 s rollout as the decay is applied, by up to 0.683 m/s^2 at
    1.35 s — the estimator's own decay holds the plan deeper for longer, which is the whole claim.
    The 0.4 asserted leaves half of that separation as margin and still fails a revert, where the two
    arms are the same published trajectory.

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
    self.assertGreater(float(np.abs(own - msg).max()), 0.4, msg="the rollout is not running on the estimator's decay")

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
    the stopping floor binds there against a 20 m/s lead at 25 m/s — a lead 60 m out leaves the lead
    terms dormant and the plan accelerating, which is not the regime this test needs to be in.
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
  def _run(params_on=True, frames=40, experimental=False, **inputs):
    planner = _planner(params_on=params_on, params_overrides={"MoonpilotExperimentalMode": experimental})
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

  def test_a_shallow_admitted_ask_blends_in_from_the_plan(self):
    """Route 00000408 at 15 m/s: an ask admitted at -0.50 stepped the plan from -0.23 to -0.52 in 0.15 s
    and the car answered -0.84. Admitted, a shallow ask now blends in from the other candidates over
    `MOONPILOT_MODEL_BRAKE_FADE_T`, never shallower than the step-in would have been relieved by, and
    still arrives whole. At the set speed the cruise term is ~0; the onset jerk is held at
    `MOONPILOT_JERK_DOWN` so the fade is what is measured."""

    def commands():
      planner = _planner()
      for _ in range(40):
        planner.update(_inputs(v_cruise_kph=72.0, model_accel=0.0))
      sm = _inputs(v_cruise_kph=72.0, model_accel=-0.6)
      out = []
      for _ in range(30):
        planner.update(sm)
        out.append(planner.output_a_target)
      return out

    with mock.patch("moonpilot.longitudinal.MOONPILOT_JERK_ONSET", MOONPILOT_JERK_DOWN):
      faded = commands()
      with mock.patch("moonpilot.longitudinal.MOONPILOT_MODEL_BRAKE_FADE_T", 1e-9):
        stepped = commands()
    self.assertTrue(all(f >= s - 1e-12 for f, s in zip(faded, stepped, strict=True)))
    self.assertGreater(faded[4] - stepped[4], 0.15)  # held back over the first quarter second
    self.assertAlmostEqual(faded[-1], -0.6)  # and whole once the fade has run

  def test_the_published_plan_projects_the_fade(self):
    """On the admission frame the live weight is one tenth, but the command reaches the whole ask within
    `MOONPILOT_MODEL_BRAKE_FADE_T`; the published plan has to forecast that, not one tenth of it forever."""
    planner = _planner()
    for _ in range(40):
      planner.update(_inputs(v_cruise_kph=72.0, model_accel=0.0))
    planner.update(_inputs(v_cruise_kph=72.0, model_accel=-0.6))
    self.assertGreater(planner.output_a_target, -0.1)  # the command itself is still at the start of the fade
    self.assertAlmostEqual(float(planner.a_desired_trajectory[0]), planner.output_a_target, delta=1e-12)
    self.assertAlmostEqual(float(planner.a_desired_trajectory.min()), -0.6, delta=1e-6)

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

  def test_a_low_speed_leadless_model_stop_enters_the_stop_hold_early(self):
    planner = _planner()
    planner.update(_inputs(v_ego=20.0, v_cruise_kph=62.0, model_accel=-1.0))
    for _ in range(20):
      planner.update(_inputs(v_ego=1.0, v_cruise_kph=62.0, model_accel=-0.46))
    self.assertTrue(planner.output_should_stop)
    self.assertEqual(planner.source, Source.e2e)

    with_lead = _planner()
    with_lead.update(_inputs(v_ego=20.0, v_cruise_kph=62.0, model_accel=-1.0))
    for _ in range(20):
      with_lead.update(_inputs(v_ego=1.0, v_cruise_kph=62.0, model_accel=-0.46, lead=_lead(6.0, 0.2)))
    self.assertFalse(with_lead.output_should_stop)

    planner.update(_inputs(v_ego=1.0, v_cruise_kph=62.0, model_accel=0.15))
    self.assertFalse(planner.output_should_stop)

  def test_a_stop_the_model_began_is_a_stop_the_model_finishes(self):
    """Route 000003d2's defect, closed-loop: the model braked a clear-road stop from 17 m/s and its
    ask tapered back over the -0.5 deadband at 0.9 m/s, where the cruise term took the slot and was
    accelerating the car when the driver braked. Admitted once, the model now keeps the slot in the
    crawl band until it asks to go — the car stops and holds — and gives it back on its first ask past
    `should_stop`'s 0.1."""
    planner = _planner()
    v = 8.0
    for _ in range(round(20.0 / DT_MDL)):
      ask = max(-1.5, -0.35 * v - 0.05)  # a model stop that eases off as the car slows
      planner.update(_inputs(v_ego=v, v_cruise_kph=62.0, model_accel=ask))
      v = max(0.0, v + planner.output_a_target * DT_MDL)
    self.assertLess(v, 0.05)
    self.assertTrue(planner.output_should_stop)
    self.assertEqual(planner.source, Source.e2e)

    planner.update(_inputs(v_ego=0.0, v_cruise_kph=62.0, model_accel=0.05))
    self.assertTrue(planner.output_should_stop)  # under the release: still the model's stop
    for _ in range(3):
      planner.update(_inputs(v_ego=0.0, v_cruise_kph=62.0, model_accel=0.15))
    self.assertFalse(planner.output_should_stop)
    self.assertEqual(planner.source, Source.cruise)

  def test_the_hold_on_the_slot_is_a_low_speed_hysteresis_only(self):
    """At speed the release is the entry deadband itself, so a model that braked and then relaxed to
    -0.4 hands back to cruise exactly as a model that never braked — no ratchet below the set speed."""
    self.assertEqual(model_release(MOONPILOT_MODEL_STOP_SPEED), MOONPILOT_MODEL_BRAKE_THRESHOLD)
    self.assertEqual(model_release(0.0), MOONPILOT_MODEL_STOP_RELEASE)
    braked = _planner()
    # 80 frames: the brake-exit ramp (`MOONPILOT_BRAKE_EXIT_JERK`) has to finish before the two meet.
    for ask, frames in ((-1.5, 20), (-0.4, 80)):
      for _ in range(frames):
        braked.update(_inputs(v_ego=20.0, v_cruise_kph=108.0, model_accel=ask))
    self.assertEqual(braked.source, Source.cruise)
    self.assertAlmostEqual(braked.output_a_target, self._run(model_accel=-0.4).output_a_target, delta=1e-12)

  def test_the_feature_row_matches_the_params_default(self):
    feature = next(f for f in FEATURES if f.key == "MoonpilotModelBraking")
    # It changes driving behavior, so the row is offroad-only. It ships on: the candidate can only
    # add braking, and the corpus behind that is in `model_candidate`.
    self.assertTrue(feature.offroad_only)
    self.assertFalse(feature.requires)
    text = (ROOT / "moonpilot" / "params_keys.h").read_text()
    self.assertTrue('{"MoonpilotModelBraking", {PERSISTENT, BOOL, "1"}}' in text)


def _plant_sm(sm):
  """Upstream's plant stopped building carControl and vehicleParameters once its own planner stopped
  reading them (#39097). plannerd still subscribes both for the fork planner, so the plant's dict gets
  their defaults here: no pose, no steering offset."""
  return {"carControl": messaging.new_message("carControl").carControl,
          "vehicleParameters": messaging.new_message("vehicleParameters").vehicleParameters, **sm}


class _ModeParams:
  """The real store with `MoonpilotExperimentalMode` pinned: upstream's maneuvers dropped their `e2e`
  argument along with chill mode, and the fork planner reads its mode from this param instead."""

  def __init__(self, real, e2e):
    self._real, self._e2e = real, e2e

  def get(self, key, **kwargs):
    return self._e2e if key == "MoonpilotExperimentalMode" else self._real.get(key, **kwargs)

  def __getattr__(self, name):
    return getattr(self._real, name)


class TestUpstreamManeuvers(unittest.TestCase):
  """Upstream's maneuver suite, run through the seam: the plant builds whatever the seam returns,
  so patching the symbol it binds is exactly how plannerd picks the fork planner."""

  class _PlantPlanner(MoonpilotLongitudinalPlanner):
    def update(self, sm):
      super().update(_plant_sm(sm))

  def test_upstream_maneuvers_pass(self):
    for e2e, force_decel in itertools.product([True, False], repeat=2):
      def factory(CP, init_v=0.0, init_a=0.0, dt=DT_MDL, e2e=e2e):
        planner = self._PlantPlanner(CP, init_v=init_v, init_a=init_a, dt=dt)
        planner.params = cast(Params, _ModeParams(planner.params, e2e))
        return planner

      with mock.patch.object(plant_mod, "LongitudinalPlanner", factory):
        for maneuver in create_maneuvers({"force_decel": force_decel}):
          with self.subTest(title=maneuver.title, e2e=e2e, force_decel=force_decel):
            valid, _ = maneuver.evaluate()
            assert valid


if __name__ == "__main__":
  unittest.main()
