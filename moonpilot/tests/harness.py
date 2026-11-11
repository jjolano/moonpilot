"""The flight harness the planner tests and the offline lab share.

A planner, its `SubMaster` of fake messages, the lead and `CarParams` builders, the maneuver plant's defaults and the
point-mass flight: the lab (`moonpilot/lab/`) flies the same planner the tests do, so the harness lives here rather
than in either. This module has no `test_` prefix so the runner does not collect it, and it holds no assertions.

It sits under `tests/` rather than in the lab package because the suite is the larger consumer: the reverse edge would
pull the whole lab (log reader, route tables, every submodule) into every test import.
"""

from time import monotonic
import numpy as np
from openpilot.cereal import custom, log, messaging
from opendbc.car.structs import car as car_struct
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.toyota.interface import CarInterface as ToyotaInterface
from opendbc.car.toyota.values import ToyotaFlags
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.slam import MOONPILOT_SLAM_CORR_STD_MIN

from moonpilot.tests.fakes import params as fake_params

Personality = log.LongitudinalPersonality
Source = log.LongitudinalPlan.LongitudinalPlanSource


def car_params(car=CAR.HONDA_CIVIC, openpilot_longitudinal=True):
  """A car's real `CarParams`, as its interface builds it: what the planner and controller tests fly."""
  cp = CarInterface.get_non_essential_params(car) if car == CAR.HONDA_CIVIC else ToyotaInterface.get_non_essential_params(car)
  cp.openpilotLongitudinalControl = openpilot_longitudinal
  return cp


def custom_car_params(brand="toyota", *, pcm_cruise=True, flags=0, safety_param=73, safety_configs=1,
                passive=False, lateral_tuning="pid", steer_control_type=None, openpilot_longitudinal=True):
  """A `CarParams` message carrying exactly the fields named, for the gates and panels that read nothing else."""
  cp = car_struct.CarParams.new_message()
  cp.brand = brand
  cp.pcmCruise = pcm_cruise
  cp.passive = passive
  cp.flags = int(flags)
  cp.lateralTuning.init(lateral_tuning)
  cp.steerControlType = car_struct.CarParams.SteerControlType.torque if steer_control_type is None else steer_control_type
  cp.openpilotLongitudinalControl = openpilot_longitudinal
  for config in cp.init("safetyConfigs", safety_configs):
    config.safetyParam = safety_param
  return cp


def lead(d_rel, v_lead, a_lead=0.0, a_lead_tau=1.5, model_prob=1.0, present=True):
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
  def __init__(self, data, moonpilot_leads=None, radar_age_s=0.0, recv_age_s=0.0, ego_correction=None):
    super().__init__(data)
    moonpilot = custom.MoonpilotState.new_message()
    if moonpilot_leads is not None:
      moonpilot.leads = moonpilot_leads
    if ego_correction is not None:
      correction = moonpilot.egoCorrection
      correction.valid = True
      correction.monoTime = int(monotonic() * 1e9)
      correction.age = 0.0
      correction.corrStd = MOONPILOT_SLAM_CORR_STD_MIN
      correction.dVel = ego_correction
    self["moonpilotState"] = moonpilot
    self.valid = {"moonpilotState": moonpilot_leads is not None or ego_correction is not None}
    self.alive = {"moonpilotState": moonpilot_leads is not None or ego_correction is not None}
    self.recv_time = {"modelV2": 1.0 + recv_age_s}
    # The planner ages a radar lead by the gap between the model and radar stamps, so the freshness of both inputs has
    # to be in here: equal stamps mean a fresh lead, and `radar_age_s` moves the radar one back.
    self.logMonoTime = {"modelV2": 1e9, "radarState": 1e9 - radar_age_s * 1e9}

  def all_checks(self, *services):
    return True


def inputs(v_ego=20.0, v_cruise_kph=108.0, a_ego=0.0, lead=None, lead_two=None, standstill=None,
            personality=Personality.standard, enabled=True, force_decel=False, throttle_prob=1.0, pitch=0.0,
            steer_angle_deg=0.0, model_accel=0.0, model_should_stop=False, moonpilot_leads=None,
            radar_age_s=0.0, model_age_s=0.0, recv_age_s=0.0, path=None, lat_active=True,
            steering_pressed=False, vp_valid=False, steer_ratio=0.0, stiffness_factor=0.0, roll=0.0,
            road_edges=None):
  cs = messaging.new_message("carState").carState
  cs.vEgo, cs.aEgo, cs.vCruise = float(v_ego), float(a_ego), float(v_cruise_kph)
  cs.standstill = bool(v_ego < 0.01) if standstill is None else bool(standstill)
  cs.steeringAngleDeg, cs.steeringPressed = float(steer_angle_deg), bool(steering_pressed)
  cc = messaging.new_message("carControl").carControl
  cc.orientationNED = [0.0, float(pitch), 0.0]
  cc.latActive = bool(lat_active)
  controls = messaging.new_message("controlsState").controlsState
  controls.forceDecel = bool(force_decel)
  controls.longControlState = 1
  state = messaging.new_message("selfdriveState").selfdriveState
  state.enabled, state.personality = bool(enabled), personality
  vp = messaging.new_message("vehicleParameters").vehicleParameters
  vp.valid, vp.steerRatio, vp.stiffnessFactor, vp.roll = bool(vp_valid), float(steer_ratio), float(stiffness_factor), float(roll)
  model = messaging.new_message("modelV2").modelV2
  model.timestampEof = int(round(1e9 - model_age_s * 1e9))
  model.meta.disengagePredictions.gasPressProbs = [float(throttle_prob)] * 6
  model.action.desiredAcceleration, model.action.shouldStop = float(model_accel), bool(model_should_stop)
  if path is not None:
    x, v_path, psi_rate = path
    model.position = log.XYZTData.new_message(x=[float(v) for v in x], t=ModelConstants.T_IDXS)
    model.velocity = log.XYZTData.new_message(x=[float(v) for v in v_path], t=ModelConstants.T_IDXS)
    model.orientationRate = log.XYZTData.new_message(z=[float(v) for v in psi_rate], t=ModelConstants.T_IDXS)
  if road_edges is not None:
    model.roadEdges = [log.XYZTData.new_message(x=[float(v) for v in ex], y=[float(v) for v in ey]) for ex, ey in road_edges]
  radar = messaging.new_message("radarState").radarState
  radar.leadOne = lead if lead is not None else globals()["lead"](200.0, float(v_ego), present=False)
  radar.leadTwo = lead_two if lead_two is not None else globals()["lead"](200.0, float(v_ego), present=False)
  return SubMaster({"carState": cs, "carControl": cc, "controlsState": controls, "selfdriveState": state,
                    "vehicleParameters": vp, "modelV2": model, "radarState": radar}, moonpilot_leads,
                   radar_age_s, recv_age_s)

def slot(in_path, present=True):
  trajectory = custom.MoonpilotState.LeadTrajectory.new_message()
  trajectory.present, trajectory.inPath = present, float(in_path)
  return trajectory



def path(v_path, curvature, x=None):
  t = np.array(ModelConstants.T_IDXS)
  v = np.full(len(t), float(v_path))
  x = float(v_path) * t if x is None else np.asarray(x, dtype=float)
  return x, v, np.broadcast_to(np.asarray(curvature, dtype=float), x.shape) * v


def planner(engine_coast=False, overrides=None, params_on=True, params=None, car=CAR.HONDA_CIVIC, **kwargs):
  cp = car_params(car)
  if engine_coast:
    cp.flags = int(cp.flags) | int(ToyotaFlags.MOONPILOT_ENGINE_COAST)
  return MoonpilotLongitudinalPlanner(cp, params=params or fake_params(on=params_on, overrides=overrides), **kwargs)


def fly(v0, v_set_kph, gap0, v_lead0, a_lead_fn, T, dt=DT_MDL):
  planner_instance = planner()
  v, gap, v_lead, a_prev = v0, gap0, v_lead0, 0.0
  vs, gaps, v_leads, cmds = [], [], [], []
  for frame in range(round(T / dt)):
    a_lead = float(a_lead_fn(frame * dt))
    planner_instance.update(inputs(v_ego=max(v, 0.0), v_cruise_kph=v_set_kph, a_ego=a_prev,
                                   lead=lead(max(gap, 0.0), max(v_lead, 0.0)), personality=1))
    a_prev = float(planner_instance.output_a_target)
    vs.append(v)
    gaps.append(gap)
    v_leads.append(v_lead)
    cmds.append(a_prev)
    v = max(0.0, v + a_prev * dt)
    v_lead_next = max(0.0, v_lead + a_lead * dt)
    gap = max(0.0, gap + (0.5 * (v_lead + v_lead_next) - v) * dt)
    v_lead = v_lead_next
  return {"v": vs, "vl": v_leads, "gap": gaps, "cmd": cmds, "peak": min(cmds),
          "min_gap": min(gaps), "end_gap": gaps[-1], "contact": min(gaps) < 0.4}


def plant_sm(sm):
  wrapped = SubMaster(sm)
  if "carControl" not in wrapped:
    control = messaging.new_message("carControl").carControl
    control.orientationNED = [0.0, 0.0, 0.0]
    control.latActive = False
    wrapped["carControl"] = control
  if "vehicleParameters" not in wrapped:
    wrapped["vehicleParameters"] = messaging.new_message("vehicleParameters").vehicleParameters
  wrapped.valid.update(getattr(sm, "valid", {}))
  wrapped.alive.update(getattr(sm, "alive", {}))
  wrapped.logMonoTime.update(getattr(sm, "logMonoTime", {}))
  wrapped.recv_time.update(getattr(sm, "recv_time", {}))
  return wrapped


def plant_planner(**kwargs):
  ego_correction = kwargs.pop("ego_correction", None)

  class PlantPlanner(MoonpilotLongitudinalPlanner):
    def update(self, sm):
      super().update(SubMaster(plant_sm(sm), ego_correction=ego_correction))

  def factory(CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    return PlantPlanner(CP, init_v=init_v, init_a=init_a, dt=dt, **kwargs)

  return factory
