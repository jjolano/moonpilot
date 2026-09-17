"""moonpilot's longitudinal planner: the fork's own answer to "what acceleration?".

Upstream solves an acados MPC every frame. This is closed-form — no solver, no generated C — and
every number in it is fork-owned, so tuning happens here instead of inside a generated optimization
problem. The policy is three candidates, and the smallest wins:

  - a spacing regulator that holds ``gap == STOP_DISTANCE + t_follow * v_ego`` against the nearest
    lead, gaining rigidity as the gap closes (all the slack is in the time gap, none in the gains);
  - the time gap itself, biased by ``MoonpilotLeadLateral`` when the lead is predicted to leave the
    path — the fork's lateral prediction reaches the longitudinal policy here, since there is no
    MPC danger zone to scale;
  - a kinematic brake that takes over once the required decel passes the regulator's braking
    authority, and then holds the exact constant-deceleration profile that arrives at the stop
    distance with the lead's speed — self-consistent without a solver, which is what makes the
    approach smooth. The handover into it is a step in the candidate; it is arbitration that keeps
    that step out of the output, and the jerk limit that turns what remains into a rate;
  - the cruise term, and in experimental mode the model's own accel, the same candidates upstream
    arbitrates between.

Two things are deliberately not upstream's:

  - delay compensation is done by predicting the state at the actuator delay and evaluating the
    policy there, not by inverting the published plan through ``get_accel_from_plan``. That inverse
    divides by the delay, so a step in the plan comes out amplified by ``action_t / dt`` — measured
    at 45 m/s^3 and instant ACCEL_MIN saturation in simulation.

The planner is picked once, at construction, so the toggle needs a restart, and it only exists on a
car with ``openpilotLongitudinalControl`` — cars whose stock ACC owns acceleration never reach the
seam. Upstream's MPC stays on the line as the fallback.
"""

import math
import time

import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, should_stop
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.features import LEAD_LATERAL, LONGITUDINAL, enabled
from moonpilot.lead import nearest_lead_in_path

LongCtrlState = car.CarControl.Actuators.LongControlState
# From cereal, not from long_mpc — that module imports the compiled acados solver at import time,
# and this file is on plannerd's init path.
LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
Personality = log.LongitudinalPersonality

# Starting points, all fork-owned. Tune against logs.
MOONPILOT_STOP_DISTANCE = 6.0  # m; gap held behind a stopped lead
# s; time gap per personality, keyed by the enum's raw value rather than the schema member:
# `log.LongitudinalPersonality.standard` is a plain int, but the same enum read off a message is a
# capnp _DynamicEnum whose hash is not that int, so a dict keyed by the members never matches.
MOONPILOT_T_FOLLOW = {
  int(Personality.relaxed): 1.75,
  int(Personality.standard): 1.45,
  int(Personality.aggressive): 1.25,
}
MOONPILOT_K_GAP = 0.3  # 1/s^2 on the spacing error
MOONPILOT_K_V = 0.6  # 1/s on the relative speed
MOONPILOT_APPROACH_DECEL = 1.0  # m/s^2; the decel an approach is planned at, and the
# spacing regulator's braking authority — one number, so
# the two terms meet continuously
MOONPILOT_MIN_SLACK = 0.5  # m; floor on the braking-distance denominator
MOONPILOT_LEAD_PREVIEW_T = 1.0  # s of the lead's own braking credited to the safety term
MOONPILOT_OUT_OF_PATH_T_FOLLOW = 0.7  # time-gap scale for a lead predicted to leave the path
MOONPILOT_K_CRUISE = 1.0  # 1/s on the speed error
MOONPILOT_A_CRUISE_MIN = -1.2  # m/s^2; cruise never brakes harder than this
MOONPILOT_A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]  # m/s
MOONPILOT_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]  # m/s^2
MOONPILOT_A_TOTAL_MAX_BP = [20.0, 40.0]  # m/s
MOONPILOT_A_TOTAL_MAX_V = [1.7, 3.2]  # m/s^2 combined accel budget
MOONPILOT_JERK_UP = 1.5  # m/s^3
MOONPILOT_JERK_DOWN = 4.0  # m/s^3
MOONPILOT_JERK_EMERGENCY = 10.0  # m/s^3, reached at ACCEL_MIN
MOONPILOT_ALLOW_THROTTLE_THRESHOLD = 0.4
MOONPILOT_MIN_ALLOW_THROTTLE_SPEED = 2.5  # m/s
MOONPILOT_CRASH_DISTANCE = 0.25  # m; FCW contact margin
MOONPILOT_FCW_DECEL = -4.0  # m/s^2 required decel that means it cannot be avoided
MOONPILOT_FCW_COUNT = 2  # frames above threshold before FCW latches
MOONPILOT_FCW_MODEL_PROB = 0.9
MOONPILOT_CONTROL_T_IDX = np.array(ModelConstants.T_IDXS[:CONTROL_N])  # 0 … 2.5 s, 17 points


def coast_accel(pitch: float) -> float:
  # Accel with the throttle closed on this grade; the same fitted shape upstream uses.
  return float(np.sin(pitch) * -5.65 - 0.3)


def cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle) -> float:
  """Speed-error term. In experimental mode the cap is ACCEL_MAX and neither the cornering budget
  nor the coast limit applies, because the model's own accel is the candidate that governs there."""
  cap = ACCEL_MAX if e2e else float(np.interp(v_ego, MOONPILOT_A_CRUISE_MAX_BP, MOONPILOT_A_CRUISE_MAX_V))
  if not e2e:
    a_total_max = float(np.interp(v_ego, MOONPILOT_A_TOTAL_MAX_BP, MOONPILOT_A_TOTAL_MAX_V))
    a_y = v_ego**2 * steer_angle_deg * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    cap = min(cap, math.sqrt(max(a_total_max**2 - a_y**2, 0.0)))
    if not allow_throttle:
      coast_limit = float(np.interp(v_ego, [MOONPILOT_MIN_ALLOW_THROTTLE_SPEED, 2 * MOONPILOT_MIN_ALLOW_THROTTLE_SPEED], [cap, max(accel_coast, ACCEL_MIN)]))
      cap = min(cap, coast_limit)
  return float(np.clip(MOONPILOT_K_CRUISE * (v_cruise - v_ego), MOONPILOT_A_CRUISE_MIN, cap))


def lead_accel(v_ego, gap, v_lead, a_lead, t_follow) -> float:
  """Two terms, one number.

  The spacing regulator holds gap == STOP_DISTANCE + t_follow * v_ego and matches the lead's speed;
  its braking authority is capped at the approach decel, so large speed errors do not turn into hard
  braking through the gain.

  The kinematic term is the exact deceleration that arrives at STOP_DISTANCE with the lead's
  (preview-corrected) speed, and it only binds once that exceeds the approach decel. Following it is
  self-consistent — a constant-deceleration approach keeps the required value constant — which is
  what makes the approach profile smooth without a solver.

  The handover into it is a step, not a crossover: where the required decel reaches the approach
  decel the regulator's output is whatever the spacing error says, positive when the gap is still
  wide — +19.9 m/s^2 at 25 m/s closing on a 20 m/s lead — and the kinematic term replaces it from
  there. That step is only safe because it is a *candidate*: `policy` takes the minimum, so while the
  lead asks for more than the cruise term the cruise term governs and the output never sees it. What
  the output steps by is therefore cruise's cap minus the approach decel, and the caller's jerk limit
  turns that into a rate. What this term buys is the profile after the handover, not continuity
  across it.
  """
  gap = max(float(gap), 0.0)
  v_lead = max(float(v_lead), 0.0)
  v_lead_eff = max(0.0, v_lead + min(float(a_lead), 0.0) * MOONPILOT_LEAD_PREVIEW_T)
  a_track = max(MOONPILOT_K_GAP * (gap - MOONPILOT_STOP_DISTANCE - t_follow * v_ego) + MOONPILOT_K_V * (v_lead - v_ego), -MOONPILOT_APPROACH_DECEL)
  if v_ego <= v_lead_eff:
    return a_track  # not closing: nothing to brake for
  a_req = -(v_ego**2 - v_lead_eff**2) / (2 * max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK))
  return min(a_track, a_req) if a_req < -MOONPILOT_APPROACH_DECEL else a_track


def jerk_limit(a_cmd, a_prev, dt) -> float:
  """Rate limit, asymmetric and urgency-scaled: comfort jerk normally, emergency jerk by the time
  the command reaches ACCEL_MIN."""
  down = float(np.interp(a_cmd, [ACCEL_MIN, -2.0], [MOONPILOT_JERK_EMERGENCY, MOONPILOT_JERK_DOWN]))
  return float(np.clip(a_cmd, a_prev - down * dt, a_prev + MOONPILOT_JERK_UP * dt))


def lead_state_at(lead, t, v_ego, a_ego) -> tuple[float, float, float]:
  """Gap, lead speed and lead accel at time t: lead accel decays as exp(-aLeadTau t^2 / 2)
  (radard's own decay model), ego advances at constant a_ego."""
  a_traj = lead.aLeadK * math.exp(-lead.aLeadTau * t**2 / 2.0)
  v_lead = max(0.0, lead.vLead + 0.5 * (lead.aLeadK + a_traj) * t)
  x_lead = max(0.0, lead.vLead * t + 0.25 * (lead.aLeadK + a_traj) * t * t)
  v_ego_t = max(0.0, v_ego + a_ego * t)
  x_ego = max(0.0, (v_ego + v_ego_t) / 2 * t)
  return max(0.0, lead.dRel + x_lead - x_ego), v_lead, a_traj


def required_decel(v_ego, gap, v_lead) -> float:
  """Decel needed to avoid contact, CRASH_DISTANCE margin. FCW's whole input."""
  return -(v_ego**2 - max(v_lead, 0.0) ** 2) / (2 * max(gap - MOONPILOT_CRASH_DISTANCE, 0.1))


def policy(v_ego, leads, v_cruise, t_follow, e2e, model_accel, steer_angle_deg, CP, accel_coast, allow_throttle):
  """The smallest of the candidates, and which one it was.

  `leads` is a sequence of (source, gap, v_lead, a_lead) for the leads that are present, in
  radarState order, each carrying the slot it came from so that an absent leadOne does not make
  leadTwo report itself as lead0. The model's own accel goes last, so an upstream change that lets a
  NaN through the model cannot win the comparison and land in the plan.
  """
  candidates = [(cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle), LongitudinalPlanSource.cruise)]
  candidates += [(lead_accel(v_ego, gap, v_lead, a_lead, t_follow), source) for source, gap, v_lead, a_lead in leads]
  if e2e:
    candidates.append((float(model_accel), LongitudinalPlanSource.e2e))
  accel, source = min(candidates, key=lambda c: c[0])
  return float(accel), source


class MoonpilotLongitudinalPlanner:
  """Same public surface as upstream's planner — `output_a_target`, `output_should_stop`, `fcw`,
  `publish` — because the maneuver harness and controlsd read those names, not the type."""

  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.dt = dt
    self.params = Params()
    self.action_t = CP.longitudinalActuatorDelay + DT_MDL

    self.output_a_target = init_a
    self.output_should_stop = False
    self.fcw = False
    self.crash_cnt = 0
    self.allow_throttle = True
    self.source = LongitudinalPlanSource.cruise
    self.solve_time = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.v_desired_trajectory[:] = init_v
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    start = time.monotonic()

    CS = sm['carState']
    v_ego = max(CS.vEgo, 0.0)
    a_ego = float(np.clip(CS.aEgo, ACCEL_MIN, ACCEL_MAX))

    v_cruise = min(CS.vCruise, V_CRUISE_MAX) * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0
    if not math.isfinite(v_cruise):
      # Upstream lets a NaN set speed sit in its cruise state forever and survives only because
      # min() skips it. Holding the current speed is the honest reading of "no set speed".
      v_cruise = v_ego

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = (sm['controlsState'].longControlState == LongCtrlState.off) if self.CP.openpilotLongitudinalControl else (not sm['selfdriveState'].enabled)
    reset_state = reset_state or CS.vCruise == V_CRUISE_UNSET
    if reset_state:
      self.output_a_target = a_ego

    accel_coast = coast_accel(sm['carControl'].orientationNED[1]) if len(sm['carControl'].orientationNED) == 3 else ACCEL_MAX
    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > MOONPILOT_ALLOW_THROTTLE_THRESHOLD or v_ego <= MOONPILOT_MIN_ALLOW_THROTTLE_SPEED

    steer_angle = CS.steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg
    t_follow = self._t_follow(sm)
    e2e = sm['selfdriveState'].experimentalMode
    model_accel = sm['modelV2'].action.desiredAcceleration
    leads = [
      (source, lead)
      for source, lead in ((LongitudinalPlanSource.lead0, sm['radarState'].leadOne), (LongitudinalPlanSource.lead1, sm['radarState'].leadTwo))
      if lead.present
    ]

    # Delay compensation by state prediction: where the car and the leads will be when this command
    # reaches the actuator. The policy is evaluated there, not inverted back through the plan.
    a_prev = float(self.output_a_target)
    v_pred = max(0.0, v_ego + a_prev * self.action_t)
    lead_states = [(source, *lead_state_at(lead, self.action_t, v_ego, a_prev)) for source, lead in leads]
    a_cmd, source = policy(v_pred, lead_states, v_cruise, t_follow, e2e, model_accel, steer_angle, self.CP, accel_coast, self.allow_throttle)

    a_target = float(np.clip(jerk_limit(a_cmd, a_prev, self.dt), ACCEL_MIN, ACCEL_MAX))
    if not math.isfinite(a_target):
      # One guard, so that no input can latch a NaN into the command.
      a_target = 0.0

    self.v_desired_trajectory, self.a_desired_trajectory = self._trajectory(
      v_ego, a_target, leads, v_cruise, t_follow, e2e, model_accel, steer_angle, accel_coast
    )
    self.j_desired_trajectory = np.gradient(self.a_desired_trajectory, MOONPILOT_CONTROL_T_IDX)

    crash = any(lead.modelProb > MOONPILOT_FCW_MODEL_PROB and required_decel(v_ego, lead.dRel, lead.vLead) < MOONPILOT_FCW_DECEL for _, lead in leads)
    self.crash_cnt = self.crash_cnt + 1 if crash else 0
    fcw = self.crash_cnt > MOONPILOT_FCW_COUNT and not CS.standstill
    if fcw and not self.fcw:
      cloudlog.info("moonpilot FCW triggered")
    self.fcw = fcw

    self.output_should_stop = should_stop(v_ego, a_target) or (e2e and sm['modelV2'].action.shouldStop)
    self.output_a_target = a_target
    self.source = source
    self.solve_time = time.monotonic() - start

  def _trajectory(self, v_ego, a_target, leads, v_cruise, t_follow, e2e, model_accel, steer_angle, accel_coast):
    """The same policy rolled forward over the published horizon, from (v_ego, a_target)."""
    speeds = np.zeros(CONTROL_N)
    accels = np.zeros(CONTROL_N)
    v, a, t_prev = v_ego, a_target, 0.0
    for i, t in enumerate(MOONPILOT_CONTROL_T_IDX):
      states = [(source, *lead_state_at(lead, float(t), v_ego, a)) for source, lead in leads]
      a_cmd, _ = policy(v, states, v_cruise, t_follow, e2e, model_accel, steer_angle, self.CP, accel_coast, self.allow_throttle)
      a = float(np.clip(jerk_limit(a_cmd, a, float(t) - t_prev), ACCEL_MIN, ACCEL_MAX))
      speeds[i] = v
      accels[i] = a
      v = max(0.0, v + a * (float(t) - t_prev))
      t_prev = float(t)
    return speeds, accels

  def _t_follow(self, sm):
    personality = sm['selfdriveState'].personality.raw
    # KeyError is not possible for the three current members; an upstream fourth would fall back to
    # the standard gap rather than raise in plannerd.
    base = MOONPILOT_T_FOLLOW.get(personality, MOONPILOT_T_FOLLOW[int(Personality.standard)])
    if enabled(LEAD_LATERAL, self.params):
      base *= float(np.interp(nearest_lead_in_path(sm), [0.0, 1.0], [MOONPILOT_OUT_OF_PATH_T_FOLLOW, 1.0]))
    return base

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)


def moonpilot_longitudinal_planner(CP, params: Params | None = None) -> MoonpilotLongitudinalPlanner | None:
  """The seam's fork side: the fork planner when the driver wants it, None to leave upstream's in
  place. The param is read once, here, so the toggle takes a restart."""
  if not (enabled(LONGITUDINAL, params or Params()) and CP.openpilotLongitudinalControl):
    return None
  return MoonpilotLongitudinalPlanner(CP)
