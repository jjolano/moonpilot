"""moonpilot's longitudinal planner: the fork's own answer to "what acceleration?".

Upstream solves an acados MPC every frame. This is closed-form — no solver, no generated C — and
every number in it is fork-owned, so tuning happens here instead of inside a generated optimization
problem. The policy is three candidates, and the smallest wins:

  - a spacing regulator that holds ``gap == STOP_DISTANCE + t_follow * v_ego`` against the nearest
    lead, gaining rigidity as the gap closes (all the slack is in the time gap, none in the gains);
  - the time gap itself, biased by ``MoonpilotLeadLateral`` when the lead is predicted to leave the
    path — the fork's lateral prediction reaches the longitudinal policy here, since there is no
    MPC danger zone to scale;
  - a time-to-collision approach term that takes over once the headway it wants passes the
    regulator's braking authority, and behind it a stopping floor — the old kinematic term kept as a
    bound, since a proportional TTC term ramps rather than stops and binds too late above ~34 m/s.
    The deeper of the two wins, so the fork's closing-rate response reaches the plan wherever it asks
    for more than stopping needs, and the floor covers the rest. The handover into either is a step in
    the candidate; it is arbitration that keeps that step out of the output while the jerk limit turns
    what remains into a rate;
  - the cruise term, and the model's own accel — raw in experimental mode, and outside it a
    braking-only candidate behind ``MoonpilotModelBraking`` (see ``model_candidate``) — the same
    candidates upstream arbitrates between.

``MoonpilotCurveSpeed`` adds two more terms *inside* the cruise slot, so the reported source stays
``cruise`` and ``min`` means they can only ever add braking: a kinematic pre-brake against the model
path's own curvature, and a proportional regulator on the lateral accel the car is actually pulling.
``moonpilot/curve.py`` is where the math and its constants live; both are inert without the model's
path arrays and, for the in-curve term, ``vehicleParameters.valid``.

Two things are deliberately not upstream's:

  - delay compensation is done by predicting the state at the actuator delay and evaluating the
    policy there, not by inverting the published plan through ``get_accel_from_plan``. That inverse
    divides by the delay, so a step in the plan comes out amplified by ``action_t / dt`` — measured
    at 45 m/s^3 and instant ACCEL_MIN saturation in simulation. The delay itself is measured onroad
    by ``moonpilot.latency``: ``action_t`` is ``max(CP.longitudinalActuatorDelay, that estimate) +
    the planner's own period`` (DT_MDL in production), since upstream's constant is a cookie-cutter
    default and a measured value may only lengthen the projection.

  - the lead's acceleration is the fork's own estimate — the least-squares slope of ``vLead`` over
    ``moonpilot.lead.LeadAccelEstimator``'s window — because radard's ``aLeadK`` is a Kalman filter
    with a 0.49 s time constant, and a lead braking at -3.5 m/s^2 reached -2.0 m/s^2 of command
    0.25 s later through it than through the slope. ``aLeadK`` stays as the fallback.

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
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, should_stop
from openpilot.selfdrive.modeld.constants import ModelConstants

from moonpilot.curve import (
  MOONPILOT_CURVE_BIAS_KEY,
  MOONPILOT_CURVE_BIAS_MAX,
  MOONPILOT_CURVE_BIAS_MIN_SPEED,
  MOONPILOT_CURVE_BIAS_MIN_SAMPLES,
  MOONPILOT_CURVE_BIAS_PERSIST_EVERY,
  LatAccelBiasEstimator,
  curve_accel,
  curve_targets,
  hold_speed,
  lat_accel_budget,
  lat_accel_hold,
  predicted_lat_accel,
)
from moonpilot.features import CURVE_SPEED, LEAD_LATERAL, LONGITUDINAL, MODEL_BRAKING, enabled
from moonpilot.latency import (
  MOONPILOT_LAG_BLOCKS_NEEDED,
  MOONPILOT_LAG_KEY,
  MOONPILOT_LAG_LOG_DELTA,
  MOONPILOT_LAG_MAX,
  MOONPILOT_LAG_MIN,
  MOONPILOT_LAG_MIN_SPEED,
  MOONPILOT_LAG_PERSIST_EVERY,
  LongLagEstimator,
)
from moonpilot.lead import LeadAccelEstimator, nearest_lead_in_path
from moonpilot.slam import ego_speed_correction

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
MOONPILOT_APPROACH_DECEL = 1.0  # m/s^2; the spacing regulator's braking authority, and the
# decel the approach term binds past — one number, so the
# two terms meet at the same output
MOONPILOT_TTC_TARGET = 3.0  # s; headway the approach term holds the closing rate inside. Lower is
# shallower — a_ttc = -K*(closing - slack/T) deepens with T, so this
# is the dial between the floor's plateau (T below ~2 the term never
# governs at all) and a hard late ramp: peak on a 25 m/s approach to a
# stopped lead is 1.23 here against 1.42 at 5.0 and 1.57 at 8.0, and the
# narrower window is also the *smoother* one, 4 command reversals over
# the maneuver against 13 at 5.0. What it gives up is coverage: TTC
# governs gaps 22-65 m at 25 m/s rather than 20-117 m, so a gently
# closing lead is the floor's business either way (measured: TTC governs
# 4 of 500 frames at T=3 on a -2 m/s^2 lead brake, 20 at T=5)
MOONPILOT_K_TTC = 1.0  # 1/s on the excess closing rate
MOONPILOT_MIN_SLACK = 0.5  # m; floor on the stopping term's braking-distance denominator
MOONPILOT_LEAD_PREVIEW_T = 1.0  # s of the lead's own braking credited to the safety term
MOONPILOT_A_LEAD_MIN = -10.0  # m/s^2; bounds on a lead's accel estimate, upstream's (long_mpc.process_lead)
MOONPILOT_A_LEAD_MAX = 5.0
MOONPILOT_OUT_OF_PATH_T_FOLLOW = 0.7  # time-gap scale for a lead predicted to leave the path
MOONPILOT_K_CRUISE = 1.0  # 1/s on the speed error
MOONPILOT_A_CRUISE_MIN = -1.2  # m/s^2; cruise never brakes harder than this
MOONPILOT_A_CRUISE_MAX_BP = [0.0, 10.0, 25.0, 40.0]  # m/s
MOONPILOT_A_CRUISE_MAX_V = [1.6, 1.2, 0.8, 0.6]  # m/s^2
MOONPILOT_A_TOTAL_MAX_BP = [20.0, 40.0]  # m/s
MOONPILOT_A_TOTAL_MAX_V = [1.7, 3.2]  # m/s^2 combined accel budget
MOONPILOT_JERK_UP = 1.5  # m/s^3
MOONPILOT_JERK_DOWN = 2.0  # m/s^3; the comfort jerk, and it is the approach's onset edge: stepping
# from the cruise term onto the floor's -1.0 m/s^2 takes 0.50 s here
# against 0.25 s at 4.0, which is what the step reads as from the seat.
# `jerk_limit` interpolates from this at -2.0 m/s^2 to JERK_EMERGENCY at
# ACCEL_MIN, so this softens the gentle onset and leaves emergency braking
# at 10.0 m/s^3 untouched — a command past -2.0 is on that ramp already.
# It costs one frame of brake-onset latency (-1.0 m/s^2 at 0.30 s instead
# of 0.25 s) and nothing in the delivered stopping distance: the ramp is
# shorter than the floor's own headroom, measured 6.05 m of end gap and no
# contact at 90 / 118.8 / 129.6 / 144 kph against both a stopped and a
# braking lead.
MOONPILOT_JERK_EMERGENCY = 10.0  # m/s^3, reached at ACCEL_MIN
MOONPILOT_ALLOW_THROTTLE_THRESHOLD = 0.4
MOONPILOT_MIN_ALLOW_THROTTLE_SPEED = 2.5  # m/s
MOONPILOT_MODEL_BRAKE_THRESHOLD = -0.5  # m/s^2; outside experimental mode the model is ignored
# until it asks for at least this much braking, and past that its ask goes in whole: the floor under
# it is the actuator's own ACCEL_MIN, not a fork value. See `model_candidate` for why a fork-owned
# floor above that was measured and removed.
MOONPILOT_CRASH_DISTANCE = 0.25  # m; FCW contact margin
MOONPILOT_FCW_DECEL = -4.0  # m/s^2 required decel that means it cannot be avoided
MOONPILOT_FCW_COUNT = 2  # frames above threshold before FCW latches
MOONPILOT_FCW_MODEL_PROB = 0.9
# m/s; the standstill deadband `should_stop` is gated on, applied to the ego's own speed and to the
# speed of the lead the policy is actually following. Above it on both counts the plan is asking the car
# to move, so declaring rest is wrong; at or below it on either — a parked car, or a stopped lead — the
# upstream predicate is untouched, which is the parked-behind-a-stopped-lead case it was written for.
# Matches the fork's own standstill convention (`MOONPILOT_STANDSTILL_SPEED`).
MOONPILOT_SHOULD_STOP_SPEED = 0.1
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

  The approach is two terms, and the deeper one wins against the regulator's capped output.

  The first is time-to-collision: it holds the closing rate inside what the slack affords at
  TTC_TARGET seconds of headway — ``closing <= slack / TTC_TARGET`` — and binds once honoring that
  needs more than the approach decel, i.e. ``slack < TTC_TARGET * (closing - APPROACH_DECEL /
  K_TTC)``: with the shipped numbers, ``slack < 3 * (closing - 1)``, which is a 78 m gap at 25 m/s
  on a stopped lead. Binding is not governing, though: it reaches the output only where it is also
  deeper than the floor below, which on that same approach is the gap window 22 m … 65 m — and it is
  only deeper than the floor for T above 2, below which this term never governs at all. Inside that
  window it is what makes the approach respond to a lead that starts slowing, because it reads the
  closing rate, which the lead's own speed history drives.

  The second is the stopping floor, the old kinematic term kept as a bound rather than as the
  approach: the exact decel that arrives at STOP_DISTANCE with the lead's preview-corrected speed,
  which binds from ``slack == v_ego^2 / 2``. It is not redundant. A TTC term is proportional on the
  closing rate, so it ramps — and ramping is not stopping: TTC_TARGET = 5 binds at a slack of
  ``5 * (v - 1)``, which above ~34 m/s is *inside* the ``v^2 / 7`` that stopping at ACCEL_MIN needs,
  and on the TTC term alone this car reaches a stopped lead at 14 m/s from 36 m/s. The floor binds
  earlier than the stopping distance and `min` cannot out-vote it, so it holds where the TTC term
  asks for less; where the TTC term asks for more, the TTC term wins, which is how the fork's
  response reaches the plan. Below 1 m/s of closing rate neither binds and the regulator alone owns
  the creep.

  The handover in either case is a step, not a crossover: at the crossing the regulator's output is
  whatever the spacing error says, positive while the gap is still wide — +67.9 m/s^2 at 25 m/s and
  the 318.5 m stopping crossing — and the approach terms replace it from there. That step is only
  safe because it is a *candidate*: `policy` takes the minimum, so while the lead asks for more than
  the cruise term the cruise term governs and the output never sees it. What the output steps by is
  therefore cruise's cap minus the approach decel, and the caller's jerk limit turns that into a rate.
  """
  gap = max(float(gap), 0.0)
  v_lead = max(float(v_lead), 0.0)
  v_lead_eff = max(0.0, v_lead + min(float(a_lead), 0.0) * MOONPILOT_LEAD_PREVIEW_T)
  a_track = max(MOONPILOT_K_GAP * (gap - MOONPILOT_STOP_DISTANCE - t_follow * v_ego) + MOONPILOT_K_V * (v_lead - v_ego), -MOONPILOT_APPROACH_DECEL)
  closing = v_ego - v_lead_eff
  if closing <= 0.0:
    return a_track  # not closing: nothing to brake for
  slack = max(gap - MOONPILOT_STOP_DISTANCE, 0.0)
  a_ttc = -MOONPILOT_K_TTC * (closing - slack / MOONPILOT_TTC_TARGET)
  a = min(a_track, a_ttc) if a_ttc < -MOONPILOT_APPROACH_DECEL else a_track
  # The stopping floor, and it is what makes this law safe rather than merely responsive. The TTC term
  # is proportional on the closing rate, so against a lead the ego is gaining on fast it ramps — and
  # ramping is not stopping: at 36 m/s with TTC_TARGET = 5 it binds 175 m out, where coming to rest
  # behind a stopped lead needs 185 m, and the car reaches the lead at 14 m/s. Nothing upstream of
  # this reaches that regime — the maneuver suite's fastest stopped-lead approach is 25 m/s — so the
  # floor is what covers it. It is the old kinematic term kept as a bound rather than as the approach:
  # the exact decel that arrives at STOP_DISTANCE with the lead's speed, binding from a slack of
  # v_ego^2 / 2, which is always inside the stopping distance. `min` cannot out-vote it, so it holds
  # where the TTC term asks for less — and where the TTC term asks for more, the TTC term wins, which
  # is the fork's own response and the reason both are here.
  a_stop = -(v_ego**2 - v_lead_eff**2) / (2 * max(gap - MOONPILOT_STOP_DISTANCE, MOONPILOT_MIN_SLACK))
  return min(a, a_stop) if a_stop < -MOONPILOT_APPROACH_DECEL else a


def jerk_limit(a_cmd, a_prev, dt) -> float:
  """Rate limit, asymmetric and urgency-scaled: comfort jerk normally, emergency jerk by the time
  the command reaches ACCEL_MIN."""
  down = float(np.interp(a_cmd, [ACCEL_MIN, -2.0], [MOONPILOT_JERK_EMERGENCY, MOONPILOT_JERK_DOWN]))
  return float(np.clip(a_cmd, a_prev - down * dt, a_prev + MOONPILOT_JERK_UP * dt))


def lead_accel_estimate(a_lead: float) -> float:
  """A lead's accel estimate, bounded the way upstream's MPC bounds it: radard's Kalman reports
  implausible values on a track that just appeared, and the maneuver plant differentiates a stepped
  lead speed into hundreds of m/s^2."""
  return min(max(float(a_lead), MOONPILOT_A_LEAD_MIN), MOONPILOT_A_LEAD_MAX)


def lead_state_at(lead, t, x_ego, a_lead, a_lead_tau) -> tuple[float, float, float]:
  """Gap, lead speed and lead accel at time t, given the ego's own travel x_ego by then.

  Lead accel decays as exp(-aLeadTau t^2 / 2) — radard's own decay model — and the lead's travel is
  the integral of its *clamped* speed: a lead that brakes to a stop inside the horizon has covered
  its stopping distance, where the unclamped integral walked it backwards and then clamped the travel
  to zero. Exact whenever aLeadTau == 0, which is what the estimator reports for anything braking
  hard enough to matter: the decay is dropped while |a| >= 0.5, radard's own rule (`radard.py:76-79`)
  applied to the fork's estimate instead of radard's `aLeadK`.

  Both accel and decay are the caller's, from `moonpilot.lead.LeadAccelEstimator` — they are one
  judgment, and pairing the fork's accel with radard's decay is what let the published plan's tail
  decay a brake onset away. Against a decaying accel the travel below is the trapezoid, measured
  against exact integration over aLeadTau's reachable range (0.3 from the vision path, otherwise a
  FirstOrderFilter decaying toward 0 from 1.5) within 0.1 m across the 0.55 s command horizon, and
  6.2 m at the 2.5 s tail at the clip's own -10 m/s^2 bound (3.1 m for |a| <= 5) — and that tail is
  the published plan, not the command.
  """
  a_lead = lead_accel_estimate(a_lead)
  a_traj = a_lead * math.exp(-a_lead_tau * t**2 / 2.0)
  a_avg = 0.5 * (a_lead + a_traj)
  v_lead = max(0.0, float(lead.vLead))
  v_end = v_lead + a_avg * t
  if v_end > 0.0:
    x_lead = 0.5 * (v_lead + v_end) * t
  else:  # stopped inside t: its own stopping distance at that average decel
    x_lead, v_end = (v_lead**2 / (-2.0 * a_avg) if a_avg < 0.0 else 0.0), 0.0
  return max(0.0, lead.dRel + x_lead - x_ego), v_end, a_traj


def required_decel(v_ego, gap, v_lead, a_lead=0.0) -> float:
  """Decel needed to avoid contact, CRASH_DISTANCE margin. FCW's whole input.

  Two constraints, and the harder one wins: matching the speed of a lead that holds it, and stopping
  short of where a lead that keeps braking at its current rate comes to rest. Without the second, a
  lead braking hard at matched speed reads as no threat at all until the speed measurement has
  caught up — 0.7 s at -8 m/s^2, which is most of the margin the warning exists to buy.
  """
  slack = max(gap - MOONPILOT_CRASH_DISTANCE, 0.1)
  v_lead = max(float(v_lead), 0.0)
  a_lead = lead_accel_estimate(a_lead)
  a_match = -(v_ego**2 - v_lead**2) / (2 * slack)
  if a_lead >= 0.0:
    return a_match
  return min(a_match, -(v_ego**2) / (2 * (slack + v_lead**2 / (-2 * a_lead))))


def model_candidate(model, e2e, allowed) -> float | None:
  """The model's own accel as a candidate, and whether it gets one this frame.

  Experimental mode is unchanged: the accel goes in raw, which is what that mode means. Outside it
  the model is a *braking* input on top of the deterministic policy, gated by a deadband and
  otherwise unbounded — the floor under it is the actuator's own, not a fork value.

  The deadband is because `policy` takes the minimum: a model accel dithering either side of zero
  would be a one-way ratchet against the cruise term, and the car would settle below the set speed
  on a clear road. Below the threshold the model is not asking for anything a driver would feel.

  Past it the ask goes in whole. What bounds it is `ACCEL_MIN`, and that bound is unconditional:
  enforced by the clip every command passes through in `update`, by the identical clip in the
  published rollout, and again by the controller from the car's own `accel_limits` — so a deep ask
  still reaches full authority in an emergency, at `MOONPILOT_JERK_EMERGENCY`.

  A fork-owned floor above that was tried and removed. Over the 235-minute local corpus it bound
  2.23 min in 61 of 494 admitted episodes, withholding a median 0.25 m/s^2 there and at most 1.23,
  and every one of those frames coincided with the car already braking — what it withheld was
  braking the driver had already begun. Its release valve never fired at all: `hardBrakePredicted`
  was false on all 282,554 frames, as the composite of a 5 m/s^2 head peaking at 0.118 against its
  0.15 gate and a 3 m/s^2 head that does reach 0.955.

  None means no candidate, which is exactly the planner this fork had before this existed. A NaN
  takes that branch too, since the comparison is false.
  """
  a = float(model.action.desiredAcceleration)
  if not e2e and (not allowed or not a < MOONPILOT_MODEL_BRAKE_THRESHOLD):
    return None
  return a


def policy(v_ego, leads, v_cruise, t_follow, e2e, model_accel, steer_angle_deg, CP, accel_coast, allow_throttle, curve=None, x_ego=0.0, v_hold=math.inf):
  """The smallest of the candidates, and which one it was.

  `leads` is a sequence of (source, gap, v_lead, a_lead) for the leads that are present, in
  radarState order, each carrying the slot it came from so that an absent leadOne does not make
  leadTwo report itself as lead0. The model's own accel goes last, so an upstream change that lets a
  NaN through the model cannot win the comparison and land in the plan.

  `curve` is `moonpilot.curve.curve_targets`' output and `v_hold` its in-curve setpoint, both fed into
  the cruise slot as a minimum — so the reported source stays `cruise`, and the two curve terms are
  live in experimental mode too. That is the point of putting them there rather than beside the model
  candidate: `min` means they can only ever add braking to whatever the model asked for, never
  substitute for it.
  """
  a_curve = min(curve_accel(v_ego, x_ego, curve), lat_accel_hold(v_ego, v_hold))
  a_cruise = min(cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle), a_curve)
  candidates = [(a_cruise, LongitudinalPlanSource.cruise)]
  candidates += [(lead_accel(v_ego, gap, v_lead, a_lead, t_follow), source) for source, gap, v_lead, a_lead in leads]
  if model_accel is not None:
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
    # The measured command -> delivered-accel lag (`moonpilot/latency.py`), seeded from the last
    # drive when there is one. `action_t` below is the whole of the delay compensation, and upstream's
    # `longitudinalActuatorDelay` is a cookie-cutter default for every car without an override
    # (`opendbc/car/interfaces.py`, `# TODO estimate car specific lag`). The learned value may only
    # lengthen the projection, so an unset param, a stale one and a measurement below the stock
    # constant are all exactly the planner this fork had before.
    self.long_lag = LongLagEstimator(CP, dt)
    self.long_lag_logged = self.long_lag.applied_delay()
    # `isinstance` is load-bearing: the tests' FakeParams answers every key with a bool, and a bool is
    # not a float, so no test is seeded. The real param is a FLOAT whose "0.0" default is below the
    # ROI floor and so is not a seed either.
    seeded = self.params.get(MOONPILOT_LAG_KEY, return_default=True)
    if isinstance(seeded, float) and seeded >= MOONPILOT_LAG_MIN:
      self.long_lag.seed(min(seeded, MOONPILOT_LAG_MAX), MOONPILOT_LAG_BLOCKS_NEEDED)
      self.long_lag_logged = self.long_lag.applied_delay()
    self.action_t = self.long_lag.applied_delay() + dt
    # The curve speed control's per-car calibration (`moonpilot/curve.py`): realized lateral accel over
    # what the path predicted, one-sided so it can only ever plan for less speed. Seeded from the last
    # drive, and the vehicle model is what turns a measured steer angle into a measured curvature.
    self.VM = VehicleModel(CP)
    self.lat_bias = LatAccelBiasEstimator(dt)
    seeded_scale = self.params.get(MOONPILOT_CURVE_BIAS_KEY, return_default=True)
    if isinstance(seeded_scale, float) and seeded_scale > 1.0:
      self.lat_bias.seed(min(seeded_scale, MOONPILOT_CURVE_BIAS_MAX), MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    self.frames = 0
    # One per radarState slot, ticked every frame so an absent or replaced lead resets its window.
    self.lead_accel = (LeadAccelEstimator(dt), LeadAccelEstimator(dt))

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
    # The rolling-window ego correction, when there is a fresh one to apply: the wheel speed is a
    # scale error away from the truth -- ~5 % on a worn tyre set -- and every term below plans from
    # it, so the correction goes in at the source rather than into one of the candidates. Zero when
    # the feature is off, nothing is published, the correction is stale or its publisher is dead,
    # which makes this exactly the plan this planner had before the correction existed.
    v_ego = max(0.0, v_ego + ego_speed_correction(sm))
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

    # Learn the chain's lag from the command that was in effect last frame (`output_a_target` is
    # assigned at the end of this one) and the acceleration the car answered with. Gated to the frames
    # where that command is what moves the car: when the planner resets, `output_a_target` *is*
    # `a_ego` and carries no information about the chain, the car's own ACC owns acceleration outside
    # `pid`, and a standstill, a pedal or a force-decel stop is not the policy either. Every other gate
    # — excitation, recovery buffer, blocks — belongs to the estimator.
    lag_valid = (
      not reset_state
      and sm['controlsState'].longControlState == LongCtrlState.pid
      and v_ego > MOONPILOT_LAG_MIN_SPEED
      and not CS.standstill
      and not CS.brakePressed
      and not CS.gasPressed
      and not sm['controlsState'].forceDecel
    )
    self.long_lag.update(self.output_a_target, a_ego, lag_valid)
    self.action_t = self.long_lag.applied_delay() + self.dt

    accel_coast = coast_accel(sm['carControl'].orientationNED[1]) if len(sm['carControl'].orientationNED) == 3 else ACCEL_MAX
    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > MOONPILOT_ALLOW_THROTTLE_THRESHOLD or v_ego <= MOONPILOT_MIN_ALLOW_THROTTLE_SPEED

    steer_angle = CS.steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg
    t_follow = self._t_follow(sm)
    e2e = sm['selfdriveState'].experimentalMode
    model_accel = model_candidate(sm['modelV2'], e2e, enabled(MODEL_BRAKING, self.params))

    # The curve speed control. The measured curvature is the car's own — `-VM.calc_curvature` of the
    # steer angle, the same idiom controlsd and `moonpilot/latcontrol.py` use — and the model's path
    # is what it is scored against. The measurement uses the raw `CS.vEgo` rather than the
    # slam-corrected `v_ego`: it is a measurement of the car, not a plan input.
    vp = sm['vehicleParameters']
    self.VM.update_params(max(vp.stiffnessFactor, 0.1), max(vp.steerRatio, 0.1))
    measured_curvature = -self.VM.calc_curvature(math.radians(CS.steeringAngleDeg - vp.angleOffsetDeg), CS.vEgo, vp.roll)
    curve_allowed = enabled(CURVE_SPEED, self.params)
    # Only frames where the fork's own lateral control is what steers the car, at speed and off the
    # pedals: a disengaged or overridden stretch would pair a prediction of the model's path with a
    # measurement of the driver's steering.
    bias_valid = (
      curve_allowed and vp.valid and sm['carControl'].latActive and not CS.steeringPressed and not CS.standstill and CS.vEgo > MOONPILOT_CURVE_BIAS_MIN_SPEED
    )
    self.lat_bias.update(predicted_lat_accel(sm['modelV2']), measured_curvature * CS.vEgo**2, bias_valid)
    scale = self.lat_bias.applied()
    curve = curve_targets(sm['modelV2'], curve_allowed, vp.roll, scale)
    v_hold = math.inf
    if curve_allowed:
      budget = float(lat_accel_budget(math.copysign(1.0, measured_curvature), vp.roll, scale))
      v_hold = hold_speed(measured_curvature, CS.vEgo, budget, CS.standstill, vp.valid)

    leads = []
    for estimator, (source, lead) in zip(
      self.lead_accel,
      ((LongitudinalPlanSource.lead0, sm['radarState'].leadOne), (LongitudinalPlanSource.lead1, sm['radarState'].leadTwo)),
      strict=True,
    ):
      a_lead, a_lead_tau = estimator.update(lead)  # every frame, present or not: that is what resets the window
      if lead.present:
        leads.append((source, lead, a_lead, a_lead_tau))

    # Delay compensation by state prediction: where the car and the leads will be when this command
    # reaches the actuator. The policy is evaluated there, not inverted back through the plan.
    #
    # The two inputs are not the same age, and `action_t` projects both by the same amount. radarState
    # lands ~one full cycle behind the modelV2 tick that polls it — measured over 22.8k ticks on 25
    # routes, median age 46.2 ms against carState's 4.1 ms, and radar the staler input on 100 % of
    # them — so the gap the message reports is from before `lead_age` of closing. Ageing the lead by
    # exactly that and the ego by nothing is what closes the difference; it needs no new constant
    # because upstream's own `commIssue` disengages at ten periods, so `lead_age` cannot usefully
    # exceed ~0.5 s, and the correction is self-limiting well before that (a stopped lead at 1 s of
    # staleness moves the stop by 0.42 m, against 3.54 m uncorrected).
    #
    # Defaulted rather than required: the maneuver plant hands a bare `dict` with no `logMonoTime` at
    # all, and a missing or equal stamp has to give the zero correction this planner applied before
    # it existed.
    log_mono = getattr(sm, 'logMonoTime', None) or {}
    model_mono = log_mono.get('modelV2')
    radar_mono = log_mono.get('radarState')
    lead_age = max(0.0, (model_mono - radar_mono) / 1e9) if model_mono and radar_mono else 0.0
    # Both halves of the stale interval, not just the lead's: the message's `dRel` predates the
    # command by `lead_age`, and `x_pred` carries only the travel over `action_t`. The lead's own
    # travel over `lead_age` rides on the extra time; the ego's rides on `x_stale`.
    x_stale = v_ego * lead_age

    a_prev = float(self.output_a_target)
    v_pred = max(0.0, v_ego + a_prev * self.action_t)
    x_pred = 0.5 * (v_ego + v_pred) * self.action_t
    lead_states = [(source, *lead_state_at(lead, self.action_t + lead_age, x_pred + x_stale, a_lead, a_lead_tau)) for source, lead, a_lead, a_lead_tau in leads]
    a_cmd, source = policy(
      v_pred,
      lead_states,
      v_cruise,
      t_follow,
      e2e,
      model_accel,
      steer_angle,
      self.CP,
      accel_coast,
      self.allow_throttle,
      curve=curve,
      x_ego=x_pred,
      v_hold=v_hold,
    )

    a_target = float(np.clip(jerk_limit(a_cmd, a_prev, self.dt), ACCEL_MIN, ACCEL_MAX))
    if not math.isfinite(a_target):
      # One guard, so that no input can latch a NaN into the command.
      a_target = 0.0

    self.v_desired_trajectory, self.a_desired_trajectory = self._trajectory(
      v_ego, a_target, leads, v_cruise, t_follow, e2e, model_accel, steer_angle, accel_coast, lead_age, curve, v_hold
    )
    self.j_desired_trajectory = np.gradient(self.a_desired_trajectory, MOONPILOT_CONTROL_T_IDX)

    crash = any(
      lead.modelProb > MOONPILOT_FCW_MODEL_PROB and required_decel(v_ego, lead.dRel, lead.vLead, a_lead) < MOONPILOT_FCW_DECEL for _, lead, a_lead, _ in leads
    )
    self.crash_cnt = self.crash_cnt + 1 if crash else 0
    fcw = self.crash_cnt > MOONPILOT_FCW_COUNT and not CS.standstill
    if fcw and not self.fcw:
      cloudlog.info("moonpilot FCW triggered")
    self.fcw = fcw

    # `should_stop`'s predicate is upstream's, and so is its `v_ego < 0.3` gate — but upstream's planner
    # only reaches those speeds on the way to rest, where declaring a stop is right. The fork's spacing
    # regulator trails a lead at whatever speed the lead is doing, and while it does the plan is asking
    # for motion: the command sits on zero, inside the predicate's 0.1 threshold, so the state machine
    # is thrown into `stopping` ~13 times a second and the stopping ramp spends the creep fighting the
    # plan. Measured against a lead creeping at 0.30 m/s the car parked 0.37 m behind its own setpoint
    # with 202 mm/s of speed ripple, and the command dithering -0.14..+0.11 where the plan wanted ~0;
    # with the gate the same car sits on the setpoint to three decimals with 1.2 mm/s of ripple.
    #
    # Gated on the lead the policy actually followed — `source` is the candidate that won the minimum,
    # so the `lead_states` entry in that slot is the one being trailed — never on every present lead. A
    # moving `leadTwo` in the next lane while parked behind a stopped `leadOne` would otherwise lift
    # should-stop and with it `LongControlState.stopping`'s brake hold, and the car rolls: measured, the
    # delivered command went from CP.stopAccel to 0.00 on that case alone.
    #
    # The ego's own speed is required too, so a stationary car whose lead's reported speed jitters over
    # the deadband cannot lose that hold either. At a standstill the gate is therefore inert and the
    # upstream predicate decides on its own, exactly as it did before this existed.
    #
    # Everything else is untouched: a stopped or absent lead is the case the predicate was written for,
    # and resume still works because a lead that pulls away lifts the command over the 0.1 threshold by
    # itself — both `long_control_state_trans`'s cruise-standstill pin and controlsd's resume read this
    # flag, so the car releases on the first frame rather than waiting for the gap to open.
    # The model's own `shouldStop` stays experimental-only, unlike its accel. It is
    # `should_stop(v_ego, desiredAcceleration)` computed in modeld (`modeld.py:59`), i.e. `v_ego <
    # 0.3 and a < 0.1` — precisely the creeping-behind-a-lead state the `trailing` gate above exists
    # to keep out of `LongControlState.stopping`, so admitting it here would undo that fix from the
    # other side.
    trailing = v_ego > MOONPILOT_SHOULD_STOP_SPEED and any(s == source and v_lead > MOONPILOT_SHOULD_STOP_SPEED for s, _, v_lead, _ in lead_states)
    self.output_should_stop = (should_stop(v_ego, a_target) and not trailing) or (e2e and sm['modelV2'].action.shouldStop)
    self.output_a_target = a_target
    self.source = source
    # Persist the learned value so the next boot projects through it from the first frame. Gated on a
    # trusted estimate, so an unestimated or invalid one never becomes the next drive's constant.
    self.frames += 1
    if self.frames % MOONPILOT_LAG_PERSIST_EVERY == 0 and self.long_lag.status == 'estimated':
      self._persist_lag()
    if self.lat_bias.frames % MOONPILOT_CURVE_BIAS_PERSIST_EVERY == 0 and self.lat_bias.status == 'estimated':
      self._persist_lat_scale()
    self.solve_time = time.monotonic() - start

  def _persist_lag(self):
    value = round(self.long_lag.estimate, 3)
    self.params.put(MOONPILOT_LAG_KEY, value)
    if abs(value - self.long_lag_logged) > MOONPILOT_LAG_LOG_DELTA:
      cloudlog.info(f"moonpilot longitudinal lag {value:.3f} s over {self.long_lag.valid_blocks} blocks, action_t {self.action_t:.3f} s")
      self.long_lag_logged = value

  def _persist_lat_scale(self):
    """Persist the learned value so the next boot plans with it from the first frame. Gated on a
    trusted estimate, so an unestimated or invalid one never becomes the next drive's scale."""
    value = round(self.lat_bias.estimate, 3)
    self.params.put(MOONPILOT_CURVE_BIAS_KEY, value)
    cloudlog.info(f"moonpilot curve lateral scale {value:.3f} over {self.lat_bias.samples} paired frames")

  def _trajectory(self, v_ego, a_target, leads, v_cruise, t_follow, e2e, model_accel, steer_angle, accel_coast, lead_age=0.0, curve=None, v_hold=math.inf):
    """The same policy rolled forward over the published horizon, from (v_ego, a_target). The ego's
    own travel is carried in x, so the gap the leads are rolled against is the gap this plan
    produces. The old loop advanced the state by the *backward* interval and rolled the leads against
    a gap recomputed from the initial speed and the loop's current accel: on a closing lead at
    25 m/s, its speeds sat up to 0.42 m/s away from the consistent rollout's. `lead_age` carries the
    same staleness correction `update` applies, so the published plan is the same prediction the
    command was taken from rather than a fresher one. `curve` and `v_hold` ride along for the same
    reason: the rollout is the policy the command came from, with the curve's own travel accumulated
    in `x` and the measured curvature held across the horizon."""
    speeds = np.zeros(CONTROL_N)
    accels = np.zeros(CONTROL_N)
    v, a, x, t_prev = v_ego, a_target, 0.0, 0.0
    for i, t_idx in enumerate(MOONPILOT_CONTROL_T_IDX):
      t = float(t_idx)
      states = [(source, *lead_state_at(lead, t + lead_age, x + v_ego * lead_age, a_lead, a_lead_tau)) for source, lead, a_lead, a_lead_tau in leads]
      a_cmd, _ = policy(
        v, states, v_cruise, t_follow, e2e, model_accel, steer_angle, self.CP, accel_coast, self.allow_throttle, curve=curve, x_ego=x, v_hold=v_hold
      )
      a = float(np.clip(jerk_limit(a_cmd, a, t - t_prev), ACCEL_MIN, ACCEL_MAX))
      speeds[i] = v
      accels[i] = a
      dt = float(MOONPILOT_CONTROL_T_IDX[i + 1]) - t if i + 1 < CONTROL_N else 0.0
      v_next = max(0.0, v + a * dt)
      x += 0.5 * (v + v_next) * dt
      v, t_prev = v_next, t
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
