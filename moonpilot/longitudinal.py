"""moonpilot's longitudinal planner: the fork's own answer to "what acceleration?".

Upstream solves an acados MPC every frame. This is closed-form — no solver, no generated C — and
every number in it is fork-owned, so tuning happens here instead of inside a generated optimization
problem. Braking candidates use `min` arbitration, and the cruise slot remains the set-speed
authority: an oversized lead gap never raises it above the ordinary speed-error request. The policy
is:

  - a lead-following law, bounded by a stopping floor;
  - a cruise candidate and the model's braking candidate, arbitrated by ``min``;
  - optional curve and corridor candidates inside the cruise slot.

The lead law aims for a band behind the lead's predicted speed and arrives at its edges
speed-matched. The stopping floor bounds that request while preserving a standstill margin.
``MoonpilotLeadLateral`` changes the time gap using a two-state prediction, not a continuous scale.
``MoonpilotCoastGrade`` adjusts only the cruise candidate; no safety candidate is weakened.

The planner predicts the state at ``action_t`` (actuator delay plus ``DT_MDL``) and evaluates the
policy there. The lead's acceleration estimate comes from ``moonpilot.lead.LeadAccelEstimator``;
radard's estimate remains the fallback.

``MoonpilotCurveSpeed`` adds three terms inside the cruise slot; their math lives in
``moonpilot/curve.py``. ``MoonpilotSqueeze`` adds a corridor candidate there too, and the feature
toggles are read per frame.

The delay estimate, lead law, and measured behavior are described in ``moonpilot/docs/longitudinal.md``.
"""

from typing import NamedTuple
import math
import time
from types import SimpleNamespace

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
  MOONPILOT_CURVE_BIAS_MIN_SPEED,
  MOONPILOT_CURVE_BIAS_TRACKING_TOLERANCE,
  MOONPILOT_CURVE_EXIT_JERK,
  MOONPILOT_CURVE_EXIT_LAT_ACCEL,
  MOONPILOT_CURVE_PATH_MAX_AGE,
  MOONPILOT_CURVE_PATH_CLOCK_SANITY,
  LatAccelBiasEstimator,
  curve_accel,
  curve_targets,
  hold_speed,
  lat_accel_budget,
  lat_accel_hold,
  predicted_lat_accel,
)
from moonpilot.pose import read_ego_pose
from moonpilot.corridor import RollingCorridor, squeeze_accel
from moonpilot.brake_band import MOONPILOT_BRAKE_BAND_FLOOR, BrakeBand, hold_timer
from moonpilot.engine_coast import engine_floor
from moonpilot.features import (BRAKE_BAND, COAST_GRADE, CURVE_SPEED, EXPERIMENTAL_MODE, HANDBACK_EASE, LEAD_LATERAL, LONGITUDINAL, MODEL_BRAKING,
                                 SQUEEZE, enabled, in_force)
from moonpilot.latency import MOONPILOT_LAG_MIN_SPEED, LongLagEstimator
from moonpilot.learned import PERSIST_EVERY
from moonpilot.longcontrol import MOONPILOT_STANDSTILL_SPEED
from moonpilot.jerk import (
  MOONPILOT_BRAKE_EXIT_A,
  MOONPILOT_BRAKE_EXIT_JERK,
  LongitudinalComfortJerkEstimator,
  jerk_limit,
  lead_braking,
  positive_ramp,
)
from moonpilot.lead import LeadAccelEstimator, nearest_lead_in_path, same_car
from moonpilot.pitch import MOONPILOT_PITCH_MIN_SPEED, PitchOffsetEstimator
from moonpilot.radar_latency import RadarLatencyEstimator
from moonpilot.slam import ego_speed_correction
from moonpilot.state import moonpilot_state


class Lead(NamedTuple):
  source: int
  lead: object
  a_lead: float
  a_lead_tau: float
  a_slow: float


class LeadState(NamedTuple):
  source: int
  gap: float
  v_lead: float
  a_lead: float
  sustain: float
  a_slow: float


class PlannerFeatures(NamedTuple):
  handback_ease: bool
  coast_grade: bool
  experimental_mode: bool
  model_braking: bool
  curve_speed: bool
  squeeze: bool
  brake_band: bool
  lead_lateral: bool

LongCtrlState = car.CarControl.Actuators.LongControlState
# From cereal, not from long_mpc — that module imports the compiled acados solver at import time,
# and this file is on plannerd's init path.
LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
Personality = log.LongitudinalPersonality

MOONPILOT_STOP_DISTANCE = 6.0  # m; standstill gap
# s; time gap by personality, keyed on raw enum values.
MOONPILOT_T_FOLLOW = {
  int(Personality.relaxed): 1.75,
  int(Personality.standard): 1.45,
  int(Personality.aggressive): 1.25,
}
MOONPILOT_T_FOLLOW_SLEW_DOWN = 0.5
MOONPILOT_T_FOLLOW_SLEW_UP = 2.0
MOONPILOT_APPROACH_DECEL = 1.0  # m/s^2; follow decel used for the cruise runway cap
MOONPILOT_FOLLOW_BAND_T = 0.3  # s of predicted lead speed past t_follow; inside, the plan holds speed
MOONPILOT_PREDICT_T = 1.0  # s of the lead's slow accel credited to its speed
MOONPILOT_ARRIVE_T = 9.0  # s; an edge this far away at the current closing starts being arrived at...
MOONPILOT_ARRIVE_RAMP_T = 3.0  # s; ...wholly by this much sooner
MOONPILOT_ARRIVE_MIN_T = 1.5  # s; the near edge's arrival never takes less than this much closing time
# At rest the minimum arrival time blends down to `MOONPILOT_ARRIVE_MIN_T_REST`; moving leads keep the standard minimum.
MOONPILOT_ARRIVE_MIN_T_REST = 0.75  # s
MOONPILOT_LEAVE_MIN_T = 2.0  # s; nor the far edge's, a lead pulling away
# The braking match share is scheduled by time headway.
MOONPILOT_TRACK_BRAKE_BP = [1.5, 3.0, 4.0]  # s of headway
MOONPILOT_TRACK_BRAKE_V = [0.85, 0.4, 0.2]
MOONPILOT_TRACK_BRAKE_T = 3.0  # s; cap on how much braking-lead speed loss is matched
MOONPILOT_TRACK_ACCEL = 0.5  # share of positive lead acceleration matched while arriving
MOONPILOT_RECOVER_T = 6.0  # s; outside the band its edge is closed in at the gap error over this...
MOONPILOT_RESPONSE_T = 1.5  # s; ...with the speed error answered over this
# ...except for a car at rest behind a lead that is leaving it: it cannot back up, the gap opens by itself, and the
# recovery would otherwise cancel the far edge's ask for the whole launch. Faded out over these two, so neither the
# jerk limiter nor the brake band sees a step.
MOONPILOT_RECOVER_REST_SPEED = 1.0  # m/s; below it, at `MOONPILOT_LEAVE_DEADBAND` of opening and past
MOONPILOT_RECOVER_OPEN_BAND = 0.3  # m/s of opening past the deadband the fade completes
MOONPILOT_CATCH_UP_CLOSING = 0.5  # m/s; the fastest a lead beyond the band is closed on
MOONPILOT_REACH_LAG_T = 0.5  # s of closing held back from the physical gap a close-in arrival may spend
MOONPILOT_CATCH_UP_SLACK = 2.0  # m past the far edge left alone: a car parked a little long behind a stopped lead stays put
MOONPILOT_LEAVE_DEADBAND = 0.2  # m/s of a lead's opening ignored as its speed noise at rest
# The stopping-floor admission is speed-scheduled and blended over a band below its plateau.
MOONPILOT_FLOOR_ADMISSION_BP = [8.0, 14.0]  # m/s
MOONPILOT_FLOOR_ADMISSION_V = [1.0, 1.6]  # m/s^2
MOONPILOT_FLOOR_ADMISSION_BAND = 0.3  # m/s^2
# The band's near edge at rest, 0.8 m inside STOP_DISTANCE (where the floor anchors at speed). Behind a lead moving at
# creep speed or more it rises to STOP_DISTANCE + MOVING_STOP_RESERVE; the reserve melts linearly as the lead slows.
MOONPILOT_STOP_REST = 5.2  # m
MOONPILOT_MOVING_STOP_RESERVE = 2.0  # m
# Slack holds back the closing the planner cannot observe before the stop.
MOONPILOT_STOP_LAG_T = 0.15  # s
MOONPILOT_MIN_SLACK = 1.0  # m; lower bound on floor slack
# The anchor moves toward `MOONPILOT_STOP_REST` as ego speed falls; above 4 m/s it is
# `MOONPILOT_STOP_DISTANCE`.
MOONPILOT_STOP_TAPER_T = 0.2
# Additional age for radar leads, learned by `moonpilot/radar_latency.py` per car.
MOONPILOT_LEAD_PREVIEW_T = 0.5  # s of a braking lead's accel credited into the floor's lead speed
# The floor credits a braking lead's measured deceleration over this preview.
MOONPILOT_LEAD_BRAKE_SUSTAIN_T = 2.0  # s; minimum braking horizon used by `stopping_decel`
MOONPILOT_LEAD_BRAKE_SUSTAIN_PER_A = 1.0  # s per m/s^2; scales the horizon with lead deceleration
MOONPILOT_A_LEAD_MIN = -10.0  # m/s^2; bounds on a lead's accel estimate, upstream's (long_mpc.process_lead)
MOONPILOT_A_LEAD_MAX = 5.0
MOONPILOT_OUT_OF_PATH_T_FOLLOW = 0.7  # time-gap scale for a lead predicted to leave the path
# Two-state hysteresis applies the out-of-path time-gap scale; see docs/longitudinal.md for measurements.
MOONPILOT_OUT_OF_PATH_ENTER = 0.10
MOONPILOT_OUT_OF_PATH_LEAVE = 0.30
# The measured slow and fast accel ladders are interpolated by speed; see docs/longitudinal.md.
MOONPILOT_LADDER_V_BP = [0.0, 2.5, 7.5, 12.5, 20.0]  # m/s
MOONPILOT_SLOW_ACCEL_V = [0.77, 0.77, 0.67, 0.42, 0.31]  # m/s^2
MOONPILOT_FAST_ACCEL_BP = [0.5, 1.5, 2.5, 3.5, 4.5, 7.0, 9.0, 11.0, 13.5, 17.5]  # m/s
MOONPILOT_FAST_ACCEL_V = [0.83, 1.70, 1.96, 1.97, 1.81, 1.60, 1.47, 1.27, 1.13, 0.87]  # m/s^2
MOONPILOT_FAST_ACCEL_TRACK_BP = [0.0, 6.0]  # m/s
MOONPILOT_FAST_ACCEL_TRACK_V = [0.786, 1.0]  # dimensionless; 1.0 above the last breakpoint
# Coast and brake rungs are measured; details are in docs/longitudinal.md.
MOONPILOT_COASTING_BP = [3.5, 7.5, 15.0]  # m/s
MOONPILOT_COASTING_V = [-0.10, -0.30, -0.33]  # m/s^2
MOONPILOT_BRAKE_LIGHT = -0.75  # m/s^2
MOONPILOT_BRAKE_MEDIUM = -1.25  # m/s^2; also the cruise floor, and forceDecel's
# Speed-error breakpoints map the brake, coast, hold and acceleration rungs.
MOONPILOT_CRUISE_ERR_BP = [-5.0, -3.0, -1.5, 0.0, 1.0, 5.0]  # m/s
# Personality scales the error axis, keyed by raw value like `MOONPILOT_T_FOLLOW`.
MOONPILOT_ERR_BP_SCALE = {
  int(Personality.aggressive): 0.7,
  int(Personality.standard): 1.0,
  int(Personality.relaxed): 1.4,
}
# The crawl band: idle creep and the fast-crawl climb to it (docs/longitudinal.md, "The crawl band").
MOONPILOT_CREEP_SPEED = 1.9  # m/s
MOONPILOT_FAST_CRAWL = 0.31  # m/s^2
# Combined longitudinal and lateral acceleration budget.
MOONPILOT_A_TOTAL_MAX_BP = [4.5, 7.0, 20.0, 40.0]  # m/s
MOONPILOT_A_TOTAL_MAX_V = [2.0, 1.7, 1.7, 3.2]  # m/s^2
MOONPILOT_COAST_BAND = 1.5  # m/s; allowed set-speed drift on a grade, above on a descent and below on a climb
MOONPILOT_COAST_GRADE_MIN = 0.4  # m/s^2; minimum road term (`accel_coast - coast_accel(0.0)`) before this applies
# Only high-grade roads exceed the application threshold.
MOONPILOT_COAST_ACCEL_MAX = 1.0  # m/s^2; bound on the coast command for garbage pitch and the ACCEL_MAX sentinel
# Both bad pose input and the missing-pose sentinel must not turn into an unbounded coast command.
MOONPILOT_COAST_RECOVERY_ACCEL = 0.15  # m/s^2; positive-ask cap just past the climb band edge
# After gas-led control, cap positive acceleration and rebuild the cap gradually.
MOONPILOT_HANDBACK_A = 0.3  # m/s^2
MOONPILOT_HANDBACK_JERK = 0.25  # m/s^3; the cap rebuilds gradually after gas-led control
MOONPILOT_HANDBACK_GAS_T = 0.5  # s
MOONPILOT_ALLOW_THROTTLE_THRESHOLD = 0.4
# Throttle returns above its release threshold after remaining there for the configured interval.
MOONPILOT_ALLOW_THROTTLE_RELEASE = 0.5
MOONPILOT_ALLOW_THROTTLE_RETURN_T = 0.5  # s the probability must stay over the release before throttle returns.
MOONPILOT_MIN_ALLOW_THROTTLE_SPEED = 2.5  # m/s
# Retain the last radar lead as a candidate when radard loses its association at long range; close or
# stopped leads are released immediately.
MOONPILOT_RADAR_HOLD_T = 2.0  # s
MOONPILOT_RADAR_HOLD_MIN_V = 5.0  # m/s of ego speed
MOONPILOT_RADAR_HOLD_MIN_GAP = 40.0  # m
MOONPILOT_MODEL_BRAKE_THRESHOLD = -0.5  # m/s^2; minimum braking ask for admission outside experimental mode
# The admitted slot uses hysteresis: a lead-specific release prevents cycling without ratcheting
# the clear-road cruise candidate.
MOONPILOT_MODEL_STOP_SPEED = 5.0  # m/s
MOONPILOT_MODEL_STOP_RELEASE = 0.1  # m/s^2; `should_stop`'s threshold
# The shallower release applies only with a lead; on a clear road it would hold the model brake
# against the cruise candidate.
MOONPILOT_MODEL_BRAKE_RELEASE = -0.25  # m/s^2; an admitted model gives the slot back above this
MOONPILOT_MODEL_BRAKE_FADE_T = 0.5  # s; an admitted model ask blends in from the other candidates over this
MOONPILOT_MODEL_BRAKE_FULL = -2.0  # m/s^2; an ask this deep goes in whole at once, and shallower ones in proportion
MOONPILOT_MODEL_STOP_ENTER_SPEED = 1.5  # m/s
MOONPILOT_CRASH_DISTANCE = 0.25  # m; FCW contact margin
MOONPILOT_FCW_DECEL = -4.0  # m/s^2 required decel that means it cannot be avoided
MOONPILOT_FCW_COUNT = 2  # frames above threshold before FCW latches
MOONPILOT_FCW_MODEL_PROB = 0.9
# m/s; the standstill deadband `should_stop` is gated on, applied to the ego's own speed and to the
# speed of the lead the policy is actually following. Above it on both counts the plan is asking the car
# to move, so declaring rest is wrong; at or below it on either — a parked car, or a stopped lead — the
# upstream predicate is untouched, which is the parked-behind-a-stopped-lead case it was written for.
MOONPILOT_SHOULD_STOP_SPEED = MOONPILOT_STANDSTILL_SPEED  # the fork's one standstill convention, from longcontrol
MOONPILOT_CONTROL_T_IDX = np.array(ModelConstants.T_IDXS[:CONTROL_N])  # 0 … 2.5 s, 17 points


def coast_accel(pitch: float) -> float:
  # Accel with the throttle closed on this grade; the same fitted shape upstream uses.
  return float(np.sin(pitch) * -5.65 - 0.3)


MOONPILOT_COAST_FLAT_ACCEL = coast_accel(0.0)  # the fit's level-road loss term, kept derived so the
# road's own contribution is computed from the fit rather than duplicating its -0.3 literal.


def cruise_cap(v_ego, e2e, steer_angle_deg, CP, accel_coast, allow_throttle) -> float:
  """The ceiling on a *positive* accel ask, and the budget the closing ask below is bounded by: the
  driver's fast-accel rung `MOONPILOT_FAST_ACCEL_V`, the combined accel budget less the cornering
  demand, and the coast limit when the model expects the driver on the gas. `ACCEL_MAX` in
  experimental mode, where neither the cornering budget nor the coast limit applies."""
  cap = ACCEL_MAX if e2e else float(np.interp(v_ego, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
  if not e2e:
    a_total_max = float(np.interp(v_ego, MOONPILOT_A_TOTAL_MAX_BP, MOONPILOT_A_TOTAL_MAX_V))
    a_y = v_ego**2 * steer_angle_deg * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    cap = min(cap, math.sqrt(max(a_total_max**2 - a_y**2, 0.0)))
    if not allow_throttle:
      coast_limit = float(np.interp(v_ego, [MOONPILOT_MIN_ALLOW_THROTTLE_SPEED, 2 * MOONPILOT_MIN_ALLOW_THROTTLE_SPEED], [cap, max(accel_coast, ACCEL_MIN)]))
      cap = min(cap, coast_limit)
  return cap


def _coast_grade(accel_coast) -> tuple[float, float]:
  """(coast, grade) for this frame: the fitted coast accel for the pitch, and its road term."""
  coast = float(np.clip(accel_coast, -MOONPILOT_COAST_ACCEL_MAX, MOONPILOT_COAST_ACCEL_MAX))
  return coast, coast - MOONPILOT_COAST_FLAT_ACCEL


def _coast_applies(v_ego, v_cruise, e2e, accel_coast, allow_throttle, coast_band) -> bool:
  _, grade = _coast_grade(accel_coast)
  error = v_cruise - v_ego
  return (
    coast_band > 0.0
    and not e2e
    and abs(grade) > MOONPILOT_COAST_GRADE_MIN
    and abs(error) < coast_band
    and error * grade < 0.0
    and (allow_throttle or grade > 0.0)
  )


def _climb_recovery_applies(v_ego, v_cruise, e2e, accel_coast, allow_throttle, coast_band) -> bool:
  """Just past the climb band the plain law would pull back with a full ladder rung; hold the positive
  ask to a gentle constant so recovery from the sag is slow and steady on the same steep grade."""
  _, grade = _coast_grade(accel_coast)
  error = v_cruise - v_ego
  return coast_band > 0.0 and not e2e and grade < -MOONPILOT_COAST_GRADE_MIN and allow_throttle and coast_band <= error < 2.0 * coast_band


def cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle, coast_band: float = 0.0, err_bp=MOONPILOT_CRUISE_ERR_BP) -> float:
  """Return the speed-error cruise candidate, clipped to its normal cap.

  ``err_bp`` is the speed-error breakpoint vector the rungs are interpolated over — the personality
  dial's surface: a scaled copy of ``MOONPILOT_CRUISE_ERR_BP`` (see ``MOONPILOT_ERR_BP_SCALE``)
  moves when each measured rung binds without touching any rung magnitude. The default is the
  standard axis, which is the ladder as measured.

  With ``coast_band`` enabled, only inside the band and only when the grade pushes the car away from
  the set speed, the command is upstream's fitted coast acceleration for that grade, tapered linearly
  to zero at the band edge. A descent may therefore run up to ``v_cruise + coast_band`` and a climb
  may sag to ``v_cruise - coast_band``, settling at the edge where proportional authority resumes; the
  caller's jerk limit makes entry gradual. Below the band on a descent, above it on a climb, on level
  road (below ``MOONPILOT_COAST_GRADE_MIN``), with ``coast_band == 0.0`` (feature off or pose missing),
  or in ``e2e`` mode, the plain speed-error law is unchanged.

  """
  cap = cruise_cap(v_ego, e2e, steer_angle_deg, CP, accel_coast, allow_throttle)
  if v_cruise <= 0.0:
    # forceDecel: the pre-ladder law (1/s on the speed, floored at medium brake), because the ladder's
    # coasting rung would soften driver-monitoring escalation at low speed.
    return min(max(-v_ego, MOONPILOT_BRAKE_MEDIUM), cap)
  rungs = [
    MOONPILOT_BRAKE_MEDIUM,
    MOONPILOT_BRAKE_LIGHT,
    float(np.interp(v_ego, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V)),
    0.0,
    float(np.interp(v_ego, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V)),
    float(np.interp(v_ego, MOONPILOT_FAST_ACCEL_BP, MOONPILOT_FAST_ACCEL_V))
    * float(np.interp(v_ego, MOONPILOT_FAST_ACCEL_TRACK_BP, MOONPILOT_FAST_ACCEL_TRACK_V)),
  ]
  a = min(float(np.interp(v_cruise - v_ego, err_bp, rungs)), cap)
  if _coast_applies(v_ego, v_cruise, e2e, accel_coast, allow_throttle, coast_band):
    coast, _ = _coast_grade(accel_coast)
    # Only where the hill is the thing moving the car — it pushes the car away from the set speed —
    # and only inside the band. A descent needs this even when the model disallows throttle: the
    # existing cap limits positive acceleration but never relaxes braking above the set speed. On a
    # climb with throttle already disallowed, that existing cap is the requested coast behavior.
    a = coast * (coast_band - abs(v_cruise - v_ego)) / coast_band
  elif _climb_recovery_applies(v_ego, v_cruise, e2e, accel_coast, allow_throttle, coast_band):
    a = min(a, MOONPILOT_COAST_RECOVERY_ACCEL)
  return float(a)


def brake_sustain(a_lead) -> float:
  """How long `stopping_decel` believes a braking lead keeps braking: longer for a harder brake."""
  return max(MOONPILOT_LEAD_BRAKE_SUSTAIN_T, -min(float(a_lead), 0.0) * MOONPILOT_LEAD_BRAKE_SUSTAIN_PER_A)


def stopping_decel(v_ego, v_lead, a_lead, slack, sustain=None) -> float:
  """Constant deceleration that preserves `slack` in the relative frame.

  The lead brakes for at most `sustain` (default `brake_sustain(a_lead)`) and never past rest,
  then holds its resulting speed. For an early crossing, while the lead is still braking, use
  `|a_lead| + closing² / (2 * slack)`; otherwise account for the lead's braking-phase travel
  before applying the relative-speed stopping term. `required_decel` uses an unbounded horizon
  for FCW; the planner uses the comfort horizon.
  """
  v_lead = max(float(v_lead), 0.0)
  closing = float(v_ego) - v_lead
  if a_lead < 0.0 and v_lead > 0.0:
    # No `closing > 0` here: a lead braking hard at *matched* speed is the case the prediction
    # exists for, and gating on the present closing rate would read it as no threat until the
    # speed measurement caught up. `required_decel(20, 20, 20, -8)` is -4.47 with the lead
    # predicted to rest and -0.0 with such a gate. The early-crossing test below is false for a
    # non-positive `closing`, so those states take the prediction branch, which handles them.
    if sustain is None:
      sustain = brake_sustain(a_lead)
    t = min(sustain, v_lead / -a_lead)
    if 2.0 * slack <= closing * t:  # the ego matches it before its braking ends
      return -(-a_lead + closing**2 / (2.0 * slack))
    slack += 0.5 * -a_lead * t**2
    v_lead = max(0.0, v_lead + a_lead * t)
  return -(max(float(v_ego) - v_lead, 0.0) ** 2) / (2.0 * slack)


def _arrive(room, closing, min_t, a_track, reach=math.inf) -> float:
  """The accel magnitude that takes `closing` to zero within `room` (a constant decel, spread over at least `min_t` of
  closing time but never past `reach`, the physical room left), plus the share `a_track` of the lead's own accel the ego
  must also match — weighted in as the edge comes within `MOONPILOT_ARRIVE_T` of closing time, whole
  `MOONPILOT_ARRIVE_RAMP_T` sooner. Zero while not closing."""
  if closing <= 0.0:
    return 0.0
  w = min(max((MOONPILOT_ARRIVE_T - max(room, 0.0) / closing) / MOONPILOT_ARRIVE_RAMP_T, 0.0), 1.0)
  return w * (closing * closing / (2.0 * max(room, min(closing * min_t, reach))) + a_track)


def lead_band(v_lead, t_follow, a_slow=0.0):
  v_pred = np.maximum(np.asarray(v_lead, dtype=float) + float(a_slow) * MOONPILOT_PREDICT_T, 0.0)
  moving = np.minimum(v_pred / MOONPILOT_CREEP_SPEED, 1.0)
  rest = MOONPILOT_STOP_REST + (MOONPILOT_STOP_DISTANCE + MOONPILOT_MOVING_STOP_RESERVE - MOONPILOT_STOP_REST) * moving
  near = np.maximum(rest, float(t_follow) * v_pred)
  return near, near + MOONPILOT_FOLLOW_BAND_T * v_pred


def recover_share(v_ego, closing) -> float:
  """The near-edge recovery's share: full unless the car is at rest with the lead leaving it — `1 - rest * open` over
  `MOONPILOT_RECOVER_REST_SPEED` and `MOONPILOT_RECOVER_OPEN_BAND` past `MOONPILOT_LEAVE_DEADBAND`, both clamped, so
  it reaches 0 without stepping."""
  rest = float(np.clip(1.0 - v_ego / MOONPILOT_RECOVER_REST_SPEED, 0.0, 1.0))
  opening = float(np.clip((-closing - MOONPILOT_LEAVE_DEADBAND) / MOONPILOT_RECOVER_OPEN_BAND, 0.0, 1.0))
  return 1.0 - rest * opening


def lead_accel(v_ego, gap, v_lead, a_lead, t_follow, sustain=None, a_slow=0.0) -> float:
  """The lead law: where to be, by when, from where the lead is going — then the stopping floor as the hard bound.

  Where: a band behind the lead's predicted speed (`v_pred`: its speed plus `MOONPILOT_PREDICT_T` of its slow accel),
  from `t_follow` to `MOONPILOT_FOLLOW_BAND_T` beyond, never nearer than the rest gap plus a moving-lead reserve. Inside
  it the plan holds speed: radar noise there moves nothing, and a lead's wander inside it is not chased.

  When: once the near edge is `MOONPILOT_ARRIVE_T` away at the current closing, the constant decel that arrives there
  speed-matched, plus the headway-scheduled `MOONPILOT_TRACK_BRAKE_V` share of a slowing lead's slow accel — no more than would shed its predicted
  speed in `MOONPILOT_TRACK_BRAKE_T`, so a lead about to stop is not braked for twice; once the far edge is that close at the
  current opening, the constant accel that matches the lead before it, plus `MOONPILOT_TRACK_ACCEL` of an accelerating
  lead's. Both are kinematic predictions rather than reactions, and continuous through zero closing.

  Outside the band the edge is recovered gently: inside the near edge over `MOONPILOT_RECOVER_T` and
  `MOONPILOT_RESPONSE_T`; beyond the far edge, closing at no more than `MOONPILOT_CATCH_UP_CLOSING`. The near-edge
  recovery stands down (`recover_share`) for a car at rest behind a lead that is leaving it: it cannot back up, and
  the recovery would otherwise cancel the far edge's ask for the whole launch.

  The stopping floor is the kinematic term kept as a bound, in the frame the gap actually closes in: the constant decel
  that keeps the slack past a rest anchor, less `MOONPILOT_STOP_LAG_T` of closing, against the lead's raw,
  sustain-bounded braking (`stopping_decel`). It is admitted past the speed-scheduled `MOONPILOT_FLOOR_ADMISSION_*`
  plateau and blended in over `MOONPILOT_FLOOR_ADMISSION_BAND` below it, so it only ever adds braking.
  """
  gap, v_lead, a_lead, a_slow = max(float(gap), 0.0), max(float(v_lead), 0.0), float(a_lead), float(a_slow)
  v_pred = max(v_lead + a_slow * MOONPILOT_PREDICT_T, 0.0)
  moving = min(v_pred / MOONPILOT_CREEP_SPEED, 1.0)
  near, far = lead_band(v_lead, t_follow, a_slow)
  closing = v_ego - v_pred
  # An opening slower than the lead-speed noise at rest is not the lead leaving: radar jitter there reads 0.0/0.2.
  a = _arrive(far - gap, max(-closing - MOONPILOT_LEAVE_DEADBAND, 0.0), MOONPILOT_LEAVE_MIN_T, MOONPILOT_TRACK_ACCEL * max(a_slow, 0.0))
  reach = max(gap - MOONPILOT_CRASH_DISTANCE - MOONPILOT_REACH_LAG_T * max(closing, 0.0), 0.05)
  share = float(np.interp(gap / max(v_ego, 1.0), MOONPILOT_TRACK_BRAKE_BP, MOONPILOT_TRACK_BRAKE_V))
  a_track = share * min(max(-a_slow, 0.0), v_pred / MOONPILOT_TRACK_BRAKE_T)
  a -= _arrive(gap - near, closing, MOONPILOT_ARRIVE_MIN_T_REST + (MOONPILOT_ARRIVE_MIN_T - MOONPILOT_ARRIVE_MIN_T_REST) * moving, a_track, reach)
  a += recover_share(v_ego, closing) * min(gap - near, 0.0) / (MOONPILOT_RECOVER_T * MOONPILOT_RESPONSE_T)
  catch_up = min(max(gap - far - MOONPILOT_CATCH_UP_SLACK, 0.0) / MOONPILOT_RECOVER_T, MOONPILOT_CATCH_UP_CLOSING)
  a += max(catch_up - max(closing, 0.0), 0.0) / MOONPILOT_RESPONSE_T
  # The floor reads the raw estimate, its braking credited over MOONPILOT_LEAD_PREVIEW_T: a real brake, without lag.
  v_lead_eff = max(0.0, v_lead + min(a_lead, 0.0) * MOONPILOT_LEAD_PREVIEW_T)
  anchor = MOONPILOT_STOP_REST + min(MOONPILOT_STOP_TAPER_T * v_ego, MOONPILOT_STOP_DISTANCE - MOONPILOT_STOP_REST)
  slack = gap - anchor - MOONPILOT_STOP_LAG_T * max(v_ego - v_lead_eff, 0.0)
  a_stop = stopping_decel(v_ego, v_lead_eff, a_lead, max(slack, MOONPILOT_MIN_SLACK), sustain)
  admission = float(np.interp(v_ego, MOONPILOT_FLOOR_ADMISSION_BP, MOONPILOT_FLOOR_ADMISSION_V))
  return a + min(max((-a_stop - admission) / MOONPILOT_FLOOR_ADMISSION_BAND + 1.0, 0.0), 1.0) * min(a_stop - a, 0.0)




def lead_accel_estimate(a_lead: float) -> float:
  """A lead's accel estimate, bounded the way upstream's MPC bounds it: radard's Kalman reports
  implausible values on a track that just appeared, and the maneuver plant differentiates a stepped
  lead speed into hundreds of m/s^2."""
  return min(max(float(a_lead), MOONPILOT_A_LEAD_MIN), MOONPILOT_A_LEAD_MAX)


def lead_state_at(lead, t, x_ego, a_lead, a_lead_tau, t_decay=None) -> tuple[float, float, float]:
  """Project a lead from its current gap, speed and estimated acceleration.

  Acceleration decays using radard's model, and travel integrates the clamped speed so a lead that
  stops during the horizon travels only its stopping distance.
  """
  a_lead = lead_accel_estimate(a_lead)
  a_traj = a_lead * math.exp(-a_lead_tau * (t if t_decay is None else t_decay) ** 2 / 2.0)
  a_avg = 0.5 * (a_lead + a_traj)
  v_lead = max(0.0, float(lead.vLead))
  v_end = v_lead + a_avg * t
  if v_end > 0.0:
    x_lead = 0.5 * (v_lead + v_end) * t
  else:  # stopped inside t: its own stopping distance at that average decel
    x_lead, v_end = (v_lead**2 / (-2.0 * a_avg) if a_avg < 0.0 else 0.0), 0.0
  return max(0.0, lead.dRel + x_lead - x_ego), v_end, a_traj


def required_decel(v_ego, gap, v_lead, a_lead=0.0) -> float:
  """FCW deceleration against the contact margin, using an unbounded lead-braking horizon."""
  return stopping_decel(v_ego, max(float(v_lead), 0.0), lead_accel_estimate(a_lead), max(gap - MOONPILOT_CRASH_DISTANCE, 0.1), sustain=math.inf)


def model_candidate(model, e2e, allowed, threshold=MOONPILOT_MODEL_BRAKE_THRESHOLD) -> float | None:
  """Return the model acceleration when admitted by mode and braking threshold.

  Experimental mode passes the raw ask. Otherwise the candidate is braking-only, admitted through
  a deadband and blended over `MOONPILOT_MODEL_BRAKE_FADE_T`; `ACCEL_MIN` remains the lower bound.

  No fork-owned floor: it would have bound 2.23 min in 61 of 494 episodes, withholding braking the
  driver had already begun.
  """
  a = float(model.action.desiredAcceleration)
  if not e2e and (not allowed or not a < threshold):
    return None
  return a


def model_release(v_ego, lead_present: bool = False) -> float:
  """The ask an admitted model has to rise past to give its slot back: `should_stop`'s 0.1 in the crawl band, then
  `MOONPILOT_MODEL_BRAKE_RELEASE` behind a lead and the entry deadband on a clear road.

  The separation is only where the evidence is. Behind a lead the equal thresholds let an ask dithering on the entry
  cycle the slot every second or so (measured on six engaged drives: 24 runs, 13 under 2 s, 8 re-admitted within 2 s;
  18, 6 and 1 with the separation), and the lead law governs the command anyway. On a clear road the same
  separation would be a one-way ratchet against the cruise term — a model that once crossed the entry and then settled
  at -0.3 would hold the car below the set speed for as long as it stayed there, which is exactly what the deadband
  exists to prevent; unguarded, the band is held 2.3 s on a clear road against 1.1 s gated. Reading `lead_present` raw
  adds a release path, a lead dropout flipping the threshold back: measured, 0 of 12 slot releases on three drives
  happened with the ask in the band and no lead, so it needs no latch."""
  held = MOONPILOT_MODEL_BRAKE_RELEASE if lead_present else MOONPILOT_MODEL_BRAKE_THRESHOLD
  return float(np.interp(v_ego, [MOONPILOT_CREEP_SPEED, MOONPILOT_MODEL_STOP_SPEED], [MOONPILOT_MODEL_STOP_RELEASE, held]))


def policy(
  v_ego,
  leads,
  v_cruise,
  t_follow,
  e2e,
  model_accel,
  steer_angle_deg,
  CP,
  accel_coast,
  allow_throttle,
  curve=None,
  x_ego=0.0,
  v_hold=math.inf,
  coast_band: float = 0.0,
  squeeze=ACCEL_MAX,
  exit_cap=ACCEL_MAX,
  err_bp=MOONPILOT_CRUISE_ERR_BP,
  model_weight=1.0,
):
  """Return the minimum policy candidate and its source.

  Each lead is `(source, gap, v_lead, a_lead[, sustain])`. Curve and corridor terms are minima in
  the cruise slot; `model_weight` blends an admitted model candidate from the other candidates'
  minimum toward its raw ask.
  """
  a_curve = min(curve_accel(v_ego, x_ego, curve), lat_accel_hold(v_ego, v_hold), squeeze, exit_cap)
  a_cruise_raw = cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle, coast_band, err_bp)
  a_cruise = min(a_cruise_raw, a_curve)
  slow = float(np.interp(v_ego, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V))
  if a_cruise > slow and v_cruise > v_ego:
    a_open = a_cruise
    for _, gap, v_lead, a_lead, *_ in leads:
      if v_lead >= v_cruise or a_lead >= MOONPILOT_FAST_CRAWL:
        continue
      target = max(MOONPILOT_STOP_DISTANCE, t_follow * max(v_ego, v_lead))
      runway = gap - target
      closing_at_set = v_cruise - v_lead
      closing_now = v_ego - v_lead
      # Room to reach set speed, then match the lead at the ordinary follow decel. One extra
      # follow gap fades the fast rung in; with less room, asking for it only buys a later brake.
      # ponytail: assume constant lead speed; use its trajectory if this caps real pull-aways.
      needed = (closing_at_set**2 - closing_now**2) / (2.0 * a_open) + closing_at_set**2 / (2.0 * MOONPILOT_APPROACH_DECEL)
      if needed > 0.0:
        a_cruise = min(a_cruise, float(np.interp(runway, [needed, needed + target], [slow, a_open])))
  candidates = [(a_cruise, LongitudinalPlanSource.cruise)]
  candidates += [(lead_accel(v_ego, gap, v_lead, a_lead, t_follow, *extra), source) for source, gap, v_lead, a_lead, *extra in leads]
  if model_accel is not None:
    # Blended from where the other candidates are, so admission is not a step (see `model_weight`).
    rest = min(c[0] for c in candidates)
    model = float(model_accel)
    if model < rest:
      model = rest + model_weight * (model - rest)
    candidates.append((model, LongitudinalPlanSource.e2e))
  accel, source = min(candidates, key=lambda c: c[0])
  return float(accel), source


class MoonpilotLongitudinalPlanner:
  """Same public surface as upstream's planner — `output_a_target`, `output_should_stop`, `fcw`,
  `publish` — because the maneuver harness and controlsd read those names, not the type."""

  def __init__(self, CP, init_v=0.0, init_a=0.0, dt=DT_MDL, params=None, clock=time):
    self.CP = CP
    self.dt = dt
    self.params = Params() if params is None else params
    self.clock = clock
    self.long_lag = LongLagEstimator(CP, dt)
    self.long_lag.restore(self.params)
    self.long_lag_logged = self.long_lag.applied_delay()
    self.long_jerk = LongitudinalComfortJerkEstimator(dt)
    self.long_jerk.restore(self.params)

    self.action_t = self.long_lag.applied_delay() + dt
    self.VM = VehicleModel(CP)
    self.lat_bias = LatAccelBiasEstimator(dt)
    self.lat_bias.restore(self.params)
    self.pitch_offset = PitchOffsetEstimator(dt)
    self.pitch_offset.restore(self.params)
    self.frames = 0
    self.lead_accel = (LeadAccelEstimator(dt), LeadAccelEstimator(dt))
    self.radar_latency = RadarLatencyEstimator()
    self.radar_latency.restore(self.params)
    # What persists, in this order: each one writes on its own evidence rule, so a drive that earned a block or a
    # handful of samples hands them on instead of restarting, and an estimator with nothing to say writes nothing.
    self.learned = (self.long_lag, self.long_jerk, self.lat_bias, self.pitch_offset, self.radar_latency)
    # Pose-stitched free-space history for squeeze: empty until a valid egoPose arrives, cleared
    # whenever the pose goes invalid so a rebase cannot mix two origins.
    self.corridor = RollingCorridor()

    self.output_a_target = init_a
    self.output_should_stop = False
    self.fcw = False
    self.crash_cnt = 0
    self.allow_throttle = True
    self.throttle_return_t = 0.0  # seconds the gas-press probability has held over the release while coasting
    self.radar_hold = None  # (last radar leadOne snapshot, a_lead, a_lead_tau, seconds held, ego travel since)
    self.source = LongitudinalPlanSource.cruise
    self.model_braking = False  # the model held a braking slot last frame: `model_release` applies
    self.model_fade = 0.0  # seconds-scaled weight an admitted model ask has earned (`MOONPILOT_MODEL_BRAKE_FADE_T`)
    self.t_follow = None
    self.lead_leaving = False
    # The curve exit ramp's ceiling on the positive ask: zero while a curve term brakes, rebuilt at
    # `MOONPILOT_CURVE_EXIT_JERK` after it lets go, `ACCEL_MAX` (inert) once rebuilt or disengaged.
    self.curve_exit_cap = ACCEL_MAX
    # The hand-back cap: `MOONPILOT_HANDBACK_A` on the first frame after a gas-led reset, rebuilt at `MOONPILOT_HANDBACK_JERK`;
    # `ACCEL_MAX` (inert) otherwise. `gas_recent` is how long a gas press during the reset still counts.
    self.handback_cap = ACCEL_MAX
    self.gas_recent = 0.0
    self.was_reset = True
    self.brake_band = BrakeBand()
    self.solve_time = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.v_desired_trajectory[:] = init_v
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def _lead_states(self, leads, t, x_ego, v_ego, lead_age, decay_t):
    states = []
    for item in leads:
      age = self.lead_age(item.lead, lead_age)
      gap, v_lead, a_lead = lead_state_at(item.lead, t + age, x_ego + v_ego * age, item.a_lead, item.a_lead_tau, decay_t)
      states.append(LeadState(item.source, gap, v_lead, a_lead, brake_sustain(item.a_lead), item.a_slow))
    return states

  def update(self, sm):
    features = PlannerFeatures(
      enabled(HANDBACK_EASE, self.params),
      enabled(COAST_GRADE, self.params),
      enabled(EXPERIMENTAL_MODE, self.params),
      enabled(MODEL_BRAKING, self.params),
      enabled(CURVE_SPEED, self.params),
      enabled(SQUEEZE, self.params),
      enabled(BRAKE_BAND, self.params),
      enabled(LEAD_LATERAL, self.params),
    )
    start = self.clock.monotonic()

    CS = sm['carState']
    v_ego = max(CS.vEgo, 0.0)
    # Apply the rolling-window ego correction at the source so every term below uses it.
    # See `slam.ego_speed_correction` for its gates and pass-through behavior.
    v_ego = max(0.0, v_ego + ego_speed_correction(sm, v_ego))
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
      self.model_braking = False
      self.model_fade = 0.0
      self.curve_exit_cap = ACCEL_MAX
      self.handback_cap = ACCEL_MAX
      self.gas_recent = MOONPILOT_HANDBACK_GAS_T if CS.gasPressed else max(self.gas_recent - self.dt, 0.0)
      self.brake_band.reset()
    elif self.was_reset:
      if self.gas_recent > 0.0 and features.handback_ease:
        self.handback_cap = MOONPILOT_HANDBACK_A
      self.gas_recent = 0.0
    else:
      self.handback_cap = min(self.handback_cap + MOONPILOT_HANDBACK_JERK * self.dt, ACCEL_MAX)
    self.was_reset = reset_state

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
    self.long_jerk.update(self.output_a_target, a_ego, self.long_lag.applied_delay(), lag_valid)
    jerk_scale = self.long_jerk.applied()

    self.action_t = self.long_lag.applied_delay() + self.dt

    pose_valid = len(sm['carControl'].orientationNED) == 3
    pitch = float(sm['carControl'].orientationNED[1]) if pose_valid else 0.0
    # Learn the device-and-car pitch bias while driving; apply it to each grade term.
    self.pitch_offset.update(pitch, pose_valid and CS.vEgo > MOONPILOT_PITCH_MIN_SPEED and not CS.standstill)
    accel_coast = coast_accel(pitch - self.pitch_offset.applied()) if pose_valid else ACCEL_MAX
    coast_band = MOONPILOT_COAST_BAND if pose_valid and features.coast_grade else 0.0
    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    if self.allow_throttle:
      self.throttle_return_t = 0.0
      self.allow_throttle = throttle_prob > MOONPILOT_ALLOW_THROTTLE_THRESHOLD
    else:
      self.throttle_return_t, self.allow_throttle = hold_timer(
        self.throttle_return_t,
        throttle_prob > MOONPILOT_ALLOW_THROTTLE_RELEASE,
        self.dt,
        MOONPILOT_ALLOW_THROTTLE_RETURN_T,
      )
    self.allow_throttle = self.allow_throttle or v_ego <= MOONPILOT_MIN_ALLOW_THROTTLE_SPEED

    # The radar lead's own sensor latency, learned from near-rest frames: one call per radarState slot, ticked
    # every frame so an absent or replaced lead resets its window.
    self.radar_latency.update(sm['radarState'].leadOne, a_ego)
    self.radar_latency.update(sm['radarState'].leadTwo, a_ego)

    steer_angle = CS.steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg
    t_follow = self.time_gap(sm, features.lead_lateral)
    err_bp = self._err_bp(sm)
    e2e = features.experimental_mode
    # The model's stop hold yields to a departing leadOne once the model asks to go.
    one = sm['radarState'].leadOne
    go = one.present and one.vLead > MOONPILOT_SHOULD_STOP_SPEED and sm['modelV2'].action.desiredAcceleration > 0.0
    threshold = model_release(v_ego, one.present) if self.model_braking and not go else MOONPILOT_MODEL_BRAKE_THRESHOLD
    model_accel = model_candidate(sm['modelV2'], e2e, features.model_braking, threshold)
    self.model_braking = model_accel is not None and not e2e
    self.model_fade = min(self.model_fade + self.dt / MOONPILOT_MODEL_BRAKE_FADE_T, 1.0) if self.model_braking else 0.0
    model_weight = 1.0
    if model_accel is not None and not e2e:
      # A deep ask is not held back by the fade: proportionally more of it goes in at once.
      model_weight = max(self.model_fade, float(np.interp(model_accel, [MOONPILOT_MODEL_BRAKE_FULL, threshold], [1.0, 0.0])))

    # Use raw `CS.vEgo`: this measures the car, not the corrected plan input.
    vp = sm['vehicleParameters']
    self.VM.update_params(max(vp.stiffnessFactor, 0.1), max(vp.steerRatio, 0.1))
    measured_curvature = -self.VM.calc_curvature(math.radians(CS.steeringAngleDeg - vp.angleOffsetDeg), CS.vEgo, vp.roll)
    curve_allowed = features.curve_speed
    # Only learn when lateral control, not the driver, owns the measured steering.
    lateral_state = getattr(sm['controlsState'], "lateralControlState", None)
    tracking_valid = True
    if lateral_state is not None and lateral_state.which() == "torqueState":
      torque_state = lateral_state.torqueState
      if getattr(torque_state, "version", 0):
        tracking_valid = (
          torque_state.active
          and not torque_state.saturated
          and abs(torque_state.actualLateralAccel - torque_state.desiredLateralAccel) <= MOONPILOT_CURVE_BIAS_TRACKING_TOLERANCE
        )
    bias_valid = (
      curve_allowed
      and vp.valid
      and sm['carControl'].latActive
      and not CS.steeringPressed
      and not CS.standstill
      and CS.vEgo > MOONPILOT_CURVE_BIAS_MIN_SPEED
      and tracking_valid
    )
    self.lat_bias.update(predicted_lat_accel(sm['modelV2']), measured_curvature * CS.vEgo**2, bias_valid)
    scale = self.lat_bias.applied()
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
        leads.append(Lead(source, lead, a_lead, a_lead_tau, estimator.a_slow))
    # FCW keeps both slots: dedupe only the planner candidates, not the warning inputs.
    fcw_leads = list(leads)
    one, two = (lead for _, lead, *_ in leads) if len(leads) == 2 else (None, None)
    if one is not None and one.radar and two.radar and same_car(one, two):
      # radard matches both model leads to the nearest radar track, so a stopped car often fills both
      # slots, on two tracks of its own. One vehicle is one candidate: two noisy readings of it under
      # `min` hand the winning slot back and forth and the command pulses (route 00000400 seg 4).
      leads.pop()
    leads += self._radar_hold(sm['radarState'].leadOne, leads, v_ego)

    # Predict each input to the command's application time; compensate the radar message's age separately.
    # Missing timestamps leave the radar-age correction at zero.
    log_mono = getattr(sm, 'logMonoTime', None) or {}
    model_mono = log_mono.get('modelV2')
    radar_mono = log_mono.get('radarState')
    lead_age = max(0.0, (model_mono - radar_mono) / 1e9) if model_mono and radar_mono else 0.0

    # Re-reference the model path to this tick; the path-age bounds are in `curve.py`.
    path_age = 0.0
    capture_time = sm['modelV2'].timestampEof * 1e-9
    if math.isfinite(capture_time) and capture_time > 0.0:
      recv_time = getattr(sm, 'recv_time', None) or {}
      for stamp in (recv_time.get('modelV2'), (model_mono / 1e9) if model_mono else None):
        if stamp is None or not math.isfinite(stamp):
          continue
        age = stamp - capture_time
        if not 0.0 < age <= MOONPILOT_CURVE_PATH_CLOCK_SANITY:
          continue  # not this source's clock: the stamp-only read is the fallback
        path_age = age if age <= MOONPILOT_CURVE_PATH_MAX_AGE else 0.0
        break
    x_path = v_ego * path_age
    curve = curve_targets(sm['modelV2'], curve_allowed, vp.roll, scale, ahead=x_path)
    # Pose history bridges edge dropouts; without a valid pose, use current road edges alone.
    squeeze = ACCEL_MAX
    if features.squeeze:
      edges = sm['modelV2'].roadEdges
      if len(edges) >= 2:
        pose = read_ego_pose(moonpilot_state(sm), self.clock.monotonic_ns())
        if pose is not None:
          now = self.clock.monotonic()
          self.corridor.push(now, pose, edges[0], edges[1])
          squeeze = self.corridor.squeeze_accel(pose, ModelConstants.X_IDXS, x_ego=x_path)
        else:
          self.corridor.clear()
          squeeze = squeeze_accel(edges[0], edges[1], ModelConstants.X_IDXS, x_ego=x_path)

    a_prev = float(self.output_a_target)
    if CS.standstill:
      # Parked, the stop hold owns the brake and a negative target is a command nothing tracks: ramping
      # a launch up out of it spent 0.36 s from the -0.44 a stop inside the standstill gap leaves. The
      # comfort first step (`jerk_limit`) still keeps one frame of a lead's speed from lifting the hold.
      a_prev = max(a_prev, 0.0)
    v_pred = max(0.0, v_ego + a_prev * self.action_t)
    x_pred = 0.5 * (v_ego + v_pred) * self.action_t
    # The exit ramp: while a curve term brakes or the plan brakes past MOONPILOT_BRAKE_EXIT_A, the positive
    # ask is held where the command is; after release it rebuilds at MOONPILOT_CURVE_EXIT_JERK while still
    # cornering or with a curve ahead, else at MOONPILOT_BRAKE_EXIT_JERK — the lead-free positive ramp's own pace.
    exit_jerk = MOONPILOT_CURVE_EXIT_JERK
    if min(curve_accel(v_pred, x_pred, curve), lat_accel_hold(v_pred, v_hold), squeeze) < 0.0 or a_prev < MOONPILOT_BRAKE_EXIT_A:
      self.curve_exit_cap = max(a_prev, 0.0)
    else:
      cornering = abs(measured_curvature) * CS.vEgo**2 >= MOONPILOT_CURVE_EXIT_LAT_ACCEL
      curve_ahead = curve_accel(max(v_cruise, v_pred), x_pred, curve) < 0.0
      if not (cornering or curve_ahead):
        exit_jerk = MOONPILOT_BRAKE_EXIT_JERK
      self.curve_exit_cap = min(self.curve_exit_cap + exit_jerk * self.dt, ACCEL_MAX)
    lead_states = self._lead_states(leads, self.action_t, x_pred, v_ego, lead_age, self.action_t)
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
      coast_band=coast_band,
      curve=curve,
      x_ego=x_pred,
      v_hold=v_hold,
      squeeze=squeeze,
      exit_cap=self.curve_exit_cap,
      err_bp=err_bp,
      model_weight=model_weight,
    )
    # The hand-back ceiling bounds the policy's result, not a candidate: an admitted model ask is blended from the other
    # candidates' minimum, so lowering one of them would deepen a partly faded model brake. A ceiling of at least
    # MOONPILOT_HANDBACK_A on the result leaves every nonpositive ask exactly as the policy gave it.
    a_cmd = min(a_cmd, self.handback_cap)

    braking_lead = lead_braking(lead_states)
    # The brake band shapes what the car is asked for, after the policy picked it and before the comfort ramp.
    # It never lifts the ask over `cruise_cap`, which the policy's output already sits under through the cruise
    # slot's `min`: with throttle denied that is the coast accel, and nothing downstream enforces `allowThrottle`
    # — only `aTarget` reaches the car, so a raised ask there would be held on throttle while the model expects
    # the driver off the gas.
    if not reset_state and features.brake_band and not e2e:
      engine = engine_floor(self.CP, v_ego)
      shaped = self.brake_band.update(a_cmd, v_ego, self.dt, urgent=braking_lead, floor=engine if engine is not None else MOONPILOT_BRAKE_BAND_FLOOR)
      a_cmd = min(shaped, cruise_cap(v_pred, e2e, steer_angle, self.CP, accel_coast, self.allow_throttle))
    else:
      self.brake_band.reset()
    braking = braking_lead
    a_target = float(np.clip(jerk_limit(a_cmd, a_prev, self.dt, v_ego, jerk_scale, braking, positive_ramp(lead_states, v_ego)), ACCEL_MIN, ACCEL_MAX))

    if not math.isfinite(a_target):
      # One guard, so that no input can latch a NaN into the command.
      a_target = 0.0

    self.v_desired_trajectory, self.a_desired_trajectory = self._trajectory(
      v_ego,
      a_target,
      leads,
      v_cruise,
      t_follow,
      e2e,
      model_accel,
      steer_angle,
      accel_coast,
      lead_age,
      curve,
      v_hold,
      jerk_scale,
      coast_band=coast_band,
      squeeze=squeeze,
      exit_cap=self.curve_exit_cap,
      exit_jerk=exit_jerk,
      handback_cap=self.handback_cap,
      err_bp=err_bp,
      model_weight=model_weight,
    )

    self.j_desired_trajectory = np.gradient(self.a_desired_trajectory, MOONPILOT_CONTROL_T_IDX)

    crash = any(
      lead.modelProb > MOONPILOT_FCW_MODEL_PROB and required_decel(v_ego, lead.dRel, lead.vLead, a_lead) < MOONPILOT_FCW_DECEL
      for _, lead, a_lead, *_ in fcw_leads
    )
    self.crash_cnt = self.crash_cnt + 1 if crash else 0
    fcw = self.crash_cnt > MOONPILOT_FCW_COUNT and not CS.standstill
    if fcw and not self.fcw:
      cloudlog.info("moonpilot FCW triggered")
    self.fcw = fcw

    # A moving lead can keep the policy at near-zero acceleration below upstream's stop threshold.
    # Suppress should-stop only while the winning candidate is a moving lead, so stopping does not
    # fight follow motion; unrelated leads cannot release a stop behind the lead being followed.
    # The model's own shouldStop remains experimental-only; modeld computes it with `should_stop`
    # at `modeld.py:68`.
    trailing = v_ego > MOONPILOT_SHOULD_STOP_SPEED and any(s == source and v_lead > MOONPILOT_SHOULD_STOP_SPEED for s, _, v_lead, *_ in lead_states)
    model_stop = not e2e and self.model_braking and not leads and v_ego <= MOONPILOT_MODEL_STOP_ENTER_SPEED and a_target < 0.0
    self.output_should_stop = (should_stop(v_ego, a_target) and not trailing) or (e2e and sm['modelV2'].action.shouldStop) or model_stop
    self.output_a_target = a_target
    self.source = source
    # Persist learned values and their evidence every `PERSIST_EVERY` frames.
    self.frames += 1
    if self.frames % PERSIST_EVERY == 0:
      for estimator in self.learned:
        estimator.persist(self.params)
    self.solve_time = self.clock.monotonic() - start

  def _trajectory(
    self,
    v_ego,
    a_target,
    leads,
    v_cruise,
    t_follow,
    e2e,
    model_accel,
    steer_angle,
    accel_coast,
    lead_age=0.0,
    curve=None,
    v_hold=math.inf,
    comfort_scale=1.0,
    coast_band: float = 0.0,
    squeeze=ACCEL_MAX,
    exit_cap=ACCEL_MAX,
    exit_jerk=MOONPILOT_CURVE_EXIT_JERK,
    err_bp=MOONPILOT_CRUISE_ERR_BP,
    model_weight=1.0,
    handback_cap=ACCEL_MAX,
  ):
    """Roll the command's policy forward over the published horizon, accumulating ego travel in `x`.

    The same lead age and curve, corridor, and handback constraints used for the command are carried
    through the rollout.
    """
    speeds = np.zeros(CONTROL_N)
    accels = np.zeros(CONTROL_N)
    v, a, x, t_prev = v_ego, a_target, 0.0, 0.0
    for i, t_idx in enumerate(MOONPILOT_CONTROL_T_IDX):
      t = float(t_idx)
      states = self._lead_states(leads, t, x, v_ego, lead_age, t)
      a_cmd, _ = policy(
        v,
        states,
        v_cruise,
        t_follow,
        e2e,
        model_accel,
        steer_angle,
        self.CP,
        accel_coast,
        self.allow_throttle,
        coast_band=coast_band,
        curve=curve,
        x_ego=x,
        v_hold=v_hold,
        squeeze=squeeze,
        exit_cap=min(exit_cap + exit_jerk * t, ACCEL_MAX),
        err_bp=err_bp,
        model_weight=max(model_weight, min(self.model_fade + t / MOONPILOT_MODEL_BRAKE_FADE_T, 1.0)),
      )
      a_cmd = min(a_cmd, handback_cap + MOONPILOT_HANDBACK_JERK * t)  # the hand-back ceiling, rebuilt over the horizon as in `update`
      a = float(np.clip(jerk_limit(a_cmd, a, t - t_prev, v, comfort_scale, lead_braking(states), positive_ramp(states, v)), ACCEL_MIN, ACCEL_MAX))
      speeds[i] = v
      accels[i] = a
      dt = float(MOONPILOT_CONTROL_T_IDX[i + 1]) - t if i + 1 < CONTROL_N else 0.0
      v_next = max(0.0, v + a * dt)
      x += 0.5 * (v + v_next) * dt
      v, t_prev = v_next, t
    return speeds, accels

  def _radar_hold(self, lead, leads, v_ego):
    """The last radar leadOne, projected forward, as an extra lead0 candidate while radard has handed
    the slot to a vision-only lead or to nothing (MOONPILOT_RADAR_HOLD_T). The current lead stays in
    `leads` beside it, so `min` takes the lower command."""
    if lead.present and lead.radar:
      a_lead, a_lead_tau = next((a, tau) for s, _, a, tau, _ in leads if s == LongitudinalPlanSource.lead0)
      snapshot = SimpleNamespace(dRel=float(lead.dRel), vLead=float(lead.vLead), radar=True)
      self.radar_hold = (snapshot, a_lead, a_lead_tau, 0.0, 0.0)
      return []
    if self.radar_hold is None:
      return []
    snapshot, a_lead, a_lead_tau, held, travel = self.radar_hold
    held += self.dt
    travel += v_ego * self.dt
    gap, v_lead, a_now = lead_state_at(snapshot, held, travel, a_lead, a_lead_tau)
    if held > MOONPILOT_RADAR_HOLD_T or v_ego <= MOONPILOT_RADAR_HOLD_MIN_V or gap <= MOONPILOT_RADAR_HOLD_MIN_GAP:
      self.radar_hold = None
      return []
    self.radar_hold = (snapshot, a_lead, a_lead_tau, held, travel)
    # A held track is a fallback for braking, not something to answer early: no slow accel.
    return [Lead(LongitudinalPlanSource.lead0, SimpleNamespace(dRel=gap, vLead=v_lead, radar=True), a_now, a_lead_tau, 0.0)]

  def lead_age(self, lead, lead_age):
    return lead_age + (self.radar_latency.applied() if lead.radar else 0.0)

  def time_gap(self, sm, lateral=None):
    personality = sm['selfdriveState'].personality.raw
    base = MOONPILOT_T_FOLLOW.get(personality, MOONPILOT_T_FOLLOW[int(Personality.standard)])
    lateral = enabled(LEAD_LATERAL, self.params) if lateral is None else lateral
    target = base
    if lateral:
      # Two-state, hysteretic: see MOONPILOT_OUT_OF_PATH_ENTER. The latch is the whole point — a
      # continuous map put a moving target under the regulator, and inPath rises instantly.
      in_path = nearest_lead_in_path(sm)
      leaving = in_path < (MOONPILOT_OUT_OF_PATH_LEAVE if self.lead_leaving else MOONPILOT_OUT_OF_PATH_ENTER)
      self.lead_leaving = leaving
      if leaving:
        target *= MOONPILOT_OUT_OF_PATH_T_FOLLOW

    lead = sm['radarState'].leadOne
    if not lead.present:
      self.t_follow = base
      self.lead_leaving = False
      return base
    if self.t_follow is None:
      self.t_follow = target

    if not lateral:
      self.t_follow = base
      return base
    if target < self.t_follow:
      self.t_follow = max(target, self.t_follow - MOONPILOT_T_FOLLOW_SLEW_DOWN * self.dt)
    else:
      self.t_follow = min(target, self.t_follow + MOONPILOT_T_FOLLOW_SLEW_UP * self.dt)
    return self.t_follow

  def _err_bp(self, sm):
    personality = sm['selfdriveState'].personality.raw
    scale = MOONPILOT_ERR_BP_SCALE.get(personality, MOONPILOT_ERR_BP_SCALE[int(Personality.standard)])
    return [e * scale for e in MOONPILOT_CRUISE_ERR_BP]

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime - sm.logMonoTime['modelV2']) / 1e9
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
  """The fork planner when the driver wants it on a car it can run on, None to leave upstream's in place; read
  once, so a restart."""
  if not in_force(LONGITUDINAL, params or Params(), CP):
    return None
  return MoonpilotLongitudinalPlanner(CP, params=params or Params())
