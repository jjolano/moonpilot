"""moonpilot's longitudinal planner: the fork's own answer to "what acceleration?".

Upstream solves an acados MPC every frame. This is closed-form — no solver, no generated C — and
every number in it is fork-owned, so tuning happens here instead of inside a generated optimization
problem. Braking candidates use `min` arbitration, and the cruise slot remains the set-speed
authority: an oversized lead gap never raises it above the ordinary speed-error request. The policy
is:

  - a spacing regulator whose exact equilibrium is
    ``gap == max(STOP_DISTANCE, t_follow * v_ego)``, with a one-meter inward cushion while closing
    and relative-speed recovery after the cushion is spent;
    the standstill distance is a floor under the headway rather than an offset on top of it;
    its *positive* side carries less authority above 20 m/s (``MOONPILOT_POS_AUTH_V``), because a
    regulator stiff enough to be responsive up there saturates the cruise candidate's fast-accel
    rung and hands the whole follow response to a constant — see that constant for the measurement;
    and its *negative* side is not the gap target's to enforce while the ego is the slower car: the
    1.45 s headway is unreachable at whatever speed the lead chooses, so following at a set speed
    above the lead's would otherwise mean braking to the lead's pace and staying there
    (``MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T``);
  - the time gap itself, biased by ``MoonpilotLeadLateral`` when the lead is predicted to leave the
    path — the fork's lateral prediction reaches the longitudinal policy here, since there is no
    MPC danger zone to scale. That prediction is a two-state, hysteretic claim rather than a
    continuous scale, because ``inPath`` at highway speed is mostly the model's own lateral
    uncertainty and a linear map from it put a moving target under the regulator
    (``MOONPILOT_OUT_OF_PATH_ENTER``);
  - one relative-speed arrival term targeting the greater of the soft rest gap and the lead's
    headway, continuous from a stopped lead through a crawl to ordinary moving following;
  - a time-to-collision approach term that takes over once the headway it wants passes the
    regulator's braking authority, and behind it a stopping floor — the old kinematic term kept as a
    bound, since a proportional TTC term ramps rather than stops and binds too late above ~34 m/s.
    The deeper bound wins, so closing-rate braking reaches the plan wherever it asks for more than
    stopping needs, and the floor covers the rest. The jerk limit turns candidate admission steps
    into a rate without relaxing the emergency braking budget;
  - the cruise term, and the model's own accel — raw in experimental mode, and outside it a
    braking-only candidate behind ``MoonpilotModelBraking`` (see ``model_candidate``) — the same
    candidates upstream arbitrates between.

``MoonpilotCurveSpeed`` adds two more terms *inside* the cruise slot, so the reported source stays
``cruise`` and ``min`` means they can only ever add braking: a kinematic pre-brake against the model
path's own curvature, and a proportional regulator on the lateral accel the car is actually pulling.
``moonpilot/curve.py`` is where the math and its constants live; both are inert without the model's
path arrays and, for the in-curve term, ``vehicleParameters.valid``.
``MoonpilotSqueeze`` adds one more: when the free corridor from ``modelV2.roadEdges`` pinches under
~2 car widths over the next 40 m, a bounded ramp into the same slot — see ``moonpilot/corridor.py``.
``MoonpilotCoastGrade`` adds a grade-aware coast band inside the cruise slot: when the road pushes the
car away from its set speed, the cruise candidate tapers toward upstream's fitted coast acceleration as
the drift approaches 1.5 m/s, allowing a descent to run up and a climb to sag. It can only replace the
cruise candidate; lead, TTC, stopping-floor, curve and model-braking candidates are still ``min``'d
against it upstream of the actuator clip, so no safety term is weakened. The flag is read every frame,
and missing pose or experimental mode leaves this feature inert.


Two things are deliberately not upstream's:

  - delay compensation is done by predicting the state at the actuator delay and evaluating the
    policy there, not by inverting the published plan through ``get_accel_from_plan``. That inverse
    divides by the delay, so a step in the plan comes out amplified by ``action_t / dt`` — measured
    at 45 m/s^3 and instant ACCEL_MIN saturation in simulation. The delay itself is measured onroad
    by ``moonpilot.latency``: ``action_t`` is ``max(CP.longitudinalActuatorDelay, that estimate) +
    the planner's own period`` (DT_MDL in production), since upstream's constant is a cookie-cutter
    default and a measured value may only lengthen the projection. The model path is the third stale
    stream that prediction has to re-reference: its ``position.x`` is measured from the frame the
    model saw, so the curve candidate reads it ``v_ego * (logMonoTime - timestampEof)`` of the way in,
    the same instant the lead pair is evaluated at.

  - the lead's acceleration is the fork's own estimate — the least-squares slope of ``vLead`` over
    ``moonpilot.lead.LeadAccelEstimator``'s window — because radard's ``aLeadK`` is a Kalman filter
    with a 0.49 s time constant, and a lead braking at -3.5 m/s^2 from a settled follow reached
    -2.0 m/s^2 of command 0.65 s later through it than through the slope (1.15 s against 0.50).
    ``aLeadK`` stays as the fallback.

The planner is picked once, at construction, so the toggle needs a restart, and it only exists on a
car with ``openpilotLongitudinalControl`` — cars whose stock ACC owns acceleration never reach the
seam. Upstream's MPC stays on the line as the fallback.
"""

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
from openpilot.common.params import Params, UnknownKeyName
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
from moonpilot.pose import MOONPILOT_POSE_MAX_DT
from moonpilot.corridor import RollingCorridor, squeeze_accel
from moonpilot.features import COAST_GRADE, CURVE_SPEED, LEAD_LATERAL, LONGITUDINAL, MODEL_BRAKING, SQUEEZE, enabled
from moonpilot.latency import (
  MOONPILOT_LAG_BLOCKS_KEY,
  MOONPILOT_LAG_HISTORY_KEY,
  MOONPILOT_LAG_KEY,
  MOONPILOT_LAG_LOG_DELTA,
  MOONPILOT_LAG_MIN_SPEED,
  MOONPILOT_LAG_PERSIST_EVERY,
  LongLagEstimator,
  completed_blocks,
  persisted_blocks,
)
from moonpilot.jerk import (
  MOONPILOT_LONG_JERK_MIN_SAMPLES,
  MOONPILOT_LONG_JERK_PERSIST_EVERY,
  MOONPILOT_LONG_JERK_SCALE_KEY,
  LongitudinalComfortJerkEstimator,
)

from moonpilot.lead import MOONPILOT_LEAD_ACCEL_TAU_RESET, LeadAccelEstimator, nearest_lead_in_path, same_car
from moonpilot.pitch import (
  MOONPILOT_PITCH_MIN_SAMPLES,
  MOONPILOT_PITCH_MIN_SPEED,
  MOONPILOT_PITCH_OFFSET_KEY,
  MOONPILOT_PITCH_PERSIST_EVERY,
  PitchOffsetEstimator,
)
from moonpilot.radar_latency import (
  MOONPILOT_RADAR_LATENCY_KEY,
  MOONPILOT_RADAR_LATENCY_PERSIST_EVERY,
  MOONPILOT_RADAR_LATENCY_SAMPLES_KEY,
  RadarLatencyEstimator,
  persisted_seed as radar_latency_seed,
)
from moonpilot.slam import ego_speed_correction

LongCtrlState = car.CarControl.Actuators.LongControlState
# From cereal, not from long_mpc — that module imports the compiled acados solver at import time,
# and this file is on plannerd's init path.
LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
Personality = log.LongitudinalPersonality

# Starting points, all fork-owned. Tune against logs.
MOONPILOT_STOP_DISTANCE = 6.0  # m; the standstill gap — the time-gap setpoint's floor, where the
# stopping floor arrives, and the gap held behind a stopped lead
# s; time gap per personality, keyed by the enum's raw value rather than the schema member:
# `log.LongitudinalPersonality.standard` is a plain int, but the same enum read off a message is a
# capnp _DynamicEnum whose hash is not that int, so a dict keyed by the members never matches.
MOONPILOT_T_FOLLOW = {
  int(Personality.relaxed): 1.75,
  int(Personality.standard): 1.45,
  int(Personality.aggressive): 1.25,
}
MOONPILOT_T_FOLLOW_SLEW_DOWN = 0.5
MOONPILOT_T_FOLLOW_SLEW_UP = 2.0
MOONPILOT_K_GAP = 0.3  # 1/s^2 on the spacing error
MOONPILOT_K_V = 0.6  # 1/s on the relative speed
MOONPILOT_FOLLOW_CUSHION = 1.0  # m; a closing approach may spend this much of the nominal gap before
# the spacing regulator adds its full speed-matching brake. The moving cushion is closing speed times
# K_V / K_GAP, capped here: below the cap it exactly cancels the relative-speed term at the nominal
# gap, then vanishes as ego becomes slower so the same regulator smoothly reopens the exact time gap.
# It also tapers out before the time-gap target reaches the standstill floor. TTC and stopping-floor
# terms never read it.
MOONPILOT_FOLLOW_TAPER_M = 3.0
# The headway below which the gap is a safety argument rather than a comfort one, and so the only
# thing that may enforce it is the brake. 1.0 s: the driver's own highway-speed follow on the manual
# corpus rides a 2.56 s median headway with a p10 of 1.55 s (10,949 frames, 255 segments, routes
# 000003c2/c6/c8 and 000003bc), so this sits a third below the driver's own worst decile and half
# below `t_follow`. Its only consumer is the speed floor in `lead_accel`, and the direction it errs
# in is the safe one: too high and the car holds a short headway, too low and it sheds speed to
# satisfy a headway the lead itself set.
MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T = 1.0
# How fast the lead may be slowing before the gap becomes a reason to brake again. -0.1 m/s^2, which
# is `LeadAccelEstimator`'s own noise floor against a lead holding speed (its error against a centered
# reference is 0.19 m/s^2 sd on the corpus, so this is a permissive bound, not a tight one) and well
# under the -0.5 a real onset reads within 0.1 s. The direction it errs in is the safe one: too
# permissive and the car coasts for a moment into a lead's brake, which the suite measured as 2.5 m
# of lost end gap on a mid-approach brake.
MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD = -0.1
# The floor's two gates are fades, not switches, because both sit on the steady follow's own
# equilibrium: at a settled follow `v_lead_match - v_ego` and `a_lead` straddle zero by radar noise
# alone (0.03-0.05 m/s of `vLead`, ~0.1 m/s^2 of slope), and a hard gate there swapped the floor's ~0
# for the gap term's -0.2 on alternate frames (route 000003f8 segment 44). The floor now fades out over
# this much ego excess speed and this much further lead braking, so it is whole exactly where it was
# whole before and gone where a real closing or a real brake starts. The speed fade is kept to about
# two sigma of that noise on purpose: wider, and a car that is genuinely a little faster keeps most of
# the floor and drifts inside the target instead of being braked back out of it.
MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V = 0.1  # m/s of ego speed above the matched lead speed
MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_A = 0.2  # m/s^2 of lead braking past MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD
# The positive side's authority: a multiplier on the regulator's output, and only on the output, and
# only where it is already positive. 1.0 at and below 20 m/s, so the whole low-speed follow law — every
# approach, every stop, every maneuver this fork has already been driven on — is bit-identical to what
# it was; above it the spacing regulator can no longer reach the cruise ladder's saturated fast-accel
# rung, which is what turned a proportional response into a step.
#
# Why the output and not the two terms: the terms carry opposite signs whenever the ego is both too
# far back and too fast, and scaling them individually moves that *sum* toward zero — a state that
# braked at the -1.0 approach clamp reads -0.84 under a 0.35 authority. `test_the_multipliers_move_
# the_positive_side_and_nothing_else` is that measurement, and it is why this is one number.
#
# The measured fault, route 000003df segment 17, 53.5-55.5 s: a lead pulling away at ~0.8 m/s^2 with
# the gap 1.5 m past its own target. The old regulator read +0.99 there, so `policy`'s `min` handed
# the whole positive side to the cruise ladder's flat +0.87 for two seconds — the surge from the seat,
# while the ego trailed by 0.4-0.95 m/s and the gap opened 3 m. Upstream's MPC, fed the identical
# radarState at the identical tick, read +0.27..+0.41 and wanted *more* distance than this planner
# does (33-40 m against 31 m): the difference is the response shape, not the spacing target. 0.35 is
# what puts this regulator on upstream's number, and it moves the point at which the speed term alone
# reaches the rung from 1.45 m/s of lag — which was exactly the rung — out to 4.4.
#
# The ramp is deliberately narrow (20 to 22 m/s, 72 to 79 kph). The rung it has to stay under is
# already flat above 17.5 m/s, so this is the band where the ladder rather than the follow law was
# setting the magnitude, and a gain that finished falling at 25 would still have been 0.68 at 22 —
# back inside the range the drive complained about. Above 22 m/s it eases to 0.28 by 30.
#
# One cost, stated: the positive region is `[[0, -1], [f*K_GAP, -f*K_V]]`, so the damping ratio
# scales as sqrt(f) — 0.33 here against 0.55, with the period stretching from 11 s to 19 s. That is
# the price of a three-times gentler ask, the closed-loop suite still settles, and a slow drift of a
# meter or two over twenty seconds is a better thing to feel than a two-second step.
MOONPILOT_POS_AUTH_V_BP = [0.0, 20.0, 22.0, 30.0]  # m/s
MOONPILOT_POS_AUTH_V = [1.0, 1.0, 0.35, 0.28]
MOONPILOT_APPROACH_DECEL = 1.0  # m/s^2; the spacing regulator's braking authority, and the
# decel the approach terms bind past — one number, so the
# candidates meet at the same output
# s; the arrival term's slack is floored at this many seconds of the current closing, so it reads
# -closing / (2 * T) at and inside its target instead of -closing**2 / (2 * slack) with both going to
# zero together. Unfloored, that ratio is the radar's dRel noise: on route 000003f8 segment 44 the last
# meter of an approach at 0.2-0.45 m/s of closing read -1.00, -0.10, -0.34, -1.00 on consecutive frames
# and the car took -0.77 m/s^2 actual while matched to within 0.3 m/s. Above `slack = T * closing` the
# law is unchanged. The cost is where the approach lands: the last meter is no longer bought at the
# approach clamp, so a closing approach finishes a little inside its target and the speed floor rests
# it there — 8.79 m at 6.5 m/s (1.35 s) on `test_mid_approach_lead_brake_stays_safe`, against 9.36 m
# when the end was a -1.0 step. Swept 1-2 s on the route: 1.5 s takes the follow's high-frequency
# command noise from 0.141 to 0.122 m/s^2 and its frames under -0.5 from 171 to 117; 2 s buys 0.006 more
# for another 0.15 m inside the target.
MOONPILOT_ARRIVAL_MIN_T = 1.5
# Below set speed, a far moving-lead approach may take the slow rung while a gentle coast still has
# room to arrive. Flown on `_fly`: 12->10 m/s from 120 m at 64.8 kph never settles within 0.5 m in
# 100 s without it and settles at 58 s with it; 24->22 m/s from 100 m at 108 kph 65 s -> 46 s; route
# 000003e4's -0.098 m/s^2 peak and 46.5 s arrival unchanged. A lead braking later meets the same state
# response an at-set approach gets (-1.685 against a fresh at-set -1.701 m/s^2 from 23.8 m/s, 56 m).
# Catch-up stops for a lead at `MOONPILOT_LEAD_ACCEL_TAU_RESET` (-0.5 m/s^2), the estimator's own
# braking boundary, not -0.1: a steady lead reads under -0.1 on 11 % of frames at 0.03 m/s of `vLead`
# noise, and that flipping doubled to tripled the approach jitter catch-up adds.
MOONPILOT_CATCH_UP_COAST_FRACTION = 0.2
# The stopping floor's admission, scheduled on the ego's speed: it is also the approach plateau, because
# the floor is a constant-decel stop that takes over once it needs this much. Measured on this driver's
# 47 manual stops behind a stopped lead: a median plateau near -1.0 from 6-12 m/s and -1.5..-1.7 from
# 12-20 m/s, held to ~2 m/s. Held at 1.0 below 8 m/s, so it matches the regulator's and TTC term's
# shared gate (MOONPILOT_APPROACH_DECEL) there. This is the admission split that was once measured and
# reverted for raising the average decel and peak; that raise is now the point, on the driver's data.
MOONPILOT_FLOOR_ADMISSION_BP = [8.0, 14.0]  # m/s
MOONPILOT_FLOOR_ADMISSION_V = [1.0, 1.6]  # m/s^2
# The soft rest point used by the shared arrival law. Moving leads transition continuously onto
# their time-gap target; a stopped lead is not a separate approach mode. The spacing regulator and
# stopping floor still protect the nominal gap, so this is a comfort aim, not a contact margin.
MOONPILOT_STOP_REST = 5.2  # m
MOONPILOT_TTC_TARGET = 3.0  # s; headway the approach term holds the closing rate inside. Lower is
# shallower — a_ttc = -K*(closing - slack/T) deepens with T, so this
# is the dial between the floor's plateau (T below ~2 the term never
# governs at all) and a hard late ramp: peak on a 25 m/s approach to a
# stopped lead is 1.23 here against 1.42 at 5.0 and 1.57 at 8.0, and the
# narrower window is also the *smoother* one, 4 command reversals over
# the maneuver against 13 at 5.0. What it gives up is coverage: TTC
# governs gaps 22-65 m at 25 m/s rather than 20-117 m, so a gently
# closing lead is the floor's business either way (measured on a
# settled follow at 25 m/s behind a -2 m/s^2 lead brake, TTC is the
# governing term on 14 % of the lead-term calls at T=3 and 17 % at T=5 —
# a minority either way, which is what the dial is choosing between)
# The TTC headway on the ego's speed: MOONPILOT_TTC_TARGET from 8 m/s up, 1.5 s by 2 m/s. At 3 s all the
# way down, the term governs a stop from ~5 m/s at -1.8 m/s^2 — harder than the driver brakes there —
# which reaches 2 m/s with ~3.3 m to go and leaves a ~4 s creep. The driver reaches 2 m/s ~2.3 m before
# rest and finishes in ~2 s (median of 47 manual stops behind a stopped lead). The shared arrival
# law and the unchanged spacing regulator now ease the crawl without an exact-distance finish.
MOONPILOT_TTC_TARGET_BP = [2.0, 8.0]  # m/s
MOONPILOT_TTC_TARGET_V = [1.5, MOONPILOT_TTC_TARGET]  # s
MOONPILOT_K_TTC = 1.0  # 1/s on the excess closing rate
MOONPILOT_MIN_SLACK = 1.0  # m; the standstill target is soft inside this final meter: the exact
# kinematic floor gives way to the TTC/regulator profile there, while higher-speed stops still bind
# s; the stopping floor's anchor slides from the standstill gap to `MOONPILOT_STOP_REST` below
# `(STOP_DISTANCE - STOP_REST) / this` (4 m/s) of ego speed, as this much of the ego's own travel.
# Aimed at a fixed point, a constant-decel stop is marginal: any closing the plan did not see — the
# radar's own latency is the measured one — is owed at the end, where the demand grows fastest. On
# route 000003d2 that was a -1.45 plateau that rose to -1.92 at 2.7 m/s and let go to -1.1 a second
# later. Aimed `this * v_ego` short of the rest point, a law that is followed exactly asks less and
# less (d|a|/dt = c v^4 / 4 q^3 with q the remaining slack) and still arrives at the rest point: the
# stop eases in the way a driver's does. Above 4 m/s the anchor is the standstill gap it always was,
# so nothing at speed moves — a braking lead matched at 20 m/s is planned exactly as before.
MOONPILOT_STOP_TAPER_T = 0.2
# s; how much older a radar lead's `dRel` is than the carState it is planned against, beyond the
# message age `lead_age` already carries — learned per car by `moonpilot/radar_latency.py` from
# near-rest radar frames (`vLead / aEgo` under excitation), not keyed by fingerprint. A car whose
# `CP.radarDelay` already aligns the composition reads ~0 there, so the residual is applied only
# where it is measured. The gap is that stale too — 0.75 m at 5 m/s on the RAV4 — which is the
# closing the stopping floor then owed at the end.
MOONPILOT_LEAD_PREVIEW_T = 0.5  # s of a lead's own accel credited into the speed each side reads
# One length; direction picks which term it feeds. Braking goes to the safety terms' lead speed
# (`v_lead_eff`): a lead that is braking is matched as if it had already shed this much speed, which
# is what makes the onset early — measured on a settled follow at 20 m/s, a lead braking at
# -3.5 m/s^2 brings the command to -2.0 m/s^2 at 0.50 s here against 0.95 s with no credit — and it
# is also the one thing here that can brake for a lead whose brake is not real. It stops at this
# length rather than a full second on both counts: measured closed-loop from 60 m at 30 m/s, a lead
# tapping -5 m/s^2 for 0.5 s peaks the command at 1.58 here against 3.36 at a full second, and the
# ego stays 1.33 m/s above the slowest the lead ever went where the full second put it 1.98 m/s
# below; while the margin on a *sustained* brake is untouched — from the 1.45 s follow gap at 20 m/s,
# a lead holding -5 m/s^2 to rest leaves 5.77 m of end gap against 5.33 m at a full second and 0.60 m
# with no credit at all. Acceleration goes to the *regulator's* lead-speed term only
# (`v_lead_match`), where the cruise cap already bounds what it can ask for: the speed being matched
# is the lead's next one, so the ego anticipates the gap opening instead of chasing it. That half is
# one-sided the other way — a braking lead gets nothing there — and it never reaches the safety
# terms, so no predicted launch can release braking. Cost of the accel half, by construction: while a
# lead accelerates at `a_lead` the regulator's equilibrium moves from `gap == target` to
# `gap == target - (K_V / K_GAP) * a_lead * this` — 1 m closer per m/s^2 of lead accel, held only as
# long as the estimate says the lead is still gaining. Measured: a settled follow behind a lead
# holding 0.5 m/s^2 sits 0.35 m closer than the setpoint where the uncredited law sits 0.13 m behind
# it, and a lead that launches at 1 m/s^2 for 3 s and then brakes at -3 to rest ends 6 cm *further*
# back — the transient does not survive the approach. At a launch it is a live term on its own
# against a slow lead — a 0.5 m/s^2 launch draws 0.292 m/s^2 at 0.25 s against 0.165, and 0.15 m of
# gap by 2 s — while against a hard one (2 m/s^2) the comfort ramp is what binds and this term moves
# 0.005 m until `MOONPILOT_JERK_LAUNCH` unlocks it. Nothing else about a lead's accel is predicted:
# the TTC term reads the same braking-credited speed as the safety terms, and the estimator's accel
# reaches both terms one other way — the delay match rather than a credit — `update` hands the
# policy `lead_state_at`'s speed at `action_t + lead_age`, which carries ~0.25 s of it. This
# constant is the extra prediction on top of that.
MOONPILOT_LEAD_BRAKE_SUSTAIN_T = 2.0  # s; how long `stopping_decel` assumes a braking lead keeps
# braking. It is the one dial between the two ways that term can be wrong, and both ends were flown
# through this planner (closed loop, `_planner`/`_inputs`, contact = gap < 0.4 m):
#   0 s   — no prediction at all, the plain relative-frame match. Contacts a lead braking -3.5 or -5
#           to rest from the 1.45 s follow gap at 20 m/s, and a 15 m cut-in that then brakes at -4.
#   inf   — the lead is predicted all the way to rest. Safe, but it turns a lead *tapping* -5 for
#           half a second at 30 m/s from 60 m into a -3.50 m/s^2 command where the absolute-frame
#           law it replaced asked -1.89 and the end gap is ~40 m either way. That is deeper than the
#           3.36 that capped `MOONPILOT_LEAD_PREVIEW_T` at 0.5 s, on that same scenario.
#   2.0 s — the tap costs -1.00 m/s^2, below both, while the sustained end gaps hold: 5.00 m at
#           -3.5 to rest (5.09 before), 5.70 m at -5 (5.51), 4.91 m on the cut-in (4.97), and the
#           four pinned stopped-lead approaches are bit-identical. 1.5 s was measured too and gives
#           2.88 m on the -5 case; 3.0 s holds the margins but puts the tap back at -1.86.
# A lead braking at -8 from either geometry contacts at every horizon including infinity: the ego
# needs more than ACCEL_MIN there, which is physics rather than this constant.
MOONPILOT_A_LEAD_MIN = -10.0  # m/s^2; bounds on a lead's accel estimate, upstream's (long_mpc.process_lead)
MOONPILOT_A_LEAD_MAX = 5.0
MOONPILOT_OUT_OF_PATH_T_FOLLOW = 0.7  # time-gap scale for a lead predicted to leave the path
# ...and the evidence that asks for it, as a two-state decision with hysteresis rather than a
# continuous scale. `moonpilot.lead.inPath` is a probability that the lead's own lateral Gaussian
# lies inside the ego path corridor, evaluated over a 4 s horizon and filtered with an instant rise
# and a 1.0 s decay. Two things about it are measured, on route 000003df segments 5, 13, 17 and 18,
# 689 samples where radard's yRel and the published inPath for the same slot agree within 0.25 s:
#
#   lead                 n     inPath median   p10    p90
#   in lane (<1.8 m)   621        0.54        0.23   0.92
#   adjacent (>3 m)     50        0.13        0.02   0.95
#
# So the number is a real signal but a weak one — corr(|yRel|, inPath) is -0.22, and the p90s nearly
# touch — and mapping it linearly onto a time gap was the mistake. A lead the car was comfortably
# following in lane read 0.54 at best and 0.23 at its p10, so the linear map delivered a permanent
# 1.10-1.30 s headway where the personality asks 1.45: a 36 m target silently given as 28-33 m,
# breathing with the estimate. That is a moving target under a proportional regulator, which is the
# second half of the reported fault.
#
# These two thresholds are the fix, and 0.10 is where the separation is: it fires on 4.2 % of in-lane
# leads and 46.9 % of adjacent-lane ones, the best ratio available (0.20 gives 7.6 % against 63.3 %,
# 0.30 gives 15.6 % against 67.3 %, and above 0.40 the in-lane false-fires double). The reduction is
# a claim about the lead *leaving*, it takes strong evidence for it, and it latches until the
# evidence is clearly gone — the band between the two is the hysteresis, wide enough that inPath's own
# jitter cannot cross it. The failure direction is the safe one: inPath that says nothing leaves the
# car at the personality's own headway, and the only thing the feature can fail to do is close up.
MOONPILOT_OUT_OF_PATH_ENTER = 0.10
MOONPILOT_OUT_OF_PATH_LEAVE = 0.30
# The driver's ladder: one accel per named driving state, measured from ~4 h of manual longitudinal
# on the RAV4 (308 segments: `~/route-corpus` plus routes 000003c2/c6/c8 and 000003bc; stock ACC off,
# gear drive). The planner interpolates between rungs instead of scaling a gain, so what it asks for
# is always something this driver does. Grade is removed (pitch relative to its segment median).
# Slow accel is the median of accelerating gas episodes' mean accel per start-speed band (0,
# 0.5-5, 5-10, 10-15, 15+ m/s); the 0 m/s point is held at the 2.5 m/s value so slow never exceeds
# fast. Fast accel is instantaneous: p75 of accel at each speed while the episode is still >= 5 m/s
# short of the speed it ends at — the ladder's own fast-rung condition — so it carries the shape of a
# launch rather than its average: ramping in to ~1.97 at 2.5-3.5 m/s and tapering after (a noisy
# 5.5 m/s bin, 1.50 between 1.81 and 1.60, is dropped). `cruise_cap`'s combined budget
# (`MOONPILOT_A_TOTAL_MAX_V`) is 2.0 through the launch band and 1.7 from 7 to 20 m/s, so the rungs'
# launch peak is what binds there. The corpus tops out at 25 m/s; interp holds the last rung above it.
MOONPILOT_LADDER_V_BP = [0.0, 2.5, 7.5, 12.5, 20.0]  # m/s
MOONPILOT_SLOW_ACCEL_V = [0.77, 0.77, 0.67, 0.42, 0.31]  # m/s^2
MOONPILOT_FAST_ACCEL_BP = [0.5, 1.5, 2.5, 3.5, 4.5, 7.0, 9.0, 11.0, 13.5, 17.5]  # m/s
MOONPILOT_FAST_ACCEL_V = [0.83, 1.70, 1.96, 1.97, 1.81, 1.60, 1.47, 1.27, 1.13, 0.87]  # m/s^2
# The fast rung above is what the driver's feet did, which is not what the car delivers for it. Swept
# across 434 cached segments (29 engaged events commanding >= 1.0 m/s^2 for >= 0.4 s at >= 2 m/s, the
# peak `aEgo` searched 2 s past the run for the vehicle's own lag): the car lands 1.27x the ask below
# 3 m/s, 1.00x at 3-6, 1.07x at 6-10 and 1.13x at 10-15, so it overshoots nearly everywhere and does
# so worst at the launch peak where the ask is largest. Four separate routes show the same shape at
# 2 m/s (cmd 1.97 -> a 2.43..2.51). The ladder value stays the measurement; this is the factor that
# turns it into a setpoint the plant can hold, applied only to the fast rung and only where the
# overshoot was measured, so a merge at speed keeps the authority the driver measured. 0.786 is the
# inverse of the 0-3 m/s median ratio; it reaches 1.0 by 6 m/s, where the sweep found none.
MOONPILOT_FAST_ACCEL_TRACK_BP = [0.0, 6.0]  # m/s
MOONPILOT_FAST_ACCEL_TRACK_V = [0.786, 1.0]  # dimensionless; 1.0 above the last breakpoint
# Coasting: no pedals, level road, median by speed (2-5, 5-10, 10-25 m/s). Agrees with `coast_accel`'s -0.3.
MOONPILOT_COASTING_BP = [3.5, 7.5, 15.0]  # m/s
MOONPILOT_COASTING_V = [-0.10, -0.30, -0.33]  # m/s^2
MOONPILOT_LEAD_COAST_MAX_V = 4.0
MOONPILOT_LEAD_COAST_MAX_GAP = 10.0
MOONPILOT_LEAD_COAST_BRAKE_A = 0.3
# Brake rungs: mean decel of 316 brake episodes starting above 5 m/s — light p90, medium median,
# hard p10. Speed-independent above 5 m/s. Hard brake (-1.75, peaks near -2.9) is the driver's own
# emergency-free ceiling and is not a cruise rung: set-speed control never brakes that hard.
MOONPILOT_BRAKE_LIGHT = -0.75  # m/s^2
MOONPILOT_BRAKE_MEDIUM = -1.25  # m/s^2; also the cruise floor, and forceDecel's
# Speed error (v_cruise - v_ego) at each cruise rung: medium brake, light brake, coasting, hold, slow
# accel, fast accel. The slow rung sits at +1 m/s, measured: taking the plateau a manual speed-up
# settles on as the driver's target (117 speed-ups on cached routes 000002ac-00000405, no lead within
# 80 m through the rise or the plateau), the driver still asks +0.19..+0.41 inside the last 0.5 m/s and
# +0.25..+0.48 at 0.5-1 m/s, where a slow rung at +2 tapered to +0.04..+0.24. Flown from those
# speed-ups with the plateau as set speed, +2 reached within 0.5 m/s of it in a median 9.8 s against
# the driver's 4.9; +1 reaches it in 5.8 s (per flight a median 0.6 s and a p90 2.5 s behind the
# driver), its median accel curve stays within 0.06 m/s^2 of the driver's median for the first 6 s
# (single flights differ by a median 0.39 at their worst instant), and no flight overshoots the target.
# It asks up to ~0.2 more than the driver at 1-2 m/s of error below 10 m/s.
# The braking side is unchanged; grade is not removed from these samples.
MOONPILOT_CRUISE_ERR_BP = [-5.0, -3.0, -1.5, 0.0, 1.0, 5.0]  # m/s
# Personality scales this error axis rather than the rung magnitudes: one factor on every breakpoint
# (0 stays 0, so the ladder keeps its shape), so aggressive binds the same measured rungs at less
# speed error and relaxed at more. Keyed by the enum's raw value, like MOONPILOT_T_FOLLOW — a
# _DynamicEnum read off a message does not hash as its int. Hand-picked: the manual corpus cannot
# derive it, since during manual driving vCruise is valid on only 5 % of frames (stock ACC off); the
# plateau measurement above stands in for the standard axis only.
MOONPILOT_ERR_BP_SCALE = {
  int(Personality.aggressive): 0.7,
  int(Personality.standard): 1.0,
  int(Personality.relaxed): 1.4,
}
# The crawl band: idle creep and the fast-crawl climb to it (docs/longitudinal.md, "The crawl band").
MOONPILOT_CREEP_SPEED = 1.9  # m/s
MOONPILOT_FAST_CRAWL = 0.31  # m/s^2
# Combined accel budget. 2.0 up to 4.5 m/s lets the measured launch rungs (1.96-1.97 at 2.5-3.5 m/s)
# through: at a flat 1.7 the planner held a trapezoid at the clip where 20 of 31 manual launches behind
# a departing lead (cached routes 000002ac-00000405) peaked above 1.7, median 1.92 at 2.8 s. Flown from
# those same states the planner now peaks at a median 1.97 at 2.8 s, with a median 6 s speed of 8.36
# against the driver's 8.24 m/s.
# Lateral demand at launch speeds is small, so the cornering share of the budget does not bind there.
MOONPILOT_A_TOTAL_MAX_BP = [4.5, 7.0, 20.0, 40.0]  # m/s
MOONPILOT_A_TOTAL_MAX_V = [2.0, 1.7, 1.7, 3.2]  # m/s^2
MOONPILOT_COAST_BAND = 1.5  # m/s; allowed set-speed drift on a grade, above on a descent and below on a climb
# 3.4 mph / 5.4 km/h is large enough to let a hill spend itself without making the set speed meaningless.
MOONPILOT_COAST_GRADE_MIN = 0.4  # m/s^2; minimum road term (`accel_coast - coast_accel(0.0)`) before this applies
# Below this is a moderate road, about a 7 % grade; only high-grade hills get the coast band.
MOONPILOT_COAST_ACCEL_MAX = 1.0  # m/s^2; bound on the coast command for garbage pitch and the ACCEL_MAX sentinel
# Both bad pose input and the missing-pose sentinel must not turn into an unbounded coast command.
MOONPILOT_COAST_RECOVERY_ACCEL = 0.15  # m/s^2; positive-ask cap just past the climb band edge
# At the edge the coast taper is already 0; the plain ladder's slow/fast rung would surge back toward
# set speed on the same hill that just earned the sag. Cap holds only in [band, 2*band) of underspeed —
# further out (a launch, a long deficit) the plain law still runs.
MOONPILOT_JERK_UP = 1.5  # m/s^3
MOONPILOT_JERK_LAUNCH = 4.0  # m/s^3; the up-limit at and below MOONPILOT_JERK_LAUNCH_SPEED, tapering
# back to `MOONPILOT_JERK_UP` at twice it, and only from the second frame of a ramp — `jerk_limit`
# keeps the first step at the comfort value, which is what holds a parked car's brake hold out of one
# frame of a lead's reported speed. Measured gains: docs/longitudinal.md, "The up-jerk is scheduled".
MOONPILOT_JERK_LAUNCH_SPEED = 2.5  # m/s
MOONPILOT_JERK_DOWN = 2.0  # m/s^3 at a -2.0 ask; 6.05 m end gap, no contact at 90-144 kph
MOONPILOT_JERK_EMERGENCY = 10.0  # m/s^3, reached at ACCEL_MIN
MOONPILOT_JERK_ONSET = 1.0  # m/s^3 above JERK_ONSET_A; this and below: docs/longitudinal.md
MOONPILOT_JERK_ONSET_A = -1.0  # m/s^2
MOONPILOT_JERK_ONSET_LEAD_A = -0.2  # m/s^2; a lead slowing past this keeps JERK_DOWN
MOONPILOT_CRAWL_REST = 4.0  # m
MOONPILOT_CRAWL_STOP_DECEL = 1.5  # m/s^2; the driver's crawl-stop peak
MOONPILOT_BRAKE_EXIT_A = -0.5  # m/s^2; braking past this holds the positive ask, as curve braking does
MOONPILOT_BRAKE_EXIT_JERK = 0.5  # m/s^3; its rebuild with the wheel straight, above the launch band
MOONPILOT_ALLOW_THROTTLE_THRESHOLD = 0.4
# Throttle comes back only once the model's gas-press probability clears this, not the 0.4 that took it
# away. On route 000003f8 the probability hovered on 0.4 while following a slowing lead and the cruise
# cap flipped between its rung and the coast accel: a third of the no-throttle runs were one frame,
# median three, and the command dithered -0.2 <-> -0.42 behind them. The signal itself is good — in
# that route's manual driving the driver was on the gas on 3 % of frames under 0.4 and 78 % above it —
# so the fix is hysteresis on it, not a different threshold.
MOONPILOT_ALLOW_THROTTLE_RELEASE = 0.5
MOONPILOT_ALLOW_THROTTLE_RETURN_T = 0.5  # s the probability must stay over the release before throttle returns.
# The 0.1 hysteresis is narrower than the signal's own frame-to-frame swing: on route 0000040e it read
# 0.31 -> 0.53 -> 0.38 -> 0.66 inside 0.2 s, and the cruise term stepped +0.08 <-> -0.30 three times in a
# second ahead of a stop (338.4-339.3 s). Losing throttle stays immediate, so the coast starts as early.
MOONPILOT_MIN_ALLOW_THROTTLE_SPEED = 2.5  # m/s
# Soft landing behind a stopped lead. The spacing regulator is the only term braking inside the standstill
# gap at a crawl (TTC and the stopping floor are gated on ~1.0 m/s^2 there), and its gap-deficit brake held
# -0.54..-0.77 m/s^2 down to 0.3 m/s on route 0000040e (126-138 s): the car came to rest from -0.7, a
# 0.61 m/s^2 step in 0.1 s on the IMU against 0.17 for the model-led stop that tapered to -0.2. Below
# `MOONPILOT_SOFT_STOP_V` and behind a lead slower than `MOONPILOT_SOFT_STOP_LEAD_V`, the regulator's brake
# is replaced by the kinematic stop to a rest point that moves in as the car slows,
# `MOONPILOT_STOP_REST + SOFT_STOP_REST_PER_V * v`: at the band edge it is the approach's own constant-decel
# plateau, so the handover has no step, and the shrinking `v^2` tapers it out toward rest. A decel tapering
# linearly with speed was tried first: it asked ~2 m/s^2 at 3.7 m/s against a 1.2 plateau, so the band edge
# stepped deeper before easing. Flown against stopped leads from 5-12 m/s the car rests 4.93-5.16 m behind
# (5.0-5.26 before) and late detections or braking leads 4.63-4.71 (4.80-4.86), at -0.11..-0.15 m/s^2 as
# it stops against -0.25..-0.38, with every peak unchanged.
MOONPILOT_SOFT_STOP_REST_PER_V = 0.15  # s; the rest point's extra distance per m/s of speed
MOONPILOT_SOFT_STOP_V = [3.0, 5.0]  # m/s: fully in below, out above
MOONPILOT_SOFT_STOP_LEAD_V = [0.5, 1.5]  # m/s of braking-credited lead speed: fully in below, out above
MOONPILOT_SOFT_STOP_MIN_T = 0.5  # s; the slack floor, in travel time, once the rest point is crossed while
# still moving: -v / (2 * MIN_T) instead of a near-zero denominator (-4.6 m/s^2 at 0.68 m/s past it).
MOONPILOT_SOFT_STOP_HANDOFF_V = 0.1  # m/s of closing over which the law hands standstill back to the regulator
# A far radar lead outlives a lapse in radard's association. Toyota's radar and the model disagree on a
# far lead's range by ~14 m (median) and the model's speed by +-6-7 m/s (p10/p90) there, so beyond ~60 m
# radard keeps losing the match and handing leadOne to the vision-only lead: 84 % of vision-only leads on
# route 000003f8 were past 60 m (median 93 m). On segment 42 that alternated a 14 m/s radar track with a
# 21 m/s vision lead 20 m nearer, plus 0.25-1.5 s dropouts, and the arrival command stepped -0.5 <-> +0.1
# seven times in four seconds. The last radar lead, projected forward, stays a *candidate* for this long
# after it stops being leadOne; the current lead (vision or radar) stays one too, so `min` still brakes
# for a stopped car only vision sees. Only above `MOONPILOT_RADAR_HOLD_MIN_V` and while the projected
# gap is over `MOONPILOT_RADAR_HOLD_MIN_GAP`, where association is what fails, never for a stop: a lead
# that leaves the slot at rest or close in is released at once, as upstream does.
MOONPILOT_RADAR_HOLD_T = 2.0  # s
MOONPILOT_RADAR_HOLD_MIN_V = 5.0  # m/s of ego speed
MOONPILOT_RADAR_HOLD_MIN_GAP = 40.0  # m
MOONPILOT_MODEL_BRAKE_THRESHOLD = -0.5  # m/s^2; outside experimental mode the model is ignored
# until it asks for at least this much braking, and past that its ask blends in over
# `MOONPILOT_MODEL_BRAKE_FADE_T` and then goes in whole: the floor under it is the actuator's own
# ACCEL_MIN, not a fork value. See `model_candidate` for why a fork-owned floor above that was removed.
# Once admitted, the model keeps its slot until its ask rises past this release, which climbs from the
# entry threshold at `MOONPILOT_MODEL_STOP_SPEED` to upstream `should_stop`'s own 0.1 at creep speed.
# The model tapers its ask as it finishes a stop — on route 000003d2 it braked from 17 m/s and crossed
# back over -0.5 at 0.9 m/s — so a fixed deadband handed the last meter to the cruise term, which
# asked +1.0 and was already moving the car when the driver braked. Below creep speed the release is
# exactly experimental mode's stop contract: held while the model's ask is under 0.1.
MOONPILOT_MODEL_STOP_SPEED = 5.0  # m/s
MOONPILOT_MODEL_STOP_RELEASE = 0.1  # m/s^2; `should_stop`'s threshold
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
# Matches the fork's own standstill convention (`MOONPILOT_STANDSTILL_SPEED`).
MOONPILOT_SHOULD_STOP_SPEED = 0.1
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

  The taper's sign follows the grade, so it cannot command toward the set speed against gravity. This
  only changes the cruise candidate: lead, TTC, stopping-floor, curve and model-braking candidates are
  ``min``'d against it upstream of the actuator clip, so no safety term is weakened.
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


def stopping_decel(v_ego, v_lead, a_lead, slack, sustain=MOONPILOT_LEAD_BRAKE_SUSTAIN_T) -> float:
  """The constant decel that keeps `slack` meters of gap, in the frame the gap actually closes in.

  One law, relative frame throughout, against the speed the lead is *predicted* to hold: it keeps
  its current braking for at most `MOONPILOT_LEAD_BRAKE_SUSTAIN_T` and never past its own rest,
  then holds whatever speed that left it at. The lead's travel beyond a lead permanently at that
  speed cancels analytically and leaves ``0.5 * |a_lead| * t^2`` of extra room, so the whole thing
  is one expression with no case split at the horizon.

  Two things it is deliberately not. It is **not absolute-frame**: the gap closes at the relative
  rate while the lead keeps travelling, so what spends the slack is ``(v_ego - v_lead)^2 / 2a``,
  not ``(v_ego^2 - v_lead^2) / 2a``. That older form is exact only for a *stopped* lead and is
  deeper than the geometry by ``(v_ego + v_lead) / (v_ego - v_lead)`` for every moving one —
  measured on route 000003c8's engaged block (70-115 m, 4-7 m/s of closing), 0.82 m/s^2 deeper on
  45 of 45 engaged braking ticks, a 4.4-5.3x over-ask, and with the regulator positive and the TTC
  term not yet binding it was the only admitted braking candidate, so it governed by default and
  the command sat flat on it: a decel floor where the geometry asked for a ramp.

  And the planner's own call does **not predict the lead all the way to rest**, which is the
  unbounded end of this same family (`sustain` = inf, so `t` is the time to rest) and is unsafe in
  the other direction: a lead that taps -5 m/s^2 for half a second at 30 m/s from 60 m draws a
  -3.50 m/s^2 command out of it against -1.89 from the absolute-frame law it replaced, for an end
  gap that is the same 40 m either way. That is the same trade `MOONPILOT_LEAD_PREVIEW_T` is capped
  at 0.5 s for, and the bound is sized the same way — see the constant. `required_decel` passes
  `inf` instead, because FCW is a worst-case warning behind its own -4 m/s^2 gate and costs nothing
  when it is early: at v_ego 30 behind a lead doing 28 at 25 m and braking at -6, the unbounded
  horizon reads -5.00 (fires, and contact is genuinely unavoidable — 90.1 m of room needs more than
  ACCEL_MIN) where the 2 s bound reads -2.67 and says nothing.

  Both of those are the *late* crossing, where the ego is still faster than the lead when the
  lead's braking ends. Where it is not — where the ego matches the lead's speed while the lead is
  still braking — crediting the lead's whole braking-phase travel credits travel it has not made
  yet: at v_ego 25 behind a lead doing 10 and braking at -0.5 with 25 m of slack, that reads -2.50
  where the true constant decel is -5.00, because the ego comes to rest at 10 s having been given
  the lead's full 20 s of travel. That case has its own exact closed form — the relative speeds
  converge at ``a - |a_lead|``, so the requirement is ``|a_lead| + closing^2 / (2 * slack)``, the
  lead's own deceleration plus the relative-frame term — and it applies exactly when the crossing
  lands inside the braking phase, ``2 * slack <= closing * t``. The two agree at that boundary, so
  this is one continuous law in two pieces rather than a blend. It binds mostly on the FCW arm,
  where `t` is a time to rest measured in tens of seconds.

  A stopped lead is unchanged by any of it: ``-v_ego^2 / (2 * slack)``, which is what every pinned
  stopping-distance number in this file was measured on.
  """
  v_lead = max(float(v_lead), 0.0)
  closing = float(v_ego) - v_lead
  if a_lead < 0.0 and v_lead > 0.0:
    # No `closing > 0` here: a lead braking hard at *matched* speed is the case the prediction
    # exists for, and gating on the present closing rate would read it as no threat until the
    # speed measurement caught up. `required_decel(20, 20, 20, -8)` is -4.47 with the lead
    # predicted to rest and -0.0 with such a gate. The early-crossing test below is false for a
    # non-positive `closing`, so those states take the prediction branch, which handles them.
    t = min(sustain, v_lead / -a_lead)
    if 2.0 * slack <= closing * t:  # the ego matches it before its braking ends
      return -(-a_lead + closing**2 / (2.0 * slack))
    slack += 0.5 * -a_lead * t**2
    v_lead = max(0.0, v_lead + a_lead * t)
  return -(max(float(v_ego) - v_lead, 0.0) ** 2) / (2.0 * slack)


def crawl_accel(a_track, v_ego, v_lead, a_lead) -> float:
  """The spacing regulator's positive ask, capped at the driver's fast-crawl rung while both cars are
  at creep speed — idle creep, not a launch. A lead accelerating at least `MOONPILOT_FAST_CRAWL` is
  pulling away, and that launch is bounded by the cruise ladder instead. Fades out between one and two
  creep speeds of the faster car, so the band edge is not a step. Braking is never shaped here.
  """
  if a_track <= MOONPILOT_FAST_CRAWL or a_lead >= MOONPILOT_FAST_CRAWL:
    return a_track
  w = float(np.interp(max(v_ego, v_lead), [MOONPILOT_CREEP_SPEED, 2 * MOONPILOT_CREEP_SPEED], [1.0, 0.0]))
  return a_track + w * (MOONPILOT_FAST_CRAWL - a_track)


def soft_stop(a_track, v_ego, gap, v_lead_eff) -> float:
  """The spacing regulator's brake near rest behind a (near-)stopped lead, replaced by the kinematic stop
  to a rest point that moves in as the car slows (see `MOONPILOT_SOFT_STOP_REST_PER_V`). Only a brake while still
  closing: a positive ask (a creep, a launch) is untouched, and the law hands back over the last
  `MOONPILOT_SOFT_STOP_HANDOFF_V` of closing so the stop hold and the launch logic own standstill exactly
  as before. Fades out above `MOONPILOT_SOFT_STOP_V`, for a lead above `MOONPILOT_SOFT_STOP_LEAD_V`,
  and over physical gaps from twice `MOONPILOT_MIN_SLACK` to that minimum, where the regulator owns braking."""
  if a_track >= 0.0:
    return a_track
  w = float(np.interp(v_ego, MOONPILOT_SOFT_STOP_V, [1.0, 0.0])) * float(np.interp(v_lead_eff, MOONPILOT_SOFT_STOP_LEAD_V, [1.0, 0.0]))
  w *= float(np.interp(v_ego - v_lead_eff, [0.0, MOONPILOT_SOFT_STOP_HANDOFF_V], [0.0, 1.0]))
  # Comfort never replaces the short-gap brake: retain the regulator inside its existing one-meter
  # minimum slack, fading the taper in over the next meter. A time-only floor can otherwise crawl
  # through a close stopped lead because neither TTC nor the stopping floor is admitted at this speed.
  w *= min(max(gap / MOONPILOT_MIN_SLACK - 1.0, 0.0), 1.0)
  if w <= 0.0:
    return a_track
  rest = MOONPILOT_STOP_REST + MOONPILOT_SOFT_STOP_REST_PER_V * v_ego
  a_soft = -(v_ego**2) / (2.0 * max(gap - rest, MOONPILOT_SOFT_STOP_MIN_T * v_ego, 0.05))
  return a_track + w * (a_soft - a_track)


def pos_authority(v_ego) -> float:
  """The multiplier on the spacing regulator's output where that output is positive.

  One reader, so nothing can apply it to the terms instead of the result — the terms carry opposite
  signs whenever the ego is both too far back and too fast, and scaling them individually moves that
  sum toward zero, i.e. weakens a brake. 1.0 at and below 20 m/s.
  """
  return float(np.interp(v_ego, MOONPILOT_POS_AUTH_V_BP, MOONPILOT_POS_AUTH_V))


def departure_crawl(a_track, v_ego, v_lead, a_lead) -> float:
  """The spacing regulator's ask, floored at the crawl a lead pulling away is owed before the gap has
  reopened.

  A stop finishes inside the standstill gap — `MOONPILOT_STOP_REST` is 0.8 m short of it and the car
  rests nearer 4.5 m on the road — so at the launch the regulator's spacing error is a brake: -0.44
  m/s^2 from 4.5 m, held until the lead's speed alone outweighed it. On route 000003d2 that kept the
  stop hold on for a second after the lead moved and the driver took over with the gas. The driver's
  own launch is a brake release into idle creep the moment the lead goes, so while the lead is moving
  and faster than the ego, the regulator's relative-speed term alone — capped at the fast-crawl rung
  — is a floor under it, fading out over the crawl band like `crawl_accel`. The lead's speed carries
  its accel both ways (`MOONPILOT_LEAD_PREVIEW_T`), so a lead that is braking is owed nothing until it
  would still be the faster car. Speed matched, the floor is zero and the regulator is untouched: the
  gap is not recovered at a crawl, only kept, and every braking term below still reads the physical gap.
  """
  if v_lead <= MOONPILOT_SHOULD_STOP_SPEED:
    return a_track
  w = float(np.interp(v_ego, [MOONPILOT_CREEP_SPEED, 2 * MOONPILOT_CREEP_SPEED], [1.0, 0.0]))
  floor = w * min(MOONPILOT_K_V * (v_lead + float(a_lead) * MOONPILOT_LEAD_PREVIEW_T - v_ego), MOONPILOT_FAST_CRAWL)
  return max(a_track, floor) if floor > 0.0 else a_track


def arrival_accel(closing, slack, a_lead) -> float:
  """The constant decel that closes `closing` exactly where `slack` runs out, clamped to the approach
  decel, behind a lead accelerating at `a_lead`.

  The lead's own positive accel is credited: closing decays at the ego's decel plus the lead's accel,
  so the ego owes that much less. It used to be a switch instead (off above +0.1 m/s^2), and that sat
  inside a far lead's slope noise — on route 000003f8 segment 42 a lead 60-120 m out read +0.1..+0.35
  and the command stepped between this term's ~-0.5 and the cruise rung every half second of the
  approach. A lead pulling away faster than the term asks for still retires it: the credited ask turns
  positive and `min` no longer selects it. Braking is not credited here — `closing` is already measured
  against the braking-credited `v_lead_eff`. `slack` may be zero or negative (at or inside the follow
  target): it is floored at `MOONPILOT_ARRIVAL_MIN_T` of closing, so the term fades with closing."""
  return max(-(closing**2) / (2.0 * max(slack, closing * MOONPILOT_ARRIVAL_MIN_T)), -MOONPILOT_APPROACH_DECEL) + max(float(a_lead), 0.0)


def lead_accel(v_ego, gap, v_lead, a_lead, t_follow, catch_up_accel=0.0, catch_up_coast=0.0) -> float:
  """The spacing regulator and one relative-speed approach for every slower lead.

  The spacing regulator holds gap == max(STOP_DISTANCE, t_follow * v_ego) and matches the lead's
  speed; its braking authority is capped at the approach decel, so large speed errors do not turn
  into hard braking through the gain. STOP_DISTANCE is a *floor* rather than an offset added to
  every headway: added, the effective headway was ``t_follow + STOP_DISTANCE / v_ego`` — 1.65 s at
  30 m/s where the driver chose 1.45, 2.05 s at 10 m/s, 3.45 s at 3 m/s — so the car hung back
  further the slower it went. It only governs below ``STOP_DISTANCE / t_follow`` (4.1 m/s at the
  standard personality), which is the region where there is no time gap left to hold.

  The nominal follow gap is a soft boundary while closing. The regulator moves its working target
  inward by at most `MOONPILOT_FOLLOW_CUSHION`, with the uncapped distance equal to closing speed
  times `K_V / K_GAP`. Below the cap that exactly cancels speed-matching brake at the nominal gap:
  relative motion may spend the cushion instead of treating the target as a wall. Once ego is slower
  the cushion is zero, and the unmodified relative-speed term lets the gap reopen before adding
  acceleration. Exact speed-match equilibrium stays at the nominal gap, and the cushion tapers out
  with the time-gap headroom so it can never move the target inside `STOP_DISTANCE`. This changes
  only the spacing regulator; the TTC and stopping-floor terms below still read the physical gap.

  The approach is three terms, and the deeper one wins against the regulator's capped output.

  The first is time-to-collision: it holds the closing rate inside what the slack affords at
  TTC_TARGET seconds of headway — ``closing <= slack / TTC_TARGET`` — and binds once honoring that
  needs more than the approach decel, i.e. ``slack < TTC_TARGET * (closing - APPROACH_DECEL /
  K_TTC)``: with the shipped numbers, ``slack < 3 * (closing - 1)``, which is a 78 m gap at 25 m/s
  on a stopped lead. Binding is not governing, though: it reaches the output only where it is also
  deeper than the floor below, which on that same approach is the gap window 22 m … 65 m — and it is
  only deeper than the floor for T above 2, below which this term never governs at all. Inside that
  window it is what makes the approach respond to a lead that starts slowing, because it reads the
  closing rate, which the lead's own speed history drives.

  The arrival term matches the lead's braking-previewed speed at
  ``max(STOP_REST, t_follow * v_lead_eff)``. The target is continuous through a crawl and a stop:
  no lead-speed threshold switches between full-stop and moving-lead approaches. Its slack is
  floored at ``ARRIVAL_MIN_T * closing``, so the request eases as closing disappears even when the
  preferred rest point has already been crossed. Positive lead acceleration retires it continuously.
  Below set speed, the slow-rung catch-up is available only while coasting still has room to match;
  that credit fades to zero as the lead's speed falls to rest. TTC and the stopping floor remain
  independent minima, with their braking preview, sustain and admission unchanged.

  The third is the stopping floor, the kinematic term kept as a bound rather than as the approach,
  and it is measured in the frame the gap actually closes in. Outside `MOONPILOT_MIN_SLACK` it is
  the harder of two constant decels, `stopping_decel`: matching the lead's *braking-credited* speed
  before the slack is spent, and stopping short of where a lead that keeps braking comes to rest.
  Inside that final meter its denominator stops shrinking: the standstill distance becomes a soft
  target and the TTC/regulator profile can shape the crawl stop instead of the exact-distance term
  deepening without bound. It is not redundant outside that comfort region. A TTC term is
  proportional on the closing rate, so it ramps — and ramping is not stopping: TTC_TARGET = 5 binds
  at a slack of ``5 * (v - 1)``, which above ~34 m/s is *inside* the ``v^2 / 7`` that stopping at
  ACCEL_MIN needs, and on the TTC term alone this car reaches a stopped lead at 14 m/s from 36 m/s.
  The floor binds earlier and `min` cannot out-vote it, so it holds where the TTC term asks for
  less; where the TTC term asks for more, the TTC term wins.

  The handover in either case is a step, not a crossover: at the crossing the regulator's output is
  whatever the spacing error says, positive while the gap is still wide — +69.7 m/s^2 at 25 m/s and
  the 318.5 m stopping crossing — and the approach terms replace it from there. That step is only
  safe because it is a *candidate*: `policy` takes the minimum, so while the lead asks for more than
  the cruise term the cruise term governs and the output never sees it. What the output steps by is
  therefore cruise's cap minus the approach decel, and the caller's jerk limit turns that into a rate.
  """
  gap = max(float(gap), 0.0)
  v_lead = max(float(v_lead), 0.0)
  # The lead's own accel, credited forward into the two speeds it belongs in, one direction each.
  # Braking goes to the speed both terms below are measured against — the cautious direction, and the
  # only prediction either of them makes (`MOONPILOT_LEAD_PREVIEW_T`) — while acceleration goes to the
  # speed *this* regulator matches, and never to the safety terms: crediting it there would release
  # braking on nothing but a predicted launch.
  v_lead_eff = max(0.0, v_lead + min(float(a_lead), 0.0) * MOONPILOT_LEAD_PREVIEW_T)
  v_lead_match = v_lead + max(float(a_lead), 0.0) * MOONPILOT_LEAD_PREVIEW_T
  gap_target = max(MOONPILOT_STOP_DISTANCE, t_follow * v_ego)
  closing_match = max(v_ego - v_lead_match, 0.0)
  gap_cushion = min(
    MOONPILOT_FOLLOW_CUSHION,
    closing_match * MOONPILOT_K_V / MOONPILOT_K_GAP,
    max(t_follow * v_ego - MOONPILOT_STOP_DISTANCE, 0.0),
  )
  a_track = MOONPILOT_K_GAP * (gap - gap_target + gap_cushion) + MOONPILOT_K_V * (v_lead_match - v_ego)
  if a_track > 0.0:
    a_track *= pos_authority(v_ego)
  a_track = max(
    a_track,
    -MOONPILOT_APPROACH_DECEL,
  )
  # The time gap is not a reason to brake while the ego is the slower car. Every settled follow at a
  # set speed above the lead's is in that state — the ego sits inside `gap_target` at whatever speed
  # the lead chooses — and the gap term alone then asks for the full approach brake. Measured on
  # route 000003df segments 17-18: 30 m behind a lead holding 24 m/s with the set speed at 111 kph,
  # `a_track` reads -1.00 at 24, 25 and 26 m/s, i.e. at *exactly* the lead's speed too, and -1.00 at a
  # 10 m gap with the ego 1 m/s slower, while `stopping_decel` reads -0.00 to -0.08 in the same states.
  # Nothing there needs stopping for; the command is the headway the lead itself set, bought with the
  # throttle shut. That is the ego decaying to below the lead's speed and staying there.
  #
  # So the response is to keep pace instead, and the floor is the speed the regulator matches. Four
  # conditions, and each is a distinct way this could be wrong — all four found by the suite.
  #
  # `v_lead_match >= v_ego` — the floor's own value is the speed match, so it must be inert where the
  # ego is clearly the *faster* car: there the closing approach and its cushion are the whole
  # mechanism, and holding the command at zero let a fly cross 7.5 m inside the target against a 1 m
  # cushion. Equal counts, because "at the lead's speed" is exactly the state the floor exists for.
  # It fades out over `MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V` of excess speed rather than switching,
  # because a settled follow straddles equality by radar noise alone.
  # `a_track < 0` — the floor raises a *negative* ask. Where the ask is already positive the scaled
  # regulator owns it, and letting the unscaled floor raise that would undo the authority above (at
  # the measured surge state it read +0.54 against the authority's +0.33).
  # `a_lead > MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD` — the lead is not slowing. Without this the
  # floor is wrong in the most expensive direction, and quietly: holding the ego's speed against a
  # lead braking at -3.5 makes the ego relatively *faster*, so the gap closes instead of opening —
  # `test_mid_approach_lead_brake_stays_safe` used to lose 2.5 m of end gap to it, and with the
  # arrival term now sharing the approach it loses 1 cm and 0.012 m/s of resting speed (the arrival
  # term lands the car on the target, the floor holds it there). The 0.5 s braking credit inside
  # `v_lead_eff` buys grace exactly where grace is not wanted. `LeadAccelEstimator` reads a real
  # onset inside 0.1 s, so a bound at -0.1 keeps estimator noise out without keeping the brake; the
  # floor fades over `MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_A` past it, for the same noise reason.
  # the headway — below `MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T` the gap *is* the safety argument
  # rather than a comfort one, and the full brake stands. This also keeps the hole this would
  # otherwise open narrow and honest, because the fork has no other term that brakes a short headway
  # when the ego is the slower car: TTC is skipped entirely, being a function of closing rate, and the
  # stopping floor is shallow against a lead that is not itself slowing.
  # `v_ego > MOONPILOT_CREEP_SPEED` — a moving follow, not a hold. At rest the headway test is
  # vacuous and `v_lead_eff` clamps to zero, so the condition is trivially true and the car would
  # accelerate away from a stopped lead inside the standstill gap.
  #
  # Unscaled, deliberately: this is not a comfort quantity, it is "do not lose the lead's speed".
  # TTC and the stopping floor are `min`'d on top and still own every case that needs braking.
  if a_track < 0.0 and v_ego > MOONPILOT_CREEP_SPEED and gap >= MOONPILOT_FOLLOW_SPEED_FLOOR_HEADWAY_T * v_ego:
    w_v = (v_lead_match - v_ego) / MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_V + 1.0
    w_a = (a_lead - MOONPILOT_FOLLOW_SPEED_FLOOR_MIN_A_LEAD) / MOONPILOT_FOLLOW_SPEED_FLOOR_FADE_A + 1.0
    w = min(max(w_v, 0.0), 1.0) * min(max(w_a, 0.0), 1.0)
    a_track += w * max(MOONPILOT_K_V * (v_lead_match - v_ego) - a_track, 0.0)
  a_track = crawl_accel(a_track, v_ego, v_lead, a_lead)
  a_track = departure_crawl(a_track, v_ego, v_lead, a_lead)
  a_track = soft_stop(a_track, v_ego, gap, v_lead_eff)
  if a_track > 0.0 and v_ego > v_lead_match and gap > gap_target:
    a_track *= min(1.0, (gap - gap_target) / MOONPILOT_FOLLOW_TAPER_M)
  closing = v_ego - v_lead_eff
  if closing <= 0.0:
    return a_track  # not closing: nothing to brake for
  slack = max(gap - MOONPILOT_STOP_DISTANCE, 0.0)
  a_ttc = -MOONPILOT_K_TTC * (closing - slack / float(np.interp(v_ego, MOONPILOT_TTC_TARGET_BP, MOONPILOT_TTC_TARGET_V)))
  a = min(a_track, a_ttc) if a_ttc < -MOONPILOT_APPROACH_DECEL else a_track
  # One arrival law for stopped, crawling and faster moving leads. The rest target is soft:
  # never divide by a tenth of a meter merely because the preferred gap has already been spent.
  arrival_target = max(MOONPILOT_STOP_REST, t_follow * v_lead_eff)
  a_arrive = arrival_accel(closing, gap - arrival_target, a_lead)
  if catch_up_accel > 0.0 and catch_up_coast < 0.0 and a_lead > -MOONPILOT_LEAD_ACCEL_TAU_RESET:
    used = a_arrive / catch_up_coast
    if used < 1.0:
      # Stop-and-go gets no catch-up throttle. The credit fades in continuously where the headway
      # target leaves the soft rest gap, reaching full strength at the old moving-lead boundary.
      weight = float(np.interp(t_follow * v_lead_eff, [MOONPILOT_STOP_REST, MOONPILOT_STOP_DISTANCE], [0.0, 1.0]))
      a_arrive += catch_up_accel * weight * (1.0 - used)
  a = min(a, a_arrive)
  # The stopping floor makes this law safe outside the final soft meter rather than merely responsive.
  # A TTC term ramps on closing rate, and ramping is not stopping: at 36 m/s with TTC_TARGET = 5 it
  # binds 175 m out, where coming to rest behind a stopped lead needs 185 m. The kinematic term
  # remains the bound throughout that regime. Only inside MOONPILOT_MIN_SLACK does its denominator
  # stop shrinking, so the exact standstill point is not chased with increasing crawl-speed braking.
  anchor = MOONPILOT_STOP_REST + min(MOONPILOT_STOP_TAPER_T * v_ego, MOONPILOT_STOP_DISTANCE - MOONPILOT_STOP_REST)
  a_stop = stopping_decel(v_ego, v_lead_eff, a_lead, max(gap - anchor, MOONPILOT_MIN_SLACK))
  a = min(a, a_stop) if a_stop < -float(np.interp(v_ego, MOONPILOT_FLOOR_ADMISSION_BP, MOONPILOT_FLOOR_ADMISSION_V)) else a
  # Crawl stop: spend the rest gap toward MOONPILOT_CRAWL_REST before braking past the driver's peak.
  cap = min(stopping_decel(v_ego, v_lead_eff, a_lead, max(gap - MOONPILOT_CRAWL_REST - MOONPILOT_MIN_SLACK, 0.05)), -MOONPILOT_CRAWL_STOP_DECEL)
  return a + float(np.interp(v_ego, MOONPILOT_SOFT_STOP_V, [1.0, 0.0])) * max(cap - a, 0.0)


def lead_braking(lead_states) -> bool:
  return any(a_lead < MOONPILOT_JERK_ONSET_LEAD_A for _, _, _, a_lead in lead_states)


def jerk_limit(a_cmd, a_prev, dt, v_ego, comfort_scale=1.0, lead_braking=False) -> float:
  """Rate limit, asymmetric and urgency-scaled.

  The learned scale is deliberately only on the positive, comfort-side ramp. The braking-side
  interpolation and its emergency endpoint are the stopping-distance-tested constants.
  """
  comfort_scale = float(np.clip(comfort_scale, 0.5, 1.0))
  up = (
    MOONPILOT_JERK_UP
    if a_prev <= 0.0
    else float(
      np.interp(
        v_ego,
        [MOONPILOT_JERK_LAUNCH_SPEED, 2 * MOONPILOT_JERK_LAUNCH_SPEED],
        [MOONPILOT_JERK_LAUNCH, MOONPILOT_JERK_UP],
      )
    )
  ) * comfort_scale
  # The gentle onset is off at a crawl and while a lead brakes, where a shallow ask is urgent.
  onset = float(np.interp(v_ego, [MOONPILOT_JERK_LAUNCH_SPEED, 2 * MOONPILOT_JERK_LAUNCH_SPEED], [MOONPILOT_JERK_DOWN, MOONPILOT_JERK_ONSET]))
  onset = MOONPILOT_JERK_DOWN if lead_braking else onset
  down = float(np.interp(a_cmd, [ACCEL_MIN, -2.0, MOONPILOT_JERK_ONSET_A], [MOONPILOT_JERK_EMERGENCY, MOONPILOT_JERK_DOWN, onset]))
  return float(np.clip(a_cmd, a_prev - down * dt, a_prev + up * dt))


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

  `stopping_decel` against the contact margin instead of the standstill gap, and with the lead
  predicted all the way to rest (`sustain=inf`) rather than to the planner's comfort horizon. The
  two differ deliberately: the planner's bound exists so a lead's brake *tap* does not buy a
  -3.5 m/s^2 command, and a warning has no comfort to protect — it sits behind its own -4 m/s^2
  gate, so the only thing a shorter horizon can do there is stay silent through a real one. At
  v_ego 30 behind a lead doing 28 at 25 m braking at -6, to-rest reads -5.00 and fires where the
  planner's 2 s horizon reads -2.67, and the contact is genuinely unavoidable: 90.1 m of room from
  30 m/s needs more than ACCEL_MIN.

  Without the lead's own stopping distance in the room, a lead braking hard at matched speed reads
  as no threat at all until the speed measurement has caught up — 0.7 s at -8 m/s^2, most of the
  margin the warning exists to buy. And the speed match is relative-frame for the reason
  `stopping_decel` gives: a lead holding 28 m/s 12 m ahead of a car doing 30 needs 0.17 m/s^2, not
  the 4.94 the absolute-frame form read — which was an FCW on an ordinary overtake.
  """
  return stopping_decel(v_ego, max(float(v_lead), 0.0), lead_accel_estimate(a_lead), max(gap - MOONPILOT_CRASH_DISTANCE, 0.1), sustain=math.inf)


def model_candidate(model, e2e, allowed, threshold=MOONPILOT_MODEL_BRAKE_THRESHOLD) -> float | None:
  """The model's own accel as a candidate, and whether it gets one this frame.

  Experimental mode is unchanged: the accel goes in raw, which is what that mode means. Outside it
  the model is a *braking* input on top of the deterministic policy, gated by a deadband and
  otherwise unbounded — the floor under it is the actuator's own, not a fork value.

  The deadband is because `policy` takes the minimum: a model accel dithering either side of zero
  would be a one-way ratchet against the cruise term, and the car would settle below the set speed
  on a clear road. Below the threshold the model is not asking for anything a driver would feel.

  Past it the ask blends in from the other candidates over `MOONPILOT_MODEL_BRAKE_FADE_T` (faster the
  deeper it is, at once from `MOONPILOT_MODEL_BRAKE_FULL`) and then goes in whole — a timed entry, not
  a floor. What bounds it is `ACCEL_MIN`, and that bound is unconditional:
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

  `threshold` is the planner's hysteresis: the entry deadband until the model has a slot, then
  `model_release` — so a stop the model began is a stop the model finishes.
  """
  a = float(model.action.desiredAcceleration)
  if not e2e and (not allowed or not a < threshold):
    return None
  return a


def model_release(v_ego) -> float:
  """The ask an admitted model has to rise past to give its slot back: the entry deadband at speed,
  `should_stop`'s 0.1 in the crawl band, so the model holds a stop it is finishing until it asks to go."""
  return float(np.interp(v_ego, [MOONPILOT_CREEP_SPEED, MOONPILOT_MODEL_STOP_SPEED], [MOONPILOT_MODEL_STOP_RELEASE, MOONPILOT_MODEL_BRAKE_THRESHOLD]))


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
  """The smallest of the candidates, and which one it was.

  `leads` is a sequence of (source, gap, v_lead, a_lead) for the leads that are present, in
  radarState order, each carrying the slot it came from so that an absent leadOne does not make
  leadTwo report itself as lead0. The model's own accel goes last, so an upstream change that lets a
  NaN through the model cannot win the comparison and land in the plan.

  `curve` is `moonpilot.curve.curve_targets`' output and `v_hold` its in-curve setpoint, both fed into
  the cruise slot as a minimum — so the reported source stays `cruise`, and the two curve terms are
  live in experimental mode too. That is the point of putting them there rather than beside the model
  candidate: `min` means they can only ever add braking to whatever the model asked for, never
  substitute for it. `squeeze` is `moonpilot.corridor.squeeze_accel`'s bounded ramp, same slot, same
  rule; the `ACCEL_MAX` default is the no-candidate case.

  The cruise slot remains the set-speed authority. At or above `v_cruise` its ordinary speed-error
  output is non-positive even when a lead gap is oversized, so `min` cannot select positive
  acceleration merely to recover follow distance. A transient brake can therefore leave a larger
  gap behind a lead holding the set speed; closing it waits for the lead to pull away or the ego to
  fall below set speed. That is deliberate: follow-distance recovery never buys overspeed. A slower
  lead that is not pulling away also caps the cruise slot's fast rung when there is too little
  runway to reach set speed and then match the lead at the ordinary follow decel. Below set speed,
  the moving-lead arrival term may use the slow rung only while arrival by coasting still fits. TTC,
  regulator braking and the stopping floor are unchanged.

  `model_weight` is how much of an admitted model ask below the other candidates goes in: the
  candidate is blended from their minimum toward the model by it, so admission past the deadband
  starts where the plan already is instead of stepping onto the model. 1.0 is the raw ask.
  """
  a_curve = min(curve_accel(v_ego, x_ego, curve), lat_accel_hold(v_ego, v_hold), squeeze, exit_cap)
  a_cruise_raw = cruise_accel(v_ego, v_cruise, e2e, steer_angle_deg, CP, accel_coast, allow_throttle, coast_band, err_bp)
  a_cruise = min(a_cruise_raw, a_curve)
  slow = float(np.interp(v_ego, MOONPILOT_LADDER_V_BP, MOONPILOT_SLOW_ACCEL_V))
  coast = float(np.interp(v_ego, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V)) + _coast_grade(accel_coast)[1]
  catch_up = min(a_cruise_raw, a_curve, slow) if v_ego < v_cruise else 0.0
  if a_cruise > slow and v_cruise > v_ego:
    a_open = a_cruise
    for _, gap, v_lead, a_lead in leads:
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
  lead_asks = []
  for source, gap, v_lead, a_lead in leads:
    ask = lead_accel(v_ego, gap, v_lead, a_lead, t_follow, catch_up, MOONPILOT_CATCH_UP_COAST_FRACTION * coast)
    if (
      v_lead < v_ego <= MOONPILOT_LEAD_COAST_MAX_V
      and gap <= MOONPILOT_LEAD_COAST_MAX_GAP
      and a_lead <= -MOONPILOT_LEAD_COAST_BRAKE_A
    ):
      ask = min(ask, float(np.interp(v_ego, MOONPILOT_COASTING_BP, MOONPILOT_COASTING_V)))
    lead_asks.append((ask, source))
  candidates = [(a_cruise, LongitudinalPlanSource.cruise)]
  candidates += lead_asks
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
    # The persisted blocks, restored exactly (`moonpilot.latency.persisted_blocks`); the value and count
    # persisted beside them are what `modeld` reads to decode the model's longitudinal ask.
    self.long_lag.restore(persisted_blocks(self.params))
    self.long_lag_logged = self.long_lag.applied_delay()
    self.long_jerk = LongitudinalComfortJerkEstimator(dt)
    seeded_jerk = self.params.get(MOONPILOT_LONG_JERK_SCALE_KEY, return_default=True)
    if isinstance(seeded_jerk, float) and seeded_jerk < 1.0:
      self.long_jerk.seed(seeded_jerk, MOONPILOT_LONG_JERK_MIN_SAMPLES)

    self.action_t = self.long_lag.applied_delay() + dt
    # The curve speed control's per-car calibration (`moonpilot/curve.py`): realized lateral accel over
    # what the path predicted, one-sided so it can only ever plan for less speed. Seeded from the last
    # drive, and the vehicle model is what turns a measured steer angle into a measured curvature.
    self.VM = VehicleModel(CP)
    self.lat_bias = LatAccelBiasEstimator(dt)
    seeded_scale = self.params.get(MOONPILOT_CURVE_BIAS_KEY, return_default=True)
    if isinstance(seeded_scale, float) and seeded_scale > 1.0:
      self.lat_bias.seed(min(seeded_scale, MOONPILOT_CURVE_BIAS_MAX), MOONPILOT_CURVE_BIAS_MIN_SAMPLES)
    # The standing offset in the reported pitch (`moonpilot/pitch.py`), subtracted before any grade
    # term reads it. Seeded from the last drive because the window is longer than most of them.
    self.pitch_offset = PitchOffsetEstimator(dt)
    seeded_pitch = self.params.get(MOONPILOT_PITCH_OFFSET_KEY, return_default=True)
    if isinstance(seeded_pitch, float) and seeded_pitch != 0.0:
      self.pitch_offset.seed(seeded_pitch, MOONPILOT_PITCH_MIN_SAMPLES)
    self.frames = 0
    # One per radarState slot, ticked every frame so an absent or replaced lead resets its window.
    self.lead_accel = (LeadAccelEstimator(dt), LeadAccelEstimator(dt))
    # The radar lead's own sensor latency (`moonpilot/radar_latency.py`), added to a radar lead's age
    # in `_lead_age`. Seeded from the last drive; zero until enough gated near-rest frames accumulate.
    self.radar_latency = RadarLatencyEstimator()
    radar_seed = radar_latency_seed(self.params)
    if radar_seed is not None:
      self.radar_latency.seed(*radar_seed)
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
    # it, so the correction goes in at the source rather than into one of the candidates. Bounded by
    # a fraction of the raw speed it corrects, which is what makes it exactly zero at a standstill:
    # a scale error is proportional to speed, and the guards that declare rest read this same
    # `v_ego`. Zero when the feature is off, nothing is published, the correction is stale or its
    # publisher is dead, which makes this exactly the plan this planner had before the correction
    # existed.
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
    # Learn from the raw signal whenever the car is driving — the bias belongs to the device and the
    # car, not to who is steering — and correct every grade term with what is trusted so far. Nothing
    # learned is exactly zero, which is the grade input this planner had before.
    self.pitch_offset.update(pitch, pose_valid and CS.vEgo > MOONPILOT_PITCH_MIN_SPEED and not CS.standstill)
    accel_coast = coast_accel(pitch - self.pitch_offset.applied()) if pose_valid else ACCEL_MAX
    # Zero when the pose is missing (accel_coast is the ACCEL_MAX sentinel) or the driver turned the
    # row off — the band is the feature's whole surface, so both collapse to the planner this fork
    # had before it. The normal path reads the toggle on every frame, so a row change applies immediately.
    # An older local params database can predate this row; its shipped default is on, so retain the
    # same behavior until the row is present.
    try:
      coast_enabled = enabled(COAST_GRADE, self.params)
    except UnknownKeyName:
      coast_enabled = True
    coast_band = MOONPILOT_COAST_BAND if pose_valid and coast_enabled else 0.0
    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    if self.allow_throttle:
      self.throttle_return_t = 0.0
      self.allow_throttle = throttle_prob > MOONPILOT_ALLOW_THROTTLE_THRESHOLD
    else:
      self.throttle_return_t = self.throttle_return_t + self.dt if throttle_prob > MOONPILOT_ALLOW_THROTTLE_RELEASE else 0.0
      self.allow_throttle = self.throttle_return_t >= MOONPILOT_ALLOW_THROTTLE_RETURN_T - 1e-9
    self.allow_throttle = self.allow_throttle or v_ego <= MOONPILOT_MIN_ALLOW_THROTTLE_SPEED

    steer_angle = CS.steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg
    t_follow = self._t_follow(sm)
    err_bp = self._err_bp(sm)
    e2e = sm['selfdriveState'].experimentalMode
    # The model's stop hold yields to a departing leadOne once the model asks to go.
    one = sm['radarState'].leadOne
    go = one.present and one.vLead > MOONPILOT_SHOULD_STOP_SPEED and sm['modelV2'].action.desiredAcceleration > 0.0
    threshold = model_release(v_ego) if self.model_braking and not go else MOONPILOT_MODEL_BRAKE_THRESHOLD
    model_accel = model_candidate(sm['modelV2'], e2e, enabled(MODEL_BRAKING, self.params), threshold)
    self.model_braking = model_accel is not None and not e2e
    self.model_fade = min(self.model_fade + self.dt / MOONPILOT_MODEL_BRAKE_FADE_T, 1.0) if self.model_braking else 0.0
    model_weight = 1.0
    if model_accel is not None and not e2e:
      # A deep ask is not held back by the fade: proportionally more of it goes in at once.
      model_weight = max(self.model_fade, float(np.interp(model_accel, [MOONPILOT_MODEL_BRAKE_FULL, threshold], [1.0, 0.0])))

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
    # measurement of the driver's steering. A zero torque-log version is the schema default from an
    # older or different writer, so keep the pre-gate fallback on those cars.
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
        leads.append((source, lead, a_lead, a_lead_tau))
    # FCW keeps both slots: the dedupe below picks a candidate for the plan, and radard's two matches of
    # one car can carry different `modelProb`, so dropping one could drop the only one over the FCW bar.
    # A held track's `modelProb` is zero, so slots alone are every lead FCW could ever fire on.
    fcw_leads = list(leads)
    one, two = (lead for _, lead, _, _ in leads) if len(leads) == 2 else (None, None)
    if one is not None and one.radar and two.radar and same_car(one, two):
      # radard matches both model leads to the nearest radar track, so a stopped car often fills both
      # slots, on two tracks of its own. One vehicle is one candidate: two noisy readings of it under
      # `min` hand the winning slot back and forth and the command pulses (route 00000400 seg 4).
      leads.pop()
    leads += self._radar_hold(sm['radarState'].leadOne, leads, v_ego)
    # Learn the radar's sensor latency from the same pair of slots, on every frame (present or not):
    # the gates inside `update` decide what teaches, so an absent lead is simply a no-op.
    self.radar_latency.update(sm['radarState'].leadOne, a_ego)
    self.radar_latency.update(sm['radarState'].leadTwo, a_ego)

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
    # staleness moves the stop by 0.26 m, against 2.35 m uncorrected).
    #
    # Defaulted rather than required: the maneuver plant hands a bare `dict` with no `logMonoTime` at
    # all, and a missing or equal stamp has to give the zero correction this planner applied before
    # it existed.
    log_mono = getattr(sm, 'logMonoTime', None) or {}
    model_mono = log_mono.get('modelV2')
    radar_mono = log_mono.get('radarState')
    lead_age = max(0.0, (model_mono - radar_mono) / 1e9) if model_mono and radar_mono else 0.0
    # Both halves of the stale interval, not just the lead's: the message's `dRel` predates the
    # command by the lead's age, and `x_pred` carries only the travel over `action_t`. The lead's own
    # travel over that age rides on the extra time; the ego's on `v_ego * age` (`_lead_age`).

    # The model path is the third stale stream, and the only one nothing re-references: its
    # `position.x` is measured from the pose of the frame the model saw, whose exposure ended at
    # `timestampEof`, while the path is re-referenced to the planner tick + `action_t`. That is
    # deliberately a tick *later* than the lead pair's own instant: `lead_age` is a `logMonoTime`
    # delta and both projection windows are `action_t + lead_age` long, so the pair lands at the
    # model's *publish* + `action_t` — one publish-to-send interval (15.9 ms median, 0.48 m at
    # 30 m/s) behind this anchor. The car covers `v_ego * path_age` of that path
    # before this command exists, and the age is the tick's rather than the publish stamp's: msgq
    # buffers, so `recv_time` is set when this process actually reads the message. Measured over
    # 282,554 corpus plans, the publish lag is 31 ms median and the interval from it to the plan's own
    # send another 15.9 ms (21.4 p95) against a 0.58 ms solve — the loop, not the planner — so the
    # tick anchor is worth up to ~0.48 m more at 30 m/s. A clock that does not line up with the stamps (a
    # replay) falls back to the publish lag, which the stamps alone carry, and both candidates are
    # bounded the way `moonpilot/curvature.py` bounds this message's age. Absent, zero or impossible
    # stamps give the zero shift this planner had before it existed — the safe direction, since a
    # shift too large empties the pre-brake's binding set and stops it asking at all.
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
    # The path is re-referenced here, once, rather than folded into each caller's `x_ego`: both the
    # command and the published rollout then pass plain ego travel and share one target.
    curve = curve_targets(sm['modelV2'], curve_allowed, vp.roll, scale, ahead=x_path)
    # The corridor squeeze: same re-reference as the curve (so the window is meters ahead of the
    # car now), same cruise-slot min, gated on the toggle every frame like the other path consumers.
    # Empty or unknown edges give the ACCEL_MAX sentinel and leave the cruise term alone. When a
    # valid egoPose is published the free-space strips are pose-stitched over a short history so a
    # single frame's edge dropout does not release the brake; without a pose (SLAM off, first fix,
    # invalid) this is the single-frame squeeze it was before the history existed.
    squeeze = ACCEL_MAX
    if enabled(SQUEEZE, self.params):
      edges = sm['modelV2'].roadEdges
      if len(edges) >= 2:
        pose = self._ego_pose(sm)
        if pose is not None:
          now = time.monotonic()
          self.corridor.push(now, pose, edges[0], edges[1], now=now)
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
    # cornering or with a curve ahead, else at MOONPILOT_BRAKE_EXIT_JERK above the launch band.
    exit_jerk = MOONPILOT_CURVE_EXIT_JERK
    if min(curve_accel(v_pred, x_pred, curve), lat_accel_hold(v_pred, v_hold), squeeze) < 0.0 or a_prev < MOONPILOT_BRAKE_EXIT_A:
      self.curve_exit_cap = max(a_prev, 0.0)
    else:
      cornering = abs(measured_curvature) * CS.vEgo**2 >= MOONPILOT_CURVE_EXIT_LAT_ACCEL
      curve_ahead = curve_accel(max(v_cruise, v_pred), x_pred, curve) < 0.0
      if not (cornering or curve_ahead):
        exit_jerk = float(np.interp(v_ego, [MOONPILOT_JERK_LAUNCH_SPEED, 2 * MOONPILOT_JERK_LAUNCH_SPEED], [MOONPILOT_JERK_LAUNCH, MOONPILOT_BRAKE_EXIT_JERK]))
      self.curve_exit_cap = min(self.curve_exit_cap + exit_jerk * self.dt, ACCEL_MAX)
    lead_states = []
    for source, lead, a_lead, a_lead_tau in leads:
      age = self._lead_age(lead, lead_age)
      lead_states.append((source, *lead_state_at(lead, self.action_t + age, x_pred + v_ego * age, a_lead, a_lead_tau)))
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

    a_target = float(np.clip(jerk_limit(a_cmd, a_prev, self.dt, v_ego, jerk_scale, lead_braking(lead_states)), ACCEL_MIN, ACCEL_MAX))

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
      err_bp=err_bp,
      model_weight=model_weight,
    )

    self.j_desired_trajectory = np.gradient(self.a_desired_trajectory, MOONPILOT_CONTROL_T_IDX)

    crash = any(
      lead.modelProb > MOONPILOT_FCW_MODEL_PROB and required_decel(v_ego, lead.dRel, lead.vLead, a_lead) < MOONPILOT_FCW_DECEL
      for _, lead, a_lead, _ in fcw_leads
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
    # flag. How fast depends on the lead, and the threshold is `0.3 * dgap + 0.6 * v_lead >= 0.1`, i.e.
    # `dgap + 2 * v_lead >= 0.333 m`: a lead above ~0.17 m/s clears it on speed alone and the car moves
    # within a frame, while a slower one has to open that third of a meter first — 1.6 s of it at
    # 0.1 m/s, 5.4 s at 0.05, 16.6 s at 0.02, and the car is *meant* to sit there rather than creep off
    # after a lead that has barely moved. Below ~0.1 m/s of lead speed the gate's own deadband then
    # makes the flag chatter (measured 20-40 flips in 20 s), which is why `MOONPILOT_SHOULD_STOP_SPEED`
    # is a deadband and not a liveness test.
    # The model's own `shouldStop` stays experimental-only, unlike its accel. It is
    # `should_stop(v_ego, desiredAcceleration)` computed in modeld (`modeld.py:59`), i.e. `v_ego <
    # 0.3 and a < 0.1` — precisely the creeping-behind-a-lead state the `trailing` gate above exists
    # to keep out of `LongControlState.stopping`, so admitting it here would undo that fix from the
    # other side.
    trailing = v_ego > MOONPILOT_SHOULD_STOP_SPEED and any(s == source and v_lead > MOONPILOT_SHOULD_STOP_SPEED for s, _, v_lead, _ in lead_states)
    model_stop = not e2e and self.model_braking and not leads and v_ego <= MOONPILOT_MODEL_STOP_ENTER_SPEED and a_target < 0.0
    self.output_should_stop = (should_stop(v_ego, a_target) and not trailing) or (e2e and sm['modelV2'].action.shouldStop) or model_stop
    self.output_a_target = a_target
    self.source = source
    # Persist the learned value *and its evidence* so the next boot continues from where this drive
    # got to: the count decides whether the value is applied (see the seeding above), so a drive that
    # only earned a block or two hands them on rather than restarting.
    self.frames += 1
    if self.frames % MOONPILOT_LAG_PERSIST_EVERY == 0 and self.long_lag.valid_blocks > 0:
      self._persist_lag()
    if self.frames % MOONPILOT_LONG_JERK_PERSIST_EVERY == 0 and self.long_jerk.status == 'estimated':
      self._persist_long_jerk()
    if self.pitch_offset.samples % MOONPILOT_PITCH_PERSIST_EVERY == 0 and self.pitch_offset.status == 'estimated':
      self._persist_pitch_offset()
    if self.radar_latency.samples > 0 and self.frames % MOONPILOT_RADAR_LATENCY_PERSIST_EVERY == 0:
      self._persist_radar_latency()

    if self.lat_bias.frames % MOONPILOT_CURVE_BIAS_PERSIST_EVERY == 0 and self.lat_bias.status == 'estimated':
      self._persist_lat_scale()
    self.solve_time = time.monotonic() - start

  def _persist_lag(self):
    value = round(self.long_lag.estimate, 3)
    self.params.put(MOONPILOT_LAG_KEY, value)
    self.params.put(MOONPILOT_LAG_BLOCKS_KEY, self.long_lag.valid_blocks)
    self.params.put(MOONPILOT_LAG_HISTORY_KEY, [round(b, 4) for b in completed_blocks(self.long_lag.block_avg)])
    if abs(value - self.long_lag_logged) > MOONPILOT_LAG_LOG_DELTA:
      cloudlog.info(f"moonpilot longitudinal lag {value:.3f} s over {self.long_lag.valid_blocks} blocks, action_t {self.action_t:.3f} s")
      self.long_lag_logged = value

  def _persist_long_jerk(self):
    value = round(self.long_jerk.applied(), 3)
    self.params.put(MOONPILOT_LONG_JERK_SCALE_KEY, value)
    cloudlog.info(f"moonpilot longitudinal comfort jerk scale {value:.3f} over {self.long_jerk.samples} paired ramps")

  def _persist_pitch_offset(self):
    """Persist the learned offset: the window is longer than most drives, so without this it would
    restart from zero every boot and never finish converging."""
    value = round(self.pitch_offset.applied(), 4)
    self.params.put(MOONPILOT_PITCH_OFFSET_KEY, value)
    cloudlog.info(f"moonpilot pitch offset {math.degrees(value):.2f} deg over {self.pitch_offset.samples} frames")

  def _persist_radar_latency(self):
    """Persist the learned latency *and its evidence* so the next boot continues from where this
    drive got to: the count decides whether the value is applied, so a partial drive hands its
    samples on rather than restarting."""
    value = round(self.radar_latency.estimate, 4)
    self.params.put(MOONPILOT_RADAR_LATENCY_KEY, value)
    self.params.put(MOONPILOT_RADAR_LATENCY_SAMPLES_KEY, self.radar_latency.samples)
    cloudlog.info(f"moonpilot radar latency {value:.3f} s over {self.radar_latency.samples} samples")

  def _persist_lat_scale(self):
    """Persist the learned value so the next boot plans with it from the first frame. Gated on a
    trusted estimate, so an unestimated or invalid one never becomes the next drive's scale."""
    value = round(self.lat_bias.estimate, 3)
    self.params.put(MOONPILOT_CURVE_BIAS_KEY, value)
    cloudlog.info(f"moonpilot curve lateral scale {value:.3f} over {self.lat_bias.samples} paired frames")

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
  ):
    """The same policy rolled forward over the published horizon, from (v_ego, a_target). The ego's
    own travel is carried in x, so the gap the leads are rolled against is the gap this plan
    produces. The old loop advanced the state by the *backward* interval and rolled the leads against
    a gap recomputed from the initial speed and the loop's current accel: on a closing lead at
    25 m/s, its speeds sat up to 0.42 m/s away from the consistent rollout's. `lead_age` carries the
    same staleness correction `update` applies, so the published plan is the same prediction the
    command was taken from rather than a fresher one.     `curve` and `v_hold` ride along for the same
    reason: the rollout is the policy the command came from, with the curve's own travel accumulated
    in `x` and the measured curvature held across the horizon — the curve target already carries the
    path re-reference, so `x_ego` is plain ego travel here as it is in `update`. `squeeze` rides the
    same way: the corridor candidate is constant over the rollout (it was computed once for this tick)."""
    speeds = np.zeros(CONTROL_N)
    accels = np.zeros(CONTROL_N)
    v, a, x, t_prev = v_ego, a_target, 0.0, 0.0
    for i, t_idx in enumerate(MOONPILOT_CONTROL_T_IDX):
      t = float(t_idx)
      states = [
        (source, *lead_state_at(lead, t + self._lead_age(lead, lead_age), x + v_ego * self._lead_age(lead, lead_age), a_lead, a_lead_tau))
        for source, lead, a_lead, a_lead_tau in leads
      ]
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
      a = float(np.clip(jerk_limit(a_cmd, a, t - t_prev, v, comfort_scale, lead_braking(states)), ACCEL_MIN, ACCEL_MAX))
      speeds[i] = v
      accels[i] = a
      dt = float(MOONPILOT_CONTROL_T_IDX[i + 1]) - t if i + 1 < CONTROL_N else 0.0
      v_next = max(0.0, v + a * dt)
      x += 0.5 * (v + v_next) * dt
      v, t_prev = v_next, t
    return speeds, accels

  def _radar_hold(self, lead, leads, v_ego):
    """The last radar leadOne, projected forward, as an extra lead0 candidate while radard has handed
    the slot to a vision-only lead or to nothing (MOONPILOT_RADAR_HOLD_T). Its `modelProb` is zero so
    FCW never fires on a held track, and the current lead stays in `leads` beside it, so `min` takes
    whichever brakes more."""
    if lead.present and lead.radar:
      a_lead, a_lead_tau = next((a, tau) for s, _, a, tau in leads if s == LongitudinalPlanSource.lead0)
      snapshot = SimpleNamespace(dRel=float(lead.dRel), vLead=float(lead.vLead), radar=True, modelProb=0.0)
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
    return [(LongitudinalPlanSource.lead0, SimpleNamespace(dRel=gap, vLead=v_lead, radar=True, modelProb=0.0), a_now, a_lead_tau)]

  def _lead_age(self, lead, lead_age):
    """How old a lead's `dRel` is at this tick: the message's age, and a radar lead's own sensor
    latency on top where it is measured (`moonpilot/radar_latency.applied()`)."""
    return lead_age + (self.radar_latency.applied() if lead.radar else 0.0)

  def _ego_pose(self, sm):
    """`(x, y, yaw)` from a fresh, valid `moonpilotState.egoPose`, or None.

    Same getattr probes as `ego_speed_correction`: the maneuver harness passes a plain dict as
    `sm`, and a dead publisher or an unset pose must fall through to the single-frame squeeze
    rather than raise.
    """
    if not (getattr(sm, "valid", {}).get("moonpilotState", False) and getattr(sm, "alive", {}).get("moonpilotState", False)):
      return None
    pose = sm["moonpilotState"].egoPose
    if not pose.valid:
      return None
    age = time.monotonic_ns() - pose.monoTime
    if age < 0 or age > MOONPILOT_POSE_MAX_DT * 1e9:
      return None
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
      return None
    return (pose.x, pose.y, pose.yaw)

  def _t_follow(self, sm):
    personality = sm['selfdriveState'].personality.raw
    # KeyError is not possible for the three current members; an upstream fourth would fall back to
    # the standard gap rather than raise in plannerd.
    base = MOONPILOT_T_FOLLOW.get(personality, MOONPILOT_T_FOLLOW[int(Personality.standard)])
    lateral = enabled(LEAD_LATERAL, self.params)
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
    """The speed-error breakpoints for this frame's personality: the measured axis scaled by
    `MOONPILOT_ERR_BP_SCALE`, raw-value keyed and standard-fallback like `_t_follow`."""
    personality = sm['selfdriveState'].personality.raw
    scale = MOONPILOT_ERR_BP_SCALE.get(personality, MOONPILOT_ERR_BP_SCALE[int(Personality.standard)])
    return [e * scale for e in MOONPILOT_CRUISE_ERR_BP]

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


def moonpilot_longitudinal_active(CP, params: Params | None = None) -> bool:
  """Whether the fork's planner is the one in place on this car, without building one.

  The same decision `moonpilot_longitudinal_planner` makes, so the two cannot disagree — the UI
  asks this to widen the displayed road camera below its speed threshold when the fork planner is
  driving, and a fork planner that is not in place must not have that say. `CP` is None until
  `carParams` lands, which reads as "not yet" rather than as a verdict on the car.
  """
  if CP is None:
    return False
  return bool(enabled(LONGITUDINAL, params or Params()) and CP.openpilotLongitudinalControl)


def moonpilot_longitudinal_planner(CP, params: Params | None = None) -> MoonpilotLongitudinalPlanner | None:
  """The seam's fork side: the fork planner when the driver wants it, None to leave upstream's in
  place. The param is read once, here, so the toggle takes a restart."""
  if not moonpilot_longitudinal_active(CP, params):
    return None
  return MoonpilotLongitudinalPlanner(CP)
