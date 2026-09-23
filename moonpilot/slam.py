"""moonpilot's rolling-window ego-speed correction: what the car's own wheel speed got wrong.

The prior is `carState`'s wheel speed -- what the car believes about its own speed. The observation
is `cameraOdometry`'s forward translation rate -- what the model saw move, with its std. Each
interval of a fixed-lag window blends the two by inverse variance, and the published `dVel` is the
blend minus the raw prior over the window's recent end; a consumer that ignores it (`valid` false)
reduces to exactly the behavior it had before. That is the whole mechanism: numpy and the signals
already on the bus, no new vision frontend, no new service, so it runs on the device's own CPU.
`moonpilot/leadd.py` publishes it on `moonpilotState.egoCorrection`.

Two things about it are not obvious.

- **What the window buys is lag, not accumulation.** With independent per-tick errors, inverse-
  variance weights are constant and the window length would not matter. Errors that are *not*
  independent are what matters, and they are exactly the error this exists to correct: a wheel-speed
  scale error (a 2 % error at 20 m/s is 0.4 m/s of the planner's own input, 1.2 m of gap over a 3 s
  approach). The window is how long an estimate of that is allowed to lag.
- **The prior is read at the odometry's pose time, not at the frame's own.** `cameraOdometry`'s
  `trans`/`rot` describe the pose as of `timestampEof - MOONPILOT_SLAM_POSE_DELAY` (locationd's own
  `CAM_ODO_POSE_DELAY`, and the reason it computes its observation time that way). Pairing "latest
  odometry" with "latest carState" instead puts an `a * 0.1 s` error on every interval -- 0.15-0.35
  m/s while braking, the same order as the signal this file measures and correlated with exactly the
  approach regime the planner's gap terms live in. So `PriorChannel` samples the wheel speed with
  its own message times and interpolates it onto that third time, and a frame whose pose time
  the prior cannot span is skipped rather than extrapolated. The exposure stamp is what the message
  carries the pose for; the publish time is 30.5 ms later on the device and reading the prior there
  gives back 30 % of the error, so `moonpilot/leadd.py` stamps the node from `timestampEof`.

Two approximations are stage-1 on purpose. **The device frame is still treated as the car frame**:
for each newly ingested pose, trusted `extrinsicsCalibration.rpyCalib` rotates `trans` from the
calibration frame into the device frame with `rot_from_euler`, and rotates its std with
`rotate_std`; invalid, absent, malformed or out-of-bound calibration leaves that transform as
identity. This uses the latest calibration even though the pose is 0.1 s older, which is acceptable
while calibration converges over minutes, and a calibration change affects only new nodes rather than
re-deriving older window samples. **The outputs are clamped**, which means the two sources disagree
by more than a sensor plausibly can; that is a tuning question the constants below own.

The only correction published is `dVel`, the one the planner can act on: the fork planner's gaps
are radar-measured, so shifting them by an odometry position offset would inject error rather than
remove it.
"""

import math
import time
from collections import deque
from typing import NamedTuple

import numpy as np

from moonpilot.lead import resample

# Starting points, fork-owned. Tune against logs: plot `egoCorrection` against `cameraOdometry` and the
# wheel speed in PlotJuggler, and watch how often `valid` falls out.
MOONPILOT_SLAM_WINDOW_S = 5.0  # s; how long an estimate of a motion error may lag
MOONPILOT_SLAM_RATE_HZ = 20.0  # window nodes per second; cameraOdometry and carState both run here
MOONPILOT_SLAM_MIN_NODES = 10  # below this the window says nothing, so it says nothing
# How wrong the prior may be, in the same per-tick currency as the odometry stds. 1 m/s is ~5 % at
# highway speed: the tyre wear, the grade and the slip a wheel-speed scale error is made of. Larger
# means the correction leans further onto the model's odometry.
MOONPILOT_SLAM_RAW_V_STD = 1.0  # m/s
# Mirrors of locationd's own numbers, pinned by test_slam: the std floor it refuses below, and the
# multiplier it discounts temporally correlated odometry noise by.
MOONPILOT_SLAM_MIN_STD = 1e-5  # m/s; locationd.MIN_STD_SANITY_CHECK
MOONPILOT_SLAM_TRANS_STD_MULT = 4  # locationd.CAM_ODO_TRANS_STD_MULT
# s; locationd.CAM_ODO_POSE_DELAY. cameraOdometry's trans/rot describe the pose as of this long
# before the message, and the prior is read there so both sides of an interval cover the same span.
MOONPILOT_SLAM_POSE_DELAY = 0.1
MOONPILOT_SLAM_PRIOR_WINDOW_S = 1.0  # s of prior samples kept; the pose time is 0.1 s behind at most
MOONPILOT_SLAM_PRIOR_RATE_HZ = 100.0  # nominal carState rate
MOONPILOT_SLAM_MAX_CORR_VEL = 2.0  # m/s; hard clamp on |dVel|
MOONPILOT_SLAM_MAX_AGE = 0.30  # s; the transport budget, on top of MOONPILOT_SLAM_POSE_DELAY: every
# correction is that pose delay old by construction, and charging the consumer's budget for it would
# spend a third of the staleness allowance on the sensor's own lag.
MOONPILOT_SLAM_VEL_WINDOW_S = 1.0  # s of the window dVel speaks for; the whole window lags a change
MOONPILOT_SLAM_CORR_STD_MIN = 0.02  # m; 1/(1 + corrStd) is the trust a consumer scales in, so the
MOONPILOT_SLAM_CORR_STD_MAX = 2.0  # floor is nearly full trust and the ceiling is a third of it
MOONPILOT_SLAM_MAX_SCALE_ERR = 0.1  # fraction of the speed being corrected, and the low-speed bound
# A wheel-speed scale error is proportional to the speed it is measured on -- 2 % of 20 m/s is
# 0.4 m/s, 2 % of 0 is 0 -- so the correction is bounded by a fraction of that speed rather than by
# an absolute number alone. At a standstill the prior is exactly 0 and the weights hand ~90 % of the
# answer to posenet, whose translation is least trustworthy there (locationd stops checking its
# health below 5 m/s at all, `locationd.py:242`), and the result lands on the same `v_ego` both
# standstill guards measure: MOONPILOT_SHOULD_STOP_SPEED in the planner's trailing gate and
# upstream's 0.3 in `should_stop`. Replaying 988 s of genuinely parked corpus through the window
# put a positive correction past 0.1 m/s on 0.05 % of frames, worst +0.578, which is enough to
# clear should-stop and take the delivered command off `CP.stopAccel`. This bound is 0 there by
# construction and inert where the feature earns its keep: 2 m/s of headroom at 20 m/s, against a
# measured moving p99 of 0.19.


class Node(NamedTuple):
  """One window sample: the prior and the odometry that closed the interval ending here.

  Forward only, which the device and car frames share, so the ingest needs no sign conversion.
  """

  mono_time: float  # s, monotonic; the odometry's pose time, which the prior is read at
  v_ego: float  # m/s, forward, wheel speed
  trans_x: float  # m/s, forward
  trans_std_x: float  # m/s


class PriorChannel:
  """One channel of the car's motion prior, sampled in time and read by interpolation.

  A publisher pushes each sample with its own message time and reads it at the *odometry's* pose
  time -- `timestampEof - MOONPILOT_SLAM_POSE_DELAY`, the exposure-stamped pose time the module
  header explains -- which is why this exists instead of reading
  the latest value off the subscription. `at` returns None rather than extrapolating: a pose time
  the samples do not span is a frame the prior cannot speak for, and the window would rather skip a
  sample than invent one.

  Interpolation is `moonpilot.lead.resample` -- the fork's one time-alignment entry point -- on a
  window short enough that its linear-on-purpose caveat does not apply: the samples are 10 ms apart,
  not the model's 2 s.
  """

  def __init__(self, window_s: float = MOONPILOT_SLAM_PRIOR_WINDOW_S, rate_hz: float = MOONPILOT_SLAM_PRIOR_RATE_HZ):
    self.window_s = float(window_s)
    self.window: deque[tuple[float, float]] = deque(maxlen=max(2, int(window_s * rate_hz) + 2))

  def push(self, mono_time: float, value: float) -> None:
    if not (math.isfinite(mono_time) and math.isfinite(value)):
      return
    self.window.append((mono_time, value))
    newest = self.window[-1][0]
    while self.window and self.window[0][0] < newest - self.window_s:
      self.window.popleft()

  def at(self, mono_time: float) -> float | None:
    if not math.isfinite(mono_time) or len(self.window) < 2:
      return None
    times = [sample[0] for sample in self.window]
    if not (times[0] <= mono_time <= times[-1]):
      return None
    return float(resample(times, [sample[1] for sample in self.window], [mono_time])[0])


def invalid(mono_time: float = 0.0) -> dict:
  """The correction a consumer may always ignore. One shape, so no caller branches on a missing key."""
  return {
    "valid": False,
    "mono_time": int(mono_time * 1e9),
    "age": 0.0,
    "dVel": 0.0,
    "corrStd": MOONPILOT_SLAM_CORR_STD_MIN,
  }


class RotatingPoseWindow:
  """The fixed-lag window: push a sample per frame, `update()` for the correction.

  `push` drops anything non-finite and anything older than the window, so the window is never
  poisoned by one bad frame and never grows: the maxlen and the time-based trim are the same rule
  twice, the time-based one handling a gap and the maxlen handling a rate faster than RATE_HZ.
  """

  def __init__(self, window_s: float = MOONPILOT_SLAM_WINDOW_S, rate_hz: float = MOONPILOT_SLAM_RATE_HZ):
    self.window_s = float(window_s)
    self.window = deque(maxlen=max(2, int(window_s * rate_hz)))

  def push(self, node: Node) -> None:
    if not all(math.isfinite(v) for v in node):
      return
    self.window.append(node)
    newest = self.window[-1].mono_time
    while self.window and self.window[0].mono_time < newest - self.window_s:
      self.window.popleft()

  def update(self) -> dict:
    """The correction at the end of the window, or an invalid one when the window cannot speak.

    Per interval between consecutive nodes: the prior's increment (`vEgo`) and the odometry's
    (`trans` x) are the same quantity in the same frame, so they are blended by inverse variance.
    `dVel` is the blend minus the prior over the window's recent end, and `corrStd` is the 1-sigma
    of the along-track position the blended increments accumulate over the whole window.
    """
    nodes = list(self.window)
    if len(nodes) < MOONPILOT_SLAM_MIN_NODES:
      return invalid(nodes[-1].mono_time if nodes else 0.0)

    t = np.array([n.mono_time for n in nodes], dtype=float)
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
      # Out-of-order or repeated timestamps: the window has no intervals to speak of.
      return invalid(t[-1])

    # The interval from node j to node j+1 carries node j's prior and node j+1's odometry: the
    # sample each source has when the interval closes.
    prior_v = np.array([n.v_ego for n in nodes[:-1]], dtype=float)
    vo = nodes[1:]
    vo_vx = np.array([n.trans_x for n in vo], dtype=float)
    # The prior's weight is the std the driver's car state is plausible to, the odometry's is
    # posenet's inflated by the multiplier locationd discounts temporally correlated noise with.
    vo_vx_std = np.maximum([n.trans_std_x for n in vo], MOONPILOT_SLAM_MIN_STD) * MOONPILOT_SLAM_TRANS_STD_MULT

    w_prior_v = 1.0 / (MOONPILOT_SLAM_RAW_V_STD * dt) ** 2
    w_vo_v = 1.0 / (vo_vx_std * dt) ** 2

    fused_v = (w_prior_v * prior_v + w_vo_v * vo_vx) / (w_prior_v + w_vo_v)
    # The variance of each blended position increment; corrStd is their accumulated 1-sigma.
    var_v = 1.0 / (w_prior_v + w_vo_v)
    corr_std = float(np.sqrt(np.sum(var_v)))

    # dVel over the last VEL_WINDOW_S of the window: the whole 5 s would report an error that has
    # since changed, and one interval would report noise.
    recent = t[1:] >= t[-1] - MOONPILOT_SLAM_VEL_WINDOW_S
    d_vel = float(np.mean((fused_v - prior_v)[recent]))

    if not (math.isfinite(d_vel) and math.isfinite(corr_std)):
      return invalid(t[-1])

    return {
      "valid": True,
      "mono_time": int(t[-1] * 1e9),
      "age": max(0.0, time.monotonic() - float(t[-1])),
      "dVel": float(np.clip(d_vel, -MOONPILOT_SLAM_MAX_CORR_VEL, MOONPILOT_SLAM_MAX_CORR_VEL)),
      "corrStd": float(np.clip(corr_std, MOONPILOT_SLAM_CORR_STD_MIN, MOONPILOT_SLAM_CORR_STD_MAX)),
    }


def fill_ego_correction(message, corr: dict) -> None:
  """Write one correction onto a `moonpilotState` message. The daemon's half of the contract."""
  field = message.moonpilotState.egoCorrection
  field.valid = corr["valid"]
  field.monoTime = corr["mono_time"]
  field.age = corr["age"]
  field.dVel = corr["dVel"]
  field.corrStd = corr["corrStd"]


def ego_speed_correction(sm, v_ego: float) -> float:
  """The published ego-speed correction in m/s, 0.0 when there is nothing to trust.

  The planner's single entry point into the correction: a correction that is absent, invalid, stale,
  non-finite or published by a dead daemon is worth exactly nothing, which is also the pass-through
  case -- adding zero is the stock planner. Trust is scaled, never amplified: `corrStd` is the
  smoother's own 1-sigma, so a confident window is applied nearly whole and a doubtful one a third of
  the way (see MOONPILOT_SLAM_CORR_STD_*). The staleness budget is MOONPILOT_SLAM_MAX_AGE plus
  MOONPILOT_SLAM_POSE_DELAY, because every correction carries the pose delay by construction.

  `v_ego` is the raw wheel speed this correction is measured against, and the bound is a fraction of
  it (MOONPILOT_SLAM_MAX_SCALE_ERR): the error being corrected is a scale error, so it is zero at a
  standstill however confident the window is. Passed in rather than read off `carState` here so the
  bound is visible at the one call site that applies it.

  getattr, not `sm.valid`: the longitudinal maneuver harness passes a plain dict as `sm`, the same
  reason moonpilot/lead.py reads the leads that way. `alive` is what covers a daemon that died with
  a fresh-looking message left in the socket: with no publisher, `alive` goes false in half a second
  and `age` never moves again.
  """
  if not (getattr(sm, "valid", {}).get("moonpilotState", False) and getattr(sm, "alive", {}).get("moonpilotState", False)):
    return 0.0

  corr = sm["moonpilotState"].egoCorrection
  if not corr.valid or corr.age > MOONPILOT_SLAM_MAX_AGE + MOONPILOT_SLAM_POSE_DELAY:
    return 0.0
  if not all(math.isfinite(v) for v in (corr.dVel, corr.corrStd, v_ego)):
    return 0.0

  trust = 1.0 / (1.0 + float(np.clip(corr.corrStd, MOONPILOT_SLAM_CORR_STD_MIN, MOONPILOT_SLAM_CORR_STD_MAX)))
  applied = float(np.clip(corr.dVel, -MOONPILOT_SLAM_MAX_CORR_VEL, MOONPILOT_SLAM_MAX_CORR_VEL)) * trust
  bound = MOONPILOT_SLAM_MAX_SCALE_ERR * max(float(v_ego), 0.0)
  return float(np.clip(applied, -bound, bound))
