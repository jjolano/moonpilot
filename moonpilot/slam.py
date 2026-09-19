"""moonpilot's rolling-window ego-motion correction: what the car's own motion prior got wrong.

The prior is `carState`'s wheel speed and `deviceMotion`'s gyro -- what the car believes about
itself. The observation is `cameraOdometry` -- what the model saw move, with per-axis stds.
Both are integrated forward over a fixed-lag window and blended per axis by inverse variance; the
published correction is the blended chain minus the raw one at the window end, and a consumer that
ignores it (`valid` false) reduces to exactly the behavior it had before. That is the whole
mechanism: numpy and the signals already on the bus, no new vision frontend, no new service, so it
runs on the device's own CPU. `moonpilot/leadd.py` publishes it on `moonpilotState.egoCorrection`.

Four things about it are not obvious.

- **The window is re-anchored at its start on every publish**, so the estimate at the end depends on
  the whole window and the oldest data leaves it -- that is the "fixed lag", and it is what makes
  this a smoother rather than a filter that never forgets. The end node is also the only thing
  published, and a Rauch-Tung-Striebel pass equals the filter at that node, so a forward pass over
  the window *is* the smoother: there is no backward pass to write.
- **What the window buys is lag, not accumulation.** With independent per-tick errors, inverse-
  variance weights are constant and the window length would not matter. Errors that are *not*
  independent are what matters, and they are exactly the error this exists to correct: a wheel-speed
  scale error (a 2 % error at 20 m/s is 0.4 m/s of the planner's own input, 1.2 m of gap over a 3 s
  approach). The window is how long an estimate of that is allowed to lag.
- **The prior's yaw rate is the gyro, not `carState.yawRate`, which is why `deviceMotion` is
  subscribed at all.** Only ford, psa and volkswagen assign `carState.yawRate` and nothing in
  `openpilot/` reads it, so on Toyota/Lexus -- the fork's own target -- and on Honda it is exactly
  0.0. A zero prior yaw rate would never turn the raw chain, leaving `dYaw` at ~0 through a real
  corner and folding the corner's lateral travel into `dPos` as though it were along-track. The
  gyro (`deviceMotion.angularVelocityDevice`, always published, `valid` from the filter) is the
  signal that exists; it is device-frame, so its `z` is down-positive and the ingest negates it into
  the car's left-positive yaw.
- **The prior is read at the odometry's pose time, not at the frame's own.** `cameraOdometry`'s
  `trans`/`rot` describe the pose as of `logMonoTime - MOONPILOT_SLAM_POSE_DELAY` (locationd's own
  `CAM_ODO_POSE_DELAY`, and the reason it computes its observation time that way). Pairing "latest
  odometry" with "latest carState" instead puts an `a * 0.1 s` error on every interval -- 0.15-0.35
  m/s while braking, the same order as the signal this file measures and correlated with exactly the
  approach regime the planner's gap terms live in. So `PriorChannel` samples the two channels with
  their own message times and interpolates both onto that third time, and a frame whose pose time
  the prior cannot span is skipped rather than extrapolated.

Two approximations are stage-1 on purpose. **The device frame is still treated as the car frame**:
for each newly ingested pose, trusted `extrinsicsCalibration.rpyCalib` rotates `trans`/`rot` from
the calibration frame into the device frame with `rot_from_euler`, and rotates their stds with
`rotate_std`; invalid, absent, malformed or out-of-bound calibration leaves that transform as
identity. This uses the latest calibration even though the pose is 0.1 s older, which is acceptable
while calibration converges over minutes, and a calibration change affects only new nodes rather than
re-deriving older window samples. **The outputs are clamped**, which means the two sources disagree
by more than a sensor plausibly can; that is a tuning question the constants below own.

Only `dVel` has a consumer today, and it is the one the planner can act on: the fork planner's gaps
are radar-measured, so shifting them by an odometry position offset would inject error rather than
remove it. `dPos` is the pose-level statement of the same disagreement, published for the log and
for a consumer that integrates ego motion.
"""

import math
import time
from collections import deque
from typing import NamedTuple

import numpy as np

from moonpilot.lead import resample

# Starting points, fork-owned. Tune against logs: plot `egoCorrection` against `deviceMotion` and the
# wheel speed in PlotJuggler, and watch how often `valid` falls out.
MOONPILOT_SLAM_WINDOW_S = 5.0  # s; how long an estimate of a motion error may lag
MOONPILOT_SLAM_RATE_HZ = 20.0  # window nodes per second; cameraOdometry and carState both run here
MOONPILOT_SLAM_MIN_NODES = 10  # below this the window says nothing, so it says nothing
# How wrong the prior may be, in the same per-tick currency as the odometry stds. 1 m/s is ~5 % at
# highway speed: the tyre wear, the grade and the slip a wheel-speed scale error is made of. Larger
# means the correction leans further onto the model's odometry.
MOONPILOT_SLAM_RAW_V_STD = 1.0  # m/s
MOONPILOT_SLAM_RAW_YAW_STD = 0.02  # rad/s
# Mirrors of locationd's own numbers, pinned by test_slam: the stds it trusts, and the floors it
# refuses below. The multipliers are how locationd discounts temporally correlated odometry noise.
MOONPILOT_SLAM_MIN_STD = 1e-5  # m/s or rad/s; locationd.MIN_STD_SANITY_CHECK
MOONPILOT_SLAM_TRANS_STD_MULT = 4  # locationd.CAM_ODO_TRANS_STD_MULT
MOONPILOT_SLAM_ROT_STD_MULT = 10  # locationd.CAM_ODO_ROT_STD_MULT
# s; locationd.CAM_ODO_POSE_DELAY. cameraOdometry's trans/rot describe the pose as of this long
# before the message, and the prior is read there so both sides of an interval cover the same span.
MOONPILOT_SLAM_POSE_DELAY = 0.1
MOONPILOT_SLAM_PRIOR_WINDOW_S = 1.0  # s of prior samples kept; the pose time is 0.1 s behind at most
MOONPILOT_SLAM_PRIOR_RATE_HZ = 100.0  # nominal carState rate, the faster of the two prior channels
MOONPILOT_SLAM_MAX_CORR_POS = 3.0  # m; hard clamp on |dPos|
MOONPILOT_SLAM_MAX_CORR_VEL = 2.0  # m/s; hard clamp on |dVel|
MOONPILOT_SLAM_MAX_CORR_YAW = 0.05  # rad; hard clamp on |dYaw|
MOONPILOT_SLAM_MAX_AGE = 0.30  # s; the transport budget, on top of MOONPILOT_SLAM_POSE_DELAY: every
# correction is that pose delay old by construction, and charging the consumer's budget for it would
# spend a third of the staleness allowance on the sensor's own lag.
MOONPILOT_SLAM_VEL_WINDOW_S = 1.0  # s of the window dVel speaks for; the whole window lags a change
MOONPILOT_SLAM_CORR_STD_MIN = 0.02  # m; 1/(1 + corrStd) is the trust a consumer scales in, so the
MOONPILOT_SLAM_CORR_STD_MAX = 2.0  # floor is nearly full trust and the ceiling is a third of it


class Node(NamedTuple):
  """One window sample: the prior and the odometry that closed the interval ending here.

  The car frame throughout -- x forward, y left, yaw left-positive -- which is the frame
  `_integrate` and the zero-lateral prior are written in. The odometry arrives in the device frame,
  where y is right and z is down, so a publisher negates both on ingest rather than letting a left
  turn cancel itself against a right-positive odometry yaw rate.
  """

  mono_time: float  # s, monotonic; the odometry's pose time, which the prior is read at
  v_ego: float  # m/s, forward, wheel speed
  yaw_rate: float  # rad/s, + left, from the device's gyro
  trans_x: float  # m/s, forward
  trans_y: float  # m/s, + left
  rot_z: float  # rad/s, + left
  trans_std_x: float  # m/s
  rot_std_z: float  # rad/s


class PriorChannel:
  """One channel of the car's motion prior, sampled in time and read by interpolation.

  A publisher pushes each sample with its own message time and reads it at the *odometry's* pose
  time -- `logMonoTime - MOONPILOT_SLAM_POSE_DELAY` -- which is why this exists instead of reading
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
    "dPos": 0.0,
    "dVel": 0.0,
    "dYaw": 0.0,
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

    Per interval between consecutive nodes: the prior's increment (`vEgo`, the gyro's yaw rate) and
    the odometry's increment (`trans`, `rot`) are the same quantity in the same frame, so they are
    blended by inverse variance, axis by axis. The prior has no lateral evidence -- it says the car
    goes where its heading points -- so the lateral channel is the odometry alone. The blended
    increments are then integrated into a second chain beside the raw one, and the published
    correction is the difference at the end, rotated into the car's frame as it is now.
    """
    nodes = list(self.window)
    if len(nodes) < MOONPILOT_SLAM_MIN_NODES:
      return invalid(nodes[-1].mono_time if nodes else 0.0)

    t = np.array([n.mono_time for n in nodes], dtype=float)
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):
      # Out-of-order or repeated timestamps: the window has no chain to speak of.
      return invalid(t[-1])

    # The interval from node j to node j+1 carries node j's prior and node j+1's odometry: the
    # sample each source has when the interval closes.
    prior_v = np.array([n.v_ego for n in nodes[:-1]], dtype=float)
    prior_w = np.array([n.yaw_rate for n in nodes[:-1]], dtype=float)
    vo = nodes[1:]
    vo_vx = np.array([n.trans_x for n in vo], dtype=float)
    vo_vy = np.array([n.trans_y for n in vo], dtype=float)
    vo_w = np.array([n.rot_z for n in vo], dtype=float)
    # The prior's weights are the stds the driver's car state is plausible to, the odometry's are
    # posenet's inflated by the multipliers locationd discounts temporally correlated noise with.
    vo_vx_std = np.maximum([n.trans_std_x for n in vo], MOONPILOT_SLAM_MIN_STD) * MOONPILOT_SLAM_TRANS_STD_MULT
    vo_w_std = np.maximum([n.rot_std_z for n in vo], MOONPILOT_SLAM_MIN_STD) * MOONPILOT_SLAM_ROT_STD_MULT

    w_prior_v = 1.0 / (MOONPILOT_SLAM_RAW_V_STD * dt) ** 2
    w_prior_w = 1.0 / (MOONPILOT_SLAM_RAW_YAW_STD * dt) ** 2
    w_vo_v = 1.0 / (vo_vx_std * dt) ** 2
    w_vo_w = 1.0 / (vo_w_std * dt) ** 2

    fused_v = (w_prior_v * prior_v + w_vo_v * vo_vx) / (w_prior_v + w_vo_v)
    fused_w = (w_prior_w * prior_w + w_vo_w * vo_w) / (w_prior_w + w_vo_w)
    # The variance of each blended increment, which is the whole of corrStd: the accumulated
    # position error of a chain of weighted increments (yaw's contribution to it is second order and
    # left out).
    var_v = 1.0 / (w_prior_v + w_vo_v)

    fused_yaw = np.cumsum(fused_w * dt)
    raw_yaw = np.cumsum(prior_w * dt)
    fused_x, fused_y = _integrate(fused_yaw, fused_v * dt, vo_vy * dt)
    raw_x, raw_y = _integrate(raw_yaw, prior_v * dt, np.zeros_like(prior_v))

    # The correction in the car's frame as it is now, rather than in the window's start frame the two
    # chains were anchored in.
    yaw = float(raw_yaw[-1])
    dx, dy = float(fused_x[-1] - raw_x[-1]), float(fused_y[-1] - raw_y[-1])
    d_pos = math.cos(yaw) * dx + math.sin(yaw) * dy

    # dVel over the last VEL_WINDOW_S of the window: the whole 5 s would report an error that has
    # since changed, and one interval would report noise.
    recent = t[1:] >= t[-1] - MOONPILOT_SLAM_VEL_WINDOW_S
    d_vel = float(np.mean((fused_v - prior_v)[recent]))

    values = (d_pos, d_vel, float(fused_yaw[-1] - raw_yaw[-1]), float(np.sqrt(np.sum(var_v))))
    if not all(math.isfinite(v) for v in values):
      return invalid(t[-1])

    d_pos, d_vel, d_yaw, corr_std = values
    return {
      "valid": True,
      "mono_time": int(t[-1] * 1e9),
      "age": max(0.0, time.monotonic() - float(t[-1])),
      "dPos": float(np.clip(d_pos, -MOONPILOT_SLAM_MAX_CORR_POS, MOONPILOT_SLAM_MAX_CORR_POS)),
      "dVel": float(np.clip(d_vel, -MOONPILOT_SLAM_MAX_CORR_VEL, MOONPILOT_SLAM_MAX_CORR_VEL)),
      "dYaw": float(np.clip(d_yaw, -MOONPILOT_SLAM_MAX_CORR_YAW, MOONPILOT_SLAM_MAX_CORR_YAW)),
      "corrStd": float(np.clip(corr_std, MOONPILOT_SLAM_CORR_STD_MIN, MOONPILOT_SLAM_CORR_STD_MAX)),
    }


def _integrate(yaw: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Body-frame increments rolled into the chain's start frame, one heading per increment.

  No loop and no matrix: the heading each increment is rotated by is the running sum of the heading
  increments, which is known before the positions are.
  """
  cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
  return np.cumsum(cos_yaw * dx - sin_yaw * dy), np.cumsum(sin_yaw * dx + cos_yaw * dy)


def fill_ego_correction(message, corr: dict) -> None:
  """Write one correction onto a `moonpilotState` message. The daemon's half of the contract."""
  field = message.moonpilotState.egoCorrection
  field.valid = corr["valid"]
  field.monoTime = corr["mono_time"]
  field.age = corr["age"]
  field.dPos = corr["dPos"]
  field.dVel = corr["dVel"]
  field.dYaw = corr["dYaw"]
  field.corrStd = corr["corrStd"]


def ego_speed_correction(sm) -> float:
  """The published ego-speed correction in m/s, 0.0 when there is nothing to trust.

  The planner's single entry point into the correction: a correction that is absent, invalid, stale,
  non-finite or published by a dead daemon is worth exactly nothing, which is also the pass-through
  case -- adding zero is the stock planner. Trust is scaled, never amplified: `corrStd` is the
  smoother's own 1-sigma, so a confident window is applied nearly whole and a doubtful one a third of
  the way (see MOONPILOT_SLAM_CORR_STD_*). The staleness budget is MOONPILOT_SLAM_MAX_AGE plus
  MOONPILOT_SLAM_POSE_DELAY, because every correction carries the pose delay by construction.

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
  if not all(math.isfinite(v) for v in (corr.dVel, corr.corrStd)):
    return 0.0

  trust = 1.0 / (1.0 + float(np.clip(corr.corrStd, MOONPILOT_SLAM_CORR_STD_MIN, MOONPILOT_SLAM_CORR_STD_MAX)))
  return float(np.clip(corr.dVel, -MOONPILOT_SLAM_MAX_CORR_VEL, MOONPILOT_SLAM_MAX_CORR_VEL)) * trust
