"""moonpilot's rolling-window ego-speed correction: what the car's own wheel speed got wrong.

The prior is `carState`'s wheel speed -- what the car believes about its own speed. The observation
is `cameraOdometry`'s forward translation rate -- what the model saw move, with its std. Each
interval of a fixed-lag window blends the two by inverse variance, and the published `dVel` is the
blend minus the raw prior over the window's recent end; a consumer that ignores it (`valid` false)
reduces to exactly the behavior it had before. That is the whole mechanism: numpy and the signals
already on the bus, no new vision frontend, no new service, so it runs on the device's own CPU.
`moonpilot/leadd.py` publishes it on `moonpilotState.egoCorrection`.

Two things about it are not obvious.

- **What the window buys is lag, not accumulation.** With independent per-tick errors, inverse-variance weights are
  constant; correlated wheel-speed scale error is what the window estimates.
- **The prior is read at the odometry's pose time.** `cameraOdometry` describes the pose at
  `timestampEof - CAM_ODO_POSE_DELAY`; `leadd` stamps nodes from `timestampEof`, and `PriorChannel`
  interpolates wheel speed from its message times without extrapolating. Replay measurements are in `docs/slam.md`.

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
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.locationd.helpers import rotate_std
from openpilot.selfdrive.locationd.locationd import (
  CALIB_RPY_SANITY_CHECK,
  CAM_ODO_POSE_DELAY,
  CAM_ODO_TRANS_STD_MULT,
  MIN_STD_SANITY_CHECK,
)

from moonpilot.state import moonpilot_state

# Starting points, fork-owned. Tune against logs: plot `egoCorrection` against `cameraOdometry` and the
# wheel speed in PlotJuggler, and watch how often `valid` falls out.
MOONPILOT_SLAM_WINDOW_S = 5.0  # s; how long an estimate of a motion error may lag
MOONPILOT_SLAM_RATE_HZ = 20.0  # nominal cameraOdometry sample rate
MOONPILOT_SLAM_MIN_NODES = 10
# How wrong the prior may be, in the same per-tick currency as the odometry stds. 1 m/s is ~5 % at
# highway speed: the tyre wear, the grade and the slip a wheel-speed scale error is made of. Larger
# means the correction leans further onto the model's odometry.
MOONPILOT_SLAM_RAW_V_STD = 1.0  # m/s
MOONPILOT_SLAM_PRIOR_WINDOW_S = 1.0  # s of prior samples kept
MOONPILOT_SLAM_MAX_CORR_VEL = 2.0  # m/s; hard clamp on |dVel|
MOONPILOT_SLAM_MAX_AGE = 0.30  # s; transport budget, excluding pose delay
MOONPILOT_SLAM_VEL_WINDOW_S = 1.0
# Trust scale bounds, in m: near the floor is nearly full trust and at the ceiling it is a third.
MOONPILOT_SLAM_CORR_STD_MIN = 0.02
MOONPILOT_SLAM_CORR_STD_MAX = 2.0
MOONPILOT_SLAM_MAX_SCALE_ERR = 0.1  # fraction of the speed being corrected, and the low-speed bound
# Parked and moving-route measurements are documented in `docs/slam.md`.

class OdometrySample(NamedTuple):
  mono_time: float
  v_ego: float | None
  trans: np.ndarray
  trans_std: np.ndarray
  rot: np.ndarray

def device_from_calib(rpy_calib, valid=True):
  if not valid:
    return None
  rpy_calib = np.asarray(rpy_calib, dtype=float)
  if rpy_calib.shape != (3,) or not np.all(np.isfinite(rpy_calib)) or np.any(np.abs(rpy_calib) > CALIB_RPY_SANITY_CHECK):
    return None
  return None if not np.any(rpy_calib) else rot_from_euler(rpy_calib)


def odometry_sample(odometry, prior: "PriorChannel", device_from_calib=None, valid=True) -> OdometrySample | None:
  """Validate and align one camera-odometry message to its pose-time wheel-speed sample."""
  if not valid or min(len(odometry.trans), len(odometry.transStd), len(odometry.rot)) < 3:
    return None
  mono_time = odometry.timestampEof * 1e-9 - CAM_ODO_POSE_DELAY
  trans = np.asarray(odometry.trans, dtype=float)[:3]
  trans_std = np.asarray(odometry.transStd, dtype=float)[:3]
  if device_from_calib is not None:
    trans = device_from_calib @ trans
    trans_std = rotate_std(device_from_calib, trans_std)
  return OdometrySample(mono_time, prior.at(mono_time), trans, trans_std, np.asarray(odometry.rot, dtype=float)[:3])


def slam_node(sample: OdometrySample) -> "Node | None":
  if sample.v_ego is None:
    return None
  return Node(sample.mono_time, sample.v_ego, float(sample.trans[0]), float(sample.trans_std[0]))


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

  The pose-time alignment is described in the module docstring. `at` returns None rather than
  extrapolating when samples do not span the requested time.
  """

  def __init__(self, window_s: float = MOONPILOT_SLAM_PRIOR_WINDOW_S):
    self.window_s = float(window_s)
    self.window: deque[tuple[float, float]] = deque()

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
    return float(np.interp(mono_time, times, [sample[1] for sample in self.window]))


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
  """The fixed-lag window: push a sample per frame, `update()` for the correction."""
  def __init__(self, window_s: float = MOONPILOT_SLAM_WINDOW_S):
    self.window_s = float(window_s)
    self.window = deque()

  def push(self, node: Node) -> None:
    if not all(math.isfinite(v) for v in node):
      return
    self.window.append(node)
    newest = self.window[-1].mono_time
    while self.window and self.window[0].mono_time < newest - self.window_s:
      self.window.popleft()

  def update(self) -> dict:
    """The correction at the end of the window, or an invalid one when the window cannot speak."""
    nodes = list(self.window)
    if len(nodes) < MOONPILOT_SLAM_MIN_NODES:
      return invalid(nodes[-1].mono_time if nodes else 0.0)

    t = np.array([n.mono_time for n in nodes], dtype=float)
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0.0):

      return invalid(t[-1])

    # The interval from node j to node j+1 carries node j's prior and node j+1's odometry: the
    # sample each source has when the interval closes.
    prior_v = np.array([n.v_ego for n in nodes[:-1]], dtype=float)
    vo = nodes[1:]
    vo_vx = np.array([n.trans_x for n in vo], dtype=float)
    # The prior's weight is the std the driver's car state is plausible to, the odometry's is
    # posenet's inflated by the multiplier locationd discounts temporally correlated noise with.
    vo_vx_std = np.maximum([n.trans_std_x for n in vo], MIN_STD_SANITY_CHECK) * CAM_ODO_TRANS_STD_MULT

    w_prior_v = 1.0 / (MOONPILOT_SLAM_RAW_V_STD * dt) ** 2
    w_vo_v = 1.0 / (vo_vx_std * dt) ** 2

    fused_v = (w_prior_v * prior_v + w_vo_v * vo_vx) / (w_prior_v + w_vo_v)

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
  CAM_ODO_POSE_DELAY, because every correction carries the pose delay by construction.

  `v_ego` is the raw wheel speed this correction is measured against, and the bound is a fraction of
  it (MOONPILOT_SLAM_MAX_SCALE_ERR): the error being corrected is a scale error, so it is zero at a
  standstill however confident the window is. Passed in rather than read off `carState` here so the
  bound is visible at the one call site that applies it.


  """
  state = moonpilot_state(sm)
  if state is None:
    return 0.0

  corr = state.egoCorrection
  if not corr.valid or corr.age > MOONPILOT_SLAM_MAX_AGE + CAM_ODO_POSE_DELAY:
    return 0.0
  if not all(math.isfinite(v) for v in (corr.dVel, corr.corrStd, v_ego)):
    return 0.0

  trust = 1.0 / (1.0 + float(np.clip(corr.corrStd, MOONPILOT_SLAM_CORR_STD_MIN, MOONPILOT_SLAM_CORR_STD_MAX)))
  applied = float(np.clip(corr.dVel, -MOONPILOT_SLAM_MAX_CORR_VEL, MOONPILOT_SLAM_MAX_CORR_VEL)) * trust
  bound = MOONPILOT_SLAM_MAX_SCALE_ERR * max(float(v_ego), 0.0)
  return float(np.clip(applied, -bound, bound))
