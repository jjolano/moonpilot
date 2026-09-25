"""The fork's curve speed control: what speed the model's *predicted* path is worth taking.

`moonpilot/longitudinal.py` answers "what acceleration?" from the car's measured state — `cruise_accel`
caps positive accel by the lateral accel the *measured* steer angle implies, so a curve the car is
already in limits throttle, and nothing anywhere slows the car for a curve it can see coming. This
module is the path half of that: three bounded braking terms, all fed back as candidates into
`policy`'s existing `min`, so every one of them can only ever *add* braking to whichever of the
deterministic or model terms was governing.

  - **`curve_accel`, the pre-brake.** The kinematic term the stopping floor uses, against the curve's
    own distance: the decel that arrives at the curve at the speed the curve is worth taking. It reads
    the whole admitted span of the path and takes the deepest point, so the nearest binding sample
    wins.
  - **the jerk ceiling folded into `curve_targets`.** Lateral jerk at constant speed is
    ``v^3 * |dk/ds|``, so a ceiling on jerk *is* a ceiling on speed — and it is what makes a
    fast-tightening entry slow earlier rather than harder, which is the whole reason a preview beats a
    measured lateral-accel limit.
  - **`lat_accel_hold`, the in-curve regulator.** Once the car is inside the curve there is no distance
    to an event left, so this one is proportional on the lateral accel the car is *actually* pulling —
    from the measured steer angle and the vehicle model — against the speed that accel is worth. It is
    the one term that does not need the model's path arrays at all.

Every term is inert unless the model message carries path arrays (`orientationRate` and a path speed)
and, for the in-curve term, `vehicleParameters.valid` — which paramsd only sets once its own sensors,
angle offset and roll are trustworthy, and which is false in a bare capnp message. That is what keeps
upstream's maneuver plant (it fills `position` and `velocity` but never `orientationRate`) and every
existing fork test exactly as they were, feature on or off.

The bank correction is upstream's own, copied from `clip_curvature` rather than re-derived: the
achievable lateral accel band is shifted by ``g * roll``, so a banked left-hander is worth more and a
right-hander less by the same amount. None of these numbers are upstream's, and they are all
fork-owned starting points — the header constants carry the unit and the reason.

`LatAccelBiasEstimator` is the per-car calibration, and it is one-sided on purpose: `applied()` may
only ever return a scale of 1.0 or more. A car that realizes *more* lateral accel than its path
predicted is in a tighter curve than the model thought and should slow more; the opposite direction
would hand the feature *less* braking than the geometry justifies, which is the one direction this
cannot be allowed to fail in.
"""

import math
from collections import deque
from typing import NamedTuple

import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.interfaces import ACCEL_MAX

from moonpilot.lead import resample  # the fork's one interpolation entry point, shared rather than re-derived

MOONPILOT_CURVE_T_IDX = np.array(ModelConstants.T_IDXS)  # the model path's own time grid
MOONPILOT_CURVE_A_LAT = 1.7  # m/s^2; lateral accel a curve is worth taking at
MOONPILOT_CURVE_A_LAT_MIN = 1.0  # m/s^2; floor after the bank correction, so a large roll cannot zero the budget
MOONPILOT_CURVE_J_LAT = 3.0  # m/s^3; lateral jerk the entry is shaped to, below upstream's 5.0 ISO command limit
MOONPILOT_CURVE_JERK_STEP = 2.0  # m; the length the jerk ceiling's |dk/ds| is measured over, rather than
# between the model's own samples. The model's spacing runs 0.1 m to 12 m along one path, so a
# sample-scale gradient's value depends on where it happens to sit, and a single sample can carry a step
# the trend around it does not have. Measured over 143,058 corpus frames, the frames where only this
# ceiling asks for braking split by exactly that: the 33 % this window drops sit at a median 0.47 m
# spacing with incoherent steps — net over summed |dk| 0.42, and |k| at 0.6x its own +/-5 m
# neighbourhood — while the 67 % it keeps are coherent ramps at 1.7 m spacing with |k| at 1.7x its
# neighbourhood, at full depth. 5 m dropped coherent ramps too (the dropped set's coherence median 1.0),
# which is why this is sized to the sub-metre class rather than to the ramp the term exists for. The
# derivative is a centered +/-STEP/2 difference clamped to the path's own ends and divided by the span
# it actually used, so a linear ramp reads its own slope exactly wherever the window fits inside it,
# and the cost is one-way: only structure narrower than the window reads shallow.
MOONPILOT_CURVE_PREVIEW_T = 4.0  # s of path admitted; past this the prediction is not worth braking on
MOONPILOT_CURVE_PATH_MAX_AGE = 2 * DT_MDL  # s; how old the path may be before `moonpilot/longitudinal.py`
# stops re-referencing it. The path's `position.x` is measured from the pose of the frame the model
# saw, so the planner passes the car's travel since that frame as `curve_targets`' `ahead` — the age at
# the planner tick that consumes the message, not its publish stamp: msgq buffers, so `recv_time` is set
# when the planner actually reads it. Measured over 282,554 corpus plans, the publish lag is 31 ms
# median and the interval from that publish to the plan's own send another 15.9 ms (21.4 p95) against
# a 0.58 ms solve, so the tick anchor is worth up to ~0.48 m more at 30 m/s. Two model periods, the same
# bound `moonpilot/curvature.py` puts on the age of this same message: past it the shift is dropped
# rather than extrapolated, which leaves the planner this fork had before the correction — the failure
# direction that only brakes less.
MOONPILOT_CURVE_PATH_CLOCK_SANITY = 10.0  # s; an age past this is a clock that does not line up with
# the message's stamps — a replay reads wall time against a log — not a late planner, so the consumer
# falls back to the publish lag, which the stamps alone carry, instead of the zero shift a mismatched
# clock would otherwise give.
MOONPILOT_CURVE_V_MIN = 5.0  # m/s; floor on any target speed, so a spurious curvature cannot ask for a stop
MOONPILOT_CURVE_ACCEL_MIN = -1.5  # m/s^2; the terms' shared floor, well above ACCEL_MIN
MOONPILOT_CURVE_MIN_SLACK = 1.0  # m; floor on the braking-distance denominator
MOONPILOT_CURVE_MIN_PATH_SPEED = 1.0  # m/s; floor under the path speed the curvature is divided by
MOONPILOT_CURVE_K_HOLD = 0.6  # 1/s on the in-curve speed error
MOONPILOT_CURVE_HOLD_MARGIN = 1.1  # slack on the budget before the in-curve term engages at all
MOONPILOT_CURVE_HOLD_MIN_SPEED = 5.0  # m/s; below it the steering angle implies curvatures no plan should chase
MOONPILOT_CURVE_HOLD_MIN_CURVATURE = 1e-4  # 1/m (10 km radius); below it the measurement is noise
MOONPILOT_CURVE_BIAS_T = 1.0  # s between a prediction and the measurement it is scored against
MOONPILOT_CURVE_BIAS_RC = 20.0  # s time constant of the ratio filter
MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL = 1.0  # m/s^2; below this both signals are noise
MOONPILOT_CURVE_BIAS_TRACKING_TOLERANCE = 0.5  # m/s^2; allow normal torque-tracking noise without learning saturation as model bias
MOONPILOT_CURVE_BIAS_MIN_SPEED = 10.0  # m/s
MOONPILOT_CURVE_BIAS_MIN = 0.7  # clamp on a single ratio sample
MOONPILOT_CURVE_BIAS_MAX = 1.4
MOONPILOT_CURVE_BIAS_MIN_SAMPLES = 200  # paired frames before the scale is applied, 10 s of curve
MOONPILOT_CURVE_BIAS_PERSIST_EVERY = 1200  # frames between param writes, 60 s at DT_MDL
MOONPILOT_CURVE_BIAS_KEY = "MoonpilotCurveLatScale"


def lat_accel_budget(direction, roll, scale=1.0):
  """The lateral acceleration a curve in `direction` (+1 left, -1 right) is worth taking at.

  The bank is credited exactly the way `clip_curvature` credits it (`drive_helpers.py:36-39`): the
  achievable band is shifted by `g * roll`, so a positive roll adds budget to a left turn and takes
  it from a right one. `scale` is the learned realization bias and only ever divides — see
  `LatAccelBiasEstimator` for why it is clamped at 1.0.
  """
  budget = MOONPILOT_CURVE_A_LAT + np.asarray(direction, dtype=float) * ACCELERATION_DUE_TO_GRAVITY * float(roll)
  return np.maximum(budget, MOONPILOT_CURVE_A_LAT_MIN) / max(float(scale), 1.0)


class CurveTarget(NamedTuple):
  """Where the path is worth slowing to: `x` in meters ahead of the car now, `v` in m/s there.
  `curve_targets` re-references the model's own `position.x` to the car by the travel the caller
  passes as `ahead`, so every consumer compares it against plain ego travel."""

  x: np.ndarray
  v: np.ndarray


def jerk_ceiling_speed(curv, x) -> np.ndarray:
  """The speed each path sample is worth under `v^3 * |dk/ds| <= MOONPILOT_CURVE_J_LAT`.

  `|dk/ds|` is a centered difference over `MOONPILOT_CURVE_JERK_STEP` meters rather than between the
  model's own samples, because the pre-brake takes the *deepest* sample of the window and the
  sample-scale gradient hands that choice to whichever sample is noisiest. Measured over 143,058
  corpus frames, the frames where only this ceiling asks for braking (741, 0.52 %) split by exactly
  that: the 33 % the window drops sit at a median 0.47 m spacing with incoherent steps (net over
  summed |dk| 0.42, |k| at 0.6x its +/-5 m neighbourhood) — a single sample's step, which no road has
  and the model's own 0.1 s sampling cannot have resolved — while the 67 % it keeps are coherent
  ramps at 1.7 m spacing with |k| at 1.7x its neighbourhood, at full depth. A linear ramp reads its
  own slope exactly wherever the window fits inside it, so what the measurement costs is only
  structure narrower than the window, which is one-way: shallower, never deeper.

  Unconstrained (`inf`) wherever the derivative cannot speak: fewer than two usable samples, or a
  degenerate span — a path is left alone rather than braked on a NaN. At the ends the window shrinks
  to what the path has and the difference is divided by that shorter span, so the slope stays
  unbiased rather than reading half of itself.
  """
  curv = np.asarray(curv, dtype=float)
  x = np.asarray(x, dtype=float)
  v_jerk = np.full(x.shape, np.inf)
  finite = np.isfinite(x) & np.isfinite(curv)
  if finite.sum() < 2:
    return v_jerk
  xs, cs = x[finite], curv[finite]
  order = np.argsort(xs)
  xs, cs = xs[order], cs[order]
  half = 0.5 * MOONPILOT_CURVE_JERK_STEP
  left = np.maximum(xs[0], xs - half)
  right = np.minimum(xs[-1], xs + half)
  dk_ds = np.abs(resample(xs, cs, right) - resample(xs, cs, left)) / np.maximum(right - left, 1e-9)
  v_jerk[finite] = np.cbrt(MOONPILOT_CURVE_J_LAT / np.maximum(dk_ds, 1e-9))
  return v_jerk


def curve_targets(model, allowed, roll=0.0, scale=1.0, ahead=0.0) -> CurveTarget | None:
  """The speed the model's own path is worth taking, per sample, or None for no candidate.

  Curvature is `orientationRate.z / velocity.x` over the model's own time grid, which is the same
  pair upstream's `get_curvature_from_plan` reads. The budget is the bank-corrected lateral accel for
  the direction the path turns; the jerk ceiling is the same budget expressed as a speed, since
  lateral jerk at constant speed is `v^3 * |dk/ds|` — and it is what makes a fast-tightening entry
  slow *earlier* rather than harder, the one thing a measured lateral-accel limit cannot see.

  Only the derivative is windowed. The budget stays per-sample on purpose, and that is measured
  rather than assumed: over the corpus's 5,286 budget-binding frames the binding sample's |k| is
  1.00x the median of its own +/-2 m neighbourhood (1.02x at +/-5 m, above 1.5x on 0.2 % of them),
  because the deepest ask comes from the *nearest* binding sample of a sustained curve — which is
  what the budget is for — rather than from a single sample's step.

  None means no candidate, which is the planner this fork had before any of this existed — a model
  message with no path arrays takes that branch, which is why upstream's maneuver plant (it fills
  `position` and `velocity` but never `orientationRate`) and every existing fork test are unaffected
  whether the feature is on or off.

  `ahead` is the distance the car has already covered since the frame this path was predicted from:
  `position.x` is measured from that pose, so subtracting the travel the caller can account for is
  what makes the returned `x` meters ahead of the car now — the reading every distance comparison
  against it assumes, and the reason both the command and the published rollout pass plain ego travel
  as `x_ego`. Zero is the no-correction case, which is what an absent, future-dated or stale stamp
  has to give rather than an extrapolated one.
  """
  if not allowed:
    return None
  ahead = float(ahead)
  if not math.isfinite(ahead) or ahead <= 0.0:
    ahead = 0.0
  x = np.asarray(model.position.x, dtype=float)
  psi_rate = np.asarray(model.orientationRate.z, dtype=float)
  v_path = np.asarray(model.velocity.x, dtype=float)
  n = min(len(x), len(psi_rate), len(v_path), len(MOONPILOT_CURVE_T_IDX))
  if n == 0:
    return None
  keep = MOONPILOT_CURVE_T_IDX[:n] <= MOONPILOT_CURVE_PREVIEW_T
  x, psi_rate, v_path = x[:n][keep], psi_rate[:n][keep], v_path[:n][keep]
  curv = psi_rate / np.maximum(v_path, MOONPILOT_CURVE_MIN_PATH_SPEED)
  budget = lat_accel_budget(np.sign(curv), roll, scale)
  v_accel = np.sqrt(budget / np.maximum(np.abs(curv), 1e-6))
  v_target = np.maximum(np.minimum(v_accel, jerk_ceiling_speed(curv, x)), MOONPILOT_CURVE_V_MIN)
  finite = np.isfinite(x) & np.isfinite(v_target)
  if not finite.any():
    return None
  return CurveTarget(x[finite] - ahead, v_target[finite])


def curve_accel(v_ego, x_ego, curve) -> float:
  """The pre-brake: the decel that arrives at the deepest binding curve sample at its own target
  speed. `ACCEL_MAX` is the inactive sentinel — a straight path, a target above the current speed, or
  every point already behind the car — so `min` leaves the cruise term exactly as it was."""
  if curve is None:
    return ACCEL_MAX
  d = curve.x - x_ego
  binding = (d > 0.0) & (curve.v < v_ego)
  if not binding.any():
    return ACCEL_MAX
  slack = np.maximum(d[binding], MOONPILOT_CURVE_MIN_SLACK)
  a = (curve.v[binding] ** 2 - v_ego**2) / (2.0 * slack)
  return float(max(a.min(), MOONPILOT_CURVE_ACCEL_MIN))


def hold_speed(measured_curvature, v_ego, budget, standstill, params_valid) -> float:
  """The in-curve term's setpoint: the speed the measured curvature is worth at `budget`, with the
  hold margin's slack on it. `math.inf` whenever the measurement cannot speak — a standstill, a
  paramsd that has not validated its own calibration, a speed too low for the steer angle to mean
  anything, or a curvature at the noise floor."""
  if standstill or not params_valid or v_ego <= MOONPILOT_CURVE_HOLD_MIN_SPEED:
    return math.inf
  curv = abs(float(measured_curvature))
  if not math.isfinite(curv) or curv < MOONPILOT_CURVE_HOLD_MIN_CURVATURE:
    return math.inf
  return math.sqrt(MOONPILOT_CURVE_HOLD_MARGIN * budget / curv)


def lat_accel_hold(v_ego, v_hold) -> float:
  """The in-curve regulator: proportional, because there is no distance to an event when the car is
  already in the curve. `ACCEL_MAX` when it is not over the setpoint (and for the `inf` setpoint)."""
  if not math.isfinite(v_hold) or v_ego <= v_hold:
    return ACCEL_MAX
  return max(MOONPILOT_CURVE_K_HOLD * (v_hold - v_ego), MOONPILOT_CURVE_ACCEL_MIN)


def predicted_lat_accel(model, t=MOONPILOT_CURVE_BIAS_T) -> float:
  """What the path predicted it would pull at t: `|curvature| * v^2` reduces to `|orientationRate.z| *
  velocity.x`, both interpolated onto `t` over the model's own grid. `0.0` when the arrays are
  shorter than two samples, which is below the estimator's floor and so contributes nothing."""
  psi_rate = np.asarray(model.orientationRate.z, dtype=float)
  v_path = np.asarray(model.velocity.x, dtype=float)
  n = min(len(psi_rate), len(v_path), len(MOONPILOT_CURVE_T_IDX))
  if n < 2:
    return 0.0
  t_idx = MOONPILOT_CURVE_T_IDX[:n]
  a = float(np.interp(t, t_idx, np.abs(psi_rate[:n])) * np.interp(t, t_idx, v_path[:n]))
  return a if math.isfinite(a) else 0.0


class LatAccelBiasEstimator:
  """Realized lateral accel over what the path predicted, learned onroad and persisted per car.

  A prediction is paired with the measurement `MOONPILOT_CURVE_BIAS_T` later — the prediction the path
  made about the moment the car is in *now* — and the ratio is filtered. Straight-line frames teach
  nothing, because both magnitudes have to clear `MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL` before a pair
  counts, and an invalid frame (not engaged, driver overriding, too slow) empties the queue so a
  prediction never scores a measurement from the other side of a disengage.

  `applied()` is clamped at 1.0, and that is the whole safety argument for this class: a car that
  realizes *more* lateral accel than the path predicted is effectively in a tighter curve and should
  slow more, while the opposite direction would hand the feature *less* braking than the geometry
  says — the same one-sided rule `moonpilot/latency.py` applies to the lag it learns. The raw ratio
  conflates lateral-controller tracking error with genuine model bias; nothing here separates them,
  which is the second reason the applied value is one-sided and bounded at `MOONPILOT_CURVE_BIAS_MAX`.
  """

  def __init__(self, dt: float = DT_MDL):
    self.delay_frames = max(1, int(round(MOONPILOT_CURVE_BIAS_T / dt)))
    self.pair_len = self.delay_frames + 1  # predictions held, so the one paired is `MOONPILOT_CURVE_BIAS_T` old
    self.predicted: deque[float] = deque(maxlen=self.pair_len)
    self.filter = FirstOrderFilter(1.0, MOONPILOT_CURVE_BIAS_RC, dt)
    self.samples = 0
    self.frames = 0  # every frame it was fed, not just the paired ones: the persistence cadence

  def seed(self, scale: float, samples: int) -> None:
    """Resume a persisted scale, the `lagd.reset` pattern: the filter starts at the value with
    enough samples behind it, so `applied()` uses it on the first frame."""
    self.filter.x = float(scale)
    self.filter.initialized = True
    self.samples = int(samples)

  @property
  def status(self) -> str:
    return "estimated" if self.samples >= MOONPILOT_CURVE_BIAS_MIN_SAMPLES else "measuring"

  @property
  def estimate(self) -> float:
    """The filtered ratio, for logging and persistence. Before the sample threshold it is still the
    filter's own seed, which is exactly why `applied()` does not use it."""
    return float(self.filter.x)

  def applied(self) -> float:
    """The scale the curve targets are computed with. One-sided and bounded — see the docstring."""
    if self.status != "estimated":
      return 1.0
    return min(max(self.estimate, 1.0), MOONPILOT_CURVE_BIAS_MAX)

  def update(self, predicted, measured, valid: bool) -> None:
    self.frames += 1
    if not valid:
      self.predicted.clear()
      return
    self.predicted.append(abs(float(predicted)))
    if len(self.predicted) < self.pair_len:
      return
    p, m = self.predicted[0], abs(float(measured))
    if not (math.isfinite(p) and math.isfinite(m)):
      return
    if p <= MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL or m <= MOONPILOT_CURVE_BIAS_MIN_LAT_ACCEL:
      return
    self.filter.update(min(max(m / p, MOONPILOT_CURVE_BIAS_MIN), MOONPILOT_CURVE_BIAS_MAX))
    self.samples += 1
