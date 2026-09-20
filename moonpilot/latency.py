"""The fork's own estimate of the longitudinal chain's lag.

`moonpilot/longitudinal.py` compensates the actuator by predicting the ego and the leads forward to
`action_t`, and that has been `CP.longitudinalActuatorDelay + DT_MDL` — upstream's `0.15` for every
car without an override, marked `# TODO estimate car specific lag` (`opendbc/car/interfaces.py`).
Nothing measures it: upstream's carcontroller compensates the PCM's own slow response with a nested
PID, but the planner never learns the delay it plans against. This module measures it, on the one
signal pair the planner already has: the command it published and the acceleration the car delivered.

The identification math is upstream's, not a copy — `openpilot/selfdrive/locationd/lagd.py` already
solves exactly this problem for the *lateral* chain, always onroad, with a masked normalized
cross-correlation over a 60 s sliding window and a block average over the estimates that clear its
gates. So this is the same machinery (`Points`, `BlockAverage`, `masked_symmetric_moving_average`,
`LateralLagEstimator.actuator_delay`) driven by a different signal pair and its own gates, rather
than a second implementation of an FFT cross-correlation that would drift from it. `lagd` is not
edited: its thresholds (`MIN_NCC`, `MIN_CONFIDENCE`, `MAX_LAG_STD`, `SMOOTH_K`, `SMOOTH_SIGMA`) are
imported, not re-typed, so the two estimators cannot disagree about what a good estimate looks like.

Three things are this module's own:

- **The signals.** `expected` is the planner's published `a_target`, `actual` is `carState.aEgo`, so
  the lag measured is the whole chain from the command to the road — planner output, the fork's
  acceleration controller, the car's own longitudinal actuator. `aEgo` is the wheel-speed KF
  (`opendbc/car/interfaces.py`, gains 0.174 / 1.659 at DT_CTRL): 50 % of a step at 0.04 s and 90 % at
  0.08 s, i.e. at or below this ROI's floor, so a measured lag is the chain's and not the filter's to
  within one frame. The command fed in is the *previous* frame's, the value that has been in effect
  for a full period.
- **The ROI (0.05 … 0.60 s).** `lagd`'s floor is 0.15 s because a steering actuator is slower than
  this; `CP.longitudinalActuatorDelay` is 0.15 s *before* the controller and the car, so the floor
  moves down to one 20 Hz frame. The ceiling mirrors `lagd`'s `MAX_LAG`: past 0.6 s the planner's
  projection is longer than the horizon it makes decisions over.
- **The cadence, and what a block is.** The estimate runs every 5th frame (4 Hz), as `lagd` does —
  but `lagd`'s block is 100 accepted estimates, i.e. 25 s of them, against a lateral signal that only
  excites in corners. A longitudinal estimate should converge inside one drive, so a block here is
  5 s: `MOONPILOT_LAG_BLOCK_SIZE = 20` accepted estimates at that 4 Hz cadence. The block count and
  the number of blocks needed are upstream's, which puts the first usable estimate 25 s after the
  window fills (5 blocks × 5 s) and the ring at 250 s.
- **The window fill.** `Points` prefills its deque with zeros and `okay=False`, so `lagd`'s
  `points_enough` — `num_points >= okay_window_sec / dt`, against a deque that is always full — is
  true from the first frame, and the masked moving average returns NaN for the unfilled region, which
  an FFT turns into an all-NaN correlation. Here the window is required to be genuinely full first,
  and an estimate with a non-finite delay, correlation or confidence is dropped rather than fed to the
  block average, where a single NaN would poison the running mean for the life of the drive. Both are
  the conservative direction: no estimate is better than a wrong one.

`applied_delay()` is the consumer's entry point, and it may only ever *lengthen* the projection: the
maneuver suite and the stopping floor's bounds were validated at or above the stock value, so a
shorter projection is the direction that would need its own evidence. Unestimated, invalid, or a
measurement below the stock constant all give the stock constant, i.e. the planner this fork had
before any of this existed.

**What the corpus says, measured rather than assumed.** The offline corpus this fork already has
(`/home/coder/route-corpus`, 199 rlog segments off the car, read through the frozen `custom` worktree
whose cereal matches it) runs the estimator to `unestimated` everywhere — the intended outcome on a
corpus that cannot support the measurement — and the reason is *not* the excitation floor. Over 42,980
estimate attempts, **98.4 % were rejected on `num_okay`**: not enough valid samples inside one 60 s
window, because the corpus holds only ~2.4 min of engaged longitudinal control in total. The
excitation-range gate bound nothing, and only 24 estimates were ever accepted against the 100 a first
trusted block needs. So the contingency to reach for is more engaged driving, never a lower
`MOONPILOT_LAG_RANGE`.

**Which is why the evidence count is persisted with the value, and not the value alone.** Replayed
per *drive* — the estimator spans log segments, and feeding it segment by segment would restart it
every minute — the two routes that carry engaged long control give the split above a shape: 3,688
attempts, of which **82 % lack valid samples**, 14 % fail the confidence gate and 4 % the NCC gate;
the attempts that clear both quality gates read 0.245 s (sd 0.016) and 0.350 s (sd 0.043), i.e. the
estimator is right where it can speak. What it cannot do is speak often enough: those drives reach
**1 block of the 5** a trusted mean needs, and a sweep of the sizing constants changes nothing —
window 40/90/120/180 s × valid gate 5/15/25 s × speed gate 1/2/5 m/s all end at 0 trusted blocks,
best case 1. The gate is not mis-tuned, the evidence is thin. Carrying the count across drives is
therefore the one lever the data supports: a drive that earns a block adds it to what earlier drives
left, the mean is applied only once the blocks are there, and `MAX_LAG_STD` is what checks that the
blocks agree. `MoonpilotLongLagBlocks` is that count; a value written before the key existed was only
ever written when trusted, so a missing count reads as the needed one.

Two more numbers from the same run, both worth knowing before reading a device result. Where the
estimator did get to run it reported 0.33–0.40 s with NCC 0.965–0.986, and an independent high-pass
lag scan of the same data — which shares no machinery with `lagd` — peaks at 0.34 s (r 0.576 against
0.510 at zero lag, and ~0.01 against a shuffled command), so the chain is measurably slower than the
0.15 s this car's `carParams` actually carry. But the identification is soft: 99.8 % of the command's
variance in that corpus sits above a 2 s timescale, so the raw correlation is nearly flat across every
lag (0.970 at 0 s, 0.976 at 0.35 s) and the confidence gate lands exactly on its 0.700 threshold. A
sharp on-device number should not be expected; a lengthened projection, eventually, is.

**And one route replayed as a control, which settled the risk that mattered most.** The TOYOTA2
`process_replay` segment (`0982d79ebb0de295|2021-01-03--20-03-36--6`, a 2017 RAV4) fed through this
class ends `unestimated` with `valid_blocks == 0` and `applied_delay()` at the stock constant — the
outcome the plan predicted, but *not* for the reason it predicted. The plan expected the car's
longitudinal to be off; its `carParams` say `openpilotLongitudinalControl = True`, `longControlState`
is `pid` on **all 402** planner ticks, and the planner's own `lag_valid` gate admits every one. What
stops the estimate is the window: 402 ticks at 20 Hz is 20 s against `MOONPILOT_LAG_WINDOW_SEC`, so
the estimate step never even runs. The gate that answers "is the planner's command what moves the
car" is therefore *not* what this route exercised — the corpus sweep above is what does — and reading
a pass here as evidence for it would be reading the wrong mechanism.

The same replay does settle the one silent-death risk in this wiring: upstream reads
`sm['controlsState'].longControlState == LongCtrlState.off` and this module's caller reads
`== LongCtrlState.pid`, and a capnp enum read off a message is a `_DynamicEnum`, not the plain `int`
the schema member is — `int()` on it raises, which is why `moonpilot/longitudinal.py`'s personality
lookup is keyed by the raw value. Equality still holds across that boundary (**402/402** ticks
matched `pid` on a real message), so the gate is live on a car and not merely in the tests' plain-int
fakes. If it ever stops matching, the estimator goes quietly dead rather than wrong.
"""

import math

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.locationd.lagd import (
  MAX_LAG_STD,
  MIN_CONFIDENCE,
  MIN_NCC,
  SMOOTH_K,
  SMOOTH_SIGMA,
  BlockAverage,
  LateralLagEstimator,
  Points,
  masked_symmetric_moving_average,
)

MOONPILOT_LAG_KEY = "MoonpilotLongLag"  # the param the learned value persists in, as FLOAT
MOONPILOT_LAG_BLOCKS_KEY = "MoonpilotLongLagBlocks"  # its evidence, in blocks, as INT
MOONPILOT_LAG_LOG_DELTA = 0.05  # s of movement worth a log line, so the qlog can be read per drive
MOONPILOT_LAG_MIN = 0.05  # s; ROI floor, one frame of the 20 Hz command
MOONPILOT_LAG_MAX = 0.60  # s; ROI ceiling, mirrors lagd's MAX_LAG
MOONPILOT_LAG_MIN_SPEED = 5.0  # m/s; below this the stopping ramp owns the command, not the policy
MOONPILOT_LAG_MAX_ABS_AEGO = 4.0  # m/s^2; sanity gate on the measurement itself
MOONPILOT_LAG_RANGE = 0.5  # m/s^2; excitation floor, mirrors lagd's MIN_LAT_ACCEL_RANGE
MOONPILOT_LAG_WINDOW_SEC = 90.0  # s of samples per correlation: lagd's 60 s, extended because the
# gate that decides whether an estimate is possible is the *valid-sample* count inside the window, and
# this car's engaged long control comes in ~60 s stretches (measured: at 60 s the best engaged route
# earns 0 blocks, at 90 s it earns 1). A longer window is how sparse engagement accumulates into one
# correlation; the quality gates (NCC, confidence, block spread) are untouched, and the estimates the
# two windows produce agree to 0.06 s.
MOONPILOT_LAG_MIN_OKAY_SEC = 25.0  # s of valid samples in the window before an estimate is used
MOONPILOT_LAG_RECOVERY_SEC = 2.0  # s of hold-off after an invalid frame, lagd's own buffer
MOONPILOT_LAG_BLOCK_SIZE = 20  # estimates per block: 5 s at the 4 Hz estimate cadence below
MOONPILOT_LAG_BLOCK_COUNT = 50  # blocks in the ring, 250 s of history
MOONPILOT_LAG_BLOCKS_NEEDED = 5  # blocks before the mean is trusted: 25 s of estimates
MOONPILOT_LAG_ESTIMATE_EVERY = 5  # frames between estimates, 4 Hz at DT_MDL, as lagd's `sm.frame % 5`
MOONPILOT_LAG_PERSIST_EVERY = 1200  # frames between param writes, 60 s at DT_MDL, as lagd's cache


class LongLagEstimator:
  """Command -> delivered-accel delay, measured live. numpy only, and no messaging: the planner
  feeds it, so it needs neither a service nor a process of its own."""

  def __init__(self, CP, dt: float = DT_MDL):
    self.CP = CP
    self.dt = dt
    self.t = 0.0
    self.frame = 0
    self.last_invalid_t = -math.inf
    self.last_estimate_t = 0.0
    self.reset(0.0, 0)

  def reset(self, delay: float, valid_blocks: int) -> None:
    self.points = Points(int(MOONPILOT_LAG_WINDOW_SEC / self.dt))
    self.block_avg = BlockAverage(MOONPILOT_LAG_BLOCK_COUNT, MOONPILOT_LAG_BLOCK_SIZE, valid_blocks, float(delay))

  def seed(self, delay: float, valid_blocks: int) -> None:
    """Resume a persisted estimate, the `lagd.reset` pattern: the block average starts at the value
    with enough blocks behind it, so `applied_delay` uses it on the first frame."""
    self.reset(delay, valid_blocks)

  def update(self, cmd: float, a_ego: float, valid: bool) -> None:
    """One frame. `valid` is the caller's judgment that the command is what is moving the car;
    the magnitude, finiteness and recovery gates are this module's."""
    a_ego = float(a_ego)
    okay = bool(valid) and math.isfinite(a_ego) and abs(a_ego) <= MOONPILOT_LAG_MAX_ABS_AEGO
    if not okay:
      self.last_invalid_t = self.t
    recovered = self.t - self.last_invalid_t >= MOONPILOT_LAG_RECOVERY_SEC
    self.points.update(self.t, float(cmd), a_ego, okay and recovered)

    if self.frame % MOONPILOT_LAG_ESTIMATE_EVERY == 0:
      self._update_estimate()
    self.frame += 1
    self.t += self.dt

  def _update_estimate(self) -> None:
    # The window has to be full of real samples: `Points` prefills with zeros, and a moving average
    # over an unfilled region is NaN, which the FFT below would spread over the whole correlation.
    if self.frame * self.dt < MOONPILOT_LAG_WINDOW_SEC:
      return

    times, cmd, a_ego, okay = self.points.get()
    is_valid = self.points.num_okay >= int(MOONPILOT_LAG_MIN_OKAY_SEC / self.dt)
    # Both signals have to move: a correlation of two constants has no peak to find. Not the gate that
    # ever binds in practice — the corpus sweep in the module docstring has `num_okay` rejecting 98.4 %
    # of attempts and this one nothing — but it is the cheap guard against a degenerate window.
    is_valid = is_valid and (cmd.max() - cmd.min() >= MOONPILOT_LAG_RANGE) and (a_ego.max() - a_ego.min() >= MOONPILOT_LAG_RANGE)
    # Only estimate on data newer than the last accepted estimate, so the block average averages
    # estimates of different windows rather than the same one repeatedly.
    if self.last_estimate_t != 0.0 and times[0] <= self.last_estimate_t:
      new_values_start_idx = next(-i for i, t in enumerate(reversed(times)) if t <= self.last_estimate_t)
      is_valid = is_valid and not (new_values_start_idx == 0 or not np.any(okay[new_values_start_idx:]))

    cmd = masked_symmetric_moving_average(cmd, okay, SMOOTH_K, SMOOTH_SIGMA)
    a_ego = masked_symmetric_moving_average(a_ego, okay, SMOOTH_K, SMOOTH_SIGMA)

    delay, corr, confidence = LateralLagEstimator.actuator_delay(cmd, a_ego, okay, self.dt, MOONPILOT_LAG_MIN, MOONPILOT_LAG_MAX)
    # A NaN passes every `<` comparison below, so it is excluded here rather than by them.
    if not (np.isfinite(delay) and np.isfinite(corr) and np.isfinite(confidence)):
      return
    if corr < MIN_NCC or confidence < MIN_CONFIDENCE or not is_valid:
      return

    self.block_avg.update(float(delay))
    self.last_estimate_t = self.t

  def applied_delay(self) -> float:
    """What `action_t` must use. The live value can only lengthen the projection — see the module
    docstring — and every state but a trusted estimate is the stock constant."""
    if self.status != "estimated":
      return self.CP.longitudinalActuatorDelay
    return max(self.CP.longitudinalActuatorDelay, min(MOONPILOT_LAG_MAX, max(MOONPILOT_LAG_MIN, self.estimate)))

  @property
  def estimate(self) -> float:
    """The raw block mean, for logging and persisting. The stock constant while there is no mean."""
    valid_mean, _ = self.block_avg.get()[:2]
    return self.CP.longitudinalActuatorDelay if not math.isfinite(valid_mean) else float(valid_mean)

  @property
  def status(self) -> str:
    valid_mean, valid_std = self.block_avg.get()[:2]
    if self.block_avg.valid_blocks >= MOONPILOT_LAG_BLOCKS_NEEDED and math.isfinite(valid_mean) and math.isfinite(valid_std):
      return "invalid" if valid_std > MAX_LAG_STD else "estimated"
    return "unestimated"

  @property
  def valid_blocks(self) -> int:
    return self.block_avg.valid_blocks
