"""The fork's live longitudinal lag estimate (`moonpilot/latency.py`), and its wiring into the planner.

`TestLongLagEstimator` drives the estimator on a plant that is a pure dead time — `a_veh[k] =
cmd[k - lag]` — so the answer is known and only the estimator's own gates are under test: it has to
find a 0.30 s dead time, refuse to find one where there is nothing to find (a measurement that never
moves, a command that never moves), keep its own output inside the ROI, and hold off around invalid
frames.

`TestPlannerWiring` holds the two ends of the electrical connection. The gate is the planner's
judgment about which frames carry information — an unengaged car is not being commanded by this
planner, so nothing may be learned from it — and `action_t` is the estimator's only consumer.
"""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import moonpilot.longitudinal as longitudinal_mod
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL

from moonpilot.latency import (
  MOONPILOT_LAG_BLOCKS_NEEDED,
  MOONPILOT_LAG_BLOCKS_KEY,
  MOONPILOT_LAG_BLOCK_SIZE,
  MOONPILOT_LAG_MAX,
  MOONPILOT_LAG_WINDOW_SEC,
  LongLagEstimator,
  applied_long_delay,
  persisted_seed,
)
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.tests.test_longitudinal import _inputs, _lead, _planner

CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)

# The window has to fill before anything can be estimated — `Points` prefills with zeros and a
# correlation over the unfilled region is NaN — and then `MOONPILOT_LAG_BLOCKS_NEEDED` blocks of
# `MOONPILOT_LAG_BLOCK_SIZE` estimates have to clear the gates: the window plus 25 s at 20 Hz. Derived
# from the window constant so a change there cannot silently starve these tests, and 40 s of margin
# over it means the exact recovery timing below is the only thing measured against a clock.
SETTLED_FRAMES = int((MOONPILOT_LAG_WINDOW_SEC + 40.0) / DT_MDL)


def _square(hz: float, amplitude: float):
  """A square wave of period 1/hz — the command shape a driver's own braking produces."""
  return lambda k: amplitude * (1.0 if int(k * DT_MDL * hz * 2) % 2 == 0 else -1.0)


def _plant(est: LongLagEstimator, command, lag: int, frames: int, valid=True):
  """`frames` steps of a pure dead time, with the estimator fed the command the car is answering."""
  history: list[float] = []
  for k in range(frames):
    cmd = float(command(k))
    history.append(cmd)
    est.update(cmd, history[k - lag] if k >= lag else 0.0, valid)


class TestLongLagEstimator(unittest.TestCase):
  def test_recovers_a_known_dead_time(self):
    """Six frames of dead time, 0.30 s, read back to within one frame. This is also the case where
    the planner's projection actually changes: `applied_delay` is above the stock constant."""
    est = LongLagEstimator(CP)
    _plant(est, _square(0.25, 1.5), lag=6, frames=SETTLED_FRAMES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.estimate, 0.30, delta=DT_MDL)
    self.assertAlmostEqual(est.applied_delay(), 0.30, delta=DT_MDL)

  def test_the_applied_delay_never_drops_below_stock(self):
    """Two frames, 0.10 s: measured, believed, and *not* applied — the live value may only lengthen
    the projection, so a chain faster than upstream's constant leaves the constant standing."""
    est = LongLagEstimator(CP)
    _plant(est, _square(0.25, 1.5), lag=2, frames=SETTLED_FRAMES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_the_applied_delay_is_clamped_to_the_roi(self):
    """`seed` is the persistence path and it takes whatever a drive left in the param: an absurd
    value is planned around no further than the ROI allows."""
    est = LongLagEstimator(CP)
    est.seed(5.0, MOONPILOT_LAG_BLOCKS_NEEDED)
    self.assertAlmostEqual(est.applied_delay(), MOONPILOT_LAG_MAX, delta=1e-9)

  def test_quasi_static_commands_are_not_identifiable(self):
    """A command and a measurement dithering inside +-0.02 m/s^2 for two minutes. The correlation is
    near-perfect there — measured `corr` 1.0 at 0.85 of confidence, reporting 0.35 s for a 0.30 s
    plant — so it is the excitation floor, not the NCC gate, that keeps this out of the estimate. A
    steady follow and a car idling in park are both this case."""
    est = LongLagEstimator(CP)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(SETTLED_FRAMES)
    _plant(est, lambda k: 0.02 * (1.0 if noise[k // 10] > 0 else -1.0), lag=6, frames=SETTLED_FRAMES)
    self.assertEqual(est.status, "unestimated")
    self.assertEqual(est.valid_blocks, 0)
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_a_constant_measurement_is_inert(self):
    """`carState.aEgo` held at zero against an excited command — the maneuver plant, and every fork
    harness that feeds the planner a fixed `a_ego`. There is nothing to correlate against, so nothing
    is learned and `action_t` stays exactly the planner this fork had before the estimator existed."""
    est = LongLagEstimator(CP)
    command = _square(0.25, 1.5)
    for k in range(SETTLED_FRAMES):
      est.update(command(k), 0.0, True)
    self.assertEqual(est.status, "unestimated")
    self.assertEqual(est.valid_blocks, 0)
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_invalid_frames_are_held_off(self):
    """One second of invalid frames, then valid ones: nothing advances until
    `MOONPILOT_LAG_RECOVERY_SEC` past the last invalid frame, and then the next estimate closes the
    block. The hold-off is what keeps an estimate from being computed across a discontinuity — an
    unengaged stretch, a pedal, a standstill — and it is applied by marking the two seconds *after* an
    invalid frame not-okay, so what it costs is a hole in the window rather than a single sample.

    The run stops one estimate short of a block boundary so the boundary is available to move: 591
    frames past the window's fill is 119 estimates at 4 Hz, and the boundary is the 120th.
    """
    est = LongLagEstimator(CP)
    command = _square(0.25, 1.5)
    _plant(est, command, lag=6, frames=int(MOONPILOT_LAG_WINDOW_SEC / DT_MDL) + 591)
    before = (est.valid_blocks, est.block_avg.idx)
    self.assertGreater(before[0], 0)
    self.assertEqual(before[1], MOONPILOT_LAG_BLOCK_SIZE - 1)

    # 1 s of invalid frames: every attempt in it is blocked, because every sample since the last
    # accepted estimate is not-okay.
    _plant(est, command, lag=6, frames=20, valid=False)
    self.assertEqual((est.valid_blocks, est.block_avg.idx), before)

    # 35 frames of valid data is still strictly inside the hold-off the last invalid frame starts;
    # the next estimate is past it, and closes the block.
    _plant(est, command, lag=6, frames=35)
    self.assertEqual((est.valid_blocks, est.block_avg.idx), before)
    _plant(est, command, lag=6, frames=15)
    self.assertEqual(est.valid_blocks, before[0] + 1)

  def test_an_inconsistent_estimate_is_not_applied(self):
    """Blocks that disagree by more than upstream's `MAX_LAG_STD` give `invalid`, which is what keeps
    a drive that answered 0.05 s once and 0.55 s once from planning on their average."""
    est = LongLagEstimator(CP)
    est.seed(0.30, MOONPILOT_LAG_BLOCKS_NEEDED)
    self.assertEqual(est.status, "estimated")
    est.block_avg.values[0] = 0.05
    est.block_avg.values[1] = 0.55
    self.assertEqual(est.status, "invalid")
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)


class TestPlannerWiring(unittest.TestCase):
  # The stopping floor's onset at 15 m/s against a 9 m/s lead, where the projection's extra meters
  # decide whether the floor asks for anything at all: 0.45 s of lag is 0.30 s past the stock
  # constant, which at this 6 m/s closing rate is 1.8 m of the approach.
  ONSET = {"v_ego": 15.0, "v_cruise_kph": 54.0, "d_rel": 79.5, "v_lead": 9.0}

  @classmethod
  def _commands(cls, seed, frames=80):
    """Held at one state — the fork's own open-loop pattern — so both planners reach their level:
    `jerk_limit` moves at most 0.1 m/s^2 per frame from a zero baseline."""
    planner = _planner()
    if seed is not None:
      planner.long_lag.seed(seed, MOONPILOT_LAG_BLOCKS_NEEDED)
    sm = _inputs(v_ego=cls.ONSET["v_ego"], v_cruise_kph=cls.ONSET["v_cruise_kph"], lead=_lead(cls.ONSET["d_rel"], cls.ONSET["v_lead"]))
    commands = []
    for _ in range(frames):
      planner.update(sm)
      commands.append(planner.output_a_target)
    return planner, commands

  def test_the_planner_projects_through_the_seeded_lag(self):
    """The seeded lag reaches the command through the projection and nothing else.

    A single frame cannot show it: from the planner's zero baseline the jerk limit allows 0.1 m/s^2,
    so two first-frame commands are the same clamped value whatever the delay. Held at one state for
    80 frames the two planners separate qualitatively — on the stock constant the lead is still 1.8 m
    further out and the floor asks for nothing, while through 0.45 s it has crossed its onset.
    """
    unseeded_planner, unseeded = self._commands(None)
    seeded_planner, seeded = self._commands(0.45)

    self.assertAlmostEqual(unseeded_planner.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)
    self.assertAlmostEqual(seeded_planner.action_t, 0.45 + DT_MDL, delta=1e-9)

    self.assertAlmostEqual(min(unseeded), 0.0, delta=0.05)
    self.assertLess(min(seeded), min(unseeded) - 0.2)

  def test_the_gate_follows_the_long_control_state(self):
    """Nothing is learned while the planner's command is not what moves the car. An unengaged car is
    the clearest case: `reset_state` holds, `output_a_target` is set to `a_ego`, and the estimator
    would be correlating the driver's own driving against itself."""
    planner = _planner()
    sm = _inputs(enabled=False)
    for _ in range(600):
      planner.update(sm)
    self.assertEqual(planner.long_lag.valid_blocks, 0)
    self.assertAlmostEqual(planner.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

  def test_a_persisted_value_is_seeded_on_construction(self):
    """`MoonpilotLongLag` is the whole of the persistence: a value the last drive left there is read
    at construction and applied before the first frame. The `isinstance` guard is what keeps the
    existing tests out of this path — their `FakeParams` answers every key with a bool, and a bool is
    not a float — and what keeps the declared `"0.0"` default, below the ROI floor, from seeding."""
    with mock.patch.object(longitudinal_mod, "Params", lambda: _StoredParams(0.45)):
      seeded = MoonpilotLongitudinalPlanner(CP)
    self.assertEqual(seeded.long_lag.status, "estimated")
    self.assertAlmostEqual(seeded.action_t, 0.45 + DT_MDL, delta=1e-9)

    with mock.patch.object(longitudinal_mod, "Params", lambda: _StoredParams(0.0)):
      unmeasured = MoonpilotLongitudinalPlanner(CP)
    self.assertEqual(unmeasured.long_lag.status, "unestimated")
    self.assertAlmostEqual(unmeasured.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

  def test_a_partial_evidence_count_is_carried_not_trusted(self):
    """The count that rides with the value decides whether it is applied, which is what lets a drive
    with too little engaged long control hand its block on instead of having it thrown away — and
    what keeps a mean from one block out of `action_t`. A value written before the count existed was
    only ever written when trusted, so a store that answers 0 for it still seeds as trusted."""
    with mock.patch.object(longitudinal_mod, "Params", lambda: _StoredParams(0.45, 1)):
      carried = MoonpilotLongitudinalPlanner(CP)
    self.assertEqual(carried.long_lag.valid_blocks, 1)
    self.assertEqual(carried.long_lag.status, "unestimated")
    self.assertAlmostEqual(carried.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

    with mock.patch.object(longitudinal_mod, "Params", lambda: _StoredParams(0.45, MOONPILOT_LAG_BLOCKS_NEEDED)):
      trusted = MoonpilotLongitudinalPlanner(CP)
    self.assertEqual(trusted.long_lag.status, "estimated")
    self.assertAlmostEqual(trusted.action_t, 0.45 + DT_MDL, delta=1e-9)

  def test_evidence_accumulates_across_seeds(self):
    """Blocks carried from earlier drives plus the one this drive earns is a trusted mean: the count
    is the evidence, and the ring keeps filling from where the last drive left off rather than
    restarting. The value stays inside the same gates — it is the *amount* of evidence that is being
    accumulated, not its quality."""
    est = LongLagEstimator(CP, DT_MDL)
    est.seed(0.30, MOONPILOT_LAG_BLOCKS_NEEDED - 1)
    self.assertEqual(est.status, "unestimated")
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)

    _plant(est, _square(0.25, 1.5), lag=6, frames=SETTLED_FRAMES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.estimate, 0.30, delta=DT_MDL)
    self.assertAlmostEqual(est.applied_delay(), 0.30, delta=DT_MDL)


class _StoredParams:
  """A param store holding the persisted lag pair, for the seeding path `FakeParams`'s bool cannot
  reach. `blocks` answers the evidence key and nothing else, so the value still reads as the value."""

  def __init__(self, value, blocks=None):
    self.value = value
    self.blocks = blocks

  def get(self, key, block=False, return_default=False):
    if key == MOONPILOT_LAG_BLOCKS_KEY:
      return self.blocks if self.blocks is not None else 0
    return self.value


class TestAppliedLongDelay(unittest.TestCase):
  """The horizon `modeld` decodes the longitudinal ask at is the number the planner projects through,
  plus the model's smoothing: one validator (`persisted_seed`) decides what a persisted value is worth
  for both consumers, so a boot cannot compute the ask and the plan for different horizons."""

  def test_the_stock_constant_is_the_floor_and_the_fallback(self):
    self.assertIsNone(persisted_seed(_StoredParams(0.0)))
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(0.0)), CP.longitudinalActuatorDelay, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(0.0), 0.3), CP.longitudinalActuatorDelay + 0.3, delta=1e-9)

  def test_a_trusted_value_is_applied_with_its_evidence(self):
    trusted = _StoredParams(0.45, MOONPILOT_LAG_BLOCKS_NEEDED)
    self.assertAlmostEqual(applied_long_delay(CP, trusted), 0.45, delta=1e-9)
    # a value written before the count existed was only ever written when trusted
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(0.45)), 0.45, delta=1e-9)
    # below the requirement the value is carried, not applied
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(0.45, 1)), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_the_roi_bounds_the_value(self):
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(1.5, MOONPILOT_LAG_BLOCKS_NEEDED)), MOONPILOT_LAG_MAX, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, _StoredParams(0.02, MOONPILOT_LAG_BLOCKS_NEEDED)), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_modeld_decodes_at_it(self):
    """Both the horizon it starts with and the one it re-reads: a boot with a learned value and a
    boot that learns one mid-drive must decode the ask at the plan's own horizon."""
    text = (Path(__file__).resolve().parents[2] / "openpilot/selfdrive/modeld/modeld.py").read_text()
    self.assertEqual(text.count("applied_long_delay(CP, params, long_smooth)"), 2)


if __name__ == "__main__":
  unittest.main()
