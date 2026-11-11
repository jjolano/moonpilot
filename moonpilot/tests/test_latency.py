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

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL

from moonpilot.latency import (
  MOONPILOT_LAG_BLOCKS_NEEDED,
  MOONPILOT_LAG_BLOCKS_KEY,
  MOONPILOT_LAG_BLOCK_SIZE,
  MOONPILOT_LAG_HISTORY_KEY,
  MOONPILOT_LAG_MAX,
  MOONPILOT_LAG_WINDOW_SEC,
  MOONPILOT_LAG_KEY,
  LongLagEstimator,
  applied_long_delay,
  persisted_seed,
)
from moonpilot.learned import PERSIST_EVERY, completed_blocks
from moonpilot.longitudinal import MoonpilotLongitudinalPlanner
from moonpilot.tests.fakes import params
from moonpilot.tests.harness import inputs, lead, planner as make_planner

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
    """`restore` is the persistence path and it takes whatever a drive left in the param: an absurd
    value is planned around no further than the ROI allows."""
    est = LongLagEstimator(CP)
    est.restore(params(on=False, overrides={MOONPILOT_LAG_KEY: 5.0, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED, MOONPILOT_LAG_HISTORY_KEY: None}))
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
    """A drive that answered 0.05 s once and 0.55 s once cannot plan on their average."""
    est = LongLagEstimator(CP)
    est.restore(params(on=False, overrides={MOONPILOT_LAG_KEY: 0.30, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED, MOONPILOT_LAG_HISTORY_KEY: None}))
    self.assertEqual(est.status, "estimated")
    est.block_avg.values[0] = 0.05
    est.block_avg.values[1] = 0.55
    self.assertEqual(est.status, "unestimated")
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)


def _lag_params(seed: float, blocks: int = MOONPILOT_LAG_BLOCKS_NEEDED, history=None):
  """A store holding the learned lag as a drive would leave it: the value, its evidence, and optionally the blocks."""
  return params(on=False, overrides={MOONPILOT_LAG_KEY: seed, MOONPILOT_LAG_BLOCKS_KEY: blocks, MOONPILOT_LAG_HISTORY_KEY: history})


class TestPlannerWiring(unittest.TestCase):
  # A 20 m/s follow closing on an 18 m/s lead at 29 m. The class comment used to say the spacing
  # regulator governs here and reads the projected speed; under the lead law the projection nearly
  # cancels at this state (0.004 m/s^2), so the reach is pinned on a heavier closing below.
  ONSET = {"v_ego": 20.0, "v_cruise_kph": 72.0, "d_rel": 29.0, "v_lead": 18.0}

  @classmethod
  def _commands_at(cls, seed, v_ego, v_lead, d_rel, frames=80):
    """Held at one state — the fork's own open-loop pattern — so a planner reaches its level:
    `jerk_limit` moves at most 0.1 m/s^2 per frame from a zero baseline."""
    return cls._commands(seed, frames=frames, state=(v_ego, v_lead, d_rel))[1]

  @classmethod
  def _commands(cls, seed, frames=80, state=None):
    v_ego, v_lead, d_rel = state or (cls.ONSET["v_ego"], cls.ONSET["v_lead"], cls.ONSET["d_rel"])
    planner = make_planner(params=_lag_params(seed) if seed is not None else params(on=False))
    sm = inputs(v_ego=v_ego, v_cruise_kph=cls.ONSET["v_cruise_kph"], lead=lead(d_rel, v_lead))
    commands = []
    for _ in range(frames):
      planner.update(sm)
      commands.append(planner.output_a_target)
    return planner, commands

  def test_the_planner_projects_through_the_seeded_lag(self):
    """The seeded lag reaches the command through the projection and nothing else.

    A single frame cannot show it: from the planner's zero baseline the jerk limit allows 0.1 m/s^2,
    so two first-frame commands are the same clamped value whatever the delay. `action_t` is the
    wiring itself — 0.50 s against 0.20 s here — and the command has to move with it, so a second
    state is flown where the projection has room to act: closing 6 m/s at 16 m/s, where the extra
    0.30 s of horizon moves the settled ask by 0.32 m/s^2. On the settled approach it nearly
    cancels (0.004 m/s^2, deeper), because the projection shrinks the gap and the closing together,
    which is why the reach is pinned on the state that shows it and not on the direction.
    """
    unseeded_planner, unseeded = self._commands(None)
    seeded_planner, seeded = self._commands(0.45)

    self.assertAlmostEqual(unseeded_planner.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)
    self.assertAlmostEqual(seeded_planner.action_t, 0.45 + DT_MDL, delta=1e-9)

    self.assertLess(min(unseeded), -0.5)
    self.assertNotEqual(seeded, unseeded)
    for state in ((16.0, 10.0, 18.0), (20.0, 15.0, 24.0)):
      with self.subTest(state=state):
        deltas = [abs(min(self._commands_at(seed, *state)) - min(self._commands_at(None, *state))) for seed in (0.45,)]
        self.assertGreater(deltas[0], 0.05)

  def test_the_gate_follows_the_long_control_state(self):
    """Nothing is learned while the planner's command is not what moves the car. An unengaged car is
    the clearest case: `reset_state` holds, `output_a_target` is set to `a_ego`, and the estimator
    would be correlating the driver's own driving against itself."""
    planner = make_planner()
    sm = inputs(enabled=False)
    for _ in range(600):
      planner.update(sm)
    self.assertEqual(planner.long_lag.valid_blocks, 0)
    self.assertAlmostEqual(planner.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

  def test_a_persisted_value_is_seeded_on_construction(self):
    """Without a block history — a store written before `MoonpilotLongLagHistory` existed — a value the
    last drive left in `MoonpilotLongLag` is read
    at construction and applied before the first frame. These cases use explicit typed overrides to
    exercise persisted values and counts, while the declared `"0.0"` default stays below the ROI floor."""
    seeded = MoonpilotLongitudinalPlanner(CP, params=_lag_params(0.45, blocks=0))
    self.assertEqual(seeded.long_lag.status, "estimated")
    self.assertAlmostEqual(seeded.action_t, 0.45 + DT_MDL, delta=1e-9)

    unmeasured = MoonpilotLongitudinalPlanner(CP, params=_lag_params(0.0, blocks=0))
    self.assertEqual(unmeasured.long_lag.status, "unestimated")
    self.assertAlmostEqual(unmeasured.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

  def test_a_partial_evidence_count_is_carried_not_trusted(self):
    """The count that rides with the value decides whether it is applied, which is what lets a drive
    with too little engaged long control hand its block on instead of having it thrown away — and
    what keeps a mean from one block out of `action_t`. A value written before the count existed was
    only ever written when trusted, so a store that answers 0 for it still seeds as trusted."""
    carried = MoonpilotLongitudinalPlanner(CP, params=_lag_params(0.45, blocks=1))
    self.assertEqual(carried.long_lag.valid_blocks, 1)
    self.assertEqual(carried.long_lag.status, "unestimated")
    self.assertAlmostEqual(carried.action_t, CP.longitudinalActuatorDelay + DT_MDL, delta=1e-9)

    trusted = MoonpilotLongitudinalPlanner(CP, params=_lag_params(0.45))
    self.assertEqual(trusted.long_lag.status, "estimated")
    self.assertAlmostEqual(trusted.action_t, 0.45 + DT_MDL, delta=1e-9)

  def test_a_boot_resumes_the_persisted_blocks_not_their_mean(self):
    """The history wins over the value and count: the blocks come back as they were, so their spread
    is still what `MAX_LAG_STD` judges, and the next persist writes them back out unchanged."""
    history = [0.40, 0.50, 0.40, 0.50, 0.45]
    planner = MoonpilotLongitudinalPlanner(CP, params=_lag_params(0.45, history=history))
    np.testing.assert_allclose(completed_blocks(planner.long_lag.block_avg), history, atol=1e-9)
    self.assertAlmostEqual(planner.long_lag.block_avg.get()[1], float(np.std(history)), delta=1e-9)
    self.assertAlmostEqual(planner.action_t, 0.45 + DT_MDL, delta=1e-9)

    planner = make_planner(params=_lag_params(0.45, history=history))
    for _ in range(PERSIST_EVERY):
      planner.update(inputs())
    self.assertEqual(planner.params.puts.get(MOONPILOT_LAG_HISTORY_KEY), history)

  def test_evidence_accumulates_across_seeds(self):
    """Blocks carried from earlier drives plus the one this drive earns is a trusted mean: the count
    is the evidence, and the ring keeps filling from where the last drive left off rather than
    restarting. The value stays inside the same gates — it is the *amount* of evidence that is being
    accumulated, not its quality."""
    est = LongLagEstimator(CP, DT_MDL)
    est.restore(_lag_params(0.30, blocks=MOONPILOT_LAG_BLOCKS_NEEDED - 1))
    self.assertEqual(est.status, "unestimated")
    self.assertAlmostEqual(est.applied_delay(), CP.longitudinalActuatorDelay, delta=1e-9)

    _plant(est, _square(0.25, 1.5), lag=6, frames=SETTLED_FRAMES)
    self.assertEqual(est.status, "estimated")
    self.assertAlmostEqual(est.estimate, 0.30, delta=DT_MDL)
    self.assertAlmostEqual(est.applied_delay(), 0.30, delta=DT_MDL)




class TestAppliedLongDelay(unittest.TestCase):
  """Modeld and the planner accept the same persisted estimate."""

  def test_the_stock_constant_is_the_floor_and_the_fallback(self):
    zero = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.0, MOONPILOT_LAG_BLOCKS_KEY: 0, MOONPILOT_LAG_HISTORY_KEY: None})
    self.assertIsNone(persisted_seed(zero))
    self.assertAlmostEqual(applied_long_delay(CP, zero), CP.longitudinalActuatorDelay, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, zero, 0.3), CP.longitudinalActuatorDelay + 0.3, delta=1e-9)

  def test_a_trusted_value_is_applied_with_its_evidence(self):
    trusted = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.45, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED, MOONPILOT_LAG_HISTORY_KEY: None})
    uncounted = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.45, MOONPILOT_LAG_BLOCKS_KEY: 0, MOONPILOT_LAG_HISTORY_KEY: None})
    partial = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.45, MOONPILOT_LAG_BLOCKS_KEY: 1, MOONPILOT_LAG_HISTORY_KEY: None})
    self.assertAlmostEqual(applied_long_delay(CP, trusted), 0.45, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, uncounted), 0.45, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, partial), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_the_roi_bounds_the_value(self):
    high = params(on=False, overrides={MOONPILOT_LAG_KEY: 1.5, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED, MOONPILOT_LAG_HISTORY_KEY: None})
    low = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.02, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED, MOONPILOT_LAG_HISTORY_KEY: None})
    self.assertAlmostEqual(applied_long_delay(CP, high), MOONPILOT_LAG_MAX, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, low), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_wide_history_is_rejected_by_modeld_and_planner(self):
    wide = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.40, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED,
                                       MOONPILOT_LAG_HISTORY_KEY: [0.10, 0.20, 0.50, 0.55, 0.60]})
    planner = LongLagEstimator(CP)
    planner.restore(wide)
    self.assertEqual(planner.status, "unestimated")
    self.assertAlmostEqual(applied_long_delay(CP, wide), CP.longitudinalActuatorDelay, delta=1e-9)

  def test_tight_history_is_applied_by_modeld_and_planner(self):
    tight = params(on=False, overrides={MOONPILOT_LAG_KEY: 0.40, MOONPILOT_LAG_BLOCKS_KEY: MOONPILOT_LAG_BLOCKS_NEEDED,
                                        MOONPILOT_LAG_HISTORY_KEY: [0.39, 0.40, 0.40, 0.41, 0.40]})
    planner = LongLagEstimator(CP)
    planner.restore(tight)
    self.assertEqual(planner.status, "estimated")
    self.assertAlmostEqual(planner.applied_delay(), 0.40, delta=1e-9)
    self.assertAlmostEqual(applied_long_delay(CP, tight), 0.40, delta=1e-9)


if __name__ == "__main__":
  unittest.main()
