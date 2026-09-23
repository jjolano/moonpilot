import contextlib
import unittest
import numpy as np
import pyray as rl
from types import SimpleNamespace
from unittest import mock

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_DANGER_FACTOR
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.ui.ui_state import ui_state
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.modeld.constants import ModelConstants
from moonpilot.lead import (
  MOONPILOT_INPATH_GRID,
  MOONPILOT_INPATH_HORIZON,
  MOONPILOT_INPATH_RC,
  MOONPILOT_LEAD_ACCEL_MIN_SAMPLES,
  MOONPILOT_LEAD_ACCEL_FAST_STREAK,
  MOONPILOT_LEAD_ACCEL_TAU,
  MOONPILOT_LEAD_ACCEL_WINDOW,
  MOONPILOT_LEAD_PATH_PX_PER_M,
  MOONPILOT_LEAD_PATH_WIDTH,
  MOONPILOT_LEAD_PROB_GATE,
  MOONPILOT_LEAD_PROB_RC,
  MOONPILOT_LEAD_SPEED_JUMP,
  MOONPILOT_MIN_Y_STD,
  MOONPILOT_OUT_OF_PATH_DANGER,
  MOONPILOT_PATH_HALF_WIDTH,
  MOONPILOT_RADAR_TO_CAMERA,
  LeadAccelEstimator,
  lead_danger_factor,
  lead_in_path,
  lead_in_path_prob,
  lead_yaw_rel,
  normalize_lead,
  resample,
)
from moonpilot.tests.fakes import _params

STRAIGHT_PATH_X = list(np.linspace(0.0, 60.0, 33))
STRAIGHT_PATH_Y = [0.0] * 33

# A model-style lead: 6 samples, 2 s apart, drifting laterally.
LEAD_T = ModelConstants.LEAD_T_IDXS


class ModelLead:
  def __init__(self, x, y, v=None, y_std=None, prob=0.9):
    n = len(x)
    self.t = list(LEAD_T[:n])
    self.x = list(x)
    self.y = list(y)
    self.v = list(v if v is not None else [10.0] * n)
    self.a = [0.0] * n
    self.xStd = [1.0] * n
    self.yStd = list(y_std if y_std is not None else [0.2] * n)
    self.vStd = [1.0] * n
    self.aStd = [1.0] * n
    self.prob = prob
    self.probTime = 0.0


class FusedLead:
  def __init__(self, d_rel, y_rel, v_lead=10.0, a_lead=0.0, present=True, radar=False, model_prob=0.9, a_lead_tau=1.5, track_id=0):
    self.dRel = d_rel
    self.yRel = y_rel
    self.vLead = v_lead
    self.aLeadK = a_lead
    self.present = present
    self.radar = radar
    self.modelProb = model_prob
    self.aLeadTau = a_lead_tau
    self.radarTrackId = track_id


class FakeSubMaster:
  def __init__(self, leads, valid=True, alive=True):
    self.valid = {"moonpilotState": valid}
    self.alive = {"moonpilotState": alive}
    self._leads = leads

  def __getitem__(self, s):
    assert s == "moonpilotState"
    return self

  @property
  def leads(self):
    return self._leads


class RendererSubMaster:
  def __init__(self, state, valid=True, alive=True):
    self.recv_frame = {"extrinsicsCalibration": 1, "modelV2": 1}
    self.updated = {"carParams": False, "modelV2": False, "radarState": False}
    self.valid = {"moonpilotState": valid, "radarState": False}
    self.alive = {"moonpilotState": alive}
    self._values = {
      "carOutput": SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)),
      "extrinsicsCalibration": SimpleNamespace(height=[0.0]),
      "modelV2": SimpleNamespace(),
      "moonpilotState": state,
      "radarState": None,
      "selfdriveState": SimpleNamespace(experimentalMode=False),
    }

  def __getitem__(self, service):
    return self._values[service]


class FakeLead:
  def __init__(self, present=True, in_path=1.0):
    self.present = present
    self.inPath = in_path


def _filter(x0=1.0):
  return FirstOrderFilter(x0, MOONPILOT_INPATH_RC, DT_MDL)


def _prob_filter(x0=1.0):
  """The gate's own filter (radard's shape), separate from the inPath one."""
  return FirstOrderFilter(x0, MOONPILOT_LEAD_PROB_RC, DT_MDL)


class TestResample(unittest.TestCase):
  def test_interpolates(self):
    np.testing.assert_allclose(resample([0.0, 2.0], [0.0, 10.0], [1.0]), [5.0])

  def test_too_short_is_empty(self):
    assert resample([0.0], [1.0], [0.0, 1.0]).size == 0


class TestRadardMirrors(unittest.TestCase):
  """Every constant lead.py mirrors instead of importing, pinned to the radard line it copies.

  The reason to mirror rather than import is that `radard.py` pulls messaging and opendbc while this
  module is on plannerd's and both renderers' import path — but that only holds if drift fails
  loudly. `_LEAD_ACCEL_TAU` has a module-level constant in radard to compare against; the filter's
  time constant and the reset threshold are inline literals there (`radard.py:55,76`), so nothing can
  be asserted about them and the estimator's docstring is the only record.
  """

  def test_radar_to_camera_matches_radard(self):
    # upstream changing radard's value must fail here rather than silently shifting published x
    from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

    self.assertEqual(MOONPILOT_RADAR_TO_CAMERA, RADAR_TO_CAMERA)

  def test_lead_accel_tau_matches_radard(self):
    from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

    self.assertEqual(MOONPILOT_LEAD_ACCEL_TAU, _LEAD_ACCEL_TAU)


class TestLeadAccelEstimator(unittest.TestCase):
  """The estimator that replaced radard's `aLeadK` as the planner's lead accel.

  Every case feeds a constant-accel speed history with `aLeadK = 0.0` (or a sentinel), so a return
  that reads the field instead of the window cannot pass.
  """

  @staticmethod
  def _feed(estimator, v0=20.0, a=-3.0, frames=MOONPILOT_LEAD_ACCEL_WINDOW, a_lead=0.0, track_id=0, present=True, radar=True):
    """Feed `frames` samples of v = v0 + a*i*DT_MDL, oldest first; returns the last (accel, tau)."""
    out = None
    for i in range(frames):
      out = estimator.update(FusedLead(35.0, 0.0, v_lead=v0 + a * i * DT_MDL, a_lead=a_lead, present=present, radar=radar, track_id=track_id))
    return out

  @classmethod
  def _accel(cls, *args, **kwargs):
    """Just the accel, for the cases whose subject is the slope rather than the decay."""
    return cls._feed(*args, **kwargs)[0]

  def test_a_braking_lead_is_recognized_within_the_window(self):
    """The whole point: a -3 m/s^2 ramp reads as -3 m/s^2 after three frames, where radard's filter
    is at -0.12 m/s^2 on that same frame and needs 1.2 s to reach 90 % of it."""
    self.assertAlmostEqual(self._accel(LeadAccelEstimator(), frames=MOONPILOT_LEAD_ACCEL_WINDOW), -3.0, delta=0.01)
    self.assertLess(self._accel(LeadAccelEstimator(), frames=MOONPILOT_LEAD_ACCEL_MIN_SAMPLES), -2.0)

  @staticmethod
  def _step_times(bare_window=False):
    """Settled follow, then a step onto -3.5 m/s^2 at frame 0 of the second loop. Times are measured
    from that frame; `bare_window` is the same estimator with the onset window made inert, which is
    the arm this change is measured against."""
    patch = mock.patch("moonpilot.lead.MOONPILOT_LEAD_ACCEL_FAST_STREAK", 10**9) if bare_window else contextlib.nullcontext()
    hit = {}
    with patch:
      estimator = LeadAccelEstimator()
      for _ in range(30):  # 1.5 s settled, so the full window is full of pre-onset samples
        estimator.update(FusedLead(35.0, 0.0, v_lead=20.0, radar=True))
      for k in range(15):
        a_lead, _ = estimator.update(FusedLead(35.0, 0.0, v_lead=20.0 - 3.5 * k * DT_MDL, radar=True))
        for threshold in (-2.0, -3.0):
          if threshold not in hit and a_lead <= threshold:
            hit[threshold] = k * DT_MDL
    return hit

  def test_the_onset_window_reads_a_brake_before_the_full_window_does(self):
    """The bias the onset window exists for: after a step the seven-sample slope is still half
    pre-onset samples, so it needs 0.20 s to pass -2.0 m/s^2 and 0.25 s to pass -3.0, where the
    newest three samples have it in 0.10 s. Both arms are asserted, so a change that made the bare
    window slower would fail here too — and the closed-loop consequence of this estimator difference
    is pinned in test_longitudinal's braking-lead test."""
    with_window = self._step_times()
    bare = self._step_times(bare_window=True)

    self.assertLessEqual(with_window[-2.0], 0.10)
    self.assertLessEqual(with_window[-3.0], 0.10)
    self.assertLessEqual(bare[-2.0], 0.20)
    self.assertLessEqual(bare[-3.0], 0.25)
    for threshold in (-2.0, -3.0):
      self.assertLess(with_window[threshold], bare[threshold], f"the onset window did not beat the bare window at {threshold}")

  def test_the_onset_window_can_only_deepen_the_estimate(self):
    """The property the policy rests on. `moonpilot/longitudinal.py` only ever uses `a_lead` to add
    braking (`min(a_lead, 0.0)` in the preview, and the floor/TTC terms key off the resulting closing
    rate), so a deeper estimate is more braking and a shallower one is less — which makes "may only
    deepen" the direction that cannot cost safety. It is also rare on a signal with no real onset:
    11 of 600 frames of a 0.05 m/s random walk, the deepest of them 1.67 m/s^2.
    """
    rng = np.random.default_rng(0)
    v = 20.0 + np.cumsum(rng.normal(0.0, 0.05, 600))
    series = []
    for streak in (MOONPILOT_LEAD_ACCEL_FAST_STREAK, 10**9):
      estimator = LeadAccelEstimator()
      with mock.patch("moonpilot.lead.MOONPILOT_LEAD_ACCEL_FAST_STREAK", streak):
        series.append(np.array([estimator.update(FusedLead(35.0, 0.0, v_lead=float(x), radar=True))[0] for x in v]))

    self.assertFalse(bool((series[0] > series[1] + 1e-12).any()), "the onset window lifted the estimate")
    self.assertLess(float((series[0] < series[1] - 1e-12).mean()), 0.05)

  def test_a_lone_deep_frame_is_not_an_onset(self):
    """The shape the streak gate is for: one frame of radar noise — 0.5 m/s at 20 m/s, a -10 m/s^2
    spike in the newest sample — must leave the estimate exactly where the bare window leaves it.
    The next frame's short window reads the recovery, so the streak never reaches two."""
    v = np.array([20.0] * 20 + [19.5] + [20.0] * 20)
    series = []
    for streak in (MOONPILOT_LEAD_ACCEL_FAST_STREAK, 10**9):
      estimator = LeadAccelEstimator()
      with mock.patch("moonpilot.lead.MOONPILOT_LEAD_ACCEL_FAST_STREAK", streak):
        series.append(np.array([estimator.update(FusedLead(35.0, 0.0, v_lead=float(x), radar=True))[0] for x in v]))

    np.testing.assert_array_equal(series[0], series[1])

  def test_noise_is_averaged_not_amplified(self):
    """0.05 m/s of measurement noise on a steady follower: bounded, and far below the jerk limits.
    A one-frame difference would put 1.4 m/s^2 of jitter on the same input."""
    estimator = LeadAccelEstimator()
    rng = np.random.default_rng(0)
    estimates = np.array([estimator.update(FusedLead(35.0, 0.0, v_lead=20.0 + rng.normal(0.0, 0.05)))[0] for _ in range(200)])
    self.assertLess(float(np.abs(estimates).max()), 0.6)
    self.assertLess(float(estimates[MOONPILOT_LEAD_ACCEL_MIN_SAMPLES:].std()), 0.25)

  def test_a_vision_lead_keeps_the_model_accel(self):
    """Only a radar-matched lead has a speed history to fit; the model's own accel is already there."""
    estimator = LeadAccelEstimator()
    for frame in range(10):
      self.assertEqual(self._accel(estimator, frames=1, a_lead=-1.7, radar=False, v0=20.0 - 3.0 * frame * DT_MDL), -1.7)

  def test_a_new_track_does_not_inherit_the_old_one(self):
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._accel(estimator), -3.0, delta=0.01)
    # a different radar track entirely: radard's value stands until this track's own window fills
    self.assertEqual(self._accel(estimator, frames=1, v0=20.0 - step, a_lead=0.0, track_id=1), 0.0)
    self.assertAlmostEqual(self._accel(estimator, frames=2, v0=20.0 - 2 * step, track_id=1), -3.0, delta=0.01)

  def test_a_speed_jump_is_reassociation_not_braking(self):
    """A lead that re-associates 3.5 m/s away is a new object, not a 70 m/s^2 brake."""
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._accel(estimator), -3.0, delta=0.01)
    jumped = 20.0 - 3.0 * (MOONPILOT_LEAD_ACCEL_WINDOW - 1) * DT_MDL - (MOONPILOT_LEAD_SPEED_JUMP + 1.0)
    self.assertEqual(self._accel(estimator, frames=1, v0=jumped, a_lead=-9.9), -9.9)
    self.assertEqual(self._accel(estimator, frames=1, v0=jumped - step, a_lead=-9.9), -9.9)  # still two samples
    self.assertAlmostEqual(self._accel(estimator, frames=1, v0=jumped - 2 * step, a_lead=-9.9), -3.0, delta=0.01)

  def test_the_decay_describes_the_estimate_it_travels_with(self):
    """The decay must match the accel it is returned with, and the fork's estimate goes deep before
    radard's Kalman does: on a -3.5 m/s^2 onset the fork crosses the 0.5 m/s^2 reset on its third
    sample (0.10 s) where `aLeadK` needs ~0.25 s, so pairing radard's `aLeadTau` with the fork's
    accel is what let the published plan's tail decay the braking away."""
    estimator = LeadAccelEstimator()
    tau = []
    for i in range(MOONPILOT_LEAD_ACCEL_WINDOW):
      a_lead, a_lead_tau = estimator.update(FusedLead(35.0, 0.0, v_lead=20.0 - 3.5 * i * DT_MDL, a_lead_tau=0.3, radar=True))
      tau.append((round(a_lead, 2), round(a_lead_tau, 4)))

    # the first two frames have no window yet: radard's own pair, untouched
    self.assertEqual(tau[0], (0.0, 0.3))
    self.assertEqual(tau[1], (0.0, 0.3))
    # from the third frame the fork's accel is deep, so its own decay takes over. Not a reset to 1.5
    # on that frame: radard decays from 1.5 too (radard.py:76-79), it just starts 0.15 s later
    self.assertLess(tau[2][0], -2.0)
    self.assertEqual(tau[2][1], MOONPILOT_LEAD_ACCEL_TAU * 0.9)
    self.assertTrue(all(a_lead_tau < MOONPILOT_LEAD_ACCEL_TAU for _, a_lead_tau in tau[3:]))
    self.assertTrue(all(later < earlier for (_, earlier), (_, later) in zip(tau[3:], tau[4:], strict=False)))

    # and a lead that is not braking keeps the long decay, radard's own judgment
    settled = LeadAccelEstimator()
    for _ in range(MOONPILOT_LEAD_ACCEL_WINDOW + 5):
      _, a_lead_tau = settled.update(FusedLead(35.0, 0.0, v_lead=20.0, radar=True))
    self.assertEqual(a_lead_tau, MOONPILOT_LEAD_ACCEL_TAU)

  def test_the_decay_re_arms_so_one_brake_does_not_flatten_it_for_the_drive(self):
    """radard re-arms its filter's own state when |a| falls back under the reset threshold
    (`radard.py:77`), not merely the value it reports. Setting only the reported value leaves the
    filter ratcheting toward 0, so after the first hard brake every later onset — and every later
    lead — would report tau ~ 0 and the rollout would be told the accel holds for the whole horizon.
    """
    estimator = LeadAccelEstimator()
    v_lead = 20.0
    for _ in range(40):  # a long brake, which drives the filter's state to ~0
      v_lead -= 3.5 * DT_MDL
      estimator.update(FusedLead(35.0, 0.0, v_lead=v_lead, radar=True))
    self.assertLess(estimator._tau.x, 0.1)  # the state really did collapse

    for _ in range(40):  # two seconds of holding speed
      _, a_lead_tau = estimator.update(FusedLead(35.0, 0.0, v_lead=v_lead, radar=True))
    self.assertEqual(a_lead_tau, MOONPILOT_LEAD_ACCEL_TAU)
    self.assertEqual(estimator._tau.x, MOONPILOT_LEAD_ACCEL_TAU)  # re-armed, not just reported

    # so a second onset decays from the long tau again, rather than from the first brake's residue
    for _ in range(MOONPILOT_LEAD_ACCEL_MIN_SAMPLES):
      v_lead -= 3.5 * DT_MDL
      _, a_lead_tau = estimator.update(FusedLead(35.0, 0.0, v_lead=v_lead, radar=True))
    self.assertGreater(a_lead_tau, 1.0)

  def test_a_new_track_starts_its_decay_fresh(self):
    """radard builds a whole new `Track`, filter included, for a new `radarTrackId`. Inheriting the
    previous lead's collapsed decay would describe this lead's accel as holding far longer than
    anything known about it."""
    estimator = LeadAccelEstimator()
    v_lead = 20.0
    for _ in range(40):
      v_lead -= 3.5 * DT_MDL
      estimator.update(FusedLead(35.0, 0.0, v_lead=v_lead, radar=True, track_id=1))
    self.assertLess(estimator._tau.x, 0.1)

    for i in range(MOONPILOT_LEAD_ACCEL_MIN_SAMPLES):
      _, a_lead_tau = estimator.update(FusedLead(35.0, 0.0, v_lead=13.0 - 3.5 * i * DT_MDL, radar=True, track_id=2))
    # the third frame's estimate is -3.5, so the decay has begun — from 1.5, not from the old track's
    self.assertEqual(a_lead_tau, MOONPILOT_LEAD_ACCEL_TAU * 0.9)

  def test_an_absent_slot_clears_the_window(self):
    """The planner ticks every instance every frame for this reason: an absent or replaced lead may
    not leave samples behind for the next one to inherit."""
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._accel(estimator), -3.0, delta=0.01)
    self.assertEqual(self._accel(estimator, frames=1, present=False, a_lead=-9.9), -9.9)
    self.assertEqual(self._accel(estimator, frames=MOONPILOT_LEAD_ACCEL_MIN_SAMPLES - 1, v0=20.0 - step, a_lead=-9.9), -9.9)
    self.assertAlmostEqual(self._accel(estimator, frames=1, v0=20.0 - 3 * step, a_lead=-9.9), -3.0, delta=0.01)


class TestNormalizeLead(unittest.TestCase):
  def test_anchors_index_zero_to_fused_lead(self):
    model = ModelLead(x=[41.52, 61.52, 81.52], y=[-0.5, -0.6, -0.7])
    out = normalize_lead(0, model, FusedLead(40.0, 0.5), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter())

    assert out["present"]
    self.assertAlmostEqual(out["x"][0], 40.0, places=6)
    self.assertAlmostEqual(out["y"][0], 0.5, places=6)
    self.assertAlmostEqual(out["x"][1], 40.0 + (61.52 - 41.52), places=6)
    self.assertEqual(out["v"], model.v)
    self.assertEqual(out["yStd"], model.yStd)

  def test_model_y_is_right_positive(self):
    # model_y increasing = drifting right = published y decreases, heading turns negative.
    model = ModelLead(x=[41.52 + 20 * i for i in range(3)], y=[-0.5, 0.5, 1.5])
    out = normalize_lead(0, model, FusedLead(40.0, 0.5), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter())

    assert out["y"][1] < out["y"][0]
    assert all(yaw < 0.0 for yaw in out["yawRel"])

  def test_vision_only_uses_raw_conventions(self):
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5])
    out = normalize_lead(2, model, None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter())

    self.assertEqual(out["source"], "vision")
    self.assertAlmostEqual(out["x"][0], 41.52 - 1.52, places=6)
    self.assertAlmostEqual(out["y"][0], 0.5, places=6)

  def test_radar_only_override_has_no_trajectory(self):
    # A low model prob on purpose: radard's low-speed override is a track it saw, not a slot the
    # model rates, so the gate below must not reach this branch.
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5], prob=0.0)
    out = normalize_lead(0, model, FusedLead(12.0, 0.3, radar=True, model_prob=0.0), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter())

    assert out["present"]
    self.assertEqual(out["source"], "radar")
    self.assertEqual(out["t"], [0.0])
    self.assertEqual(out["x"], [12.0])
    self.assertEqual(out["yawRel"], [])
    self.assertEqual(out["inPathProb"], [])
    self.assertEqual(out["inPath"], 1.0)

  def test_a_slot_the_model_does_not_believe_is_published_empty(self):
    """The gate is what keeps a phantom slot off the line and out of the planner's time gap."""
    model = ModelLead(x=[41.52, 61.52, 81.52], y=[-0.5, -0.6, -0.7], prob=MOONPILOT_LEAD_PROB_GATE - 0.01)
    out = normalize_lead(0, model, None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(0.3), _prob_filter(0.0))

    assert not out["present"]
    self.assertEqual(out["source"], "none")
    self.assertEqual(out["x"], [])
    self.assertEqual(out["inPath"], 1.0)

  def test_the_gate_is_the_models_own_probability(self):
    for prob, present in ((MOONPILOT_LEAD_PROB_GATE, True), (MOONPILOT_LEAD_PROB_GATE - 1e-6, False)):
      out = normalize_lead(0, ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5], prob=prob), None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter(0.0))
      self.assertEqual(out["present"], present)

  def test_a_lead_already_believed_survives_one_low_frame(self):
    """Radard's filter shape is why: a rise is instant, a fall decays, so the gate cannot blink the
    line on and off at 20 Hz -- measured on the corpus, 29 % of the crossings under the gate are a
    single frame long."""
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5], prob=0.02)
    prob_filter = _prob_filter(1.0)
    out = normalize_lead(0, model, None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), prob_filter)

    assert out["present"]
    assert MOONPILOT_LEAD_PROB_GATE <= out["prob"] < 1.0, out["prob"]

  def test_a_sustained_low_prob_drops_the_lead(self):
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5], prob=0.02)
    prob_filter = _prob_filter(1.0)
    for _ in range(20):  # 1 s at DT_MDL, five times the filter's own time constant
      out = normalize_lead(0, model, None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), prob_filter)

    assert not out["present"]
    self.assertEqual(out["source"], "none")

  def test_a_fused_lead_is_not_gated_on_the_models_prob(self):
    """The other direction: a track radard published is not the model's to drop, whatever the model
    thinks of the slot behind it."""
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5], prob=0.0)
    out = normalize_lead(0, model, FusedLead(40.0, 0.5, model_prob=0.3), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter(0.0))

    assert out["present"]
    self.assertAlmostEqual(out["prob"], 0.3, places=6)

  def test_missing_data_is_a_planner_no_op(self):
    for model, fused in ((None, None), (ModelLead(x=[], y=[]), None)):
      out = normalize_lead(0, model, fused, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter(), _prob_filter())
      assert not out["present"]
      self.assertEqual(out["source"], "none")
      self.assertEqual(out["inPath"], 1.0)
      self.assertEqual(out["inPathProb"], [])
      self.assertEqual(out["x"], [])

    no_path = normalize_lead(0, ModelLead(x=[41.52], y=[-0.5]), None, [], [], _filter(), _prob_filter())
    assert not no_path["present"]


class TestLeadYawRel(unittest.TestCase):
  def test_straight_lead_has_no_heading(self):
    np.testing.assert_allclose(lead_yaw_rel([0.0, 2.0, 4.0], [0.0, 0.0, 0.0], [10.0] * 3), [0.0] * 3)

  def test_too_short_is_empty(self):
    self.assertEqual(lead_yaw_rel([0.0], [0.0], [10.0]), [])


def _wire(y, y_std=None):
  """A constant-lateral 6-sample lead at 20 m spacing, as lead_in_path's arguments."""
  n = len(LEAD_T)
  return {
    "t": list(LEAD_T),
    "x": [30.0 + 20.0 * i for i in range(n)],
    "y": [y] * n,
    "y_std": [0.2] * n if y_std is None else y_std,
    "ego_path_x": STRAIGHT_PATH_X,
    "ego_path_y": STRAIGHT_PATH_Y,
  }


class TestLeadInPathProb(unittest.TestCase):
  def test_centered_lead_is_in_path(self):
    probs = lead_in_path_prob([30.0], [0.0], [0.2], STRAIGHT_PATH_X, STRAIGHT_PATH_Y)
    assert probs[0] > 0.99, probs

  def test_far_lead_is_out_of_path(self):
    probs = lead_in_path_prob([30.0], [4.0], [0.2], STRAIGHT_PATH_X, STRAIGHT_PATH_Y)
    assert probs[0] < 0.01, probs

  def test_corridor_edge_is_half(self):
    probs = lead_in_path_prob([30.0], [MOONPILOT_PATH_HALF_WIDTH], [MOONPILOT_MIN_Y_STD], STRAIGHT_PATH_X, STRAIGHT_PATH_Y)
    self.assertAlmostEqual(probs[0], 0.5, places=6)

  def test_no_ego_path_is_empty(self):
    self.assertEqual(lead_in_path_prob([30.0], [0.0], [0.2], [], []), [])

  def test_missing_y_std_falls_back_to_the_floor(self):
    probs = lead_in_path_prob([30.0], [0.0], [], STRAIGHT_PATH_X, STRAIGHT_PATH_Y)
    assert probs[0] > 0.99, probs


class TestLeadInPath(unittest.TestCase):
  def _scalar(self, y, y_std=None):
    return lead_in_path(**_wire(y, y_std), filt=_filter(1.0))

  def test_continuity_is_why_this_is_a_probability(self):
    values = [self._scalar(0.1 * i) for i in range(41)]  # y from 0 m to 4 m

    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1)), values
    assert max(abs(values[i + 1] - values[i]) for i in range(len(values) - 1)) <= 0.1, values

  def test_asymmetric_filter(self):
    # Rises instantly, decays one first-order step per frame. A 50 m lateral offset sits
    # outside the Gaussian's support, so its raw probability is exactly 0.0.
    alpha = DT_MDL / (MOONPILOT_INPATH_RC + DT_MDL)
    self.assertAlmostEqual(lead_in_path(**_wire(y=50.0), filt=_filter(1.0)), 1.0 - alpha, places=9)

    # Resting low, one in-path frame restores stock behavior exactly.
    assert lead_in_path(**_wire(y=0.0), filt=_filter(0.0)) == 1.0

  def test_grid_covers_the_cut_in_window(self):
    assert MOONPILOT_INPATH_GRID[0] == 0.0
    assert MOONPILOT_INPATH_GRID[-1] >= 4.0
    assert MOONPILOT_INPATH_GRID[-1] < LEAD_T[-1]


class TestRendererLeadPath(unittest.TestCase):
  """The on-device line cannot be observed headlessly (EGL window init is unavailable here),
  so project a synthetic lead through each tree's own renderer and check the line."""

  W, H = 1920, 1080

  @staticmethod
  def _transform():
    # Synthetic forward camera: u = W/2 + fx*y/x, v = cy + fz*z/x.
    m = np.zeros((3, 3), dtype=np.float32)
    m[0] = [TestRendererLeadPath.W / 2, 1600.0, 0.0]
    m[1] = [350.0, 0.0, 2000.0]
    m[2] = [1.0, 0.0, 0.0]
    return m

  def _renderer(self, tree):
    if tree == "tizi":
      from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
    else:
      from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer

    r = ModelRenderer()
    r.set_transform(self._transform())
    r.set_rect(rl.Rectangle(0, 0, self.W, self.H))
    r._clip_region = rl.Rectangle(-500, -500, self.W + 1000, self.H + 1000)
    x = np.linspace(0.0, 60.0, 33)
    r._path.raw_points = np.array([x, np.zeros_like(x), np.full_like(x, 1.2)], dtype=np.float32).T
    r._path_offset_z = 0.0
    return r

  @staticmethod
  def _state(y, present=True, in_path=1.0, y_std=None):
    state = type("State", (), {})()
    state.leads = []
    if present:
      lead = ModelLead(x=[40.0, 45.0, 50.0, 55.0, 60.0, 65.0], y=y, y_std=y_std)
      lead.present = True
      lead.inPath = in_path
      state.leads = [lead]
    return state

  def _projected(self, tree, y, y_std=None):
    r = self._renderer(tree)
    r._update_lead_path(self._state(y, y_std=y_std), np.linspace(0.0, 60.0, 33).astype(np.float32))
    return r._lead_path.projected_points, r._lead_in_path, r._lead_path_widths

  def _render_lead_path(self, renderer, state, valid=True, alive=True):
    renderer._transform_dirty = True
    sm = RendererSubMaster(state, valid=valid, alive=alive)
    rect = rl.Rectangle(0, 0, self.W, self.H)
    with (
      mock.patch.object(ui_state, "sm", sm),
      mock.patch.object(ui_state, "started_frame", 0),
      mock.patch.object(ui_state, "params", _params()),
      mock.patch.object(renderer, "_update_model"),
      mock.patch.object(renderer, "_draw_lane_lines"),
      mock.patch.object(renderer, "_draw_path"),
      mock.patch.object(renderer, "_draw_lead_path"),
    ):
      renderer._render(rect)

  def test_projects_a_line(self):
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        points, in_path, widths = self._projected(tree, [0.0] * 6)
        assert points.ndim == 2 and points.shape[1] == 2, points.shape
        # One point per 0.5 s sample out to the planner's horizon, and one width per point.
        assert 2 <= points.shape[0] <= int(MOONPILOT_INPATH_HORIZON / 0.5) + 1, points.shape
        assert widths.shape[0] == points.shape[0], (widths.shape, points.shape)
        assert np.isfinite(points).all()
        self.assertEqual(in_path, 1.0)

  def test_line_stops_at_the_planner_horizon(self):
    # The lead's own grid runs 40..65 m over 10 s; the drawn line ends at the 4 s sample.
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        r = self._renderer(tree)
        r._update_lead_path(self._state([0.0] * 6), np.linspace(0.0, 60.0, 33).astype(np.float32))
        raw = r._lead_path.raw_points
        self.assertAlmostEqual(float(raw[0, 0]), 40.0, places=4)
        self.assertAlmostEqual(float(raw[-1, 0]), 50.0, places=4)

  def test_width_comes_from_the_models_own_std(self):
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        floor, ceiling = MOONPILOT_LEAD_PATH_WIDTH
        np.testing.assert_allclose(self._projected(tree, [0.0] * 6, y_std=[0.0] * 6)[2], floor)
        np.testing.assert_allclose(self._projected(tree, [0.0] * 6, y_std=[10.0] * 6)[2], ceiling)
        middle = np.clip(1.0 * MOONPILOT_LEAD_PATH_PX_PER_M, floor, ceiling)
        np.testing.assert_allclose(self._projected(tree, [0.0] * 6, y_std=[1.0] * 6)[2], middle)

  def test_a_short_y_std_falls_back_instead_of_raising(self):
    # resample() is np.interp, which raises unless values is as long as t, so a partially
    # populated yStd must not reach it.
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        points, _, widths = self._projected(tree, [0.0] * 6, y_std=[0.9])
        assert points.shape[0] >= 2, points.shape
        expected = np.clip(MOONPILOT_MIN_Y_STD * MOONPILOT_LEAD_PATH_PX_PER_M, *MOONPILOT_LEAD_PATH_WIDTH)
        np.testing.assert_allclose(widths, expected)

  def test_lateral_prediction_reaches_the_projection(self):
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        centered = self._projected(tree, [0.0] * 6)[0]
        # y is left positive, and screen x grows to the right.
        leftward = self._projected(tree, [2.0] * 6)[0]
        assert leftward[:, 0].mean() < centered[:, 0].mean() - 20, (centered[:, 0].mean(), leftward[:, 0].mean())

  def test_no_lead_or_missing_samples_draws_nothing(self):
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        r = self._renderer(tree)
        path_x = np.linspace(0.0, 60.0, 33).astype(np.float32)

        r._update_lead_path(self._state([0.0] * 6, present=False), path_x)
        assert r._lead_path.projected_points.size == 0
        assert r._lead_path_widths.size == 0

        single = self._state([0.0])
        r._update_lead_path(single, path_x)
        assert r._lead_path.projected_points.size == 0
        assert r._lead_path_widths.size == 0

  def test_toggle_off_clears_the_line(self):
    """The line shows the inPath the fork planner acts on, so it follows that feature's toggle:
    with upstream's planner back in the line there is no decision for it to draw."""
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        r = self._renderer(tree)
        path_x = np.linspace(0.0, 60.0, 33).astype(np.float32)
        r._update_lead_path(self._state([0.0] * 6), path_x)
        assert r._lead_path.projected_points.size > 0

        with mock.patch.object(ui_state, "params", _params(on=False)):
          r._update_lead_path(self._state([0.0] * 6), path_x)
        assert r._lead_path.projected_points.size == 0
        assert r._lead_path_widths.size == 0
        self.assertEqual(r._lead_in_path, 1.0)


  def test_invalid_or_dead_state_clears_and_redraws(self):
    for valid, alive in ((False, True), (True, False)):
      for tree in ("tizi", "mici"):
        with self.subTest(tree=tree, valid=valid, alive=alive):
          r = self._renderer(tree)
          state = self._state([0.0] * 6)

          self._render_lead_path(r, state)
          assert r._lead_path.projected_points.size > 0

          self._render_lead_path(r, state, valid=valid, alive=alive)
          assert r._lead_path.projected_points.size == 0
          assert r._lead_path_widths.size == 0
          self.assertEqual(r._lead_in_path, 1.0)

          self._render_lead_path(r, state)
          assert r._lead_path.projected_points.size > 0


class TestLeadDangerFactor(unittest.TestCase):
  def test_toggle_off_is_the_upstream_default(self):
    sm = FakeSubMaster([FakeLead(in_path=0.0)])
    assert lead_danger_factor(sm, _params(on=False), 0.75) == 0.75

  def test_invalid_state_is_the_upstream_default(self):
    for valid, alive in ((False, True), (True, False)):
      sm = FakeSubMaster([FakeLead(in_path=0.0)], valid=valid, alive=alive)
      assert lead_danger_factor(sm, _params(), 0.75) == 0.75

  def test_no_lead_is_the_upstream_default(self):
    for leads in ([], [FakeLead(present=False)]):
      assert lead_danger_factor(FakeSubMaster(leads), _params(), 0.75) == 0.75

  def test_plain_dict_submaster_is_the_upstream_default(self):
    # The longitudinal maneuver harness passes a dict, not a SubMaster.
    assert lead_danger_factor({"radarState": None}, _params(), 0.75) == 0.75

  def test_in_path_lead_is_bit_identical(self):
    sm = FakeSubMaster([FakeLead(in_path=1.0)])
    assert lead_danger_factor(sm, _params(), 0.75) == 0.75

  def test_out_of_path_lead_relaxes_the_constraint(self):
    sm = FakeSubMaster([FakeLead(in_path=0.0)])
    assert lead_danger_factor(sm, _params(), 0.75) == MOONPILOT_OUT_OF_PATH_DANGER

  def test_halfway_is_linear(self):
    sm = FakeSubMaster([FakeLead(in_path=0.5)])
    expected = (MOONPILOT_OUT_OF_PATH_DANGER + 0.75) / 2.0
    self.assertAlmostEqual(lead_danger_factor(sm, _params(), 0.75), expected, places=6)


if __name__ == "__main__":
  unittest.main()


class PlannerSubMaster:
  """A real planner's inputs, with moonpilotState spliced in — the seam's actual call site."""

  def __init__(self, in_path):
    car_state = messaging.new_message("carState")
    car_state.carState.vEgo = 20.0
    car_state.carState.vCruise = 100.0
    car_state.carState.aEgo = 0.0
    car_state.carState.standstill = False
    car_state.carState.steeringAngleDeg = 0.0

    car_control = messaging.new_message("carControl")
    car_control.carControl.orientationNED = [0.0, 0.0, 0.0]

    controls_state = messaging.new_message("controlsState")
    controls_state.controlsState.forceDecel = False
    controls_state.controlsState.longControlState = log.LongitudinalPlan.LongitudinalPlanSource.lead0

    selfdrive_state = messaging.new_message("selfdriveState")
    selfdrive_state.selfdriveState.experimentalMode = False
    selfdrive_state.selfdriveState.enabled = True

    vehicle_parameters = messaging.new_message("vehicleParameters")
    vehicle_parameters.vehicleParameters.angleOffsetDeg = 0.0

    model = messaging.new_message("modelV2")
    model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6
    model.modelV2.action.desiredAcceleration = 0.0
    model.modelV2.action.shouldStop = False

    radar = messaging.new_message("radarState")
    radar.radarState.leadOne.present = True
    radar.radarState.leadOne.dRel = 60.0
    radar.radarState.leadOne.vLead = 22.0
    radar.radarState.leadOne.aLeadK = 0.0
    radar.radarState.leadOne.aLeadTau = 0.3
    radar.radarState.leadOne.modelProb = 0.9

    self._data = {
      "carState": car_state.carState,
      "carControl": car_control.carControl,
      "controlsState": controls_state.controlsState,
      "selfdriveState": selfdrive_state.selfdriveState,
      "vehicleParameters": vehicle_parameters.vehicleParameters,
      "modelV2": model.modelV2,
      "radarState": radar.radarState,
      "moonpilotState": FakeSubMaster([FakeLead(in_path=in_path)]),
    }
    self.valid = {"moonpilotState": True}
    self.alive = {"moonpilotState": True}

  def __getitem__(self, s):
    return self._data[s]


class TestPlannerSeam(unittest.TestCase):
  """The number the MPC actually solves with, not just the helper's return value."""

  @staticmethod
  def _mpc_danger_factor(in_path, toggle_on=True):
    planner = LongitudinalPlanner(CarInterface.get_non_essential_params(CAR.HONDA_CIVIC), init_v=20.0)
    planner.params = _params(on=toggle_on)
    planner.update(PlannerSubMaster(in_path))
    return float(planner.mpc.params[0, 5])

  def test_in_path_lead_uses_the_upstream_factor(self):
    self.assertEqual(self._mpc_danger_factor(1.0), LEAD_DANGER_FACTOR)

  def test_out_of_path_lead_relaxes_the_mpc_constraint(self):
    self.assertEqual(self._mpc_danger_factor(0.0), MOONPILOT_OUT_OF_PATH_DANGER)

  def test_toggle_off_ignores_the_prediction(self):
    self.assertEqual(self._mpc_danger_factor(0.0, toggle_on=False), LEAD_DANGER_FACTOR)
