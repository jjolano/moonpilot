import unittest
from typing import cast

import numpy as np
import pyray as rl

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LEAD_DANGER_FACTOR
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.modeld.constants import ModelConstants
from moonpilot.lead import (
  MOONPILOT_INPATH_GRID,
  MOONPILOT_INPATH_RC,
  MOONPILOT_LEAD_ACCEL_MIN_SAMPLES,
  MOONPILOT_LEAD_ACCEL_WINDOW,
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


class FakeLead:
  def __init__(self, present=True, in_path=1.0):
    self.present = present
    self.inPath = in_path


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store."""

  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


def _filter(x0=1.0):
  return FirstOrderFilter(x0, MOONPILOT_INPATH_RC, DT_MDL)


class TestResample(unittest.TestCase):
  def test_interpolates(self):
    np.testing.assert_allclose(resample([0.0, 2.0], [0.0, 10.0], [1.0]), [5.0])

  def test_too_short_is_empty(self):
    assert resample([0.0], [1.0], [0.0, 1.0]).size == 0


class TestRadarOffset(unittest.TestCase):
  def test_matches_radard(self):
    # lead.py mirrors this constant instead of importing it, so pin them: upstream changing
    # radard's value must fail here rather than silently shifting published x.
    from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA

    self.assertEqual(MOONPILOT_RADAR_TO_CAMERA, RADAR_TO_CAMERA)


class TestLeadAccelEstimator(unittest.TestCase):
  """The estimator that replaced radard's `aLeadK` as the planner's lead accel.

  Every case feeds a constant-accel speed history with `aLeadK = 0.0` (or a sentinel), so a return
  that reads the field instead of the window cannot pass.
  """

  @staticmethod
  def _feed(estimator, v0=20.0, a=-3.0, frames=MOONPILOT_LEAD_ACCEL_WINDOW, a_lead=0.0, track_id=0, present=True, radar=True):
    """Feed `frames` samples of v = v0 + a*i*DT_MDL, oldest first; returns the last estimate."""
    out = None
    for i in range(frames):
      out = estimator.update(FusedLead(35.0, 0.0, v_lead=v0 + a * i * DT_MDL, a_lead=a_lead, present=present, radar=radar, track_id=track_id))
    return out

  def test_a_braking_lead_is_recognized_within_the_window(self):
    """The whole point: a -3 m/s^2 ramp reads as -3 m/s^2 after three frames, where radard's filter
    is at -0.12 m/s^2 on that same frame and needs 1.2 s to reach 90 % of it."""
    self.assertAlmostEqual(self._feed(LeadAccelEstimator(), frames=MOONPILOT_LEAD_ACCEL_WINDOW), -3.0, delta=0.01)
    self.assertLess(self._feed(LeadAccelEstimator(), frames=MOONPILOT_LEAD_ACCEL_MIN_SAMPLES), -2.0)

  def test_noise_is_averaged_not_amplified(self):
    """0.05 m/s of measurement noise on a steady follower: bounded, and far below the jerk limits.
    A one-frame difference would put 1.4 m/s^2 of jitter on the same input."""
    estimator = LeadAccelEstimator()
    rng = np.random.default_rng(0)
    estimates = np.array([estimator.update(FusedLead(35.0, 0.0, v_lead=20.0 + rng.normal(0.0, 0.05))) for _ in range(200)])
    self.assertLess(float(np.abs(estimates).max()), 0.6)
    self.assertLess(float(estimates[MOONPILOT_LEAD_ACCEL_MIN_SAMPLES:].std()), 0.25)

  def test_a_vision_lead_keeps_the_model_accel(self):
    """Only a radar-matched lead has a speed history to fit; the model's own accel is already there."""
    estimator = LeadAccelEstimator()
    for frame in range(10):
      self.assertEqual(self._feed(estimator, frames=1, a_lead=-1.7, radar=False, v0=20.0 - 3.0 * frame * DT_MDL), -1.7)

  def test_a_new_track_does_not_inherit_the_old_one(self):
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._feed(estimator), -3.0, delta=0.01)
    # a different radar track entirely: radard's value stands until this track's own window fills
    self.assertEqual(self._feed(estimator, frames=1, v0=20.0 - step, a_lead=0.0, track_id=1), 0.0)
    self.assertAlmostEqual(self._feed(estimator, frames=2, v0=20.0 - 2 * step, track_id=1), -3.0, delta=0.01)

  def test_a_speed_jump_is_reassociation_not_braking(self):
    """A lead that re-associates 3.5 m/s away is a new object, not a 70 m/s^2 brake."""
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._feed(estimator), -3.0, delta=0.01)
    jumped = 20.0 - 3.0 * (MOONPILOT_LEAD_ACCEL_WINDOW - 1) * DT_MDL - (MOONPILOT_LEAD_SPEED_JUMP + 1.0)
    self.assertEqual(self._feed(estimator, frames=1, v0=jumped, a_lead=-9.9), -9.9)
    self.assertEqual(self._feed(estimator, frames=1, v0=jumped - step, a_lead=-9.9), -9.9)  # still two samples
    self.assertAlmostEqual(self._feed(estimator, frames=1, v0=jumped - 2 * step, a_lead=-9.9), -3.0, delta=0.01)

  def test_an_absent_slot_clears_the_window(self):
    """The planner ticks every instance every frame for this reason: an absent or replaced lead may
    not leave samples behind for the next one to inherit."""
    estimator = LeadAccelEstimator()
    step = 3.0 * DT_MDL
    self.assertAlmostEqual(self._feed(estimator), -3.0, delta=0.01)
    self.assertEqual(self._feed(estimator, frames=1, present=False, a_lead=-9.9), -9.9)
    self.assertEqual(self._feed(estimator, frames=MOONPILOT_LEAD_ACCEL_MIN_SAMPLES - 1, v0=20.0 - step, a_lead=-9.9), -9.9)
    self.assertAlmostEqual(self._feed(estimator, frames=1, v0=20.0 - 3 * step, a_lead=-9.9), -3.0, delta=0.01)


class TestNormalizeLead(unittest.TestCase):
  def test_anchors_index_zero_to_fused_lead(self):
    model = ModelLead(x=[41.52, 61.52, 81.52], y=[-0.5, -0.6, -0.7])
    out = normalize_lead(0, model, FusedLead(40.0, 0.5), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter())

    assert out["present"]
    self.assertAlmostEqual(out["x"][0], 40.0, places=6)
    self.assertAlmostEqual(out["y"][0], 0.5, places=6)
    self.assertAlmostEqual(out["x"][1], 40.0 + (61.52 - 41.52), places=6)
    self.assertEqual(out["v"], model.v)
    self.assertEqual(out["yStd"], model.yStd)

  def test_model_y_is_right_positive(self):
    # model_y increasing = drifting right = published y decreases, heading turns negative.
    model = ModelLead(x=[41.52 + 20 * i for i in range(3)], y=[-0.5, 0.5, 1.5])
    out = normalize_lead(0, model, FusedLead(40.0, 0.5), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter())

    assert out["y"][1] < out["y"][0]
    assert all(yaw < 0.0 for yaw in out["yawRel"])

  def test_vision_only_uses_raw_conventions(self):
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5])
    out = normalize_lead(2, model, None, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter())

    self.assertEqual(out["source"], "vision")
    self.assertAlmostEqual(out["x"][0], 41.52 - 1.52, places=6)
    self.assertAlmostEqual(out["y"][0], 0.5, places=6)

  def test_radar_only_override_has_no_trajectory(self):
    model = ModelLead(x=[41.52, 61.52], y=[-0.5, -0.5])
    out = normalize_lead(0, model, FusedLead(12.0, 0.3, radar=True, model_prob=0.0), STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter())

    self.assertEqual(out["source"], "radar")
    self.assertEqual(out["t"], [0.0])
    self.assertEqual(out["x"], [12.0])
    self.assertEqual(out["yawRel"], [])
    self.assertEqual(out["inPathProb"], [])
    self.assertEqual(out["inPath"], 1.0)

  def test_missing_data_is_a_planner_no_op(self):
    for model, fused in ((None, None), (ModelLead(x=[], y=[]), None)):
      out = normalize_lead(0, model, fused, STRAIGHT_PATH_X, STRAIGHT_PATH_Y, _filter())
      assert not out["present"]
      self.assertEqual(out["source"], "none")
      self.assertEqual(out["inPath"], 1.0)
      self.assertEqual(out["inPathProb"], [])
      self.assertEqual(out["x"], [])

    no_path = normalize_lead(0, ModelLead(x=[41.52], y=[-0.5]), None, [], [], _filter())
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
  """The on-device ribbon cannot be observed headlessly (EGL window init is unavailable here),
  so project a synthetic lead through each tree's own renderer and check the ribbon."""

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
  def _state(y, present=True, in_path=1.0):
    state = type("State", (), {})()
    state.leads = []
    if present:
      lead = ModelLead(x=[40.0, 45.0, 50.0, 55.0, 60.0, 65.0], y=y)
      lead.present = True
      lead.inPath = in_path
      state.leads = [lead]
    return state

  def _projected(self, tree, y):
    r = self._renderer(tree)
    r._update_lead_path(self._state(y), np.linspace(0.0, 60.0, 33).astype(np.float32))
    return r._lead_path.projected_points, r._lead_in_path

  def test_projects_a_ribbon(self):
    for tree in ("tizi", "mici"):
      with self.subTest(tree=tree):
        points, in_path = self._projected(tree, [0.0] * 6)
        assert points.ndim == 2 and points.shape[1] == 2, points.shape
        assert points.shape[0] >= 4, points.shape
        assert points.shape[0] % 2 == 0, points.shape
        assert np.isfinite(points).all()
        # Twenty-one densified samples minus clipping, doubled for the two ribbon edges.
        assert points.shape[0] <= 42, points.shape
        self.assertEqual(in_path, 1.0)

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

        single = self._state([0.0])
        r._update_lead_path(single, path_x)
        assert r._lead_path.projected_points.size == 0


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
