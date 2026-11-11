"""What the runtime does with a committed selection, and the two conventions it must not confuse.

The runtime is the last place a wrong model can reach the car, so these tests are mostly about its
refusals: a bundled entry, a build that is not there, a build made for another protocol, a member
whose bytes have changed. Each returns `None` -- "run upstream's model" -- and writes the reason to
`MoonpilotModelsActive*`, the only thing a panel can show for a load that failed after the boot
commit had already approved it.

The decode conventions get their own group because the two are indistinguishable from a model's
shape: `curvature-100` is a curvature times a hundred, `lateral-accel` is a lateral acceleration,
and applying the wrong one is a wrong steering request, silently. The same numbers through both are
therefore asserted to *differ* in the way the conventions differ.

The pickle round trip checks the fork's writer against the runtime's reader.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from moonpilot import modelprotocols, models
from moonpilot.tests.fakes import FakeParams
from moonpilot.tests.model_store import TempStoreMixin, write_build, write_package
ROOT = Path(__file__).resolve().parents[2]
RECIPE = "a" * 64


class PrevAction:
  """The fields `get_action` reads off `log.ModelDataV2.Action`, without building a message."""

  def __init__(self, curvature: float = 0.0, accel: float = 0.0):
    self.desiredCurvature = curvature
    self.desiredAcceleration = accel


def flat_plan() -> np.ndarray:
  """A straight, steady plan trajectory: 33 samples of the 15-wide plan, which is enough for
  `get_accel_from_plan`/`get_curvature_from_plan` to produce a number."""
  from openpilot.selfdrive.modeld.constants import ModelConstants, Plan

  plan = np.zeros((ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH), dtype=np.float32)
  plan[:, Plan.VELOCITY][:, 0] = 20.0
  plan[:, Plan.T_FROM_CURRENT_EULER][:, 2] = [t * 0.01 for t in range(ModelConstants.IDX_N)]
  return plan[np.newaxis]


def driver_runtime(contract: str, *, lat: float = 0.0, long: float = 0.3):
  from moonpilot import modelruntime

  entry = {
    "kind": "custom",
    "id": RECIPE,
    "protocol": modelprotocols.SUPERCOMBO.id,
    "build": "/nonexistent",
    "members": {},
    "configuration": {"LAT_SMOOTH_SECONDS": lat, "LONG_SMOOTH_SECONDS": long},
  }
  return modelruntime.DriverRuntime(entry, {"action_contract": contract}, None)


class TestActionConventions(unittest.TestCase):
  OUTPUT = {"action": np.array([[42.0, -1.5]])}
  V_EGO = 20.0

  def test_the_old_convention_divides_by_a_hundred(self):
    # No smoothing (both constants 0), so the head's numbers reach the message unchanged.
    action = driver_runtime("curvature-100", lat=0.0, long=0.0)._action_from_curvature_100(self.OUTPUT, PrevAction(), self.V_EGO)
    # 42 / 100, not 42 / v² = 0.105: that head is a curvature, not an acceleration.
    self.assertAlmostEqual(action.desiredCurvature, 0.42, places=6)
    self.assertAlmostEqual(action.desiredAcceleration, -1.5, places=6)
    self.assertFalse(action.shouldStop)

  def test_the_two_conventions_disagree_on_the_same_numbers(self):
    from openpilot.selfdrive.modeld.modeld import get_action_from_model

    current = get_action_from_model({"action": self.OUTPUT["action"], "plan": flat_plan()}, PrevAction(), 0.15, 0.15, self.V_EGO)
    old = driver_runtime("curvature-100", lat=0.0, long=0.0)._action_from_curvature_100(self.OUTPUT, PrevAction(), self.V_EGO)
    self.assertAlmostEqual(current.desiredCurvature, 42.0 / (20.0**2), places=6)
    self.assertNotAlmostEqual(current.desiredCurvature, old.desiredCurvature, places=3)

  def test_the_current_function_carries_the_plan_contract_too(self):
    # `plan` and `lateral-accel` both end in upstream's function; what this pins is that the runtime
    # does not re-implement it, and that a plan with no `action` slice still decodes.
    runtime = driver_runtime("plan")
    with mock.patch("openpilot.selfdrive.modeld.modeld.get_action_from_model", return_value="upstream") as upstream:
      self.assertEqual(runtime.get_action({"plan": flat_plan()}, PrevAction(), 0.1, 0.1, 5.0), "upstream")
    self.assertEqual(upstream.call_count, 1)

  def test_the_builds_smoothing_constants_are_applied(self):
    # 0.3 s of longitudinal smoothing means the first frame carries a fraction of the ask, which is
    # the whole reason the constants come from the build rather than from a fork default.
    smoothed = driver_runtime("curvature-100", long=0.3)._action_from_curvature_100(self.OUTPUT, PrevAction(), self.V_EGO)
    self.assertLess(smoothed.desiredAcceleration, 0.0)
    self.assertGreater(smoothed.desiredAcceleration, -1.5)
    unsmoothed = driver_runtime("curvature-100", long=0.0)._action_from_curvature_100(self.OUTPUT, PrevAction(), self.V_EGO)
    self.assertAlmostEqual(unsmoothed.desiredAcceleration, -1.5, places=6)

  def test_should_stop_needs_to_be_slow_and_asking_to_stop(self):
    slow = driver_runtime("curvature-100")._action_from_curvature_100({"action": np.array([[0.0, -0.5]])}, PrevAction(), 0.1)
    self.assertTrue(slow.shouldStop)
    moving = driver_runtime("curvature-100")._action_from_curvature_100({"action": np.array([[0.0, -0.5]])}, PrevAction(), 5.0)
    self.assertFalse(moving.shouldStop)

  def test_below_the_control_speed_the_previous_curvature_is_held(self):
    action = driver_runtime("curvature-100")._action_from_curvature_100({"action": np.array([[42.0, 0.0]])}, PrevAction(curvature=0.123), 0.1)
    self.assertAlmostEqual(action.desiredCurvature, 0.123, places=6)


class TestEntries(TempStoreMixin, unittest.TestCase):
  """`runtime_from_entry` is the seam's whole decision: `None` means upstream's model runs."""

  def setUp(self):
    super().setUp()
    from moonpilot import modelruntime

    self.runtime = modelruntime
  def entry(self, *, selection: str = RECIPE, **overrides) -> dict:
    """A committed entry, with the build directory its own selection would have produced."""
    entry = {
      "kind": "custom",
      "id": selection,
      "protocol": modelprotocols.SUPERCOMBO.id,
      "build": models.build_dir(selection, modelprotocols.SUPERCOMBO.id, models.CAMERA_RESOLUTIONS),
      "members": {},
      "configuration": {},
    }
    entry.update(overrides)
    return entry


  def test_a_bundled_entry_runs_upstream(self):
    params = FakeParams()
    self.assertIsNone(self.runtime.runtime_from_entry(modelprotocols.DRIVING, {"kind": "bundled", "id": ""}, params))
    self.assertTrue(models.ACTIVE_DRIVING_KEY not in params.values)

  def test_a_malformed_entry_is_not_a_verdict_on_the_car(self):
    for entry in (None, {}, {"kind": "custom"}, {"kind": "custom", "id": RECIPE, "protocol": "nope", "build": ""}):
      with self.subTest(entry=entry):
        self.assertIsNone(self.runtime.runtime_from_entry(modelprotocols.DRIVING, entry, FakeParams()))

  def test_a_missing_build_falls_back_with_the_reason(self):
    params = FakeParams()
    self.assertIsNone(self.runtime.runtime_from_entry(modelprotocols.DRIVING, self.entry(), params))
    self.assertTrue(str(params.values[models.ACTIVE_DRIVING_KEY]).startswith("stock: "))

  def test_a_record_without_a_contract_decodes_from_the_plan(self):
    # A build written before the field existed falls back to upstream's own decode -- never to the
    # old split family's `/100`, which would be a wrong steering request on a model nobody checked.
    self.selection = write_package("supercombo")
    write_build(self.selection, modelprotocols.SUPERCOMBO.id, action_contract=None)
    runtime = self.runtime.runtime_from_entry(modelprotocols.DRIVING, self.entry(selection=self.selection), FakeParams())
    assert isinstance(runtime, self.runtime.DriverRuntime)
    self.assertEqual(runtime.action_contract, "plan")

  def test_a_build_of_another_protocol_is_refused(self):
    self.selection = write_package("supercombo")
    write_build(self.selection, modelprotocols.SUPERCOMBO.id, record_protocol=modelprotocols.SPLIT_VISION_OFF_ON.id,
                action_contract="lateral-accel")
    params = FakeParams()
    self.assertIsNone(self.runtime.runtime_from_entry(modelprotocols.DRIVING, self.entry(selection=self.selection), params))
    self.assertTrue("another protocol" in str(params.values[models.ACTIVE_DRIVING_KEY]))

  def test_the_wrong_kind_never_constructs_a_runtime(self):
    # The boot commit already refuses this; the runtime refuses it again because the commit ran in
    # another process, and a driving runtime handed to `dmonitoringmodeld` would load a driving
    # pickle into the monitoring path.
    write_build(RECIPE, modelprotocols.SUPERCOMBO.id, action_contract="lateral-accel")
    params = FakeParams()
    self.assertIsNone(self.runtime.runtime_from_entry(modelprotocols.MONITORING, self.entry(), params))

  def test_a_verified_build_yields_the_runtime_for_its_kind(self):
    self.selection = write_package("supercombo")
    write_build(self.selection, modelprotocols.SUPERCOMBO.id, action_contract="lateral-accel")
    runtime = self.runtime.runtime_from_entry(modelprotocols.DRIVING, self.entry(selection=self.selection), FakeParams())
    assert isinstance(runtime, self.runtime.DriverRuntime)
    self.assertEqual(runtime.protocol, modelprotocols.SUPERCOMBO.id)
    self.assertEqual(runtime.smoothness, (0.0, 0.3))
    # The contract comes from the record, because that is what the model's `action` head means.
    self.assertEqual(runtime.action_contract, "lateral-accel")


class TestLoadFailure(TempStoreMixin, unittest.TestCase):
  """The path the boot commit cannot cover: the build is there, it is just not loadable.

  This is the on-device corruption case -- a truncated or overwritten `model.pkl` -- and what has to
  happen is the bundled model, not a crash loop and not a half-loaded one. `create_model` is where
  the two meet: it returns `None` and the reason is left on the Active param for the panel.
  """

  def setUp(self):
    super().setUp()
    from moonpilot import modelruntime

    self.runtime = modelruntime
    self.params = FakeParams()
    self.build = os.path.join(self.tmp.name, "build")
    os.makedirs(self.build, exist_ok=True)
    with open(os.path.join(self.build, "model.pkl"), "wb") as handle:
      handle.write(b"not a pickle at all")
    self.entry = {"kind": "custom", "id": RECIPE, "protocol": modelprotocols.SUPERCOMBO.id, "build": self.build,
                  "members": {}, "configuration": {"LAT_SMOOTH_SECONDS": 0.0, "LONG_SMOOTH_SECONDS": 0.3}}
  def test_a_corrupt_build_falls_back_to_the_bundled_model(self):
    runtime = self.runtime.DriverRuntime(self.entry, {"action_contract": "lateral-accel"}, self.params)
    self.assertIsNone(runtime.create_model(1928, 1208))
    self.assertTrue(str(self.params.values[models.ACTIVE_DRIVING_KEY]).startswith("stock: "))

  def test_the_monitoring_half_fails_the_same_way(self):
    runtime = self.runtime.MonitoringRuntime({**self.entry, "protocol": modelprotocols.DMONITORING.id}, self.params)
    self.assertIsNone(runtime.create_model(1928, 1208))
    self.assertTrue(str(self.params.values[models.ACTIVE_MONITORING_KEY]).startswith("stock: "))



class TestMonitoringParse(unittest.TestCase):
  """The DM heads the tree reads, and the ones a legacy model does not have."""

  class Model:
    def __init__(self, heads):
      self.output_slices = {name: slice(0, 1) for name in heads}

  SHARED = (
    "face_descs_lhd",
    "face_descs_rhd",
    "face_prob_lhd",
    "face_prob_rhd",
    "left_eye_prob_lhd",
    "right_eye_prob_lhd",
    "left_blink_prob_lhd",
    "right_blink_prob_lhd",
    "sunglasses_prob_lhd",
    "using_phone_prob_lhd",
    "wheel_on_right",
  )

  def sliced(self, heads) -> dict:
    return {name: np.full((1, 1), 2.0, dtype=np.float32) for name in heads if "face_descs" not in name} | {
      name: np.full((1, 12), 0.5, dtype=np.float32) for name in heads if "face_descs" in name
    }

  def test_the_sleep_head_is_optional_and_the_rest_are_identical(self):
    from moonpilot import modelruntime

    runtime = modelruntime.MonitoringRuntime(
      {"kind": "custom", "id": RECIPE, "protocol": modelprotocols.DMONITORING.id, "build": "/nonexistent", "members": {}, "configuration": {}}, None
    )
    with_sleep = tuple(self.SHARED) + ("sleep_prob_lhd", "sleep_prob_rhd")
    old = runtime.parse(self.sliced(with_sleep), self.Model(with_sleep))
    without_sleep = tuple(self.SHARED)
    new = runtime.parse(self.sliced(without_sleep), self.Model(without_sleep))

    for head, value in old.items():
      if head.startswith("sleep_prob"):
        continue
      self.assertTrue(np.array_equal(new[head], value), head)
    # `fill_driver_data` reads this key unconditionally, so a model without the head supplies a
    # zero rather than raising -- the same value the capnp default would have been.
    self.assertEqual(new["sleep_prob_lhd"].tolist(), [[0.0]])
    self.assertEqual(new["sleep_prob_rhd"].tolist(), [[0.0]])
    self.assertTrue(np.allclose(old["sleep_prob_lhd"], 1.0 / (1.0 + np.exp(-2.0))))

  def test_a_sigmoid_head_is_not_pass_through(self):
    from moonpilot import modelruntime

    runtime = modelruntime.MonitoringRuntime(
      {"kind": "custom", "id": RECIPE, "protocol": modelprotocols.DMONITORING.id, "build": "/nonexistent", "members": {}, "configuration": {}}, None
    )
    heads = ("wheel_on_right",)
    parsed = runtime.parse(self.sliced(heads), self.Model(heads))
    self.assertTrue(np.allclose(parsed["wheel_on_right"], 1.0 / (1.0 + np.exp(-2.0))))


class TestBuildAgreement(unittest.TestCase):
  """The fork's pickle writer and runtime reader must agree."""



  def test_dump_oob_round_trips_through_runtime_codec(self):
    from moonpilot.modelpickle import dump_oob, load_oob

    payload = {"metadata": {"input_shapes": {"img": [1, 12, 128, 256]}}, "values": np.arange(16, dtype=np.float32).reshape(4, 4)}
    with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryFile() as handle:
      dump_oob(payload, handle, scratch)
      handle.seek(0)
      loaded = load_oob(handle)
    self.assertEqual(sorted(loaded), sorted(payload))
    values = loaded["values"]
    assert isinstance(values, np.ndarray)
    self.assertTrue(np.array_equal(values, payload["values"]))
    self.assertEqual(loaded["metadata"], payload["metadata"])

  def test_the_queues_are_rebuilt_from_the_record(self):
    # The compiled JIT is called with these buffers: a shape the runtime re-derives differently from
    # the compiler is a reshape error on the road, so the record is the only source and this pins
    # that the runtime reads it rather than recomputing it.
    from moonpilot import modelruntime

    metadata = {
      "queue_shapes": [["img_q", [8, 6, 128, 256], "uint8"], ["big_img_q", [8, 6, 128, 256], "uint8"]],
      "npy_shapes": [["tfm", [3, 3]], ["big_tfm", [3, 3]], ["prev_feat", [1, 512]]],
      "packed_npy_size": (9 + 9 + 512) * 4,
      "input_keys": ["img_q", "big_img_q", "packed_npy_inputs"],
    }
    queues, npy, frame_views = modelruntime.make_queues(metadata, "NPY", 4096)
    # The packed buffer is the one queue the record does not carry a shape for: its size is the
    # recorded float region plus the camera's own frame copy size.
    self.assertEqual(sorted(queues), ["big_img_q", "img_q", "packed_npy_inputs"])
    self.assertEqual(queues["packed_npy_inputs"].shape, (metadata["packed_npy_size"] + 2 * 4096,))
    self.assertEqual(list(npy), ["tfm", "big_tfm", "prev_feat"])
    self.assertEqual(npy["prev_feat"].shape, (1, 512))
    self.assertEqual(frame_views["img"].shape, (4096,))
    self.assertIs(frame_views["img"].base, frame_views["big_img"].base)


class TestSeamFallback(unittest.TestCase):
  """Both seams drop the runtime when a build will not load, and that is the whole safety argument
  for a failed load: `create_model` returns `None`, so a runtime left in place would drive the
  *bundled* model through the custom runtime's own smoothing constants and action decode -- a
  `curvature-100` build divides a head the bundled model does not have, which is a wrong steering
  request rather than a missing one. `moonpilot/modelruntime.py` cannot enforce this: the collapse
  is in the seam, and `main` needs visionipc and a camera to run, so the assertion here is on the
  seam's own lines -- the same shape `test_offroad.py` uses for `hardwared`, whose seam is likewise
  unreachable from a unit test.
  """

  SEAMS = {
    "openpilot/selfdrive/modeld/modeld.py": [
      # the load, the collapse, and the two consumers that must see the collapse
      "lat_smooth, long_smooth = runtime.smoothness if runtime is not None else (LAT_SMOOTH_SECONDS, LONG_SMOOTH_SECONDS)",
      "action = (runtime.get_action(model_output, prev_action, lat_action_t, long_action_t, v_ego) if runtime is not None",
    ],
    "openpilot/selfdrive/modeld/dmonitoringmodeld.py": [
      "model_output = runtime.parse(model_output, model) if runtime is not None else parse_model_output(model_output)",
    ],
  }

  def test_a_failed_load_collapses_the_runtime_in_both_seams(self):
    for path, fragments in self.SEAMS.items():
      with self.subTest(seam=path):
        text = (ROOT / path).read_text()
        for fragment in fragments:
          assert fragment in text, f"{path} no longer collapses the runtime on a failed load"
        assert re.search(r"(?m)^[ ]+if model is None:[^\n]*\n[ ]+runtime = None", text), f"{path} does not clear runtime after failed load"


class TestLogImport(unittest.TestCase):
  def test_the_runtime_builds_capnp_actions_not_dicts(self):
    # `get_action`'s return value is assigned straight onto `modelV2.action`, so it has to be the
    # capnp struct the loop publishes; a dict would be a capnp type error at 20 Hz.
    action = driver_runtime("curvature-100", long=0.0)._action_from_curvature_100({"action": np.array([[0.0, 0.0]])}, PrevAction(), 1.0)
    self.assertLessEqual({"desiredCurvature", "desiredAcceleration", "shouldStop"}, set(action.to_dict()))
    self.assertEqual(action.to_dict()["shouldStop"], False)


if __name__ == "__main__":
  unittest.main()
