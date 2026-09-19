"""Running a model the catalog published: load it, drive it, decode its action.

This module is the fork side of two seams -- `openpilot/selfdrive/modeld/modeld.py` and
`.../dmonitoringmodeld.py` -- and both of them import it *at call time*, inside `main`, so a broken
copy here cannot stop the stock path from loading. Each factory returns `None` for anything it is
not certain of, and `None` means "run upstream's model", which is the whole contract:

- **The boot snapshot is the input, and nothing else.** `MoonpilotModelsBoot` was written once by
  the manager after it had resolved the selection against the store, verified every artifact against
  its recorded digest and checked that a build exists for exactly this (selection, protocol,
  resolutions). This module re-checks the cheap half of that (a store on the data partition can be
  truncated or edited between the commit and a model process's start): the files exist, their sizes
  match, the build record names the same protocol and parses as built. The expensive half -- the
  digests -- is the commit's job, run a moment earlier in the same boot.
- **Fail at load, fall back, never mid-drive.** Nothing here hot-swaps or caches a prediction: a
  model that cannot load gives `None`, the process runs the bundled model, and
  MoonpilotModelsActive* records which of the two happened. A failure *after* the model is running
  propagates exactly as upstream's does -- the process exits; while `should_run` remains true,
  `ensure_running` leaves it down; the next manager/boot cycle re-runs the same check.
- **The build drives the buffers.** Every queue shape and every packed view is recorded by
  `moonpilot/modelbuild.py` in `model.pkl`'s metadata and rebuilt here from that record, so the JIT
  sees the buffers it was compiled against by construction rather than by two files agreeing about
  arithmetic. The one thing this file derives is the NV12 frame copy size, which is a function of
  the camera resolution and nothing else (upstream's `ModelState` does the same).
- **The action decode is the protocol's, and `curvature-100` is the one that is not upstream's.**
  `plan` and `lateral-accel` both hand the action to upstream `get_action_from_model`, which already
  branches on the presence of an `action` slice; `curvature-100` is the pre-2026-06 convention
  (`action[0, 0] / 100`, the head's first column a curvature times a hundred) that
  `openpilot@9757bf10` used for the split families it carried. Applying the wrong one of the two
  scales is a wrong steering request that no shape in a model can reveal, which is why the protocol
  pins it at admission and the build record carries it forward.
"""

import os
import pickle
import time

import numpy as np
from tinygrad.tensor import Tensor

from openpilot.cereal import log
from openpilot.common.file_chunker import open_file_chunked
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import smooth_value
from openpilot.selfdrive.modeld.compile_modeld import nv12_copy_size
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.helpers import MODELS_DIR, get_tg_input_devices, load_oob
from openpilot.selfdrive.modeld.parse_model_outputs import Parser, safe_exp, sigmoid
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from moonpilot import models
from typing import Any

SEND_RAW_PRED = os.getenv("SEND_RAW_PRED")

MIN_LAT_CONTROL_SPEED = 0.3
DM_PROCESS_NAME = "openpilot.selfdrive.modeld.dmonitoringmodeld"
# The DM warp is model-independent -- it maps the camera's NV12 frame to the model's input size,
# which the build refuses to change (`models.DM_INPUT_SIZE`) -- so it is read from the checkout
# rather than copied into every build.
DM_WARP = "dm_warp_{w}x{h}_tinygrad.pkl"


def _active(kind: str, text: str, params: Any) -> None:
  """Record what actually loaded, for the panel. One writer per param: `modeld` writes the driving
  one, `dmonitoringmodeld` the monitoring one."""
  if params is None:
    return
  params.put(models.ACTIVE_KEY[kind], text, block=True)


def _fallback(kind: str, reason: str, params: Any) -> None:
  cloudlog.warning(f"moonpilot models: running the bundled model, {reason}")
  _active(kind, f"stock: {reason}", params)


def _field_key(names: set[str], role: str, field: str) -> str:
  """The packed-view name for one role's field. A single-role family uses upstream's own names
  (`desire`, `prev_feat`); a split namespaces them per role (`desire_on_policy`). Recorded by the
  compiler as `npy_shapes`; this resolves which convention it used, and a miss is a build this
  runtime cannot drive rather than a wrong buffer."""
  if field in names:
    return field
  if f"{field}_{role}" in names:
    return f"{field}_{role}"
  raise KeyError(f"no packed field for {role}.{field}: {sorted(names)}")


def _make_queues(metadata: dict, device: str, frame_copy_size: int):
  """Rebuild the JIT's input queues and the views into the packed buffer.

  Mirrors upstream `compile_modeld.make_input_queues`, but from the recorded shapes: the compiler
  captured its JIT against exactly these, and the runtime cannot re-derive them without repeating
  the family's layout rules.
  """
  input_queues = {}
  for key, shape, dtype in metadata["queue_shapes"]:
    if dtype == "uint8":
      input_queues[key] = Tensor(np.zeros(tuple(shape), dtype=np.uint8), device=device).contiguous().realize()
    else:
      input_queues[key] = Tensor(np.zeros(tuple(shape), dtype=np.float32), device=device).contiguous().realize()

  packed_size = int(metadata["packed_npy_size"])
  # The packed buffer is the one thing whose size depends on the camera, so it is not in the record:
  # the compiler built its own from the same function and the same resolution.
  packed_input = np.zeros(packed_size + 2 * frame_copy_size, dtype=np.uint8)
  input_queues["packed_npy_inputs"] = Tensor(packed_input, device="NPY").realize()
  npy_inputs = packed_input[:packed_size].view(np.float32)
  shapes = [(name, tuple(shape)) for name, shape in metadata["npy_shapes"]]
  sizes = [int(np.prod(shape)) for _name, shape in shapes]
  npy = {name: part.reshape(shape) for (name, shape), part in zip(shapes, np.split(npy_inputs, np.cumsum(sizes[:-1])), strict=True)}
  frames = packed_input[packed_size:]
  frame_views = {"img": frames[:frame_copy_size], "big_img": frames[frame_copy_size:]}
  return input_queues, npy, frame_views


class DriverModel:
  """One loaded driving build, with upstream `ModelState`'s interface.

  A role's flat output is sliced with that role's own slice map and the results are merged into one
  dict, because that is what the consumers read: `fill_model_msg`, `fill_pose_msg` and the parser
  take names off a single model-output dict and do not care which member produced them.
  """

  def __init__(self, entry: dict, cam_w: int, cam_h: int):
    payload = load_oob(open_file_chunked(os.path.join(entry["build"], "model.pkl")))

    self.metadata = payload["metadata"]
    self.model_device = payload["input_devices"]["model"]
    self.role_order = [str(role) for role in self.metadata["role_order"]]
    self.role_slices = {role: {name: slice(*bounds) for name, bounds in self.metadata["roles"][role]["slices"].items()} for role in self.role_order}
    self.features_role = str(self.metadata["features_role"])
    self.vision_input_names = [str(name) for name in self.metadata["vision_input_names"]]
    self.frame_skip = int(self.metadata["frame_skip"])
    self.output_slices = {name: bounds for role in self.role_order for name, bounds in self.role_slices[role].items()}
    self.policy_roles = [role for role in self.role_order if "desire_pulse" in self.metadata["roles"][role]["input_shapes"]]

    npy_names = {str(name) for name, _shape in self.metadata["npy_shapes"]}
    self.npy_keys = {
      role: {
        "desire": _field_key(npy_names, role, "desire"),
        "traffic_convention": _field_key(npy_names, role, "traffic_convention"),
        "action_t": _field_key(npy_names, role, "action_t") if "action_t" in self.metadata["roles"][role]["input_shapes"] else None,
        "prev_feat": _field_key(npy_names, role, "prev_feat"),
      }
      for role in self.policy_roles
    }

    self.frame_copy_size = nv12_copy_size(*get_nv12_info(cam_w, cam_h)[:3])
    self.run_model = payload["run_model"][(cam_w, cam_h)]
    self.input_queues, self.npy, self.frame_views = _make_queues(self.metadata, self.model_device, self.frame_copy_size)
    self.parser = Parser()
    self.chestnut = False  # a custom selection never loads as chestnut; modeld gates its big-model path on this
    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

  def slice_outputs(self, flat: np.ndarray, role: str) -> dict[str, np.ndarray]:
    return {name: flat[np.newaxis, bounds] for name, bounds in self.role_slices[role].items()}

  def run(self, bufs: dict, transforms: dict[str, np.ndarray], inputs: dict[str, np.ndarray], after_enqueue=None) -> dict[str, np.ndarray]:
    for key, buf in bufs.items():
      np.copyto(self.frame_views[key], np.frombuffer(buf.data, dtype=np.uint8, count=self.frame_copy_size))

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs["desire_pulse"][0] = 0
    pulse = np.where(inputs["desire_pulse"] - self.prev_desire > 0.99, inputs["desire_pulse"], 0)
    for keys in self.npy_keys.values():
      self.npy[keys["desire"]][:] = pulse
      self.npy[keys["traffic_convention"]][:] = inputs["traffic_convention"]
      if keys["action_t"] is not None:
        self.npy[keys["action_t"]][:] = inputs["action_t"]
    self.prev_desire[:] = inputs["desire_pulse"]
    self.npy["tfm"][:, :] = transforms["img"][:, :]
    self.npy["big_tfm"][:, :] = transforms["big_img"][:, :]

    outs = self.run_model(**{key: self.input_queues[key] for key in self.metadata["input_keys"]})
    if after_enqueue is not None:
      after_enqueue()
    flat = {role: out.numpy()[0] for role, out in zip(self.role_order, outs, strict=True)}

    outputs_dict: dict[str, np.ndarray] = {}
    for role in self.role_order:
      outputs_dict.update(self.slice_outputs(flat[role], role))
    outputs_dict = self.parser.parse_outputs(outputs_dict)

    # The recurrence, fed for the next frame: whatever produced `hidden_state` is the feature the
    # policy roles' queues carry.
    features = self.role_slices[self.features_role].get("hidden_state")
    if features is not None:
      for keys in self.npy_keys.values():
        self.npy[keys["prev_feat"]][:] = flat[self.features_role][features]

    if SEND_RAW_PRED:
      outputs_dict["raw_pred"] = np.concatenate([flat[role] for role in self.role_order]).copy()
    return outputs_dict


class DriverRuntime:
  """A verified custom driving selection. `modeld` reads the smoothing constants before it loads
  anything, so they come from the boot entry rather than from the pickle."""

  def __init__(self, entry: dict, record: dict, params: Any):
    self.entry = entry
    self.record = record
    self.selection = str(entry.get("id", ""))
    self.protocol = str(entry.get("protocol", ""))
    self.name = models.entry_name(self.selection) or self.selection[:12]
    self.action_contract = str(record.get("action_contract") or "plan")
    self.configuration = entry.get("configuration") or {}
    self._params: Any = params

  @property
  def smoothness(self) -> tuple[float, float]:
    return (float(self.configuration.get("LAT_SMOOTH_SECONDS", 0.0)), float(self.configuration.get("LONG_SMOOTH_SECONDS", 0.3)))

  def create_model(self, cam_w: int, cam_h: int) -> DriverModel | None:
    try:
      model = DriverModel(self.entry, cam_w, cam_h)
    except Exception as exc:
      cloudlog.exception("moonpilot models: custom model failed to load")
      _active(models.DRIVING, f"stock: {type(exc).__name__}: {exc}", self._params)
      return None
    _active(models.DRIVING, f"{self.name} ({self.protocol})", self._params)
    cloudlog.warning(f"moonpilot models: driving model {self.name} ({self.protocol}) loaded")
    return model

  def get_action(self, model_output: dict[str, np.ndarray], prev_action, lat_action_t: float, long_action_t: float, v_ego: float):
    if self.action_contract == "curvature-100" and "action" in model_output:
      return self._action_from_curvature_100(model_output, prev_action, v_ego)
    if self.action_contract == "curvature-100":
      # A record that says `curvature-100` while the model publishes no `action` is a build whose
      # selection changed under it, and the plan decode is the one that cannot be wrong about a head
      # that is not there. The compiler writes the contract from the members, so this is a guard.
      cloudlog.warning("moonpilot models: no action head for a curvature-100 build, decoding the plan")
    # `plan` and `lateral-accel` are both upstream's own function: it branches on the presence of an
    # `action` slice and decodes `plan` otherwise.
    from openpilot.selfdrive.modeld.modeld import get_action_from_model

    return get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)

  def _action_from_curvature_100(self, model_output: dict[str, np.ndarray], prev_action, v_ego: float):
    """`openpilot@9757bf109e0673eb7481246d7dad91b3a70b51e9`'s decode, verbatim: the head's first
    column is a curvature times a hundred, not a lateral acceleration."""
    lat_smooth, long_smooth = self.smoothness
    unscaled_curvature, desired_accel = model_output["action"][0]
    desired_curvature = unscaled_curvature / 100
    should_stop = v_ego < MIN_LAT_CONTROL_SPEED and desired_accel < 0.1

    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, long_smooth)
    if v_ego > MIN_LAT_CONTROL_SPEED:
      desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, lat_smooth)
    else:
      desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature), desiredAcceleration=float(desired_accel), shouldStop=bool(should_stop))


class MonitoringModel:
  """One loaded driver-monitoring build, with upstream `dmonitoringmodeld.ModelState`'s interface
  (plus `output_slices`, which the seam's `slice_outputs` call reads)."""

  def __init__(self, entry: dict, cam_w: int, cam_h: int):
    with open(os.path.join(entry["build"], "model.pkl"), "rb") as handle:
      payload = pickle.load(handle)

    metadata = payload["metadata"]
    self.model_run = payload["model_run"]
    self.input_shapes = metadata["input_shapes"]
    self.output_slices = {name: slice(*bounds) for name, bounds in metadata["slices"].items()}

    self.DEV = get_tg_input_devices(DM_PROCESS_NAME, chestnut=False)["DEV"]
    self.numpy_inputs = {"calib": np.zeros(self.input_shapes["calib"], dtype=np.float32)}
    self.warp_inputs_np = {"transform": np.zeros((3, 3), dtype=np.float32)}
    self.warp_inputs = {key: Tensor(value, device="NPY") for key, value in self.warp_inputs_np.items()}
    self.frame_buf_params = get_nv12_info(cam_w, cam_h)
    self.tensor_inputs = {key: Tensor(value, device="NPY").realize() for key, value in self.numpy_inputs.items()}
    self._blob_cache: dict[int, Tensor] = {}
    with open(os.path.join(MODELS_DIR, DM_WARP.format(w=cam_w, h=cam_h)), "rb") as handle:
      self.image_warp = pickle.load(handle)

  def run(self, buf, calib: np.ndarray, transform: np.ndarray) -> tuple[np.ndarray, float]:
    self.numpy_inputs["calib"][0, :] = calib

    t1 = time.perf_counter()

    ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
    # There is a ringbuffer of imgs, just cache tensors pointing to all of them
    if ptr not in self._blob_cache:
      self._blob_cache[ptr] = Tensor.from_blob(ptr, (self.frame_buf_params[3],), dtype="uint8", device=self.DEV)

    self.warp_inputs_np["transform"][:] = transform[:]
    self.tensor_inputs["input_img"] = self.image_warp(self._blob_cache[ptr], self.warp_inputs["transform"])

    output = self.model_run(**self.tensor_inputs).numpy().flatten()

    t2 = time.perf_counter()
    return output, t2 - t1


class MonitoringRuntime:
  def __init__(self, entry: dict, record: dict, params: Any):
    self.entry = entry
    self.record = record
    self.selection = str(entry.get("id", ""))
    self.protocol = str(entry.get("protocol", ""))
    self.name = models.entry_name(self.selection) or self.selection[:12]
    self._params: Any = params

  def create_model(self, cam_w: int, cam_h: int) -> MonitoringModel | None:
    try:
      model = MonitoringModel(self.entry, cam_w, cam_h)
    except Exception as exc:
      cloudlog.exception("moonpilot models: monitoring model failed to load")
      _active(models.MONITORING, f"stock: {type(exc).__name__}: {exc}", self._params)
      return None
    _active(models.MONITORING, f"{self.name} ({self.protocol})", self._params)
    cloudlog.warning(f"moonpilot models: monitoring model {self.name} ({self.protocol}) loaded")
    return model

  def parse(self, sliced: dict[str, np.ndarray], model: Any) -> dict[str, Any]:
    """Upstream `dmonitoringmodeld.parse_model_output`, made tolerant of the heads a model does not
    have. `model` only has to expose `output_slices`, which the seam's bundled `ModelState` does
    too; the returned dict is upstream's own loose shape, because the caller adds `raw_pred` bytes.

    `sleep_prob_*` exists only on the newest DM models (553 slices against a legacy 551), and
    `driverStateV2` has no consumer for `features` in this tree, so the one field upstream's
    `fill_driver_data` reads unconditionally is supplied as a zero when the model lacks it -- which
    is what the capnp default would have been anyway."""
    heads = set(model.output_slices)
    parsed: dict[str, np.ndarray] = {}
    if "wheel_on_right" in heads:
      parsed["wheel_on_right"] = sigmoid(sliced["wheel_on_right"])
    for suffix in ("lhd", "rhd"):
      face_descs = sliced.get(f"face_descs_{suffix}")
      if face_descs is not None:
        parsed[f"face_descs_{suffix}"] = face_descs[:, :-6]
        parsed[f"face_descs_{suffix}_std"] = safe_exp(face_descs[:, -6:])
      for head in ("face_prob", "left_eye_prob", "right_eye_prob", "left_blink_prob", "right_blink_prob", "sunglasses_prob", "using_phone_prob"):
        if f"{head}_{suffix}" in heads:
          parsed[f"{head}_{suffix}"] = sigmoid(sliced[f"{head}_{suffix}"])
      if f"sleep_prob_{suffix}" in heads:
        parsed[f"sleep_prob_{suffix}"] = sigmoid(sliced[f"sleep_prob_{suffix}"])
      else:
        parsed[f"sleep_prob_{suffix}"] = np.zeros((1, 1), dtype=np.float32)
    return parsed


def runtime_from_entry(kind: str, entry: Any, params: Any = None):
  """A runtime for a committed custom entry, or `None` to run the bundled model.

  Every reason for `None` is logged, and the ones that mean the driver's selection did not load are
  written to `MoonpilotModelsActive*` so the panel can say so. The checks are the cheap half of the
  boot commit's -- presence and size, not digest -- because the commit hashed these files moments
  earlier in the same boot; a store edited since then fails to load rather than running wrong bytes.
  """
  if not isinstance(entry, dict) or entry.get("kind") != "custom":
    return None
  proto = models.protocol(str(entry.get("protocol", "")))
  if proto is None or proto.kind != kind:
    _fallback(kind, "the committed protocol is not one this fork runs", params)
    return None
  selection = str(entry.get("id", ""))
  build = str(entry.get("build", ""))
  key = os.path.basename(build)
  state, error = models.build_state(key)
  if state != "built" or not build:
    _fallback(kind, error or models.NOT_BUILT, params)
    return None
  record = models.load_build(key) or {}
  if record.get("protocol") != proto.id:
    _fallback(kind, models.BUILD_MISMATCH, params)
    return None
  resolved = models.selection_record(selection)
  if resolved is None or resolved["protocol"].id != proto.id:
    _fallback(kind, models.NOT_INSTALLED, params)
    return None
  for role, member in resolved["members"].items():
    try:
      if os.path.getsize(member["path"]) != member["size"]:
        _fallback(kind, f"{role}: {models.ARTIFACT_CHANGED}", params)
        return None
    except OSError:
      _fallback(kind, f"{role}: {models.NOT_INSTALLED}", params)
      return None
  if kind == models.DRIVING:
    return DriverRuntime(entry, record, params)
  return MonitoringRuntime(entry, record, params)


def moonpilot_model_runtime() -> DriverRuntime | None:
  """The driving half of the boot snapshot, or `None` for the bundled model."""
  try:
    from openpilot.common.params import Params

    params = Params()
    return runtime_from_entry(models.DRIVING, models.boot(params)[models.DRIVING], params)
  except Exception:
    cloudlog.exception("moonpilot models: driving runtime unavailable")
    return None


def moonpilot_dm_runtime() -> MonitoringRuntime | None:
  """The monitoring half of the boot snapshot, or `None` for the bundled model."""
  try:
    from openpilot.common.params import Params

    params = Params()
    return runtime_from_entry(models.MONITORING, models.boot(params)[models.MONITORING], params)
  except Exception:
    cloudlog.exception("moonpilot models: monitoring runtime unavailable")
    return None
