#!/usr/bin/env python3
"""Compile a selection's members into the JIT the runtime loads.

`moonpilot/modelsd.py` calls `build()` once per selection per resolution set, offroad, and this is
the only module that runs tinygrad's compiler. Five things are not obvious.

**The tinygrad flags are the SConscript's, and they must be in the environment before tinygrad is
imported.** `openpilot/selfdrive/modeld/SConscript` prefixes its compile commands with them and the
device build pins the compiler to CPU 7 (`isolcpus` on AGNOS); this module does the same, and
`moonpilot/tests/test_modelruntime.py` fails if the two strings drift. The import order is why the
flags are set at module import rather than in a function: `DEV=` is read when a device is opened.

**The families are compiled by different code, all of it fork-owned.** `comma.supercombo.v1` follows
the deleted `openpilot/selfdrive/modeld/compile_modeld.py`, moved here verbatim, because the fork's
runtime has to reproduce exactly the buffers and the recurrence that model was trained against.
`comma.split-vision-policy.v1` and `comma.split-vision-off-on.v1` need the historical split glue,
which no longer exists upstream: the vision member runs once, its `hidden_state` slice feeds each
policy member's feature queue, and the JIT returns one flat tensor per role -- which is why the
runtime merges per-role slices instead of slicing one buffer. `comma.dmonitoring.v1` is a single
`TinyJit` over the model, pickled plainly.

**Every queue and view is recorded, not re-derived.** The metadata written into `model.pkl` carries
the exact shapes the JIT was captured against (`queue_shapes`, `npy_shapes`, `packed_npy_size`,
`input_keys`), and `moonpilot/modelruntime.py` rebuilds the buffers from that record. One thing is
deliberately *not* recorded: the packed buffer's size includes the camera's NV12 frame copy size,
which differs per resolution, so the runtime computes it from the camera it was handed -- the same
function the compiler uses (`models.nv12_copy_size`).

**A half-finished build is never valid.** `build.json` is written last, from a temp directory inside
the store, and `models.build_state` treats a build without it -- or without `model.pkl` -- as not
built. A compiler failure is a record with `state: "failed"` and the message in `error`, never a
silent fallback: the runtime falls back to the bundled model at load, and the panel says why.

**Nothing here reads a downloaded file with a full unpickler.** Interfaces come from
`moonpilot/vendor/openmodels/metadata.py`, which hand-decodes the ONNX protobuf and refuses anything
but `slice` in the embedded metadata pickle. The only `pickle.load` in this file is of the model the
*compiler itself* just wrote, or of an ONNX whose bytes the caller already checked.
"""

import argparse
import datetime
import io
import math
import os
import pickle
import platform
import shutil
import struct
import tempfile
import time
from functools import partial
from typing import Any, NamedTuple

import numpy as np

from moonpilot import models

# Exactly the strings `openpilot/selfdrive/modeld/SConscript` compiles with; the test pins them.
TG_FLAGS_QCOM = "DEV=QCOM IMAGE=1 FLOAT16=1 NOLOCALS=1 JIT_BATCH_SIZE=0 OPENPILOT_HACKS=1"
TG_FLAGS_CPU = "DEV=CPU:LLVM"
TG_FLAGS_METAL = "DEV=METAL JIT=2"  # JIT=2 disables graph batching, which mixes buffers up on Metal


def tg_flags() -> str:
  """The flags for the machine this is running on, as the SConscript picks them."""
  machine, system = platform.machine(), platform.system()
  if machine in ("aarch64", "arm64") and system == "Linux":
    return TG_FLAGS_QCOM
  if system == "Darwin":
    return TG_FLAGS_METAL
  return TG_FLAGS_CPU


TG_FLAGS = tg_flags()

# `DEV=` and the rest are read when tinygrad enumerates devices, so they have to be in the
# environment before it is imported. `setdefault`: an explicit DEV in the environment (the smoke
# path's `DEV=CPU:LLVM`) is the caller saying what it wants.
for _flag in TG_FLAGS.split():
  _name, _value = _flag.split("=", 1)
  os.environ.setdefault(_name, _value)
os.environ.setdefault("GMMU", "0")  # for chestnut fast loading, noop for qcom

from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import Context
from tinygrad.tensor import Tensor


class NV12Frame(NamedTuple):
  """Camera frame geometry: the resolution plus what `get_nv12_info` reports. Moved from deleted
  `openpilot/selfdrive/modeld/compile_modeld.py`; current upstream `compile_warp.py` defines the
  same shape but lives behind `examples.openpilot.helpers`, which is not importable from the
  project root without a path hack."""
  width: int
  height: int
  stride: int
  y_height: int
  uv_height: int
  size: int


MODELD_INPUTS = ["img_q", "big_img_q", "feat_q", "desire_q", "packed_npy_inputs"]


def read_file_chunked_to_disk(path: str) -> str:
  """The selected ONNX as a path `OnnxRunner` can open. Deleted `file_chunker` only existed for
  upstream's chunked checkouts; the fork's members are single files, so an unchunked path is used
  directly and never copied to a `.unchunked` staging file."""
  return path


def warp_perspective_tinygrad(src_flat, M_inv, dst_shape, src_shape, stride_pad, border_fill_val=None):
  w_dst, h_dst = dst_shape
  h_src, w_src = src_shape

  x = Tensor.arange(w_dst).reshape(1, w_dst).expand(h_dst, w_dst).reshape(-1)
  y = Tensor.arange(h_dst).reshape(h_dst, 1).expand(h_dst, w_dst).reshape(-1)

  # inline 3x3 matmul as elementwise to avoid reduce op (enables fusion with gather)
  src_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
  src_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
  src_w = M_inv[2, 0] * x + M_inv[2, 1] * y + M_inv[2, 2]

  src_x = src_x / src_w
  src_y = src_y / src_w

  x_round = Tensor.round(src_x)
  y_round = Tensor.round(src_y)
  x_nn_clipped = x_round.clip(0, w_src - 1).cast("int")
  y_nn_clipped = y_round.clip(0, h_src - 1).cast("int")
  idx = y_nn_clipped * (w_src + stride_pad) + x_nn_clipped
  sampled = src_flat[idx]

  if border_fill_val is None:
    return sampled

  in_bounds = ((x_round >= 0) & (x_round <= w_src - 1) &
               (y_round >= 0) & (y_round <= h_src - 1)).cast(sampled.dtype)
  return sampled * in_bounds + Tensor(border_fill_val, dtype=sampled.dtype) * (1 - in_bounds)


def frames_to_tensor(frames):
  H = (frames.shape[0] * 2) // 3
  W = frames.shape[1]
  in_img1 = Tensor.cat(frames[0:H:2, 0::2],
                       frames[1:H:2, 0::2],
                       frames[0:H:2, 1::2],
                       frames[1:H:2, 1::2],
                       frames[H:H+H//4].reshape((H//2, W//2)),
                       frames[H+H//4:H+H//2].reshape((H//2, W//2)), dim=0).reshape((6, H//2, W//2))
  return in_img1


def make_frame_prepare(nv12: NV12Frame, model_w, model_h):
  cam_w, cam_h, stride, y_height, uv_height, _ = nv12
  uv_offset = stride * y_height
  stride_pad = stride - cam_w

  def frame_prepare_tinygrad(input_frame, M_inv):
    M_inv = M_inv.to(Device.DEFAULT).realize()
    # UV_SCALE @ M_inv @ UV_SCALE_INV simplifies to elementwise scaling
    M_inv_uv = M_inv * Tensor([[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], device=Device.DEFAULT)
    # deinterleave NV12 UV plane (UVUV... -> separate U, V)
    uv = input_frame[uv_offset:uv_offset + uv_height * stride].reshape(uv_height, stride)
    with Context(SPLIT_REDUCEOP=0):
      y = warp_perspective_tinygrad(input_frame[:cam_h*stride],
                                    M_inv, (model_w, model_h),
                                    (cam_h, cam_w), stride_pad).realize()
      u = warp_perspective_tinygrad(uv[:cam_h//2, :cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0).realize()
      v = warp_perspective_tinygrad(uv[:cam_h//2, 1:cam_w:2].flatten(),
                                    M_inv_uv, (model_w//2, model_h//2),
                                    (cam_h//2, cam_w//2), 0).realize()
    yuv = y.cat(u).cat(v).reshape((model_h * 3 // 2, model_w))
    tensor = frames_to_tensor(yuv)
    return tensor
  return frame_prepare_tinygrad


def get_policy_npy_shapes(input_shapes):
  dp = input_shapes["desire_pulse"]  # (1, 25, 8)
  tc = input_shapes["traffic_convention"]  # (1, 2)
  at = input_shapes["action_t"]  # (1, 2)
  fb = input_shapes["features_buffer"]  # (1, T-1, ...) e.g. (1, 24, 32, 512) with spatial features
  feat_dim = math.prod(fb[2:])
  # TODO prev_feat shouldn't exist and be handled inside the JIT, but corrupt on QCOM for now
  shapes = {"desire": (dp[2],), "traffic_convention": tuple(tc), "action_t": tuple(at), "prev_feat": (fb[0], feat_dim)}
  return shapes, [math.prod(s) for s in shapes.values()]


def make_input_queues(input_shapes, frame_skip, device, frame_copy_size):
  img = input_shapes["img"]  # (1, 12, 128, 256)
  fb = input_shapes["features_buffer"]  # (1, T-1, ...), past features only; the model appends the current frame's feature
  feat_dim = math.prod(fb[2:])
  dp = input_shapes["desire_pulse"]  # (1, 25, 8)
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])

  policy_shapes, _ = get_policy_npy_shapes(input_shapes)
  shapes = {"tfm": (3, 3), "big_tfm": (3, 3)} | policy_shapes
  sizes = [math.prod(s) for s in shapes.values()]
  packed_npy_size = sum(sizes) * np.dtype(np.float32).itemsize
  packed_input = np.zeros(packed_npy_size + 2 * frame_copy_size, dtype=np.uint8)
  packed_npy_inputs = packed_input[:packed_npy_size].view(np.float32)
  frames = packed_input[packed_npy_size:]
  frame_views = {"img": frames[:frame_copy_size], "big_img": frames[frame_copy_size:]}
  # views into the packed inputs, to be refilled at runtime
  npy = {k: v.reshape(s) for (k, s), v in zip(shapes.items(), np.split(packed_npy_inputs, np.cumsum(sizes[:-1])), strict=True)}
  input_queues = {
    "img_q": Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    "big_img_q": Tensor(np.zeros(img_buf_shape, dtype=np.uint8), device=device).contiguous().realize(),
    "feat_q": Tensor(np.zeros((frame_skip * fb[1], fb[0], feat_dim), dtype=np.float32), device=device).contiguous().realize(),
    "desire_q": Tensor(np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float32), device=device).contiguous().realize(),
    "packed_npy_inputs": Tensor(packed_input, device="NPY").realize(),
  }
  return input_queues, npy, frame_views


def shift_and_sample(buf, new_val, sample_fn):
  buf.assign(buf[1:].cat(new_val, dim=0).contiguous())
  return sample_fn(buf)


def sample_skip(buf, frame_skip):
  return buf[::frame_skip].contiguous().flatten(0, 1).unsqueeze(0)


def sample_desire(buf, frame_skip):
  return buf.reshape(-1, frame_skip, *buf.shape[1:]).max(1).flatten(0, 1).unsqueeze(0)


def make_warp(nv12, model_w, model_h):
  frame_prepare = make_frame_prepare(nv12, model_w, model_h)

  def warp(tfm, big_tfm, frame, big_frame):
    tfm = tfm.to(Device.DEFAULT)
    big_tfm = big_tfm.to(Device.DEFAULT)
    frame = frame.to(Device.DEFAULT)
    big_frame = big_frame.to(Device.DEFAULT)
    Tensor.realize(tfm, big_tfm, frame, big_frame)

    warped_frame = frame_prepare(frame, tfm).unsqueeze(0)
    warped_big_frame = frame_prepare(big_frame, big_tfm).unsqueeze(0)
    return Tensor.cat(warped_frame, warped_big_frame)

  return warp


def make_run_policy(model_runner, model_metadata, frame_skip):
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  npy_shapes, npy_sizes = get_policy_npy_shapes(model_metadata["input_shapes"])
  model_input_dtypes = {name: spec.dtype for name, spec in model_runner.graph_inputs.items()}

  def run_policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_npy_inputs = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_npy_inputs, warped)

    img = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
    big_img = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)

    desire, traffic_convention, action_t, prev_feat = (t.reshape(s) for t, s in zip(packed_npy_inputs.split(npy_sizes), npy_shapes.values(), strict=True))
    desire_buf = shift_and_sample(desire_q, desire.reshape(1, 1, -1), sample_desire_fn)
    feat_buf = shift_and_sample(feat_q, prev_feat.reshape(1, 1, -1), sample_skip_fn)

    inputs = {
      "img": img,
      "big_img": big_img,
      "features_buffer": feat_buf.reshape(model_metadata["input_shapes"]["features_buffer"]),
      "desire_pulse": desire_buf,
      "traffic_convention": traffic_convention,
      "action_t": action_t,
    }
    inputs = {name: value.cast(model_input_dtypes[name]) for name, value in inputs.items()}
    out = next(iter(model_runner(inputs).values())).cast("float32")
    return out,
  return run_policy


def make_run_model(warp, run_policy, model_metadata, frame_copy_size):
  _, policy_sizes = get_policy_npy_shapes(model_metadata["input_shapes"])
  packed_npy_size = (18 + sum(policy_sizes)) * np.dtype(np.float32).itemsize

  def run_model(img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
    packed_input = packed_npy_inputs.to(Device.DEFAULT)
    Tensor.realize(packed_input)
    packed_npy_inputs = packed_input[:packed_npy_size].bitcast("float32")
    frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
    big_frame = packed_input[packed_npy_size + frame_copy_size:]
    tfm, big_tfm, policy_inputs = packed_npy_inputs.split([9, 9, sum(policy_sizes)])
    warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)
    return run_policy(warped, img_q, big_img_q, feat_q, desire_q, policy_inputs)
  return run_model


def compile_jit(jit, input_keys, make_queues, benchmark_runs):
  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")

  SEED = 42
  def random_inputs_run(fn, seed, n_runs, test_val=None, test_buffers=None, expect_match=True):
    input_queues, npy, frame_views = make_queues(Device.DEFAULT)
    rng = np.random.default_rng(seed)

    for i in range(n_runs):
      for v in npy.values():
        v[:] = rng.standard_normal(v.shape).astype(v.dtype)
      for v in frame_views.values():
        v[:] = rng.integers(0, 256, size=v.shape, dtype=np.uint8)
      Device.default.synchronize()
      st = time.perf_counter()
      outs = fn(**{k: input_queues[k] for k in input_keys})
      mt = time.perf_counter()
      Device.default.synchronize()
      et = time.perf_counter()
      print(f"  [{i+1}/{n_runs}] enqueue {(mt-st)*1e3:6.2f} ms -- total {(et-st)*1e3:6.2f} ms")

      if i == 0:
        val = [np.copy(v.numpy()) for v in outs]
        buffers = [np.copy(v.numpy().copy()) for v in input_queues.values()]

    if test_val is not None:
      match = all(np.array_equal(a, b) for a, b in zip(val, test_val, strict=True))
      assert match == expect_match, f"outputs {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    if test_buffers is not None:
      match = all(np.array_equal(a, b) for a, b in zip(buffers, test_buffers, strict=True))
      assert match == expect_match, f"buffers {'differ from' if expect_match else 'match'} baseline (seed={seed})"
    return val, buffers

  print("capture + replay")
  test_val, test_buffers = random_inputs_run(jit, SEED, 3)
  print(f"pickle round trip ({benchmark_runs} runs per seed)")
  with tempfile.TemporaryFile(dir=_scratch_dir()) as f:
    dump_oob(jit, f)
    f.seek(0)
    from moonpilot.modelruntime import _load_fork_oob
    loaded_jit = _load_fork_oob(f)
  random_inputs_run(loaded_jit, SEED, benchmark_runs, test_val, test_buffers, expect_match=True)
  random_inputs_run(loaded_jit, SEED+1, benchmark_runs, test_val, test_buffers, expect_match=False)
  # Keep the original so per-resolution JITs share model weight buffers in the final pickle.
  return jit


from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

from moonpilot.vendor.openmodels import metadata as onnx_metadata

BENCHMARK_RUNS = 1


def prepare_environment() -> None:
  """Idempotent: the flags are already set at import, and this is for a caller that wants to be
  explicit about it before touching anything."""
  for flag in TG_FLAGS.split():
    name, value = flag.split("=", 1)
    os.environ.setdefault(name, value)


def pin_compiler() -> None:
  """The SConscript pins the device's compiler to CPU 7, which AGNOS isolates; the worker's own
  compile should not compete with realtime processes either. Best effort on purpose: a container
  whose affinity mask excludes CPU 7 keeps whatever scheduling it had."""
  if platform.machine() not in ("aarch64", "arm64"):
    return
  try:
    os.sched_setaffinity(0, {7})
  except (AttributeError, OSError):
    pass


def dump_oob(obj: Any, handle) -> None:
  """The fork's OOB writer, matching `moonpilot/modelruntime.py::_load_fork_oob`. The scratch file
  lives in the store's temp directory, not the checkout: a checkout temp means writing the model's
  buffers into git state (AGENTS.md, Storage)."""
  with tempfile.TemporaryFile(dir=_scratch_dir()) as tmp:

    def buffer_callback(pickle_buffer: pickle.PickleBuffer):
      view = pickle_buffer.raw()
      tmp.write(struct.pack("<q", view.nbytes))
      tmp.write(view)
      pickle_buffer.release()  # keep peak ram at ~1 buffer

    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    handle.write(struct.pack("<q", len(opcodes)))
    handle.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, handle)


def _scratch_dir() -> str:
  scratch = models.tmp_dir()
  os.makedirs(scratch, exist_ok=True)
  return scratch


def failure_record(error: str) -> dict:
  """The `build.json` body for a failed build, so the worker and this module cannot disagree about
  its shape. `models.build_state` reads `state` and `error` and nothing else."""
  return {"schema": models.SCHEMA, "state": "failed", "error": error}


def _interface(path: str) -> dict:
  """What one member's bytes declare: `input_shapes` and `slices`, read without executing anything."""
  record = onnx_metadata.parse(path)
  if record.get("output_slices_error") is not None or record.get("output_slices") is None:
    raise ValueError(f"{os.path.basename(path)}: output_slices unreadable")
  return {"input_shapes": record["input_shapes"], "slices": record["output_slices"]}


def _model_size(input_shapes: dict) -> tuple[int, int]:
  """The warp's target size, from the model's own `img` input: `[1, 12, H//2, W//2]` is a YUV420
  frame whose halves are the UV planes, so the full frame is twice each dimension."""
  img = input_shapes["img"]
  return int(img[3]) * 2, int(img[2]) * 2


def _npy_layout(input_shapes: dict) -> tuple[list[list[Any]], list[int]]:
  """The packed float views `make_input_queues` builds, in its order: the transforms, then
  the policy scalars and the recurrent feature. `sizes` is the same list in elements, so a split
  family can lay its per-role fields out with the same arithmetic."""
  desire_pulse = input_shapes["desire_pulse"]
  traffic = input_shapes["traffic_convention"]
  features = input_shapes["features_buffer"]
  shapes: list[list[Any]] = [["tfm", [3, 3]], ["big_tfm", [3, 3]]]
  for name, shape in (("desire", (desire_pulse[2],)), ("traffic_convention", tuple(traffic))):
    shapes.append([name, list(shape)])
  if "action_t" in input_shapes:
    shapes.append(["action_t", list(input_shapes["action_t"])])
  shapes.append(["prev_feat", [features[0], int(_product(features[2:]))]])
  sizes = [int(_product(shape)) for _name, shape in shapes]
  return shapes, sizes


def _product(shape) -> int:
  total = 1
  for value in shape:
    total *= int(value)
  return total


def _sfx(role: str, name: str) -> str:
  """A packed field's name for one role. Single-role families keep upstream's own names, so the
  supercombo metadata is the metadata `make_input_queues` builds; a split namespaces them, because
  two policy members have two `desire` views. `moonpilot/modelruntime.py` resolves either."""
  return name if role == models.SUPERCOMBO.roles[0] else f"{name}_{role}"


def effective_contract(proto, interfaces: dict[str, dict]) -> str:
  """The decode this *build* needs. A protocol's action contract describes the head its family
  carries, and upstream's own decode branches on the head's presence: a selection with no `action`
  slice is decoded from its plan, which is what the older family-C recipes need. Admission has
  already refused an action head on the wrong side of the change, so what is decided here is
  presence and nothing else."""
  return proto.action_contract if any("action" in interface["slices"] for interface in interfaces.values()) else "plan"


def _family_metadata(
  proto, interfaces: dict[str, dict], configuration: dict, npy_shapes: list, packed_npy_size: int, queue_shapes: list, input_keys: list[str]
) -> dict:
  return {
    "family": proto.id,
    "action_contract": effective_contract(proto, interfaces),
    "frame_skip": int(configuration["frame_skip"]) if "frame_skip" in configuration else 0,
    "features_role": models.features_role(proto),
    "role_order": list(proto.roles),
    "roles": {
      role: {"input_shapes": interfaces[role]["input_shapes"], "slices": interfaces[role]["slices"], "index": proto.roles.index(role)} for role in proto.roles
    },
    "vision_input_names": [name for name in interfaces[proto.roles[0]]["input_shapes"] if "img" in name],
    "npy_shapes": npy_shapes,
    "packed_npy_size": packed_npy_size,
    "queue_shapes": queue_shapes,
    "input_keys": input_keys,
  }


def _compile_supercombo(proto, members: dict[str, str], interfaces: dict[str, dict], configuration: dict, camera_resolutions, payload: dict) -> None:
  """The single-role path, with the selected ONNX instead of the bundled one. The JIT it captures
  already returns a one-tuple, which is the tuple-of-roles contract every family follows."""
  from tinygrad.nn.onnx import OnnxRunner

  role = proto.roles[0]
  input_shapes = interfaces[role]["input_shapes"]
  frame_skip = int(configuration["frame_skip"])
  model_w, model_h = _model_size(input_shapes)
  runner = OnnxRunner(read_file_chunked_to_disk(members[role]))
  run_policy = make_run_policy(runner, {"input_shapes": input_shapes}, frame_skip)

  shapes, sizes = _npy_layout(input_shapes)
  npy_shapes = shapes
  packed_npy_size = sum(sizes) * 4
  vision_input_names = [name for name in input_shapes if "img" in name]
  queues: list[list[Any]] = []
  for cam_w, cam_h in camera_resolutions:
    nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
    frame_copy_size = models.nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
    make_queues = partial(make_input_queues, input_shapes, frame_skip, frame_copy_size=frame_copy_size)
    warp = make_warp(nv12, model_w, model_h)
    jit = TinyJit(make_run_model(warp, run_policy, {"input_shapes": input_shapes}, frame_copy_size), prune=True)
    payload["run_model"][(cam_w, cam_h)] = compile_jit(jit, MODELD_INPUTS, make_queues, BENCHMARK_RUNS)
    if not queues:
      queues = _queue_shapes_for_record(input_shapes, frame_skip)

  payload["metadata"] = _family_metadata(proto, interfaces, configuration, npy_shapes, packed_npy_size, queues, list(MODELD_INPUTS))
  # The runtime rebuilds the queues from the record, so they have to be the two the capture built.
  assert set(vision_input_names) == {"img", "big_img"}, vision_input_names


def _queue_shapes_for_record(input_shapes: dict, frame_skip: int) -> list[list[Any]]:
  """The queues `make_input_queues` creates, minus the packed buffer: its size depends on
  the camera's frame copy size, so the runtime computes it from the camera it is handed."""
  img = input_shapes["img"]
  features = input_shapes["features_buffer"]
  desire_pulse = input_shapes["desire_pulse"]
  n_frames = img[1] // 6
  img_buf_shape = [frame_skip * (n_frames - 1) + 1, 6, img[2], img[3]]
  return [
    ["img_q", img_buf_shape, "uint8"],
    ["big_img_q", list(img_buf_shape), "uint8"],
    ["feat_q", [frame_skip * features[1], features[0], int(_product(features[2:]))], "float32"],
    ["desire_q", [frame_skip * desire_pulse[1], desire_pulse[0], desire_pulse[2]], "float32"],
  ]


def _compile_split(proto, members: dict[str, str], interfaces: dict[str, dict], configuration: dict,
                   camera_resolutions, payload: dict) -> None:
  """The historical split glue, on the moved compiler helpers.

  The vision member runs once per frame; its `hidden_state` slice is the feature the policy members'
  queues carry -- one queue per policy role, because two policy members may declare different
  `features_buffer` shapes and the runtime cannot assume they agree. The packed float region holds
  the transforms and then, per policy role in `role_order`, that role's scalar fields and
  `prev_feat`; the JIT returns `(vision_out, *policy_outs)` in the same order, which is the
  tuple-of-roles contract the runtime merges.

  The queues are named by the policy role's *position* (`feat_q_0`, `desire_q_0`, ...), because the
  JIT is called by keyword and its parameter names have to be fixed while the role names in a
  catalog are not. `role_order` is what maps a position back to a role.
  """
  from tinygrad.nn.onnx import OnnxRunner

  vision_role = models.features_role(proto)
  policy_roles = [role for role in proto.roles if role != vision_role]
  frame_skip = int(configuration["frame_skip"])
  vision_shapes = interfaces[vision_role]["input_shapes"]
  vision_hidden = interfaces[vision_role]["slices"]["hidden_state"]
  model_w, model_h = _model_size(vision_shapes)

  vision_runner = OnnxRunner(read_file_chunked_to_disk(members[vision_role]))
  policy_runners = {role: OnnxRunner(read_file_chunked_to_disk(members[role])) for role in policy_roles}
  input_dtypes = {role: {name: spec.dtype for name, spec in policy_runners[role].graph_inputs.items()} for role in policy_roles}

  # The packed layout, in call order: the transforms, then each policy role's fields.
  npy_shapes: list[list[Any]] = [["tfm", [3, 3]], ["big_tfm", [3, 3]]]
  role_fields: dict[str, list[list[Any]]] = {}
  for role in policy_roles:
    shapes, _sizes = _npy_layout(interfaces[role]["input_shapes"])
    role_fields[role] = [[_sfx(role, name), shape] for name, shape in shapes]
    npy_shapes.extend(role_fields[role])
  sizes = [int(_product(shape)) for _name, shape in npy_shapes]
  packed_npy_size = sum(sizes) * 4
  role_sizes = {role: [int(_product(shape)) for _name, shape in role_fields[role]] for role in policy_roles}

  queue_shapes: list[list[Any]] = []
  queue_keys: list[str] = []
  img = vision_shapes["img"]
  n_frames = img[1] // 6
  for name in ("img_q", "big_img_q"):
    queue_shapes.append([name, [frame_skip * (n_frames - 1) + 1, 6, img[2], img[3]], "uint8"])
    queue_keys.append(name)
  for index, role in enumerate(policy_roles):
    features = interfaces[role]["input_shapes"]["features_buffer"]
    desire_pulse = interfaces[role]["input_shapes"]["desire_pulse"]
    queue_shapes.append([f"feat_q_{index}", [frame_skip * features[1], features[0], int(_product(features[2:]))], "float32"])
    queue_shapes.append([f"desire_q_{index}", [frame_skip * desire_pulse[1], desire_pulse[0], desire_pulse[2]], "float32"])
    queue_keys.extend([f"feat_q_{index}", f"desire_q_{index}"])

  sample_skip_fn = partial(sample_skip, frame_skip=frame_skip)
  sample_desire_fn = partial(sample_desire, frame_skip=frame_skip)
  hidden_slice = slice(*vision_hidden)

  def make_split_run_policy(warp, frame_copy_size):
    """One `run_policy` per camera resolution: the warp and the frame copy size are the resolution's,
    and the JIT captured against them is what gets pickled for that resolution."""

    def run_policy(img_q, big_img_q, packed_npy_inputs, *queues):
      packed_input = packed_npy_inputs.to(Device.DEFAULT)
      Tensor.realize(packed_input)
      floats = packed_input[:packed_npy_size].bitcast("float32")
      frame = packed_input[packed_npy_size:packed_npy_size + frame_copy_size]
      big_frame = packed_input[packed_npy_size + frame_copy_size:]
      tfm, big_tfm, rest = floats.split([9, 9, sum(sizes) - 18])
      warped = warp(tfm.reshape(3, 3), big_tfm.reshape(3, 3), frame, big_frame)

      img_tensor = shift_and_sample(img_q, warped[0:1], sample_skip_fn)
      big_img_tensor = shift_and_sample(big_img_q, warped[1:2], sample_skip_fn)
      vision_out = next(iter(vision_runner({"img": img_tensor, "big_img": big_img_tensor}).values())).cast("float32")
      new_feature = vision_out[:, hidden_slice].reshape(1, -1).unsqueeze(0)

      outs = [vision_out]
      offset = 0
      for index, role in enumerate(policy_roles):
        fields = role_fields[role]
        chunk = rest[offset:offset + sum(role_sizes[role])].split(role_sizes[role])
        offset += sum(role_sizes[role])
        values = {name: tensor.reshape(shape) for (name, shape), tensor in zip(fields, chunk, strict=True)}
        feat_q, desire_q = queues[2 * index:2 * index + 2]
        inputs = {
          "features_buffer": shift_and_sample(feat_q, new_feature, sample_skip_fn).reshape(
            interfaces[role]["input_shapes"]["features_buffer"]),
          "desire_pulse": shift_and_sample(desire_q, values[_sfx(role, "desire")].reshape(1, 1, -1), sample_desire_fn),
          "traffic_convention": values[_sfx(role, "traffic_convention")],
        }
        if _sfx(role, "action_t") in values:
          inputs["action_t"] = values[_sfx(role, "action_t")]
        declared = interfaces[role]["input_shapes"]
        if set(inputs) - set(declared):
          raise ValueError(f"{role}: packing inputs the model does not declare: {sorted(set(inputs) - set(declared))}")
        inputs = {name: value.cast(input_dtypes[role][name]) for name, value in inputs.items()}
        outs.append(next(iter(policy_runners[role](inputs).values())).cast("float32"))
      return tuple(outs)

    return run_policy

  for cam_w, cam_h in camera_resolutions:
    nv12 = NV12Frame(cam_w, cam_h, *get_nv12_info(cam_w, cam_h))
    frame_copy_size = models.nv12_copy_size(nv12.stride, nv12.y_height, nv12.uv_height)
    warp = make_warp(nv12, model_w, model_h)
    run_policy = make_split_run_policy(warp, frame_copy_size)
    make_queues = partial(_make_split_queues, queue_shapes, packed_npy_size, frame_copy_size)
    jit = TinyJit(run_policy, prune=True)
    payload["run_model"][(cam_w, cam_h)] = compile_jit(jit, queue_keys + ["packed_npy_inputs"], make_queues, BENCHMARK_RUNS)

  payload["metadata"] = _family_metadata(proto, interfaces, configuration, npy_shapes, packed_npy_size, queue_shapes,
                                         queue_keys + ["packed_npy_inputs"])


def _make_split_queues(queue_shapes, packed_npy_size, frame_copy_size, device):
  """This module's `compile_jit` queue factory: the same buffers the runtime rebuilds, per camera resolution."""
  queues = {}
  for key, shape, dtype in queue_shapes:
    array = np.zeros(tuple(shape), dtype=np.uint8 if dtype == "uint8" else np.float32)
    queues[key] = Tensor(array, device=device).contiguous().realize()
  packed_input = np.zeros(packed_npy_size + 2 * frame_copy_size, dtype=np.uint8)
  queues["packed_npy_inputs"] = Tensor(packed_input, device="NPY").realize()
  return queues


def _compile_dmonitoring(proto, members: dict[str, str], interfaces: dict[str, dict], configuration: dict, camera_resolutions, payload: dict) -> None:
  """One JIT over the model, warm, pickled plainly -- the shape `dmonitoringmodeld` loads today.

  The warp is not compiled here: it maps the camera's NV12 frame to the model's input size and does
  not depend on the model, so the runtime puts the bundled `dm_warp_<w>x<h>_tinygrad.pkl` in front of
  this. What is checked here is that the model's `input_img` is the size that warp produces.
  """
  from tinygrad.nn.onnx import OnnxRunner

  role = proto.roles[0]
  input_shapes = interfaces[role]["input_shapes"]
  input_img = input_shapes.get("input_img")
  if input_img is None or int(_product(input_img)) != int(_product(models.DM_INPUT_SIZE)):
    raise ValueError(f"input_img {input_img} is not the {models.DM_INPUT_SIZE} the DM warp produces")

  runner = OnnxRunner(read_file_chunked_to_disk(members[role]))

  # The parameter names and their order are the runtime's: `dmonitoringmodeld` calls the loaded JIT
  # with `**self.tensor_inputs`, which is `{'calib': ..., 'input_img': ...}`. A JIT is captured
  # against the names it is called with, so the warmup call below has to use the same keywords.
  def run_onnx(calib, input_img):
    return next(iter(runner({"input_img": input_img.to(Device.DEFAULT), "calib": calib.to(Device.DEFAULT)}).values())).cast("float32")

  jit = TinyJit(run_onnx, prune=True)
  dummy_img = Tensor(np.zeros(tuple(input_img), dtype=np.uint8), device=Device.DEFAULT).realize()
  dummy_calib = Tensor(np.zeros(tuple(input_shapes["calib"]), dtype=np.float32), device="NPY").realize()
  for _ in range(3):
    jit(calib=dummy_calib, input_img=dummy_img).numpy()

  payload["model_run"] = jit
  payload["metadata"] = {"input_shapes": input_shapes, "slices": interfaces[role]["slices"], "output_slices": interfaces[role]["slices"]}
  payload["input_devices"] = {"model": Device.DEFAULT}
  payload.pop("run_model", None)


def build(
  proto,
  members: dict[str, str],
  configuration: dict,
  camera_resolutions=models.CAMERA_RESOLUTIONS,
  out_dir: str | None = None,
  *,
  selection: str = "",
) -> str:
  """Compile one selection and write its build directory. Returns the directory.

  Raises on anything that would make a wrong build: a member whose bytes do not declare the
  interface the protocol requires, a family this module does not compile, a DM model the warp cannot
  feed. The caller (`moonpilot/modelsd.py`) records the failure and leaves `model.pkl` out, so
  nothing downstream can load a build that did not finish.
  """
  interfaces = {role: _interface(path) for role, path in members.items()}
  views = {
    role: {"input_shapes": interfaces[role]["input_shapes"], "slices": interfaces[role]["slices"], "format": "onnx", "targets": ["QCOM"]} for role in interfaces
  }
  # No `published_at`: the action head's era is admission's decision, made against the catalog's
  # recorded publication dates, and a build only ever runs a selection that already passed it.
  if (reason := models.set_reason(proto, views)) is not None:
    raise ValueError(reason)

  directory = out_dir or models.build_dir(selection, proto.id, camera_resolutions)
  os.makedirs(directory, exist_ok=True)
  payload: dict[str, Any] = {"metadata": {}, "input_devices": {"model": Device.DEFAULT}, "run_model": {}}

  if proto.id == models.SUPERCOMBO.id:
    _compile_supercombo(proto, members, interfaces, configuration, camera_resolutions, payload)
  elif proto.id in (models.SPLIT_VISION_POLICY.id, models.SPLIT_VISION_OFF_ON.id):
    _compile_split(proto, members, interfaces, configuration, camera_resolutions, payload)
  elif proto.id == models.DMONITORING.id:
    _compile_dmonitoring(proto, members, interfaces, configuration, camera_resolutions, payload)
  else:
    raise ValueError(f"no compiler for {proto.id}")

  model_path = os.path.join(directory, "model.pkl")
  # Staged inside the build directory, not in the store's scratch space: `os.replace` is only atomic
  # within a filesystem, and a build directory can be anywhere (the smoke path's /tmp, a store root
  # that moved). A leftover staging file is not a build -- only `model.pkl` and `build.json` are read.
  staging = tempfile.NamedTemporaryFile(dir=directory, prefix=".model-", delete=False)
  staging.close()
  try:
    with open(staging.name, "wb") as handle:
      if proto.kind == models.DRIVING:
        dump_oob(payload, handle)
      else:
        pickle.dump(payload, handle)
    os.replace(staging.name, model_path)
  finally:
    if os.path.exists(staging.name):
      os.remove(staging.name)

  key = os.path.basename(directory)
  models.write_json(
    models.build_record(key),
    {
      "schema": models.SCHEMA,
      "state": "built",
      "error": None,
      "selection": selection,
      "protocol": proto.id,
      "roles": list(proto.roles),
      "members": {role: _member_record(members[role]) for role in members},
      "configuration": {key: configuration[key] for key in proto.required_configuration if key in configuration},
      # Wall clock, from a clock the repo's lint rule trusts (`time.time` is banned tree-wide).
      "created_at": datetime.datetime.now(datetime.UTC).timestamp(),
      "provenance": {"version": _fork_version(), "tinygrad": _tinygrad_revision(), "flags": TG_FLAGS},
      "camera_resolutions": [[w, h] for w, h in camera_resolutions],
      "action_contract": effective_contract(proto, interfaces),
    },
  )
  return directory


def _member_record(path: str) -> dict:
  """One member as the record keeps it: the digest and the size of the bytes that were compiled."""
  import hashlib

  with open(path, "rb") as handle:
    return {"sha256": hashlib.file_digest(handle, "sha256").hexdigest(), "size": os.path.getsize(path)}


def _fork_version() -> str:
  from moonpilot import features

  return features.version()


def _tinygrad_revision() -> str:
  """The tinygrad the build was compiled with, if this checkout can tell. Informational: nothing
  invalidates a build on a version bump (moonpilot/modelsd.py)."""
  import subprocess

  root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tinygrad_repo")
  try:
    result = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False)
    return result.stdout.strip() or "unknown"
  except (OSError, subprocess.SubprocessError):
    return "unknown"


def smoke(onnx_path: str, protocol_id: str | None, camera: tuple[int, int], build_dir: str | None) -> int:
  """Build one ONNX, load it back through the runtime and run one frame of zeros.

  The runnable proof that compile -> pickle -> load -> inference works, offline, on the machine this
  runs on. `smoke` never touches the store's selection params: it goes straight at the runtime
  classes, so it needs neither a boot snapshot nor a package on disk.
  """
  from moonpilot import modelruntime

  interface = _interface(onnx_path)
  if protocol_id is None:
    inputs = interface["input_shapes"]
    if "input_img" in inputs:
      protocol_id = models.DMONITORING.id
    elif "features_buffer" in inputs and "action_t" in inputs:
      protocol_id = models.SUPERCOMBO.id
    else:
      print(f"cannot tell which family {onnx_path} is; pass --protocol")
      return 2
  proto = models.protocol(protocol_id)
  if proto is None:
    print(f"unknown protocol {protocol_id}")
    return 2

  role = proto.roles[0]
  members = {role: onnx_path}
  configuration = {"frame_skip": 4, "LAT_SMOOTH_SECONDS": 0.0, "LONG_SMOOTH_SECONDS": 0.3}
  directory = build_dir or tempfile.mkdtemp(prefix="mp-smoke-")
  print(f"building {proto.id} into {directory} with {TG_FLAGS}")
  started = time.monotonic()
  build(proto, members, configuration, models.CAMERA_RESOLUTIONS, directory, selection="smoke")
  print(f"built in {time.monotonic() - started:.0f}s, {os.path.getsize(os.path.join(directory, 'model.pkl')) / 1e6:.1f} MB")
  print("record:", models.load_build(os.path.basename(directory)))

  entry = {"kind": "custom", "id": "smoke", "protocol": proto.id, "build": directory, "members": members, "configuration": configuration}
  import numpy as np

  if proto.kind == models.DRIVING:
    runtime = modelruntime.DriverRuntime(entry, {"action_contract": proto.action_contract}, None)
    model = runtime.create_model(camera[0], camera[1])
    if model is None:
      print("model failed to load")
      return 1
    print("vision inputs:", model.vision_input_names, "roles:", model.role_order)

    class Buf:
      def __init__(self, size):
        self.data = bytearray(size)

    bufs = {name: Buf(model.frame_copy_size) for name in model.vision_input_names}
    transforms = {name: np.eye(3, dtype=np.float32) for name in model.vision_input_names}
    inputs = {
      "desire_pulse": np.zeros(8, dtype=np.float32),
      "traffic_convention": np.array([0.0, 1.0], dtype=np.float32),
      "action_t": np.array([0.25, 0.25], dtype=np.float32),
    }
    started = time.monotonic()
    out = model.run(bufs, transforms, inputs)
    print(f"one frame in {(time.monotonic() - started) * 1e3:.0f}ms")
    print("outputs:", sorted(out))
    for key in sorted(out):
      value = out[key]
      print(f"  {key:24s} {value.shape} finite={bool(np.all(np.isfinite(value)))}")
    required = sorted(models.DRIVING_SLICES)
    missing = [name for name in required if name not in out]
    if missing:
      print("missing required outputs:", missing)
      return 1
  else:
    runtime = modelruntime.MonitoringRuntime(entry, {}, None)
    model = runtime.create_model(camera[0], camera[1])
    if model is None:
      print("model failed to load")
      return 1
    flat, gpu_time = model.run(_DmBuf(model.frame_buf_params[3]), np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32))
    print(f"one frame in {gpu_time * 1e3:.0f}ms, {flat.shape[0]} outputs")
    sliced = {name: flat[np.newaxis, bounds] for name, bounds in model.output_slices.items()}
    parsed = runtime.parse(sliced, model)
    print("parsed heads:", sorted(parsed))
    missing = [name for name in sorted(models.DM_SLICES) if name not in parsed]
    if missing:
      print("missing required heads:", missing)
      return 1
  return 0


class _DmBuf:
  """A VisionBuf-shaped object for the smoke path: the DM runtime only ever reads `buf.data`."""

  def __init__(self, size: int):
    self.data = bytearray(size)


def main() -> int:
  parser = argparse.ArgumentParser(description="compile one selection into a build directory")
  sub = parser.add_subparsers(dest="command", required=True)
  smoke_parser = sub.add_parser("smoke", help="build one ONNX, load it back and run one frame")
  smoke_parser.add_argument("--onnx", required=True)
  smoke_parser.add_argument("--protocol", default=None)
  smoke_parser.add_argument("--build-dir", default=None)
  smoke_parser.add_argument("--camera", default="1928x1208")
  args = parser.parse_args()

  if args.command == "smoke":
    width, height = (int(part) for part in args.camera.lower().split("x"))
    return smoke(args.onnx, args.protocol, (width, height), args.build_dir)
  return 2


if __name__ == "__main__":
  raise SystemExit(main())
