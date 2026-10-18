"""Lateral replays: controlsd's request chain and the fork torque controller over recorded control frames.

  python -m moonpilot.lab torque   # MoonpilotSteerHysteresis: torque reversals per second (AGENTS.md, The lateral controller)
  python -m moonpilot.lab smooth   # MoonpilotPathSmooth: request lag behind the model's action (docs/curvature.md)

`extract` is the shared lateral extract; other lab modules may read it.
"""
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

from opendbc.car.car_helpers import interfaces
from opendbc.car.structs import car
from opendbc.car.vehicle_model import VehicleModel
import openpilot.cereal.messaging as messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.lib.logreader import LogReader

from moonpilot.curvature import MOONPILOT_PREVIEW_MIN_SPEED, MOONPILOT_SMOOTH_TAUS, PathSmooth, response_aligned_curvature
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand, lag_s, param_index, predates, reversals_per_s
from moonpilot.latcontrol import MOONPILOT_DAMPING_T, MOONPILOT_TORQUE_HYSTERESIS, MoonpilotLatControlTorque

EXTRACT_VERSION = 2  # bump whenever _build() changes what it stores; v1's steer_limited paired the wrong frames
ROUTE_SETS = {
  # MoonpilotSteerHysteresis's claim; both drove PathSmooth before d8b15bd1b, which the as-driven config reconstructs.
  "hysteresis": ("0000040d--b31ecdf74b", "0000040e--20803b67bd"),
  # MoonpilotPathSmooth's lag table (moonpilot/docs/curvature.md).
  "smooth": ("00000413--e204529ce6", "00000414--32736d038c", "00000417--e0a9135c4e", "00000418--c554e5658b"),
}
SMOOTH_IN_DELAY_UNTIL = "d8b15bd1b"  # before this commit controlsd added PathSmooth's tau to the controller's lat_delay


def _build(segs: list[str]) -> dict[str, np.ndarray]:
  n_t = ModelConstants.IDX_N
  rows: list[tuple] = []
  models: dict[str, list] = {k: [] for k in ("m_t", "m_action", "m_eof", "m_grid", "m_yaw", "m_yaw_rate", "m_valid")}
  latest: dict[str, tuple] = {}
  cp, sds_t, co_torque, steer_limited, prev_torque = b"", None, None, False, None
  for m in LogReader(segs, sort_by_time=True):
    w, t = m.which(), m.logMonoTime * 1e-9
    if w == "carParams":
      cp = m.carParams.as_builder().to_bytes()
    elif w == "carState":
      latest[w] = (m.carState.vEgo, m.carState.steeringAngleDeg, m.carState.steeringRateDeg, m.carState.steeringPressed)
    elif w == "vehicleParameters":
      vp = m.vehicleParameters
      latest[w] = (vp.angleOffsetDeg, vp.roll, vp.stiffnessFactor, vp.steerRatio)
    elif w == "lateralDelay":
      latest[w] = (m.lateralDelay.lateralDelay,)
    elif w == "lateralTorqueParameters":
      tp = m.lateralTorqueParameters
      latest[w] = (m.valid and tp.useParams, tp.latAccelFactorFiltered, tp.latAccelOffsetFiltered, tp.frictionCoefficientFiltered)
    elif w == "selfdriveState":
      sds_t = t
    elif w == "carOutput":
      co_torque = m.carOutput.actuatorsOutput.torque
    elif w == "modelV2":
      mv = m.modelV2
      grid, yaw, rate = list(mv.orientation.t), list(mv.orientation.z), list(mv.orientationRate.z)
      ok = m.valid and len(grid) == len(yaw) == len(rate) == n_t and grid == list(mv.orientationRate.t)
      for k, v in (("m_t", t), ("m_action", mv.action.desiredCurvature), ("m_eof", mv.timestampEof * 1e-9), ("m_valid", ok),
                   ("m_grid", grid if ok else [0.0] * n_t), ("m_yaw", yaw if ok else [0.0] * n_t), ("m_yaw_rate", rate if ok else [0.0] * n_t)):
        models[k].append(v)
    elif w == "carControl":
      cc = m.carControl
      if sds_t is not None and models["m_t"] and all(k in latest for k in ("carState", "vehicleParameters", "lateralDelay", "lateralTorqueParameters")):
        rows.append((t, sds_t, len(models["m_t"]) - 1, cc.latActive, cc.actuators.curvature, cc.actuators.torque, steer_limited,
                     *latest["carState"], *latest["vehicleParameters"], *latest["lateralDelay"], *latest["lateralTorqueParameters"]))
      # controlsd judges the frame it just published against the newest carOutput, and the next frame reads the verdict.
      prev_torque = cc.actuators.torque
      if co_torque is not None:
        steer_limited = abs(prev_torque - co_torque) > 1e-2

  if not rows or not cp:
    raise ValueError("no complete control frames or no carParams")
  cols = np.array(rows, dtype=float).T
  keys = ("t", "now", "model_idx", "lat_active", "curvature", "torque", "steer_limited", "v_ego", "angle", "rate", "pressed",
          "angle_offset", "roll", "stiffness", "steer_ratio", "lat_delay", "tp_ok", "tp_factor", "tp_offset", "tp_friction")
  d = dict(zip(keys, cols, strict=True))
  d["model_idx"] = d["model_idx"].astype(int)
  for k in ("lat_active", "steer_limited", "pressed", "tp_ok"):
    d[k] = d[k].astype(bool)
  d.update({k: np.array(v, dtype=bool if k == "m_valid" else float) for k, v in models.items()})
  d["cp"] = np.frombuffer(cp, dtype=np.uint8)
  return d


def extract(route: str, root: str = DEFAULT_ROOT) -> dict[str, np.ndarray]:
  """One row per carControl frame, holding the newest of every input controlsd had when it published that frame,
  plus every modelV2's action and path (`model_idx` maps a frame to the model it held)."""
  return cached(route, "lateral", EXTRACT_VERSION, root, _build)


class TorqueConfig(NamedTuple):
  hysteresis: float = 0.0
  damping_t: float = 0.0
  delay_offset: float = 0.0  # s added to the logged lateralDelay


def as_driven(route: str, root: str = DEFAULT_ROOT) -> TorqueConfig:
  """replay_torque's settings as the drive had them, from its initData params and commit.
  Ceiling: a marketplace model's LAT_SMOOTH_SECONDS is not reconstructed (the bundled model's is 0)."""
  commit, params = drive_config(route, root)
  tau = MOONPILOT_SMOOTH_TAUS[min(param_index(params, "MoonpilotPathSmooth"), len(MOONPILOT_SMOOTH_TAUS) - 1)]
  return TorqueConfig(
    hysteresis=MOONPILOT_TORQUE_HYSTERESIS[min(param_index(params, "MoonpilotSteerHysteresis"), len(MOONPILOT_TORQUE_HYSTERESIS) - 1)],
    damping_t=MOONPILOT_DAMPING_T[min(param_index(params, "MoonpilotSteerDamping"), len(MOONPILOT_DAMPING_T) - 1)],
    delay_offset=tau if predates(commit, SMOOTH_IN_DELAY_UNTIL) else 0.0,
  )


def replay_torque(d: dict[str, np.ndarray], cfg: TorqueConfig, lac_out: list | None = None) -> np.ndarray:
  """The fork torque controller's output per frame (carControl's sign), driven the way controlsd drives it.
  Pass `lac_out` to collect each frame's torque log."""
  CP = messaging.log_from_bytes(d["cp"].tobytes(), car.CarParams)
  lac = MoonpilotLatControlTorque(CP, interfaces[CP.carFingerprint](CP), DT_CTRL, cfg.hysteresis, cfg.damping_t)
  VM = VehicleModel(CP)
  out = np.zeros(len(d["t"]))
  for k in range(len(out)):
    VM.update_params(max(d["stiffness"][k], 0.1), max(d["steer_ratio"][k], 0.1))
    if d["tp_ok"][k]:
      lac.update_torque_parameters(float(d["tp_factor"][k]), float(d["tp_offset"][k]), float(d["tp_friction"][k]))
    active = bool(d["lat_active"][k])
    if not active:
      lac.reset()
    CS = SimpleNamespace(vEgo=d["v_ego"][k], steeringAngleDeg=d["angle"][k], steeringRateDeg=d["rate"][k], steeringPressed=bool(d["pressed"][k]))
    lp = SimpleNamespace(angleOffsetDeg=d["angle_offset"][k], roll=d["roll"][k])
    out[k], _, torque_log = lac.update(active, CS, VM, lp, bool(d["steer_limited"][k]), float(d["curvature"][k]), False,
                                       float(d["lat_delay"][k]) + cfg.delay_offset)
    if lac_out is not None:
      lac_out.append(torque_log)
  return out


def fidelity(d: dict[str, np.ndarray], torque: np.ndarray) -> float:
  """p99 |replayed - logged| torque over steering frames."""
  return float(np.percentile(np.abs(torque - d["torque"])[d["lat_active"]], 99))


def action(d: dict[str, np.ndarray]) -> np.ndarray:
  """modelV2.action.desiredCurvature as controlsd held it at each frame."""
  return d["m_action"][d["model_idx"]]


def model_at(d: dict[str, np.ndarray], i: int) -> SimpleNamespace:
  """The fields of model `i` that moonpilot.curvature reads, shaped like a modelV2 reader."""
  return SimpleNamespace(timestampEof=d["m_eof"][i] * 1e9, orientation=SimpleNamespace(z=d["m_yaw"][i], t=d["m_grid"][i]),
                         orientationRate=SimpleNamespace(z=d["m_yaw_rate"][i], t=d["m_grid"][i]))


def replay_smooth(d: dict[str, np.ndarray], tau: float, lead: bool) -> np.ndarray:
  """The curvature request after MoonpilotPathSmooth at `tau`, with or without the reference's lead of `tau`."""
  request = action(d)
  if tau == 0.0:
    return request
  smooth, out = PathSmooth(tau), np.zeros(len(request))
  for k, i in enumerate(d["model_idx"]):
    active, u = bool(d["lat_active"][k]), float(request[k])
    if lead and active:
      u = response_aligned_curvature(model_at(d, i), u, v_ego=float(d["v_ego"][k]), lat_delay=float(d["lat_delay"][k]),
                                     model_recv_time=float(d["m_t"][i]), now=float(d["now"][k]), model_valid=bool(d["m_valid"][i]), lead=tau)
    out[k] = smooth.update(u, active)
  return out


def _torque(args):
  for route in expand(args.routes, ROUTE_SETS):
    d, cfg = extract(route, args.root), as_driven(route, args.root)
    active = d["lat_active"]
    err = fidelity(d, replay_torque(d, cfg))
    print(f"{route}: {active.sum() * DT_CTRL:.0f} s steering, as driven {cfg}, replay vs logged p99 |err| {err:.4f}, "
          + f"logged {reversals_per_s(d['torque'], active):.1f} reversals/s")
    for h in MOONPILOT_TORQUE_HYSTERESIS:
      print(f"  hysteresis {h * 100:.1f}%: {reversals_per_s(replay_torque(d, cfg._replace(hysteresis=h)), active):5.1f} reversals/s")


def _smooth(args):
  pad = int(round(0.5 / DT_CTRL)) + 1  # keeps the lag search from pairing frames across two routes
  parts: dict[str, list[np.ndarray]] = {}
  for route in expand(args.routes, ROUTE_SETS):
    d = extract(route, args.root)
    series = {"mask": d["lat_active"] & (d["v_ego"] > MOONPILOT_PREVIEW_MIN_SPEED), "action": action(d)}
    for tau in MOONPILOT_SMOOTH_TAUS[1:]:
      series[f"{tau} no lead"] = replay_smooth(d, tau, lead=False)
      series[f"{tau} lead"] = replay_smooth(d, tau, lead=True)
    for k, v in series.items():
      parts.setdefault(k, []).extend((v, np.zeros(pad, dtype=v.dtype)))
  s = {k: np.concatenate(v) for k, v in parts.items()}
  print(f"{s['mask'].sum() * DT_CTRL:.0f} s steering above {MOONPILOT_PREVIEW_MIN_SPEED:.0f} m/s; lag behind the model's action:")
  for tau in MOONPILOT_SMOOTH_TAUS[1:]:
    no_lead, led = (lag_s(s["action"], s[f"{tau} {k}"], s["mask"]) * 1e3 for k in ("no lead", "lead"))
    print(f"  tau {tau:.2f} s: {no_lead:4.0f} ms no lead, {led:4.0f} ms lead tau")


def register(sub):
  for name, fn, default, help_text in (
    ("torque", _torque, "hysteresis", "MoonpilotSteerHysteresis: torque reversals per second"),
    ("smooth", _smooth, "smooth", "MoonpilotPathSmooth: request lag behind the model's action"),
  ):
    p = sub.add_parser(name, help=help_text, description=help_text)
    p.add_argument("--routes", nargs="+", default=[default], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
    p.set_defaults(fn=fn)
