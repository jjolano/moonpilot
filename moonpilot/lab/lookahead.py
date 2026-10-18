"""MoonpilotPathLookahead's recorded-turn claims: start advance, apex, open-loop drift, today's-request fidelity.

  python -m moonpilot.lab lookahead

The claim's routes 000003f8-00000410 steer on only four drives (3f8/400/410, plus the engaged 414 on the fade
commit); the newest engaged drives 43e/43f ran the look-ahead as-driven at HEAD~2 and carry the fidelity line.
Every replay is the current code over the drive's own MoonpilotPathSmooth/MoonpilotPathLookahead settings; the
turn metrics then vary one setting (the look-ahead level) from that same per-route configuration. A recorded turn
is a >=0.2 s same-sign run of today's request peaking at >=0.5 m/s2 lateral acceleration; its start is the first
crossing of 20% of that peak and its drift the double integral of the lead-in's extra lateral acceleration."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import clip_curvature

from moonpilot import curvature as curvature_mod
from moonpilot.curvature import MOONPILOT_LOOKAHEAD, MOONPILOT_SMOOTH_TAUS, PathSmooth, response_aligned_curvature
from moonpilot.features import PATH_LOOKAHEAD
from moonpilot.lab import drive_config, expand, param_index
from moonpilot.lab.lateral import action, extract, model_at

ROUTE_SETS = {
  # The claim's routes 000003f8-00000410, restricted to the drives that steered; 414 reuses lateral.py's extract.
  "turns": ("000003f8", "00000400", "00000410", "00000414--32736d038c", "0000043e--d12558e19c", "0000043f--7b506ff942"),
}
LEVEL_NAMES = PATH_LOOKAHEAD.choices
BANDS = ("5-10", "10-17", ">17")
DRIVER_LEAD = (1.5, 1.0, 1.5)  # the unoffered driver-lead variant the claim mentions
UNFADED_SPEEDS = ([0.0, 1.0], [1.0, 1.0])  # MOONPILOT_LOOKAHEAD_SPEEDS with the low-speed fade disabled


def _config(route: str, root: str) -> SimpleNamespace:
  """The drive's smoothing tau and look-ahead level, from its own initData params."""
  _, params = drive_config(route, root)
  tau = MOONPILOT_SMOOTH_TAUS[min(param_index(params, "MoonpilotPathSmooth"), len(MOONPILOT_SMOOTH_TAUS) - 1)]
  level = min(param_index(params, "MoonpilotPathLookahead"), len(MOONPILOT_LOOKAHEAD) - 1)
  return SimpleNamespace(tau=tau, level=level)


def reference(d: dict, tau: float, lookahead, unfaded: bool = False) -> np.ndarray:
  """The request controlsd's curvature reference returns, before MoonpilotPathSmooth and clip_curvature;
  today's request at lookahead=None (the action plus the smoothing lead)."""
  request = action(d)
  out = np.array(request, copy=True)
  ctx = mock.patch.object(curvature_mod, "MOONPILOT_LOOKAHEAD_SPEEDS", UNFADED_SPEEDS) if unfaded else nullcontext()
  with ctx:
    for k, i in enumerate(d["model_idx"]):
      if d["lat_active"][k]:
        out[k] = response_aligned_curvature(model_at(d, int(i)), float(request[k]), v_ego=float(d["v_ego"][k]),
                                            lat_delay=float(d["lat_delay"][k]), model_recv_time=float(d["m_t"][i]),
                                            now=float(d["now"][k]), model_valid=bool(d["m_valid"][i]),
                                            lookahead=lookahead, lead=tau)
  return out


def chain(d: dict, cfg: SimpleNamespace, lookahead) -> tuple[np.ndarray, np.ndarray]:
  """controlsd's whole request path: reference, MoonpilotPathSmooth, clip_curvature. Returns (pre-clip, post-clip).
  While not steering controlsd feeds the tracked road curvature; the logged request is the state clip left."""
  smooth = PathSmooth(cfg.tau) if cfg.tau > 0.0 else None
  request = action(d)
  pre = np.array(request, copy=True)
  post = np.array(request, copy=True)
  prev = 0.0
  for k, i in enumerate(d["model_idx"]):
    if d["lat_active"][k]:
      u = response_aligned_curvature(model_at(d, int(i)), float(request[k]), v_ego=float(d["v_ego"][k]),
                                     lat_delay=float(d["lat_delay"][k]), model_recv_time=float(d["m_t"][i]),
                                     now=float(d["now"][k]), model_valid=bool(d["m_valid"][i]), lookahead=lookahead, lead=cfg.tau)
      if smooth is not None:
        u = smooth.update(u, True)
      pre[k] = u
      post[k] = clip_curvature(float(d["v_ego"][k]), prev, u, float(d["roll"][k]))[0]
      prev = post[k]
    else:
      pre[k] = post[k] = prev = float(d["curvature"][k])
      if smooth is not None:
        smooth.update(float(d["curvature"][k]), False)
  return pre, post


def fidelity(d: dict, pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
  """Median |replay - logged carControl.actuators.curvature| on steering frames: the pre-clip request where
  clip_curvature did not bind (the logged value is the pre-clip request there), and the full post-clip chain."""
  active = d["lat_active"]
  unclipped = active & (pre == post)
  pre_err = float(np.median(np.abs(pre - d["curvature"])[unclipped])) if unclipped.any() else float("nan")
  return pre_err, float(np.median(np.abs(post - d["curvature"])[active]))


def altered_fidelity(d: dict, pre: np.ndarray) -> float:
  """Median |pre-clip replay - logged| on steering frames where the reference or smoothing actually moved the request."""
  fired = d["lat_active"] & (pre != action(d))
  return float(np.median(np.abs(pre - d["curvature"])[fired])) if fired.any() else float("nan")


def _turns(x: np.ndarray, mask: np.ndarray) -> list[slice]:
  """The recorded turns: contiguous steering frames of one curvature sign, at least 0.2 s."""
  sign = np.sign(x)
  sign[np.abs(x) < 1e-5] = 0
  turns: list[slice] = []
  start = None
  current = 0
  for i, (s, valid) in enumerate(zip(sign, mask, strict=True)):
    s = int(s) if valid else 0
    if s != current:
      if start is not None and i - start >= 20:
        turns.append(slice(start, i))
      start, current = (i, s) if s else (None, 0)
  if start is not None and len(x) - start >= 20:
    turns.append(slice(start, len(x)))
  return turns


def turn_values(base: np.ndarray, changed: np.ndarray, speed: np.ndarray, mask: np.ndarray) -> list[tuple]:
  """Per recorded turn: (speed band, start advance s, apex ratio, inside drift m, outward drift m).
  Start = first crossing of 20% of the turn's peak; drift = double integration of the lead-in's extra lateral
  acceleration over the turn, signed toward the inside."""
  out = []
  for turn in _turns(base, mask):
    b, c, v = base[turn], changed[turn], speed[turn]
    peak = float(np.max(np.abs(b)))
    median_v = float(np.median(v))
    if peak * median_v ** 2 < 0.5 or median_v < 5.0:  # a corner, not lane noise: the turn's peak lateral acceleration
      continue
    turn_sign = float(np.sign(b[int(np.argmax(np.abs(b)))]))
    threshold = 0.2 * peak
    base_at = np.flatnonzero(turn_sign * b >= threshold)
    ahead_at = np.flatnonzero(turn_sign * c >= threshold)
    if not len(base_at) or not len(ahead_at):
      continue
    band = BANDS[0] if median_v < 10.0 else BANDS[1] if median_v < 17.0 else BANDS[2]
    apex = int(np.argmax(np.abs(b)))
    delta = (c - b) * v ** 2 * turn_sign
    y = np.cumsum(np.cumsum(delta)) * DT_CTRL ** 2
    out.append((band, (base_at[0] - ahead_at[0]) * DT_CTRL, float(abs(c[apex] / b[apex])),
                float(np.max(y)), float(max(0.0, -np.min(y)))))
  return out


def _med(values) -> float:
  return float(np.median(values)) if values else float("nan")


def _report(name: str, rows: list[tuple]):
  def at(band):
    return [r for r in rows if r[0] == band]
  counts = "/".join(str(len(at(band))) for band in BANDS)
  advance = "/".join(f"{_med([r[1] for r in at(band)]):.2f}" for band in BANDS)
  drift = "/".join(f"{_med([r[3] for r in at(band)]):.3f}" for band in BANDS)
  ratios = [r[2] for r in rows]
  p10 = np.percentile(ratios, 10) if ratios else float("nan")
  print(f"  {name}: turns {counts}; start advance {advance} s at 5-10/10-17/>17")
  print(f"  {name}: apex ratio median/p10 {_med(ratios):.3f}/{p10:.3f}; outward drift median {_med([r[4] for r in rows]):.3f} m")
  print(f"  {name}: open-loop inside drift {drift} m at 5-10/10-17/>17")


def _lookahead(args):
  routes = expand(args.routes, ROUTE_SETS)
  rows: dict[str, list[tuple]] = {name: [] for name in LEVEL_NAMES[1:]}
  rows["driver"] = []
  unfaded_510: dict[str, list] = {name: [] for name in LEVEL_NAMES[1:]}
  errors, steering = [], 0.0
  for route in routes:
    d, cfg = extract(route, args.root), _config(route, args.root)
    mask = d["lat_active"] & d["m_valid"][d["model_idx"]]
    steering += int(mask.sum()) * DT_CTRL
    pre, post = chain(d, cfg, MOONPILOT_LOOKAHEAD[cfg.level])
    pre_err, post_err = fidelity(d, pre, post)
    errors.append(pre_err)
    print(f"  {route}: as-driven {LEVEL_NAMES[cfg.level]} + smooth {cfg.tau:.2f} s; median |err| pre-clip {pre_err:.2e}, post-clip {post_err:.2e} m/s²")
    altered = altered_fidelity(d, pre)
    print(f"  {route}: median |err| where the request was altered {altered:.2e} m/s²" if np.isfinite(altered) else
          f"  {route}: the reference altered no frame in this drive")
    today = reference(d, cfg.tau, None)
    for level in range(1, len(MOONPILOT_LOOKAHEAD)):
      name = LEVEL_NAMES[level]
      rows[name].extend(turn_values(today, reference(d, cfg.tau, MOONPILOT_LOOKAHEAD[level]), d["v_ego"], mask))
      full = reference(d, cfg.tau, MOONPILOT_LOOKAHEAD[level], unfaded=True)
      unfaded_510[name].extend(r[3] for r in turn_values(today, full, d["v_ego"], mask) if r[0] == BANDS[0])
    rows["driver"].extend(turn_values(today, reference(d, cfg.tau, DRIVER_LEAD), d["v_ego"], mask))
  print(f"lookahead: {len(routes)} routes, {steering:.0f} s steering; today's request (as-driven) reconstructed to {_med(errors):.2e} m/s² median")
  print(f"  route range {min(errors):.2e}-{max(errors):.2e} vs logged post-clip carControl.actuators.curvature, pre-clip where clip_curvature did not bind")
  for level in range(1, len(MOONPILOT_LOOKAHEAD)):
    name = LEVEL_NAMES[level]
    _report(f"{name} {MOONPILOT_LOOKAHEAD[level]}", rows[name])
    faded = _med([r[3] for r in rows[name] if r[0] == BANDS[0]])
    print(f"  {name}: 5-10 m/s inside drift with the fade disabled {_med(unfaded_510[name]):.3f} m vs faded {faded:.3f} m")
  _report(f"driver lead-in {DRIVER_LEAD}", rows["driver"])


def register(sub):
  parser = sub.add_parser("lookahead", help="replay MoonpilotPathLookahead turn timing and drift")
  parser.add_argument("--routes", nargs="+", default=["turns"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  parser.set_defaults(fn=_lookahead)
