"""Identify the longitudinal plant from the drives' own engaged commands."""
import math

import numpy as np

from moonpilot.features import ENGINE_COAST
from moonpilot.lab import drive_config, expand, lag_s, param_index
from moonpilot.lab.planner import SCORECARD_HOLDOUT, _split_routes, control_rate_dt, extract_control
from moonpilot.lab.routes import add_routes_arg

PLANT_BANDS = ((0.0, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 15.0), (15.0, 40.0))
PLANT_CENTERS = tuple((lo + hi) / 2 for lo, hi in PLANT_BANDS)
PLANT_MIN_FRAMES = 50


def _plant_route(job):
  """One drive's engaged, pedal-free target and response at the chain's own lag, with the steady frames that identify
  the gain and the engine-coast setting that changes the braking plant."""
  route, root = job
  try:
    c = extract_control(route, root)
    dt = control_rate_dt(c["t"])
  except (ValueError, FileNotFoundError, OSError):
    return route, None
  ok = c["long_active"] & (c["v_ego"] > 0.5) & ~c["brake_pressed"] & ~c["gas_pressed"] & ~c["should_stop"]
  if ok.sum() * dt < 5.0:
    return route, None
  lag = lag_s(c["a_target"], c["a_ego"], ok, dt=dt, max_lag=1.0)
  k = max(int(round(lag / dt)), 1)
  half = max(int(round(0.5 / dt)), 1)
  change = np.full(len(c["t"]), np.inf)  # the target's change over the surrounding second
  change[half:-half] = np.abs(c["a_target"][2 * half:] - c["a_target"][:-2 * half])
  cmd, a, v = c["a_target"][:-k], c["a_ego"][k:], c["v_ego"][:-k]
  mask = ok[:-k] & ok[k:]
  try:
    ec = param_index(drive_config(route, root)[1], ENGINE_COAST.key)
  except (ValueError, FileNotFoundError, OSError):
    ec = 0
  environment = {"lag": lag, "cmd": cmd[mask], "a": a[mask], "v": v[mask], "steady": (change[:-k] < 0.3)[mask], "ec": ec}
  return route, environment


def _plant_results(root, routes):
  from concurrent.futures import ProcessPoolExecutor
  with ProcessPoolExecutor(4) as pool:
    return [(route, d) for route, d in pool.map(_plant_route, [(r, root) for r in routes]) if d is not None]


def _plant_table(results):
  """Median `aEgo / aTarget` on steady frames by sign and speed band, and the frame counts behind each cell.
  The braking table leaves engine-coast drives out: their friction brakes are held off near the coast."""
  table, counts = {}, {}
  for sign in ("accel", "brake"):
    gains = []
    for lo, hi in PLANT_BANDS:
      pieces = []
      for _, d in results:
        if sign == "brake" and d["ec"]:
          continue
        want = d["cmd"] > 0.2 if sign == "accel" else d["cmd"] < -0.2
        m = d["steady"] & (d["v"] >= lo) & (d["v"] < hi) & want
        if m.any():
          pieces.append(d["a"][m] / d["cmd"][m])
      x = np.concatenate(pieces) if pieces else np.array([])
      counts[(sign, lo, hi)] = len(x)
      gains.append(float(np.median(x)) if len(x) >= PLANT_MIN_FRAMES else 1.0)
    table[sign] = np.asarray(gains)
  return table, counts


class _PlantGain:
  """The identified gain table as a picklable `(cmd, v) -> factor`, so a fitted plant can cross a process pool."""

  def __init__(self, table):
    self.accel = np.asarray(table["accel"], dtype=float)
    self.brake = np.asarray(table["brake"], dtype=float)

  def __call__(self, cmd, v):
    accel_gain, brake_gain = np.interp(v, PLANT_CENTERS, self.accel), np.interp(v, PLANT_CENTERS, self.brake)
    return np.where(np.asarray(cmd) > 0.2, accel_gain, np.where(np.asarray(cmd) < -0.2, brake_gain, 1.0))




def _plant_validation(results, table):
  """Median |aEgo - aTarget| of the open-loop gain prediction against gain=1, on non-engine-coast drives."""
  gain, base, fitted = _PlantGain(table), [], []
  for _, d in results:
    if d["ec"]:
      continue
    base.append(np.abs(d["a"] - d["cmd"]))
    fitted.append(np.abs(d["a"] - d["cmd"] * gain(d["cmd"], d["v"])))
  if not base:
    return math.nan, math.nan
  return float(np.median(np.concatenate(base))), float(np.median(np.concatenate(fitted)))


def _plant_table_text(table, counts):
  parts = []
  for sign in ("accel", "brake"):
    cells = " ".join(f"{lo:.0f}-{hi:.0f}:{gain:.2f}({counts[(sign, lo, hi)]})" for (lo, hi), gain in zip(PLANT_BANDS, table[sign], strict=True))
    parts.append(f"{sign} {cells}")
  return " | ".join(parts)


def _plant_fit(args):
  """The longitudinal plant from engaged drives: the chain's lag, the steady aEgo/aTarget gain by sign and speed band,
  and the gain's held-out prediction error. The gain table seeds `scorecard --plant fitted`; it is identification, not a
  replacement for the car's PCM/brake arbitration."""
  routes = [r.split("--")[0] for r in expand(args.routes, args.root)]
  train, holdout = _split_routes(routes, args.holdout, args.seed)
  train_results, holdout_results = _plant_results(args.root, train), _plant_results(args.root, holdout)
  if not train_results:
    raise ValueError("no engaged drives with enough pedal-free frames")
  table, counts = _plant_table(train_results)
  lags = np.asarray([d["lag"] for _, d in (*train_results, *holdout_results)])
  print(f"plant-fit: {len(routes)} routes, {len(train)} train / {len(holdout)} holdout; "
        + f"lag median {np.median(lags):.3f} s (p10 {np.percentile(lags, 10):.3f}, p90 {np.percentile(lags, 90):.3f})")
  print("  train  " + _plant_table_text(table, counts))
  if holdout_results:
    held_table, held_counts = _plant_table(holdout_results)
    print("  holdout " + _plant_table_text(held_table, held_counts))
  for name, results in (("train", train_results), ("holdout", holdout_results)):
    base, fitted = _plant_validation(results, table)
    print(f"  {name:7s} open-loop |aEgo - aTarget| median: gain=1 {base:.3f}, fitted {fitted:.3f}")


def register(sub):
  parser = sub.add_parser("plant-fit", help="identify the longitudinal plant from engaged drives: lag, gain by sign and speed, held-out error")
  add_routes_arg(parser, ["longitudinal-engaged"])
  parser.add_argument("--holdout", type=float, default=SCORECARD_HOLDOUT, help="fraction of routes held out")
  parser.add_argument("--seed", type=int, default=0, help="split seed")
  parser.set_defaults(fn=_plant_fit)
