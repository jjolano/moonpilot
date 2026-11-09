"""The braking half of `jerk_limit` (`MOONPILOT_JERK_ONSET` / `_DOWN` / `_EMERGENCY` in `moonpilot/longitudinal.py`): how hard
the driver's own brake presses build, what the down-jerk schedule allows at each ask depth, and what a shallower shallow
onset would do to the decels the planner flies. Command-path evidence: the car is held as recorded.
"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from openpilot.common.realtime import DT_MDL
from moonpilot.lab.driver import _decel_line, _fly_stages, _load
from moonpilot.lab.planner import _calibration, _decel_jobs, _hard_brake_run

DRIVER_KEYS = ("t", "manual", "v", "a", "brake", "lead", "gap", "v_lead")
BUILD_SPAN_S, BUILD_WINDOW_S, MIN_BUILD_SPEED = 0.5, 2.0, 5.0  # the change over this span, this far into the press, above this speed
SHALLOW_PEAK = 1.2  # m/s^2; an episode peaking under this is the shallow brake the shallow onset governs
DEPTHS = (-0.3, -1.0, -2.0)  # m/s^2 of ask whose rate is printed beside the driver's build


def _driver_build(d):
  """Per manual brake press above `MIN_BUILD_SPEED`: the steepest `BUILD_SPAN_S` drop of aEgo inside the first
  `BUILD_WINDOW_S` from the press, the episode's peak decel, and the decel the drop began at. Nothing is read *at* the press
  frame: the switch lags the pedal and aEgo is still rising on a quarter of presses, so a press-frame slope is a lower bound
  on the application, not the application."""
  n_span, n_win = round(BUILD_SPAN_S / DT_MDL), round(BUILD_WINDOW_S / DT_MDL)
  brake, a, v, t = d["brake"] > 0, d["a"], d["v"], d["t"]
  rows = []
  for i in np.flatnonzero(brake[1:] & ~brake[:-1]) + 1:
    if v[i] < MIN_BUILD_SPEED or i + n_win >= len(a):
      continue
    seg = a[i:i + n_win]
    if not np.all(np.isfinite(seg)) or np.any(np.diff(t[i - 1:i + n_win]) > 0.2):  # one drive, contiguous
      continue
    drop = seg[:-n_span] - seg[n_span:]
    k = int(np.argmax(drop))
    rows.append((v[i], drop[k] / BUILD_SPAN_S, -seg.min(), -seg[k]))
  return np.asarray(rows, dtype=float).reshape(-1, 4)


def _down_rate(a_cmd, v_ego=10.0, lead_braking=False):
  """`jerk_limit`'s own down rate at one ask, read back off its output rather than restated."""
  from moonpilot.longitudinal import jerk_limit
  return (0.05 - jerk_limit(a_cmd, 0.05, DT_MDL, v_ego, lead_braking=lead_braking)) / DT_MDL


def _fly_one(job, onset):
  """One lead-caused decel flown through the planner with `MOONPILOT_JERK_ONSET` set to `onset`."""
  import moonpilot.longitudinal as longitudinal
  saved, longitudinal.MOONPILOT_JERK_ONSET = longitudinal.MOONPILOT_JERK_ONSET, onset
  try:
    return _fly_stages(job)
  finally:
    longitudinal.MOONPILOT_JERK_ONSET = saved


def _decel_ramp(args):
  """The driver's own brake build by episode depth, then the schedule `jerk_limit` allows at each ask, then the planner flown
  through every lead-caused manual decel at each `--onsets` value with `decel-peaks`' pump/jitter/press/peak/gap line and the
  hard brake's gap beside it. The shallow row is the brake the shallow onset governs; the deep asks are the ones a hard brake
  depends on."""
  import moonpilot.longitudinal as longitudinal
  d = _load(args.root, DRIVER_KEYS)
  x = _driver_build(d)
  print(f"driver: {len(x)} manual brake presses above {MIN_BUILD_SPEED:g} m/s over {len(np.unique(d['route_id']))} drives; aEgo's "
        + f"steepest {BUILD_SPAN_S:g} s drop inside {BUILD_WINDOW_S:g} s, and the episode's peak (median, p75)")
  bins = (("shallow, peak < 1.2", x[:, 2] < SHALLOW_PEAK), ("medium, 1.2-2.5", (x[:, 2] >= 1.2) & (x[:, 2] < 2.5)),
          ("deep, >= 2.5", x[:, 2] >= 2.5))
  for name, m in bins:
    if m.any():
      print(f"  {name:22s} n {m.sum():4d}  build {np.median(x[m, 1]):.2f} ({np.percentile(x[m, 1], 75):.2f}) m/s^3  "
            + f"peak {np.median(x[m, 2]):.2f} m/s^2, rising at {np.median(x[m, 3]):.2f}")
  print("planner at 10 m/s, no lead braking: " + ", ".join(f"ask {a:+.1f}: {_down_rate(a):.2f} m/s^3" for a in DEPTHS)
        + f", ACCEL_MIN: {_down_rate(longitudinal.ACCEL_MIN):.2f}; below {longitudinal.MOONPILOT_JERK_LAUNCH_SPEED:g} m/s or while any"
        + f" lead brakes: {_down_rate(-2.0, lead_braking=True):.2f}")

  a_s = np.concatenate([np.convolve(d["a"][d["route_id"] == r], np.ones(10) / 10, mode="same") for r in np.unique(d["route_id"])])
  d["route"] = d["route_id"]  # `_decel_jobs` groups by route; `_load` names drives by id, not by name
  jobs, drv = _decel_jobs(d, a_s)
  if not jobs:
    print("lead-caused decels: none")
    return
  calibration = _calibration(args.calibration, args.root)
  print(f"lead-caused decels: {len(jobs)} flown closed-loop with the shallow onset at each value")
  for onset in args.onsets:
    with ProcessPoolExecutor(max_workers=4) as pool:
      flown = list(pool.map(partial(_fly_one, onset=onset), [(*job, calibration) for job in jobs]))
    saved = longitudinal.MOONPILOT_JERK_ONSET
    longitudinal.MOONPILOT_JERK_ONSET = onset
    try:
      gap = _hard_brake_run("fork+engine", 20.0, -3.5, True)[0]
    finally:
      longitudinal.MOONPILOT_JERK_ONSET = saved
    print(f"  {onset:4.2f}  " + _decel_line(flown, drv) + f"  |  hard brake 20 m/s -3.5: gap {gap:.2f} m")
  print("  an onset well under the driver's own shallow build brakes slower than he does; a hard brake's gap is set by the deep"
        + " asks, so what a shallower shallow onset changes is felt in the gentle brake, not bought back in the stop")


def register(sub):
  parser = sub.add_parser("decel-ramp", help="the braking schedule: the driver's brake build by depth, jerk_limit's rates, decels at each onset")
  parser.add_argument("--onsets", nargs="+", type=float, default=[1.0, 0.5], help="shallow onsets to fly, m/s^3")
  parser.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  parser.set_defaults(fn=_decel_ramp)
