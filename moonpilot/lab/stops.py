"""How a stop behind a stopped or stopping lead ends: the driver's own against the lead law flown on the lab's plant.

  python -m moonpilot.lab stop-finish   # MOONPILOT_ARRIVE_MIN_T_REST (moonpilot/longitudinal.py)

Measured by speed rather than time, because the lab's plant has no stopping state: from 2 m/s to the 0.3 m/s where
`should_stop` hands the stop to the controller, the decel as the car passes 1.5 / 1.0 / 0.5 m/s, the gap there, and the
peak before 2 m/s. Lead law only (no model ask): the drive's recorded ask belongs to the driver's path, and replayed
open-loop it keeps braking a flown car that has fallen behind that path.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import moonpilot.longitudinal as longitudinal
from moonpilot.lab import driver
from moonpilot.lab.planner import DT_MDL, _calibration, _fly_logged_lead

KEYS = ("t", "manual", "v", "a", "lead", "gap", "v_lead")
ONSET_A = -0.3  # m/s^2 over 0.5 s: the braking run that ends at rest
PRE_S, POST_S = 5.0, 8.0  # s of the same lead before the driver's onset, and past his rest, that a flight covers


def _stops(d):
  """Each manual stop from 4 m/s or more behind a lead at rest (under 1 m/s, within 15 m), one continuous lead from the
  braking onset to rest: (window start, onset, rest, window end) indices."""
  v, gap, v_lead, a = d["v"], d["gap"], d["v_lead"], d["a"]
  n = len(v)
  rid = np.unique(d["route"], return_inverse=True)[1]
  lead = d["lead"] > 0
  joined = np.r_[False, (np.abs(np.diff(gap)) <= 1.5) & (np.abs(np.diff(v_lead)) <= 1.0) & (rid[1:] == rid[:-1])
                 & (np.diff(d["t"]) <= 1.5 * DT_MDL)] & lead
  run = np.cumsum(~joined)
  a_s = np.convolve(a, np.ones(10) / 10, mode="same")
  manual = d["manual"] > 0
  out = []
  for kr in np.flatnonzero((v[1:] < 0.1) & (v[:-1] >= 0.1)) + 1:
    lo = max(kr - round(20.0 / DT_MDL), 0)
    if rid[lo] != rid[kr] or v[lo:kr].max() < 5.0 or not manual[lo:kr + 1].all() or not (lead[kr] and v_lead[kr] <= 1.0 and gap[kr] <= 15.0):
      continue
    on, quiet = kr, 0
    while on > lo:
      quiet = 0 if a_s[on - 1] < ONSET_A else quiet + 1
      if quiet > 10:
        break
      on -= 1
    on += quiet
    if kr - on < round(1.5 / DT_MDL) or v[on] < 4.0 or run[on] != run[kr]:
      continue
    start, end = on, min(kr + round(POST_S / DT_MDL), n)
    while start > on - round(PRE_S / DT_MDL) and run[start - 1] == run[on]:
      start -= 1
    while end > kr and run[end - 1] != run[kr]:
      end -= 1
    out.append((start, on, kr, end))
  return out


def finish(v, a, gap):
  """From 2 m/s to 0.3 m/s: its time, the decel (0.5 s mean) passing 1.5 / 1.0 / 0.5 m/s, the gap at 0.3 m/s, and the
  peak decel before 2 m/s. None for a stop that never reaches 0.3 m/s."""
  a_s = np.convolve(a, np.ones(10) / 10, mode="same")
  below2 = np.flatnonzero(v < 2.0)
  if not len(below2):
    return None
  k2 = below2[0]
  below03 = np.flatnonzero(v[k2:] < 0.3)
  if not len(below03):
    return None
  k03 = k2 + below03[0]
  at = [a_s[k2 + np.flatnonzero(v[k2:k03 + 1] < s)[0]] for s in (1.5, 1.0, 0.5)]
  return ((k03 - k2) * DT_MDL, *at, gap[k03], a_s[:k2].min() if k2 else np.nan)


def _line(name, rows, closest=None):
  x = np.array([r for r in rows if r is not None])
  tail = f"  closest {closest:4.1f} m" if closest is not None else ""
  print(f"  {name:18s} {len(x):3d}  {np.median(x[:, 0]):4.1f} s (p90 {np.percentile(x[:, 0], 90):3.1f})  "
        + " / ".join(f"{np.median(x[:, i]):+.2f}" for i in (1, 2, 3)) + f"   {np.median(x[:, 4]):4.1f} m   {np.median(x[:, 5]):+.2f}" + tail)


def _stop_finish(args):
  """The driver's stops behind a stopped or stopping lead, and the lead law flown through them at each
  MOONPILOT_ARRIVE_MIN_T_REST: how long the end takes, how firm it is, and where it lands."""
  d = driver._load(args.root, KEYS)
  stops = _stops(d)
  calibration = _calibration(args.calibration, args.root)
  jobs = []
  for start, _on, kr, end in stops:
    w = slice(start, end)
    jobs.append((d["v"][w], d["gap"][w], d["v_lead"][w], float(d["a"][start]), calibration, max(d["v"][start:kr].max(), 1.0) * 3.6 + 5.0,
                 np.zeros(end - start)))
  print(f"{len(stops)} manual stops behind a lead at rest; flown from up to {PRE_S:g} s before the driver's onset, lead law only")
  print(f"  {'':18s}   n  2 -> 0.3 m/s       decel at 1.5 / 1.0 / 0.5 m/s   gap at 0.3   peak above 2 m/s")
  _line("driver", [finish(d["v"][s:e], d["a"][s:e], d["gap"][s:e]) for s, _, _, e in stops])
  saved = longitudinal.MOONPILOT_ARRIVE_MIN_T_REST
  try:
    for min_t in args.min_t:
      longitudinal.MOONPILOT_ARRIVE_MIN_T_REST = min_t  # forked workers inherit it
      with ProcessPoolExecutor(3) as pool:
        flown = list(pool.map(_fly_logged_lead, jobs, chunksize=4))
      _line(f"rest min_t {min_t:g} s", [finish(f["v"], f["a"], f["gap"]) for f in flown], min(f["gap"].min() for f in flown))
  finally:
    longitudinal.MOONPILOT_ARRIVE_MIN_T_REST = saved


def register(sub):
  p = sub.add_parser("stop-finish", help="how a stop behind a stopped lead ends: the driver against the lead law per rest min_t",
                     description=_stop_finish.__doc__)
  p.add_argument("--min-t", nargs="+", type=float, default=[longitudinal.MOONPILOT_ARRIVE_MIN_T, longitudinal.MOONPILOT_ARRIVE_MIN_T_REST],
                 help="MOONPILOT_ARRIVE_MIN_T_REST values to fly, s (the first default is the moving-lead value, the law before it)")
  p.add_argument("--calibration", default="00000451", help="drive whose recorded lag and radar latency seed the flights")
  p.set_defaults(fn=_stop_finish)
