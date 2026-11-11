"""Scorecard regression checks and scripted robustness probes."""
import math

import numpy as np

from moonpilot.lab import moving_mean
from moonpilot.lab.driver import _pumps
from moonpilot.lab.planner import _fly_lagged


def flight_peak(f):
  felt = moving_mean(f["a"], 10)
  moving = felt[f["v"] > 2.0]
  return float(moving.min()) if len(moving) else math.nan


def _scorecard_checks(results, hard_rows):
  """The explicit regression thresholds `--check` enforces. They are deliberately loose around the driver envelope:
  the scorecard is a gate against contact, oscillation and a collapsed stop shape, not a target to fit."""
  checks = []

  def add(name, ok, detail):
    checks.append((name, None if ok is None else bool(ok), detail))

  for split, r in results.items():
    if r.get("following"):
      d, p = r["following"]["driver"], r["following"]["planner"]
      if min(len(d), len(p)) < 20:
        add(f"{split} following", None, f"skipped ({len(d)} stretches)")
      else:
        d90, p90 = np.median(d[:, 0]), np.median(p[:, 0])
        d99, p99 = np.median(d[:, 1]), np.median(p[:, 1])
        add(f"{split} following p90", p90 <= max(1.5 * d90, d90 + 0.2), f"{p90:.2f} vs driver {d90:.2f}")
        add(f"{split} following p99", p99 <= max(1.5 * d99, d99 + 0.2), f"{p99:.2f} vs driver {d99:.2f}")
        dhw, phw = np.median(d[:, 3]), np.median(p[:, 3])
        add(f"{split} headway", phw <= max(2.5, 1.5 * dhw), f"{phw:.2f} s vs driver {dhw:.2f} s")
    if r.get("decels"):
      dec = r["decels"]
      gap = min(float(f["gap"].min()) for f in dec["flown"])
      add(f"{split} decel contact", gap >= 0.4, f"closest {gap:.2f} m")
      stops = int(dec["stops"].sum())
      if stops >= 10:
        stop_flights = [f for f, is_stop in zip(dec["flown"], dec["stops"], strict=True) if is_stop]
        pumps = sum(_pumps(f["cmd"]) for f in stop_flights)
        add(f"{split} stop pumps", pumps / stops <= 2.5, f"{pumps / stops:.2f} per stop ({pumps}/{stops})")
      else:
        add(f"{split} stop pumps", None, f"skipped ({stops} stops)")
      driver_peak = np.nanmedian(dec["drv"][:, 1])
      planner_peak = np.nanmedian([flight_peak(f) for f in dec["flown"]])
      add(f"{split} decel peak", not np.isfinite(planner_peak) or planner_peak >= driver_peak - 0.6,
          f"{planner_peak:.2f} vs driver {driver_peak:.2f} m/s^2")
    if r.get("launches") and r["launches"]["keep"]:
      launch = r["launches"]
      if len(launch["keep"]) < 5:
        add(f"{split} launches", None, f"skipped ({len(launch['keep'])} flights)")
      else:
        gap = float(np.nanmedian(launch["gap"]))
        add(f"{split} launch gap", gap >= 3.0, f"{gap:.2f} m at 6-8 s")
        release = np.nanmedian(launch["x"][launch["keep"], 3])
        ask = np.nanmedian(launch["ask"])
        add(f"{split} launch ask", ask <= release + 0.5, f"{ask:.2f} s vs release {release:.2f} s")

  for _, a_brake, gap, _, fcw in hard_rows:
    if a_brake == -3.5:
      add(f"hard brake {a_brake:+.1f}", gap >= 0.4, f"min gap {gap:.2f} m")
    else:
      add(f"hard brake FCW {a_brake:+.1f}", math.isfinite(fcw), f"FCW {fcw:.2f} s")
  return checks


def _adversarial(args):
  """Scripted leads with no manual reference: a close cut-in, a hard brake tap, a radar dropout, and a slow crawl.
  These are robustness probes for the policy; they do not tune to the driver."""
  scenarios = {
    # name: (v0, set kph, gap0, v_lead0, lead accel (t, v), seconds, visible(t))
    "cut-in 12 m": (20.0, 72.0, 12.0, 16.0, lambda t, v: 0.0, 20.0, lambda t: t >= 1.0),
    "brake tap -5/0.5 s": (30.0, 108.0, 45.0, 30.0, lambda t, v: -5.0 if 2.0 <= t < 2.5 else 0.0, 20.0, None),
    "radar dropout 1 s": (20.0, 72.0, 30.0, 15.0, lambda t, v: 0.0, 20.0, lambda t: not (5.0 <= t < 6.0)),
    "crawl 0.3 m/s": (0.4, 10.0, 4.0, 0.3, lambda t, v: 0.0, 20.0, None),
  }
  print("scenario                min gap  peak cmd  end gap  pumps  contact")
  for name, (v0, v_set, gap0, v_lead0, a_lead, seconds, visible) in scenarios.items():
    r = _fly_lagged(v0, v_set, gap0, v_lead0, a_lead, seconds, engine_coast=False, visible_fn=visible)
    gap = float(r["gap"].min())
    print(f"  {name:22s} {gap:6.2f}  {r['cmd'].min():8.2f}  {r['gap'][-1]:7.2f}  {_pumps(r['cmd']):5d}  {'CONTACT' if gap < 0.4 else ''}")


def register(sub):
  sub.add_parser("adversarial", help="scripted cut-in, brake tap, radar dropout and crawl probes").set_defaults(fn=_adversarial)
