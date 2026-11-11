"""lead-reaction: how soon the planner answers a followed lead's speed change, beside the driver's own answer to the same one.

Every 30 s of manual following behind one radar track (`driver-follow`'s stretches) is flown through the planner on the lab's
plant behind the lead's recorded path, engine coast off as the device runs unless the arm says otherwise. A lead speed change
is its +-0.5 s slope passing `EVENT_A` after a quiet second and holding for one; its onset is the breakpoint of a two-slope fit
of `vLead`, so every time printed is from the moment the radar shows the change (the radar is `MoonpilotRadarLatency` behind
the car it tracks). Per arm, the driver first:

- reaction: s to the command moving 0.1 m/s^2, to the car's accel moving 0.15, to the car at half the lead's accel change,
  then the lag and the share of the lead's accel the car's accel matches (cross-correlation);
- feet: how many have changed pedal state 1 s and 2 s after the onset — the driver off the gas, on the brake, back on the gas;
  the planner asking to slow, asking for the friction brakes, asking to go;
- cost: command jitter, gas<->brake alternations, friction-brake time and presses, pumps, and the car's accel spread.

A lift is a reaction too, and on this car it is all a shallow ask can be: the brake band raises one to its lamp-free floor,
so "friction brakes" is a command at `BRAKE_ASK` or deeper or a band press, and with engine coast a band press alone.

The flights carry no model ask unless the arm says so, on purpose. In manual following the logged ask is the driver's own
braking read back — over these stretches it correlates 0.82 with the car's aEgo at no lag, and the driver is on the brake on
45% of the frames it asks -0.5 or deeper against 1% of the rest (the report's second line) — so feeding it to a flight that
is somewhere else scores his decisions as the planner's. The "model ask" arms fly it anyway, as the bracket from above.

Simulation against a recorded lead that does not react, on a unity-gain plant; the driver's rows are the car as recorded.
"""
import contextlib
from concurrent.futures import ProcessPoolExecutor
from unittest import mock

import numpy as np

import moonpilot.lead as lead_module
import moonpilot.longitudinal as longitudinal
from openpilot.common.realtime import DT_MDL
from moonpilot.lab import runs
from moonpilot.lab.driver import _follow_stretches, _load, _pumps
from moonpilot.lab.planner import _calibration, _centered_slope, _fly_logged_lead, _hf_sd
from moonpilot.lab.routes import add_calibration_arg

EVENT_A = 0.4  # m/s^2; the lead's +-0.5 s slope that counts as a speed change, and the least its two-slope fit must turn by
QUIET_A = 0.3  # m/s^2; the slope stays under this from 1.5 s to 0.5 s before, and so does the fit's slope before the onset
BRAKE_ASK = -0.25  # m/s^2; a sent command this deep holds the car's brakes on (docs/longitudinal.md, the brake band's switching rates)
ASK = 0.05  # m/s^2; a command past this either way is asking to slow or to go
BEFORE, HORIZON = round(2.0 / DT_MDL), round(5.0 / DT_MDL)  # the state a pedal change is counted from, and how long a response may take

# name: (constants patched for the flight, engine coast, the drive's logged model ask fed in)
ARMS: dict[str, tuple[tuple, bool, bool]] = {
  "today": ((), False, False),
  "model ask": ((), False, True),
  # the model as an early lift: admitted from -0.15 instead of -0.5, released above it
  "model lift": (((longitudinal, "MOONPILOT_MODEL_BRAKE_THRESHOLD", -0.15), (longitudinal, "MOONPILOT_MODEL_BRAKE_RELEASE", -0.05)), False, True),
  # every latency the lead law holds on purpose, shortened at once
  "quicker": (((longitudinal, "MOONPILOT_PREDICT_T", 1.5), (longitudinal, "MOONPILOT_FOLLOW_BAND_T", 0.15), (longitudinal, "MOONPILOT_ARRIVE_T", 15.0),
               (longitudinal, "MOONPILOT_ARRIVE_RAMP_T", 5.0), (longitudinal, "MOONPILOT_LEAVE_DEADBAND", 0.1),
               (lead_module, "MOONPILOT_LEAD_ACCEL_SLOW_RC", 0.5)), False, False),
  "engine coast": ((), True, False),
  "engine coast, model ask": ((), True, True),
  # the braking match inside 1.5 s of headway, 0.85 shipped
  "match 0.45": (((longitudinal, "MOONPILOT_TRACK_BRAKE_V", [0.45, 0.4, 0.2]),), False, False),
}
# The follow band's width, 0.3 s of lead speed shipped: inside it the plan holds speed and the gap absorbs the lead.
ARMS.update({f"{'engine coast, ' if coast else ''}band {band:g}": (((longitudinal, "MOONPILOT_FOLLOW_BAND_T", band),), coast, False)
             for band in (0.6, 1.0) for coast in (False, True)})
# The wide band anchored at the aggressive personality's time gap (the flights fly standard, raw value 1): what it gives back of the headway.
ARMS["band 1, gap 1.25"] = ((*ARMS["band 1"][0], (longitudinal, "MOONPILOT_T_FOLLOW", {**longitudinal.MOONPILOT_T_FOLLOW, 1: 1.25})), False, False)


def onset(v, i, half=30):
  """(index, slope before, slope after) of the slope change in `v` near i: the breakpoint, from 1 s before i to 0.5 s after,
  of the best continuous two-slope fit over i +- `half` samples."""
  t, y, best = np.arange(-half, half + 1) * DT_MDL, v[i - half:i + half + 1], (np.inf, i, 0.0, 0.0)
  for k in range(-20, 11):
    basis = np.column_stack([np.ones(len(t)), np.minimum(t - k * DT_MDL, 0.0), np.maximum(t - k * DT_MDL, 0.0)])
    coef = np.linalg.lstsq(basis, y, rcond=None)[0]
    sse = float(np.sum((basis @ coef - y) ** 2))
    if sse < best[0]:
      best = (sse, i + k, float(coef[1]), float(coef[2]))
  return best[1:]


def lead_events(v_lead):
  """(onset index, +1 speeding up / -1 slowing, accel change) of each lead speed change, with 5 s of the stretch after it."""
  slope, events = _centered_slope(v_lead, 10), []
  for sign in (1.0, -1.0):
    x, last = sign * slope, -HORIZON
    for i in np.flatnonzero((x[1:] >= EVENT_A) & (x[:-1] < EVENT_A)) + 1:
      if i < 45 or i + HORIZON + 10 > len(v_lead) or i - last < 60:
        continue
      if np.max(np.abs(slope[i - 30:i - 10])) >= QUIET_A or np.mean(x[i:i + 20]) < EVENT_A:
        continue
      k, before, after = onset(v_lead, i)
      if sign * (after - before) >= EVENT_A and abs(before) < QUIET_A:
        events.append((k, sign, abs(after - before)))
        last = i
  return events


def _first(mask):
  hit = np.flatnonzero(mask)
  return hit[0] * DT_MDL if len(hit) else np.nan


def _moved(y, k, sign, by):
  """s from k until `y` has moved `by` in the event's direction; nan if it has not in `HORIZON`."""
  return _first(sign * (y[k:k + HORIZON] - y[k]) >= by)


def _changed(state, k):
  """s from k to the first frame, from `BEFORE` ahead of it, where `state` holds; None unless it did not hold at that start."""
  if k < BEFORE or state[k - BEFORE]:
    return None
  return _first(state[k - BEFORE:k + HORIZON]) - BEFORE * DT_MDL


def _alternations(x, by=0.15):
  """Switches between at least +`by` and at most -`by`."""
  side = np.sign(x[np.abs(x) >= by])
  return int(np.sum(side[1:] != side[:-1]))


def _fly(task):
  arm, job, model = task
  patches, engine_coast, with_model = ARMS[arm]
  with contextlib.ExitStack() as stack:
    for module, name, value in patches:
      stack.enter_context(mock.patch.object(module, name, value))
    return _fly_logged_lead((*job, model) if with_model else job, engine_coast=engine_coast)


def _q(x, p):
  x = np.asarray(x, dtype=float)
  x = x[np.isfinite(x)]
  return float(np.percentile(x, p)) if len(x) else np.nan


def _feet(name, stretches):
  """Each stretch as (speed, gap, command or None, slowing, brakes, going): the driver's pedals, or the planner's asks."""
  if name == "driver":
    return [(s["v"], s["gap"], None, ~s["gas"], s["brake"], s["gas"]) for s in stretches]
  coast = ARMS[name][1]
  return [(f["v"], f["gap"], f["cmd"], f["cmd"] <= -ASK, f["pressing"] if coast else f["pressing"] | (f["cmd"] <= BRAKE_ASK), f["cmd"] >= ASK)
          for f in stretches]


def _row(name, stretches, leads, events, minutes):
  """One arm's three lines: reaction, feet, cost."""
  lags, rows = np.arange(round(4.0 / DT_MDL) + 1), _feet(name, stretches)
  cross, lead_power, own_power, moved, feet, accel, headway = np.zeros(len(lags)), np.zeros(len(lags)), np.zeros(len(lags)), [], [], [], []
  for (v, gap, cmd, slowing, brakes, going), v_lead, found in zip(rows, leads, events, strict=True):
    a, ref = _centered_slope(np.asarray(v, dtype=float), 5), _centered_slope(v_lead, 10)
    accel.append(a[5:-5])
    headway.append(np.percentile(gap / np.maximum(v, 0.1), (5, 50, 95)))
    for lag in lags:
      x, y = ref[:len(ref) - lag], a[lag:]
      ok = np.isfinite(x) & np.isfinite(y)
      cross[lag] += np.sum(x[ok] * y[ok])
      lead_power[lag] += np.sum(x[ok] ** 2)
      own_power[lag] += np.sum(y[ok] ** 2)
    for k, sign, size in found:
      moved.append((sign, _moved(cmd, k, sign, 0.1) if cmd is not None else np.nan, _moved(a, k, sign, 0.15), _moved(a, k, sign, 0.5 * size)))
      feet.append((sign, _changed(slowing, k), _changed(brakes, k), _changed(going, k), _changed(~brakes, k)))
  moved, lag = np.asarray(moved), int(np.argmax(cross / np.sqrt(lead_power * own_power)))
  line = f"  {name:22s}"
  for sign, label in ((-1.0, "slows"), (1.0, "speeds up")):
    m = moved[moved[:, 0] == sign]
    line += (f" {label}: command {_q(m[:, 1], 50):4.2f} (p75 {_q(m[:, 1], 75):4.2f}) car {_q(m[:, 2], 50):4.2f} (p75 {_q(m[:, 2], 75):4.2f}) "
             + f"half {_q(m[:, 3], 50):4.2f}, never {100 * np.mean(~np.isfinite(m[:, 3])):2.0f}% |")
  low, mid, high = np.median(headway, axis=0)
  print(line + f" lag {lag * DT_MDL:4.2f} s, matches {cross[lag] / lead_power[lag]:4.2f}, headway {mid:4.2f} s ({low:4.2f}-{high:4.2f} within a stretch)")

  line = f"  {'':22s}"
  for sign, column, label in ((-1.0, 1, "slowing"), (-1.0, 2, "brakes"), (1.0, 3, "going"), (1.0, 4, "off the brake")):
    x = np.asarray([f[column] for f in feet if f[0] == sign and f[column] is not None], dtype=float)
    if len(x):
      line += f" {label}: {100 * np.mean(x <= 1.0):3.0f}% / {100 * np.mean(x <= 2.0):3.0f}%, median {_q(x, 50):+5.2f} s (n {len(x)}) |"
  print(line)

  accel, presses = np.concatenate(accel), [x.stop - x.start for row in rows for x in runs(row[4])]
  line = (f"  {'':22s} brakes {100 * np.mean(np.concatenate([row[4] for row in rows])):4.1f}% of the time, {len(presses) / minutes:4.2f} presses/min "
          + f"({sum(n < round(1.0 / DT_MDL) for n in presses) / minutes:4.2f} under 1 s)"
          + f" | accel sd {np.std(accel):.3f}, p1 {np.percentile(accel, 1):+.2f}, under -0.5 {100 * np.mean(accel < -0.5):4.1f}%")
  if name != "driver":
    cmds = [f["cmd"] for f in stretches]
    line += (f" | jitter {np.median([_hf_sd(c, np.ones(len(c), dtype=bool)) for c in cmds]):.3f}, "
             + f"gas<->brake {sum(_alternations(c) for c in cmds) / minutes:4.2f}/min, pumps {sum(_pumps(c) for c in cmds) / minutes:4.2f}/min, "
             + f"closest {min(float(f['gap'].min()) for f in stretches):.1f} m")
  print(line)


def _report(args):
  d = _load(args.root, ("t", "manual", "v", "a", "gas", "brake", "lead", "lead_radar", "gap", "v_lead", "desired_a"))
  windows = []
  jobs, _ = _follow_stretches(d, _calibration(args.calibration, args.root), windows)
  minutes, leads = len(jobs) * 30.0 / 60, [job[2] for job in jobs]
  events = [lead_events(v_lead) for v_lead in leads]
  sizes = {sign: [size for found in events for _, s, size in found if s == sign] for sign in (-1.0, 1.0)}
  print(f"{len(jobs)} stretches ({minutes:.0f} min) of manual following, calibration {args.calibration}: {len(sizes[-1.0])} lead slow-downs and "
        + f"{len(sizes[1.0])} speed-ups (median {np.median(sizes[-1.0]):.2f} / {np.median(sizes[1.0]):.2f} m/s^2).")
  # Why the flights carry no model ask: over these stretches it is the driver's own braking read back.
  ask, a, brake = (np.concatenate([np.nan_to_num(d[key][w]) for w in windows]) for key in ("desired_a", "a", "brake"))
  entry = longitudinal.MOONPILOT_MODEL_BRAKE_THRESHOLD
  deep = ask <= entry
  print(f"The logged model ask correlates {np.corrcoef(ask, a)[0, 1]:.2f} with the car's own aEgo at no lag; it is at or under {entry:g} on "
        + f"{100 * deep.mean():.1f}% of frames, the driver on the brake on {100 * np.mean(brake[deep] > 0):.0f}% of those and "
        + f"{100 * np.mean(brake[~deep] > 0):.0f}% of the rest.")
  print("Per arm — reaction: s from the lead's onset to the command moving 0.1, the car's accel moving 0.15, the car at half the lead's change; "
        + "feet: share changed by +1 s / +2 s among those not already there 2 s before; cost.")
  _row("driver", [{"v": d["v"][w], "gap": d["gap"][w], "gas": d["gas"][w] > 0, "brake": d["brake"][w] > 0} for w in windows], leads, events, minutes)
  model = [np.nan_to_num(d["desired_a"][w]) for w in windows]
  with ProcessPoolExecutor(4) as pool:
    for arm in args.arms:
      _row(arm, list(pool.map(_fly, [(arm, job, m) for job, m in zip(jobs, model, strict=True)], chunksize=4)), leads, events, minutes)


def register(sub):
  parser = sub.add_parser("lead-reaction", help="how soon the planner answers a followed lead's speed change, against the driver's own feet")
  add_calibration_arg(parser)
  parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=["today"], help="planner variants to fly beside the driver")
  parser.set_defaults(fn=_report)
