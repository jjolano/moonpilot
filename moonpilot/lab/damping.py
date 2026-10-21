"""MoonpilotSteerDamping's claims: replay each choice over recorded curves; measure the wheel, P and the rate sensor the band was written against.

  python -m moonpilot.lab damping   # D's share of the command, the turn build-up and release (AGENTS.md, The lateral controller)
  python -m moonpilot.lab weave     # the wheel-motion and rate-sensor measurements beside it
  python -m moonpilot.lab release   # where a low-speed turn's release comes from: car, request, torque, the command's terms

The windows follow the analyses that produced the claims: 6 s of data every 3 s; curve windows are hands-off
and all-active at a mean 3-17 m/s with at least 1 m/s^2 of lateral acceleration (device yaw x speed); turns
are peaks of the 0.2 s-smoothed lateral acceleration whose whole 10%-of-peak entry and exit stay under one
mask. Replays feed logged messages on their own logMonoTime, vary only the damping time from each drive's own
configuration, and cache every level so `damping` and `weave` share one pass.
"""
import os

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.tools.lib.logreader import LogReader

from moonpilot.lab import DEFAULT_CACHE, cached, expand, lag_s, runs, segments
from moonpilot.lab.lateral import as_driven, extract, fidelity, replay_torque
from moonpilot.latcontrol import MOONPILOT_DAMPING_SPEEDS, MOONPILOT_DAMPING_T

EXTRACT_VERSION = 1  # bump whenever _yaw_build changes what it stores
REPLAY_VERSION = 1  # bump whenever the replay cache's arrays change meaning
ROUTE_SETS = {
  # The routes the claims cite (AGENTS.md: 151 min manual, 100 min engaged); routes without rlogs are skipped.
  "curve": tuple(f"0000{i:04x}" for i in range(0x3F8, 0x411)),
}
WINDOW = 600  # frames (6 s), stepped by 300
STEP = 300
G = 9.81


def _yaw_build(segs: list[str]) -> dict[str, np.ndarray]:
  """Device yaw and selfdriveState.active per message; the shared lateral extract carries neither."""
  mt, yaw, st, sd = [], [], [], []
  for m in LogReader(segs, sort_by_time=True):
    w, t = m.which(), m.logMonoTime * 1e-9
    if w == "deviceMotion":
      mt.append(t)
      yaw.append(m.deviceMotion.angularVelocityDevice.z)
    elif w == "selfdriveState":
      st.append(t)
      sd.append(m.selfdriveState.active)
  return {"mt": np.array(mt), "yaw": np.array(yaw), "st": np.array(st), "sd": np.array(sd, dtype=float)}


def _series(route: str, root: str):
  """The lateral extract plus device yaw (rad/s) and selfdriveState.active on its frames."""
  d = extract(route, root)
  e = cached(route, "damping", EXTRACT_VERSION, root, _yaw_build)
  yaw = np.interp(d["t"], e["mt"], e["yaw"], left=0.0, right=0.0) if len(e["mt"]) else np.zeros(len(d["t"]))
  sd = np.interp(d["t"], e["st"], e["sd"], left=0.0, right=0.0) > 0.5 if len(e["st"]) else np.zeros(len(d["t"]), dtype=bool)
  return d, yaw, sd


def _replays(d: dict[str, np.ndarray], route: str, root: str):
  """The drive's own config and, per damping level, torque plus p/i/d/f and the setpoint; cached per route."""
  cfg = as_driven(route, root)
  names = [os.path.basename(os.path.dirname(s)) for s in segments(route, root)]
  path = os.path.join(DEFAULT_CACHE, f"{route}.damping-replay.v{REPLAY_VERSION}.npz")
  key = np.array([cfg.hysteresis, cfg.delay_offset, *MOONPILOT_DAMPING_T])
  if os.path.exists(path):
    z = dict(np.load(path))
    if z["segments"].tolist() == names and np.array_equal(z["cfg"], key):
      return cfg, z
  z = {k: np.zeros((len(MOONPILOT_DAMPING_T), len(d["t"]))) for k in ("p", "i", "d", "f", "setpoint", "torque")}
  for level, t in enumerate(MOONPILOT_DAMPING_T):
    logs = []
    z["torque"][level] = replay_torque(d, cfg._replace(damping_t=t), lac_out=logs)
    for k, field in (("p", "p"), ("i", "i"), ("d", "d"), ("f", "f"), ("setpoint", "desiredLateralAccel")):
      z[k][level] = [getattr(log, field) for log in logs]
  z.update(segments=np.array(names), cfg=key)
  os.makedirs(DEFAULT_CACHE, exist_ok=True)
  np.savez(file=path, **z)  # ty: ignore[invalid-argument-type]  # z is one ndarray per array name
  return cfg, z


def _load(route: str, root: str):
  """One route's extract, yaw-derived lateral acceleration, class masks, config and replay cache; None without rlogs."""
  try:
    d, yaw, sd = _series(route, root)
    cfg, z = _replays(d, route, root)
  except FileNotFoundError as e:
    print(f"{route}: skipped, {e}")
    return None
  la = np.abs(yaw * d["v_ego"])
  masks = {"manual": ~sd & ~d["lat_active"] & (d["v_ego"] > 3), "op": sd & d["lat_active"] & ~d["pressed"] & (d["v_ego"] > 3)}
  return d, la, masks, cfg, z


def _fft(x: np.ndarray):
  return np.fft.rfft((x - x.mean()) * np.hanning(len(x))), np.fft.rfftfreq(len(x), DT_CTRL)


def _band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
  """The signal rebuilt from one band (the claims' band estimator)."""
  X, f = _fft(x)
  X[(f < lo) | (f > hi)] = 0
  return np.fft.irfft(X, len(x))


def _band_rms(x: np.ndarray, lo: float, hi: float) -> float:
  return float(np.sqrt(np.mean(_band(x, lo, hi) ** 2)))


def _power(x: np.ndarray):
  """A detrended window's power spectrum and its frequencies."""
  y = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
  X = np.fft.rfft(y * np.hanning(len(y)))
  return np.abs(X) ** 2, np.fft.rfftfreq(len(y), DT_CTRL)


def _hf_share(x: np.ndarray) -> float:
  """Share of the wheel's motion amplitude (RMS of power) at 1-5 Hz of all its 0.1-5 Hz motion."""
  p, f = _power(x)
  return float(np.sqrt(p[(f >= 1) & (f <= 5)].sum() / p[(f >= 0.1) & (f <= 5)].sum()))


def _split(x: np.ndarray):
  """(below 1 Hz, 1-5 Hz) RMS: turn-scale correction against flutter."""
  X, f = _fft(x)
  return tuple(float(np.sqrt(np.mean(np.fft.irfft(X * band, len(x)) ** 2))) for band in (f < 1, (f >= 1) & (f <= 5)))


def _smooth(x: np.ndarray, n: int = 20) -> np.ndarray:
  return np.convolve(x, np.ones(n) / n, mode="same")


def _curve_windows(d: dict[str, np.ndarray], la: np.ndarray) -> list[int]:
  """Starts of 6 s every 3 s that are all hands-off and active, unbroken, at a mean 3-17 m/s, reaching 1 m/s^2."""
  act = d["lat_active"] & ~d["pressed"]
  t, v = d["t"], d["v_ego"]
  return [s for s in range(0, len(t) - WINDOW, STEP) if act[s:s + WINDOW].all() and np.all(np.diff(t[s:s + WINDOW]) < 0.05)
          and 3 <= v[s:s + WINDOW].mean() < 17 and la[s:s + WINDOW].max() >= 1]


def _class_windows(mask: np.ndarray, t: np.ndarray) -> list[int]:
  """Starts of 6 s every 3 s fully inside the mask, with unbroken timestamps."""
  out = []
  for r in runs(mask):
    for s in range(r.start, r.stop - WINDOW + 1, STEP):
      if np.all(np.diff(t[s:s + WINDOW]) < 0.05):
        out.append(s)
  return out


def _turns(la: np.ndarray, mask: np.ndarray) -> list[int]:
  """Peaks of the smoothed lateral acceleration >= 1 m/s^2 that are the highest within +-2 s, under mask."""
  return [int(k) for k in np.flatnonzero((la >= 1) & mask) if la[k] == la[max(k - 200, 0):k + 201].max()]


def _stretches(la: np.ndarray, k: int, mask: np.ndarray):
  """(entry, exit) index pairs from the 10% point to the peak at k and back, or None per side."""
  out = []
  for step in (-1, 1):
    i = k
    while 0 < i < len(la) - 1 and abs(i - k) < 600 and la[i] >= 0.1 * la[k]:
      i += step
    lo, hi = sorted((i, k))
    out.append((lo, hi) if la[i] < 0.1 * la[k] and mask[lo:hi + 1].all() else None)
  return out


def _shape(la: np.ndarray, t: np.ndarray, k: int, step: int, mask: np.ndarray):
  """Steepest rate / peak over one side of the turn at k, from the 10% point; None if it leaves the mask."""
  peak, i, c10, c90 = la[k], k, None, None
  while 0 < i < len(la) - 1 and abs(t[i] - t[k]) < 6:
    i += step
    if c90 is None and la[i] < 0.9 * peak:
      c90 = i
    if la[i] < 0.1 * peak:
      c10 = i
      break
  if c10 is None or c90 is None:
    return None
  lo, hi = sorted((c10, k))
  if not mask[lo:hi + 1].all() or np.any(np.diff(t[lo:hi + 1]) > 0.05):
    return None
  return float(np.abs(np.gradient(la[lo:hi + 1], t[lo:hi + 1])).max() / peak)


def _rate_lag(angle: np.ndarray, rate: np.ndarray):
  """How far the rate sensor trails the angle's derivative, and their correlation there."""
  deriv = np.gradient(np.convolve(angle, np.ones(5) / 5, mode="same"), DT_CTRL)
  lag = lag_s(deriv, rate, np.ones(len(angle), dtype=bool), max_lag=0.3)
  i = max(int(round(lag / DT_CTRL)), 0)
  a, b = deriv[10:len(deriv) - 10 - i], rate[10 + i:len(rate) - 10]
  return lag, (float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 else float("nan"))


def _plan_ff(d: dict[str, np.ndarray], sl: slice) -> np.ndarray:
  """The plan feedforward the claims measure D against, in lateral-acceleration units."""
  return d["curvature"][sl] * d["v_ego"][sl] ** 2 - d["roll"][sl] * G - d["tp_offset"][sl]


def _damping(args):
  weave = {t: [] for t in MOONPILOT_DAMPING_T}
  buzz = {t: [] for t in MOONPILOT_DAMPING_T}
  entry = {t: [] for t in MOONPILOT_DAMPING_T}
  exit_ = {t: [] for t in MOONPILOT_DAMPING_T}
  release = {"manual": [], "op": []}
  for name in expand(args.routes, ROUTE_SETS):
    got = _load(name, args.root)
    if got is None:
      continue
    d, la, masks, cfg, z = got
    las = _smooth(la)
    i_as = MOONPILOT_DAMPING_T.index(cfg.damping_t)
    active = d["lat_active"]
    for cls in ("manual", "op"):  # before the guard: an all-manual drive still counts toward the driver's release
      for k in _turns(las, masks[cls]):
        if 3 <= d["v_ego"][k] < 10:
          shape = _shape(las, d["t"], k, 1, masks[cls])
          if shape is not None:
            release[cls].append(shape)
    if not active.any():
      print(f"{name}: skipped, no steering frames")
      continue
    err = fidelity(d, z["torque"][i_as])
    can = float(np.median(np.abs(z["torque"][i_as] - d["torque"])[active]) * 1500)
    windows = _curve_windows(d, la)
    print(f"{name}: as driven T={cfg.damping_t:.2f}s, replay vs logged p99 |err| {err:.4f} (median {can:.1f} CAN), {len(windows)} curve windows")
    for s in windows:
      sl = slice(s, s + WINDOW)
      for level, t in enumerate(MOONPILOT_DAMPING_T):
        weave[t].append(_band_rms(z["d"][level][sl], 1, 2) / max(_band_rms(z["p"][level][sl], 1, 2), 1e-12))
        buzz[t].append(_band_rms(z["torque"][level][sl], 5, 50))
    for k in _turns(las, masks["op"]):
      turn = float(np.sign(z["setpoint"][i_as][k])) or 1.0
      for side, span in zip(("entry", "exit"), _stretches(las, k, masks["op"]), strict=True):
        if span is None:
          continue
        sl = slice(span[0], span[1] + 1)
        push = float(np.sum(np.abs(_plan_ff(d, sl))))
        if push < 1e-6:
          continue
        for level, t in enumerate(MOONPILOT_DAMPING_T):
          (entry if side == "entry" else exit_)[t].append(float(np.sum(turn * z["d"][level][sl]) / push))
  off = float(np.median(buzz[0.0]))
  for t in MOONPILOT_DAMPING_T:
    med_entry = 100 * float(np.median(entry[t])) if entry[t] else float("nan")
    p10 = 100 * float(np.percentile(entry[t], 10)) if entry[t] else float("nan")
    pos, n_exit = sum(x > 0 for x in exit_[t]), len(exit_[t])
    print(f"T={t:.2f}s: D/P 1-2Hz {100 * float(np.median(weave[t])):.0f}%, command >5Hz {100 * (float(np.median(buzz[t])) / off - 1):+.0f}%, "
          + f"turn build-up {med_entry:+.1f}% of plan feedforward (p10 {p10:+.1f}%, n={len(entry[t])}), "
          + f"release {100 * float(np.median(exit_[t])) if n_exit else float('nan'):+.1f}% ({pos}/{n_exit} exits positive)")
  man = float(np.median(release["manual"])) if release["manual"] else float("nan")
  op = float(np.median(release["op"])) if release["op"] else float("nan")
  print(f"steepest release / peak at 3-10 m/s (median): manual {man:.2f} (n={len(release['manual'])}), "
        + f"openpilot {op:.2f} (n={len(release['op'])})")
  lead = [f"{np.degrees(np.arctan(2 * np.pi * 1.2 * t)):.1f}" for t in MOONPILOT_DAMPING_T[1:]]
  print(f"analytic feedback lead at 1.2 Hz: {' / '.join(lead)} degrees; damping speed schedule {MOONPILOT_DAMPING_SPEEDS}")


def _weave(args):
  hf = {"manual": [], "op": []}
  spec = {"manual": [], "op": []}
  flutter = []
  lags, corrs, fids = [], [], []
  minutes = {"manual": 0.0, "op": 0.0}
  for name in expand(args.routes, ROUTE_SETS):
    got = _load(name, args.root)
    if got is None:
      continue
    d, la, masks, cfg, z = got
    i_as = MOONPILOT_DAMPING_T.index(cfg.damping_t)
    if d["lat_active"].any():
      err = fidelity(d, z["torque"][i_as])
      fids.append(err)
      shown = f"{err:.4f}"
    else:
      shown = "n/a (no steering)"
    for cls in ("manual", "op"):
      minutes[cls] += float(masks[cls].sum() * DT_CTRL / 60)
      for s in _class_windows(masks[cls], d["t"]):
        sl = slice(s, s + WINDOW)
        v, turn = float(d["v_ego"][sl].mean()), float(la[sl].max())
        if turn < 1 or not 3 <= v < 17:
          continue
        hf[cls].append((v, turn, _hf_share(d["angle"][sl])))
        p, f = _power(d["angle"][sl])
        band = (f >= 1) & (f <= 5)
        spec[cls].append(p[band] / p[band].sum())
        lag, r = _rate_lag(d["angle"][sl], d["rate"][sl])
        lags.append(lag)
        corrs.append(r)
        if cls == "op":
          p_low, p_high = _split(z["p"][i_as][sl])
          ff_low, _ = _split(_plan_ff(d, sl))
          wheel, command = _band(d["angle"][sl], 1, 5), _band(z["p"][i_as][sl], 1, 5)
          flutter.append((v, p_high / max(p_low, 1e-12), p_low / max(ff_low, 1e-12), abs(float(np.corrcoef(wheel, command)[0, 1]))))
    print(f"{name}: manual {minutes['manual']:.0f} min, openpilot {minutes['op']:.0f} min (running), replay vs logged p99 |err| {shown}")
  for lo, hi in ((3, 10), (10, 17)):
    for t_lo, t_hi in ((1, 2), (2, 3), (3, 9)):
      cells = {cls: [r[2] for r in hf[cls] if lo <= r[0] < hi and t_lo <= r[1] < t_hi] for cls in ("manual", "op")}
      if all(len(v) >= 3 for v in cells.values()):
        e, m = 100 * float(np.median(cells["op"])), 100 * float(np.median(cells["manual"]))
        print(f"wheel motion above 1 Hz, {lo}-{hi} m/s, turn {t_lo}-{t_hi} m/s^2: "
              + f"engaged {e:.0f}% (n={len(cells['op'])}), manual {m:.0f}% (n={len(cells['manual'])})")
  for cls in ("manual", "op"):
    share = np.mean(spec[cls], axis=0)
    print(f"{cls} wheel 1-5 Hz power: {100 * share[:6].sum():.0f}% of it at 1-2 Hz ({len(spec[cls])} windows)")
  for lo, hi in ((3, 7), (7, 12), (12, 17)):
    rows = [r for r in flutter if lo <= r[0] < hi]
    if rows:
      med = np.median(np.array([r[1:] for r in rows]), axis=0)
      print(f"engaged P at {lo}-{hi} m/s: 1-5 Hz / below-1Hz {med[0]:.2f}, below-1Hz / plan feedforward {med[1]:.2f}, "
            + f"|corr| with wheel 1-5 Hz {med[2]:.2f} (n={len(rows)})")
  print(f"analyzed at v>3 m/s: {minutes['manual']:.0f} min manual, {minutes['op']:.0f} min engaged")
  if lags:
    fid = float(np.median(fids)) if fids else float("nan")
    print(f"rate sensor behind the angle's derivative: median {1000 * float(np.median(lags)):+.0f} ms, "
          + f"r {float(np.nanmedian(corrs)):.2f} (n={len(lags)} curve windows, replay vs logged p99 |err| median {fid:.4f})")


def _release(args):
  """Where a low-speed turn's release comes from, on recorded turns peaking at 3-10 m/s: the steepest release / peak of
  the car's lateral accel by hand and with openpilot steering, and of the model's action, the request, the logged
  torque and the wheel angle on the openpilot turns; then the replayed PID sum's change (p + i + d + f, before the
  PID's clip, the torque map and hysteresis, so not the delivered torque) over the 0.4 s after its own peak, by term,
  as a share of that peak, beside the request's, the setpoint's and the car's over the same 0.4 s."""
  from moonpilot.lab.lateral import action
  shapes = {k: [] for k in ("manual car", "openpilot car", "model action", "request", "torque", "wheel angle")}
  terms = {k: [] for k in ("pid sum", "p", "i", "d", "friction", "request", "setpoint", "car")}
  for name in expand(args.routes, ROUTE_SETS):
    got = _load(name, args.root)
    if got is None:
      continue
    d, la, masks, cfg, z = got
    t, v, level = d["t"], d["v_ego"], MOONPILOT_DAMPING_T.index(cfg.damping_t)
    las, request = _smooth(la), d["curvature"] * v ** 2
    p, i, dd, f, sp = (z[k][level] for k in ("p", "i", "d", "f", "setpoint"))
    command = p + i + dd + f
    series = {"model action": np.abs(action(d) * v ** 2), "request": np.abs(request), "torque": np.abs(d["torque"]), "wheel angle": np.abs(d["angle"])}
    series = {n: _smooth(x) for n, x in series.items()}
    for cls in ("manual", "op"):
      for k in _turns(las, masks[cls]):
        if not 3 <= v[k] < 10 or (s := _shape(las, t, k, 1, masks[cls])) is None:
          continue
        shapes["manual car" if cls == "manual" else "openpilot car"].append(s)
        if cls == "manual":
          continue
        for n, x in series.items():
          w = slice(max(k - 100, 0), k + 101)
          if (sh := _shape(x, t, w.start + int(np.argmax(x[w])), 1, masks[cls])) is not None:
            shapes[n].append(sh)
        sign = np.sign(command[k])
        w = slice(max(k - 150, 0), k + 50)
        top = w.start + int(np.argmax(sign * command[w]))
        j, peak = min(top + 40, len(t) - 1), sign * command[top]
        if peak > 0.2:
          for n, x in (("pid sum", command), ("p", p), ("i", i), ("d", dd), ("friction", f - request), ("request", request), ("setpoint", sp)):
            terms[n].append(sign * (x[j] - x[top]) / peak)
          terms["car"].append((las[j] - las[top]) / peak)
  print("steepest release / peak at 3-10 m/s, median (n): " + ", ".join(f"{n} {np.median(x):.2f} ({len(x)})" for n, x in shapes.items() if x))
  if terms["pid sum"]:
    print(f"openpilot turns ({len(terms['pid sum'])}), 0.4 s after the PID sum's peak, change as a share of it: "
          + ", ".join(f"{n} {np.median(x):+.2f}" for n, x in terms.items()))


def register(sub):
  for name, fn, help_ in (
    ("damping", _damping, "per damping level: D/P in the weave band, command buzz, turn build-up and release"),
    ("weave", _weave, "the wheel spectrum, P's flutter and the rate sensor's lag the damping band was written against"),
    ("release", _release, "where a low-speed turn's release comes from: driver vs openpilot, request, torque and the command's terms"),
  ):
    p = sub.add_parser(name, help=help_)
    p.add_argument("--routes", nargs="+", default=["curve"], help="route names or sets: curve (routes 3f8-410)")
    p.set_defaults(fn=fn)
