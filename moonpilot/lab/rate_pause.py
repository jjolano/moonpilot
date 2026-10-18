"""Replay Toyota steering-rate gating over recorded control frames."""
import numpy as np

from openpilot.tools.lib.logreader import LogReader
from moonpilot.engage import ActuatorGate
from moonpilot.lab import DEFAULT_ROOT, cached, drive_config, expand

ROUTE_SETS = {"rate-pause": ("00000408", "0000040d", "0000040e")}
EXTRACT_VERSION = 3


class _Params:
  def __init__(self, value):
    self.value = value

  def get(self, key, return_default=False):
    return self.value if key == "MoonpilotLateralEngage" else None


def _build(segs):
  rows = []
  latest = None
  latest_allowed = False
  cp = b""
  for m in LogReader(segs, sort_by_time=True):
    kind = m.which()
    if kind == "carParams":
      cp = m.carParams.as_builder().to_bytes()
    elif kind == "pandaStates":
      latest_allowed = m.pandaStates[0].controlsAllowedLateral
    elif kind == "carState":
      cs = m.carState
      latest = (cs.steeringRateDeg, cs.steeringAngleDeg, cs.vEgo, cs.steerFaultTemporary, cs.steerFaultPermanent)
    elif kind == "carControl" and latest is not None:
      rows.append((m.logMonoTime, m.carControl.latActive, latest_allowed, *latest))
  if not rows or not cp:
    raise ValueError("no carControl frames, carState or carParams")
  cols = np.asarray(rows, dtype=float).T
  return {"t": cols[0], "active": cols[1].astype(bool), "lateral_allowed": cols[2].astype(bool), "rate": cols[3], "angle": cols[4], "speed": cols[5],
          "temporary": cols[6].astype(bool), "permanent": cols[7].astype(bool), "cp": np.frombuffer(cp, dtype=np.uint8)}


def extract(route, root=DEFAULT_ROOT):
  return cached(route, "rate-pause", EXTRACT_VERSION, root, _build)


class _PandaState:
  """The one field `ActuatorGate.lateral` reads, as the panda publishes it."""
  controlsAllowedLateral: bool = False


class _CarState:
  steeringRateDeg: float = 0.0


def replay(d, enabled=True):
  from moonpilot.engage import car
  from openpilot.cereal.messaging import log_from_bytes

  CP = log_from_bytes(d["cp"].tobytes(), car.CarParams)
  # The fakes stand in for Params and the cereal readers: duck-typed on the fields the gate reads.
  gate = ActuatorGate(CP, _Params(enabled))  # ty: ignore[invalid-argument-type]  # a Params stand-in
  cs = _CarState()
  panda = [_PandaState()]
  released = np.empty(len(d["active"]), dtype=bool)
  stock = np.empty(len(d["active"]), dtype=bool)
  hot = 0
  for i, rate in enumerate(d["rate"]):
    cs.steeringRateDeg = float(rate)
    panda[0].controlsAllowedLateral = bool(d["lateral_allowed"][i])
    released[i] = gate.lateral(panda, cs)
    if d["active"][i] and abs(rate) >= 100:
      hot += 1
    else:
      hot = 0
    stock[i] = not (d["active"][i] and hot > 17)
    if hot >= 18:
      hot = 0
  return released, stock


def _episodes(mask):
  return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def _rate_pause(args):
  for route in expand(args.routes, ROUTE_SETS):
    d = extract(route, args.root)
    _, params = drive_config(route, args.root)
    as_driven, stock = replay(d, params.get("MoonpilotLateralEngage") == "1")
    current, _ = replay(d)
    active = d["active"]
    hot = np.abs(d["rate"]) >= 100
    # The logs for the pre-current-pause routes already contain the old first-hot pause, so the
    # request upstream would have made without that pause is reconstructed on hot, granted frames.
    request = active | (hot & d["lateral_allowed"])
    turns = (d["speed"] > 0.5) & (np.abs(d["angle"]) > 90)
    current_active = request & current
    stock_active = request & stock
    err = int(np.count_nonzero((active & as_driven) != active))
    lockouts = _episodes(d["temporary"] | d["permanent"])
    print(f"{route}: as-driven lateral-engage={params.get('MoonpilotLateralEngage', 'unset')}, "
          + f"gate-vs-logged latActive mismatches {err}/{len(active)} frames; "
          + f"lockouts {lockouts} (temporary {_episodes(d['temporary'])}, permanent {_episodes(d['permanent'])}); "
          + f"gate releases {int(current_active.sum())}/{int(request.sum())} reconstructed requests, "
          + f"stock 17-frame cuts {int((request & ~stock).sum())}; "
          + f"moving >90° turns assist {current_active[turns].mean() * 100:.1f}% gate / "
          + f"{stock_active[turns].mean() * 100:.1f}% stock ({int(turns.sum())} frames)")


def register(sub):
  p = sub.add_parser("rate-pause", help="replay Toyota rapid-wheel pause")
  p.add_argument("--routes", nargs="+", default=["rate-pause"], help=f"route names or sets: {', '.join(ROUTE_SETS)}")
  p.set_defaults(fn=_rate_pause)

