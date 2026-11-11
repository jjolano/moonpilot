"""Route names and command-line options shared by offline lab replays."""
import argparse
import glob
import os
from collections.abc import Sequence

DEFAULT_ROOT = ".cache/moonpilot-logs"

CURVE_ROUTES = tuple(f"0000{i:04x}" for i in range(0x3F8, 0x411))
LURCH_EVENT = ("0000046b--fbae8b11b5", 667.85, 671.85)
CALIBRATION_ROUTE = "00000451"
STOP_SHAPE_ROUTE = "00000468--34a8e259a0"

_LONGITUDINAL = ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e",
                "0000043f")

_LONGITUDINAL_INPATH = (*_LONGITUDINAL, "0000044d")
_MANUAL_APPROACH = ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000042b",
                    "0000042c", "0000043e", "0000043f", "0000044d")
_DECEL_ROUTES = (*_MANUAL_APPROACH, "0000044e", "0000044f", "00000450", "00000451")
_LONGITUDINAL_ENGAGED = ("000003f8", "000003fe", "00000400", "00000401", "00000408", "0000040d", "0000040e", "0000040f", "00000410", "00000414",
                         "00000417", "0000041d", "00000424", "00000429", "0000042a", "0000042b", "00000439", "0000043a", "0000043e", "0000043f",
                         "0000044d", "00000450", "00000451", "0000045a", "0000045d", "0000045e")

ROUTE_SETS: dict[str, tuple[str, ...]] = {
  "longitudinal": _LONGITUDINAL,
  "longitudinal-inpath": _LONGITUDINAL_INPATH,
  "lead-approach": ("000003f8", "00000400", "00000401", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  "manual-approach": _MANUAL_APPROACH,
  "decel": _DECEL_ROUTES,
  "longitudinal-engaged": _LONGITUDINAL_ENGAGED,
  "longitudinal-plus-468": (*_LONGITUDINAL_ENGAGED, "00000468"),
  "stop-cues": (*_DECEL_ROUTES, *_LONGITUDINAL_ENGAGED, "00000468"),
  # The estimator replays' own engaged drives, which are a different set from the planner's.
  "estimator-engaged": ("000003f8", "000003fe", "00000400", "00000408", "0000040d", "0000040e", "00000410", "00000414", "0000043e", "0000043f"),
  "model-braking-engaged": _LONGITUDINAL_INPATH,
  "manual": _MANUAL_APPROACH,
  "curve": CURVE_ROUTES,
  "all": (),
  "smooth": ("00000413--e204529ce6", "00000414--32736d038c", "00000417--e0a9135c4e", "00000418--c554e5658b"),
  "hysteresis": ("0000040d--b31ecdf74b", "0000040e--20803b67bd"),
  "lanes": (*CURVE_ROUTES, "00000413", "00000414", "00000417", "00000418", "0000043e", "0000043f", "0000044d", "0000044e", "0000044f"),
  "stick": (*CURVE_ROUTES, "00000413", "00000414", "00000417", "00000418", "0000043e", "0000043f", "0000044d", LURCH_EVENT[0]),
  "turns": ("000003f8", "00000400", "00000410", "00000414--32736d038c", "0000043e--d12558e19c", "0000043f--7b506ff942"),
  "claims": tuple(f"{i:08x}" for i in range(0x3F3, 0x40B)),
  "closed-loop": ("00000408", "0000040e", "0000040d"),
  "rate-pause": ("00000408", "0000040d", "0000040e"),
  "latest": ("0000043f", "0000043e"),
  "recenter": (*CURVE_ROUTES, "00000413", "00000414", "00000417", "00000418", "0000043e", "0000043f", "0000044d", "0000044e", "0000044f", LURCH_EVENT[0]),
}


def expand(names: Sequence[str], root: str) -> list[str]:
  """Expand route-set names against this invocation's root, preserving first-seen order."""
  result = []
  for name in names:
    routes = ROUTE_SETS.get(name, (name,))
    if name == "all":
      routes = sorted({
        os.path.basename(os.path.dirname(path)).split("--", 1)[0]
        for path in glob.glob(os.path.join(root, "*--*", "rlog.zst"))
      })
    elif name == "claims":
      routes = tuple(
        route for route in routes if glob.glob(os.path.join(root, f"{route}--*", "rlog.zst"))
      )
    result.extend(routes)
  return list(dict.fromkeys(result))


def add_routes_arg(parser: argparse.ArgumentParser, default):
  parser.add_argument(
    "--routes", nargs="*" if default == [] else "+", default=default,
    help=f"route names or sets: {', '.join(ROUTE_SETS)}",
  )


def add_calibration_arg(parser: argparse.ArgumentParser):
  parser.add_argument(
    "--calibration", default=CALIBRATION_ROUTE,
    help="drive whose recorded lag and radar latency seed the flights",
  )
