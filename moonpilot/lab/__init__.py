"""moonpilot's offline lab: replay recorded drives through fork code and measure the result.

The replay numbers in moonpilot/docs and AGENTS.md come from runs like these. Each submodule here owns one
subsystem's extraction, replay and metrics and registers its subcommands; `python -m moonpilot.lab -h` lists them.

Inputs are rlogs under --root, one `<route>--<seg>/rlog.zst` per segment, pulled the way the log-analysis skill
does. Extracts in .cache/moonpilot-extract are a derived cache holding only what a replay reads: keep the rlogs,
they are what a new question re-extracts from.

Every replay runs the code as it is now. A replay that has a logged counterpart first runs the drive's own
configuration — the `Moonpilot*` params and commit its initData recorded — and prints the replay-vs-logged error
beside the metric, then varies one setting from that same configuration. A large error means the drive ran code
this replay does not reconstruct, not that the metric is wrong; read it before the metric.

Command-path evidence only: what is recorded (the wheel, the car's speed) is held as recorded, so a replay says
what the fork would have asked for, never how the car would have moved. Dev tooling; nothing on a device imports it.
"""
import argparse
import glob
import importlib
import os
import pkgutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np

from openpilot.common.realtime import DT_CTRL
from openpilot.tools.lib.logreader import LogReader

DEFAULT_ROOT = ".cache/moonpilot-logs"
DEFAULT_CACHE = ".cache/moonpilot-extract"


def segments(route: str, root: str = DEFAULT_ROOT) -> list[str]:
  """The route's rlogs in segment order."""
  paths = glob.glob(os.path.join(root, f"{route}--*", "rlog.zst"))
  if not paths:
    raise FileNotFoundError(f"no {route}--*/rlog.zst under {root}")
  return sorted(paths, key=lambda p: int(os.path.basename(os.path.dirname(p)).rsplit("--", 1)[1]))


def cached(route: str, name: str, version: int, root: str, build) -> dict[str, np.ndarray]:
  """`build(segment_paths)`'s arrays, cached per route, extract name and version; rebuilt when the segments on disk change."""
  segs = segments(route, root)
  names = [os.path.basename(os.path.dirname(s)) for s in segs]
  path = os.path.join(DEFAULT_CACHE, f"{route}.{name}.v{version}.npz")
  if os.path.exists(path):
    d = dict(np.load(path))
    if d["segments"].tolist() == names:
      return d
  d = build(segs)
  d["segments"] = np.array(names)
  os.makedirs(DEFAULT_CACHE, exist_ok=True)
  np.savez(path, **d)
  return d


@cache
def drive_config(route: str, root: str = DEFAULT_ROOT) -> tuple[str, dict[str, str]]:
  """(git commit, Moonpilot* params) as the drive's initData recorded them."""
  for m in LogReader(segments(route, root)[0]):
    if m.which() == "initData":
      params = {e.key: e.value.decode(errors="replace") for e in m.initData.params.entries if e.key.startswith("Moonpilot")}
      return m.initData.gitCommit, params
  raise ValueError(f"{route}: no initData")


def param_index(params: dict[str, str], key: str) -> int:
  """A choice or toggle param as the drive had it; absent or unreadable reads as 0, which is every row's off."""
  try:
    return int(params.get(key, "0"))
  except ValueError:
    return 0


@cache
def predates(commit: str, ref: str) -> bool:
  """True when `ref` is not in the drive's history, so the drive ran code from before that change."""
  code = subprocess.run(["git", "merge-base", "--is-ancestor", ref, commit], capture_output=True).returncode
  if code not in (0, 1):
    raise ValueError(f"commit {commit} is not in this checkout; fetch it to replay the drive's configuration")
  return code == 1


def runs(mask: np.ndarray) -> list[slice]:
  """The contiguous True stretches of a boolean mask."""
  edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return [slice(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def reversals_per_s(x: np.ndarray, mask: np.ndarray, dt: float = DT_CTRL) -> float:
  """Direction changes of a command per second of masked time. A frame that holds still continues the last direction,
  so a held (hysteresis) output only counts the moves that reach the car."""
  count = 0
  for run in runs(mask):
    step = np.sign(np.diff(x[run]))
    step = step[step != 0]
    count += int(np.count_nonzero(step[1:] != step[:-1]))
  return count / max(mask.sum() * dt, dt)


def lag_s(ref: np.ndarray, x: np.ndarray, mask: np.ndarray, dt: float = DT_CTRL, max_lag: float = 0.5) -> float:
  """How far `x` trails `ref`: the shift minimizing their mean squared difference over frames masked at both ends,
  refined to below a frame by a parabola through the minimum."""
  cost = []
  for s in range(int(round(max_lag / dt)) + 1):
    both = mask[s:] & mask[:len(mask) - s]
    cost.append(np.mean((x[s:][both] - ref[:len(ref) - s][both]) ** 2))
  i = int(np.argmin(cost))
  if 0 < i < len(cost) - 1:
    a, b, c = cost[i - 1], cost[i], cost[i + 1]
    i += 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c > 0 else 0.0
  return i * dt


def expand(names: list[str], sets: Mapping[str, Sequence[str]]) -> list[str]:
  """Route names, with any set name replaced by its routes."""
  return [r for n in names for r in sets.get(n, (n,))]


def main(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--root", default=DEFAULT_ROOT, help="directory of <route>--<seg>/rlog.zst")
  sub = parser.add_subparsers(required=True)
  # Each submodule with a `register(sub)` adds its own subcommands, each setting `fn(args)`. One that fails to import
  # costs its own subcommands, named on stderr, not the whole lab.
  for info in pkgutil.iter_modules(__path__):
    if info.name.startswith("_"):
      continue
    try:
      module = importlib.import_module(f"{__name__}.{info.name}")
      if hasattr(module, "register"):
        module.register(sub)
    except Exception as e:
      print(f"moonpilot.lab: skipping {info.name}: {e!r}", file=sys.stderr)
  args = parser.parse_args(argv)
  args.fn(args)
