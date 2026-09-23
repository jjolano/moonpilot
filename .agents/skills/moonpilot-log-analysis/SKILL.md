---
name: moonpilot-log-analysis
description: Locates, pulls and reads moonpilot drive data — device route segments under /data/media/0/realdata and the offline corpus at /home/coder/route-corpus — with LogReader, and names the fork's own logged signals. Use when the user asks to analyse a drive or route, pull qlogs/rlogs off the device, count or plot messages from a log, compare engaged vs manual stretches, replay logged messages through fork code, check what moonpilotState recorded, or open a route in PlotJuggler/jotpluggler/cabana/replay. Not for live on-device faults (moonpilot-device-debug), deploying (moonpilot-device-deploy) or the edit-build loop (moonpilot-develop).
---

## Where the data is

One directory per segment under `/data/media/0/realdata/<route>--<seg>/`:

```bash
ssh moonpilot 'ls /data/media/0/realdata/000003b8--c071d4392d--0; du -sh /data/media/0/realdata'
```

Observed: 419 segment dirs across 24 routes, 63G total, `/data` 90% full. Each segment holds
`rlog.zst qlog.zst fcamera.hevc ecamera.hevc qcamera.ts` — ~156M, of which the logs are ~10M.
A 10-segment route is ~1.5G on disk but ~100M of log.

Offline corpus: `/home/coder/route-corpus/` (266 segments, 2.6G), pulled 2026-06-12 off this
device. Read its `README.md`; three things decide whether it answers your question:

- It is almost entirely **manual** driving — ~2.4 min engaged in the whole corpus. Any
  engaged-control baseline you think you see there is not backed by data.
- Its `baselines/` were baked by the *other* fork's `openpilot/tools/drive_lab`, run from a
  sunnypilot worktree. moonpilot has no `tools/drive_lab`; those reports can be read, not re-run.
- The logs predate moonpilot's schema. Tag `@107` in them carries sunnypilot's struct, so
  moonpilot's cereal names it `moonpilotState` and then raises
  `KjException: Schema mismatch` on access. Skip that field on corpus logs.

## Pull a segment

```bash
mkdir -p /tmp/mp-logs && scp moonpilot:/data/media/0/realdata/000003b8--c071d4392d--0/qlog.zst /tmp/mp-logs/
rsync -a --include='*/' --include='?log.zst' --exclude='*' \
  moonpilot:/data/media/0/realdata/000003b8--c071d4392d--{0,1,2} /tmp/mp-logs/
```

No trailing slash on the sources, or all segments flatten into one directory. `rsync`, `tar`
and `zstd` are on the device. Pull logs only unless you need video — cameras are 15x the bytes.
`system/loggerd/deleter.py` reclaims space only under `Paths.log_root()`
(`/data/media/0/realdata/` on device), so copies parked elsewhere on `/data` are never cleaned
up and eat the 9G free.

## Reading a log

`openpilot/tools/lib/logreader.py`. `LogReader(path_or_list)` iterates capnp events; each has
`.which()` and `.logMonoTime`. `ReadMode.RLOG/QLOG/AUTO` picks which file a route identifier
resolves to; a direct file path needs no mode.

qlog is not a smaller rlog, it is a decimated one: `cereal/services.py` gives each service a
decimation and `openpilot/system/loggerd/loggerd.cc:292` keeps a message only when
`service.freq != -1 && service.counter++ % service.freq == 0`.
So `decimation: None` means **never in qlog at all** — that is `modelV2`. `can` is 2053
(3 msgs per segment), `carState`/`controlsState` 10, `moonpilotState` 5. qlog answers
engagement windows and event timelines; model output, CAN and controller timing need rlog.

Corpus route names are not comma route IDs. `/home/coder/route-corpus/run_tool.py` monkeypatches
`LogReader._parse_identifier` to glob `logs/<route>--*/{rlog,qlog}.zst` — reuse the idea.

## Worked example

```bash
cd /home/coder/projects/moonpilot && .venv/bin/python -c "
from collections import Counter
from openpilot.tools.lib.logreader import LogReader
c, ts = Counter(), []
for m in LogReader('/home/coder/route-corpus/logs/000001b4--c6ab5af113--3/rlog.zst'):
  c[m.which()] += 1; ts.append(m.logMonoTime)
print({k: c[k] for k in ('carState','modelV2','radarState','can')})
print(f'span {(max(ts)-min(ts))/1e9:.1f}s, {sum(c.values())} msgs, {len(c)} services')"
```

Prints `{'carState': 5809, 'modelV2': 1200, 'radarState': 1200, 'can': 6000}` and
`span 241.4s, 105376 msgs, 59 services`. The repo `.venv` has capnp; the system `python3` does not.

## Fork signals

- `moonpilotState` — the fork's own published service, 20Hz, schema in `cereal/custom.capnp`
  (`MoonpilotState`: `leads`, `egoCorrection`). Its field comments carry units and sign
  conventions; AGENTS.md **Features** says what they are for.
- `pandaStates[].controlsAllowedLateral` (`cereal/log.capnp:573`) and the `lateralEngageOff`
  onroad event (`log.capnp:120`) — the half-engagement state, AGENTS.md
  `moonpilot/docs/engage.md`. The seam table in AGENTS.md maps each field to the file that writes it.

## Patterns worth keeping

- **Replay logged messages on their own `logMonoTime`.** Fork code in `moonpilot/` ages its
  inputs (`slam.py`, `longitudinal.py`, `leadd.py` all read stamps); re-stamping payloads with
  synthetic evenly-spaced times deletes the staleness, jitter and rejection paths you came for.
- **Split engaged from manual before aggregating.** `carControl.enabled` /
  `selfdriveState.enabled` per message; a route mean over mostly-manual driving describes the
  driver, not the controller. On this corpus that is nearly always the case.

## Interactive tools

- `tools/op.sh juggle <route>` — PlotJuggler, multi-signal time-series plots.
- `openpilot/tools/jotpluggler/jotpluggler [--layout <l>] [--demo] [route]` — newer in-repo
  plotter, `--stream --address <ip>` for live device data; no `op.sh` subcommand.
- `tools/op.sh cabana <route>` — CAN bus / DBC signal inspection.
- `tools/op.sh replay <route>` — republish a log onto msgq so real processes consume it. One
  publisher per service: stop the process owning it first.
