---
name: moonpilot-log-analysis
description: Locates, pulls and reads moonpilot drive data — device route segments under /data/media/0/realdata and the offline corpus at /home/coder/route-corpus — with LogReader, synchronized camera inspection, and the fork's own logged signals. Use when the user asks to analyse a drive or route, inspect driving video, pull qlogs/rlogs off the device, count or plot messages from a log, compare engaged vs manual stretches and driver inputs, replay logged messages through fork code, check what moonpilotState recorded, or open a route in PlotJuggler/jotpluggler/cabana/replay. Not for live on-device faults (moonpilot-device-debug), deploying (moonpilot-device-deploy) or the edit-build loop (moonpilot-develop).
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

From the repository root, use `.cache/moonpilot-logs/`; `.cache/` is already gitignored.

```bash
mkdir -p .cache/moonpilot-logs && scp moonpilot:/data/media/0/realdata/000003b8--c071d4392d--0/qlog.zst .cache/moonpilot-logs/
rsync -a --include='*/' --include='?log.zst' --exclude='*' \
  moonpilot:/data/media/0/realdata/000003b8--c071d4392d--{0,1,2} .cache/moonpilot-logs/
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

## Visual evidence

For lead following, stops, intersections, lane choice or suspected perception errors, inspect
the camera alongside the signals. Start with logs to shortlist episodes, then pull only those
segments' `qcamera.ts` into the same cache directories. Use `fcamera.hevc` or `ecamera.hevc`
only when qcamera cannot resolve the question; do not pull cabin footage without a need.

- **Align capture time, not segment arithmetic.** In rlog, `qNarrowRoadEncodeIdx` identifies
  qcamera frames; `narrowRoadEncodeIdx` and `wideRoadEncodeIdx` identify the full cameras.
  `segmentNum` selects the file, `segmentId` is its presentation-order frame index, and
  `timestampSof` / `timestampEof` share the logs' monotonic clock. Encoding `logMonoTime`
  is later than capture. Verify media PTS/frame count with `ffprobe`; do not assume
  route time is `segment * 60 + video offset`, especially at boot or across dropped frames.
- **Decode and actually inspect.** Use host `ffmpeg` for qcamera's MPEG-TS/H.264; the repo's
  `openpilot.tools.lib.framereader.FrameReader` handles full HEVC. Make a contact sheet for
  context, then inspect denser frames or the clip around the event. Label frames with
  capture-derived route time and correlate `longActive`, speed, acceleration, pedal inputs,
  lead gap/speed/source and planner/controller commands using timestamp-aligned samples.
- **Separate observation from inference.** Record what the image establishes: queue,
  visible lead motion, signal state, lane, cut-in or clear road. Radar track IDs are not
  physical identities. A contact sheet alone cannot prove a false lead, exact distance,
  brake pressure or the driver's intent. State missing video, resolution or timing limits.

After deriving the start offset within the selected file, a 12-second context sheet:

```bash
ffmpeg -hide_banner -loglevel error -ss <start-offset-seconds> \
  -i .cache/moonpilot-logs/<route>--<seg>/qcamera.ts -t 12 -vf 'fps=1,scale=640:-1,tile=4x3' \
  -frames:v 1 -y .cache/moonpilot-logs/visual.png
```

Open the resulting image; generating it is not visual verification. If ffmpeg is unavailable,
report that limit rather than presenting log-only scene assumptions as observed facts.

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
- **Split actuator authority from driver input before aggregating.** For longitudinal work,
  use `carControl.longActive`, not just `enabled`: lateral-only engagement can coexist with
  manual speed control. Separate gas overrides and stock ACC from manual pedal driving.
  Compare matched speed/gap/scene episodes; a mostly-manual route mean is not a controller
  baseline. Pedal booleans show switch timing, not pressure, and can lag analog input.
  Decode car-specific CAN with the existing opendbc parser when modulation matters.
  Keep uncalibrated pressure in raw counts; an unpopulated pedal field is not zero input.

## Interactive tools

- `tools/op.sh juggle <route>` — PlotJuggler, multi-signal time-series plots.
- `openpilot/tools/jotpluggler/jotpluggler [--layout <l>] [--demo] [route]` — newer in-repo
  plotter, `--stream --address <ip>` for live device data; no `op.sh` subcommand.
- `tools/op.sh cabana <route>` — CAN bus / DBC signal inspection.
- `tools/op.sh replay <route>` — republish a log onto msgq so real processes consume it. One
  publisher per service: stop the process owning it first.
