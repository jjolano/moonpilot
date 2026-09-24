# Phase 0 — true-SLAM research spike

Numbers from route-corpus `000001b7--1e83052604`, segments 1–10 (≈4.8 km of `cameraOdometry` +
`liveLocationKalmanDEPRECATED`), measured offline with the PC venv. Truth is LLK
`positionGeodetic` (corpus has no `gpsLocationExternal`; `gpsLocation` is qcom at 1 Hz).
VO pose time is `timestampEof − 0.1 s` (already nanoseconds). Extrinsics applied when present.

## Front-end options

| Option | Result |
| --- | --- |
| **A** — posenet only (`cameraOdometry`) | Viable for short-horizon pose and for the existing speed correction. Dead-reckoned alone it does not hold a map-scale trajectory. |
| **B** — own feature tracks on narrow frames (cv2) | Not measured here (no cv2 on PC; aarch64 wheel `opencv-python-headless==4.12.0.88` resolves). Required if landmark mapping is wanted. |
| **C** — model intermediates | **No spatial feature maps.** Parser exposes only structured heads (`pose`, `road_transform`, `wide_from_device_euler`, lane lines, road edges, leads, plan, desire, meta) plus a 512-d `hidden_state`. `modelV2.rawPredictions` is empty unless `SEND_RAW_PRED`. Not usable for landmark SLAM. |

**Pick: A for Phase 1 ego pose; C-shaped geometry (`modelV2.roadEdges` / `laneLines`) for Phase 2 corridor occupancy; B only if Phase 3+ needs landmarks.**

## VO dead-reckoning vs LLK

Per-segment: integrate `trans`/`rot` in the calib frame (rotated by extrinsics), Procrustes-align
(yaw + xy + scale) to LLK ENU over the overlap after a 2 s settle, report RMSE.

| metric | value |
| --- | --- |
| segments | 10 (total ≈ 4793 m ref path) |
| RMSE | med **3.86 m**, mean 11.4, max 54.5 (seg 7 scale collapse 0.62) |
| drift % of path | med **0.63 %**, mean 3.40, max 17.97 |
| path length VO/ref | **0.981** (4870 / 4966) — scale ≈ 1 |
| chained 5 segs, no mid-route re-align | dist 2627 m → RMSE **341 m (13 %)**, final 198 m |

Reading: raw VO is excellent over a ~60 s window (sub-meter to a few metres on clean segs) and
falls apart over minutes without a correction source. That is exactly what the rolling-window
correction already papers over for *speed*, and why a map or GPS/LLK gate is required for *pose*.

### Forward speed (same route, moving > 3 m/s)

| comparison | RMSE | mean | median scale |
| --- | --- | --- | --- |
| \|VO trans_x\| vs `carState.vEgo` | 0.17–0.85 m/s | ±0.35 | **0.99–1.02** |
| \|VO trans_x\| vs LLK ground speed | 0.26–0.92 m/s | — | same |

Confirms `moonpilot/slam.py`’s stage-1 claim: the signal is a scale/lag disagreement of order
0.1–0.5 m/s, not a different speed scale.

## Phase 2 occupancy input

`modelV2.roadEdges`: 2 edges × 33 XYZT points, horizon to **x = 192 m**, `y` up to ~55 m,
`roadEdgeStds` ~1.2 m at the far end. Lane lines same shape. No new frontend needed for a
corridor / occupancy strip along the planned path — model geometry alone is dense enough.
Persistent free-space maps across drives were dropped from scope; this is the on-road tier only.

## CPU budget

`openpilot/selfdrive/test/test_onroad.py` allocates **342 % of `MAX_TOTAL_CPU = 350 %`** →
**8 % headroom** for any new process. Corpus `procLog` on this route measured ~425 % system-wide
(includes non-budget procs and a sunnypilot build) — do not treat that as the budget.

Implication: a Phase 1/2 worker must either (a) be tiny (numpy pose math, piggyback an existing
process or a low-duty `PythonProcess`), or (b) free budget elsewhere. Feature tracking (option B)
at 20 Hz on narrow frames is unlikely to fit in 8 % without measurement on-device.

## Packaging

- No venv on device; third-party deps only via `moonpilot/deps.py` + `deps.lock` (aarch64 pins).
- `opencv-python-headless==4.12.0.88` resolves for `aarch64-unknown-linux-gnu` / py3.12.
- g2o / ceres Python bindings: not probed; avoid unless option B + nonlinear optimization is chosen.
- PC spike ran in `/tmp/moonpilot-spike/drift.py` against the project venv; script is not checked in.

## Exit (Phase 0)

1. Front-end: **A** now, **B** only if landmark mapping returns to scope.
2. Phase 1: better ego pose = fuse A with wheel + LLK/GPS gates over a longer window; keep
   observation-only until RMSE-vs-truth improves or holds.
3. Phase 2 occupancy: **`modelV2.roadEdges`/`laneLines` only** — no new vision, no new service
   until a consumer exists.
4. Next gate: a replay that scores chained-pose RMSE against LLK the way `test_slam.py` scores
   speed, before any control-path consumer.
