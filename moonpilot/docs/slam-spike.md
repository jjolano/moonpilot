# SLAM spike measurements

These offline measurements use route-corpus `000001b7--1e83052604`, segments 1–10 (about 4.8 km
of `cameraOdometry` and `liveLocationKalmanDEPRECATED`). Truth is LLK `positionGeodetic`; the
corpus has no `gpsLocationExternal`, and `gpsLocation` is qcom at 1 Hz. VO pose time is
`timestampEof - 0.1 s`; extrinsics are applied when present.

## VO dead-reckoning vs LLK

Each segment integrates `trans`/`rot` in the calibration frame, rotates by extrinsics, then
Procrustes-aligns yaw, XY, and scale to LLK ENU after a 2 s settle.

| metric | value |
| --- | --- |
| segments | 10 (about 4793 m reference path) |
| RMSE | median **3.86 m**, mean 11.4, max 54.5 (segment 7 scale collapse: 0.62) |
| drift as path fraction | median **0.63 %**, mean 3.40, max 17.97 |
| VO / reference path length | **0.981** (4870 / 4966) |
| five chained segments, no mid-route realignment | 2627 m; RMSE **341 m (13 %)**; final error 198 m |

Raw VO is useful over short windows but drifts over minutes without a correction source.

## Forward speed

For samples moving faster than 3 m/s:

| comparison | RMSE | mean | median scale |
| --- | --- | --- | --- |
| `abs(trans_x)` vs `carState.vEgo` | 0.17–0.85 m/s | ±0.35 | **0.99–1.02** |
| `abs(trans_x)` vs LLK ground speed | 0.26–0.92 m/s | — | same |

This supports the `moonpilot/slam.py` scale/lag correction; it does not establish long-term pose
accuracy.

## Model geometry

`modelV2.roadEdges` has 2 edges × 33 XYZT points, reaching **x = 192 m** and `y` about 55 m;
`roadEdgeStds` is about 1.2 m at the far end. Lane lines have the same shape. This geometry
supports on-road corridor occupancy without a separate vision frontend. Persistent free-space maps
across drives remain out of scope.
