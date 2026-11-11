# The rolling-window ego correction
Implementation: `moonpilot/slam.py` publishes `egoCorrection`; replay measurements follow.

- **Align the wheel-speed prior to odometry exposure time.** Latest-with-latest pairing adds 0.15–0.35 m/s error while braking; publish-time pairing retains 30 % of it. Publish lag measured 30.5 ms median (p95 33.5, max 200) over 13k frames; publish-time alignment correlated −0.85 with `aEgo` and under-corrected by +0.019 m/s. On `0000043f` / `0000043e`, latency was 32.9 / 33.2 ms median (p95 36.0 / 34.2), and publish-time alignment differed by −0.984 / −0.982 correlation and +0.059 / +0.038 m/s while braking. `PriorChannel` implements exposure-time interpolation.
- **Parked-route replay.** Over 988 s, positive corrections exceeded 0.1 m/s on 0.05 % of frames (worst +0.578). A 14-segment route measured 0.80 % (32/4,014, worst +0.114), with no excursion during engaged longitudinal control; `0000043f` / `0000043e` measured 1.19 % / 0.26 % (worst +0.148 / +0.111), again with no engaged-longitudinal frame. Closed-loop, a +0.15 m/s correction beside a lead at 0.12 m/s cleared should-stop and changed the delivered command from `CP.stopAccel` to −0.02 m/s², after which `cruiseControl.resume` requested pull-away. The 0.1 speed-fraction bound leaves 2 m/s headroom at 20 m/s against a moving p99 of 0.19 m/s.
`carState.valid` is not a gate: `card.py` derives it from `CS.canValid`, so a CAN-health hiccup would invalidate the correction. The shared valid/alive check is in `moonpilot/state.py`.

`ego_speed_correction()` applies the accepted correction in the planner. The cached routes lack `gpsLocationExternal`; the results available here use qcom `gpsLocation` and do not verify the external-truth criterion.
On `0000043f` / `0000043e`, corrected speed RMSE is 0.553 / 0.391 m/s versus 0.517 / 0.339 raw; `egoCorrection.valid` duty is 99.96 % (22,028/22,037) and 99.84 % (5,624/5,633), and the correction reproduces the logged value to 0.019 / 0.018 m/s RMSE.

An on-device replay measured `egoCorrection.valid` at 100.00 % (14,955/14,955) against `gpsLocationExternal`; that route and truth service are not in this checkout. The cached-route results above use qcom `gpsLocation` and do not verify that criterion.

## The ego pose

Pose behavior and coordinate conventions are documented in `moonpilot/pose.py`.

