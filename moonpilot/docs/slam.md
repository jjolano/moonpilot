# The rolling-window ego correction
`moonpilot/slam.py` is the fork's answer to a single question: is the speed the planner plans from
right? It blends two sources the car already has — the prior, which is `carState`'s wheel speed,
against the observation, which is `cameraOdometry`'s forward `trans` and what the model saw move —
over a 5 s rotating window, and publishes the disagreement at the window's end as
`moonpilotState.egoCorrection.dVel`. Stage 1, on purpose: no new vision frontend, no new service,
numpy only, so it runs on the device's own CPU. Seven things about it are not obvious.

- **One process publishes `moonpilotState`, and it is `leadd`.** msgq allows exactly one publisher
  per service: a second `PubMaster` on the same service connects, overwrites the write uid, and the
  first publisher's next send fails with `MultiplePublishersError` (msgq's own
  `test_multiple_publishers_exception`, and `Killing old publisher` on stderr) — after which the
  manager never restarts that process, because `ensure_running` reaps only on the `should_run`-false
  path. So the correction rides `leadd`, and the gate is read per frame in its loop; a second
  process for it would silently take the leads down with it.
- **Calibration is applied at ingest.** `cameraOdometry`'s `trans` is in the camera's calibration frame, which differs from the device frame by the mounting angle `extrinsicsCalibration.rpyCalib` measures — upstream `locationd` applies exactly that rotation (`device_from_calib`, `rot_from_euler`, `rotate_std`). `leadd` ingests the newest trusted calibration and applies it to `trans` and its std, with identity whenever the message is absent, invalid, malformed or outside the sanity bound, so a device without a trusted calibration behaves exactly as before. The device frame is still treated as the car frame; a calibration shift mid-window affects only newly ingested samples.


- **The prior is read at the odometry's pose time, which is the frame's exposure and not the
  message's.** `cameraOdometry` describes the pose as of `timestampEof - CAM_ODO_POSE_DELAY` (which
  is why `locationd` computes its observation time that way, `locationd.py:162`), so `leadd` stamps
  each node from `timestampEof`. Pairing latest-with-latest instead puts an `a * 0.1 s` error on
  every interval — 0.15–0.35 m/s while braking, the same order as the signal this measures and
  correlated with exactly the approach regime the planner's gap terms live in. Reading the prior at
  `logMonoTime` instead is 30 % of that same error rather than none of it: the publish time trails
  the exposure by modeld's inference and send latency, a measured 30.5 ms median (p95 33.5, max
  200) over 13k corpus frames, and replaying the corpus through both arms puts the difference at
  −0.85 correlation with `aEgo` and +0.019 m/s of under-correction while braking, 23 % of the
  correction's own size there. `PriorChannel` samples the wheel speed with its own message times and
  interpolates it onto that time; a frame the samples do not span is skipped, never extrapolated.
- **The gate is live, not restart-gated.** `leadd` feeds the window whether or not `MoonpilotSlam`
  is on, so turning it on is immediate instead of five seconds of refill. That is also why the
  description does not claim a restart, unlike the fork's controller toggles.
- **The correction is `smoothed minus the raw prior`, and that sign is load-bearing.** A consumer
  adds it to the wheel speed it was computed against, so a wheel speed reading low has to come back
  positive — at 19.5 against 20.0 the correction is +0.5. The opposite convention would add the
  wheel's own scale error back rather than remove it (truth 20, wheel 19.5, VO 20.0 → 19.0). The
  plan that produced this feature pinned the inverted sign, which the replay RMSE check is the
  arbiter for; `moonpilot/tests/test_slam.py` pins this one.
- **What the window buys is lag, not accumulation.** With independent per-tick errors, inverse
  variance weights would be constant and the window length would not matter. Correlated error is the
  error this exists for — a wheel-speed scale error, where 2 % at 20 m/s is 0.4 m/s and 1.2 m of gap
  over a 3 s approach — and the window is how long an estimate of it is allowed to lag.
- **The bound is a fraction of the speed being corrected, so a standstill gets nothing.** A scale
  error is proportional to the speed it is measured on: 2 % of 20 m/s is 0.4 m/s, 2 % of 0 is 0. At
  rest the prior is exactly 0 and the inverse-variance weights hand ~90 % of the window's answer to
  posenet, which is least trustworthy there — upstream stops checking its health below 5 m/s at all
  (`locationd.py:242`) — and the result lands on the same `v_ego` both standstill guards read:
  `MOONPILOT_SHOULD_STOP_SPEED` in the planner's trailing gate, and upstream's 0.3 in `should_stop`.
  Replaying 988 s of genuinely parked corpus through the real window put a positive correction past
  0.1 m/s on 0.05 % of frames (worst +0.578), while this 14-segment route measures 0.80 %
  (32 of 4,014 parked frames, worst +0.114) — the rate is route-dependent, and 0 of the 32
  excursions coincide with engaged longitudinal control, so latent clamp behavior on this drive
  rather than realized creep. Measured closed-loop, a +0.15 correction beside a
  lead reporting 0.12 m/s cleared should-stop and took the delivered command from `CP.stopAccel` to
  −0.02 — after which `controlsd`'s own `cruiseControl.resume` asks the car's ACC to pull away.
  `MOONPILOT_SLAM_MAX_SCALE_ERR` (0.1 of `v_ego`) is zero there by construction on any route and inert where the
  feature earns its keep: 2 m/s of headroom at 20 m/s against a measured moving p99 of 0.19.

Two more things the wiring depends on. `leadd` fills the message's own `valid` from
`all_checks(['modelV2', 'radarState'])` rather than from every service it subscribes to, so a
`cameraOdometry` outage cannot invalidate the leads and take the lead line off both UIs with it; the
correction carries its own `valid`, and the reader checks that *plus* `sm.alive`, which is what
catches a daemon that died with a fresh-looking message left in the socket. And `carState`'s `valid`
is deliberately *not* a gate: `card.py` sets it from `CS.canValid`, a CAN-health flag, so a bus
hiccup would silently kill the correction — the prior's freshness comes from the pose-time read,
which is the stricter test anyway.

The consumer is `MoonpilotLongitudinalPlanner.update`, one line: `v_ego` gains the correction before
anything else reads it, so the spacing regulator, the time-to-collision approach term, the cruise
term and the published rollout all plan from the corrected speed — not one candidate, since every
one of them starts from `v_ego`. Trust is scaled by `1 / (1 + corrStd)` and never amplified, `a_ego` is
untouched, and absent, invalid, stale or dead-publisher all give zero, which is exactly the planner
this fork had before the correction existed. Staleness is honest: `age` is the window end to now and
therefore includes `MOONPILOT_SLAM_POSE_DELAY`, so the consumer's budget is `MAX_AGE +
POSE_DELAY` rather than spending a third of a 0.3 s allowance on the sensor's own lag. The numbers
are in `slam.py`'s header block — including which of them are mirrors of `locationd`'s own std floor
and odometry multiplier, which `test_slam.py` pins against that file. That test also pins the sign
convention, the clamps, the staleness contract, the ingest's calibration and pose-time stamp and the
pass-through, and then drives upstream's plant — the 20 m/s follow-then-stop maneuver, with the
correction and without it — to hold that a correction of half a meter per second moves the plan
without reshaping it. The on-device proof the plan calls for is a replay: logged `carState` +
`cameraOdometry` through the window, against `gpsLocationExternal` as truth, RMSE down or equal, and
`egoCorrection.valid` duty ≥ 95 % onroad — met on this route at 100.00 % (14,955/14,955): PASS.

## The ego pose

The same toggle and the same publisher also fill `moonpilotState.egoPose` from `moonpilot/pose.py`:
observation-only, dead-reckon + GPS gate, valid from the first accepted fix (the origin) onward.
x/y are meters east/north of that fix, yaw is rad CCW from east, `monoTime` is the last odom pose
time stamped the same way as the correction's window end. Off (`MoonpilotSlam` unset or false)
leaves `valid` false — the pass-through case. Its first control-path consumer is the pose-stitched
corridor below: without a valid pose that consumer is the single-frame form it had before.

## Corridor occupancy (Phase 2)

`moonpilot/corridor.py` is the on-road free-space strip from `modelV2.roadEdges` alone — pure
numpy for the geometry, no publisher, no toggle of its own — so a planner seam or a replay can call
it the same way `lead_in_path` is called. At each sample along a path it returns the free lateral
bounds (ordered by y, never by edge index), the free width, and an in-corridor flag with a small
margin; a sample the edges do not span is NaN, not free. Lane lines are the same shape and remain
available if a consumer needs them; a persistent map across drives is out of scope (`slam-spike.md`).

Two consumers ship. `MoonpilotSqueeze` (`squeeze_accel`) is the cruise-slot brake when free width
pinches under about two car widths over the next 40 m — see `moonpilot/docs/longitudinal.md`.
`MoonpilotPathOutside` (`path_outside_alert`) is selfdrived's banner when enough of the near model
path leaves that corridor while `carControl.latActive` is true — see `moonpilot/docs/longitudinal.md`
and the seam table in `AGENTS.md`.

## Pose-stitched free space (Phase 3)

The model repaints `roadEdges` every frame; a strip solid for half a second and missing for one
frame should not release squeeze or clear the path-outside banner on that frame. `RollingCorridor`
keeps a short history (1.5 s / 40 frames) of strips, each stamped with the `egoPose` the frame was
captured in, and warps them into the caller's current pose by relative SE2 before intersecting the
free regions (max of lefts, min of rights — the narrowest strip any frame saw). Unknown samples stay
unknown: intersection only tightens where both frames had an answer. A `pose is None` or an empty
history is all-NaN / `ACCEL_MAX`, and an invalid pose clears the history so a rebase cannot mix two
origins. The planner's squeeze path is the first caller: with a valid `moonpilotState.egoPose` it
pushes each frame and queries the fused strip; without a pose (SLAM off, first fix, dead publisher)
it clears and falls back to the single-frame `squeeze_accel` — exactly the planner this fork had
before the history existed.

