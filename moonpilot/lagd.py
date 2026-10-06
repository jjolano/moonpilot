"""The fork's lateral delay estimator: upstream `lagd`'s machinery, with a memory that can follow the car.

It runs under upstream's process name and publishes upstream's `lateralDelay`, so every consumer —
modeld's lateral action horizon, controlsd's controller and path reference, torqued's sample alignment,
both calibration rows — reads it unchanged (`moonpilot_procs` in `moonpilot/procs.py` swaps the row).
The identification is `LateralLagEstimator` itself: same signal pair (`desiredCurvature * v²` against
`yaw_rate * v`), same correlation, quality gates, ROI and block size. Three things differ, each a
measured defect of the stock learner on this car:

- **Speed gate 10 m/s, not 50 mph.** Stock learns only above 22.35 m/s and applies the one number at
  every speed. Replayed over routes 000003f3…0000040a (stock estimator, gated per band), the delay is
  flat where it can be measured: 0.39 s at 10–17 m/s, 0.36 at 17–22, 0.40 above. Below 10 m/s too few
  windows clear the gates to say anything, so the gate stops there.
- **A 10-block ring, not 50.** A completed block weighs 1/(valid blocks), 1/50 at steady state, and
  highway blocks are rare: those routes earned one, moving the applied value 0.215 → 0.228 against a
  measured ~0.40. Ten blocks — nine valid once the ring wraps, `BlockAverage` excludes the slot being
  written — at the lower speed gate refresh within a drive or two, which is also what retires blocks
  learned under a different controller, the stock cache keying only on the fingerprint.
- **Persistence keeps the blocks.** Stock saves one number and `BlockAverage.__init__` tiles it into
  every slot on boot, so the history is flattened each boot, the spread check reads 0, and the saved
  `current_mean` gives a partial block a full block's weight. Here `points` carries the completed
  blocks oldest first and the boot restores exactly them; the partial block is not saved. `version` is
  in the fork band, so a stock cache (version 1) is discarded rather than seeded — the stale value this
  replaces — and stock `lagd` would discard this one in turn.

Unestimated (fewer than 5 blocks) or invalid (block spread over `MAX_LAG_STD`) starts from stock's
`steerActuatorDelay + 0.2`. Whatever the source, the *applied* `lateralDelay` is capped at
`MOONPILOT_LAT_LAG_MAX_APPLIED`; `lateralDelayEstimate` and the blocks stay uncapped, so the learner's
view is still logged. The loop below mirrors upstream's `main()` line for line; a change there
does not reach this copy by itself.
"""
import math

import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.locationd.lagd import MAX_LAG, MIN_LAG, LateralLagEstimator

from moonpilot.latency import completed_blocks, restore_blocks

MOONPILOT_LAT_LAG_MIN_SPEED = 10.0  # m/s
MOONPILOT_LAT_LAG_BLOCK_COUNT = 10
MOONPILOT_LAT_LAG_VERSION = 1000  # fork band; stock lagd's VERSION is 1
# ponytail: a fixed ceiling, because the estimate is measured in closed loop. The torque controller holds
# the wheel to the request from lat_delay ago, so desiredCurvature -> yaw lag reads about the applied delay
# plus the vehicle's response: 408 applied 0.228 and measured ~0.31; 40e applied 0.32 (+0.05 smoothing) and its
# first partial block read 0.469. Left alone the applied value ratchets toward MAX_LAG. Lift the cap once the
# estimate is taken against something the controller's delay does not shape (open-loop torque -> yaw).
MOONPILOT_LAT_LAG_MAX_APPLIED = 0.25  # s
MOONPILOT_LAT_LAG_KEY = "LiveDelay"  # upstream's cache, so both reset-calibration buttons clear it


def persisted_blocks(params, CP) -> list[float]:
  """The cached blocks when they are this estimator's, for this car, and in range; else none."""
  try:
    with log.Event.from_bytes(params.get(MOONPILOT_LAT_LAG_KEY)) as msg, \
         car.CarParams.from_bytes(params.get("CarParamsPrevRoute")) as last_CP:
      ld = msg.lateralDelay
      blocks = [float(b) for b in ld.points]
      if ld.version != MOONPILOT_LAT_LAG_VERSION or last_CP.carFingerprint != CP.carFingerprint:
        return []
  except Exception:
    return []
  if len(blocks) >= MOONPILOT_LAT_LAG_BLOCK_COUNT or not all(math.isfinite(b) and MIN_LAG <= b <= MAX_LAG for b in blocks):
    return []
  return blocks


class MoonpilotLagEstimator(LateralLagEstimator):
  def __init__(self, CP, dt: float):
    super().__init__(CP, dt, block_count=MOONPILOT_LAT_LAG_BLOCK_COUNT, min_vego=MOONPILOT_LAT_LAG_MIN_SPEED)

  def restore(self, blocks: list[float]) -> None:
    restore_blocks(self.block_avg, blocks)

  def get_msg(self, valid: bool, debug: bool = False):
    msg = super().get_msg(valid)
    msg.lateralDelay.lateralDelay = min(msg.lateralDelay.lateralDelay, MOONPILOT_LAT_LAG_MAX_APPLIED)
    msg.lateralDelay.points = completed_blocks(self.block_avg)
    msg.lateralDelay.version = MOONPILOT_LAT_LAG_VERSION
    return msg


def main():
  config_realtime_process([0, 1, 2, 3], 5)

  pm = messaging.PubMaster(['lateralDelay'])
  sm = messaging.SubMaster(['deviceMotion', 'extrinsicsCalibration', 'carState', 'controlsState', 'carControl'], poll='deviceMotion')

  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  lag_learner = MoonpilotLagEstimator(CP, 1. / SERVICE_LIST['deviceMotion'].frequency)
  lag_learner.restore(persisted_blocks(params, CP))

  while True:
    sm.update()
    if sm.all_checks():
      for which in sorted(sm.updated.keys(), key=lambda x: sm.logMonoTime[x]):
        if sm.updated[which]:
          lag_learner.handle_log(sm.logMonoTime[which] * 1e-9, which, sm[which])
      lag_learner.update_points()

    # 4Hz driven by deviceMotion
    if sm.frame % 5 == 0:
      lag_learner.update_estimate()
      lag_msg_dat = lag_learner.get_msg(sm.all_checks()).to_bytes()
      pm.send('lateralDelay', lag_msg_dat)

      if sm.frame % 1200 == 0:  # cache every 60 seconds
        params.put(MOONPILOT_LAT_LAG_KEY, lag_msg_dat)


if __name__ == "__main__":
  main()
