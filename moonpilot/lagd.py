"""Fork lateral-delay learner; upstream's process row is replaced in `moonpilot/procs.py`."""
import capnp
import math

import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process
from openpilot.selfdrive.locationd.lagd import MAX_LAG, MIN_LAG, LateralLagEstimator

from moonpilot.learned import PERSIST_EVERY, completed_blocks, restore_blocks

MOONPILOT_LAT_LAG_MIN_SPEED = 10.0  # m/s
MOONPILOT_LAT_LAG_BLOCK_COUNT = 10
MOONPILOT_LAT_LAG_VERSION = 1000  # fork band; stock lagd's VERSION is 1
# The measured applied-delay replay is documented in `docs/curvature.md`.
MOONPILOT_LAT_LAG_MAX_APPLIED = 0.25  # s
MOONPILOT_LAT_LAG_KEY = "LiveDelay"  # upstream's cache, so both reset-calibration buttons clear it


def cached_lag_blocks(params, CP) -> list[float]:
  cached = params.get(MOONPILOT_LAT_LAG_KEY)
  previous_car = params.get("CarParamsPrevRoute")
  if cached is None or previous_car is None:
    return []
  try:
    with log.Event.from_bytes(cached) as msg, car.CarParams.from_bytes(previous_car) as last_CP:
      ld = msg.lateralDelay
      blocks = [float(b) for b in ld.points]
      if ld.version != MOONPILOT_LAT_LAG_VERSION or last_CP.carFingerprint != CP.carFingerprint:
        return []
  except (capnp.lib.capnp.KjException, ValueError, TypeError):
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
    msg = super().get_msg(valid, debug)
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
  lag_learner.restore(cached_lag_blocks(params, CP))

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

      if sm.frame % PERSIST_EVERY == 0:
        params.put(MOONPILOT_LAT_LAG_KEY, lag_msg_dat)


if __name__ == "__main__":
  main()
