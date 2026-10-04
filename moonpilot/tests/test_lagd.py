import unittest

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.selfdrive.locationd.lagd import BLOCK_NUM_NEEDED, BLOCK_SIZE, MIN_OKAY_WINDOW_SEC, VERSION, LateralLagEstimator

from moonpilot.lagd import (
  MOONPILOT_LAT_LAG_BLOCK_COUNT,
  MOONPILOT_LAT_LAG_KEY,
  MoonpilotLagEstimator,
  persisted_blocks,
)

DT = 0.05
FINGERPRINT = "TOYOTA_RAV4_TSS2"


def _cp(fingerprint=FINGERPRINT):
  return car.CarParams(steerActuatorDelay=0.12, carFingerprint=fingerprint)


def _feed(estimator, v_ego, lag_frames=7, seconds=None):
  """upstream test_lagd's synthetic drive: a 0.3 m/s^2 weave, the yaw response lag_frames behind."""
  n = int((seconds or (MIN_OKAY_WINDOW_SEC + 5)) / DT) + BLOCK_NUM_NEEDED * BLOCK_SIZE
  for i in range(n):
    t = i * DT
    desired_la = np.cos(10 * t) * 0.3
    actual_la = np.cos(10 * (t - lag_frames * DT)) * 0.3
    for w, m in (
      ("carControl", car.CarControl(latActive=True)),
      ("carState", car.CarState(vEgo=v_ego, steeringPressed=False)),
      ("controlsState", log.ControlsState(desiredCurvature=float(desired_la / v_ego ** 2))),
      ("deviceMotion", log.DeviceMotion(angularVelocityDevice=log.DeviceMotion.XYZMeasurement(z=float(actual_la / v_ego), valid=True),
                                        posenetOK=True, inputsOK=True)),
      ("extrinsicsCalibration", log.ExtrinsicsCalibration(rpyCalib=[0, 0, 0], calStatus=log.ExtrinsicsCalibration.Status.calibrated)),
    ):
      estimator.handle_log(t, w, m)
    estimator.update_points()
    estimator.update_estimate()


class _Params:
  def __init__(self, values):
    self.values = values

  def get(self, key):
    return self.values.get(key)


def _cache(msg, fingerprint=FINGERPRINT):
  return _Params({MOONPILOT_LAT_LAG_KEY: msg.to_bytes(),
                  "CarParamsPrevRoute": car.CarParams.new_message(carFingerprint=fingerprint).to_bytes()})


class TestLagd(unittest.TestCase):
  def test_a_city_drive_earns_an_estimate_stock_cannot(self):
    fork = MoonpilotLagEstimator(_cp(), DT)
    stock = LateralLagEstimator(_cp(), DT)
    for est in (fork, stock):
      _feed(est, v_ego=15.0)
    fork_msg, stock_msg = fork.get_msg(True).lateralDelay, stock.get_msg(True).lateralDelay
    self.assertEqual(fork_msg.status, "estimated")
    self.assertAlmostEqual(fork_msg.lateralDelay, 7 * DT, delta=0.01)
    self.assertEqual(stock_msg.status, "unestimated")

  def test_below_the_speed_gate_nothing_is_learned(self):
    est = MoonpilotLagEstimator(_cp(), DT)
    _feed(est, v_ego=9.0)
    msg = est.get_msg(True).lateralDelay
    self.assertEqual(msg.status, "unestimated")
    self.assertAlmostEqual(msg.lateralDelay, 0.12 + 0.2, delta=1e-6)

  def test_the_cache_restores_the_blocks_not_their_mean(self):
    blocks = [0.30, 0.32, 0.40, 0.36, 0.34]
    est = MoonpilotLagEstimator(_cp(), DT)
    est.restore(blocks)
    est.block_avg.update(0.6)  # a partial block: published in the estimate, never cached
    msg = est.get_msg(True)
    self.assertEqual(msg.lateralDelay.status, "estimated")
    self.assertAlmostEqual(msg.lateralDelay.lateralDelay, np.mean(blocks), delta=1e-6)

    restored = persisted_blocks(_cache(msg), _cp())
    np.testing.assert_allclose(restored, blocks, atol=1e-6)
    again = MoonpilotLagEstimator(_cp(), DT)
    again.restore(restored)
    ld = again.get_msg(True).lateralDelay
    self.assertAlmostEqual(ld.lateralDelay, np.mean(blocks), delta=1e-6)
    self.assertAlmostEqual(ld.lateralDelayEstimateStd, np.std(blocks), delta=1e-6)  # the spread survives a boot

  def test_a_full_ring_retires_its_oldest_block(self):
    old = [0.20 + 0.01 * i for i in range(MOONPILOT_LAT_LAG_BLOCK_COUNT - 1)]
    est = MoonpilotLagEstimator(_cp(), DT)
    est.restore(old)
    for _ in range(BLOCK_SIZE):
      est.block_avg.update(0.5)
    np.testing.assert_allclose(list(est.get_msg(True).lateralDelay.points), old[1:] + [0.5], atol=1e-6)

  def test_a_foreign_or_corrupt_cache_starts_fresh(self):
    good = MoonpilotLagEstimator(_cp(), DT)
    good.restore([0.3] * 5)
    stock_msg = good.get_msg(True)
    stock_msg.lateralDelay.version = VERSION
    out_of_range = good.get_msg(True)
    out_of_range.lateralDelay.points = [0.3, 2.0]
    cases = {
      "stock lagd's cache": _cache(stock_msg),
      "another car": _cache(good.get_msg(True), fingerprint="HONDA_CIVIC"),
      "a block outside the ROI": _cache(out_of_range),
      "nothing cached": _Params({}),
    }
    for name, params in cases.items():
      with self.subTest(name):
        self.assertEqual(persisted_blocks(params, _cp()), [])

  def test_the_lagd_row_runs_the_fork_estimator(self):
    from openpilot.system.manager.process_config import managed_processes, only_onroad
    lagd = managed_processes["lagd"]
    self.assertEqual(getattr(lagd, "module", None), "moonpilot.lagd")
    self.assertIs(lagd.should_run, only_onroad)


if __name__ == "__main__":
  unittest.main()
