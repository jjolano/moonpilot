import math
import unittest

from opendbc.car.car_helpers import interfaces
from opendbc.car.gm.values import CAR as GM
from opendbc.car.structs import car
from opendbc.car.toyota.values import CAR as TOYOTA
from opendbc.car.vehicle_model import VehicleModel

from openpilot.cereal import log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL

from moonpilot.latcontrol import (
  MOONPILOT_JERK_LOOKAHEAD_T,
  MOONPILOT_KI,
  MOONPILOT_MEAS_CUTOFF_HZ,
  MoonpilotLatControlTorque,
  moonpilot_latcontrol,
)
from moonpilot.tests.fakes import _params

# A request small enough that the controller runs inside its own limits, so the integrator
# is not anti-windup frozen at zero (the PID freezes it once the sum clips).
LINEAR_REQUEST = 0.3 / 625


def _controller(car_name=TOYOTA.TOYOTA_RAV4):
  CP = interfaces[car_name].get_non_essential_params(car_name)
  CI = interfaces[car_name](CP)
  return MoonpilotLatControlTorque(CP.as_reader(), CI, DT_CTRL), VehicleModel(CP), CI


def _state(VEgo=25.0):
  CS = car.CarState.new_message()
  CS.vEgo = VEgo
  CS.steeringPressed = False
  return CS


def _run(lac, VM, CS, params, frames, desired_curvature=0.0, active=True, lat_delay=0.2, curvature_limited=False, steer_limited=False, torques=None):
  lac_log = None
  for _ in range(frames):
    steer, _, lac_log = lac.update(active, CS, VM, params, steer_limited, desired_curvature, curvature_limited, lat_delay)
    if torques is not None:
      torques.append(steer)
  return lac_log


class TestMoonpilotLatControlTorque(unittest.TestCase):
  def test_setpoint_is_the_request_one_delay_ago(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()

    # 22.5 frames of delay: the setpoint is the midpoint of the 22- and 23-frame samples.
    for i in range(200):
      lac_log = _run(lac, VM, CS, params, 1, desired_curvature=1e-4 * i, lat_delay=0.225)

    sample_22 = 1e-4 * (199 - 22) * 625
    sample_23 = 1e-4 * (199 - 23) * 625
    self.assertAlmostEqual(lac_log.desiredLateralAccel, 1e-4 * (199 - 22.5) * 625, delta=1e-3)
    self.assertAlmostEqual(lac_log.desiredLateralAccel, (sample_22 + sample_23) / 2, delta=1e-3)

  def test_feedback_setpoint_holds_delay_on_tightening_and_looks_ahead_on_unwind(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()
    lat_delay = 0.2

    _run(lac, VM, CS, params, 100, desired_curvature=1.0 / 625, lat_delay=lat_delay)
    lac_log = _run(lac, VM, CS, params, 10, desired_curvature=2.0 / 625, lat_delay=lat_delay)
    self.assertAlmostEqual(lac_log.desiredLateralAccel, 1.0, delta=1e-3)  # tightening remains delay-matched

    _run(lac, VM, CS, params, 100, desired_curvature=2.0 / 625, lat_delay=lat_delay)
    lac_log = _run(lac, VM, CS, params, 2, desired_curvature=1.0 / 625, lat_delay=lat_delay)
    self.assertAlmostEqual(lac_log.desiredLateralAccel, 1.0, delta=1e-3)  # same-sign unwind starts early

    _run(lac, VM, CS, params, 100, desired_curvature=1.0 / 625, lat_delay=lat_delay)
    lac_log = _run(lac, VM, CS, params, 2, desired_curvature=-2.0 / 625, lat_delay=lat_delay)
    self.assertAlmostEqual(lac_log.desiredLateralAccel, 0.0, delta=1e-3)  # reversal releases but is not anticipated

  def test_jerk_tracks_a_constant_rate_request(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()

    lac_log = None
    for i in range(500):
      lac_log = _run(lac, VM, CS, params, 1, desired_curvature=1.0 * i * DT_CTRL / 625)
    self.assertAlmostEqual(lac_log.desiredLateralJerk, 1.0, delta=0.02)

  def test_jerk_is_read_at_the_lookahead_not_at_the_newest_request(self):
    """A car with a large estimated delay must not anticipate the whole delay: a request that starts
    ramping is not in the jerk until it has aged past delay - lookahead (26 frames here)."""
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()
    lat_delay = MOONPILOT_JERK_LOOKAHEAD_T + 0.26
    _run(lac, VM, CS, params, 200, desired_curvature=0.0, lat_delay=lat_delay)

    request, lac_log = 0.0, None
    for _ in range(10):  # 0.1 s of ramp: nothing has reached the lookahead yet
      request += 1.0 * DT_CTRL / 625
      lac_log = _run(lac, VM, CS, params, 1, desired_curvature=request, lat_delay=lat_delay)
    self.assertLess(abs(lac_log.desiredLateralJerk), 0.05)

    for _ in range(100):
      request += 1.0 * DT_CTRL / 625
      lac_log = _run(lac, VM, CS, params, 1, desired_curvature=request, lat_delay=lat_delay)
    self.assertAlmostEqual(lac_log.desiredLateralJerk, 1.0, delta=0.05)

  def test_roll_is_taken_out_of_the_feedforward(self):
    lac, VM, _ = _controller()
    CS = _state()

    offsets = []
    for roll in (0.0, 0.05):
      params = log.VehicleParameters.new_message()
      params.roll = roll
      # Far past FRICTION_THRESHOLD, so the friction term is identical for both rolls.
      lac_log = _run(lac, VM, CS, params, 1, desired_curvature=2.0 / 625)
      offsets.append(lac_log.f)

    self.assertAlmostEqual(offsets[1] - offsets[0], -0.05 * ACCELERATION_DUE_TO_GRAVITY, delta=1e-4)

  def test_learner_feeds_the_conversion(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()

    baseline = _run(lac, VM, CS, params, 50, desired_curvature=2.0 / 625, active=True).f
    lac.update_torque_parameters(lac.torque_params.latAccelFactor, 0.3, lac.torque_params.friction)
    self.assertAlmostEqual(_run(lac, VM, CS, params, 1, desired_curvature=2.0 / 625).f - baseline, -0.3, delta=1e-4)

    # A doubled latAccelFactor halves the torque the same request asks for, on RAV4's linear map.
    # Friction is zeroed for both runs: get_friction scales with latAccelFactor, so leaving it in
    # would grow the doubled run's feedforward by the same factor and hide the conversion.
    single, _, _ = _controller()
    single.update_torque_parameters(single.torque_params.latAccelFactor, 0.0, 0.0)
    before: list[float] = []
    _run(single, VM, CS, params, 50, desired_curvature=LINEAR_REQUEST, torques=before)
    single.update_torque_parameters(2 * single.torque_params.latAccelFactor, 0.0, 0.0)
    after: list[float] = []
    _run(single, VM, CS, params, 1, desired_curvature=LINEAR_REQUEST, torques=after)
    self.assertAlmostEqual(after[-1], before[-1] / 2, delta=1e-3)

  def test_saturation(self):
    for car_name in (TOYOTA.TOYOTA_RAV4, GM.CHEVROLET_BOLT_EUV):
      lac, VM, _ = _controller(car_name)
      CS, params = _state(30.0), log.VehicleParameters.new_message()

      self.assertTrue(_run(lac, VM, CS, params, 1000, curvature_limited=True).saturated)
      self.assertFalse(_run(lac, VM, CS, params, 1000).saturated)
      self.assertTrue(_run(lac, VM, CS, params, 1000, desired_curvature=1.0).saturated)

  def test_reset_drops_the_integral_and_keeps_the_delay_line(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()

    lac_log = _run(lac, VM, CS, params, 300, desired_curvature=LINEAR_REQUEST)
    self.assertNotEqual(lac_log.i, 0.0)

    lac.reset()
    lac_log = _run(lac, VM, CS, params, 1, desired_curvature=LINEAR_REQUEST)
    self.assertLessEqual(abs(lac_log.i), MOONPILOT_KI * DT_CTRL * abs(lac_log.error) + 1e-6)
    self.assertNotEqual(lac.requests[-1], 0.0)

  def test_inactive_outputs_nothing_but_keeps_tracking(self):
    lac, VM, _ = _controller()
    CS, params = _state(), log.VehicleParameters.new_message()

    torques: list[float] = []
    lac_log = _run(lac, VM, CS, params, 100, desired_curvature=0.002, active=False, torques=torques)

    self.assertTrue(all(t == 0.0 for t in torques))
    self.assertFalse(lac_log.active)
    self.assertNotEqual(lac_log.desiredLateralAccel, 0.0)
    self.assertEqual(lac_log.actualLateralAccel, 0.0)
    self.assertEqual(lac.requests[-1], 0.002 * 625)

  def test_integrator_freezes_when_the_output_is_not_ours(self):
    for reason in ("safety", "driver", "slow"):
      lac, VM, _ = _controller()
      CS = _state(4.0 if reason == "slow" else 25.0)
      CS.steeringPressed = reason == "driver"
      params = log.VehicleParameters.new_message()

      lac_log = _run(lac, VM, CS, params, 200, desired_curvature=LINEAR_REQUEST, steer_limited=reason == "safety")
      self.assertEqual(lac_log.i, 0.0)

  def test_output_stays_finite_and_bounded(self):
    for car_name in (TOYOTA.TOYOTA_RAV4, GM.CHEVROLET_BOLT_EUV):
      lac, VM, _ = _controller(car_name)
      params = log.VehicleParameters.new_message()

      for v, curvature in ((0.0, 0.05), (30.0, 1.0)):
        torques: list[float] = []
        _run(lac, VM, _state(v), params, 100, desired_curvature=curvature, torques=torques)
        self.assertTrue(all(abs(t) <= 1.0 for t in torques))

  def test_factory_follows_the_toggle(self):
    CP = interfaces[TOYOTA.TOYOTA_RAV4].get_non_essential_params(TOYOTA.TOYOTA_RAV4).as_reader()
    CI = interfaces[TOYOTA.TOYOTA_RAV4](CP)

    self.assertIsNone(moonpilot_latcontrol(CP, CI, DT_CTRL, _params(on=False)))
    self.assertIsInstance(moonpilot_latcontrol(CP, CI, DT_CTRL, _params(on=True)), MoonpilotLatControlTorque)

  def test_measurement_filter_rejects_a_single_frame_angle_spike(self):
    """A road step on steeringAngleDeg must not land whole in feedback: friction tracks error
    and was the term driving torque sign-flips on a rough-highway route."""
    lac, VM, _ = _controller()
    CS, params = _state(25.0), log.VehicleParameters.new_message()
    _run(lac, VM, CS, params, 100, desired_curvature=0.0)
    self.assertEqual(lac.meas_filter.x, 0.0)

    CS.steeringAngleDeg = 2.0
    raw = -VM.calc_curvature(math.radians(2.0), 25.0, 0.0) * 25.0**2
    lac_log = _run(lac, VM, CS, params, 1)
    # First frame of an 8 Hz RC at 100 Hz: alpha ~ 0.33 of the step, not the whole thing.
    self.assertLess(abs(lac_log.actualLateralAccel), 0.5 * abs(raw))
    self.assertGreater(abs(lac_log.actualLateralAccel), 0.1 * abs(raw))

    # Held, it converges to the raw measurement — the filter is a lag, not a clamp.
    lac_log = _run(lac, VM, CS, params, 200)
    self.assertAlmostEqual(lac_log.actualLateralAccel, raw, delta=0.05 * abs(raw))

    # And it keeps tracking the wheel across a reset, like the delay line.
    CS.steeringAngleDeg = 0.0
    _run(lac, VM, CS, params, 100)
    lac.reset()
    lac_log = _run(lac, VM, CS, params, 1)
    self.assertAlmostEqual(lac_log.actualLateralAccel, 0.0, delta=0.05 * abs(raw))
    self.assertEqual(MOONPILOT_MEAS_CUTOFF_HZ, 8.0)

  def test_angle_square_wave_does_not_flip_output_every_frame(self):
    """High-frequency angle noise (the bump signature) must not produce a torque sign-flip
    per half-cycle once the measurement is low-passed."""
    lac, VM, _ = _controller()
    CS, params = _state(25.0), log.VehicleParameters.new_message()
    _run(lac, VM, CS, params, 100, desired_curvature=LINEAR_REQUEST)

    torques: list[float] = []
    for i in range(100):
      CS.steeringAngleDeg = 1.5 if (i // 2) % 2 == 0 else -1.5
      _run(lac, VM, CS, params, 1, desired_curvature=LINEAR_REQUEST, torques=torques)

    flips = sum(
      1 for a, b in zip(torques, torques[1:], strict=False)
      if abs(a) > 1e-4 and abs(b) > 1e-4 and (a > 0) != (b > 0)
    )
    # Unfiltered, each 2-frame half-cycle of ±1.5° at 25 m/s is a multi-m/s² measurement step
    # and would flip every cycle (up to 50). The LPF must keep that well under half.
    self.assertLess(flips, 25)


if __name__ == "__main__":
  unittest.main()
