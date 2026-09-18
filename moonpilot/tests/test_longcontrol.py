"""The fork acceleration controller: the tracking contract controlsd calls it with.

The controller must hold up its half of controlsd's contract — same `update` signature, the
LongControlState that controlsState and both UIs read, a command inside the car's own accel limits —
and the three things the fork owns on top of upstream's loop: the output rate limit (and the one
place it is deliberately dropped, the handover out of the stopping state), the integrator freeze, and
the stopping ramp.
"""

import unittest
from typing import cast

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL

from moonpilot.longcontrol import (
  MOONPILOT_ACCEL_JERK,
  MOONPILOT_STANDSTILL_SPEED,
  MoonpilotLongControl,
  moonpilot_longcontrol,
)

LongCtrlState = car.CarControl.Actuators.LongControlState

ACCEL_LIMITS = (-4.0, 2.0)


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store."""

  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


def _controller():
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
  return MoonpilotLongControl(CP), CP.as_reader()


def _state(v_ego=20.0, a_ego=0.0, brake=False, gas=False, standstill=False):
  CS = car.CarState.new_message()
  CS.vEgo = float(v_ego)
  CS.aEgo = float(a_ego)
  CS.brakePressed = bool(brake)
  CS.gasPressed = bool(gas)
  CS.cruiseState.standstill = bool(standstill)
  return CS


def _step(controller, CS, a_target, should_stop=False, active=True, frames=1, limits=ACCEL_LIMITS):
  for _ in range(frames):
    output = controller.update(active, CS, a_target, should_stop, limits)
  return output


class TestMoonpilotLongControl(unittest.TestCase):
  def test_off_is_a_zero_command_and_a_reset_integrator(self):
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=0.0)
    _step(controller, CS, -1.0, frames=10)  # wind the integrator up first
    self.assertNotEqual(controller.pid.i, 0.0)

    output = _step(controller, CS, -1.0, active=False)
    self.assertEqual(output, 0.0)
    self.assertEqual(controller.pid.i, 0.0)
    self.assertEqual(controller.long_control_state, LongCtrlState.off)

  def test_a_matched_target_is_applied_as_feedforward(self):
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=1.0)
    # enough frames for the output rate limit to walk the command up to the matched target
    self.assertAlmostEqual(_step(controller, CS, 1.0, frames=20), 1.0, delta=1e-6)
    self.assertEqual(controller.pid.i, 0.0)

  def test_the_command_is_rate_limited(self):
    """A step in the plan is not a step at the actuator: the whole point of the first fork change."""
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=0.0)
    bound = MOONPILOT_ACCEL_JERK * DT_CTRL + 1e-9
    _step(controller, CS, 0.0, frames=5)

    previous = 0.0
    for _ in range(100):
      output = _step(controller, CS, 2.0)
      self.assertLessEqual(abs(output - previous), bound)
      previous = output
    self.assertAlmostEqual(previous, 2.0, delta=0.1)  # and it does get there

  def test_the_rate_limit_applies_to_braking_too(self):
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=0.0)
    previous = _step(controller, CS, 0.0, frames=5)
    for _ in range(100):
      output = _step(controller, CS, -3.0)
      self.assertLessEqual(abs(output - previous), MOONPILOT_ACCEL_JERK * DT_CTRL + 1e-9)
      previous = output

  def test_stopping_ramps_down_to_the_car_stop_accel(self):
    controller, CP = _controller()
    CS = _state(v_ego=5.0, a_ego=0.0, standstill=True)
    outputs = []
    for _ in range(200):
      outputs.append(_step(controller, CS, -1.0, should_stop=True))
    self.assertEqual(controller.long_control_state, LongCtrlState.stopping)
    self.assertAlmostEqual(outputs[-1], CP.stopAccel, delta=1e-6)
    self.assertGreaterEqual(min(outputs), CP.stopAccel)
    # monotone descent through the ramp
    for earlier, later in zip(outputs, outputs[1:], strict=False):
      self.assertLessEqual(later, earlier + 1e-12)

  def test_stopping_never_brakes_past_the_limits(self):
    controller, _ = _controller()
    CS = _state(v_ego=5.0, a_ego=0.0, standstill=True)
    for _ in range(200):
      output = _step(controller, CS, -1.0, should_stop=True, limits=(-1.0, 2.0))
    self.assertGreaterEqual(output, -1.0)

  def test_integrator_freezes_at_standstill(self):
    controller, _ = _controller()
    CS = _state(v_ego=MOONPILOT_STANDSTILL_SPEED / 2, a_ego=0.0)
    _step(controller, CS, 1.0, frames=50)
    self.assertEqual(controller.pid.i, 0.0)

  def test_integrator_freezes_while_the_driver_is_on_a_pedal(self):
    for brake, gas in ((True, False), (False, True)):
      controller, _ = _controller()
      CS = _state(v_ego=20.0, a_ego=0.0, brake=brake, gas=gas)
      _step(controller, CS, 1.0, frames=50)
      self.assertEqual(controller.pid.i, 0.0)

  def test_integrator_runs_while_tracking(self):
    """The freeze must be a freeze, not a permanently disabled integrator."""
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=0.0)
    _step(controller, CS, 1.0, frames=50)
    self.assertGreater(controller.pid.i, 0.0)

  def test_reset_clears_the_integrator(self):
    controller, _ = _controller()
    CS = _state(v_ego=20.0, a_ego=0.0)
    _step(controller, CS, 1.0, frames=50)
    controller.reset()
    self.assertEqual(controller.pid.i, 0.0)

  def test_the_command_stays_inside_the_car_limits(self):
    controller, _ = _controller()
    controller.last_output_accel = 1.0
    CS = _state(v_ego=20.0, a_ego=0.0)
    for _ in range(50):
      self.assertGreaterEqual(_step(controller, CS, 5.0, limits=(-1.0, 0.5)), -1.0)
    self.assertLessEqual(controller.last_output_accel, 0.5)

  def test_leaving_the_stop_hold_does_not_ramp_back_through_the_brake(self):
    """The handover out of `stopping`, and the one place the fork's own rate limit is wrong.

    In the stopping state `last_output_accel` sits at `CP.stopAccel`, but that is a brake command rather
    than a command being tracked toward the plan, and the clearance that releases the car's own
    standstill hold is downstream of the delivered accel — Toyota's carcontroller drops its PCM
    standstill request on `actuators.accel > 0`. So the first pid frame has to command the plan rather
    than walk up from -2.0, which at MOONPILOT_ACCEL_JERK * DT_CTRL is 25 frames, 240 ms of commanded
    braking after the hold came off.

    Closed loop that is the cost of a car pinned in stopping by its own brake or the PCM's standstill
    bit. Measured from the `stopping -> pid` edge, held 20 s and then cleared: pre-fix commands -1.92,
    -1.84, -1.76, -1.68 where the plan asks +0.15 and turns positive 240 ms later; with the reset it
    commands +0.08 on the first frame. The free path pays the same 240 ms from a 10 s dwell up;
    `test_a_shallow_stop_hold_does_not_delay_the_release_either` covers the shallower holds.
    """
    controller, CP = _controller()
    held = _state(v_ego=0.0, standstill=True)
    for _ in range(200):
      output = _step(controller, held, 0.0, should_stop=True)
    self.assertEqual(controller.long_control_state, LongCtrlState.stopping)
    self.assertAlmostEqual(output, CP.stopAccel, delta=1e-6)

    # the lead moves off: shouldStop clears and the plan asks for +0.5
    released = _state(v_ego=0.0)
    self.assertGreaterEqual(_step(controller, released, 0.5, should_stop=False), 0.0)
    self.assertEqual(controller.long_control_state, LongCtrlState.pid)
    # and the rate limit still owns the rest of the ramp inside the pid state
    self.assertAlmostEqual(_step(controller, released, 0.5, frames=20), 0.5, delta=1e-6)

  def test_a_shallow_stop_hold_does_not_delay_the_release_either(self):
    """The reset fires on every `stopping -> pid` edge, not only at the brake floor, and this is why.

    Everything the stopping branch produces is a brake command, so a half-ramped hold carried into the
    pid branch commands braking while the plan asks for motion — the same defect as the saturated hold,
    just for less time. Measured from the edge at a 10 s dwell the pre-fix loop commands -0.76 and the
    plan asks +0.15; at 11 s it commands -1.76. A reset makes the first frame land on the plan's side.

    Asserted on the delivered accel: a -0.2 hold releasing into a +0.5 plan comes out at +0.08, one
    rate-limit step from the reset baseline, and at -0.12 if the hold is carried over.
    """
    controller, _ = _controller()
    creeping = _state(v_ego=0.2, standstill=False)
    for _ in range(20):
      _step(controller, creeping, -0.5, should_stop=True)
    self.assertEqual(controller.long_control_state, LongCtrlState.stopping)
    self.assertAlmostEqual(controller.last_output_accel, -0.2, delta=1e-6)  # nowhere near the floor

    output = _step(controller, creeping, 0.5, should_stop=False)
    self.assertGreater(output, 0.0)  # already on the plan's side, not still ramping through the brake
    self.assertAlmostEqual(output, MOONPILOT_ACCEL_JERK * DT_CTRL, delta=1e-6)

  def test_engaging_from_a_standstill_starts_stopping_not_driving(self):
    """long_control_state_trans' contract, which is why the fork reuses upstream's function."""
    controller, _ = _controller()
    CS = _state(v_ego=0.0, standstill=True)
    _step(controller, CS, 1.0, should_stop=True)
    self.assertEqual(controller.long_control_state, LongCtrlState.stopping)


class TestLongControlSeam(unittest.TestCase):
  def test_factory_returns_the_fork_controller_only_when_wanted(self):
    CP = CarInterface.get_non_essential_params(CAR.HONDA_CIVIC)
    self.assertIsInstance(moonpilot_longcontrol(CP, _params(on=True)), MoonpilotLongControl)
    self.assertIsNone(moonpilot_longcontrol(CP, _params(on=False)))


if __name__ == "__main__":
  unittest.main()
