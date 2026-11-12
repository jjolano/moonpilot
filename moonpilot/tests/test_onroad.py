"""The fork's one onroad color: when a car is steering without holding speed.

The load-bearing question is not which color, it is when the fork is allowed to paint one at all.
Both call sites are delegates — `moonpilot_status_color(...) or BORDER_COLORS.get(...)`, and the
same over mici's `LANE_LINE_COLORS` — so a `None` is not "no color": it is upstream's own
expression running unchanged. These pin the three ways that has to come out, and the one input
that decides it.
"""

import unittest
from types import SimpleNamespace

import numpy as np

import raylib_nb as rl

from openpilot.selfdrive.ui.ui_state import UIStatus
from moonpilot.tests.fakes import params
from moonpilot.ui.onroad import HALF_ENGAGED, lateral_only, lead_path_line, moonpilot_status_color


class _PandaState:
  """The two fields the predicate reads, as the UI reads them off a real PandaState."""

  def __init__(self, controls_allowed=False, controls_allowed_lateral=False):
    self.controlsAllowed = controls_allowed
    self.controlsAllowedLateral = controls_allowed_lateral


class TestLateralOnly(unittest.TestCase):
  def test_grant_without_longitudinal_authority(self):
    # The half-engaged state: the safety layer permits steering, stock ACC is not set
    self.assertTrue(lateral_only([_PandaState(controls_allowed_lateral=True)]))

  def test_full_engagement_is_not_lateral_only(self):
    # ACC set on top: both flags, which is upstream's ordinary engaged car
    self.assertFalse(lateral_only([_PandaState(controls_allowed=True, controls_allowed_lateral=True)]))

  def test_a_car_that_never_arms_the_rule_is_not_lateral_only(self):
    """Every car the fork did not flag: the permission bit is never set, so nothing shows."""
    for panda in (_PandaState(), _PandaState(controls_allowed=True)):
      self.assertFalse(lateral_only([panda]))

  def test_any_panda_granting_is_enough(self):
    # Main and harness pandas disagree only transiently; the UI shows what is being granted
    self.assertTrue(lateral_only([_PandaState(), _PandaState(controls_allowed_lateral=True)]))


class _CarState:
  def __init__(self, steering_pressed):
    self.steeringPressed = steering_pressed


class _UIState:
  """The three fields the palette reads, in the shape the real singleton has them."""

  def __init__(self, status, panda_states, steering_pressed=False):
    self.status = status
    self.sm = {"pandaStates": panda_states, "carState": _CarState(steering_pressed)}


GRANTED = [_PandaState(controls_allowed_lateral=True)]
ENGAGED, OVERRIDE, DISENGAGED = UIStatus.ENGAGED, UIStatus.OVERRIDE, UIStatus.DISENGAGED
# A stand-in palette; the real ones are exercised end to end by the render probe.
PALETTE = {ENGAGED: rl.Color(22, 127, 64, 255), OVERRIDE: rl.Color(137, 146, 141, 255), DISENGAGED: rl.Color(58, 62, 64, 255)}


class TestStatusColor(unittest.TestCase):
  def test_a_half_engaged_car_is_blue(self):
    self.assertEqual(moonpilot_status_color(_UIState(ENGAGED, GRANTED), PALETTE), HALF_ENGAGED)

  def test_none_when_nothing_is_on(self):
    """Disengaged is the neutral reading, and it is not the fork's color to paint"""
    self.assertIsNone(moonpilot_status_color(_UIState(DISENGAGED, GRANTED), PALETTE))

  def test_none_when_engaged_without_the_grant(self):
    """A fully engaged car keeps upstream's palette, which is the overwhelming majority of drives"""
    for panda in (_PandaState(controls_allowed=True, controls_allowed_lateral=True), _PandaState(), _PandaState(controls_allowed=True)):
      self.assertIsNone(moonpilot_status_color(_UIState(ENGAGED, [panda]), PALETTE))

  def test_the_driver_steering_turns_it_gray(self):
    """`steeringPressed` is what raises upstream's own `steerOverride`, on a fully engaged car too.

    Half-engaged, the driver's hands on the wheel are doing the steering, so the car goes back to
    the neutral reading rather than claiming openpilot is driving. The safety layer cannot do this
    on any brand the rule is enabled for — its steering-override term is upstream's
    `steering_disengage`, which only Tesla's rx hook sets — and it keeps granting lateral authority
    throughout, which is why the grant alone is not enough.
    """
    self.assertTrue(lateral_only(GRANTED))
    self.assertEqual(moonpilot_status_color(_UIState(OVERRIDE, GRANTED, steering_pressed=True), PALETTE), PALETTE[OVERRIDE])
    # ...including on the frame where selfdriveState has not caught up with the override yet, which
    # is the frame upstream would otherwise paint ENGAGED green
    self.assertEqual(moonpilot_status_color(_UIState(ENGAGED, GRANTED, steering_pressed=True), PALETTE), PALETTE[OVERRIDE])

  def test_a_gas_override_stays_blue(self):
    """In half-engagement the driver's foot owns the speed, so the pedal must not drop the color.

    `gasPressedOverride` makes upstream's status OVERRIDE while the car is still steering for the
    driver, which is why the predicate is not simply "status isn't OVERRIDE".
    """
    self.assertEqual(moonpilot_status_color(_UIState(OVERRIDE, GRANTED), PALETTE), HALF_ENGAGED)

  def test_none_is_what_leaves_upstream_in_place(self):
    """The call sites are `... or BORDER_COLORS.get(...)`, so `None` has to fall through to them"""
    upstream = rl.Color(0x16, 0x7F, 0x40, 0xFF)
    self.assertEqual(moonpilot_status_color(_UIState(ENGAGED, [_PandaState()]), PALETTE) or upstream, upstream)
    self.assertEqual(moonpilot_status_color(_UIState(ENGAGED, GRANTED), PALETTE) or upstream, HALF_ENGAGED)

class TestLeadPathProjection(unittest.TestCase):
  def test_projects_the_published_lead_on_the_ui_horizon(self):
    lead = SimpleNamespace(
      present=True, t=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
      x=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
      y=[0.0] * 6, yStd=[1.0] * 6, inPath=0.75,
    )
    renderer = SimpleNamespace(
      _path=SimpleNamespace(raw_points=np.array([[x, 0.0, 1.0] for x in range(61)], dtype=np.float32)),
      _get_path_length_idx=lambda path_x, x: int(round(x)),
      _clip_region=rl.Rectangle(-100.0, -100.0, 200.0, 200.0),
      _car_space_transform=np.eye(3, dtype=np.float32),
      _path_offset_z=0.0,
    )
    state = SimpleNamespace(leads=[lead])
    raw, points, widths, in_path = lead_path_line(renderer, state, np.arange(61), params(overrides={"MoonpilotLeadLateral": True}))
    self.assertEqual(raw.shape, (9, 3))
    self.assertEqual(points.shape, (9, 2))
    self.assertEqual(widths.shape, (9,))
    self.assertEqual(float(raw[-1, 0]), 30.0)
    self.assertEqual(in_path, 0.75)


if __name__ == "__main__":
  unittest.main()
