"""The fork's on-road palette: what a car that is steering but not holding speed looks like.

Upstream renders engagement as a palette, and both trees key it off `ui_state.status` — tizi
paints the road-view border from `BORDER_COLORS`, mici its lane lines from `LANE_LINE_COLORS`,
each with the same three readings (disengaged, override, engaged). That status is
`selfdriveState.enabled` and nothing else, which on a stock-ACC car is not a complete statement:
openpilot never owns the speed there, so the engaged palette has always meant "openpilot steers,
and the ACC I set is holding speed". Half-engagement paints the same palette with nobody holding
speed — a combination the car's visual grammar has no word for, and the reading a driver is most
likely to take from a lit-up car is that it is driving.

So the fork keeps blue for the half-engaged state — the one color the palette had spent on
"disengaged", which moves to a neutral gray — because a car that is doing nothing deserves the
neutral reading and a car that is steering for you does not. The call sites keep upstream's own
expression on the line, and hand this module the tree's palette so it can name the neutral itself:

    border_color = (moonpilot_status_color(ui_state, BORDER_COLORS) or
                    BORDER_COLORS.get(ui_state.status, BORDER_COLORS[UIStatus.DISENGAGED]))

`None` means upstream's expression runs unchanged, so a car that is not half-engaged — every car
with the feature off, unavailable, or simply not one of the three brands — cannot reach the fork's
color at all.
Every frame that *is* half-engaged gets a color from here, which is the point: deferring to
upstream for the driver-steering case would paint `ENGAGED` green on any frame where `carState` has
the driver's hands on the wheel before `selfdriveState` has caught up with the override — and green
on a car that is only steering is the reading this whole module exists to prevent.

Blue is painted only while openpilot is the thing steering, which is three conditions:

- **Something is on.** `status != DISENGAGED` — upstream's own reading, and not redundant with the
  grant below: `selfdriveState` and `pandaStates` are separate messages, so for a frame or two
  after a disengage the last panda message still says the grant is held. Without this the border
  would flash blue on the way out.
- **The safety layer is still granting lateral authority.** `controlsAllowedLateral and not
  controlsAllowed` — the fork's own grant against stock ACC engagement, exact rather than inferred
  from `carState`, and false on every car where the rule was never enabled.
- **The driver is not steering.** `carState.steeringPressed`, which is what raises upstream's own
  `steerOverride` — so whatever makes a *fully* engaged car read as overridden makes a
  half-engaged car read as the same neutral, `palette[UIStatus.OVERRIDE]`, taken from the tree's
  own palette rather than copied here so the two cannot drift. It cannot come from the safety
  layer: the rule's steering-override term is upstream's `steering_disengage`, which only Tesla's
  rx hook ever sets, so no car the fork enables the rule for can reach it. Blue here would claim openpilot is steering while
  the driver's own hands are on the wheel doing the work.

`gasPressedOverride` deliberately does **not** drop the color, which is why this is not simply
"status isn't OVERRIDE": in half-engagement the driver's foot owns the speed, so they are on the
pedal most of the time, and the car is still steering for them throughout. Grey is then left
meaning exactly one of two things — nothing is on, or you are driving — and never "openpilot is
steering" when it is not.
"""

from typing import Any, Optional

import pyray as rl

from openpilot.selfdrive.ui.ui_state import UIStatus

# Saturated mid blue: the classic reading of "assist is active, but only partly", and distinct
# from upstream's engaged green and its two neutral grays.
HALF_ENGAGED = rl.Color(0x1E, 0x76, 0xD2, 0xFF)


def lateral_only(panda_states) -> bool:
  """Whether a panda is granting steering while withholding longitudinal authority."""
  return any(ps.controlsAllowedLateral and not ps.controlsAllowed for ps in panda_states)


def moonpilot_status_color(ui_state: Any, palette: dict) -> Optional[rl.Color]:  # noqa: UP045  # rl.Color is a function, so `rl.Color | None` fails
  """The fork's color for a half-engaged car, or `None` to leave upstream's in place.

  Takes the UI state itself rather than unpacked values, so both trees' call sites stay one
  expression — the whole decision is this one question, and it is the same question in tizi and
  mici. `palette` is the calling tree's `UIStatus` to color map, which is what makes the
  driver-steering color the same gray upstream paints a frame later rather than a second copy of
  it here. Loosely typed because the tests hand it a stand-in rather than the singleton.
  """
  if ui_state.status == UIStatus.DISENGAGED or not lateral_only(ui_state.sm['pandaStates']):
    return None
  if ui_state.sm['carState'].steeringPressed:
    return palette[UIStatus.OVERRIDE]
  return HALF_ENGAGED
