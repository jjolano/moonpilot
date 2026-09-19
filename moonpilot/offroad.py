"""Offroad mode: keep the device offroad while the car's ignition is on.

Upstream decides onroad/offroad in `hardwared`'s `onroad_conditions`, where every member is a
reason the device *may* be onroad: ignition, no onroad cycle, device temperature. The fork adds
one more, `moonpilot_onroad`, whose only job is `onroad_condition` below — so the whole mechanism
on the upstream side is that dict member and the line that refreshes it.

The mode itself is a param, not a process state: it is entered from the UI (the moonpilot panel's
row, or a hold on the driving view), and while it is set the device stays offroad. It is
`CLEAR_ON_IGNITION_ON`, so a new ignition cycle always comes up onroad, and `CLEAR_ON_MANAGER_START`,
so a reboot does too — neither a fresh drive nor a restart can inherit it. Both entries read the
same three functions (`requested`, `can_enter`, `enabled`) and the same labels, which is what keeps
the row and the gesture from disagreeing about when the mode may be entered: the car is on (so
`started`), it is not being driven (`v_ego` under `MOONPILOT_OFFROAD_SPEED`, the fork's standstill
convention), and openpilot is not engaged. Leaving offroad is never gated: the device has to stay
recoverable from the state the mode puts it in.
"""

MOONPILOT_OFFROAD_KEY = "MoonpilotOffroad"
MOONPILOT_OFFROAD_SPEED = 0.1  # m/s; the fork's standstill convention, mirroring
# MOONPILOT_STANDSTILL_SPEED (moonpilot/longcontrol.py) and MOONPILOT_SHOULD_STOP_SPEED
# (moonpilot/longitudinal.py); a test pins them together.

ENTER_LABEL = "TURN ON"  # the action the row and the gesture offer
EXIT_LABEL = "TURN OFF"  # the mode is set: this is the way back onroad
OFFROAD_LABEL = "OFFROAD"  # already offroad and not because of this mode: nothing to switch
NOT_PARKED_LABEL = "PARKED ONLY"  # onroad but moving, or engaged

# One literal per string, continued with a backslash: ruff's ISC set rejects implicitly
# concatenated string literals over multiple lines (pyproject.toml, allow-multiline = false),
# and this copy is too long for one 160-column line.
DESCRIPTION = "Keep the device offroad while the car stays on: settings, uploads and SSH work, but \
openpilot is off — no assist, no camera logs. Enter it while parked and disengaged, by holding \
the driving view or with this row. It ends when you turn this row off, and with the next ignition \
cycle or a reboot."
CONFIRM_TEXT = "Switch the device to offroad mode? The car stays on, and openpilot stops until you \
switch it back off in the moonpilot panel, cycle the ignition, or reboot."
SLIDE_LABEL = "slide to go offroad"


def requested(params) -> bool:
  """Whether the driver asked for the mode. Read with the declared default: an unset param is off,
  inside and outside the manager, which is why `return_default=True` is the read to use here."""
  return bool(params.get(MOONPILOT_OFFROAD_KEY, return_default=True))


def can_enter(started: bool, v_ego: float, engaged: bool) -> bool:
  """Whether the mode may be entered right now: the car is on, parked, and openpilot is not
  engaged. A car crawling in traffic at a standstill is engaged; a parked car with the engine
  running is neither."""
  return started and not engaged and v_ego < MOONPILOT_OFFROAD_SPEED


def enabled(params, parked: bool) -> bool:
  """Whether the row and the gesture are live: entering is gated, leaving never is."""
  return requested(params) or parked


def button_label(params, started: bool, parked: bool) -> str:
  """What both trees print on the row, and the reason when the action is not available."""
  if requested(params):
    return EXIT_LABEL
  if not started:
    return OFFROAD_LABEL
  return ENTER_LABEL if parked else NOT_PARKED_LABEL


def onroad_condition(params) -> bool:
  """`hardwared`'s `onroad_conditions["moonpilot_onroad"]`: False holds the device offroad."""
  return not requested(params)
