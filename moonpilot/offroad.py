MOONPILOT_OFFROAD_KEY = "MoonpilotOffroad"
MOONPILOT_OFFROAD_SPEED = 0.1

ENTER_LABEL = "TURN ON"
EXIT_LABEL = "TURN OFF"
OFFROAD_LABEL = "OFFROAD"
NOT_PARKED_LABEL = "PARKED ONLY"

DESCRIPTION = "Keep the device offroad while the car stays on: settings, uploads and SSH work, but \
openpilot is off — no assist, no camera logs. Enter it while parked and disengaged, by holding \
the driving view or with this row. It ends when you turn this row off, and with the next ignition \
cycle or a reboot."
CONFIRM_TEXT = "Switch the device to offroad mode? The car stays on, and openpilot stops until you \
switch it back off in the moonpilot panel, cycle the ignition, or reboot."
SLIDE_LABEL = "slide to go offroad"


def requested(params) -> bool:
  return bool(params.get(MOONPILOT_OFFROAD_KEY, return_default=True))


def can_enter(started: bool, v_ego: float, engaged: bool) -> bool:
  # A crawling car in traffic is engaged; a parked car with its engine running is not.
  return started and not engaged and v_ego < MOONPILOT_OFFROAD_SPEED


def set_requested(params, value: bool, started: bool, v_ego: float, engaged: bool) -> None:
  if value and not can_enter(started, v_ego, engaged):
    return
  params.put_bool(MOONPILOT_OFFROAD_KEY, value, block=True)


def enabled(params, parked: bool) -> bool:
  return requested(params) or parked


def button_label(params, started: bool, parked: bool) -> str:
  if requested(params):
    return EXIT_LABEL
  if not started:
    return OFFROAD_LABEL
  return ENTER_LABEL if parked else NOT_PARKED_LABEL


def onroad_condition(params) -> bool:
  return not requested(params)
