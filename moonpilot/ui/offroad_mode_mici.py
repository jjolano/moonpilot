"""Offroad mode's mici half: the panel row and the road view's hold gesture, in mici's own widgets.

`moonpilot/offroad.py` owns every decision and every string; this module only renders them. Kept
separate from `moonpilot/ui/offroad_mode.py` on purpose -- a fork override of a renamed upstream
method is a failure no merge can see, and each tree's widgets are its own.
"""

from moonpilot import offroad
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app


def parked() -> bool:
  """Whether the mode may be entered now, read off the UI state the panel and the road view share."""
  return offroad.can_enter(ui_state.started, ui_state.sm["carState"].vEgo, ui_state.engaged)


def row(params: Params) -> BigButton:
  button = BigButton("offroad mode", description=offroad.DESCRIPTION)
  button.set_click_callback(lambda: toggle(params))
  button.set_enabled(lambda: offroad.enabled(params, parked()))
  refresh(button, params)
  return button


def refresh(button: BigButton, params: Params) -> None:
  """Push the value line: `set_value` is not callable-resolved, unlike `set_enabled`, so the panel
  calls this every frame next to the tailscale row."""
  button.set_value(offroad.button_label(params, ui_state.started, parked()))


def toggle(params: Params) -> None:
  params.put_bool(offroad.MOONPILOT_OFFROAD_KEY, not offroad.requested(params), block=True)


def long_press() -> None:
  """The road view's hold: enter the mode behind a slide-to-confirm, only where the row is live.
  Entering only, like the tizi half: the hold can never leave the mode."""
  if not parked():
    return
  gui_app.push_widget(BigConfirmationDialog(offroad.SLIDE_LABEL, gui_app.texture("icons_mici/settings/device/power.png", 54, 64), enter))


def enter() -> None:
  ui_state.params.put_bool(offroad.MOONPILOT_OFFROAD_KEY, True, block=True)
