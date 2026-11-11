"""Offroad mode's tizi half: the panel row and the road view's hold gesture.

The decision is `moonpilot/offroad.py`'s, so this file is only what tizi needs to render it and to
ask: the row the panel appends (`moonpilot/ui/settings.py`) and the confirm dialog behind a hold on
the road view (`openpilot/selfdrive/ui/onroad/augmented_road_view.py`). `parked()` is the one
reading of the UI state both call, which is what keeps the row and the gesture in agreement.
"""

from moonpilot import offroad
from moonpilot.ui.panel_common import enter, parked, toggle
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.list_view import ListItem, button_item




def row(params: Params) -> ListItem:
  # A button, not a toggle: the param is also cleared outside this panel (the ignition edge and the
  # manager's start-up clear), and `button_text` is re-resolved on every render where a toggle's pill
  # is set once at construction -- a claim this row could not keep.
  return button_item(
    "offroad mode",
    lambda: offroad.button_label(params, ui_state.started, parked()),
    description=offroad.DESCRIPTION,
    callback=lambda: toggle(params),
    enabled=lambda: offroad.enabled(params, parked()),
  )




def long_press() -> None:
  """The road view's hold: enter the mode behind a confirmation, only where the row is live.
  Entering only -- the hold can never leave the mode, so it cannot switch a device that is already
  offroad back onroad by accident."""
  if not parked():
    return
  gui_app.push_widget(ConfirmDialog(offroad.CONFIRM_TEXT, "Go offroad", callback=lambda result: enter() if result == DialogResult.CONFIRM else None))


