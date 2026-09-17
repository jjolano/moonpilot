"""moonpilot settings panel for mici. Same feature table as the tizi panel, but a disabled
widget here swallows its own long press, so the description dialog carrying the reason is
unreachable: the reason rides along as the always-visible sub-label instead."""

from moonpilot import tailscale
from moonpilot.features import FEATURES, Feature, available, wanted
from moonpilot.ui.tailscale_qr_mici import TailscaleSignInDialogMici
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


def _feature_button(feature: Feature):
  button = BigParamControl(feature.title, feature.key, description=feature.description)
  button.set_enabled(lambda f=feature: available(f) and (not f.offroad_only or ui_state.is_offroad()))
  return button


class MoonpilotLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._rows = tuple((feature, _feature_button(feature)) for feature in FEATURES)

    # The tailscale row carries its state in the value line rather than a second toggle: it is a
    # status readout whose only interaction is opening the sign-in dialog.
    self._tailscale = BigButton("tailscale")
    self._tailscale.set_click_callback(self._show_tailscale)

    self._scroller.add_widgets([*[button for _, button in self._rows], self._tailscale])
    self._update_rows()
    ui_state.add_offroad_transition_callback(self._update_rows)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    # mici's values are pushed, not callable-resolved, so the state needs re-reading every frame.
    self._tailscale.set_value(tailscale.status_text(self._params)[0])

  def _show_tailscale(self):
    # A QR while there is something to scan; otherwise the state detail, which does not fit the
    # value line. Built at click time so neither can be stale.
    if tailscale.auth_url(self._params):
      gui_app.push_widget(TailscaleSignInDialogMici())
      return
    detail = tailscale.status_text(self._params)[1]
    if detail:
      gui_app.push_widget(BigDialog("tailscale", detail))

  def _update_rows(self):
    # set_value is not callable-resolved, unlike set_enabled, so it needs pushing here.
    for feature, button in self._rows:
      button.set_value("" if available(feature) else "unavailable")
      # BigParamControl reads get_bool, which ignores the declared default, so set the pill
      # from the driver's preference the way the feature itself reads it.
      button.set_checked(wanted(feature, self._params))
