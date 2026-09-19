"""moonpilot settings panel for mici. Same feature table as the tizi panel, but a disabled
widget here swallows its own long press, so the description dialog carrying the reason is
unreachable: the reason rides along as the always-visible sub-label instead."""

from moonpilot import tailscale
from moonpilot.engage import car_unavailable_reason
from moonpilot.features import FEATURES, Feature, available, wanted
from moonpilot.ui import offroad_mode_mici
from moonpilot.ui.tailscale_qr_mici import TailscaleSignInDialogMici
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller


def _feature_button(feature: Feature):
  button = BigParamControl(feature.title, feature.key, description=feature.description)
  # Both gates, the same pair the tizi panel resolves per render: a missing dependency and the
  # car itself. The value line only ever shows one of them (`_unavailable_value`).
  button.set_enabled(lambda f=feature: available(f) and car_unavailable_reason(f, ui_state.CP) is None and (not f.offroad_only or ui_state.is_offroad()))
  return button


def _unavailable_value(feature: Feature) -> str:
  """The value line for the row: the reason it cannot run, or empty while it can.

  mici swallows a disabled widget's long press, so the description dialog carrying the reason
  is unreachable and the reason has to ride the always-visible sub-label. The car's own reason
  where there is one, the dependency placeholder otherwise, and nothing at all when the row is
  live.
  """
  if reason := car_unavailable_reason(feature, ui_state.CP):
    return reason
  return "" if available(feature) else "unavailable"


class MoonpilotLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._rows = tuple((feature, _feature_button(feature)) for feature in FEATURES)
    self._cp = ui_state.CP

    # The tailscale row carries its state in the value line rather than a second toggle: it is a
    # status readout whose only interaction is opening the sign-in dialog.
    self._tailscale = BigButton("tailscale")
    self._tailscale.set_click_callback(self._show_tailscale)

    # Same shape, and the same reason the value line is pushed: offroad mode is a device action
    # whose label the manager and the ignition edge can change under the panel's feet.
    self._offroad = offroad_mode_mici.row(self._params)

    self._scroller.add_widgets([*[button for _, button in self._rows], self._offroad, self._tailscale])
    self._update_rows()
    ui_state.add_offroad_transition_callback(self._update_rows)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    # mici's values are pushed, not callable-resolved, so the state needs re-reading every frame.
    self._tailscale.set_value(tailscale.status_text(self._params)[0])
    offroad_mode_mici.refresh(self._offroad, self._params)
    # The car gate the value line carries is pushed too, and it can change under the panel: driven
    # off a different object rather than re-reading nine params every frame, because `ui_state`
    # re-parses CarParams on its own parameter thread and a new object is what that looks like.
    if ui_state.CP is not self._cp:
      self._cp = ui_state.CP
      self._update_rows()

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
      button.set_value(_unavailable_value(feature))
      # BigParamControl reads get_bool, which ignores the declared default, so set the pill
      # from the driver's preference the way the feature itself reads it.
      button.set_checked(wanted(feature, self._params))
