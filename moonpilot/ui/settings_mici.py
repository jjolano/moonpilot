"""moonpilot settings panel for mici. Same feature table as the tizi panel, but a disabled
widget here swallows its own long press, so the description dialog carrying the reason is
unreachable: the reason rides along as the always-visible sub-label instead.

The panel is the root -- what the device is running, one entry per group, and the two device rows --
and each group is a pushed page of the same feature rows the flat panel used to carry.
"""

from moonpilot import models, tailscale
from moonpilot.engage import car_unavailable_reason
from moonpilot.features import GROUPS, Group, Feature, available, wanted
from moonpilot.ui import models_mici, offroad_mode_mici
from moonpilot.ui.tailscale_qr_mici import TailscaleSignInDialogMici
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigParamControl
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
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

def _dependencies_row(params: Params):
  from moonpilot import deps

  row = BigButton("dependencies", deps.status_text(params)[0])

  def activate():
    state = deps.read_status(params)[0]
    if state == deps.WAITING_METERED:
      deps.request_retry(params)
    elif detail := deps.status_text(params)[1]:
      gui_app.push_widget(BigDialog("dependencies", detail))

  row.set_click_callback(activate)
  return row


def _rollback_row(params: Params):
  from moonpilot import boot

  row = BigButton("rollback", boot.rollback_text(params))
  row.set_visible(lambda: bool(boot.rollback_text(params)))
  return row


class _FeatureRows:
  """The rows of one group, with the pushing mici needs.

  mici's `set_value` and `set_checked` are pushed rather than resolved, so the state is re-read
  every frame, and two things can change it from outside the page: the car gate can change under
  the panel, and openpilot can go offroad while it is open. The first is detected by `ui_state.CP`
  being a different *object* (`ui_state` re-parses `CarParams` on its own parameter thread), the
  second by the offroad-transition callback the panel registers -- the same loop upstream's mici
  developer panel uses.
  """

  def __init__(self, features: tuple[Feature, ...]):
    self._params: Params = ui_state.params
    self._rows: tuple[tuple[Feature, BigParamControl], ...] = tuple((feature, _feature_button(feature)) for feature in features)
    self._cp = ui_state.CP
    ui_state.add_offroad_transition_callback(self._update_rows)
    self._update_rows()

  @property
  def widgets(self) -> list[Widget]:
    return [button for _feature, button in self._rows]

  def show_event(self) -> None:
    self._update_rows()

  def update_state(self) -> None:
    if ui_state.CP is not self._cp:
      self._cp = ui_state.CP
      self._update_rows()

  def _update_rows(self):
    # set_value is not callable-resolved, unlike set_enabled, so it needs pushing here.
    for feature, button in self._rows:
      button.set_value(_unavailable_value(feature))
      # BigParamControl reads get_bool, which ignores the declared default, so set the pill
      # from the driver's preference the way the feature itself reads it.
      button.set_checked(wanted(feature, self._params))


class GroupPageMici(NavScroller):
  """One group's page: the group's rows, and nothing else -- mici dismisses by swiping down, so a
  back row would be a second, redundant way out."""

  def __init__(self, group: Group):
    super().__init__()
    self._group = group
    self._rows = _FeatureRows(group.features)
    self._scroller.add_widgets(self._rows.widgets)

  def show_event(self):
    super().show_event()
    self._rows.show_event()

  def _update_state(self):
    super()._update_state()
    self._rows.update_state()


class MoonpilotLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params

    self._models = models_mici.ModelsPage(self._params)
    self._models_button = BigButton("models", description="Models from the openmodels catalog: what this car drives with and how it watches you.")
    self._models_button.set_click_callback(lambda: gui_app.push_widget(self._models))

    self._group_buttons: list[tuple[Group, BigButton]] = []
    for group in GROUPS:
      button = BigButton(group.title, description=group.description)
      button.set_click_callback(lambda group=group: gui_app.push_widget(GroupPageMici(group)))
      self._group_buttons.append((group, button))

    # The tailscale row carries its state in the value line rather than a second toggle: it is a
    # status readout whose only interaction is opening the sign-in dialog.
    self._tailscale = BigButton("tailscale")
    self._tailscale.set_click_callback(self._show_tailscale)

    # Same shape, and the same reason the value line is pushed: offroad mode is a device action
    # whose label the manager and the ignition edge can change under the panel's feet.
    self._offroad = offroad_mode_mici.row(self._params)

    self._dependencies = _dependencies_row(self._params)
    self._rollback = _rollback_row(self._params)

    self._scroller.add_widgets([self._models_button, *[button for _group, button in self._group_buttons], self._offroad, self._tailscale, self._dependencies, self._rollback])
    self._update_rows()
    ui_state.add_offroad_transition_callback(self._update_rows)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    # mici's values are pushed, not callable-resolved, so the state needs re-reading every frame.
    self._update_rows()

  def _update_rows(self):
    self._models_button.set_value(models.model_label(self._params, models.DRIVING))
    self._tailscale.set_value(tailscale.status_text(self._params)[0])
    from moonpilot import boot, deps
    self._dependencies.set_value(deps.status_text(self._params)[0])
    self._rollback.set_value(boot.rollback_text(self._params))
    offroad_mode_mici.refresh(self._offroad, self._params)

  def _show_tailscale(self):
    # A QR while there is something to scan; otherwise the state detail, which does not fit the
    # value line. Built at click time so neither can be stale.
    if tailscale.auth_url(self._params):
      gui_app.push_widget(TailscaleSignInDialogMici())
      return
    detail = tailscale.status_text(self._params)[1]
    if detail:
      gui_app.push_widget(BigDialog("tailscale", detail))
