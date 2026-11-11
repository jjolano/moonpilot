"""Moonpilot settings panel with inline feature and model pages.

Only dialogs leave this panel through the application navigation stack. Feature groups and model
pages stay in this panel's existing content rectangle and use their own leading Back row.
"""


from moonpilot import boot, deps, tailscale
from moonpilot.features import Group, GROUPS, Feature, car_reason, choice, missing_modules, version, wanted
from moonpilot.ui import models as models_ui, widgets
from moonpilot.ui import offroad_mode
from moonpilot.ui.tailscale_qr import TailscaleSignInDialog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from moonpilot.ui import panel_common
from moonpilot.ui.panel_common import PageStack
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import button_item, multiple_button_item, text_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


def _unavailable_reason(feature: Feature) -> str | None:
  modules = missing_modules(feature)
  if modules:
    return f"{feature.title} is currently unavailable until {', '.join(modules)} is installed, which happens automatically once the device is online."
  car = car_reason(feature, ui_state.CP)
  return f"{feature.title} is unavailable on this car: {car}." if car else None


def _description(feature: Feature) -> str:
  reason = _unavailable_reason(feature)
  return feature.description if reason is None else "<b>" + reason + "</b><br><br>" + feature.description


def _feature_row(feature: Feature, params: Params, parked_only: bool):
  def enabled():
    return panel_common.feature_enabled(feature, parked_only)

  if feature.choices:
    row = multiple_button_item(
      feature.title,
      lambda: _description(feature),
      buttons=list(feature.labels),
      selected_index=choice(feature, params),
      button_width=200,
      callback=lambda index: params.put(feature.key, index, block=True),
    )
    row.action_item.set_enabled(enabled)
    return row
  return toggle_item(
    feature.title,
    description=lambda: _description(feature),
    initial_state=wanted(feature, params),
    callback=lambda state: params.put_bool(feature.key, state, block=True),
    enabled=enabled,
  )


def _tailscale_row(params: Params):
  return button_item(
    "tailscale",
    lambda: "SIGN IN" if tailscale.auth_url(params) else tailscale.status_text(params)[0].upper(),
    description=lambda: tailscale.status_text(params)[1],
    callback=lambda: gui_app.push_widget(TailscaleSignInDialog()),
    enabled=lambda: bool(tailscale.auth_url(params)),
  )


def _dependencies_row(params: Params):

  def retry():
    deps.request_retry(params)

  return button_item(
    "dependencies",
    lambda: deps.status_text(params)[0],
    description=lambda: deps.status_text(params)[1],
    callback=retry,
    enabled=lambda: deps.can_retry(params),
  )


def _rollback_row(params: Params):

  row = text_item("rollback", lambda: boot.rollback_text(params))
  row.set_visible(lambda: bool(boot.rollback_text(params)))
  return row






class MoonpilotLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._stack = self._child(PageStack())
    self._root = Scroller(
      [
        text_item("version", version()),
        models_ui.row(self._params, self._stack.open, self._stack.back),
        *[widgets.submenu_item(group.title, "", group.description, lambda group=group: self._open_group(group)) for group in GROUPS],
        offroad_mode.row(self._params),
        _tailscale_row(self._params),
        _dependencies_row(self._params),
        _rollback_row(self._params),
      ],
      line_separator=True,
      spacing=0,
    )
    self._stack.set_root(self._root)

  def _open_group(self, group: Group):
    rows = [_feature_row(feature, self._params, group.parked_only) for feature in group.features]
    self._stack.open(widgets.Page(rows, group.title, group.description, self._stack.back))

  def _render(self, rect):
    self._stack.render(rect)
