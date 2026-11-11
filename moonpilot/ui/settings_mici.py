"""Moonpilot settings panel for mici with inline vertical feature/model pages."""


from moonpilot import boot, deps, modelprotocols, modelview, tailscale
from moonpilot.features import GROUPS, Group, Feature, car_reason, choice, missing_modules, wanted
from moonpilot.ui import models_mici, offroad_mode_mici, panel_common, widgets_mici
from moonpilot.ui.tailscale_qr_mici import TailscaleSignInDialogMici
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiParamToggle, BigParamControl
from openpilot.selfdrive.ui.mici.widgets.dialog import BigDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from moonpilot.ui.panel_common import PageStack
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller


def _feature_button(feature: Feature, parked_only: bool):
  if feature.choices:
    button: BigParamControl | BigMultiParamToggle = BigMultiParamToggle(feature.title, feature.key, list(feature.labels), description=feature.description)
  else:
    button = BigParamControl(feature.title, feature.key, description=feature.description)
  button.set_enabled(lambda: panel_common.feature_enabled(feature, parked_only))
  return button


def _unavailable_value(feature: Feature) -> str:
  if missing_modules(feature):
    return "unavailable"
  return car_reason(feature, ui_state.CP) or ""


def _dependencies_row(params: Params):
  row = BigButton("dependencies", deps.status_text(params)[0])

  def activate():
    if not deps.request_retry(params) and (detail := deps.status_text(params)[1]):
      gui_app.push_widget(BigDialog("dependencies", detail))

  row.set_click_callback(activate)
  return row


def _rollback_row(params: Params):

  row = BigButton("rollback", boot.rollback_text(params))
  row.set_visible(lambda: bool(boot.rollback_text(params)))
  return row




class _FeatureRows:
  def __init__(self, group: Group):
    self._params: Params = ui_state.params
    self._rows = tuple((feature, _feature_button(feature, group.parked_only)) for feature in group.features)
    self._cp = ui_state.CP
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
    for feature, button in self._rows:
      # BigParamControl reads get_bool, which ignores the declared default: push the registry's answer.
      reason = _unavailable_value(feature)
      if feature.choices:
        button.set_value(reason or feature.labels[choice(feature, self._params)])
        continue
      button.set_value(reason)
      button.set_checked(wanted(feature, self._params))




class MoonpilotLayoutMici(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._stack = self._child(PageStack())
    self._models_button = widgets_mici.SubmenuButton(
      modelview.TITLE_MODELS,
      modelview.model_label(self._params, modelprotocols.DRIVING),
      description=modelview.DESCRIPTION_MODELS,
    )
    self._models_button.set_click_callback(self._open_models)

    self._group_buttons: list[tuple[Group, BigButton]] = []
    for group in GROUPS:
      button = widgets_mici.SubmenuButton(group.title, description=group.description)
      button.set_click_callback(lambda group=group: self._open_group(group))
      self._group_buttons.append((group, button))

    self._tailscale = BigButton("tailscale")
    self._tailscale.set_click_callback(self._show_tailscale)
    self._offroad = offroad_mode_mici.row(self._params)
    self._dependencies = _dependencies_row(self._params)
    self._rollback = _rollback_row(self._params)
    self._root = Scroller(horizontal=False)
    self._root._scroller.add_widgets(  # system Scroller exposes its child for dynamic pages
      [self._models_button, *[button for _group, button in self._group_buttons], self._offroad, self._tailscale, self._dependencies, self._rollback]
    )
    self._stack.set_root(self._root)
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _render(self, rect):
    self._stack.render(rect)

  def _open_models(self):
    self._stack.open(models_mici.ModelsPage(self._params, self._stack.open, self._stack.back))

  def _open_group(self, group: Group):
    rows = _FeatureRows(group)
    page = widgets_mici.Page(rows.widgets, group.title, group.description, self._stack.back, rows.update_state)
    self._stack.open(page)

  def _update_rows(self):
    self._models_button.set_value(modelview.model_label(self._params, modelprotocols.DRIVING))
    self._tailscale.set_value(tailscale.status_text(self._params)[0])

    self._dependencies.set_value(deps.status_text(self._params)[0])
    self._rollback.set_value(boot.rollback_text(self._params))
    offroad_mode_mici.refresh(self._offroad, self._params)

  def _show_tailscale(self):
    if tailscale.auth_url(self._params):
      gui_app.push_widget(TailscaleSignInDialogMici())
      return
    detail = tailscale.status_text(self._params)[1]
    if detail:
      gui_app.push_widget(BigDialog("tailscale", detail))
