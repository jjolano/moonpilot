"""Moonpilot settings panel for mici with inline vertical feature/model pages."""

from collections.abc import Callable

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
from openpilot.system.ui.widgets.scroller import Scroller


def _feature_button(feature: Feature):
  button = BigParamControl(feature.title, feature.key, description=feature.description)
  button.set_enabled(lambda f=feature: available(f) and car_unavailable_reason(f, ui_state.CP) is None and (not f.offroad_only or ui_state.is_offroad()))
  return button


def _unavailable_value(feature: Feature) -> str:
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


class _PageStack(Widget):
  def __init__(self):
    super().__init__()
    self._pages: list[Widget] = []

  def set_root(self, page: Widget):
    self._pages = [page]

  def open(self, page: Widget):
    if self._pages:
      self._pages[-1].hide_event()
    self._pages.append(page)
    page.show_event()

  def back(self):
    if len(self._pages) <= 1:
      return
    self._pages.pop().hide_event()
    self._pages[-1].show_event()

  def show_event(self):
    super().show_event()
    if self._pages:
      self._pages[-1].show_event()

  def hide_event(self):
    if self._pages:
      self._pages[-1].hide_event()
    super().hide_event()

  def _render(self, rect):
    if self._pages:
      self._pages[-1].render(rect)


class _FeatureRows:
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
    for feature, button in self._rows:
      button.set_value(_unavailable_value(feature))
      button.set_checked(wanted(feature, self._params))


class GroupPageMici(Scroller):
  """One feature group's inline page, with an explicit leading Back card."""

  def __init__(self, group: Group, back: Callable):
    self._rows = _FeatureRows(group.features)
    back_card = BigButton(group.title, models.LABEL_BACK, description=group.description)
    back_card.set_click_callback(back)
    super().__init__(horizontal=False)
    self._scroller.add_widgets([back_card, *self._rows.widgets])

  def show_event(self):
    super().show_event()
    self._rows.show_event()

  def _update_state(self):
    super()._update_state()
    self._rows.update_state()


class MoonpilotLayoutMici(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._stack = self._child(_PageStack())
    self._models_button = models_mici.submenu_button(
      "models",
      models.model_label(self._params, models.DRIVING),
      description="Models from the openmodels catalog: what this car drives with and how it watches you.",
    )
    self._models_button.set_click_callback(self._open_models)

    self._group_buttons: list[tuple[Group, BigButton]] = []
    for group in GROUPS:
      button = models_mici.submenu_button(group.title, description=group.description)
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
    ui_state.add_offroad_transition_callback(self._update_rows)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _render(self, rect):
    self._stack.render(rect)

  def _open_models(self):
    self._stack.open(models_mici.ModelsPage(self._params, self._stack.open, self._stack.back))

  def _open_group(self, group: Group):
    self._stack.open(GroupPageMici(group, self._stack.back))

  def _update_rows(self):
    self._models_button.set_value(models.model_label(self._params, models.DRIVING))
    self._tailscale.set_value(tailscale.status_text(self._params)[0])
    from moonpilot import boot, deps

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
