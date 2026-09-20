"""Moonpilot settings panel with inline feature and model pages.

Only dialogs leave this panel through the application navigation stack. Feature groups and model
pages stay in this panel's existing content rectangle and use their own leading Back row.
"""

from collections.abc import Callable

import pyray as rl

from moonpilot import tailscale
from moonpilot.engage import car_unavailable_reason
from moonpilot.features import FEATURES, Group, GROUPS, Feature, missing_modules, version, wanted
from moonpilot.ui import models as models_ui
from moonpilot.ui import offroad_mode
from moonpilot.ui.tailscale_qr import TailscaleSignInDialog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import MousePos, Widget
from openpilot.system.ui.widgets.list_view import ItemAction, ListItem, button_item, text_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller

CHEVRON = gui_app.texture("icons/chevron_right.png", 48, 48)


def _unavailable_reason(feature: Feature) -> str | None:
  modules = missing_modules(feature)
  if modules:
    return f"{feature.title} is currently unavailable until {', '.join(modules)} is installed, which happens automatically once the device is online."
  car = car_unavailable_reason(feature, ui_state.CP)
  return f"{feature.title} is unavailable on this car: {car}." if car else None


def _description(feature: Feature) -> str:
  reason = _unavailable_reason(feature)
  return feature.description if reason is None else "<b>" + reason + "</b><br><br>" + feature.description


def _feature_toggle(feature: Feature, params: Params):
  return toggle_item(
    feature.title,
    description=lambda f=feature: _description(f),
    initial_state=wanted(feature, params),
    callback=lambda state, key=feature.key: params.put_bool(key, state, block=True),
    enabled=lambda f=feature: _unavailable_reason(f) is None and (not f.offroad_only or ui_state.is_offroad()),
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
  from moonpilot import deps

  def retry():
    if deps.read_status(params)[0] == deps.WAITING_METERED:
      deps.request_retry(params)

  return button_item(
    "dependencies",
    lambda: deps.status_text(params)[0],
    description=lambda: deps.status_text(params)[1],
    callback=retry,
    enabled=lambda: deps.read_status(params)[0] == deps.WAITING_METERED,
  )


def _rollback_row(params: Params):
  from moonpilot import boot

  row = text_item("rollback", lambda: boot.rollback_text(params))
  row.set_visible(lambda: bool(boot.rollback_text(params)))
  return row


class _ChevronAction(ItemAction):
  WIDTH = 440

  def __init__(self, enabled=True):
    super().__init__(self.WIDTH, enabled)
    self._clicked = False

  def _render(self, rect):
    rl.draw_texture_ex(
      CHEVRON,
      rl.Vector2(rect.x + rect.width - CHEVRON.width, rect.y + (rect.height - CHEVRON.height) / 2),
      0.0,
      1.0,
      rl.WHITE if self.enabled else rl.Color(255, 255, 255, 100),
    )
    clicked, self._clicked = self._clicked, False
    return clicked

  def _handle_mouse_release(self, _mouse_pos: MousePos):
    self._clicked = True


def submenu_item(title, description: str, callback: Callable, enabled=True) -> ListItem:
  return ListItem(title=title, description=description, action_item=_ChevronAction(enabled), callback=callback)


class _PageStack(Widget):
  """A small inline page stack; it never touches the application navigation stack."""

  def __init__(self):
    super().__init__()
    self._pages: list[Widget] = []

  def set_root(self, page: Widget) -> None:
    self._pages = [page]

  def open(self, page: Widget) -> None:
    if self._pages:
      self._pages[-1].hide_event()
    self._pages.append(page)
    page.show_event()

  def back(self) -> None:
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


class GroupLayout(Widget):
  """One feature group's inline page."""

  def __init__(self, group: Group, params: Params, back: Callable):
    super().__init__()
    self._scroller = Scroller(
      [
        button_item(group.title, "BACK", description=group.description, callback=back),
        *(_feature_toggle(feature, params) for feature in group.features),
      ],
      line_separator=True,
      spacing=0,
    )

  def _render(self, rect):
    self._scroller.render(rect)

  def show_event(self):
    super().show_event()
    self._scroller.show_event()

  def hide_event(self):
    self._scroller.hide_event()
    super().hide_event()


class MoonpilotLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._stack = self._child(_PageStack())
    self._root = Scroller(
      [
        text_item("version", version()),
        models_ui.row(self._params, self._stack.open, self._stack.back),
        *[submenu_item(group.title, group.description, lambda group=group: self._open_group(group)) for group in GROUPS],
        offroad_mode.row(self._params),
        _tailscale_row(self._params),
        _dependencies_row(self._params),
        _rollback_row(self._params),
      ],
      line_separator=True,
      spacing=0,
    )
    self._stack.set_root(self._root)
    assert FEATURES, GROUPS

  def _open_group(self, group: Group):
    self._stack.open(GroupLayout(group, self._params, self._stack.back))

  def _render(self, rect):
    self._stack.render(rect)
