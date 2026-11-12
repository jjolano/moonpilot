from __future__ import annotations

from collections.abc import Callable

import raylib_nb as rl

from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets import DialogResult, MousePos
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import ITEM_PADDING, ITEM_TEXT_VALUE_COLOR, ItemAction, ListItem, button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller
LABEL_BACK = "Back"
_CHEVRON_PATH = "icons/chevron_right.png"


class _ChevronAction(ItemAction):
  def __init__(self, value: str | Callable[[], str] = "", enabled: bool | Callable[[], bool] = True):
    super().__init__(0, enabled)
    self._value = value
    self._clicked = False
    self._chevron = gui_app.texture(_CHEVRON_PATH, 48, 48)

  def _render(self, rect):
    value = self._value() if callable(self._value) else self._value
    value_rect = rl.Rectangle(rect.x, rect.y, rect.width - self._chevron.width - 24, rect.height)
    if value:
      gui_label(value_rect, str(value), font_size=40, color=ITEM_TEXT_VALUE_COLOR, font_weight=FontWeight.NORMAL,
                alignment=TextAlignment.RIGHT, alignment_vertical=TextAlignmentVertical.MIDDLE)
    rl.draw_texture_ex(self._chevron,
                       rl.Vector2(rect.x + rect.width - self._chevron.width, rect.y + (rect.height - self._chevron.height) / 2),
                       0.0, 1.0, rl.WHITE if self.enabled else rl.Color(255, 255, 255, 100))
    clicked, self._clicked = self._clicked, False
    return clicked

  def _handle_mouse_release(self, _mouse_pos: MousePos):
    self._clicked = True


class _SubmenuItem(ListItem):
  def show_event(self):
    super().show_event()
    self._update_state()
    self._set_description_visible(True)

  def _update_state(self):
    previous_description = self._prev_description
    super()._update_state()
    if self.description_visible and self._prev_description != previous_description:
      self._rect.height = self.get_item_height(self._font, int(self._rect.width - ITEM_PADDING * 2))

  def set_parent_rect(self, parent_rect):
    old_width = self._rect.width
    super().set_parent_rect(parent_rect)
    if self.description_visible and self._rect.width != old_width:
      self._update_state()
      self._rect.height = self.get_item_height(self._font, int(self._rect.width - ITEM_PADDING * 2))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not self.is_visible:
      return
    if self.action_item and rl.check_collision_point_rec(rl.Vector2(mouse_pos.x, mouse_pos.y), self.get_right_item_rect(self._rect)):
      return
    if self.callback:
      self.callback()


def submenu_item(title, value, description: str | Callable[[], str] | None, callback, enabled=True) -> ListItem:
  row = _SubmenuItem(title=title, description=description, action_item=_ChevronAction(value, enabled), callback=callback)
  row.show_event()
  return row


class Page(Scroller):
  def __init__(self, rows: list, title: str, description: str = "", back: Callable | None = None):
    super().__init__([button_item(title, LABEL_BACK, description=description, callback=back), *rows], line_separator=True, spacing=0)


def confirm(text: str, label: str, action) -> None:
  def confirmed(result) -> None:
    if result == DialogResult.CONFIRM:  # Cancel calls back too, with DialogResult.CANCEL
      action()

  gui_app.push_widget(ConfirmDialog(text, label, rich=True, callback=confirmed))
