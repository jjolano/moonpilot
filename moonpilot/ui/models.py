"""Tizi model settings rendered inside Moonpilot's inline page stack.

The worker owns catalog, build and selection state. This module renders the model tasks and writes
the same request/selection parameters as the worker protocol. Choosing a model is the tree picker in
`moonpilot/ui/model_picker.py`, which is a dialog rather than a page: the catalog is a list of groups
now, and a driver picks from the whole list rather than from a paged copy of it.
"""

from collections.abc import Callable

import pyray as rl

from moonpilot import models
from moonpilot.ui.model_picker import ModelTreeDialog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import (
  ITEM_PADDING,
  ITEM_TEXT_VALUE_COLOR,
  ItemAction,
  ListItem,
  button_item,
  text_item,
)
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import MousePos

BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)
BAR_WIDTH = 620
INSTALLED_ROWS = 8
_CHEVRON_PATH = "icons/chevron_right.png"


def _offroad() -> bool:
  return ui_state.is_offroad()


class _ChevronAction(ItemAction):
  """A non-dialog row action that draws the standard right chevron."""

  def __init__(self, value: str | Callable[[], str] = "", enabled: bool | Callable[[], bool] = True):
    # Width 0 means full-row hitbox; value and chevron still draw at the rect's right edge.
    super().__init__(0, enabled)
    self._value = value
    self._clicked = False
    # Load after init_window so the texture is valid; per-instance, not module scope.
    self._chevron = gui_app.texture(_CHEVRON_PATH, 48, 48)
    self._font = gui_app.font(FontWeight.NORMAL)

  def _render(self, rect):
    value = self._value() if callable(self._value) else self._value
    value_rect = rl.Rectangle(rect.x, rect.y, rect.width - self._chevron.width - 24, rect.height)
    if value:
      gui_label(
        value_rect,
        str(value),
        font_size=40,
        color=ITEM_TEXT_VALUE_COLOR,
        font_weight=FontWeight.NORMAL,
        alignment=TextAlignment.RIGHT,
        alignment_vertical=TextAlignmentVertical.MIDDLE,
      )
    rl.draw_texture_ex(
      self._chevron,
      rl.Vector2(rect.x + rect.width - self._chevron.width, rect.y + (rect.height - self._chevron.height) / 2),
      0.0,
      1.0,
      rl.WHITE if self.enabled else rl.Color(255, 255, 255, 100),
    )
    clicked, self._clicked = self._clicked, False
    return clicked

  def _handle_mouse_release(self, _mouse_pos: MousePos):
    self._clicked = True


class _SubmenuItem(ListItem):
  """Submenu row whose description stays open across show_event resets."""

  def show_event(self):
    super().show_event()
    # Refresh dynamic descriptions before sizing so the open height fits current text.
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
    # Construction opens at the 600px base width; re-size once the panel width lands.
    if self.description_visible and self._rect.width != old_width:
      self._update_state()
      self._rect.height = self.get_item_height(self._font, int(self._rect.width - ITEM_PADDING * 2))

  def _handle_mouse_release(self, mouse_pos: MousePos):
    if not self.is_visible:
      return
    if self.action_item:
      action_rect = self.get_right_item_rect(self._rect)
      if rl.check_collision_point_rec(mouse_pos, action_rect):
        return
    if self.callback:
      self.callback()


def submenu_item(title, value, description: str | Callable[[], str] | None, callback, enabled=True) -> ListItem:
  row = _SubmenuItem(title=title, description=description, action_item=_ChevronAction(value, enabled), callback=callback)
  row.show_event()
  return row


class Page(Scroller):
  """A page inside Moonpilot's settings pane, always with an explicit leading Back row."""

  def __init__(self, rows: list, title: str, description: str = "", back: Callable | None = None):
    back_row = button_item(title, models.LABEL_BACK, description=description, callback=back)
    super().__init__([back_row, *rows], line_separator=True, spacing=0)


def _confirm(text: str, label: str, action) -> None:
  def confirmed(result) -> None:
    if result is not None:
      action()

  gui_app.push_widget(ConfirmDialog(text, label, rich=True, callback=confirmed))


class _JobAction(ItemAction):
  def __init__(self, params: Params):
    super().__init__(width=BAR_WIDTH)
    self._params = params

  def _render(self, rect):
    job = models.status(self._params)["job"]
    fraction = models.job_fraction(job)
    if fraction is None:
      gui_label(
        rect,
        models.job_text(job),
        font_size=40,
        color=ITEM_TEXT_VALUE_COLOR,
        font_weight=FontWeight.NORMAL,
        alignment=TextAlignment.RIGHT,
        alignment_vertical=TextAlignmentVertical.MIDDLE,
      )
      return False
    bar = rl.Rectangle(rect.x, rect.y + rect.height / 2 - 12, rect.width, 24)
    rl.draw_rectangle_rounded(bar, 1.0, 8, BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(bar.x, bar.y, bar.width * fraction, bar.height), 1.0, 8, BAR_FILL)
    gui_label(
      bar,
      models.job_progress_text(job),
      font_size=34,
      color=rl.WHITE,
      font_weight=FontWeight.MEDIUM,
      alignment=TextAlignment.CENTER,
      alignment_vertical=TextAlignmentVertical.MIDDLE,
    )
    return False


class ModelsLayout(Page):
  """The primary model tasks: the two pickers, downloaded models, jobs and storage."""

  def __init__(self, params: Params, open_page: Callable, back: Callable):
    self._params = params
    self._open_page = open_page
    self._back = back
    job = ListItem(title="job", description=lambda: models.job_description(self._status()["job"], self._params), action_item=_JobAction(self._params))
    job.set_visible(lambda: self._status()["job"] is not None)
    cancel = button_item(models.LABEL_CANCEL, models.LABEL_CANCEL, description=models.DESCRIPTION_CANCEL, callback=self._cancel)
    cancel.set_visible(lambda: models.job_active(self._status()["job"]))
    reboot = button_item(
      models.LABEL_REBOOT, models.LABEL_REBOOT, description=models.DESCRIPTION_REBOOT, callback=lambda: models.request_reboot(self._params), enabled=_offroad
    )
    reboot.set_visible(lambda: models.restart_needed(self._params))
    super().__init__(
      [
        submenu_item(
          models.TITLE_CURRENT_MODEL,
          lambda: models.model_label(self._params, models.DRIVING),
          models.DESCRIPTION_CURRENT_MODEL,
          lambda: self._pick(models.DRIVING),
        ),
        submenu_item(
          models.TITLE_MONITORING,
          lambda: models.model_label(self._params, models.MONITORING),
          lambda: models.model_description(self._params, models.MONITORING),
          lambda: self._pick(models.MONITORING),
        ),
        submenu_item(
          models.TITLE_INSTALLED,
          lambda: str(len(self._status()["installed"])),
          models.DESCRIPTION_INSTALLED,
          lambda: self._open_page(InstalledLayout(self._params, self._back)),
        ),
        button_item(
          models.TITLE_REFRESH,
          lambda: models.catalog_text(self._status()["catalog"]),
          description=lambda: models.catalog_description(self._status()["catalog"]),
          callback=lambda: models.request(self._params, "refresh"),
        ),
        job,
        cancel,
        reboot,
        text_item(models.TITLE_STORAGE, lambda: models.storage_text(self._status()["storage"]), description=models.DESCRIPTION_STORAGE),
      ],
      models.TITLE_MODELS,
      models.DESCRIPTION_MODELS,
      back,
    )

  def _pick(self, kind: str) -> None:
    """Open the tree picker for one kind. A row the device has not built is an install, not a
    selection, and the picker's own footer says so before the driver confirms."""
    gui_app.push_widget(ModelTreeDialog(self._params, kind, lambda selection: self._choose(kind, selection), _offroad))

  def _choose(self, kind: str, selection: str) -> None:
    if selection == models.desired(self._params, kind):
      return
    if not models.is_built(self._params, kind, selection):
      _confirm(models.INSTALL_TEXT, models.LABEL_INSTALL, lambda: models.request(self._params, "install", selection))
      return
    if not _offroad():
      return
    _confirm(models.CONFIRM_SELECT, models.LABEL_SELECT, lambda: models.select(self._params, kind, selection))

  def _cancel(self) -> None:
    if models.job_active(self._status()["job"]):
      _confirm(models.CONFIRM_CANCEL, models.LABEL_CANCEL, lambda: models.request(self._params, "cancel"))

  def _status(self) -> dict:
    return models.status(self._params)


class InstalledLayout(Page):
  """Downloaded model details/removal, with rebuild shown only for non-built entries."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    rows: list = []
    for index in range(INSTALLED_ROWS):
      info = button_item(
        lambda index=index: self._name(index),
        lambda index=index: self._state(index),
        description=lambda index=index: models.installed_detail(self._entry(index) or {}),
        callback=lambda: None,
      )
      remove = button_item(
        models.LABEL_REMOVE, models.LABEL_REMOVE, description=models.REMOVE_TEXT, callback=lambda index=index: self._remove(index), enabled=_offroad
      )
      rebuild = button_item(
        models.LABEL_REBUILD,
        models.LABEL_REBUILD,
        description=lambda index=index: models.installed_detail(self._entry(index) or {}),
        callback=lambda index=index: self._rebuild_action(index),
        enabled=_offroad,
      )
      for row in (info, remove, rebuild):
        row.set_visible(lambda index=index: self._entry(index) is not None)
      rebuild.set_visible(lambda index=index: (self._entry(index) or {}).get("state") != "built")
      rows += [info, remove, rebuild]
    super().__init__(rows, models.TITLE_INSTALLED, models.DESCRIPTION_INSTALLED, back)

  def _entry(self, index: int) -> dict | None:
    return models.page_item(models.status(self._params)["installed"], 0, INSTALLED_ROWS, index)

  def _name(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else str(entry.get("name") or models.selection_of(entry)[:12])

  def _state(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else str(entry.get("state", ""))

  def _remove(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      selection = models.selection_of(entry)
      _confirm(models.REMOVE_TEXT, models.LABEL_REMOVE, lambda: models.request(self._params, "remove", selection))

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None and entry.get("state") != "built":
      models.request(self._params, "install", models.selection_of(entry))


def row(params: Params, open_page: Callable, back: Callable) -> ListItem:
  return submenu_item(
    models.TITLE_MODELS,
    lambda: models.model_label(params, models.DRIVING),
    lambda: models.model_description(params, models.DRIVING),
    lambda: open_page(ModelsLayout(params, open_page, back)),
  )
