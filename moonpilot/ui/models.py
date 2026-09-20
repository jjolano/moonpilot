"""Tizi model settings rendered inside Moonpilot's inline page stack.

The worker owns catalog, build and selection state. This module only renders the model tasks and
writes the same request/selection parameters as the worker protocol.
"""

from collections.abc import Callable
from typing import cast

import pyray as rl

from moonpilot import models
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import (
  ITEM_TEXT_VALUE_COLOR,
  ItemAction,
  ListItem,
  button_item,
  multiple_button_item,
  text_item,
)
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import MousePos

BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)
BAR_WIDTH = 620
PAGE_ROWS = 8
INSTALLED_ROWS = 8
KIND_FILTERS = (models.FILTER_ALL, models.FILTER_DRIVING, models.FILTER_MONITORING)
CHEVRON = gui_app.texture("icons/chevron_right.png", 48, 48)
_INVALID = object()


def _offroad() -> bool:
  return ui_state.is_offroad()


class _ChevronAction(ItemAction):
  """A non-dialog row action that draws the standard right chevron."""

  WIDTH = 520

  def __init__(self, value: str | Callable[[], str] = "", enabled: bool | Callable[[], bool] = True):
    super().__init__(self.WIDTH, enabled)
    self._value = value
    self._clicked = False
    self._font = gui_app.font(FontWeight.NORMAL)

  def _render(self, rect):
    value = self._value() if callable(self._value) else self._value
    value_rect = rl.Rectangle(rect.x, rect.y, rect.width - CHEVRON.width - 24, rect.height)
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


def submenu_item(title, value, description: str | Callable[[], str] | None, callback, enabled=True) -> ListItem:
  return ListItem(title=title, description=description, action_item=_ChevronAction(value, enabled), callback=callback)


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
  """The primary model tasks: selection, marketplace, downloaded models, jobs and storage."""

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
          lambda: self._open_page(ChooserLayout(self._params, models.DRIVING, self._back)),
        ),
        submenu_item(
          models.TITLE_MONITORING,
          lambda: models.model_label(self._params, models.MONITORING),
          lambda: models.model_description(self._params, models.MONITORING),
          lambda: self._open_page(ChooserLayout(self._params, models.MONITORING, self._back)),
        ),
        submenu_item(models.TITLE_BROWSE, "", models.DESCRIPTION_BROWSE, lambda: self._open_page(CatalogLayout(self._params, self._back))),
        submenu_item(
          models.TITLE_INSTALLED,
          lambda: str(len(self._status()["installed"])),
          models.DESCRIPTION_INSTALLED,
          lambda: self._open_page(InstalledLayout(self._params, self._back)),
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

  def _cancel(self) -> None:
    if models.job_active(self._status()["job"]):
      _confirm(models.CONFIRM_CANCEL, models.LABEL_CANCEL, lambda: models.request(self._params, "cancel"))

  def _status(self) -> dict:
    return models.status(self._params)


class CatalogLayout(Page):
  """The compatible marketplace, including catalog refresh and paging."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    self._kind = 0
    self._page = 0
    self._catalog = models.browse().get("revision", "") != ""
    rows: list = [
      button_item(
        models.TITLE_REFRESH,
        lambda: models.catalog_text(self._status()["catalog"]),
        description=lambda: models.catalog_description(self._status()["catalog"]),
        callback=lambda: models.request(self._params, "refresh"),
      ),
      multiple_button_item("show", "", list(KIND_FILTERS), 0, callback=self._set_kind),
    ]
    rows += [
      button_item(
        lambda index=index: self._title(index),
        lambda index=index: self._verdict(index),
        description=lambda index=index: self._detail(index),
        callback=lambda index=index: self._install(index),
        enabled=lambda index=index: self._installable(index),
      )
      for index in range(PAGE_ROWS)
    ]
    for row in rows[2:]:
      row.set_visible(lambda row=row: self._entry_of(row) is not None)
    rows += [
      text_item("page", lambda: self._page_text(), description=models.DESCRIPTION_BROWSE),
      button_item("older", models.LABEL_OLDER, callback=lambda: self._offset(1), enabled=lambda: self._pages() > 1),
      button_item("newer", models.LABEL_NEWER, callback=lambda: self._offset(-1), enabled=lambda: self._pages() > 1),
    ]
    self._rows = rows[2 : 2 + PAGE_ROWS]
    super().__init__(rows, models.TITLE_BROWSE, models.DESCRIPTION_BROWSE, back)

  def _status(self) -> dict:
    return models.status(self._params)

  def _entries(self) -> list[dict]:
    entries = [entry for entry in models.browse().get("entries", []) if entry.get("admitted")]
    kind = KIND_FILTERS[self._kind]
    return entries if kind == models.FILTER_ALL else [entry for entry in entries if entry.get("kind") == kind]

  def _pages(self) -> int:
    return max(1, (len(self._entries()) + PAGE_ROWS - 1) // PAGE_ROWS)

  def _index_of(self, row) -> int:
    return self._rows.index(row)

  def _entry_of(self, row) -> dict | None:
    entries = self._entries()
    position = self._page * PAGE_ROWS + self._index_of(row)
    return entries[position] if position < len(entries) else None

  def _installed(self) -> set[str]:
    return {models.selection_of(entry) for entry in self._status()["installed"]}

  def _title(self, index: int) -> str:
    entry = self._entry_at(index)
    return str(entry.get("name", "")) if entry else ""

  def _verdict(self, index: int) -> str:
    entry = self._entry_at(index)
    return models.entry_action(entry, self._installed()) if entry else ""

  def _detail(self, index: int) -> str:
    entry = self._entry_at(index)
    return models.entry_detail(entry) if entry else ""

  def _installable(self, index: int) -> bool:
    entry = self._entry_at(index)
    return entry is not None and models.selection_of(entry) not in self._installed()

  def _entry_at(self, index: int) -> dict | None:
    entries = self._entries()
    position = self._page * PAGE_ROWS + index
    return entries[position] if position < len(entries) else None

  def _page_text(self) -> str:
    if not self._entries():
      return models.BROWSE_EMPTY if not self._catalog else models.BROWSE_NONE
    return f"{self._page + 1} / {self._pages()}"

  def _set_kind(self, index: int) -> None:
    self._kind, self._page = index, 0

  def _offset(self, delta: int) -> None:
    self._page = min(max(0, self._page + delta), self._pages() - 1)

  def _install(self, index: int) -> None:
    entry = self._entry_at(index)
    if entry is None or not self._installable(index):
      return
    _confirm(models.INSTALL_TEXT, models.LABEL_INSTALL, lambda: models.request(self._params, "install", models.selection_of(entry)))


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
    entries = models.status(self._params)["installed"]
    return entries[index] if index < len(entries) else None

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


class ChooserLayout(Page):
  """One kind-filtered chooser: stock, built models and admitted catalog models."""

  def __init__(self, params: Params, kind: str, back: Callable):
    self._params = params
    self._kind = kind
    self._back = back
    self._choices_cache = models.chooser_entries(self._params, self._kind)
    choices = self._choices_cache
    rows = [
      button_item(
        lambda index=index: self._name(index),
        lambda index=index: self._action(index),
        description=lambda index=index: self._detail(index),
        callback=lambda index=index: self._select(index),
        enabled=lambda index=index: self._row_enabled(index),
      )
      for index in range(len(choices))
    ]
    self._rows = rows
    for index, row in enumerate(rows):
      row.set_visible(lambda index=index: self._choice(index) is not _INVALID)
    super().__init__(rows, models.TITLE_DRIVING if kind == models.DRIVING else models.TITLE_MONITORING, models.DESCRIPTION_MODELS, back)

  def _choices(self) -> list[dict | None]:
    return self._choices_cache

  def _choice(self, index: int) -> dict | None | object:
    return self._choices_cache[index] if index < len(self._choices_cache) else _INVALID

  def _catalog_only(self, index: int) -> bool:
    choice = self._choice(index)
    return isinstance(choice, dict) and choice.get("state") != "built"

  def _row_enabled(self, index: int) -> bool:
    return _offroad() if not self._catalog_only(index) else True

  def _selection(self, index: int) -> str:
    choice = self._choice(index)
    return "" if choice is None or choice is _INVALID else models.selection_of(cast(dict, choice))

  def _name(self, index: int) -> str:
    choice = self._choice(index)
    if choice is _INVALID:
      return ""
    return "stock" if choice is None else str(choice.get("name") or models.selection_of(cast(dict, choice))[:12])

  def _detail(self, index: int) -> str:
    choice = self._choice(index)
    if choice is _INVALID:
      return ""
    if choice is None:
      return models.model_description(self._params, self._kind)
    return models.installed_detail(cast(dict, choice)) if choice.get("state") == "built" else models.entry_detail(cast(dict, choice))

  def _action(self, index: int) -> str:
    if self._catalog_only(index):
      return models.LABEL_INSTALL
    return models.REASON_SELECTED if self._selection(index) == models.desired(self._params, self._kind) else models.LABEL_SELECT

  def _select(self, index: int) -> None:
    choice = self._choice(index)
    if choice is _INVALID:
      return
    if self._catalog_only(index):
      selection = models.selection_of(cast(dict, choice))
      _confirm(models.INSTALL_TEXT, models.LABEL_INSTALL, lambda: self._install(selection))
      return
    if not _offroad():
      return
    selection = self._selection(index)
    if selection == models.desired(self._params, self._kind):
      return
    _confirm(models.CONFIRM_SELECT, models.LABEL_SELECT, lambda: models.select(self._params, self._kind, selection))

  def _install(self, selection: str) -> None:
    models.request(self._params, "install", selection)
    self._back()


def row(params: Params, open_page: Callable, back: Callable) -> ListItem:
  return submenu_item(
    models.TITLE_MODELS,
    lambda: models.model_label(params, models.DRIVING),
    lambda: models.model_description(params, models.DRIVING),
    lambda: open_page(ModelsLayout(params, open_page, back)),
  )
