"""Mici model settings rendered as inline vertical pages inside Moonpilot settings."""

from collections.abc import Callable, Sequence

import pyray as rl

from moonpilot import models
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, SettingDescriptionDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller

JOB_TRACK_COLOR = rl.Color(57, 57, 57, 255)
JOB_FILL_COLOR = rl.Color(51, 171, 76, 255)
PAGE_ROWS = 6
INSTALLED_ROWS = 8
KIND_FILTERS = (models.FILTER_ALL, models.FILTER_DRIVING, models.FILTER_MONITORING)
ICON_INSTALL = "icons_mici/settings/software.png"
ICON_REMOVE = "icons_mici/settings/device/uninstall.png"
ICON_REBUILD = "icons_mici/settings/device/update.png"
CHEVRON = gui_app.texture("icons/chevron_right.png", 48, 48)
BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)
BAR_HEIGHT = 10


def _request(params: Params, op: str, selection: str = "") -> None:
  models.request(params, op, selection)


def _describe(title: str, text: str) -> None:
  gui_app.push_widget(SettingDescriptionDialog(title, text or ".", gui_app.texture(ICON_REBUILD, 64, 75)))


def _confirm(title: str, icon: str, action, *, red: bool = False) -> None:
  gui_app.push_widget(BigConfirmationDialog(title, gui_app.texture(icon, 64, 75), action, red=red))


class _SubmenuButton(BigButton):
  def __init__(self, text: str, value: str = "", *, description: str = ""):
    super().__init__(text, value, description=description)
    self._chevron = CHEVRON

  def _title_width_hint(self) -> int:
    return super()._title_width_hint() - self._chevron.width - 20

  def _subtitle_width_hint(self) -> int:
    return super()._subtitle_width_hint() - self._chevron.width - 20

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    rl.draw_texture_ex(
      self._chevron,
      (self._rect.x + self._rect.width - self.LABEL_HORIZONTAL_PADDING - self._chevron.width, btn_y + (self._rect.height - self._chevron.height) / 2),
      0.0,
      1.0,
      rl.Color(255, 255, 255, int(255 * 0.9)) if self.enabled else rl.Color(255, 255, 255, 100),
    )


def submenu_button(text: str, value: str = "", *, description: str = "") -> BigButton:
  return _SubmenuButton(text, value, description=description)


class Page(Scroller):
  """An inline vertical page with an explicit leading Back card."""

  def __init__(self, rows: Sequence[Widget], title: str, description: str, back: Callable):
    back_card = BigButton(title, models.LABEL_BACK, description=description)
    back_card.set_click_callback(back)
    super().__init__(horizontal=False)
    self._scroller.add_widgets([back_card, *rows])


class _JobCard(BigButton):
  def __init__(self, params: Params):
    super().__init__("job")
    self._params = params

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    if (fraction := models.job_fraction(models.status(self._params)["job"])) is None:
      return
    x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
    width = self._rect.width - 2 * self.LABEL_HORIZONTAL_PADDING
    y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING + 6
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width, BAR_HEIGHT), 1.0, 8, BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width * fraction, BAR_HEIGHT), 1.0, 8, BAR_FILL)


class ModelsPage(Page):
  """The primary model tasks."""

  def __init__(self, params: Params, open_page: Callable, back: Callable):
    self._params = params
    self._open_page = open_page
    self._current = submenu_button(models.TITLE_CURRENT_MODEL, description=models.DESCRIPTION_CURRENT_MODEL)
    self._current.set_click_callback(lambda: open_page(ChooserPage(params, models.DRIVING, back)))
    self._monitoring = submenu_button(models.TITLE_MONITORING, description=models.model_description(params, models.MONITORING))
    self._monitoring.set_click_callback(lambda: open_page(ChooserPage(params, models.MONITORING, back)))
    self._job = _JobCard(params)
    self._cancel = BigButton(models.LABEL_CANCEL, models.LABEL_CANCEL, description=models.DESCRIPTION_CANCEL)
    self._cancel.set_click_callback(self._cancel_action)
    self._job.set_click_callback(lambda: _describe("job", models.job_description(models.status(params)["job"], params)))
    self._reboot = BigButton(models.LABEL_REBOOT, models.LABEL_REBOOT, description=models.DESCRIPTION_REBOOT)
    self._reboot.set_click_callback(lambda: models.request_reboot(params))
    self._browse = submenu_button(models.TITLE_BROWSE, description=models.DESCRIPTION_BROWSE)
    self._browse.set_click_callback(lambda: open_page(CatalogPage(params, back)))
    self._installed = submenu_button(models.TITLE_INSTALLED, description=models.DESCRIPTION_INSTALLED)
    self._installed.set_click_callback(lambda: open_page(InstalledPage(params, back)))
    self._storage = BigButton(models.TITLE_STORAGE, description=models.DESCRIPTION_STORAGE)
    self._storage.set_click_callback(lambda: _describe(models.TITLE_STORAGE, models.DESCRIPTION_STORAGE))
    super().__init__(
      [self._current, self._monitoring, self._browse, self._installed, self._job, self._cancel, self._reboot, self._storage],
      models.TITLE_MODELS,
      models.DESCRIPTION_MODELS,
      back,
    )
    self._update_rows()

  def _cancel_action(self) -> None:
    if models.job_active(models.status(self._params)["job"]):
      _confirm(models.CONFIRM_CANCEL, ICON_REMOVE, lambda: _request(self._params, "cancel"), red=True)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _update_rows(self):
    status = models.status(self._params)
    self._current.set_value(models.model_label(self._params, models.DRIVING))
    self._monitoring.set_value(models.model_label(self._params, models.MONITORING))
    self._job.set_text(models.job_text(status["job"]) or "job")
    self._job.set_enabled(status["job"] is not None)
    self._cancel.set_visible(models.job_active(status["job"]))
    self._installed.set_value(str(len(status["installed"])))
    self._reboot.set_visible(models.restart_needed(self._params))
    self._reboot.set_enabled(ui_state.is_offroad())


class CatalogPage(Page):
  """The compatible marketplace, including refresh and paging."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    self._page = 0
    self._kind_toggle = BigMultiToggle("show", list(KIND_FILTERS))
    self._rows = [BigButton("") for _ in range(PAGE_ROWS)]
    self._refresh = BigButton(models.TITLE_REFRESH, description=models.DESCRIPTION_REFRESH)
    self._refresh.set_click_callback(lambda: _request(params, "refresh"))
    self._page_row = BigButton("catalog")
    self._older = BigButton("older", models.LABEL_OLDER)
    self._older.set_click_callback(lambda: self._offset(1))
    self._newer = BigButton("newer", models.LABEL_NEWER)
    self._newer.set_click_callback(lambda: self._offset(-1))
    super().__init__(
      [self._refresh, self._kind_toggle, *self._rows, self._page_row, self._older, self._newer], models.TITLE_BROWSE, models.DESCRIPTION_BROWSE, back
    )
    self._update_rows()

  def show_event(self):
    super().show_event()
    self._page = 0
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _entries(self) -> list[dict]:
    entries = [entry for entry in models.browse().get("entries", []) if entry.get("admitted")]
    kind = self._kind_toggle.get_value()
    return entries if kind == models.FILTER_ALL else [entry for entry in entries if entry.get("kind") == kind]

  def _pages(self) -> int:
    return max(1, (len(self._entries()) + PAGE_ROWS - 1) // PAGE_ROWS)

  def _entry_at(self, index: int) -> dict | None:
    entries = self._entries()
    position = self._page * PAGE_ROWS + index
    return entries[position] if position < len(entries) else None

  def _offset(self, delta: int) -> None:
    self._page = min(max(0, self._page + delta), self._pages() - 1)
    self._update_rows()

  def _install(self, index: int) -> None:
    entry = self._entry_at(index)
    if entry is None:
      return
    installed = {models.selection_of(item) for item in models.status(self._params)["installed"]}
    selection = models.selection_of(entry)
    if not entry.get("admitted") or selection in installed:
      _describe(str(entry.get("name", "")), models.entry_detail(entry))
      return
    _confirm(models.INSTALL_TEXT, ICON_INSTALL, lambda: _request(self._params, "install", selection))

  def _update_rows(self):
    self._page = min(self._page, self._pages() - 1)
    installed = {models.selection_of(entry) for entry in models.status(self._params)["installed"]}
    catalog = models.status(self._params)["catalog"]
    self._refresh.set_value(models.catalog_text(catalog))
    self._refresh.set_long_press_callback(lambda: _describe(models.TITLE_REFRESH, models.catalog_description(catalog)))
    for index, row in enumerate(self._rows):
      entry = self._entry_at(index)
      row.set_visible(entry is not None)
      if entry is None:
        continue
      row.set_text(str(entry.get("name", "")))
      row.set_value(models.entry_action(entry, installed))
      row.set_enabled(True)
      row.set_click_callback(lambda index=index: self._install(index))
    entries = self._entries()
    self._page_row.set_value(
      f"{self._page + 1} / {self._pages()}" if entries else (models.BROWSE_EMPTY if not models.browse().get("revision") else models.BROWSE_NONE)
    )
    self._older.set_visible(bool(entries))
    self._newer.set_visible(bool(entries))


class InstalledPage(Page):
  """Downloaded model details/removal, with conditional rebuild."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    self._blocks: list[tuple[BigButton, BigButton, BigButton]] = []
    widgets: list[Widget] = []
    for index in range(INSTALLED_ROWS):
      info = BigButton("")
      info.set_click_callback(lambda index=index: self._describe(index))
      remove = BigButton(models.LABEL_REMOVE, models.LABEL_REMOVE, description=models.REMOVE_TEXT)
      remove.set_click_callback(lambda index=index: self._remove_action(index))
      rebuild = BigButton(models.LABEL_REBUILD, models.LABEL_REBUILD, description=models.DESCRIPTION_STORAGE)
      rebuild.set_click_callback(lambda index=index: self._rebuild_action(index))
      self._blocks.append((info, remove, rebuild))
      widgets.extend((info, remove, rebuild))
    super().__init__(widgets, models.TITLE_INSTALLED, models.DESCRIPTION_INSTALLED, back)
    self._update_rows()

  def _entry(self, index: int) -> dict | None:
    entries = models.status(self._params)["installed"]
    return entries[index] if index < len(entries) else None

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _describe(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      _describe(str(entry.get("name") or models.selection_of(entry)[:12]), models.installed_detail(entry))

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None and entry.get("state") != "built":
      _request(self._params, "install", models.selection_of(entry))

  def _remove_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      selection = models.selection_of(entry)
      _confirm(models.REMOVE_TEXT, ICON_REMOVE, lambda: _request(self._params, "remove", selection), red=True)

  def _update_rows(self):
    offroad = ui_state.is_offroad()
    for index, (info, remove, rebuild) in enumerate(self._blocks):
      entry = self._entry(index)
      present = entry is not None
      info.set_visible(present)
      remove.set_visible(present)
      rebuild.set_visible(present and entry.get("state") != "built" if entry else False)
      info.set_enabled(True)
      remove.set_enabled(offroad)
      rebuild.set_enabled(offroad)
      if entry is None:
        continue
      info.set_text(str(entry.get("name") or models.selection_of(entry)[:12]))
      info.set_value(f"{entry.get('state', '')} {models.human_size(int(entry.get('size', 0)))}".strip())


class ChooserPage(Page):
  """One kind-filtered chooser: stock, built models, and admitted catalog models."""

  def __init__(self, params: Params, kind: str, back: Callable):
    self._params = params
    self._kind = kind
    self._back = back
    self._choices_cache = models.chooser_entries(self._params, self._kind)
    self._rows: list[BigButton] = []
    super().__init__([], models.TITLE_DRIVING if kind == models.DRIVING else models.TITLE_MONITORING, models.DESCRIPTION_MODELS, back)
    self._ensure_rows(max(1, len(self._choices_cache)))
    self._update_rows()

  def show_event(self):
    super().show_event()
    self._choices_cache = models.chooser_entries(self._params, self._kind)
    self._ensure_rows(max(1, len(self._choices_cache)))
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _choices(self) -> list[dict | None]:
    return self._choices_cache

  def _ensure_rows(self, count: int) -> None:
    while len(self._rows) < count:
      index = len(self._rows)
      row = BigButton("")
      row.set_click_callback(lambda index=index: self._select(index))
      self._rows.append(row)
      self._scroller.add_widgets([row])

  def _choice(self, index: int) -> dict | None:
    choices = self._choices()
    return choices[index] if index < len(choices) else None

  def _selection(self, index: int) -> str:
    choice = self._choice(index)
    return "" if choice is None else models.selection_of(choice)

  def _install(self, selection: str) -> None:
    _request(self._params, "install", selection)
    self._back()

  def _select(self, index: int) -> None:
    choices = self._choices()
    if index >= len(choices):
      return
    choice = choices[index]
    selection = "" if choice is None else models.selection_of(choice)
    if choice is not None and choice.get("state") != "built":
      _confirm(models.INSTALL_TEXT, ICON_INSTALL, lambda: self._install(selection))
      return
    if not ui_state.is_offroad() or selection == models.desired(self._params, self._kind):
      return
    _confirm(models.CONFIRM_SELECT, ICON_INSTALL, lambda: models.select(self._params, self._kind, selection))

  def _update_rows(self):
    desired = models.desired(self._params, self._kind)
    choices = self._choices()
    self._ensure_rows(max(1, len(choices)))
    offroad = ui_state.is_offroad()
    for index, row in enumerate(self._rows):
      choice = choices[index] if index < len(choices) else None
      row.set_visible(index < len(choices))
      row.set_enabled(offroad or choice is not None and choice.get("state") != "built")
      if index >= len(choices):
        continue
      selection = "" if choice is None else models.selection_of(choice)
      row.set_text("stock" if choice is None else str(choice.get("name") or selection[:12]))
      row.set_value(
        models.LABEL_INSTALL
        if choice is not None and choice.get("state") != "built"
        else models.REASON_SELECTED
        if selection == desired
        else models.LABEL_SELECT
      )
