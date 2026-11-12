"""Mici model settings rendered as inline vertical pages inside Moonpilot settings.

The picker is a drill-down rather than a tree, because a mici screen is a column of cards: the
groups first, then one group's models. The grouping, names and stars are `moonpilot/modelview.py`'s
— the same ones the tizi tree dialog draws — and a star here is a long press, since there is no room
for a star glyph beside a card's value.
"""

from collections.abc import Callable

import raylib_nb as rl

from moonpilot import modelprotocols, modelstatus, models, modelview
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from moonpilot.ui import panel_common, widgets_mici

ICON_REMOVE = "icons_mici/settings/device/uninstall.png"
ICON_REBUILD = "icons_mici/settings/device/update.png"
ICON_INSTALL = "icons_mici/settings/software.png"
FAVORITE_MARK = "fav"






class _JobCard(BigButton):
  def __init__(self, params: Params):
    super().__init__("job")
    self._params = params

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    if (fraction := modelview.job_fraction(modelstatus.status(self._params)["job"])) is None:
      return
    x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
    width = self._rect.width - 2 * self.LABEL_HORIZONTAL_PADDING
    y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING + 6
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width, widgets_mici.BAR_HEIGHT), 1.0, 8, panel_common.BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width * fraction, widgets_mici.BAR_HEIGHT), 1.0, 8, panel_common.BAR_FILL)


class ModelsPage(widgets_mici.Page):
  """The primary model tasks."""

  def __init__(self, params: Params, open_page: Callable, back: Callable):
    self._params = params
    self._open_page = open_page
    self._current = widgets_mici.SubmenuButton(modelview.TITLE_CURRENT_MODEL, description=modelview.DESCRIPTION_CURRENT_MODEL)
    self._current.set_click_callback(lambda: open_page(GroupsPage(params, modelprotocols.DRIVING, open_page, back)))
    self._monitoring = widgets_mici.SubmenuButton(modelview.TITLE_MONITORING, description=modelview.model_description(params, modelprotocols.MONITORING))
    self._monitoring.set_click_callback(lambda: open_page(GroupsPage(params, modelprotocols.MONITORING, open_page, back)))
    self._job = _JobCard(params)
    self._cancel = BigButton(modelview.LABEL_CANCEL, modelview.LABEL_CANCEL, description=modelview.DESCRIPTION_CANCEL)
    self._cancel.set_click_callback(self._cancel_action)
    self._job.set_click_callback(lambda: widgets_mici.describe("job", modelview.job_description(modelstatus.status(params)["job"]), ICON_REBUILD))
    self._reboot = BigButton(modelview.LABEL_REBOOT, modelview.LABEL_REBOOT, description=modelview.DESCRIPTION_REBOOT)
    self._reboot.set_click_callback(lambda: models.request_reboot(params))
    self._installed = widgets_mici.SubmenuButton(modelview.TITLE_INSTALLED, description=modelview.DESCRIPTION_INSTALLED)
    self._installed.set_click_callback(lambda: open_page(InstalledPage(params, back)))
    self._refresh = BigButton(modelview.TITLE_REFRESH, description=modelview.DESCRIPTION_REFRESH)
    self._refresh.set_click_callback(self._refresh_action)
    self._storage = BigButton(modelview.TITLE_STORAGE, description=modelview.DESCRIPTION_STORAGE)
    self._storage.set_click_callback(lambda: widgets_mici.describe(modelview.TITLE_STORAGE, modelview.DESCRIPTION_STORAGE, ICON_REBUILD))
    super().__init__(
      [self._current, self._monitoring, self._installed, self._refresh, self._job, self._cancel, self._reboot, self._storage],
      modelview.TITLE_MODELS,
      modelview.DESCRIPTION_MODELS,
      back,
    )
    self._update_rows()

  def _refresh_action(self) -> None:
    modelstatus.request(self._params, modelstatus.OP_REFRESH)  # a refused op reads in the status line

  def _cancel_action(self) -> None:
    if modelstatus.job_active(modelstatus.status(self._params)["job"]):
      widgets_mici.confirm(modelview.CONFIRM_CANCEL, ICON_REMOVE, lambda: modelview.cancel(self._params), red=True)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _update_rows(self):
    status = modelstatus.status(self._params)
    state = modelview.model_page_state(self._params, status, ui_state.is_offroad())
    self._current.set_value(modelview.model_label(self._params, modelprotocols.DRIVING))
    self._monitoring.set_value(modelview.model_label(self._params, modelprotocols.MONITORING))
    self._job.set_text(modelview.job_text(status["job"]) or "job")
    self._job.set_enabled(state["job_present"])
    self._cancel.set_visible(state["cancel"])
    self._installed.set_value(str(len(status["installed"])))
    self._refresh.set_value(modelview.catalog_text(status["catalog"]))
    self._refresh.set_long_press_callback(
      lambda: widgets_mici.describe(modelview.TITLE_REFRESH, modelview.catalog_description(status["catalog"]), ICON_REBUILD)
    )
    self._refresh.set_enabled(state["refresh"])
    self._reboot.set_visible(state["reboot"])
    self._reboot.set_enabled(state["reboot_enabled"])


class InstalledPage(widgets_mici.Page):
  """Downloaded model details/removal, with conditional rebuild."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    self._blocks: list[tuple[BigButton, BigButton, BigButton]] = []
    self._row_count = -1
    super().__init__([], modelview.TITLE_INSTALLED, modelview.DESCRIPTION_INSTALLED, back)
    self._update_rows()

  def _entry(self, index: int) -> dict | None:
    entries = modelstatus.status(self._params)["installed"]
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
      widgets_mici.describe(modelview.display_name(entry), modelview.installed_detail(entry), ICON_REBUILD)

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None and entry.get("state") != models.BUILT:
      modelview.install(self._params, modelview.row_selection(entry))

  def _remove_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      widgets_mici.confirm(modelview.REMOVE_TEXT, ICON_REMOVE, lambda: modelview.request_remove(self._params, entry), red=True)

  def _update_rows(self):
    offroad = ui_state.is_offroad()
    status = modelstatus.status(self._params)
    entries = status["installed"]
    if len(entries) != self._row_count:
      self._row_count = len(entries)
      self._blocks = []
      widgets: list[Widget] = []
      for index in range(self._row_count):
        info = BigButton("")
        info.set_click_callback(lambda index=index: self._describe(index))
        remove = BigButton(modelview.LABEL_REMOVE, modelview.LABEL_REMOVE, description=modelview.REMOVE_TEXT)
        remove.set_click_callback(lambda index=index: self._remove_action(index))
        rebuild = BigButton(modelview.LABEL_REBUILD, modelview.LABEL_REBUILD, description=modelview.DESCRIPTION_STORAGE)
        self._blocks.append((info, remove, rebuild))
        widgets.extend((info, remove, rebuild))
      self.replace_rows(widgets)
    for index, (info, remove, rebuild) in enumerate(self._blocks):
      entry = self._entry(index)
      row_state = modelview.installed_row_state(entry, status, offroad)
      present = row_state["present"]
      info.set_visible(present)
      remove.set_visible(present)
      rebuild.set_visible(row_state["rebuild"])
      info.set_enabled(True)
      remove.set_enabled(row_state["remove"])
      rebuild.set_enabled(row_state["rebuild_enabled"])
      if not present:
        continue
      info.set_text(modelview.display_name(entry or {}))
      info.set_value(f"{entry.get('state', '')} {modelview.human_size(int(entry.get('size', 0)))}".strip())


class GroupsPage(widgets_mici.Page):
  """The picker's groups for one kind: the catalog's own folders, the starred models, and whatever
  the compatibility-class fallback groups. A tap opens one group."""

  def __init__(self, params: Params, kind: str, open_page: Callable, back: Callable):
    self._params = params
    self._kind = kind
    self._open_page = open_page
    self._back = back
    self._rows: list[tuple[str, BigButton]] = []
    super().__init__([], modelview.TITLE_DRIVING if kind == modelprotocols.DRIVING else modelview.TITLE_MONITORING, modelview.DESCRIPTION_MODELS, back)
    self._rebuild()

  def show_event(self):
    super().show_event()
    self._rebuild()

  def _update_state(self):
    super()._update_state()
    self._rebuild()

  def _rebuild(self):
    groups = modelview.chooser_groups(self._params, self._kind)
    if [label for label, _button in self._rows] == [label for label, _rows in groups]:
      for index, (_label, button) in enumerate(self._rows):
        button.set_value(str(len(groups[index][1])))
      return
    self._rows = []
    widgets: list[Widget] = []
    for label, rows in groups:
      first = rows[0] if rows else None
      button = widgets_mici.SubmenuButton(label or modelview.BUNDLED_LABEL, str(len(rows)),
                              description=modelview.DESCRIPTION_MODELS if first is None else modelview.entry_detail(first))
      button.set_click_callback(lambda label=label: self._open_page(GroupPage(self._params, self._kind, label, self._open_page, self._back)))
      self._rows.append((label, button))
      widgets.append(button)
    if not widgets:
      widgets = [widgets_mici.SubmenuButton(modelview.BROWSE_EMPTY)]
    self.replace_rows(widgets)


class GroupPage(widgets_mici.Page):
  """One group's models. A tap installs a model that is not built yet and selects one that is; a long
  press stars it, which is the mici stand-in for the tree's star."""

  def __init__(self, params: Params, kind: str, label: str, open_page: Callable, back: Callable):
    self._params = params
    self._kind = kind
    self._label = label
    self._back = back
    self._rows: list[tuple[BigButton, dict | None]] = []
    super().__init__([], label or modelview.TITLE_CURRENT_MODEL, modelview.DESCRIPTION_MODELS, back)
    self._rebuild()

  def show_event(self):
    super().show_event()
    self._rebuild()

  def _update_state(self):
    super()._update_state()
    self._update_values()

  def _rows_for_label(self) -> list[dict | None]:
    return next((rows for label, rows in modelview.chooser_groups(self._params, self._kind) if label == self._label), [])

  def _rebuild(self):
    entries = self._rows_for_label()
    # By selection, not by identity: an installed entry is re-merged from the index on every read, so
    # its dict is a new object each time and identity would rebuild the cards every frame.
    if [modelview.row_selection(entry) for _button, entry in self._rows] == [
        modelview.row_selection(entry) for entry in entries
      ]:
      self._update_values()
      return
    self._rows = []
    widgets: list[Widget] = []
    for entry in entries:
      button = BigButton("")
      button.set_click_callback(lambda entry=entry: self._choose(entry))
      button.set_long_press_callback(lambda entry=entry: self._star(entry))
      self._rows.append((button, entry))
      widgets.append(button)
    if not widgets:
      widgets = [widgets_mici.SubmenuButton(modelview.BROWSE_NONE)]
    self.replace_rows(widgets)
    self._update_values()

  def _update_values(self):
    status = modelstatus.status(self._params)
    starred = modelview.favorites(self._params)
    offroad = ui_state.is_offroad()
    busy = modelstatus.job_active(status["job"])
    for button, entry in self._rows:
      selection = modelview.row_selection(entry)
      action = modelview.choose_policy(selection, self._kind, self._params, offroad, busy)
      text = (
        modelview.REASON_SELECTED if action == modelview.CHOICE_CURRENT else
        modelview.LABEL_INSTALL if action == modelview.CHOICE_INSTALL else modelview.LABEL_SELECT
      )
      button.set_enabled(action in (modelview.CHOICE_SELECT, modelview.CHOICE_INSTALL))
      button.set_text(modelview.entry_display(entry, self._kind))
      button.set_value(f"{text} {FAVORITE_MARK}" if selection in starred else text)

  def _show_detail(self, entry: dict | None) -> None:
    description = modelview.model_description(self._params, self._kind) if entry is None else modelview.entry_detail(entry)
    widgets_mici.describe(modelview.entry_display(entry, self._kind), description, ICON_REBUILD)

  def _star(self, entry: dict | None) -> None:
    if entry is None:
      return
    selection = modelview.row_selection(entry)
    modelview.toggle_favorite(self._params, selection)
    self._rebuild()

  def _choose(self, entry: dict | None) -> None:
    selection = modelview.row_selection(entry)
    status = modelstatus.status(self._params)
    busy = modelstatus.job_active(status["job"])
    if busy:
      return
    action = modelview.choose_policy(selection, self._kind, self._params, ui_state.is_offroad(), busy)
    if action == modelview.CHOICE_INSTALL:
      widgets_mici.confirm(modelview.INSTALL_TEXT, ICON_INSTALL, lambda: modelview.install(self._params, selection))
    elif action == modelview.CHOICE_SELECT:
      widgets_mici.confirm(modelview.CONFIRM_SELECT, ICON_INSTALL, lambda: models.select(self._params, self._kind, selection))
    else:
      self._show_detail(entry)
