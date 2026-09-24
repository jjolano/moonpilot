"""Mici model settings rendered as inline vertical pages inside Moonpilot settings.

The picker is a drill-down rather than a tree, because a mici screen is a column of cards: the
groups first, then one group's models. The grouping, the names and the stars are `moonpilot/models.py`'s
— the same ones the tizi tree dialog draws — and a star here is a long press, since there is no room
for a star glyph beside a card's value.
"""

from collections.abc import Callable, Sequence

import pyray as rl

from moonpilot import models
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, SettingDescriptionDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller import Scroller

INSTALLED_ROWS = 8
ICON_INSTALL = "icons_mici/settings/software.png"
ICON_REMOVE = "icons_mici/settings/device/uninstall.png"
ICON_REBUILD = "icons_mici/settings/device/update.png"
_CHEVRON_PATH = "icons/chevron_right.png"
_INVALID = object()
BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)
BAR_HEIGHT = 10
FAVORITE_MARK = "fav"


def _request(params: Params, op: str, selection: str = "") -> None:
  models.request(params, op, selection)


def _describe(title: str, text: str) -> None:
  gui_app.push_widget(SettingDescriptionDialog(title, text or ".", gui_app.texture(ICON_REBUILD, 64, 75)))


def _confirm(title: str, icon: str, action, *, red: bool = False) -> None:
  gui_app.push_widget(BigConfirmationDialog(title, gui_app.texture(icon, 64, 75), action, red=red))


class _SubmenuButton(BigButton):
  def __init__(self, text: str, value: str = "", *, description: str = ""):
    super().__init__(text, value, description=description)
    # Load after init_window so the texture is valid; per-instance, not module scope.
    self._chevron = gui_app.texture(_CHEVRON_PATH, 48, 48)

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

  def replace_rows(self, rows: Sequence[Widget]) -> None:
    """Swap everything after the Back card. The picker's groups come from the catalog, so the row
    count is not known until the page opens. `_Scroller.items` is the live list, so this is an
    in-place swap rather than a rebuild."""
    self._scroller.items[:] = [*self._scroller.items[:1], *rows]


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
    self._current.set_click_callback(lambda: open_page(GroupsPage(params, models.DRIVING, open_page, back)))
    self._monitoring = submenu_button(models.TITLE_MONITORING, description=models.model_description(params, models.MONITORING))
    self._monitoring.set_click_callback(lambda: open_page(GroupsPage(params, models.MONITORING, open_page, back)))
    self._job = _JobCard(params)
    self._cancel = BigButton(models.LABEL_CANCEL, models.LABEL_CANCEL, description=models.DESCRIPTION_CANCEL)
    self._cancel.set_click_callback(self._cancel_action)
    self._job.set_click_callback(lambda: _describe("job", models.job_description(models.status(params)["job"], params)))
    self._reboot = BigButton(models.LABEL_REBOOT, models.LABEL_REBOOT, description=models.DESCRIPTION_REBOOT)
    self._reboot.set_click_callback(lambda: models.request_reboot(params))
    self._installed = submenu_button(models.TITLE_INSTALLED, description=models.DESCRIPTION_INSTALLED)
    self._installed.set_click_callback(lambda: open_page(InstalledPage(params, back)))
    self._refresh = BigButton(models.TITLE_REFRESH, description=models.DESCRIPTION_REFRESH)
    self._refresh.set_click_callback(lambda: _request(params, "refresh"))
    self._storage = BigButton(models.TITLE_STORAGE, description=models.DESCRIPTION_STORAGE)
    self._storage.set_click_callback(lambda: _describe(models.TITLE_STORAGE, models.DESCRIPTION_STORAGE))
    super().__init__(
      [self._current, self._monitoring, self._installed, self._refresh, self._job, self._cancel, self._reboot, self._storage],
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
    self._refresh.set_value(models.catalog_text(status["catalog"]))
    self._refresh.set_long_press_callback(lambda: _describe(models.TITLE_REFRESH, models.catalog_description(status["catalog"])))
    self._reboot.set_visible(models.restart_needed(self._params))
    self._reboot.set_enabled(ui_state.is_offroad())


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
    return models.page_item(models.status(self._params)["installed"], 0, INSTALLED_ROWS, index)

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


class GroupsPage(Page):
  """The picker's groups for one kind: the catalog's own folders, the starred models, and whatever
  the compatibility-class fallback groups. A tap opens one group."""

  def __init__(self, params: Params, kind: str, open_page: Callable, back: Callable):
    self._params = params
    self._kind = kind
    self._open_page = open_page
    self._back = back
    self._rows: list[tuple[str, BigButton]] = []
    super().__init__([], models.TITLE_DRIVING if kind == models.DRIVING else models.TITLE_MONITORING, models.DESCRIPTION_MODELS, back)
    self._rebuild()

  def show_event(self):
    super().show_event()
    self._rebuild()

  def _update_state(self):
    super()._update_state()
    self._rebuild()

  def _rebuild(self):
    groups = models.chooser_groups(self._params, self._kind)
    if [label for label, _button in self._rows] == [label for label, _rows in groups]:
      for index, (_label, button) in enumerate(self._rows):
        button.set_value(str(len(groups[index][1])))
      return
    self._rows = []
    widgets: list[Widget] = []
    for label, rows in groups:
      # The bundled model's group is the catalog's own unnamed group, which reads as "stock" here.
      first = rows[0] if rows else None
      button = submenu_button(label or models.BUNDLED_LABEL, str(len(rows)),
                              description=models.DESCRIPTION_MODELS if first is None else models.entry_detail(first))
      button.set_click_callback(lambda label=label: self._open_page(GroupPage(self._params, self._kind, label, self._open_page, self._back)))
      self._rows.append((label, button))
      widgets.append(button)
    if not widgets:
      widgets = [submenu_button(models.BROWSE_EMPTY)]
    self.replace_rows(widgets)


class GroupPage(Page):
  """One group's models. A tap installs a model that is not built yet and selects one that is; a long
  press stars it, which is the mici stand-in for the tree's star."""

  def __init__(self, params: Params, kind: str, label: str, open_page: Callable, back: Callable):
    self._params = params
    self._kind = kind
    self._label = label
    self._back = back
    self._rows: list[tuple[BigButton, dict | None]] = []
    super().__init__([], label or models.TITLE_CURRENT_MODEL, models.DESCRIPTION_MODELS, back)
    self._rebuild()

  def show_event(self):
    super().show_event()
    self._rebuild()

  def _update_state(self):
    super()._update_state()
    self._update_values()

  def _rows_for_label(self) -> list[dict | None]:
    return next((rows for label, rows in models.chooser_groups(self._params, self._kind) if label == self._label), [])

  def _rebuild(self):
    entries = self._rows_for_label()
    # By selection, not by identity: an installed entry is re-merged from the index on every read, so
    # its dict is a new object each time and identity would rebuild the cards every frame.
    if ["" if entry is None else models.selection_of(entry) for _button, entry in self._rows] == [
        "" if entry is None else models.selection_of(entry) for entry in entries
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
      widgets = [submenu_button(models.BROWSE_NONE)]
    self.replace_rows(widgets)
    self._update_values()

  def _update_values(self):
    desired = models.desired(self._params, self._kind)
    starred = models.favorites(self._params)
    offroad = ui_state.is_offroad()
    for button, entry in self._rows:
      selection = "" if entry is None else models.selection_of(entry)
      if entry is None or entry.get("state") == "built":
        action = models.REASON_SELECTED if selection == desired else models.LABEL_SELECT
        button.set_enabled(offroad)
      else:
        action = models.LABEL_INSTALL
        button.set_enabled(True)
      button.set_text(models.entry_display(entry, self._kind))
      button.set_value(f"{action} {FAVORITE_MARK}" if selection in starred else action)

  def _show_detail(self, entry: dict | None) -> None:
    _describe(models.entry_display(entry, self._kind), models.model_description(self._params, self._kind) if entry is None else models.entry_detail(entry))

  def _star(self, entry: dict | None) -> None:
    if entry is None:
      return
    selection = models.selection_of(entry)
    models.set_favorite(self._params, selection, selection not in models.favorites(self._params))
    self._rebuild()

  def _choose(self, entry: dict | None) -> None:
    selection = "" if entry is None else models.selection_of(entry)
    if entry is not None and entry.get("state") != "built":
      _confirm(models.INSTALL_TEXT, ICON_INSTALL, lambda: _request(self._params, "install", selection))
      return
    if selection == models.desired(self._params, self._kind) or not ui_state.is_offroad():
      self._show_detail(entry)
      return
    _confirm(models.CONFIRM_SELECT, ICON_INSTALL, lambda: models.select(self._params, self._kind, selection))
