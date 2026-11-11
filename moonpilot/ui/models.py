"""Tizi model settings rendered inside Moonpilot's inline page stack.

The worker owns catalog, build and selection state. This module renders the model tasks and writes
the same request/selection parameters as the worker protocol. Choosing a model is the tree picker in
`moonpilot/ui/model_picker.py`, which is a dialog rather than a page: the catalog is a list of groups
now, and a driver picks from the whole list rather than from a paged copy of it.
"""

from collections.abc import Callable

import pyray as rl

from moonpilot import modelprotocols, modelstatus, models, modelview
from moonpilot.ui import panel_common, widgets
from moonpilot.ui.model_picker import ModelTreeDialog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.list_view import (
  ITEM_TEXT_VALUE_COLOR,
  ItemAction,
  ListItem,
  button_item,
  text_item,
)

BAR_WIDTH = 620


def _offroad() -> bool:
  return ui_state.is_offroad()







class _JobAction(ItemAction):
  def __init__(self, params: Params):
    super().__init__(width=BAR_WIDTH)
    self._params = params

  def _render(self, rect):
    job = modelstatus.status(self._params)["job"]
    fraction = modelview.job_fraction(job)
    if fraction is None:
      gui_label(
        rect,
        modelview.job_text(job),
        font_size=40,
        color=ITEM_TEXT_VALUE_COLOR,
        font_weight=FontWeight.NORMAL,
        alignment=TextAlignment.RIGHT,
        alignment_vertical=TextAlignmentVertical.MIDDLE,
      )
      return False
    bar = rl.Rectangle(rect.x, rect.y + rect.height / 2 - 12, rect.width, 24)
    rl.draw_rectangle_rounded(bar, 1.0, 8, panel_common.BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(bar.x, bar.y, bar.width * fraction, bar.height), 1.0, 8, panel_common.BAR_FILL)
    gui_label(
      bar,
      modelview.job_progress_text(job),
      font_size=34,
      color=rl.WHITE,
      font_weight=FontWeight.MEDIUM,
      alignment=TextAlignment.CENTER,
      alignment_vertical=TextAlignmentVertical.MIDDLE,
    )
    return False


class ModelsLayout(widgets.Page):
  """The primary model tasks: the two pickers, downloaded models, jobs and storage."""

  def __init__(self, params: Params, open_page: Callable, back: Callable):
    self._params = params
    self._open_page = open_page
    self._back = back
    job = ListItem(title="job", description=lambda: modelview.job_description(self._status()["job"]), action_item=_JobAction(self._params))
    job.set_visible(lambda: self._status()["job"] is not None)
    cancel = button_item(modelview.LABEL_CANCEL, modelview.LABEL_CANCEL, description=modelview.DESCRIPTION_CANCEL, callback=self._cancel)
    cancel.set_visible(lambda: modelview.model_page_state(self._params, self._status(), _offroad())["cancel"])
    reboot = button_item(
      modelview.LABEL_REBOOT,
      modelview.LABEL_REBOOT,
      description=modelview.DESCRIPTION_REBOOT,
      callback=lambda: models.request_reboot(self._params),
      enabled=lambda: modelview.model_page_state(self._params, self._status(), _offroad())["reboot_enabled"],
    )
    reboot.set_visible(lambda: modelview.model_page_state(self._params, self._status(), _offroad())["reboot"])
    super().__init__(
      [
        widgets.submenu_item(
          modelview.TITLE_CURRENT_MODEL,
          lambda: modelview.model_label(self._params, modelprotocols.DRIVING),
          modelview.DESCRIPTION_CURRENT_MODEL,
          lambda: self._pick(modelprotocols.DRIVING),
        ),
        widgets.submenu_item(
          modelview.TITLE_MONITORING,
          lambda: modelview.model_label(self._params, modelprotocols.MONITORING),
          lambda: modelview.model_description(self._params, modelprotocols.MONITORING),
          lambda: self._pick(modelprotocols.MONITORING),
        ),
        widgets.submenu_item(
          modelview.TITLE_INSTALLED,
          lambda: str(len(self._status()["installed"])),
          modelview.DESCRIPTION_INSTALLED,
          lambda: self._open_page(InstalledLayout(self._params, self._back)),
        ),
        button_item(
          modelview.TITLE_REFRESH,
          lambda: modelview.catalog_text(self._status()["catalog"]),
          description=lambda: modelview.catalog_description(self._status()["catalog"]),
          callback=lambda: modelstatus.request(self._params, modelstatus.OP_REFRESH),
          enabled=lambda: modelview.model_page_state(self._params, self._status(), _offroad())["refresh"],
        ),
        job,
        cancel,
        reboot,
        text_item(modelview.TITLE_STORAGE, lambda: modelview.storage_text(self._status()["storage"]), description=modelview.DESCRIPTION_STORAGE),
      ],
      modelview.TITLE_MODELS,
      modelview.DESCRIPTION_MODELS,
      back,
    )

  def _pick(self, kind: str) -> None:
    """Open the tree picker for one kind. A row the device has not built is an install, not a
    selection, and the picker's own footer says so before the driver confirms."""
    gui_app.push_widget(ModelTreeDialog(self._params, kind, lambda selection: self._choose(kind, selection), _offroad))

  def _choose(self, kind: str, selection: str) -> None:
    status = self._status()
    action = modelview.choose_policy(selection, kind, self._params, _offroad(), modelstatus.job_active(status["job"]))
    if action == modelview.CHOICE_INSTALL:
      widgets.confirm(modelview.INSTALL_TEXT, modelview.LABEL_INSTALL, lambda: modelview.install(self._params, selection))
    elif action == modelview.CHOICE_SELECT:
      widgets.confirm(modelview.CONFIRM_SELECT, modelview.LABEL_SELECT, lambda: models.select(self._params, kind, selection))

  def _cancel(self) -> None:
    if modelstatus.job_active(self._status()["job"]):
      widgets.confirm(modelview.CONFIRM_CANCEL, modelview.LABEL_CANCEL, lambda: modelview.cancel(self._params))

  def _status(self) -> dict:
    return modelstatus.status(self._params)


class InstalledLayout(widgets.Page):
  """Downloaded model details/removal, with rebuild shown only for non-built entries."""

  def __init__(self, params: Params, back: Callable):
    self._params = params
    self._row_count = -1
    super().__init__([], modelview.TITLE_INSTALLED, modelview.DESCRIPTION_INSTALLED, back)
    self._sync_rows()

  def _sync_rows(self) -> None:
    count = len(modelstatus.status(self._params)["installed"])
    if count == self._row_count:
      return
    self._items = self._items[:1]
    self._row_count = count
    for index in range(count):
      info = button_item(
        lambda index=index: self._name(index),
        lambda index=index: self._state(index),
        description=lambda index=index: modelview.installed_detail(self._entry(index) or {}),
        callback=lambda: None,
      )
      remove = button_item(
        modelview.LABEL_REMOVE,
        modelview.LABEL_REMOVE,
        description=modelview.REMOVE_TEXT,
        callback=lambda index=index: self._remove(index),
        enabled=lambda index=index: self._row_state(index).get("remove", False),
      )
      rebuild = button_item(
        modelview.LABEL_REBUILD,
        modelview.LABEL_REBUILD,
        description=lambda index=index: modelview.installed_detail(self._entry(index) or {}),
        callback=lambda index=index: self._rebuild_action(index),
        enabled=lambda index=index: self._row_state(index)["rebuild_enabled"],
      )
      for row in (info, remove, rebuild):
        row.set_visible(lambda index=index: self._row_state(index)["present"])
      rebuild.set_visible(lambda index=index: self._row_state(index)["rebuild"])
      for row in (info, remove, rebuild):
        self.add_widget(row)

  def _update_state(self) -> None:
    super()._update_state()
    self._sync_rows()

  def _entry(self, index: int) -> dict | None:
    entries = modelstatus.status(self._params)["installed"]
    return entries[index] if index < len(entries) else None

  def _row_state(self, index: int) -> dict[str, bool]:
    entry = self._entry(index)
    return modelview.installed_row_state(entry, modelstatus.status(self._params), _offroad())

  def _name(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else modelview.display_name(entry)

  def _state(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else str(entry.get("state", ""))

  def _remove(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      widgets.confirm(modelview.REMOVE_TEXT, modelview.LABEL_REMOVE, lambda: modelview.request_remove(self._params, entry))

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None and entry.get("state") != models.BUILT:
      modelview.install(self._params, modelview.row_selection(entry))


def row(params: Params, open_page: Callable, back: Callable) -> ListItem:
  return widgets.submenu_item(
    modelview.TITLE_MODELS,
    lambda: modelview.model_label(params, modelprotocols.DRIVING),
    lambda: modelview.model_description(params, modelprotocols.DRIVING),
    lambda: open_page(ModelsLayout(params, open_page, back)),
  )
