"""The tizi half of the model marketplace: the panes that browse, install, compose and select.

Every rule and every string is `moonpilot/models.py`'s; this file is the rendering. It never reads
the catalog file itself -- `MoonpilotModelsStatus` is what the device knows, and `models.browse()`
is the index the worker derived from the snapshot it validated, so a pane opens on an empty store and
shows what the device actually has.

Four things about the shape of these panes:

- **Nothing here does I/O.** An action writes a request param (`models.request`) or a desired
  selection (`models.select`); the worker fetches and compiles. A row's title, value, description and
  enabled state are *callables*, which tizi re-resolves on every render, so the panes follow the
  worker with no refresh loop at all (AGENTS.md, Features).
- **Installation is not gated on offroad; selection, composing and removal are.** A download and a
  refresh happen whenever the driver asks; a change to what the car runs waits until it is parked,
  which is what `ui_state.is_offroad()` is for and what the worker enforces again.
- **A page is a `Scroller` with a leading `back` row**, which is how tizi leaves a pushed widget.
  Neither tree overrides the other's: a fork override of a renamed upstream method is exactly the
  failure a merge cannot see.
- **The catalog list is a fixed pool of rows.** `Scroller` cannot drop an item and a list rebuilt per
  frame would fight the scroll offset, so the page keeps `PAGE_ROWS` rows and hides the spares; every
  row resolves through the current page and filter, which is why the filter buttons only have to
  change two integers.
"""

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
  dual_button_item,
  multiple_button_item,
  text_item,
)
from openpilot.system.ui.widgets.scroller_tici import Scroller

BAR_TRACK = rl.Color(57, 57, 57, 255)  # upstream's own button grey (`MultipleButtonAction`)
BAR_FILL = rl.Color(51, 171, 76, 255)  # ... and its green, for the part that is done
BAR_WIDTH = 620  # the row's right-hand area while a job runs: wide enough that the label sits
# inside the bar and still leaves a visible length of track to compare the fill against

PAGE_ROWS = 8  # catalog entries per page
INSTALLED_ROWS = 8  # installed blocks per page -- more than any store the caps allow
# The order of the "show" row's buttons: the index the page keeps is this list's index.
KIND_FILTERS = (models.FILTER_ALL, models.FILTER_DRIVING, models.FILTER_MONITORING)


def _offroad() -> bool:
  return ui_state.is_offroad()


class Page(Scroller):
  """A pushed pane: a scroller whose first row goes back."""

  def __init__(self, rows: list, title: str, description: str = ""):
    back = button_item(title, models.LABEL_BACK, description=description, callback=lambda: gui_app.pop_widget())
    super().__init__([back, *rows], line_separator=True, spacing=0)


def _confirm(text: str, label: str, action) -> None:
  """Every irreversible action goes through one dialog with one shape: confirm text, and a callback
  that only runs on `DialogResult.CONFIRM`."""

  def confirmed(result) -> None:
    if result is not None:
      action()

  gui_app.push_widget(ConfirmDialog(text, label, rich=True, callback=confirmed))


class _JobAction(ItemAction):
  """The job row's right-hand side: a bar while bytes are moving, the phase line otherwise.

  Not a button (`_render` returns False), so nothing happens on a press over it -- and that includes
  the row's own tap-to-unfold: `ListItem._handle_mouse_release` returns early for any press inside its
  action rect, which is this strip. The **title** is the way to the description, where the reason and
  the cancel hint live, and it is the larger target by far (the row is 2160 px wide at 620 px of bar).
  The bar is drawn only when `models.job_fraction` has a number behind it: the phases without a total
  (`building`, `verifying`, the two waiting ones) get the phase line and, for the ones that can sit, an
  elapsed clock from `models.job_text`.
  """

  def __init__(self, params: Params):
    super().__init__(width=BAR_WIDTH)
    self._params = params

  def _render(self, rect):
    job = models.status(self._params)["job"]
    fraction = models.job_fraction(job)
    if fraction is None:
      gui_label(rect, models.job_text(job), font_size=40, color=ITEM_TEXT_VALUE_COLOR, font_weight=FontWeight.NORMAL,
                alignment=TextAlignment.RIGHT, alignment_vertical=TextAlignmentVertical.MIDDLE)
      return False
    bar = rl.Rectangle(rect.x, rect.y + rect.height / 2 - 12, rect.width, 24)
    rl.draw_rectangle_rounded(bar, 1.0, 8, BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(bar.x, bar.y, bar.width * fraction, bar.height), 1.0, 8, BAR_FILL)
    gui_label(bar, models.job_progress_text(job), font_size=34, color=rl.WHITE, font_weight=FontWeight.MEDIUM,
              alignment=TextAlignment.CENTER, alignment_vertical=TextAlignmentVertical.MIDDLE)
    return False


class ModelsLayout(Page):
  """The models subpane's root: what is in effect, what this device holds, and where to go."""

  def __init__(self, params: Params):
    self._params = params
    job = ListItem(title="job", description=lambda: models.job_description(self._status()["job"], self._params),
                   action_item=_JobAction(self._params))
    job.set_visible(lambda: self._status()["job"] is not None)
    cancel = button_item(models.LABEL_CANCEL, models.LABEL_CANCEL, description=models.DESCRIPTION_CANCEL, callback=self._cancel)
    cancel.set_visible(lambda: models.job_active(self._status()["job"]))
    # A selection only takes effect at the next boot, so the row that acts on it is here, next to the
    # two rows it applies to, and only while a restart is owed (`models.restart_needed`).
    reboot = button_item(models.LABEL_REBOOT, models.LABEL_REBOOT, description=models.DESCRIPTION_REBOOT,
                         callback=lambda: models.request_reboot(self._params), enabled=_offroad)
    reboot.set_visible(lambda: models.restart_needed(self._params))
    super().__init__(
      [
        text_item(
          models.TITLE_DRIVING,
          lambda: models.model_label(self._params, models.DRIVING),
          description=lambda: models.model_description(self._params, models.DRIVING),
        ),
        text_item(
          models.TITLE_MONITORING,
          lambda: models.model_label(self._params, models.MONITORING),
          description=lambda: models.model_description(self._params, models.MONITORING),
        ),
        job,
        cancel,
        reboot,
        button_item(
          models.TITLE_BROWSE, models.LABEL_OPEN, description=models.DESCRIPTION_BROWSE, callback=lambda: gui_app.push_widget(CatalogLayout(self._params))
        ),
        button_item(
          models.TITLE_INSTALLED,
          lambda: str(len(self._status()["installed"])),
          description=models.DESCRIPTION_INSTALLED,
          callback=lambda: gui_app.push_widget(InstalledLayout(self._params)),
        ),
        button_item(
          models.TITLE_COMPOSE,
          models.LABEL_OPEN,
          description=models.DESCRIPTION_COMPOSE,
          callback=lambda: gui_app.push_widget(ComposeLayout(self._params)),
          enabled=_offroad,
        ),
        text_item(models.TITLE_STORAGE, lambda: models.storage_text(self._status()["storage"]), description=models.DESCRIPTION_STORAGE),
        button_item(
          models.TITLE_REFRESH,
          lambda: models.catalog_text(self._status()["catalog"]),
          description=lambda: models.catalog_description(self._status()["catalog"]),
          callback=lambda: models.request(self._params, "refresh"),
        ),
      ],
      models.TITLE_MODELS,
      models.DESCRIPTION_MODELS,
    )

  def _cancel(self) -> None:
    if models.job_active(self._status()["job"]):
      _confirm(models.CONFIRM_CANCEL, models.LABEL_CANCEL, lambda: models.request(self._params, "cancel"))

  def _status(self) -> dict:
    return models.status(self._params)


class CatalogLayout(Page):
  """Every catalog entry this device knows about, and what it would do with each one."""

  def __init__(self, params: Params):
    self._params = params
    self._kind = 0
    self._runnable_only = True
    self._page = 0
    self._catalog = models.browse().get("revision", "") != ""
    rows: list = [
      multiple_button_item("show", "", [models.FILTER_ALL, models.FILTER_DRIVING, models.FILTER_MONITORING], 0, callback=self._set_kind),
      multiple_button_item(models.FILTER_RUNNABLE, "", ["off", "on"], 1, callback=self._set_runnable),
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
    super().__init__(rows, models.TITLE_BROWSE, models.DESCRIPTION_BROWSE)

  # --- the list ------------------------------------------------------------------------------------
  def _entries(self) -> list[dict]:
    entries = models.browse().get("entries", [])
    kind = KIND_FILTERS[self._kind]
    if kind != models.FILTER_ALL:
      entries = [entry for entry in entries if entry.get("kind") == kind]
    if self._runnable_only:
      entries = [entry for entry in entries if entry.get("admitted")]
    return entries

  def _pages(self) -> int:
    return max(1, (len(self._entries()) + PAGE_ROWS - 1) // PAGE_ROWS)

  def _index_of(self, row) -> int:
    return self._rows.index(row)

  def _entry_of(self, row) -> dict | None:
    entries = self._entries()
    position = self._page * PAGE_ROWS + self._index_of(row)
    return entries[position] if position < len(entries) else None

  def _installed(self) -> set[str]:
    return {models.selection_of(entry) for entry in models.status(self._params)["installed"]}

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
    return entry is not None and bool(entry.get("admitted")) and models.selection_of(entry) not in self._installed()

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

  def _set_runnable(self, index: int) -> None:
    self._runnable_only, self._page = index == 1, 0

  def _offset(self, delta: int) -> None:
    self._page = min(max(0, self._page + delta), self._pages() - 1)

  # --- actions -------------------------------------------------------------------------------------
  def _install(self, index: int) -> None:
    entry = self._entry_at(index)
    if entry is None or not self._installable(index):
      return
    selection = models.selection_of(entry)
    _confirm(models.INSTALL_TEXT, models.LABEL_INSTALL, lambda: models.request(self._params, "install", selection))


class InstalledLayout(Page):
  """What the store holds: one block per installed model -- what it is, whether it is built, and the
  three things a driver can do to it. `rebuild` re-runs the install, which is what a fork update
  needs after the compiler or tinygrad moves: nothing invalidates a build automatically, on purpose
  (moonpilot/modelsd.py)."""

  def __init__(self, params: Params):
    self._params = params
    rows: list = []
    for index in range(INSTALLED_ROWS):
      info = text_item(
        lambda index=index: self._name(index),
        lambda index=index: self._state(index),
        description=lambda index=index: models.installed_detail(self._entry(index) or {}),
      )
      actions = dual_button_item(
        lambda index=index: self._action_label(index),
        models.LABEL_REMOVE,
        left_callback=lambda index=index: self._select(index),
        right_callback=lambda index=index: self._remove(index),
        enabled=_offroad,
      )
      rebuild = button_item(
        models.LABEL_REBUILD,
        models.LABEL_REBUILD,
        description=lambda index=index: models.installed_detail(self._entry(index) or {}),
        callback=lambda index=index: self._rebuild_action(index),
        enabled=_offroad,
      )
      for row in (info, actions, rebuild):
        row.set_visible(lambda index=index: self._entry(index) is not None)
      rows += [info, actions, rebuild]
    super().__init__(rows, models.TITLE_INSTALLED, models.DESCRIPTION_INSTALLED)

  def _entry(self, index: int) -> dict | None:
    entries = models.status(self._params)["installed"]
    return entries[index] if index < len(entries) else None

  def _name(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else str(entry.get("name") or models.selection_of(entry)[:12])

  def _state(self, index: int) -> str:
    entry = self._entry(index)
    return "" if entry is None else str(entry.get("state", ""))

  def _action_label(self, index: int) -> str:
    entry = self._entry(index)
    return models.LABEL_SELECT if entry is None else models.installed_action(entry, self._params)[0]

  def _select(self, index: int) -> None:
    entry = self._entry(index)
    if entry is None:
      return
    selection = models.selection_of(entry)
    kind = str(entry.get("kind", models.DRIVING))
    if models.desired(self._params, kind) == selection:
      models.select(self._params, kind, "")
      return
    _confirm(models.SELECT_TEXT, models.LABEL_SELECT, lambda: models.select(self._params, kind, selection))

  def _remove(self, index: int) -> None:
    entry = self._entry(index)
    if entry is None:
      return
    selection = models.selection_of(entry)
    _confirm(models.REMOVE_TEXT, models.LABEL_REMOVE, lambda: models.request(self._params, "remove", selection))

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      models.request(self._params, "install", models.selection_of(entry))


class ComposeLayout(Page):
  """One model out of installed pieces.

  Driven by the protocol's *roles*, not by the recipes the catalog groups: pick a protocol with
  candidates, then one installed recipe per role, and the rule that decides whether the set can run
  (`models.composition_reason` -- the same function the worker checks a record against) is on the
  page before anything is written. A role row cycles through its candidates and lists them in the
  description: tizi has no picker widget, and the description is the one place that can hold a list
  of names without clipping.
  """

  def __init__(self, params: Params):
    self._params = params
    self._protocols = [proto for proto in models.PROTOCOLS if any(models.compose_candidates(proto.id).values())]
    self._protocol_index = 0
    roles = self._roles()
    self._picks = dict.fromkeys(roles, 0)
    rows: list = [
      multiple_button_item("protocol", models.COMPOSE_TEXT, [models.protocol_label(proto.id) for proto in self._protocols], 0, callback=self._set_protocol)
    ]
    rows[0].set_visible(lambda: len(self._protocols) > 1)
    rows += [
      button_item(
        lambda role=role: role,
        lambda role=role: self._pick_label(role),
        description=lambda role=role: self._pick_detail(role),
        callback=lambda role=role: self._cycle(role),
      )
      for role in roles
    ]
    rows += [
      text_item("compatibility", lambda: self._verdict_text(), description=models.COMPOSE_TEXT),
      button_item(models.LABEL_COMPOSE, models.LABEL_COMPOSE, description=models.COMPOSE_TEXT, callback=self._compose, enabled=self._can_compose),
    ]
    super().__init__(rows, models.TITLE_COMPOSE, models.COMPOSE_TEXT)

  def _protocol(self):
    return self._protocols[self._protocol_index] if self._protocols else None

  def _roles(self) -> tuple[str, ...]:
    proto = self._protocol()
    return proto.roles if proto is not None else ()

  def _set_protocol(self, index: int) -> None:
    self._protocol_index = index
    self._picks = dict.fromkeys(self._roles(), 0)

  def _candidates(self, role: str) -> list[dict]:
    proto = self._protocol()
    return models.compose_candidates(proto.id).get(role, []) if proto is not None else []

  def _pick_label(self, role: str) -> str:
    candidates = self._candidates(role)
    return candidates[self._picks.get(role, 0) % len(candidates)]["name"] if candidates else models.LABEL_NONE

  def _pick_detail(self, role: str) -> str:
    candidates = self._candidates(role)
    if not candidates:
      return f"no installed {role} member"
    return ", ".join(f"{candidate['name']} ({models.human_size(int(candidate['size']))})" for candidate in candidates)

  def _cycle(self, role: str) -> None:
    if candidates := self._candidates(role):
      self._picks[role] = (self._picks.get(role, 0) + 1) % len(candidates)

  def _picks_by_recipe(self) -> dict[str, str]:
    picks = {}
    for role in self._roles():
      candidates = self._candidates(role)
      if candidates:
        picks[role] = candidates[self._picks.get(role, 0) % len(candidates)]["recipe"]
    return picks

  def _verdict_text(self) -> str:
    proto = self._protocol()
    if proto is None:
      return models.LABEL_NONE
    reason = models.composition_reason(proto.id, self._picks_by_recipe())
    return models.COMPATIBLE if reason is None else reason

  def _can_compose(self) -> bool:
    proto = self._protocol()
    return bool(proto) and _offroad() and models.composition_reason(proto.id, self._picks_by_recipe()) is None

  def _compose(self) -> None:
    proto = self._protocol()
    if proto is None or not self._can_compose():
      return
    picks = self._picks_by_recipe()

    def write_and_install() -> None:
      composition, _reason = models.write_composition(proto.id, picks)
      if composition is not None:
        models.request(self._params, "install", composition)

    _confirm(models.COMPOSE_TEXT, models.LABEL_COMPOSE, write_and_install)


def row(params: Params):
  """The row the moonpilot panel itself shows: the subpane, with the model in effect on its button,
  so the answer to "what is the car running" does not need a page."""
  return button_item(
    models.TITLE_MODELS,
    lambda: models.model_label(params, models.DRIVING),
    description=lambda: models.model_description(params, models.DRIVING),
    callback=lambda: gui_app.push_widget(ModelsLayout(params)),
  )
