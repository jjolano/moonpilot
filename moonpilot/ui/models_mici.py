"""The mici half of the model marketplace: the same four panes as the tizi half, in mici's widgets.

`moonpilot/models.py` owns every rule and every string; this file only renders. The differences from
`moonpilot/ui/models.py` are the tree's, not the feature's:

- mici's pages are `NavScroller`s with no back row -- the swipe-down gesture is the way out.
- mici's values are *pushed*, not resolved, so every row re-reads `models.status()` in
  `_update_state` (one cached parse, see `models.status`) and every action re-resolves the entry it
  is about at click time rather than at construction.
- `BigButton` captures its description when it is built, so a dynamic long-press description is not
  available: the value line carries the state, and a tap opens `SettingDescriptionDialog` with the
  detail text as it is at that moment.
- `BigMultiToggle` is mici's segmented control, which is what the filters and the compose pickers
  use: it highlights one option and cycles on tap. A picker's options are the candidate *names*,
  which two recipes can share, so a name that appears twice is disambiguated with its digest prefix.
"""

import pyray as rl
from moonpilot import models
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, SettingDescriptionDialog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets.scroller import NavScroller

JOB_TRACK_COLOR = rl.Color(57, 57, 57, 255)
JOB_FILL_COLOR = rl.Color(51, 171, 76, 255)

PAGE_ROWS = 6  # catalog entries and installed blocks per page
KIND_FILTERS = (models.FILTER_ALL, models.FILTER_DRIVING, models.FILTER_MONITORING)
ICON_INSTALL = "icons_mici/settings/software.png"
ICON_REMOVE = "icons_mici/settings/device/uninstall.png"
ICON_REBUILD = "icons_mici/settings/device/update.png"
# The job card's bar: the same grey and green tizi's uses, so the two trees read alike.
BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)
BAR_HEIGHT = 10


def _request(params: Params, op: str, selection: str = "") -> None:
  """A request's id is the worker's cancellation handle, not the panel's business. `models.request`
  returns it; every actionable row here discards it, so they all go through this one line."""
  models.request(params, op, selection)


def _describe(title: str, text: str) -> None:
  gui_app.push_widget(SettingDescriptionDialog(title, text or ".", gui_app.texture(ICON_REBUILD, 64, 75)))


def _confirm(title: str, icon: str, action, *, red: bool = False) -> None:
  gui_app.push_widget(BigConfirmationDialog(title, gui_app.texture(icon, 64, 75), action, red=red))


def _labels(candidates: list[dict]) -> list[str]:
  """One label per candidate, unique: a duplicated name gets its digest prefix, which is the only
  thing that tells two recipes apart on a picker."""
  names = [str(candidate["name"]) for candidate in candidates]
  return [f"{name} · {candidate['recipe'][:8]}" if names.count(name) > 1 else name for name, candidate in zip(names, candidates, strict=True)]


class _JobCard(BigButton):
  """The job card with a bar under its value line.

  mici's card draws itself, so the bar is drawn from the card's own extension point (`_draw_content`,
  the same one `BigToggle` and `GreyBigButton` use) rather than by a second widget. It appears only
  while `models.job_fraction` has a number behind it: the value line already carries the phase, the
  bytes and -- for the phases that can sit -- an elapsed clock.
  """

  def __init__(self, params: Params):
    super().__init__("job")
    self._params = params

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    if (fraction := models.job_fraction(models.status(self._params)["job"])) is None:
      return
    x = self._rect.x + self.LABEL_HORIZONTAL_PADDING
    width = self._rect.width - 2 * self.LABEL_HORIZONTAL_PADDING
    # `BigButton` bottom-aligns the value line at `btn_y + height - LABEL_VERTICAL_PADDING`, and that
    # line wraps to three at this width, so the bar goes into the padding band *below* it -- 6 px of
    # gap, then the bar, with 7 px of card left under it.
    y = btn_y + self._rect.height - self.LABEL_VERTICAL_PADDING + 6
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width, BAR_HEIGHT), 1.0, 8, BAR_TRACK)
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, width * fraction, BAR_HEIGHT), 1.0, 8, BAR_FILL)


class ModelsPage(NavScroller):
  """The models subpane's root."""

  def __init__(self, params: Params):
    super().__init__()
    self._params = params

    self._driving = self._status_row(models.TITLE_DRIVING, models.DRIVING)
    self._monitoring = self._status_row(models.TITLE_MONITORING, models.MONITORING)
    self._job = _JobCard(params)
    self._cancel = BigButton(models.LABEL_CANCEL, models.LABEL_CANCEL, description=models.DESCRIPTION_CANCEL)
    self._cancel.set_click_callback(self._cancel_action)
    self._job.set_click_callback(lambda: _describe("job", models.job_description(models.status(self._params)["job"], self._params)))
    # A selection takes effect at the next boot, so the row that acts on it is here, beside the two
    # rows it applies to, and only while a restart is owed.
    self._reboot = BigButton(models.LABEL_REBOOT, models.LABEL_REBOOT, description=models.DESCRIPTION_REBOOT)
    self._reboot.set_click_callback(lambda: models.request_reboot(self._params))
    self._browse = BigButton(models.TITLE_BROWSE, description=models.DESCRIPTION_BROWSE)
    self._browse.set_click_callback(lambda: gui_app.push_widget(CatalogPage(self._params)))
    self._installed = BigButton(models.TITLE_INSTALLED, description=models.DESCRIPTION_INSTALLED)
    self._installed.set_click_callback(lambda: gui_app.push_widget(InstalledPage(self._params)))
    self._compose = BigButton(models.TITLE_COMPOSE, description=models.DESCRIPTION_COMPOSE)
    self._compose.set_click_callback(lambda: gui_app.push_widget(ComposePage(self._params)))
    self._storage = BigButton(models.TITLE_STORAGE, description=models.DESCRIPTION_STORAGE)
    self._storage.set_click_callback(lambda: _describe(models.TITLE_STORAGE, models.DESCRIPTION_STORAGE))
    self._refresh = BigButton(models.TITLE_REFRESH, description=models.DESCRIPTION_REFRESH)
    self._refresh.set_click_callback(lambda: _request(self._params, "refresh"))

    self._scroller.add_widgets([self._driving, self._monitoring, self._job, self._cancel, self._reboot, self._browse, self._installed,
                                self._compose, self._storage, self._refresh])
    self._update_rows()

  def _cancel_action(self) -> None:
    if models.job_active(models.status(self._params)["job"]):
      _confirm(models.CONFIRM_CANCEL, ICON_REMOVE, lambda: _request(self._params, "cancel"), red=True)

  def _status_row(self, title: str, kind: str) -> BigButton:
    row = BigButton(title)
    row.set_click_callback(lambda kind=kind: _describe(title, models.model_description(self._params, kind)))
    return row

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _update_rows(self):
    status = models.status(self._params)
    self._driving.set_value(models.model_label(self._params, models.DRIVING))
    self._monitoring.set_value(models.model_label(self._params, models.MONITORING))
    self._job.set_text(models.job_text(status["job"]) or "job")
    self._job.set_enabled(status["job"] is not None)
    self._cancel.set_visible(models.job_active(status["job"]))
    self._installed.set_value(str(len(status["installed"])))
    self._storage.set_value(models.storage_text(status["storage"]))
    self._refresh.set_value(models.catalog_text(status["catalog"]))
    self._compose.set_enabled(ui_state.is_offroad())
    self._reboot.set_visible(models.restart_needed(self._params))
    self._reboot.set_enabled(ui_state.is_offroad())


class CatalogPage(NavScroller):
  """Every catalog entry this device knows about."""

  def __init__(self, params: Params):
    super().__init__()
    self._params = params
    self._page = 0
    self._kind_toggle = BigMultiToggle("show", list(KIND_FILTERS))
    self._runnable_toggle = BigMultiToggle(models.FILTER_RUNNABLE, ["off", "on"])
    self._runnable_toggle.set_value("on")
    self._rows: list[BigButton] = []
    for _index in range(PAGE_ROWS):
      row = BigButton("")
      self._rows.append(row)
    self._page_row = BigButton("catalog")
    self._older = BigButton("older", models.LABEL_OLDER)
    self._older.set_click_callback(lambda: self._offset(1))
    self._newer = BigButton("newer", models.LABEL_NEWER)
    self._newer.set_click_callback(lambda: self._offset(-1))

    self._scroller.add_widgets([self._kind_toggle, self._runnable_toggle, *self._rows, self._page_row, self._older, self._newer])
    self._update_rows()

  def show_event(self):
    super().show_event()
    self._page = 0
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _entries(self) -> list[dict]:
    entries = models.browse().get("entries", [])
    if self._kind_toggle.get_value() != models.FILTER_ALL:
      entries = [entry for entry in entries if entry.get("kind") == self._kind_toggle.get_value()]
    if self._runnable_toggle.get_value() == "on":
      entries = [entry for entry in entries if entry.get("admitted")]
    return entries

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
    selection = models.selection_of(entry)
    _confirm(models.INSTALL_TEXT, ICON_INSTALL, lambda: _request(self._params, "install", selection))

  def _update_rows(self):
    self._page = min(self._page, self._pages() - 1)
    installed = {models.selection_of(entry) for entry in models.status(self._params)["installed"]}
    for index, row in enumerate(self._rows):
      entry = self._entry_at(index)
      row.set_visible(entry is not None)
      if entry is None:
        continue
      selection = models.selection_of(entry)
      admissible = bool(entry.get("admitted")) and selection not in installed
      row.set_text(str(entry.get("name", "")))
      row.set_value(models.entry_action(entry, installed))
      row.set_enabled(admissible)
      row.set_click_callback(
        (lambda index=index: self._install(index)) if admissible else (lambda entry=entry: _describe(str(entry.get("name", "")), models.entry_detail(entry)))
      )
    entries = self._entries()
    if entries:
      self._page_row.set_value(f"{self._page + 1} / {self._pages()}")
    else:
      self._page_row.set_value(models.BROWSE_EMPTY if not models.browse().get("revision") else models.BROWSE_NONE)
    self._older.set_visible(bool(entries))
    self._newer.set_visible(bool(entries))


class InstalledPage(NavScroller):
  """What the store holds, with the three things a driver can do to each model. `select` and
  `remove` are offroad-only; `rebuild` re-runs the install, which is what a fork update needs since
  nothing invalidates a build automatically (moonpilot/modelsd.py)."""

  def __init__(self, params: Params):
    super().__init__()
    self._params = params
    widgets: list = []
    self._blocks: list[tuple[BigButton, BigButton, BigButton, BigButton]] = []
    for index in range(PAGE_ROWS):
      info = BigButton("")
      info.set_click_callback(lambda index=index: self._describe(index))
      select = BigButton(models.LABEL_SELECT)
      select.set_click_callback(lambda index=index: self._toggle(index))
      rebuild = BigButton(models.LABEL_REBUILD, description=models.DESCRIPTION_STORAGE)
      rebuild.set_click_callback(lambda index=index: self._rebuild_action(index))
      remove = BigButton(models.LABEL_REMOVE, description=models.REMOVE_TEXT)
      remove.set_click_callback(lambda index=index: self._remove_action(index))
      self._blocks.append((info, select, rebuild, remove))
      widgets.extend((info, select, rebuild, remove))
    self._scroller.add_widgets(widgets)
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

  def _toggle(self, index: int) -> None:
    entry = self._entry(index)
    if entry is None:
      return
    selection = models.selection_of(entry)
    kind = str(entry.get("kind", models.DRIVING))
    if models.desired(self._params, kind) == selection:
      models.select(self._params, kind, "")
      return
    _confirm(models.SELECT_TEXT, ICON_INSTALL, lambda: models.select(self._params, kind, selection))

  def _rebuild_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is not None:
      _request(self._params, "install", models.selection_of(entry))

  def _remove_action(self, index: int) -> None:
    entry = self._entry(index)
    if entry is None:
      return
    selection = models.selection_of(entry)
    _confirm(models.REMOVE_TEXT, ICON_REMOVE, lambda: _request(self._params, "remove", selection), red=True)

  def _update_rows(self):
    offroad = ui_state.is_offroad()
    for index, (info, select, rebuild, remove) in enumerate(self._blocks):
      entry = self._entry(index)
      present = entry is not None
      for row in (info, select, rebuild, remove):
        row.set_visible(present)
        row.set_enabled(offroad)
      if entry is None:
        continue
      info.set_text(str(entry.get("name") or models.selection_of(entry)[:12]))
      info.set_value(f"{entry.get('state', '')} {models.human_size(int(entry.get('size', 0)))}".strip())
      select.set_text(models.installed_action(entry, self._params)[0])


class ComposePage(NavScroller):
  """One model out of installed pieces: a picker per role, the rule's verdict, and one confirm."""

  def __init__(self, params: Params):
    super().__init__()
    self._params = params
    self._protocols = [proto for proto in models.PROTOCOLS if any(models.compose_candidates(proto.id).values())]
    self._by_label = {models.protocol_label(proto.id): proto for proto in self._protocols}
    self._protocol_toggle = BigMultiToggle("protocol", list(self._by_label) or [models.LABEL_NONE], select_callback=lambda _value: self._update_rows())
    self._protocol_toggle.set_visible(lambda: len(self._protocols) > 1)
    self._roles = BigButton("roles")
    self._roles.set_click_callback(lambda: _describe(models.TITLE_COMPOSE, models.COMPOSE_TEXT))
    self._compose = BigButton(models.LABEL_COMPOSE, description=models.COMPOSE_TEXT)
    self._compose.set_click_callback(self._compose_action)

    # One picker per role of every protocol with candidates, built once and shown for the current
    # protocol: the scroller cannot drop a widget, and rebuilding on every protocol change would
    # leave the previous protocol's pickers behind it.
    self._pickers: dict[tuple[str, str], BigMultiToggle] = {}
    for proto in self._protocols:
      for role in proto.roles:
        candidates = models.compose_candidates(proto.id).get(role, [])
        if candidates:
          picker = BigMultiToggle(role, _labels(candidates))
          self._pickers[(proto.id, role)] = picker

    self._scroller.add_widgets([self._protocol_toggle, self._roles, *self._pickers.values(), self._compose])
    self._update_rows()

  def _protocol(self):
    return self._by_label.get(self._protocol_toggle.get_value())

  def _candidates(self, role: str) -> list[dict]:
    proto = self._protocol()
    return models.compose_candidates(proto.id).get(role, []) if proto is not None else []

  def _picks(self) -> dict[str, str]:
    proto = self._protocol()
    if proto is None:
      return {}
    picks = {}
    for role in proto.roles:
      picker = self._pickers.get((proto.id, role))
      candidates = self._candidates(role)
      if picker is None or not candidates:
        continue
      labels = _labels(candidates)
      if picker.get_value() in labels:
        picks[role] = candidates[labels.index(picker.get_value())]["recipe"]
    return picks

  def _verdict(self) -> str:
    proto = self._protocol()
    if proto is None:
      return models.LABEL_NONE
    reason = models.composition_reason(proto.id, self._picks())
    return models.COMPATIBLE if reason is None else reason

  def _compose_action(self) -> None:
    proto = self._protocol()
    if proto is None:
      return
    picks = self._picks()

    def write_and_install() -> None:
      composition, _reason = models.write_composition(proto.id, picks)
      if composition is not None:
        _request(self._params, "install", composition)

    _confirm(models.COMPOSE_TEXT, ICON_INSTALL, write_and_install)

  def show_event(self):
    super().show_event()
    self._update_rows()

  def _update_state(self):
    super()._update_state()
    self._update_rows()

  def _update_rows(self):
    proto = self._protocol()
    for (protocol_id, _role), picker in self._pickers.items():
      picker.set_visible(protocol_id == (proto.id if proto is not None else ""))
    self._compose.set_value(self._verdict())
    self._compose.set_enabled(ui_state.is_offroad() and proto is not None and models.composition_reason(proto.id, self._picks()) is None)
    if proto is None:
      self._roles.set_value(models.LABEL_NONE)
      return
    roles = models.compose_candidates(proto.id)
    self._roles.set_value(", ".join(f"{role}: {len(candidates)}" for role, candidates in roles.items()))
