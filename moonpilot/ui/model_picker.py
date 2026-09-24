"""The model picker: a tree of the catalog's own groups, shaped like sunnypilot's.

sunnypilot's picker is a dialog, not a page: the slot's current model is pinned at the top, groups
are `+`/`-` headers over indented rows, a star puts a row in favorites, a search box filters, and
Cancel/Select commit at the bottom. This is that, on upstream's own `MultiOptionDialog`, `Button`,
`Scroller` and `Keyboard` -- no new widget and no new upstream seam.

The rows and their grouping are `moonpilot/models.py`'s: this module draws them and writes the same
selection parameters the rest of the panel already writes. Nothing here decides what a model is
called or which group it belongs to -- the catalog owns that, and the fallback grouping is the
protocol the recipe needs.
"""

import math
from collections.abc import Callable, Sequence

import pyray as rl

from moonpilot import models
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import FontWeight, MousePos, TextAlignment, TextAlignmentVertical, gui_app
from openpilot.system.ui.widgets import DialogResult, Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.keyboard import Keyboard
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.option_dialog import (
  BUTTON_HEIGHT,
  BUTTON_SPACING,
  ITEM_HEIGHT,
  ITEM_SPACING,
  LIST_ITEM_SPACING,
  MARGIN,
  TITLE_FONT_SIZE,
  MultiOptionDialog,
)
from openpilot.system.ui.widgets.scroller_tici import Scroller

ROW_FONT_SIZE = 48
INDENT = 30
STAR_RADIUS = 26
STAR_HIT = 70
HINT_FONT_SIZE = 36
HINT_HEIGHT = 60
SEARCH_WIDTH = 620
STARRED = rl.Color(70, 91, 234, 255)
UNSTARRED = rl.Color(140, 140, 140, 255)
DARK = rl.Color(30, 30, 30, 255)


def draw_star(center_x: float, center_y: float, color: rl.Color) -> None:
  """A five-pointed star: the five points plus the inner pentagon, as eight triangles. `rl.draw_poly`
  wants a ctypes array of vertices, and the triangles are the same shape without one."""
  points = []
  for index in range(10):
    angle = -math.pi / 2 + index * math.pi / 5
    radius = STAR_RADIUS if index % 2 == 0 else STAR_RADIUS * 0.42
    points.append(rl.Vector2(center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
  for index in range(0, 10, 2):
    rl.draw_triangle(points[index - 1], points[index], points[(index + 1) % 10], color)
  rl.draw_triangle(points[1], points[3], points[5], color)
  rl.draw_triangle(points[1], points[5], points[7], color)
  rl.draw_triangle(points[1], points[7], points[9], color)


class Row(Button):
  """One tree row: a group header or a model. `text` is a callable, so a row keeps its own label
  current. The star is a hitbox on the right of a model row, and only model rows have one."""

  def __init__(self, text, *, indent: int = 0, starred: bool = False,
               on_select: Callable | None = None, on_star: Callable | None = None):
    super().__init__(text, click_callback=on_select, font_size=ROW_FONT_SIZE, font_weight=FontWeight.MEDIUM,
                     button_style=ButtonStyle.NORMAL, text_alignment=TextAlignment.LEFT,
                     text_padding=20 + indent * INDENT, elide_right=True)
    self.indent = indent
    self.starred = starred
    self._on_star = on_star

  def _render(self, _):
    rect = rl.Rectangle(_.x + self.indent, _.y, _.width - self.indent, _.height) if self.indent else _
    if self.indent:
      # The row draws inset, so its hitbox must be too, or a tap in the indent selects the row above.
      # That means `_rect` no longer matches what the scroller hands in, so the dialog resets every
      # row's rect each frame -- without that, the indent would compound.
      self._rect = rect
    super()._render(rect)
    if self._on_star is not None:
      draw_star(rect.x + rect.width - STAR_HIT / 2 - 10, rect.y + rect.height / 2, STARRED if self.starred else UNSTARRED)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    star = rl.Rectangle(self._rect.x + self._rect.width - STAR_HIT, self._rect.y, STAR_HIT, self._rect.height)
    if self._on_star is not None and rl.check_collision_point_rec(mouse_pos, star):
      self._on_star()
      return
    super()._handle_mouse_release(mouse_pos)


class TreeScroller(Scroller):
  """A `Scroller` whose rows can be swapped. The tree is rebuilt on every tap, and a fresh scroller
  would drop the driver's scroll position -- `show_event` resets the offset, and the dialog is shown
  once."""

  def replace(self, items: Sequence[Widget]) -> None:
    self._items = list(items)
    for item in items:
      item.set_touch_valid_callback(self.scroll_panel.is_touch_valid)

  @property
  def rows(self) -> list[Widget]:
    return self._items


class ModelTreeDialog(MultiOptionDialog):
  """The model picker for one kind. It opens on the model in effect and hands the chosen recipe to
  `on_choose` -- `""` for the bundled model -- which is what writes it, so this class never touches
  the worker's inbox itself."""

  def __init__(self, params: Params, kind: str, on_choose: Callable[[str], None], offroad: Callable[[], bool]):
    self._params = params
    self._kind = kind
    self._on_choose = on_choose
    self._offroad = offroad
    self._query = ""
    self._expanded: set[str] = set()
    self._selection: dict | None = None
    self._listed = False
    super().__init__(models.TITLE_SELECT_MODEL, [], "", FontWeight.MEDIUM, callback=self._committed)
    self.current = models.desired(params, kind)
    self._selection, self._listed = self._row_for(self.current)
    self.scroller = TreeScroller([], spacing=LIST_ITEM_SPACING)
    self.search = Button(self._search_text, click_callback=self._open_search, font_size=ROW_FONT_SIZE, font_weight=FontWeight.MEDIUM)
    self._rebuild()

  # --- the rows ---------------------------------------------------------------
  def _groups(self) -> list[tuple[str, list[dict | None]]]:
    return models.chooser_groups(self._params, self._kind)

  def _row_for(self, selection: str) -> tuple[dict | None, bool]:
    """The row for one selection, and whether the chooser still lists it at all."""
    for _, rows in self._groups():
      for row in rows:
        if ("" if row is None else models.selection_of(row)) == selection:
          return row, True
    return None, False

  def _rebuild(self) -> None:
    """The visible rows, from the groups, the query and the expansion state."""
    rows: list[Widget] = []
    starred = models.favorites(self._params)
    if self._listed and models.matches_query(self._selection, self._query, self._kind):
      # The model in effect leads the picker, as it leads sunnypilot's, so a driver never has to
      # expand anything to see what is driving.
      rows.append(self._model_row(self._selection, 0, starred))
    for label, group_rows in models.search_groups(self._groups(), self._query, self._kind):
      if not label:
        # The unnamed group is the bundled model, already pinned above.
        rows += [self._model_row(row, 0, starred) for row in group_rows if self._current_selection() != ("" if row is None else models.selection_of(row))]
        continue
      open_ = self._is_open(label)
      rows.append(Row(lambda label=label, open_=open_: f"{'-' if open_ else '+'} {label}", on_select=lambda label=label: self._toggle(label)))
      if open_:
        rows += [self._model_row(row, 1, starred) for row in group_rows]
    self.scroller.replace(rows)

  def _model_row(self, entry: dict | None, indent: int, starred: set[str]) -> Row:
    selection = "" if entry is None else models.selection_of(entry)
    row = Row(lambda entry=entry: models.entry_display(entry, self._kind), indent=indent,
              starred=selection in starred, on_select=lambda entry=entry: self._pick(entry),
              on_star=None if entry is None else lambda entry=entry: self._star(entry))
    row.set_button_style(ButtonStyle.PRIMARY if selection == self._current_selection() else ButtonStyle.NORMAL)
    return row

  def _current_selection(self) -> str:
    """The selection the highlight follows. By string, not by object: an installed entry is
    re-merged from the index on every read, so the row a tap handed us is not the row the next
    rebuild produces."""
    return "" if self._selection is None else models.selection_of(self._selection)

  # --- state ------------------------------------------------------------------
  def _is_open(self, label: str) -> bool:
    return label in self._expanded or bool(self._query.strip())

  def _toggle(self, label: str) -> None:
    self._expanded.symmetric_difference_update({label})
    self._rebuild()

  def _pick(self, entry: dict | None) -> None:
    self._selection = entry
    self._rebuild()

  def _star(self, entry: dict | None) -> None:
    if entry is None:
      return
    selection = models.selection_of(entry)
    models.set_favorite(self._params, selection, selection not in models.favorites(self._params))
    self._rebuild()

  def _search_text(self) -> str:
    return f"{models.LABEL_SEARCH}: {self._query}" if self._query else models.LABEL_SEARCH

  def _open_search(self) -> None:
    keyboard = Keyboard(max_text_size=64)
    keyboard.set_title(models.TITLE_SELECT_MODEL, models.DESCRIPTION_BROWSE)
    keyboard.set_text(self._query)

    def typed(result) -> None:
      if result == DialogResult.CONFIRM:
        self._query = keyboard.text.strip()
        self._rebuild()

    keyboard.set_callback(typed)
    gui_app.push_widget(keyboard)

  def _committed(self, result) -> None:
    if result == DialogResult.CONFIRM and self._listed:
      self._on_choose("" if self._selection is None else models.selection_of(self._selection))

  def _selectable(self) -> bool:
    """Whether Select is live: something other than the model already in effect, and a built model
    only while offroad -- a download may start onroad, changing what drives may not."""
    if not self._listed:
      return False
    if ("" if self._selection is None else models.selection_of(self._selection)) == self.current:
      return False
    return self._offroad() or self._selection is None or self._selection.get("state") != "built"

  def _hint(self) -> str:
    if not self._listed:
      return models.BROWSE_NONE
    return models.chooser_hint(self._selection, self._kind, self._params, self._offroad())

  # --- rendering --------------------------------------------------------------
  def _render(self, rect):
    dialog = rl.Rectangle(rect.x + MARGIN, rect.y + MARGIN, rect.width - 2 * MARGIN, rect.height - 2 * MARGIN)
    rl.draw_rectangle_rounded(dialog, 0.02, 20, DARK)
    content = rl.Rectangle(dialog.x + MARGIN, dialog.y + MARGIN, dialog.width - 2 * MARGIN, dialog.height - 2 * MARGIN)
    buttons_y = content.y + content.height - BUTTON_HEIGHT

    gui_label(rl.Rectangle(content.x, content.y, content.width - SEARCH_WIDTH, TITLE_FONT_SIZE),
              self.title, TITLE_FONT_SIZE, font_weight=FontWeight.BOLD)
    self.search.render(rl.Rectangle(content.x + content.width - SEARCH_WIDTH, content.y, SEARCH_WIDTH, 100))

    gui_label(rl.Rectangle(content.x, buttons_y - HINT_HEIGHT - ITEM_SPACING, content.width, HINT_HEIGHT), self._hint(),
              HINT_FONT_SIZE, color=rl.Color(228, 228, 228, 160), alignment=TextAlignment.LEFT,
              alignment_vertical=TextAlignmentVertical.MIDDLE)

    options_y = content.y + TITLE_FONT_SIZE + ITEM_SPACING
    options = rl.Rectangle(content.x, options_y, content.width, buttons_y - HINT_HEIGHT - ITEM_SPACING * 2 - options_y)
    for row in self.scroller.rows:
      row.set_rect(rl.Rectangle(0, 0, options.width, ITEM_HEIGHT))
    self.scroller.render(options)

    width = (content.width - BUTTON_SPACING) / 2
    self.cancel_button.render(rl.Rectangle(content.x, buttons_y, width, BUTTON_HEIGHT))
    self.select_button.set_enabled(self._selectable())
    self.select_button.render(rl.Rectangle(content.x + width + BUTTON_SPACING, buttons_y, width, BUTTON_HEIGHT))

  def show_event(self):
    super().show_event()
    self.search.show_event()
    self.scroller.show_event()
