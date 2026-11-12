from __future__ import annotations

from collections.abc import Callable, Sequence

import raylib_nb as rl

from openpilot.selfdrive.ui.mici.widgets.button import BigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationDialog, SettingDescriptionDialog
from openpilot.system.ui.widgets.scroller import Scroller
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget

CHEVRON_PATH = "icons/chevron_right.png"
BAR_HEIGHT = 10
LABEL_BACK = "Back"


def describe(title: str, text: str, icon: str) -> None:
  gui_app.push_widget(SettingDescriptionDialog(title, text or ".", gui_app.texture(icon, 64, 75)))


def confirm(title: str, icon: str, action, *, red: bool = False) -> None:
  gui_app.push_widget(BigConfirmationDialog(title, gui_app.texture(icon, 64, 75), action, red=red))


class SubmenuButton(BigButton):
  def __init__(self, text: str, value: str = "", *, description: str = ""):
    super().__init__(text, value, description=description)
    self._chevron = gui_app.texture(CHEVRON_PATH, 48, 48)

  def _title_width_hint(self) -> int:
    return super()._title_width_hint() - self._chevron.width - 20

  def _subtitle_width_hint(self) -> int:
    return super()._subtitle_width_hint() - self._chevron.width - 20

  def _draw_content(self, btn_y: float):
    super()._draw_content(btn_y)
    rl.draw_texture_ex(self._chevron,
                       rl.Vector2(self._rect.x + self._rect.width - self.LABEL_HORIZONTAL_PADDING - self._chevron.width,
                                  btn_y + (self._rect.height - self._chevron.height) / 2),
                       0.0, 1.0, rl.Color(255, 255, 255, int(255 * 0.9)) if self.enabled else rl.Color(255, 255, 255, 100))



class Page(Scroller):
  def __init__(self, rows: Sequence[Widget], title: str, description: str, back: Callable, on_update: Callable | None = None):
    self._on_update = on_update
    back_card = BigButton(title, LABEL_BACK, description=description)
    back_card.set_click_callback(back)
    super().__init__(horizontal=False)
    self._scroller.add_widgets([back_card, *rows])

  def _update_state(self):
    super()._update_state()
    if self._on_update is not None:
      self._on_update()


  def replace_rows(self, rows: Sequence[Widget]) -> None:
    self._scroller.items[:] = [*self._scroller.items[:1], *rows]

