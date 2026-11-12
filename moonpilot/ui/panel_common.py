import raylib_nb as rl

from moonpilot import offroad
from moonpilot.features import Feature, car_reason, missing_modules
from openpilot.common.qrcode import make_texture
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets import Widget


class QrTexture:
  _qr_inverted = False
  _url = ""
  _qr: "rl.Texture | None" = None

  def _update_qr(self, url: str) -> None:
    if url == self._url:
      return
    self._url = url
    if self._qr is not None and self._qr.id != 0:
      rl.unload_texture(self._qr)
      self._qr = None
    if url:
      try:
        self._qr = make_texture(url, inverted=self._qr_inverted)
      except Exception:
        cloudlog.exception("tailscale QR generation failed")
        self._qr = None

  def __del__(self):
    if self._qr is not None and self._qr.id != 0:
      rl.unload_texture(self._qr)


def render_qr(texture, rect: rl.Rectangle) -> None:
  if texture is None:
    rl.draw_text_ex(gui_app.font(FontWeight.BOLD), "QR unavailable",
                    rl.Vector2(rect.x + 20, rect.y + rect.height / 2 - 15), 30, 0.0, rl.RED)
    return
  rl.draw_texture_pro(texture, rl.Rectangle(0, 0, texture.width, texture.height), rect, rl.Vector2(0, 0), 0, rl.WHITE)

BAR_TRACK = rl.Color(57, 57, 57, 255)
BAR_FILL = rl.Color(51, 171, 76, 255)


def feature_enabled(feature: Feature, parked_only: bool) -> bool:
  return not missing_modules(feature) and not car_reason(feature, ui_state.CP) and (not parked_only or ui_state.is_offroad())


def parked() -> bool:
  return offroad.can_enter(ui_state.started, ui_state.sm["carState"].vEgo, ui_state.engaged)


def toggle(params) -> None:
  offroad.set_requested(params, not offroad.requested(params), ui_state.started, ui_state.sm["carState"].vEgo, ui_state.engaged)


def enter() -> None:
  offroad.set_requested(ui_state.params, True, ui_state.started, ui_state.sm["carState"].vEgo, ui_state.engaged)


class PageStack(Widget):
  def __init__(self):
    super().__init__()
    self._pages: list[Widget] = []

  def set_root(self, page: Widget) -> None:
    self._pages = [page]

  def open(self, page: Widget) -> None:
    if self._pages:
      self._pages[-1].hide_event()
    self._pages.append(page)
    page.show_event()

  def back(self) -> None:
    if len(self._pages) <= 1:
      return
    self._pages.pop().hide_event()
    self._pages[-1].show_event()

  def show_event(self):
    super().show_event()
    if self._pages:
      self._pages[-1].show_event()

  def hide_event(self):
    if self._pages:
      self._pages[-1].hide_event()
    super().hide_event()

  def _render(self, rect):
    if self._pages:
      self._pages[-1].render(rect)
