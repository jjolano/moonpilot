"""mici sign-in dialog: the tailscale login URL as a QR code, plus the URL itself as text.

Same job as moonpilot/ui/tailscale_qr.py on the mici tree, which renders on the dark theme, hence
`inverted=True` — the same thing upstream's mici pairing dialog passes.

    OPENPILOT_PREFIX=tssmoke python3 moonpilot/ui/tailscale_qr_mici.py
"""

import pyray as rl

from openpilot.common.qrcode import make_texture
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.nav_widget import NavWidget

from moonpilot import tailscale


class TailscaleSignInDialogMici(NavWidget):
  """Full-screen QR dialog for the pending tailscale login, on the mici nav stack."""

  def __init__(self):
    super().__init__()
    self._url = ""
    self._qr: rl.Texture | None = None
    self._title = UnifiedLabel("sign in to tailscale", font_size=48, font_weight=FontWeight.BOLD, line_height=0.8)
    # UnifiedLabel resolves a callable on every render, so the URL needs no refresh code.
    self._url_label = UnifiedLabel(lambda: tailscale.auth_url(ui_state.params), font_size=32, font_weight=FontWeight.ROMAN)

  def _update_qr(self, url: str) -> None:
    # Driven by the URL string, not a timer: a timer would leave a stale code on screen.
    if url == self._url:
      return
    self._url = url

    if self._qr is not None and self._qr.id != 0:
      rl.unload_texture(self._qr)
      self._qr = None

    if not url:
      return
    try:
      self._qr = make_texture(url, inverted=True)
    except Exception:
      cloudlog.exception("tailscale QR generation failed")
      self._qr = None

  def _update_state(self):
    super()._update_state()
    if not tailscale.auth_url(ui_state.params) and not self.is_dismissing:
      self.dismiss()

  def _render(self, rect: rl.Rectangle):
    self._update_qr(tailscale.auth_url(ui_state.params))
    self._render_qr()

    label_x = self._rect.x + 8 + self._rect.height + 24
    self._title.set_max_width(int(self._rect.width - label_x))
    self._title.set_position(label_x, self._rect.y + 16)
    self._title.render()

    self._url_label.set_max_width(int(self._rect.width - label_x))
    self._url_label.set_position(label_x, self._rect.y + 16 + self._title.rect.height + 16)
    self._url_label.render()

  def _render_qr(self) -> None:
    if self._qr is None:
      # The URL text beside the code is still usable, so say what happened instead of a blank box.
      rl.draw_text_ex(
        gui_app.font(FontWeight.BOLD), "QR unavailable", rl.Vector2(self._rect.x + 20, self._rect.y + self._rect.height // 2 - 15), 30, 0.0, rl.RED
      )
      return

    scale = self._rect.height / self._qr.height
    rl.draw_texture_ex(self._qr, rl.Vector2(round(self._rect.x + 8), round(self._rect.y)), 0.0, scale, rl.WHITE)

  def __del__(self):
    if self._qr is not None and self._qr.id != 0:
      rl.unload_texture(self._qr)


class _Background(NavWidget):
  """Stand-in for the settings panel this dialog is pushed over when running standalone.

  Without it the nav stack holds only the dialog, and pop_widget() refuses to empty the stack
  (application.py:412-415) — which is where NavWidget.dismiss() ends up — so the self-closing path
  this dialog exists for cannot be watched.
  """

  def _render(self, rect: rl.Rectangle):
    rl.clear_background(rl.BLACK)
    return -1


if __name__ == "__main__":
  gui_app.init_window("tailscale sign in")
  gui_app.push_widget(_Background())
  dialog = TailscaleSignInDialogMici()
  gui_app.push_widget(dialog)
  try:
    for _ in gui_app.render():
      pass
  finally:
    del dialog
