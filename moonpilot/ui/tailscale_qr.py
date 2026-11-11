"""Tizi sign-in dialog: a QR code of the tailscale login URL, plus the URL as text."""


import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import IconButton

from moonpilot import tailscale
from moonpilot.ui.panel_common import QrTexture, render_qr

INSTRUCTIONS = ("Scan this code with your phone", "Or open this link: {}", "Approve this device in your tailnet if your tailnet requires it")



class TailscaleSignInDialog(QrTexture, Widget):
  """Full-screen QR dialog for the pending tailscale login."""

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._close_btn = IconButton(gui_app.texture("icons/close.png", 80, 80))
    self._close_btn.set_click_callback(gui_app.pop_widget)

  def _update_state(self):
    if not tailscale.auth_url(self._params):
      gui_app.pop_widget()

  def _render(self, rect: rl.Rectangle) -> int:
    # The light background the tizi pairing dialog uses, which is what makes the default
    # black-on-white code scannable.
    rl.clear_background(rl.Color(224, 224, 224, 255))
    url = tailscale.auth_url(self._params)
    self._update_qr(url)

    margin = 70
    content = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - 2 * margin, rect.height - 2 * margin)
    y = content.y

    close_size, pad = 80, 20
    self._close_btn.render(rl.Rectangle(content.x - pad, y - pad, close_size + pad * 2, close_size + pad * 2))
    y += close_size + 40

    title_font = gui_app.font(FontWeight.NORMAL)
    left_width = int(content.width * 0.5 - 15)
    title = wrap_text(title_font, "Sign in to tailscale", 75, left_width)
    rl.draw_text_ex(title_font, "\n".join(title), rl.Vector2(content.x, y), 75, 0.0, rl.BLACK)
    y += len(title) * 75 + 60

    right_width = content.width // 2 - 20
    self._render_instructions(
      rl.Rectangle(content.x, y, left_width, content.height - (y - content.y)), [text.format(url) for text in INSTRUCTIONS]
    )

    qr_size = min(right_width, content.height) - 40
    render_qr(self._qr, rl.Rectangle(content.x + left_width + 40 + (right_width - qr_size) // 2, content.y, qr_size, qr_size))
    return -1

  def _render_instructions(self, rect: rl.Rectangle, instructions: list[str]) -> None:
    font = gui_app.font(FontWeight.BOLD)
    y = rect.y

    for i, text in enumerate(instructions):
      radius = 25
      circle_x = rect.x + radius + 15
      text_x = rect.x + radius * 2 + 40
      wrapped = wrap_text(font, text, 47, int(rect.width - (radius * 2 + 40)))
      text_height = len(wrapped) * 47
      circle_y = y + text_height // 2

      rl.draw_circle(int(circle_x), int(circle_y), radius, rl.Color(70, 70, 70, 255))
      number = str(i + 1)
      number_size = measure_text_cached(font, number, 30)
      rl.draw_text_ex(font, number, (int(circle_x - number_size.x // 2), int(circle_y - number_size.y // 2)), 30, 0, rl.WHITE)
      rl.draw_text_ex(font, "\n".join(wrapped), rl.Vector2(text_x, y), 47, 0.0, rl.BLACK)
      y += text_height + 50





