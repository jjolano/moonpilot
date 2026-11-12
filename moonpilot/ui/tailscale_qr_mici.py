"""mici sign-in dialog: the tailscale login URL as a QR code, plus the URL itself as text.

The shared QR texture and unavailable-state drawing live in `moonpilot/ui/panel_common.py`; the mici
dialog inverts the code for its dark background.

    OPENPILOT_PREFIX=tssmoke python3 moonpilot/ui/tailscale_qr_mici.py
"""

import raylib_nb as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight
from openpilot.system.ui.widgets.label import UnifiedLabel
from openpilot.system.ui.widgets.nav_widget import NavWidget

from moonpilot import tailscale
from moonpilot.ui.panel_common import QrTexture, render_qr


class TailscaleSignInDialogMici(QrTexture, NavWidget):
  """Full-screen QR dialog for the pending tailscale login, on the mici nav stack."""

  _qr_inverted = True

  def __init__(self):
    super().__init__()
    self._title = UnifiedLabel("sign in to tailscale", font_size=48, font_weight=FontWeight.BOLD, line_height=0.8)
    # UnifiedLabel resolves a callable on every render, so the URL needs no refresh code.
    self._url_label = UnifiedLabel(lambda: tailscale.auth_url(ui_state.params), font_size=32, font_weight=FontWeight.ROMAN)

  def _update_state(self):
    super()._update_state()
    if not tailscale.auth_url(ui_state.params) and not self.is_dismissing:
      self.dismiss()

  def _render(self, rect: rl.Rectangle):
    self._update_qr(tailscale.auth_url(ui_state.params))
    render_qr(self._qr, rl.Rectangle(self._rect.x + 8, self._rect.y, self._rect.height, self._rect.height))

    label_x = self._rect.x + 8 + self._rect.height + 24
    self._title.set_max_width(int(self._rect.width - label_x))
    self._title.set_position(label_x, self._rect.y + 16)
    self._title.render()

    self._url_label.set_max_width(int(self._rect.width - label_x))
    self._url_label.set_position(label_x, self._rect.y + 16 + self._title.rect.height + 16)
    self._url_label.render()





