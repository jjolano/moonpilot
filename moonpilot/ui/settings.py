from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import text_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


class MoonpilotLayout(Widget):
  def __init__(self):
    super().__init__()
    params = ui_state.params
    self._scroller = Scroller([
      text_item("version", params.get("Version") or "N/A"),
      text_item("branch", params.get("GitBranch") or "N/A"),
      text_item("commit", (params.get("GitCommit") or "N/A")[:8]),
    ], line_separator=True, spacing=0)

  def _render(self, rect):
    self._scroller.render(rect)
