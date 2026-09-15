from openpilot.selfdrive.ui.mici.widgets.button import GreyBigButton
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets.scroller import NavScroller


class MoonpilotLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    params = ui_state.params
    self._scroller.add_widgets([
      GreyBigButton("version", params.get("Version") or "N/A"),
      GreyBigButton("branch", params.get("GitBranch") or "N/A"),
      GreyBigButton("commit", (params.get("GitCommit") or "N/A")[:8]),
    ])
