from moonpilot.features import FEATURES, Feature, enabled
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import text_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


def _feature_toggle(feature: Feature, params: Params):
  return toggle_item(
    feature.title,
    description=feature.description,
    initial_state=enabled(feature, params),
    callback=lambda state, key=feature.key: params.put_bool(key, state, block=True),
    enabled=ui_state.is_offroad if feature.offroad_only else True,
  )


class MoonpilotLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._scroller = Scroller([
      *(_feature_toggle(feature, self._params) for feature in FEATURES),
      text_item("version", self._params.get("Version") or "N/A"),
      text_item("branch", self._params.get("GitBranch") or "N/A"),
      text_item("commit", (self._params.get("GitCommit") or "N/A")[:8]),
    ], line_separator=True, spacing=0)

  def _render(self, rect):
    self._scroller.render(rect)
