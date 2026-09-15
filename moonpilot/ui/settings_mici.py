from moonpilot.features import FEATURES, Feature, enabled
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, GreyBigButton
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.widgets.scroller import NavScroller


def _feature_button(feature: Feature, params: Params):
  button = BigParamControl(feature.title, feature.key, description=feature.description)
  # BigParamControl reads get_bool, which ignores the declared default, so set the
  # initial state from the same read the feature itself uses.
  button.set_checked(enabled(feature, params))
  return button


class MoonpilotLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()
    params = ui_state.params
    self._scroller.add_widgets([
      *(_feature_button(feature, params) for feature in FEATURES),
      GreyBigButton("version", params.get("Version") or "N/A"),
      GreyBigButton("branch", params.get("GitBranch") or "N/A"),
      GreyBigButton("commit", (params.get("GitCommit") or "N/A")[:8]),
    ])
