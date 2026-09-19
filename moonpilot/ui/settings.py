"""moonpilot settings panel. Rows come from moonpilot/features.py, so a feature shows up
here by existing in that table, and a feature whose dependencies are still missing reads
as unavailable instead of silently doing nothing.

The panel itself is the root: what this device is running (the models row), one row per *group* of
features, and the two device actions that own their own live labels. A group pushes its own page,
which is the same rows the flat panel used to carry -- the toggle, its description and its gates are
unchanged, so a feature moved between groups is a table edit and nothing else.
"""

from moonpilot import tailscale
from moonpilot.engage import car_unavailable_reason
from moonpilot.features import FEATURES, Group, GROUPS, Feature, missing_modules, version, wanted
from moonpilot.ui import models as models_ui
from moonpilot.ui import offroad_mode
from moonpilot.ui.tailscale_qr import TailscaleSignInDialog
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import button_item, text_item, toggle_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


def _unavailable_reason(feature: Feature) -> str | None:
  """Why this feature cannot run, or None. Two gates, one message: the dependencies the device is
  still missing, and the car itself."""
  modules = missing_modules(feature)
  if modules:
    return f"{feature.title} is currently unavailable until {', '.join(modules)} is installed, which happens automatically once the device is online."
  car = car_unavailable_reason(feature, ui_state.CP)
  return f"{feature.title} is unavailable on this car: {car}." if car else None


def _description(feature: Feature) -> str:
  reason = _unavailable_reason(feature)
  if reason is None:
    return feature.description
  # Upstream's disabled-with-reason shape: bold reason, then the description (toggles.py).
  return "<b>" + reason + "</b><br><br>" + feature.description


def _feature_toggle(feature: Feature, params: Params):
  # title/description/enabled are re-resolved every render, so callables are all the refresh
  # this panel needs. The pill follows wanted(), not enabled(): an unavailable feature the
  # driver asked for still reads as on, like upstream's disabled-but-on toggles.
  return toggle_item(
    feature.title,
    description=lambda f=feature: _description(f),
    initial_state=wanted(feature, params),
    callback=lambda state, key=feature.key: params.put_bool(key, state, block=True),
    enabled=lambda f=feature: _unavailable_reason(f) is None and (not f.offroad_only or ui_state.is_offroad()),
  )


def _tailscale_row(params: Params):
  # One row for tailscale's state and its sign-in: text, description and enabled are all
  # re-resolved every render, so it reads SIGN IN and is tappable exactly while a login URL
  # exists, and shows the state dimmed out otherwise. The description is where the URL and
  # the explanatory line go — it wraps and expands on tap, which a right-aligned value
  # would clip. Same shape as upstream's Pair Device row.
  return button_item(
    "tailscale",
    lambda: "SIGN IN" if tailscale.auth_url(params) else tailscale.status_text(params)[0].upper(),
    description=lambda: tailscale.status_text(params)[1],
    callback=lambda: gui_app.push_widget(TailscaleSignInDialog()),
    enabled=lambda: bool(tailscale.auth_url(params)),
  )


class GroupLayout(Widget):
  """One group's page: the group's own name and description, then its toggles. Pushed, so the way
  back is the leading row, and the toggles are the same objects the flat panel used to carry."""

  def __init__(self, group: Group, params: Params):
    super().__init__()
    self._scroller = Scroller(
      [
        button_item(group.title, "BACK", description=group.description, callback=lambda: gui_app.pop_widget()),
        *(_feature_toggle(feature, params) for feature in group.features),
      ],
      line_separator=True,
      spacing=0,
    )

  def _render(self, rect):
    self._scroller.render(rect)


class MoonpilotLayout(Widget):
  def __init__(self):
    super().__init__()
    self._params = ui_state.params
    self._scroller = Scroller(
      [
        text_item("version", version()),
        models_ui.row(self._params),
        *[
          button_item(
            group.title,
            "OPEN",
            description=group.description,
            callback=lambda group=group, params=self._params: gui_app.push_widget(GroupLayout(group, params)),
          )
          for group in GROUPS
        ],
        offroad_mode.row(self._params),
        _tailscale_row(self._params),
      ],
      line_separator=True,
      spacing=0,
    )
    # Every feature row lives on a group page now; this is the one place that would notice a
    # feature added to FEATURES but to no group (it cannot happen -- FEATURES is the flattening).
    assert FEATURES, GROUPS

  def _render(self, rect):
    self._scroller.render(rect)
