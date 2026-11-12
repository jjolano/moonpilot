import numpy as np
import raylib_nb as rl
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.mici.onroad import blend_colors
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer as BaseModelRenderer, STOCK_PATH_GRADIENT
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.lib.polygon import draw_polygon
from moonpilot.ui.onroad import moonpilot_status_color  # moonpilot seam, see AGENTS.md

LANE_LINE_COLORS = {
  UIStatus.DISENGAGED: rl.Color(200, 200, 200, 255),
  UIStatus.OVERRIDE: rl.Color(255, 255, 255, 255),
  UIStatus.ENGAGED: rl.Color(0, 255, 64, 255),
}


class ModelRenderer(BaseModelRenderer):
  LANE_LINE_WIDTHS = (0.12, 0.16, 0.16, 0.16)
  ROAD_EDGE_WIDTH = 0.16

  def __init__(self):
    super().__init__()
    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)

  def _draw_offset(self) -> np.ndarray:
    return np.array([self._rect.x, self._rect.y], dtype=np.float32)

  def _render(self, rect: rl.Rectangle):
    self._torque_filter.update(-ui_state.sm['carOutput'].actuatorsOutput.torque)
    super()._render(rect)

  def _get_ll_color(self, prob: float, adjacent: bool, left: bool):
    alpha = np.clip(prob, 0.0, 0.7)
    if adjacent:
      base_color = (moonpilot_status_color(ui_state, LANE_LINE_COLORS) or
                    LANE_LINE_COLORS.get(ui_state.status, LANE_LINE_COLORS[UIStatus.DISENGAGED]))
      color = rl.Color(base_color.r, base_color.g, base_color.b, int(alpha * 255))
      torque = self._torque_filter.x
      if abs(torque) > 0.6 and left == (torque > 0):
        color = blend_colors(color, rl.Color(255, 115, 0, int(alpha * 255)), np.interp(abs(torque), [0.6, 0.8], [0.0, 1.0]))
    else:
      color = rl.Color(255, 255, 255, int(alpha * 255))

    if ui_state.status == UIStatus.DISENGAGED:
      color = rl.Color(0, 0, 0, int(alpha * 255))
    return color

  def _draw_lane_lines(self):
    offset = self._draw_offset()
    for i, lane_line in enumerate(self._lane_lines):
      if lane_line.projected_points.size:
        color = self._get_ll_color(float(self._lane_line_probs[i]), i in (1, 2), i in (0, 1))
        draw_polygon(self._rect, lane_line.projected_points + offset, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size:
        color = self._get_ll_color(float(1.0 - self._road_edge_stds[i]), float(self._lane_line_probs[i + 1]) < 0.25, i == 0)
        draw_polygon(self._rect, road_edge.projected_points + offset, color)

  def _draw_path(self):
    if not self._path.projected_points.size:
      return

    path_pts = self._path.projected_points + self._draw_offset()
    if ui_state.status == UIStatus.DISENGAGED:
      draw_polygon(self._rect, path_pts, rl.Color(0, 0, 0, 90))
    elif not self._longitudinal_control:
      draw_polygon(self._rect, path_pts, gradient=STOCK_PATH_GRADIENT)
    elif len(self._path_gradient.colors) > 1:
      draw_polygon(self._rect, path_pts, gradient=self._path_gradient)
    else:
      draw_polygon(self._rect, path_pts, rl.Color(255, 255, 255, 30))

