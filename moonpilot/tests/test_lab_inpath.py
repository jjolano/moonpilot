import unittest

import numpy as np
from openpilot.cereal import custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

from moonpilot.lab.inpath import emulate_t_follow, raw_in_path
from moonpilot.lead import MOONPILOT_INPATH_GRID, MOONPILOT_INPATH_RC, _rise_hold, lead_in_path
from moonpilot.longitudinal import MOONPILOT_T_FOLLOW
from moonpilot.tests.test_longitudinal import _inputs, _lead, _planner


class TestInPathEmulation(unittest.TestCase):
  def test_raw_is_the_unfiltered_lead_in_path(self):
    """The lab's vectorized inPath is `lead.lead_in_path` before its filter, on leadd's own grid."""
    t = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    x = np.array([60.0, 90.0, 120.0, 150.0, 180.0, 210.0])
    y = np.array([0.4, 0.9, 5.0, 1.0, 3.0, -6.0])
    s = np.array([0.4, 2.5, 0.1, 1.2, 1.7, 1.8])
    path_x, path_y = np.linspace(0.0, 192.0, 33), np.linspace(0.0, 3.0, 33)  # right-positive, as modelV2 logs it
    real = lead_in_path(t, x, y, s, path_x, path_y, FirstOrderFilter(0.0, MOONPILOT_INPATH_RC, DT_MDL))
    grid = lambda v: np.interp(MOONPILOT_INPATH_GRID, t, v)[None]  # noqa: E731
    lead_x = grid(x)
    mine = raw_in_path(grid(y), grid(s), -np.interp(lead_x, path_x, path_y))
    self.assertAlmostEqual(float(mine[0]), real, places=9)

  def test_the_emulated_latch_is_the_planners(self):
    """Rise-hold filter, hysteresis latch and slew, frame for frame against the real `_t_follow`."""
    raw = np.array([1.0] * 5 + [0.02] * 60 + [0.2] * 5 + [0.6] * 3 + [0.0] * 50 + [0.9] * 60 + [0.05] * 70)
    n = len(raw)
    d = {
      "route": np.zeros(n), "mp_present": np.ones(n), "mp_source": np.ones(n), "lead": np.ones(n),
      "base": np.full(n, MOONPILOT_T_FOLLOW[1]),
    }
    tf, leaving = emulate_t_follow(raw, d, np.ones(n, dtype=bool))

    planner, filt = _planner(), FirstOrderFilter(1.0, MOONPILOT_INPATH_RC, DT_MDL)
    for i in range(n):
      slot = custom.MoonpilotState.LeadTrajectory.new_message()
      slot.present, slot.inPath = True, _rise_hold(float(raw[i]), filt)
      self.assertAlmostEqual(planner._t_follow(_inputs(lead=_lead(35.0, 20.0), moonpilot_leads=[slot])), tf[i], places=9, msg=f"frame {i}")
      self.assertEqual(planner.lead_leaving, bool(leaving[i]), f"frame {i}")
    self.assertTrue(leaving.any() and not leaving.all())


if __name__ == "__main__":
  unittest.main()
