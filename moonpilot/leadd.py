#!/usr/bin/env python3
"""Publish moonpilotState: the model's three lead trajectories, normalized.

Observation only — nothing it publishes is in the control path, so no config_realtime_process.
"""

from openpilot.cereal import messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

from moonpilot.lead import MOONPILOT_INPATH_RC, normalize_lead

N_LEADS = 3


def main() -> None:
  sm = messaging.SubMaster(['modelV2', 'radarState'], poll='modelV2')
  pm = messaging.PubMaster(['moonpilotState'])

  # Owned here and created once: per-frame filters would silently drop the smoothing.
  in_path_filters = [FirstOrderFilter(1.0, MOONPILOT_INPATH_RC, DT_MDL) for _ in range(N_LEADS)]

  while True:
    sm.update()
    if not sm.updated['modelV2']:
      continue

    model = sm['modelV2']
    radar = sm['radarState'] if sm.valid['radarState'] else None
    ego_path_x = model.position.x
    ego_path_y = model.position.y

    msg = messaging.new_message('moonpilotState')
    msg.valid = sm.all_checks()

    leads = msg.moonpilotState.init('leads', N_LEADS)
    for i, slot in enumerate(leads):
      model_lead = model.leadsV3[i] if i < len(model.leadsV3) else None
      fused_lead = None
      if radar is not None:
        fused_lead = (radar.leadOne, radar.leadTwo)[i] if i < 2 else None

      for field, value in normalize_lead(i, model_lead, fused_lead, ego_path_x, ego_path_y, in_path_filters[i]).items():
        setattr(slot, field, value)

    pm.send('moonpilotState', msg)


if __name__ == "__main__":
  main()
