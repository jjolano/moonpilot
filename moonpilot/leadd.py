#!/usr/bin/env python3
"""Publish moonpilotState: lead trajectories, the ego-motion correction, and the ego pose.

One publisher owns the service. The loop publishes leads, ego correction, and pose; failures in the
window update are isolated from the other fields.
"""

import time

from openpilot.cereal import messaging
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.swaglog import cloudlog

from moonpilot import features
from moonpilot.lead import LeadSlot
from moonpilot.pose import PoseTracker, fill_ego_pose, push_odometry_sample
from moonpilot.slam import PriorChannel, RotatingPoseWindow, device_from_calib, fill_ego_correction, invalid, odometry_sample, slam_node
RADAR_LEAD_SLOT_COUNT = len(ModelConstants.LEAD_T_OFFSETS) - 1





def main() -> None:
  params = Params()
  gps_service = get_gps_location_service(params)
  sm = messaging.SubMaster(
    ['modelV2', 'radarState', 'carState', 'cameraOdometry', 'extrinsicsCalibration', gps_service],
    poll='modelV2',
  )
  pm = messaging.PubMaster(['moonpilotState'])
  window = RotatingPoseWindow()
  # The wheel-speed prior, fed with its own message times and read at the odometry's pose time.
  prior = PriorChannel()
  pose = PoseTracker()
  lead_slots = tuple(LeadSlot(i) for i in range(len(ModelConstants.LEAD_T_OFFSETS)))


  while True:
    sm.update()

    # Keep each new carState sample for interpolation at the odometry pose time.
    if sm.updated['carState']:
      prior.push(sm.logMonoTime['carState'] * 1e-9, float(sm['carState'].vEgo))

    if sm.updated[gps_service]:
      g = sm[gps_service]
      pose.push_gps(
        float(g.latitude), float(g.longitude), horizontal_accuracy=float(g.horizontalAccuracy),
        has_fix=bool(g.hasFix), bearing_deg=float(g.bearingDeg), speed=float(g.speed),
      )
    sample = None
    if sm.updated['cameraOdometry']:
      calibration = sm['extrinsicsCalibration']
      rotation = device_from_calib(calibration.rpyCalib, sm.valid['extrinsicsCalibration'])
      sample = odometry_sample(sm['cameraOdometry'], prior, rotation, sm.valid['cameraOdometry'])
      if sample is not None:
        push_odometry_sample(pose, sample)

    if not sm.updated['modelV2']:
      continue

    # The window is fed whether or not the correction is wanted, so toggling it on is immediate
    # rather than waiting five seconds for a window to fill. `enabled`, not the raw param: a feature
    # whose dependencies are missing is off, and this is where that has to be asked.
    if sample is not None:
      node = slam_node(sample)
      if node is not None:
        window.push(node)

    slam_on = features.enabled(features.SLAM, params)
    corr = invalid()
    if slam_on:
      try:
        corr = window.update()
      except Exception:
        # Per-frame, so one bad frame costs one correction. A dead daemon is gone until the next
        # boot (the manager does not restart a process that exited), which is strictly worse.
        cloudlog.exception("moonpilot leadd: the ego-motion window update failed")

    model = sm['modelV2']
    radar = sm['radarState'] if sm.valid['radarState'] else None
    ego_path_x = model.position.x
    ego_path_y = model.position.y

    msg = messaging.new_message('moonpilotState')
    # The two services the leads need, named explicitly: cameraOdometry and carState are this
    # process's own business and must not be able to take the leads down with them.
    msg.valid = sm.all_checks(['modelV2', 'radarState'])

    leads = msg.moonpilotState.init('leads', len(lead_slots))
    for i, slot in enumerate(leads):
      model_lead = model.leadsV3[i] if i < len(model.leadsV3) else None
      fused_lead = (radar.leadOne, radar.leadTwo)[i] if radar is not None and i < RADAR_LEAD_SLOT_COUNT else None
      for field, value in lead_slots[i].normalize(model_lead, fused_lead, ego_path_x, ego_path_y).items():
        setattr(slot, field, value)

    fill_ego_correction(msg, corr)
    # Same toggle as the correction: one SLAM row, both observation fields. Off leaves egoPose at
    # its default (valid false), which is the pass-through case.
    if slam_on:
      fill_ego_pose(msg, pose, time.monotonic())
    pm.send('moonpilotState', msg)


if __name__ == "__main__":
  main()
