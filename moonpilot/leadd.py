#!/usr/bin/env python3
"""Publish moonpilotState: lead trajectories, the ego-motion correction, and the ego pose.

Observation only — nothing it publishes is in the control path, so no config_realtime_process.

One publisher, by construction: msgq allows exactly one, and a second one kills the first with
EADDRINUSE on its next send (msgq's own `test_multiple_publishers_exception`). So this process owns
the service and fills all three of its parts, and each is gated in-loop rather than by a second
process — leads are published whether the correction or the pose is on or off, and a smoother or
tracker that throws costs that frame's field and nothing else.
"""

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.locationd.helpers import rotate_std
from openpilot.selfdrive.locationd.locationd import CALIB_RPY_SANITY_CHECK

from moonpilot import features
from moonpilot.lead import MOONPILOT_INPATH_RC, MOONPILOT_LEAD_PROB_RC, normalize_lead
from moonpilot.pose import PoseTracker, fill_ego_pose
from moonpilot.slam import MOONPILOT_SLAM_POSE_DELAY, Node, PriorChannel, RotatingPoseWindow, fill_ego_correction, invalid

N_LEADS = 3


def _device_from_calib(sm):
  try:
    if not sm.valid['extrinsicsCalibration']:
      return None
    rpy_calib = np.asarray(sm['extrinsicsCalibration'].rpyCalib, dtype=float)
  except (AttributeError, KeyError, TypeError, ValueError):
    return None

  if rpy_calib.shape != (3,) or not np.all(np.isfinite(rpy_calib)) or np.any(np.abs(rpy_calib) > CALIB_RPY_SANITY_CHECK):
    return None
  if not np.any(rpy_calib):
    return None
  return rot_from_euler(rpy_calib)


def _slam_node(sm, prior: PriorChannel) -> Node | None:
  """One window sample, or None when this frame has no interval to close.

  The odometry's own `valid` is the gate that matters here -- `fill_pose_msg` sets it from extrinsics
  calibration and dropped frames, so it means the pose is trustworthy. `carState`'s does not:
  `card.py` sets it from `CS.canValid`, a CAN-health flag, so a bus hiccup would silently kill the
  correction; the prior's own freshness comes from the pose-time read instead, which is a stricter
  test than a timestamp compare. The latest trusted extrinsics calibration is applied to each new
  sample; older window nodes are not re-derived if it changes.
  """
  if not (sm.updated['cameraOdometry'] and sm.valid['cameraOdometry']):
    return None

  odometry = sm['cameraOdometry']
  if min(len(odometry.trans), len(odometry.transStd)) < 3:
    return None

  # The pose the odometry describes is one MOONPILOT_SLAM_POSE_DELAY before the frame's *end of
  # exposure*, so the prior is read there and the node is stamped there: both sides of the interval
  # then cover the same span. `timestampEof`, not `logMonoTime`, for the same reason locationd uses
  # it (`locationd.py:162`): the publish time trails the exposure by modeld's own inference and send
  # latency -- a measured 30.5 ms median (p95 33.5, max 200) over 13k corpus frames -- and pairing
  # the prior with that instead leaves 30 % of the very `a * dt` error this read exists to remove,
  # correlated -0.85 with `aEgo` and worth +0.019 m/s of under-correction while braking.
  mono_time = odometry.timestampEof * 1e-9 - MOONPILOT_SLAM_POSE_DELAY
  v_ego = prior.at(mono_time)
  if v_ego is None:
    return None

  trans, trans_std = odometry.trans, odometry.transStd
  device_from_calib = _device_from_calib(sm)
  if device_from_calib is not None:
    trans = device_from_calib @ np.asarray(odometry.trans, dtype=float)[:3]
    trans_std = rotate_std(device_from_calib, np.asarray(odometry.transStd, dtype=float)[:3])

  return Node(mono_time=mono_time, v_ego=v_ego, trans_x=float(trans[0]), trans_std_x=float(trans_std[0]))


def _pose_push(tracker: PoseTracker, sm, prior: PriorChannel) -> None:
  """One dead-reckon step at the odometry's pose time, or nothing when this frame has none.

  Same stamp and prior read as `_slam_node`: the pose the odometry describes is
  `timestampEof - MOONPILOT_SLAM_POSE_DELAY`. Wheel forward when the prior covers it (the plan
  every other consumer trusts); the VO's own forward only as the bare-replay fallback `pose.py`
  documents, so a prior gap still integrates rather than freezing. Calibration rotates the VO
  lateral the same way the correction's ingest does.
  """
  if not (sm.updated['cameraOdometry'] and sm.valid['cameraOdometry']):
    return
  odometry = sm['cameraOdometry']
  if len(odometry.trans) < 3 or len(odometry.rot) < 3:
    return
  mono_time = odometry.timestampEof * 1e-9 - MOONPILOT_SLAM_POSE_DELAY
  trans = np.asarray(odometry.trans, dtype=float)
  device_from_calib = _device_from_calib(sm)
  if device_from_calib is not None:
    trans = device_from_calib @ trans
  v_forward = prior.at(mono_time)
  if v_forward is None:
    v_forward = float(trans[0])
  tracker.push_odom(mono_time, v_forward=v_forward, yaw_rate=float(odometry.rot[2]), v_lateral=float(trans[1]))


def _gps_push(tracker: PoseTracker, sm, gps_service: str) -> None:
  """Gate and apply one GPS fix from whichever location service this build has."""
  if not sm.updated[gps_service]:
    return
  g = sm[gps_service]
  tracker.push_gps(
    sm.logMonoTime[gps_service] * 1e-9,
    float(g.latitude),
    float(g.longitude),
    horizontal_accuracy=float(g.horizontalAccuracy),
    has_fix=bool(g.hasFix),
    bearing_deg=float(g.bearingDeg),
    speed=float(g.speed),
  )


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
  # Fed whether or not MoonpilotSlam is on, same as the correction window: toggling is immediate.
  pose = PoseTracker()

  # Owned here and created once: per-frame filters would silently drop the smoothing.
  in_path_filters = [FirstOrderFilter(1.0, MOONPILOT_INPATH_RC, DT_MDL) for _ in range(N_LEADS)]
  # The confidence gate's own filter, radard's shape: 0.0 start so the first frame is a rise.
  prob_filters = [FirstOrderFilter(0.0, MOONPILOT_LEAD_PROB_RC, DT_MDL) for _ in range(N_LEADS)]

  while True:
    sm.update()

    # Fed every frame, not only on the polled one: carState runs at 100 Hz, and `PriorChannel.at`
    # can only interpolate over what it was given.
    if sm.updated['carState']:
      prior.push(sm.logMonoTime['carState'] * 1e-9, float(sm['carState'].vEgo))

    _gps_push(pose, sm, gps_service)
    _pose_push(pose, sm, prior)

    if not sm.updated['modelV2']:
      continue

    # The window is fed whether or not the correction is wanted, so toggling it on is immediate
    # rather than waiting five seconds for a window to fill. `enabled`, not the raw param: a feature
    # whose dependencies are missing is off, and this is where that has to be asked.
    node = _slam_node(sm, prior)
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

    leads = msg.moonpilotState.init('leads', N_LEADS)
    for i, slot in enumerate(leads):
      model_lead = model.leadsV3[i] if i < len(model.leadsV3) else None
      fused_lead = None
      if radar is not None:
        fused_lead = (radar.leadOne, radar.leadTwo)[i] if i < 2 else None

      for field, value in normalize_lead(i, model_lead, fused_lead, ego_path_x, ego_path_y, in_path_filters[i], prob_filters[i]).items():
        setattr(slot, field, value)

    fill_ego_correction(msg, corr)
    # Same toggle as the correction: one SLAM row, both observation fields. Off leaves egoPose at
    # its default (valid false), which is the pass-through case.
    if slam_on:
      fill_ego_pose(msg, pose)
    pm.send('moonpilotState', msg)


if __name__ == "__main__":
  main()
