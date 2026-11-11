"""The model protocol table and its structural admission rules."""

import datetime
from dataclasses import dataclass
from typing import Any

DRIVING = "driving"
MONITORING = "monitoring"
KINDS = (DRIVING, MONITORING)
BUNDLED = "bundled"
CUSTOM = "custom"
NOT_INSTALLED = "not installed"
# --- protocols: what may be selected -----------------------------------------
# A protocol is a statement about an *interchange controller*, not about a catalog entry: it names
# the roles a selection must have, the inputs the fork's runtime feeds each role, the outputs a
# consumer in this tree reads, and how the action head is contracted. Admission
# (moonpilot/modelcatalog.py) checks a recipe against exactly this table, and nothing else may
# decide what the device executes.
#
# Two things about the shape:
#
# - **The slice requirement is the union over the members, not a per-role pin.** Every consumer
#   here (`fill_model_msg`, `fill_pose_msg`, `parse_model_output`) reads names off one dict, so what
#   has to hold is that *some* member produces each required name. That is the property the real
#   split generations satisfy: a family-C set puts `lane_lines`/`lead` in the off-policy member and
#   `plan`/`desire_state` in whichever policy member carries them, and pinning the slices per role
#   refuses sets that run fine -- 24 of the catalog's family-C recipes are admissible under the
#   union and would be refused by any per-role pin that also admits them.
# - **`action_contract` is per protocol, and it is the one place a wrong guess is a steering
#   request.** `plan` decodes both lateral and longitudinal from the trajectory through upstream's
#   `get_action_from_model`; `lateral-accel` is that same current function's `action` branch
#   (`action[0,0] / max(1, v_ego)**2`, i.e. the head's second column is an acceleration);
#   `curvature-100` is the older convention (`action[0,0] / 100`, the head's first column is a
#   curvature times 100) that `openpilot@9757bf10` used. Upstream switched the meaning of that head
#   in `249cafe897deb433944c84fa6c3c6ce568f727e1` ("Prereqs deep models", 2026-06-01), and no shape
#   in the catalog distinguishes the two eras, so admission refuses an action-declaring member on
#   the wrong side of that date rather than scale it by the other convention's constant.
LATERAL_ACCEL_SINCE = datetime.date(2026, 6, 1)

# `ModelConstants`-era `STOCK_SLICES` minus `pad`: every name `fill_model_msg`, `fill_pose_msg` and
# the loop's own `prev_feat` read from the model output dict.
DRIVING_SLICES = frozenset(
  {
    "meta",
    "desire_pred",
    "pose",
    "wide_from_device_euler",
    "road_transform",
    "lane_lines",
    "lane_lines_prob",
    "road_edges",
    "lead",
    "lead_prob",
    "hidden_state",
    "plan",
    "desire_state",
  }
)

# What `dmonitoringmodeld`'s `parse_model_output`/`fill_driver_data` read. `sleep_prob_*` and
# `features` are deliberately absent: they are newer heads, nothing in this tree reads
# `driverStateV2.sleepProb` (`openpilot/selfdrive/monitoring/policy.py`), and a legacy 551-slice
# model that lacks them is safe rather than degraded.
DM_SLICES = frozenset(
  {
    "face_descs_lhd",
    "face_descs_rhd",
    "face_prob_lhd",
    "face_prob_rhd",
    "left_eye_prob_lhd",
    "left_eye_prob_rhd",
    "right_eye_prob_lhd",
    "right_eye_prob_rhd",
    "left_blink_prob_lhd",
    "left_blink_prob_rhd",
    "right_blink_prob_lhd",
    "right_blink_prob_rhd",
    "sunglasses_prob_lhd",
    "sunglasses_prob_rhd",
    "using_phone_prob_lhd",
    "using_phone_prob_rhd",
    "wheel_on_right",
  }
)

CONFIGURATION_KEYS = ("frame_skip", "LAT_SMOOTH_SECONDS", "LONG_SMOOTH_SECONDS")


ACTION_PLAN = "plan"
ACTION_LATERAL_ACCEL = "lateral-accel"
ACTION_CURVATURE_100 = "curvature-100"
QCOM_TARGET = "QCOM"


@dataclass(frozen=True)
class Protocol:


  id: str
  kind: str
  family: str
  roles: tuple[str, ...]
  features_role: str
  policy_role: str
  role_inputs: dict[str, frozenset[str]]
  optional_inputs: dict[str, frozenset[str]]
  slices: frozenset[str]
  action_contract: str
  label: str
  required_configuration: tuple[str, ...] = CONFIGURATION_KEYS

  def field(self, role: str, name: str) -> str:
    return name if self.family == "supercombo" else f"{name}_{role}"



SUPERCOMBO = Protocol(
  id="comma.supercombo.v1",
  kind=DRIVING,
  family="supercombo",
  roles=("supercombo",),
  features_role="supercombo",
  policy_role="supercombo",
  role_inputs={
    "supercombo": frozenset({"img", "big_img", "features_buffer", "desire_pulse", "traffic_convention", "action_t"}),
  },
  optional_inputs={"supercombo": frozenset()},
  slices=DRIVING_SLICES,
  action_contract=ACTION_LATERAL_ACCEL,
  label="supercombo",
)

SPLIT_VISION_POLICY = Protocol(
  id="comma.split-vision-policy.v1",
  kind=DRIVING,
  family="split",
  roles=("vision", "on_policy"),
  features_role="vision",
  policy_role="on_policy",
  role_inputs={
    "vision": frozenset({"img", "big_img"}),
    "on_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
  },
  optional_inputs={"vision": frozenset(), "on_policy": frozenset({"action_t"})},
  slices=DRIVING_SLICES,
  action_contract=ACTION_PLAN,
  label="vision + policy",
)

SPLIT_VISION_OFF_ON = Protocol(
  id="comma.split-vision-off-on.v1",
  kind=DRIVING,
  family="split",
  roles=("vision", "off_policy", "on_policy"),
  features_role="vision",
  policy_role="on_policy",
  role_inputs={
    "vision": frozenset({"img", "big_img"}),
    "off_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
    "on_policy": frozenset({"desire_pulse", "features_buffer", "traffic_convention"}),
  },
  optional_inputs={
    "vision": frozenset(),
    "off_policy": frozenset({"action_t"}),
    "on_policy": frozenset({"action_t"}),
  },
  slices=DRIVING_SLICES,
  action_contract=ACTION_CURVATURE_100,
  label="vision + off + on",
)

DMONITORING = Protocol(
  id="comma.dmonitoring.v1",
  kind=MONITORING,
  family="dmonitoring",
  roles=("dmonitoring",),
  features_role="dmonitoring",
  policy_role="dmonitoring",
  role_inputs={"dmonitoring": frozenset({"input_img", "calib"})},
  optional_inputs={"dmonitoring": frozenset()},
  slices=DM_SLICES,
  action_contract=ACTION_PLAN,
  label="driver monitoring",
  required_configuration=(),
)
PROTOCOLS: tuple[Protocol, ...] = (SUPERCOMBO, SPLIT_VISION_POLICY, SPLIT_VISION_OFF_ON, DMONITORING)



def protocol(protocol_id: str) -> Protocol | None:
  return next((p for p in PROTOCOLS if p.id == protocol_id), None)


def protocol_for_roles(roles: Any) -> Protocol | None:
  """The protocol whose role set is exactly `roles`. A role set no protocol claims is refused
  rather than mapped onto the nearest one."""
  wanted = frozenset(roles)
  return next((p for p in PROTOCOLS if frozenset(p.roles) == wanted), None)





# Nothing here needs the catalog SDK: every input is a plain dict of what a document recorded --
# `input_shapes`, `output_slices`, `targets`, `format` -- which is exactly what the catalog's
# metadata reader and the stored `recipe.json` both carry.
#
# A member that declares an input the protocol does not list is refused rather than half-fed. That
# is the rule that refuses the pre-`action_t` monoliths (`lateral_control_params`,
# `prev_desired_curv`, `nav_features`), and it is also what would refuse a member declaring
# `prev_action`: no recipe in the 2026-09-19 catalog declares it, so nothing here carries the
# queue for it, and a model that needs one would be a model this runtime cannot run yet.
UNSUPPORTED_INPUT = "needs inputs this runtime does not supply"
UNKNOWN_ROLE = "a role this protocol does not have"
MISSING_OUTPUTS = "missing outputs this runtime publishes"
HIDDEN_STATE_MISMATCH = "the vision feature width does not match the policy's features_buffer"
NOT_ONNX = "not an ONNX model"
NOT_QCOM = "chestnut/AMD models are not selectable"
NO_METADATA = "the catalog records no interface for this member"
ACTION_ERA = "predates the fork's action contract"


def member_reason(role: str, proto: Protocol, member: dict) -> str | None:
  """Why one member cannot be run as this protocol's `role`, or `None`.

  `member` carries `input_shapes` (name -> shape), `slices` (name -> [start, stop, step]), `format`,
  `targets` and `published_at` (ISO 8601, the catalog's own observation of when the artifact was
  published; absent for a stored package, where the check has already been made at install time).
  """
  required = proto.role_inputs.get(role)
  if required is None:
    return UNKNOWN_ROLE
  shapes = member.get("input_shapes")
  if not isinstance(shapes, dict) or not shapes:
    return NO_METADATA
  if member.get("format") != "onnx":
    return NOT_ONNX
  targets = member.get("targets")
  if isinstance(targets, list) and QCOM_TARGET not in targets:
    return NOT_QCOM
  unsupported = set(shapes) - required - proto.optional_inputs.get(role, frozenset())
  if unsupported:
    return UNSUPPORTED_INPUT
  if not required <= set(shapes):
    return UNSUPPORTED_INPUT
  return _action_era_reason(proto, member)


def _action_era_reason(proto: Protocol, member: dict) -> str | None:
  """See the protocol table above: the action head changed meaning on 2026-06-01."""
  if "action" not in (member.get("slices") or {}):
    return None
  published = member.get("published_at")
  if not published:
    return None
  try:
    published_date = datetime.datetime.fromisoformat(published).date()
  except (TypeError, ValueError):
    return None

  old_head = published_date < LATERAL_ACCEL_SINCE
  if proto.action_contract == ACTION_CURVATURE_100:
    return ACTION_ERA if not old_head else None
  return ACTION_ERA if old_head else None


def set_reason(proto: Protocol, members: dict[str, dict]) -> str | None:
  """Why a whole member set cannot be run, or `None`."""
  if frozenset(members) != frozenset(proto.roles):
    return NOT_INSTALLED
  for role, member in members.items():
    if (reason := member_reason(role, proto, member)) is not None:
      return f"{role}: {reason}"
  published = set()
  for member in members.values():
    published |= set(member.get("slices") or {})
  if missing := proto.slices - published:
    return f"{MISSING_OUTPUTS}: {', '.join(sorted(missing))}"
  vision = members[proto.features_role]
  hidden = (vision.get("slices") or {}).get("hidden_state")
  width = _slice_width(hidden)
  policy = members[proto.policy_role]
  buffer_shape = (policy.get("input_shapes") or {}).get("features_buffer")
  if width is not None and isinstance(buffer_shape, list) and buffer_shape:
    if width != buffer_shape[-1]:
      return HIDDEN_STATE_MISMATCH
  return None


def _slice_width(bounds: Any) -> int | None:
  try:
    start, stop = bounds[0], bounds[1]
  except (TypeError, IndexError):
    return None
  if not isinstance(start, int) or not isinstance(stop, int):
    return None
  return stop - start
