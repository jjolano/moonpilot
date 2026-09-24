from dataclasses import dataclass

from moonpilot import deps
from openpilot.common.params import Params
from openpilot.common.version import get_version


@dataclass(frozen=True)
class Feature:
  key: str  # param row in moonpilot/params_keys.h, which also carries its default
  title: str
  description: str
  offroad_only: bool = False  # changes driving behavior, so only flip it while parked
  requires: tuple[str, ...] = ()  # modules that must be importable for this feature to run


LEAD_LATERAL = Feature(
  key="MoonpilotLeadLateral",
  title="lead lateral prediction",
  description="Relax the lead following constraint for a lead predicted to leave your path. Applies immediately.",
  offroad_only=True,
)

TORQUE_LATERAL = Feature(
  key="MoonpilotTorqueLateral",
  title="moonpilot steering",
  description="moonpilot's own torque steering controller instead of openpilot's. Torque-steered cars only; takes effect after a restart.",
  offroad_only=True,
)

PATH_PREVIEW = Feature(
  key="MoonpilotPathPreview",
  title="response-aligned steering",
  description="Align the model path with when steering is expected to respond, using live model age and steering delay. Restart to apply.",
  offroad_only=True,
)

TURN_DESIRE = Feature(
  key="MoonpilotTurnDesire",
  title="turn desire",
  description=(
    "Below 32 km/h, a turn signal feeds a turn desire into the model so it steers into the corner. "
    + "The model's own turn prediction picks the direction. Restart to apply."
  ),
  offroad_only=True,
)

LONGITUDINAL = Feature(
  key="MoonpilotLongitudinal",
  title="moonpilot longitudinal",
  description="moonpilot's own longitudinal planner and acceleration controller, not openpilot's MPC. openpilot-longitudinal cars only; restart to apply.",
  offroad_only=True,
)

MODEL_BRAKING = Feature(
  key="MoonpilotModelBraking",
  title="model braking",
  description="Outside experimental mode, let the model's own braking ask through to the command. Applies immediately.",
  offroad_only=True,
)

CURVE_SPEED = Feature(
  key="MoonpilotCurveSpeed",
  title="curve speed control",
  description="Slow for a curve the model already sees, and hold the lateral acceleration it is pulling. Braking is bounded, and only added. Immediate.",
  offroad_only=True,
)

SQUEEZE = Feature(
  key="MoonpilotSqueeze",
  title="narrow corridor braking",
  description="Slow when free space ahead pinches under about two car widths. Braking is bounded, and only added. Immediate.",
  offroad_only=True,
)

COAST_GRADE = Feature(
  key="MoonpilotCoastGrade",
  title="coast on grade",
  description=(
    "Within 1.5 m/s of the set speed the car coasts with the hill instead of holding it: allowed to "
    + "run up on a descent and to sag on a climb. Following, curve and model braking are unchanged. "
    + "Applies immediately."
  ),
  offroad_only=True,
)

# No requires: `requires` gates on importable Python modules, and tailscale here is a binary.
# There is nothing to gate either way — the supervisor installs what is missing and the settings
# row says so while it does.
TAILSCALE = Feature(
  key="MoonpilotTailscale",
  title="tailscale",
  description="Join this device to your tailnet for remote access. Downloads tailscale (~35 MB) the first time, then shows a sign-in link here.",
)

SLAM = Feature(
  key="MoonpilotSlam",
  title="rolling-window ego correction",
  description="Smooths recent ego motion over a 5 s rotating window and corrects the speed the fork planner plans from. Off = the raw wheel speed.",
  offroad_only=True,
)

CATALOG_SIGNATURES = Feature(
  key="MoonpilotCatalogSignatures",
  title="catalog signatures",
  description=(
    "Refuse a model catalog that fails its detached signature check against the key pinned in this build. "
    + "Needs the cryptography package, installed automatically once the device is online. Off by default; "
    + "an unsigned catalog is accepted until the fork pins a key."
  ),
  requires=("cryptography",),
)

LATERAL_ENGAGE = Feature(
  key="MoonpilotLateralEngage",
  title="lateral engagement",
  description="Steer from the main switch; brake, gas and cancel keep steering. Where panda cannot read the switch, the host's claim is trusted. Restart.",
  offroad_only=True,
)


@dataclass(frozen=True)
class Group:
  """A page of the panel: what its rows have in common, in the driver's terms.

  The panel has one screen per group, because the flat list had grown past what a driver can read at
  a glance -- and the grouping is by *what the row changes about the car*, not by which module owns
  it, so the question a driver arrives with ("why does it steer like that") lands on one page."""

  title: str
  description: str
  features: tuple[Feature, ...]


# Behaviors the driver can swap back to upstream, grouped into the pages each panel renders. The
# panels are built from this table, so a feature is one row here, one row in params_keys.h, and its
# own code -- and a new page is one `Group` and no panel edit.
STEERING = Group(
  title="steering",
  description="How openpilot moves the wheel: whose controller, when a curve is entered, and whether openpilot can steer at all.",
  features=(LATERAL_ENGAGE, TORQUE_LATERAL, PATH_PREVIEW, TURN_DESIRE),
)

SPEED = Group(
  title="speed & distance",
  description="What the car does with the pedals: whose planner sets the speed, and how it reads the road ahead.",
  # The lead's lateral prediction sits here rather than with the steering because the fork reaches
  # it through the planner's time gap, not through a steering request.
  features=(LONGITUDINAL, MODEL_BRAKING, CURVE_SPEED, SQUEEZE, COAST_GRADE, LEAD_LATERAL, SLAM),
)

DEVICE = Group(
  title="device",
  description="What the device does for itself, off the road.",
  features=(TAILSCALE, CATALOG_SIGNATURES),
)

GROUPS: tuple[Group, ...] = (STEERING, SPEED, DEVICE)

# Depth-first over the groups: every feature, once, in the order the pages show them.
FEATURES: tuple[Feature, ...] = tuple(feature for group in GROUPS for feature in group.features)


def missing_modules(feature: Feature) -> tuple[str, ...]:
  return tuple(module for module in feature.requires if not deps.available(module))


def available(feature: Feature) -> bool:
  return not missing_modules(feature)


def wanted(feature: Feature, params: Params) -> bool:
  # Not get_bool: that ignores the default declared in params_keys.h and reports off for
  # an unset param. return_default=True is the value that holds inside and outside the
  # manager, which seeds unset params from their default at boot.
  return bool(params.get(feature.key, return_default=True))


def enabled(feature: Feature, params: Params) -> bool:
  # The seam-facing "is this behavior on" predicate: a feature the driver asked for but
  # whose dependencies are not installed yet is off, so seams never import what is missing.
  return available(feature) and wanted(feature, params)


def brand() -> str:
  # Fork identity, like version(). Always on, so no param and no toggle to carry: nobody
  # picks between "moonpilot" and "openpilot" as a name. The seams name upstream's own
  # value as a fallback, so this is the only thing standing between them and stock.
  return "moonpilot"


def version() -> str:
  # Fork identity, like brand(): COMMA_VERSION verbatim, "<upstream>-moonpilot.<fork revision>".
  # get_version() reads version.h directly, so this is right outside the manager too, where the
  # Version param is unset. See AGENTS.md, Versioning.
  return get_version()


def is_fork_build() -> bool:
  # The one identity helper that gates behavior: comma tests comma's branches, so upstream's
  # "WARNING: This branch is not tested" startup banner would sit on the road at the top of every
  # drive in a build of this tree. Keyed on the COMMA_VERSION marker rather than on the git remote,
  # because the marker is what every fork build carries however it was installed. See AGENTS.md.
  return "-moonpilot." in version()
