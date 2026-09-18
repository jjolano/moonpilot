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
  description="Relax the lead following constraint for a lead predicted to leave your path. Takes effect after a restart.",
  offroad_only=True,
)

TORQUE_LATERAL = Feature(
  key="MoonpilotTorqueLateral",
  title="moonpilot steering",
  description="moonpilot's own torque steering controller instead of openpilot's. Torque-steered cars only; takes effect after a restart.",
  offroad_only=True,
)

LONGITUDINAL = Feature(
  key="MoonpilotLongitudinal",
  title="moonpilot longitudinal",
  description="moonpilot's own longitudinal planner and acceleration controller, not openpilot's MPC. openpilot-longitudinal cars only; restart to apply.",
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

LATERAL_ENGAGE = Feature(
  key="MoonpilotLateralEngage",
  title="lateral engagement",
  description="Steer from the cruise main switch alone; ACC owns speed once set. Brake and gas no longer disengage; LKAS toggles. Toyota/stock ACC; restart.",
  offroad_only=True,
)

# Behaviors the driver can swap back to upstream. The settings panel is built from this
# table, so a feature is one row here, one row in params_keys.h, and its own code.
FEATURES: tuple[Feature, ...] = (LEAD_LATERAL, TORQUE_LATERAL, LONGITUDINAL, TAILSCALE, LATERAL_ENGAGE)


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
