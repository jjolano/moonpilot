from dataclasses import dataclass

from openpilot.common.params import Params


@dataclass(frozen=True)
class Feature:
  key: str  # param row in moonpilot/params_keys.h, which also carries its default
  title: str
  description: str
  offroad_only: bool = False  # changes driving behavior, so only flip it while parked


BRANDING = Feature(
  key="MoonpilotBranding",
  title="moonpilot branding",
  description="Show moonpilot instead of openpilot as the product name. Takes effect after a reboot.",
)

LEAD_LATERAL = Feature(
  key="MoonpilotLeadLateral",
  title="lead lateral prediction",
  description="Relax the lead following constraint for a lead predicted to leave your path. Takes effect after a restart.",
  offroad_only=True,
)

# Behaviors the driver can swap back to upstream. The settings panel is built from this
# table, so a feature is one row here, one row in params_keys.h, and its own code.
FEATURES: tuple[Feature, ...] = (BRANDING, LEAD_LATERAL)


def enabled(feature: Feature, params: Params) -> bool:
  # Not get_bool: that ignores the default declared in params_keys.h and reports off for
  # an unset param. return_default=True is the value that holds inside and outside the
  # manager, which seeds unset params from their default at boot.
  return bool(params.get(feature.key, return_default=True))


def brand(params: Params) -> str:
  # Delegates rather than replaces, so upstream's own name stays reachable.
  return "moonpilot" if enabled(BRANDING, params) else "openpilot"
