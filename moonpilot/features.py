from dataclasses import dataclass

from openpilot.common.params import Params
from openpilot.common.version import get_version


@dataclass(frozen=True)
class Feature:
  key: str  # param row in moonpilot/params_keys.h, which also carries its default
  title: str
  description: str
  offroad_only: bool = False  # changes driving behavior, so only flip it while parked


LEAD_LATERAL = Feature(
  key="MoonpilotLeadLateral",
  title="lead lateral prediction",
  description="Relax the lead following constraint for a lead predicted to leave your path. Takes effect after a restart.",
  offroad_only=True,
)

# Behaviors the driver can swap back to upstream. The settings panel is built from this
# table, so a feature is one row here, one row in params_keys.h, and its own code.
FEATURES: tuple[Feature, ...] = (LEAD_LATERAL,)


def enabled(feature: Feature, params: Params) -> bool:
  # Not get_bool: that ignores the default declared in params_keys.h and reports off for
  # an unset param. return_default=True is the value that holds inside and outside the
  # manager, which seeds unset params from their default at boot.
  return bool(params.get(feature.key, return_default=True))


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
