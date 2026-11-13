from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from moonpilot import deps
from openpilot.common.params import Params
from openpilot.common.realtime import Priority
from openpilot.common.version import get_version


@dataclass(frozen=True)
class Feature:
  key: str  # param row in moonpilot/params_keys.h, which also carries its default
  title: str
  description: str
  requires: tuple[str, ...] = ()  # modules that must be importable for this feature to run
  # A choice row instead of a toggle: (label, value) pairs, the param an INT index into them, 0 = off.
  choices: tuple[tuple[str, Any], ...] = ()
  # Why a car cannot run this, value-line sized, or None when it can.
  car: Callable[[Any], str | None] | None = None

  @property
  def labels(self) -> tuple[str, ...]:
    return tuple(label for label, _ in self.choices)

  @property
  def values(self) -> tuple[Any, ...]:
    return tuple(value for _, value in self.choices)

  def at(self, index: Any) -> Any:
    """The value a choice index selects; anything out of range is the off choice."""
    return self.choices[index][1] if isinstance(index, int) and 0 <= index < len(self.choices) else self.choices[0][1]


def _torque_steered(CP) -> str | None:
  # controlsd's own dispatch: steerControlType is tested before the tuning, so an angle- or curvature-steered
  # car never reaches the torque branch the fork controller sits in, however its tuning reads.
  if CP.steerControlType in ("angle", "curvature") or CP.lateralTuning.which() != "torque":
    return "torque-steered cars only"
  return None


def _openpilot_longitudinal(CP) -> str | None:
  return None if CP.openpilotLongitudinalControl else "openpilot-longitudinal cars only"


def _toyota_openpilot_longitudinal(CP) -> str | None:
  return None if CP.brand == "toyota" and CP.openpilotLongitudinalControl else "Toyota openpilot-longitudinal only"


def _lateral_engage(CP) -> str | None:
  from moonpilot.engage import unsupported_reason  # the safety gate lives with the safety code, which imports this module
  return unsupported_reason(CP)


LEAD_LATERAL = Feature(
  key="MoonpilotLeadLateral",
  title="lead lateral prediction",
  description="Relax the lead following constraint for a lead predicted to leave your path. Applies immediately.",
)

TORQUE_LATERAL = Feature(
  key="MoonpilotTorqueLateral",
  title="moonpilot steering",
  description="moonpilot's own torque steering controller instead of openpilot's. Torque-steered cars only; takes effect after a restart.",
  car=_torque_steered,
)

PATH_LOOKAHEAD = Feature(
  key="MoonpilotPathLookahead",
  title="turn in early",
  description=(
    "Above about 40 km/h, starts turns earlier by looking further along the planned path; "
    + "never steers less than without it, so it does not run wide. Stronger starts earlier and cuts more toward the inside. "
    + "Restart to apply."
  ),
  # (window s, gain, cap m/s^2). See `moonpilot/curvature.py` for behavior and replay measurements:
  # replayed on the recorded turns of routes 3f8-410 (python -m moonpilot.lab lookahead) the request started
  # 0.00/0.10/0.12 s (light), 0.00/0.20/0.20 s (medium) and 0.00/0.35/0.24 s (strong) earlier at 5-10/10-17/>17 m/s
  # with every turn's apex at 100% of today's (p10 included) and no outward drift at the median; the 5-10 column is
  # after the low-speed fade. Unrestricted, the same averages cut the apex to 82-96% and let turns out early --
  # running wide. The cost is a turn-in that begins inside the lane: 0.4-0.6 / 0.8-1.2 / 1.1-1.5 m median open-loop,
  # before the model's own re-planning. None makes highway builds lighter: the controller adds that abruptness.
  choices=(("off", None), ("light", (1.0, 0.3, 0.8)), ("medium", (1.0, 0.6, 1.5)), ("strong", (1.5, 0.6, 1.5))),
)

PATH_SMOOTH = Feature(
  key="MoonpilotPathSmooth",
  title="smooth planner path",
  description=(
    "Filter the model's curvature request before the controller for calmer torque steps: 0.05 s takes about half the "
    + "steps out, 0.10 s most of the rest, 0.15 s little more. Above about 18 km/h the planned path is read that much "
    + "further ahead, which removes half to three quarters of the filter's steering lag. Restart to apply."
  ),
  choices=(("off", 0.0), ("0.05 s", 0.05), ("0.10 s", 0.10), ("0.15 s", 0.15)),  # first-order time constant, s
)

PATH_RECENTER = Feature(
  key="MoonpilotPathRecenter",
  title="calm re-centering",
  description=(
    "Above about 40 km/h, the planned path may move the car sideways within its lane only gently: steering beyond the "
    + "lane's own bend is held to a small sideways acceleration, so a plan that suddenly shifts over, like re-centering "
    + "on a curb that reappears past an intersection, drifts instead of lurching. Never adds steering, and does "
    + "nothing during lane changes or where the lane lines are not seen. Restart to apply."
  ),
)

STEER_HYSTERESIS = Feature(
  key="MoonpilotSteerHysteresis",
  title="steady steering torque",
  description=(
    "moonpilot steering only. Holds the steering torque until the controller's own torque moves past a small band "
    + "(percent of maximum), so tiny back-and-forth corrections never reach the wheel. Wider is calmer and less precise. "
    + "Restart to apply."
  ),
  # Half-widths in normalized steer torque. On a RAV4 (STEER_MAX 1500) these are ~4.5 / 10.5 / 19.5 CAN units;
  # `python -m moonpilot.lab torque` cuts torque reversals on 40d/40e from 14.2/15.9 per s to 4.3/5.6 at 0.7%.
  choices=(("off", 0.0), ("0.3%", 0.003), ("0.7%", 0.007), ("1.3%", 0.013)),
)

STEER_DAMPING = Feature(
  key="MoonpilotSteerDamping",
  title="curve damping",
  description=(
    "moonpilot steering only. Resists the wheel weaving back and forth in curves between about 10 and 70 km/h, "
    + "without slowing turns the path asks for. Stronger is steadier, but passes more sensor noise to the wheel. "
    + "Restart to apply."
  ),
  # Derivative time, s: P + D = KP * (error + T * error_rate). In hands-off curves at 3-17 m/s (routes 3f8-410,
  # matched by speed and turn size against the same driver's manual curves) 22-24% of the wheel's motion was above
  # 1 Hz against the driver's 4-10%, and P carried it: P's 1-5 Hz content equalled its turn-scale correction and
  # moved with the wheel at r = 0.9, 81% of it at 1-2 Hz. At 1.2 Hz these are 21 / 37 / 48 degrees of lead;
  # `python -m moonpilot.lab damping` puts 42 / 83 / 125% of P's 1-2 Hz action into D.
  choices=(("off", 0.0), ("light", 0.05), ("medium", 0.10), ("strong", 0.15)),
)

STEER_FRICTION_RAMP = Feature(
  key="MoonpilotSteerFrictionRamp",
  title="smooth friction breakaway",
  description=(
    "moonpilot steering only. Lets the torque that breaks the wheel loose build over a couple of tenths of a second "
    + "instead of in one step, so the wheel does not snap past the path when it lets go. Releasing is always immediate. "
    + "Restart to apply."
  ),
)

TURN_DESIRE = Feature(
  key="MoonpilotTurnDesire",
  title="turn desire",
  description=(
    "Below 32 km/h, a turn signal feeds a turn desire into the model so it steers into the corner. "
    + "The model's own turn prediction picks the direction. Restart to apply."
  ),
)

LONGITUDINAL = Feature(
  key="MoonpilotLongitudinal",
  title="moonpilot longitudinal",
  description="moonpilot's own longitudinal planner and acceleration controller, not openpilot's MPC. openpilot-longitudinal cars only; restart to apply.",
  car=_openpilot_longitudinal,
)

EXPERIMENTAL_MODE = Feature(
  key="MoonpilotExperimentalMode",
  title="experimental mode",
  description=(
    "moonpilot longitudinal passes the model's own acceleration straight through, as openpilot's experimental mode "
    + "did; no-flicker braking, the model-braking deadband and coast on grade stand aside. Applies immediately."
  ),
)

MODEL_BRAKING = Feature(
  key="MoonpilotModelBraking",
  title="model braking",
  description="Outside experimental mode, let the model's own braking ask through to the command. Applies immediately.",
)

CURVE_SPEED = Feature(
  key="MoonpilotCurveSpeed",
  title="curve speed control",
  description="Slow for a curve the model already sees, and hold the lateral acceleration it is pulling. Braking is bounded, and only added. Immediate.",
)

SQUEEZE = Feature(
  key="MoonpilotSqueeze",
  title="narrow corridor braking",
  description="Slow when free space ahead pinches under about two car widths. Braking is bounded, and only added. Immediate.",
)

PATH_OUTSIDE = Feature(
  key="MoonpilotPathOutside",
  title="path outside road warning",
  description="Warn when the planned path leaves the model's free corridor. Immediate.",
)

COAST_GRADE = Feature(
  key="MoonpilotCoastGrade",
  title="coast on grade",
  description=(
    "Within 1.5 m/s of the set speed the car coasts with the hill instead of holding it: allowed to "
    + "run up on a descent and to sag on a climb. Following, curve and model braking are unchanged. "
    + "Applies immediately."
  ),
)

BRAKE_BAND = Feature(
  key="MoonpilotBrakeBand",
  title="no brake-light flicker",
  description=(
    "Above about 18 km/h a slight slowdown is done by coasting, and real braking is one firm press of at least "
    + "a second, so the brake lights do not blink on and off. Firmer braking and anything below 18 km/h are "
    + "unchanged. Applies immediately."
  ),
)

ENGINE_COAST = Feature(
  key="MoonpilotEngineCoast",
  title="engine-braking coast (test)",
  description=(
    "Above about 18 km/h a slowdown down to the engine's own coast is done without the brakes, so following a "
    + "lead lights the brake lights far less. Untested on the road: watch that the car still slows as asked. "
    + "Toyota openpilot-longitudinal only. Takes effect after a restart."
  ),
  car=_toyota_openpilot_longitudinal,
)

HANDBACK_EASE = Feature(
  key="MoonpilotHandbackEase",
  title="ease off after the gas",
  description=(
    "When you lift off the gas below the set speed, openpilot eases off and builds back up to the set speed over a couple "
    + "of seconds, instead of carrying on at the rate you were accelerating. Braking is unchanged. Applies immediately."
  ),
)

CRAWL_RELEASE = Feature(
  key="MoonpilotCrawlRelease",
  title="ease off before the stop (test)",
  description=(
    "In the last 2 km/h of a gentle stop the brake request eases toward zero, the way a driver lets off just before "
    + "the car stops, to take out the final lurch. Firmer stops are unchanged. Untested on the road: watch the stopping "
    + "distance. Needs moonpilot longitudinal; openpilot-longitudinal cars only. Takes effect after a restart."
  ),
  car=_openpilot_longitudinal,
)

TAILSCALE = Feature(
  key="MoonpilotTailscale",
  title="tailscale",
  description="Join this device to your tailnet for remote access. Downloads tailscale (~35 MB) the first time, then shows a sign-in link here.",
)

SLAM = Feature(
  key="MoonpilotSlam",
  title="rolling-window ego correction",
  description="Smooths recent ego motion over a 5 s rotating window and corrects the speed the fork planner plans from. Off = the raw wheel speed.",
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
  car=_lateral_engage,
)


@dataclass(frozen=True)
class Group:

  title: str
  description: str
  features: tuple[Feature, ...]
  parked_only: bool  # the rows change driving behavior, so they only flip while offroad


STEERING = Group(
  title="steering",
  description="How openpilot moves the wheel: whose controller, when a curve is entered, and whether openpilot can steer at all.",
  features=(LATERAL_ENGAGE, TORQUE_LATERAL, PATH_LOOKAHEAD, PATH_SMOOTH, PATH_RECENTER, STEER_HYSTERESIS, STEER_DAMPING, STEER_FRICTION_RAMP,
            TURN_DESIRE),
  parked_only=True,
)

SPEED = Group(
  title="speed & distance",
  description="What the car does with the pedals: whose planner sets the speed, and how it reads the road ahead.",
  features=(LONGITUDINAL, EXPERIMENTAL_MODE, MODEL_BRAKING, CURVE_SPEED, SQUEEZE, PATH_OUTSIDE, COAST_GRADE, BRAKE_BAND, ENGINE_COAST, CRAWL_RELEASE,
            HANDBACK_EASE, LEAD_LATERAL, SLAM),
  parked_only=True,
)

DEVICE = Group(
  title="device",
  description="What the device does for itself, off the road.",
  features=(TAILSCALE, CATALOG_SIGNATURES),
  parked_only=False,
)

GROUPS: tuple[Group, ...] = (STEERING, SPEED, DEVICE)
FEATURES: tuple[Feature, ...] = tuple(feature for group in GROUPS for feature in group.features)


def missing_modules(feature: Feature) -> tuple[str, ...]:
  return tuple(module for module in feature.requires if not deps.available(module))


def car_reason(feature: Feature, CP) -> str | None:
  """Why this car cannot run the feature, or None. A CarParams not loaded yet is no verdict on the car."""
  return None if CP is None or feature.car is None else feature.car(CP)


def wanted(feature: Feature, params: Params) -> bool:
  return bool(params.get(feature.key, return_default=True))


def enabled(feature: Feature, params: Params) -> bool:
  return not missing_modules(feature) and wanted(feature, params)


def in_force(feature: Feature, params: Params, CP) -> bool:
  return enabled(feature, params) and car_reason(feature, CP) is None


def choice(feature: Feature, params: Params) -> int:
  index = params.get(feature.key, return_default=True)
  return index if isinstance(index, int) and 0 <= index < len(feature.choices) else 0


def value(feature: Feature, params: Params) -> Any:
  return feature.at(choice(feature, params) if not missing_modules(feature) else 0)


def ui_priority(upstream: int) -> int:
  """UI shares CPU 5 with radar and planning; rendering must yield to their realtime work."""
  return min(upstream, Priority.CTRL_LOW - 1)


def brand() -> str:
  return "moonpilot"


def version() -> str:
  return get_version()


def is_fork_build() -> bool:
  # comma tests comma's branches, so upstream's "WARNING: This branch is not tested" startup banner would sit on the
  # road at the top of every drive in a build of this tree. Keyed on the COMMA_VERSION marker rather than the git
  # remote, because the marker is what every fork build carries however it was installed. See AGENTS.md.
  return "-moonpilot." in version()
