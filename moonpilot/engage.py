"""moonpilot's lateral-only engagement: steer from the cruise main switch, not from ACC set.

Upstream engages everything on one edge — stock ACC engagement — and panda enforces that edge at
its own layer, so a fork cannot widen it from openpilot alone. The fork's side of that is the
`controls_allowed_lateral` permission in forked `opendbc`/`panda` (see AGENTS.md, and
`opendbc/safety/moonpilot/lateral_engage.h` for the rule); this module is the openpilot side of
the same decision: it asks the safety param for the permission, then holds openpilot in its
enabled state while the driver is only half-engaged.

What the module does at runtime:

  - `moonpilot_engage_safety_param`, called once by card before `CarParams` is written, sets the
    Toyota safety-param flag that turns the panda rule on for this car. Only a Toyota with stock
    ACC reaches it (see `_available`), so every other car's `CarParams` is upstream's byte for
    byte.
  - `LateralEngage.update`, called from selfdrived once every event source has contributed, is
    the engage/disengage policy. It suppresses the two events whose whole purpose is the behavior
    being replaced — `pcmDisable` (level-triggered whenever stock ACC is off) and `pedalPressed`
    (the brake/gas disengage) — and asks to engage when the driver turns the cruise main switch on.
    Disengaging is upstream's: the cruise main switch, the LKAS button, a steering override,
    cancel, and every fault still land in `wrongCarMode`, `steerDisengage`, `buttonCancel` or a
    disable event, and any authoritative disable latches the half-engaged state off until the
    driver re-arms.

Scope, and the ceilings that come with it:

  - Toyota/Lexus with stock longitudinal (`CP.pcmCruise`) only, and not `UNSUPPORTED_DSU` — those
    read the cruise main switch out of `DSU_CRUISE` at 5 Hz, below panda's 10 Hz rx-check minimum,
    so the feature is inert there by construction rather than by a runtime check that would fail
    on the road.
  - The decision to be half-engaged is fixed at construction, like the fork's other behavior
    toggles: the module reads its param once, and the panel row says a restart is needed.
  - No fresh permission is invented for the acceleration side. `controlsAllowed` keeps upstream's
    meaning, so `CC.longActive` stays false and the car's own ACC owns speed until the driver sets
    it — which is the whole point of "half".
"""

from opendbc.car.structs import car
from opendbc.car.toyota.values import ToyotaFlags, ToyotaSafetyFlags
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import ET

from moonpilot.features import LATERAL_ENGAGE, enabled

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName

# The two events the half-engaged state is defined against: stock ACC being off (level-triggered,
# so it is present in every frame the driver has not set ACC) and a brake or gas tap. Everything
# else upstream raises keeps working — in particular gas keeps its softer `gasPressedOverride`,
# which is an override rather than a disable, so steering continues through it.
SUPPRESSED_EVENTS = (EventName.pcmDisable, EventName.pedalPressed)


def _available(CP) -> bool:
  """Whether this car is one the half-engaged state can be built for at all."""
  # `not CP.passive` is not redundant with the caller's placement: card sets `passive` and
  # replaces `safetyConfigs` with a single noOutput config *before* the seam runs, so without
  # this a dashcam-mode Toyota would still have the flag ORed into a config that never reads it.
  return CP.brand == 'toyota' and CP.pcmCruise and not CP.passive and not (CP.flags & ToyotaFlags.UNSUPPORTED_DSU)


def moonpilot_engage_safety_param(CP, params: Params | None = None) -> None:
  """The car side of the panda permission: the safety param the forked safety layer reads.

  Called by card after the passive/dashcam handling and before `CarParams` is written, which is
  the only window where the param can still change — panda and selfdrived both read it from there.
  """
  if not (_available(CP) and enabled(LATERAL_ENGAGE, params or Params())):
    return
  # A car with more than one safety config is not a config the fork knows how to flag, and a
  # partial flag would put openpilot and panda out of agreement. Leave it alone.
  if len(CP.safetyConfigs) != 1:
    return
  CP.safetyConfigs[0].safetyParam |= int(ToyotaSafetyFlags.LATERAL_ENGAGE)


def moonpilot_engage(CP, params: Params | None = None) -> 'LateralEngage':
  """The seam's fork side for selfdrived. Always an instance; an inert one when out of scope."""
  return LateralEngage(_available(CP) and enabled(LATERAL_ENGAGE, params or Params()))


class LateralEngage:
  """The half-engaged engage/disengage policy, run once per selfdrived cycle."""

  def __init__(self, on: bool) -> None:
    # Read once, at construction: the toggle takes a restart, which the panel row says.
    self.enabled = on
    # Latched by an authoritative disable, cleared by the driver re-arming (setting ACC, or the
    # LKAS button). Without it a disable would only last the frame it arrived in, because the
    # suppressed pcmDisable/pedalPressed are also what upstream would keep disengaging on.
    self._blocked = False
    # ACC's previous state, so re-arming is its *rising edge*. Clearing on the level instead would
    # undo the LKAS toggle one frame later for a driver who pressed it while cruising with ACC set.
    self._acc_set_prev = False

  def controls_allowed(self, ps) -> bool:
    """Whether this pandaState is one openpilot can be enabled on.

    Upstream's selfdrived treats "enabled but the panda is not allowing controls" as a mismatch
    and disengages, so half-engaged needs the panda's lateral permission to count here — the
    cross-check stays sharp instead of being switched off.
    """
    return bool(ps.controlsAllowed) or (self.enabled and bool(ps.controlsAllowedLateral))

  def update(self, CS, events, enabled: bool) -> None:
    """Adjust the cycle's engagement events in place. Runs after every other event source.

    `enabled` is selfdrived's state-machine output as of the last cycle — the state machine
    consumes the events this method edits, so the engage request is judged against the state the
    driver is about to leave.
    """
    if not self.enabled:
      return

    if not CS.cruiseState.available:
      # The cruise main switch is off, which is upstream's own disengage: the car events raise
      # wrongCarMode, and the panda rule drops its permission on acc_main's falling edge.
      self._blocked = False
      self._acc_set_prev = False
      return

    if CS.cruiseState.enabled and not self._acc_set_prev:
      # Setting ACC re-arms: the driver asked for the full stack, so it may not stay blocked.
      self._blocked = False
    self._acc_set_prev = CS.cruiseState.enabled

    # The wheel's LKAS button is a free explicit on/off: Toyota TSS2 emits this press/release pair
    # from the camera, and nothing in openpilot consumes it. Blocking has to disable through the
    # first-class event rather than only dropping the engage request, since the state machine
    # leaves the enabled state on a disable event and on nothing else.
    if any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents):
      self._blocked = not self._blocked
      if self._blocked:
        events.add(EventName.buttonCancel)

    # The point of the feature: a brake or gas tap must not take the steering with it.
    events.events = [e for e in events.events if e not in SUPPRESSED_EVENTS]

    if events.contains(ET.USER_DISABLE) or events.contains(ET.IMMEDIATE_DISABLE) or events.contains(ET.SOFT_DISABLE):
      # Cancel, main-switch-off, steer faults, and every fault latch it off until the driver
      # re-arms. SOFT_DISABLE is included so a soft-disable event without NO_ENTRY cannot make
      # this an engage/disable loop.
      self._blocked = True
    elif not enabled and not self._blocked and not events.contains(ET.NO_ENTRY) and not events.contains(ET.ENABLE):
      # The driver's arming gesture is the cruise main switch coming on, which upstream has no
      # event for on a stock-ACC car. buttonEnable carries the normal engage chime, and skipping
      # it while a NO_ENTRY is present is what keeps a standstill or an uncalibrated car from
      # nagging with refuse alerts — the attempt simply repeats once the blocker clears. ET.ENABLE
      # skips it when upstream already asked (its own pcmEnable on the same frame).
      events.add(EventName.buttonEnable)
