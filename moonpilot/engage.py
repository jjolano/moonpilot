"""Lateral-only engagement; see `moonpilot/docs/engage.md` and AGENTS.md."""

from opendbc.car.structs import car
from opendbc.car.toyota.carcontroller import MAX_STEER_RATE, MAX_STEER_RATE_FRAMES
from opendbc.car.chrysler.values import ChryslerSafetyFlags
from opendbc.car.ford.values import FordSafetyFlags
from opendbc.car.gm.values import GMSafetyFlags
from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.car.hyundai.values import HyundaiSafetyFlags
from opendbc.car.mazda.values import MazdaSafetyFlags
from opendbc.car.mg.values import MgSafetyFlags
from opendbc.car.nissan.values import NissanSafetyFlags
from opendbc.car.psa.values import PsaSafetyFlags
from opendbc.car.rivian.values import RivianSafetyFlags
from opendbc.car.subaru.values import SubaruSafetyFlags
from opendbc.car.tesla.values import TeslaSafetyFlags
from opendbc.car.toyota.values import ToyotaFlags, ToyotaSafetyFlags
from opendbc.car.volkswagen.values import VolkswagenSafetyFlags
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import ET

from moonpilot.features import ENGINE_COAST, LATERAL_ENGAGE, in_force

ButtonType = car.CarState.ButtonEvent.Type
GearShifter = car.CarState.GearShifter
EventName = log.OnroadEvent.EventName

# These are the three events the half-engaged policy suppresses. Keep all other upstream events,
# including gasPressedOverride and LKAS, on their normal paths.
SUPPRESSED_EVENTS = (EventName.pcmDisable, EventName.pedalPressed, EventName.buttonCancel)

# Safety-param bits shared with each brand's forked safety mode header.
#
# Toyota, Honda, and Volkswagen arm on the car's cruise main switch; all other listed brands arm on
# the host. Volkswagen shares one flag across MQB, MEB, PQ, and MLB.
LATERAL_ENGAGE_FLAGS = {
  'toyota': ToyotaSafetyFlags.LATERAL_ENGAGE,
  'honda': HondaSafetyFlags.LATERAL_ENGAGE,
  'volkswagen': VolkswagenSafetyFlags.LATERAL_ENGAGE,
  'chrysler': ChryslerSafetyFlags.LATERAL_ENGAGE,
  'ford': FordSafetyFlags.LATERAL_ENGAGE,
  'gm': GMSafetyFlags.LATERAL_ENGAGE,
  'hyundai': HyundaiSafetyFlags.LATERAL_ENGAGE,
  'mazda': MazdaSafetyFlags.LATERAL_ENGAGE,
  'mg': MgSafetyFlags.LATERAL_ENGAGE,
  'nissan': NissanSafetyFlags.LATERAL_ENGAGE,
  'psa': PsaSafetyFlags.LATERAL_ENGAGE,
  'rivian': RivianSafetyFlags.LATERAL_ENGAGE,
  'subaru': SubaruSafetyFlags.LATERAL_ENGAGE,
  'tesla': TeslaSafetyFlags.LATERAL_ENGAGE,
}


def unsupported_reason(CP) -> str | None:
  """Why this car cannot be half-engaged, or `None` when it can.

  `not CP.passive` is not redundant with card's placement: card sets `passive` and replaces
  `safetyConfigs` with a single noOutput config *before* the seam runs, so without this a
  dashcam-mode car would still have the flag ORed into a config that never reads it.
  """
  if CP.brand not in LATERAL_ENGAGE_FLAGS:
    return "not supported on this car"
  if CP.passive:
    return "not in dashcam mode"
  if not CP.pcmCruise:
    return "stock ACC only"
  if CP.brand == 'toyota' and CP.flags & ToyotaFlags.UNSUPPORTED_DSU:
    # Those read the main switch out of DSU_CRUISE at 5 Hz, below panda's 10 Hz rx-check minimum.
    return "not supported on this car"
  if len(CP.safetyConfigs) != 1:
    # A partial flag would put openpilot and panda out of agreement.
    return "not supported on this car"
  return None


def moonpilot_engage_safety_param(CP, params: Params | None = None) -> None:
  """The fork's CarParams hook: the safety param the forked safety layer reads, and the engine-coast
  flag the forked Toyota carcontroller reads (`opendbc/car/moonpilot/engine_coast.py`).

  Called by card after the passive/dashcam handling and before `CarParams` is written, which is
  the only window where either can still change — panda, selfdrived, plannerd and the car's own
  controller (which holds this same CarParams object) all read them from there.
  """
  params = params or Params()
  if in_force(ENGINE_COAST, params, CP):
    CP.flags |= int(ToyotaFlags.MOONPILOT_ENGINE_COAST)
  if in_force(LATERAL_ENGAGE, params, CP):
    CP.safetyConfigs[0].safetyParam |= int(LATERAL_ENGAGE_FLAGS[CP.brand])


MOONPILOT_TOYOTA_STEER_QUIET_FRAMES = 10  # 100 ms at controlsd's 100 Hz


class ActuatorGate:
  """Read panda grants back before sending the corresponding actuator commands."""

  def __init__(self, CP, params: Params | None = None) -> None:
    self._gated = in_force(LATERAL_ENGAGE, params if params is not None else Params(), CP)
    self._toyota_torque = self._gated and CP.brand == 'toyota' and CP.steerControlType == car.CarParams.SteerControlType.torque
    self._steer_hot_frames = 0
    self._steer_quiet_frames = 0

  def lateral(self, panda_states, CS) -> bool:
    """The panda's lateral grant, with Toyota torque requests paused during rapid wheel motion.

    Stateful: it must stay the last operand of controlsd's `and` chain, so its counters advance only on cycles
    upstream already permits."""
    if not self._gated:
      return True
    if not any(ps.controlsAllowedLateral for ps in panda_states):
      return False
    if self._toyota_torque:
      if abs(CS.steeringRateDeg) >= MAX_STEER_RATE:
        self._steer_hot_frames += 1
        if self._steer_hot_frames > MAX_STEER_RATE_FRAMES:
          self._steer_quiet_frames = MOONPILOT_TOYOTA_STEER_QUIET_FRAMES
      else:
        self._steer_hot_frames = 0
        if self._steer_quiet_frames:
          self._steer_quiet_frames -= 1
      if self._steer_quiet_frames:
        return False
    return True

  def longitudinal(self, panda_states) -> bool:
    """Whether openpilot may command acceleration: the panda's own longitudinal grant.

    Which is stock ACC engagement, so a half-engaged car keeps sending ACC_CONTROL — at
    ACCEL_CMD=0, the inactive value `longitudinal_accel_checks` accepts — and the PCM keeps its
    stream and cannot fault. The driver's foot owns speed until they set ACC, and setting it is
    what raises this grant and gives the normal, fully longitudinal engagement.
    """
    return (not self._gated) or any(ps.controlsAllowed for ps in panda_states)


def moonpilot_engage(CP, params: Params | None = None) -> 'LateralEngage':
  """The seam's fork side for selfdrived. Always an instance; an inert one when out of scope."""
  return LateralEngage(in_force(LATERAL_ENGAGE, params or Params(), CP))


class LateralEngage:
  """The half-engaged engage/disengage policy, run once per selfdrived cycle."""

  def __init__(self, on: bool) -> None:
    self.enabled = on
    # An authoritative disable stays latched until a re-arm gesture. ACC's rising edge, rather
    # than its level, preserves an LKAS-button disable while ACC remains set.
    self._blocked = False
    self._acc_set_prev = False
    # A gear openpilot cannot drive in raises `wrongGear` on every frame it is held, so the refusal
    # request is one-shot per entry: an engagement attempt, not a retry. Reset when the gear becomes
    # drivable again or the cruise main switch goes off.
    self._wrong_gear_seen = False

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

    # `reverseGear` is the passive full-screen alert this feature replaces. Keep `wrongGear`, whose
    # NO_ENTRY is the alert for an actual engagement attempt, and filter the passive event even when
    # the cruise main switch is off.
    events.events = [e for e in events.events if e != EventName.reverseGear]
    if CS.gearShifter == GearShifter.reverse and EventName.wrongGear not in events.events:
      # Unreachable on every car today: no interface lists reverse in `DRIVABLE_GEARS`, so `car_events` still raises
      # `wrongGear`. It is kept because it is the only thing that would, if one ever did -- `reverseGear` is filtered
      # above, so half-engaged steering in reverse would otherwise carry no alert at all.
      events.add(EventName.wrongGear)

    # Neutral is allowed only while ACC is off: that is the state this re-arm signal also changes.
    if CS.gearShifter == GearShifter.neutral and not CS.cruiseState.enabled:
      events.events = [e for e in events.events if e != EventName.wrongGear]
    wrong_gear = EventName.wrongGear in events.events
    first_wrong_gear = wrong_gear and not self._wrong_gear_seen

    if not CS.cruiseState.available:
      # The cruise main switch is off, which is upstream's own disengage: the car events raise
      # wrongCarMode, and the panda rule drops its permission on acc_main's falling edge.
      self._blocked = False
      self._acc_set_prev = False
      self._wrong_gear_seen = False
      return

    if CS.cruiseState.enabled and not self._acc_set_prev:
      self._blocked = False
    self._acc_set_prev = CS.cruiseState.enabled

    # The wheel's LKAS button is a free explicit on/off, and this is brand-generic on purpose: the
    # two brands in scope that emit `ButtonType.lkas` are Toyota (TSS2, from the camera) and Honda
    # (SCM_BUTTONS' LKAS setting button), and nothing in openpilot or opendbc consumes the event on
    # either --- no `ButtonType.lkas` reader exists in the tree. Volkswagen emits no such button at
    # all, every one of its buttons being an ACC function, so there the re-arm gestures are the two
    # that need no button: cycling the main switch above and ACC's rising edge.
    #

    lkas_pressed = any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents)
    was_blocked = self._blocked
    if lkas_pressed:
      self._blocked = not self._blocked
    lkas_rearm = lkas_pressed and was_blocked and not self._blocked

    # The point of the feature: a brake or gas tap must not take the steering with it, and neither
    # must ACC's cancel button. `reverseGear` was filtered above; `wrongGear` is still the source of
    # the one-shot "Gear not D" refusal on an engagement attempt, wherever the gear is one the car
    # cannot drive in.
    events.events = [e for e in events.events if e not in SUPPRESSED_EVENTS]

    if lkas_pressed and self._blocked:
      # Blocking needs a first-class disable event: StateMachine leaves enabled only on one.
      # This event is added after filtering so the driver's LKAS-button disable survives.
      events.add(EventName.buttonCancel)
    if events.contains(ET.USER_DISABLE) or events.contains(ET.IMMEDIATE_DISABLE):
      # The main switch going off, a steer fault, and every authoritative disable latch it off
      # until the driver re-arms.
      self._blocked = True
    elif not enabled and not self._blocked and not events.contains(ET.ENABLE):
      # The driver's arming gesture is the cruise main switch coming on, which upstream has no
      # event for on a stock-ACC car. buttonEnable carries the normal engage chime. In park or
      # reverse, emit it once on entry (or on an LKAS re-arm) even though wrongGear is present:
      # StateMachine needs ENABLE plus NO_ENTRY to show the refusal alert, and `_wrong_gear_seen`
      # prevents a refusal retry on every frame the gear is held. ET.ENABLE skips it when upstream
      # already asked.
      wrong_gear_attempt = wrong_gear and (first_wrong_gear or lkas_rearm)
      ordinary_attempt = not wrong_gear and not events.contains(ET.NO_ENTRY) and not events.contains(ET.SOFT_DISABLE)
      if wrong_gear_attempt or ordinary_attempt:
        events.add(EventName.buttonEnable)

    self._wrong_gear_seen = wrong_gear

    if wrong_gear and enabled:
      # A wrong gear quietly stops steering without latching the state. Keep `wrongGear` for the
      # one-shot "Gear not D" refusal on an engagement attempt.
      events.add(EventName.buttonCancel)

    if self._blocked:
      # Point to the one state the driver cannot otherwise see: upstream's disable events are
      # suppressed, so the latched state needs its own banner. Do not duplicate upstream alerts.
      events.add(EventName.lateralEngageOff)
