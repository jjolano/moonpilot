"""moonpilot's lateral-only engagement: steer from the cruise main switch, not from ACC set.

Upstream engages everything on one edge — stock ACC engagement — and panda enforces that edge at
its own layer, so a fork cannot widen it from openpilot alone. The fork's side of that is the
`controls_allowed_lateral` permission in forked `opendbc`/`panda` (see AGENTS.md, and
`opendbc/safety/moonpilot/lateral_engage.h` for the rule); this module is the openpilot side of
the same decision: it asks the safety param for the permission, then holds openpilot in its
enabled state while the driver is only half-engaged.

What the module does at runtime:

  - `moonpilot_engage_safety_param`, called once by card before `CarParams` is written, sets this
    brand's safety-param flag (`LATERAL_ENGAGE_FLAGS`) that turns the panda rule on for this car.
    Only a car in scope reaches it (see `_available`), so every other car's `CarParams` is
    upstream's byte for byte.
  - `car_unavailable_reason`, what both settings panels ask: why a driver cannot use a feature on
    this car, or `None`. For `LATERAL_ENGAGE` that is the panel's half of `_available` — the same
    `_unsupported_reason` the gate itself reads — so a car that can never be half-engaged says so
    instead of reading as on and doing nothing. `TORQUE_LATERAL` and `LONGITUDINAL` have one too,
    panel-facing only: each names a car whose controlsd branch the fork's seam is not in, so the
    feature is inert rather than gated, and nothing but the row's reason changes.
  - `moonpilot_actuator_gate`, built once by controlsd: the panda's two grants, read back by the
    client that produced them. A frame the safety layer rejects is silent to the sender, so a
    half-engaged car has to command nothing the panda will refuse — see `ActuatorGate`.
  - `LateralEngage.update`, called from selfdrived once every event source has contributed, is
    the engage/disengage policy. It suppresses the three events whose whole purpose is the behavior
    being replaced — `pcmDisable` (level-triggered whenever stock ACC is off), `pedalPressed` (the
    brake/gas disengage) and `buttonCancel` (ACC's cancel button) — and asks to engage when the
    driver turns the cruise main switch on. Disengaging is upstream's: the main switch, a fault and
    a lane-keeping fault still land in `wrongCarMode` or a disable event, and the LKAS button's own
    press adds the one `buttonCancel` that survives the filter. An authoritative disable — `USER_DISABLE`
    or `IMMEDIATE_DISABLE`, which on this car is the main switch, a steer fault or a lane-keeping
    fault — latches the half-engaged state off until the driver re-arms with the LKAS button or by
    cycling the main switch; a soft disable is left to upstream's own state machine, which returns
    to enabled by itself once the condition clears.

    It also raises `lateralEngageOff` while the latch is set — a permanent banner, because the
    suppressors above are exactly what make a latched car silent: the events upstream would have
    explained the disengagement with are the ones filtered out. Keyed on the latch alone, so a
    `NO_ENTRY` hold or a soft disable, which upstream already alerts on, does not double up.

    One of upstream's disengages is inert here rather than suppressed: `steerDisengage` — the panda
    rule's own `steering_disengage` — is a signal only Tesla's rx hook ever sets, so on all three
    brands a driver's torque is upstream's blending path rather than a disarm. Cancel used to be a
    per-brand fact worth this paragraph; it is a suppression now (see `SUPPRESSED_EVENTS`), which is
    the same behavior on all three brands instead of a coincidence of which wheel is wired.

Scope, and the ceilings that come with it:

  - Three brands, and one platform caveat inside the third. Toyota/Lexus, Honda and Volkswagen
    MQB/MEB, which is what `CP.pcmCruise` is for: on Honda and Volkswagen it is `not
    openpilotLongitudinalControl` and so does describe who owns speed, but on Toyota it is a default
    that stays true under openpilot longitudinal control too. Those cars are therefore in scope, and
    their half-engagement is steer-only — the driver's foot owns speed until ACC is set, which is
    the grant the longitudinal half of the gate waits for. PQ is excluded because its safety hook
    sets `acc_main_on` only under openpilot longitudinal control, and MLB never sets it.
    Toyota's `UNSUPPORTED_DSU` cars are excluded too: they read the cruise main switch out of
    `DSU_CRUISE` at 5 Hz, below panda's 10 Hz rx-check minimum, so the feature would be inert
    there by construction rather than by a runtime check that would fail on the road.
  - The decision to be half-engaged is fixed at construction, like the fork's other behavior
    toggles: the module reads its param once, and the panel row says a restart is needed.
  - No fresh permission is invented for the acceleration side. `controlsAllowed` keeps upstream's
    meaning, so `CC.longActive` waits for the panda's own longitudinal grant and the car's own ACC
    owns speed until the driver sets it — which is the whole point of "half".
"""

from opendbc.car.structs import car
from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.car.toyota.values import ToyotaFlags, ToyotaSafetyFlags
from opendbc.car.volkswagen.values import VolkswagenFlags, VolkswagenSafetyFlags
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived.events import ET

from moonpilot.features import LATERAL_ENGAGE, LONGITUDINAL, TORQUE_LATERAL, Feature, enabled

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName

# The three events the half-engaged state is defined against: stock ACC being off (level-triggered,
# so it is present in every frame the driver has not set ACC), a brake or gas tap, and ACC's cancel
# button. Cancel belongs in that list for the reason the feature exists: it stops the car's ACC, not
# the steering, and it is the one of the three a driver is most likely to press while expecting
# openpilot to keep the lane. Honda and Volkswagen emit `ButtonType.cancel` -- Toyota's wheel emits
# no cancel at all -- and `car_events.py` turns any such press into `buttonCancel` for every brand
# but Hyundai, so suppressing the event is what makes cancel behave the same way on all three.
#
# The event carries `ET.NO_ENTRY` as well as `ET.USER_DISABLE`, so suppressing it also drops the
# momentary "cancel blocks a new engage" hold. That is the same statement from the other side: on a
# half-engaged car cancel is not a request to stop driving, and the driver's own off switch is the
# LKAS button, which is still honored outright (see `update`).
#
# Everything else upstream raises keeps working — in particular gas keeps its softer
# `gasPressedOverride`, which is an override rather than a disable, so steering continues through
# it.
SUPPRESSED_EVENTS = (EventName.pcmDisable, EventName.pedalPressed, EventName.buttonCancel)

# The brands the forked safety layer carries the permission for, and the safety-param bit that
# turns it on for each. One row per brand, and the bit is a wire contract with that brand's own
# mode header -- `HONDA_PARAM_LATERAL_ENGAGE`, `TOYOTA_PARAM_LATERAL_ENGAGE`,
# `FLAG_VOLKSWAGEN_LATERAL_ENGAGE` -- so the two must change together (opendbc_repo/AGENTS.md).
#
# What a brand needs to be listed: its safety rx hook must decode the cruise main switch into
# `acc_main_on`, and the message carrying it must already be rx-checked. Toyota's the fork added;
# Honda's and Volkswagen's are upstream's own, which is why neither needs an rx check of its own.
# Everyone else needs that decode written and its rate validated on the car.
LATERAL_ENGAGE_FLAGS = {
  'toyota': ToyotaSafetyFlags.LATERAL_ENGAGE,
  'honda': HondaSafetyFlags.LATERAL_ENGAGE,
  'volkswagen': VolkswagenSafetyFlags.LATERAL_ENGAGE,
}


def _unsupported_reason(CP) -> str | None:
  """Why this car cannot be half-engaged, or `None` when it can.

  The single gate: `_available` is this, and so is the settings panel's disabled-with-reason, so the
  row a driver sees and the behavior they get cannot disagree. The strings are value-line sized --
  mici puts them there, and only tizi has room to wrap -- so the detail lives in the feature's own
  description rather than here.

  `not CP.passive` is not redundant with card's placement: card sets `passive` and replaces
  `safetyConfigs` with a single noOutput config *before* the seam runs, so without this a
  dashcam-mode car would still have the flag ORed into a config that never reads it.
  """
  if CP is None:
    # CarParams is not loaded yet, which is not a verdict on the car: no reason. The panel hoists
    # this itself; `_available` only ever sees the real thing, from card and selfdrived.
    return None
  if CP.brand not in LATERAL_ENGAGE_FLAGS:
    return "Toyota, Lexus, Honda or Volkswagen only"
  if CP.passive:
    return "not in dashcam mode"
  if not CP.pcmCruise:
    return "stock ACC only"
  if CP.brand == 'toyota' and CP.flags & ToyotaFlags.UNSUPPORTED_DSU:
    # Those read the main switch out of DSU_CRUISE at 5 Hz, below panda's 10 Hz rx-check minimum.
    return "not supported on this car"
  if CP.brand == 'volkswagen' and CP.flags & (VolkswagenFlags.PQ | VolkswagenFlags.MLB):
    # Their safety hooks never set `acc_main_on` in the stock-ACC configuration this feature is
    # for: PQ's is set only under openpilot longitudinal control, and MLB's is not set at all.
    return "not supported on this car"
  return None


def _available(CP) -> bool:
  """Whether this car is one the half-engaged state can be built for at all."""
  return _unsupported_reason(CP) is None


def _torque_steered(CP) -> bool:
  """Whether controlsd picks the branch the fork's torque controller seam sits in.

  The whole dispatch, not `lateralTuning` alone: `steerControlType` is tested first, so an
  angle- or curvature-steered car never reaches the torque branch however its tuning reads
  (openpilot/selfdrive/controls/controlsd.py:64-71).
  """
  return (CP.steerControlType not in (car.CarParams.SteerControlType.angle, car.CarParams.SteerControlType.curvature)
          and CP.lateralTuning.which() == 'torque')


def car_unavailable_reason(feature: Feature, CP) -> str | None:
  """Why a driver cannot use this feature on this car, or `None`. What the settings panels ask.

  The dependency gate in `moonpilot/features.py` cannot see CarParams, so a feature with a car
  requirement needs a second gate; this is it, and it lives here with the requirement itself rather
  than as a per-feature branch inside both panel files. `ui_state.CP` is None until carParams
  arrives offroad, which reads as "say nothing".

  Three features have a reason, and only the first of them is a safety gate. `LATERAL_ENGAGE`'s is
  the same `_unsupported_reason(CP)` `_available` reads, so the row and the behavior cannot
  disagree. `TORQUE_LATERAL`'s and `LONGITUDINAL`'s are panel-facing only — nothing gates behavior
  on them — because a car outside their seam's dispatch is a car the seam never runs on: the
  feature is inert there, not gated.
  """
  if CP is None:
    return None
  if feature is LATERAL_ENGAGE:
    return _unsupported_reason(CP)
  if feature is TORQUE_LATERAL and not _torque_steered(CP):
    return "torque-steered cars only"
  if feature is LONGITUDINAL and not CP.openpilotLongitudinalControl:
    return "openpilot-longitudinal cars only"
  return None


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
  CP.safetyConfigs[0].safetyParam |= int(LATERAL_ENGAGE_FLAGS[CP.brand])


class ActuatorGate:
  """The panda's own grants, read back by the client that produced them.

  A frame the safety layer rejects is silent to the sender: the car simply never receives the
  message. So a half-engaged car has to command nothing the panda will refuse, and it otherwise
  does two such things. Steering dies on the `desired_torque_last` reset in
  `steer_torque_cmd_checks`, which blocks every later frame until the command comes home within
  `MAX_RATE_UP` of zero — the car's lane-keeping ECU reads the resulting silence as a message
  dropout. And an ACC_CONTROL at `controls_allowed == false` never reaches the PCM, which on an
  openpilot-longitudinal Toyota has no other ACC stream and faults.

  True — upstream's behavior — on every car the feature is not enabled for, which is what keeps a
  stock config unchanged. Built once, at construction, like the fork's other behavior toggles.
  """

  def __init__(self, CP, params: Params | None = None) -> None:
    params = params if params is not None else Params()
    self._gated = enabled(LATERAL_ENGAGE, params) and _available(CP)

  def lateral(self, panda_states) -> bool:
    """Whether openpilot may command steering: the panda's lateral grant, once armed."""
    return (not self._gated) or any(ps.controlsAllowedLateral for ps in panda_states)

  def longitudinal(self, panda_states) -> bool:
    """Whether openpilot may command acceleration: the panda's own longitudinal grant.

    Which is stock ACC engagement, so a half-engaged car keeps sending ACC_CONTROL — at
    ACCEL_CMD=0, the inactive value `longitudinal_accel_checks` accepts — and the PCM keeps its
    stream and cannot fault. The driver's foot owns speed until they set ACC, and setting it is
    what raises this grant and gives the normal, fully longitudinal engagement.
    """
    return (not self._gated) or any(ps.controlsAllowed for ps in panda_states)


def moonpilot_actuator_gate(CP, params: Params | None = None) -> ActuatorGate:
  """The seam's fork side for controlsd. Always an instance; an inert one when out of scope."""
  return ActuatorGate(CP, params)


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

    # The wheel's LKAS button is a free explicit on/off, and this is brand-generic on purpose: the
    # two brands in scope that emit `ButtonType.lkas` are Toyota (TSS2, from the camera) and Honda
    # (SCM_BUTTONS' LKAS setting button), and nothing in openpilot or opendbc consumes the event on
    # either --- no `ButtonType.lkas` reader exists in the tree. Volkswagen emits no such button at
    # all, every one of its buttons being an ACC function, so there the re-arm gestures are the two
    # that need no button: cycling the main switch above and ACC's rising edge.
    #
    # Blocking has to disable through the first-class event rather than only dropping the engage
    # request, since the state machine leaves the enabled state on a disable event and on nothing
    # else.
    lkas_pressed = any(be.type == ButtonType.lkas and be.pressed for be in CS.buttonEvents)
    if lkas_pressed:
      self._blocked = not self._blocked

    # The point of the feature: a brake or gas tap must not take the steering with it, and neither
    # must ACC's cancel button.
    events.events = [e for e in events.events if e not in SUPPRESSED_EVENTS]

    if lkas_pressed and self._blocked:
      # Added after the filter, not with the rest of the button handling: this is the one
      # `buttonCancel` that has to survive, because blocking has to disable through a first-class
      # event rather than only dropping the engage request -- the state machine leaves the enabled
      # state on a disable event and on nothing else. A latched car whose event was filtered would
      # steer while the driver watched the button do nothing.
      events.add(EventName.buttonCancel)

    if events.contains(ET.USER_DISABLE) or events.contains(ET.IMMEDIATE_DISABLE):
      # The main switch going off, a steer fault, and every authoritative disable latch it off
      # until the driver re-arms.
      self._blocked = True
    elif not enabled and not self._blocked and not events.contains(ET.ENABLE) and not events.contains(ET.NO_ENTRY) and not events.contains(ET.SOFT_DISABLE):
      # The driver's arming gesture is the cruise main switch coming on, which upstream has no
      # event for on a stock-ACC car. buttonEnable carries the normal engage chime, and skipping
      # it while a NO_ENTRY is present is what keeps a standstill or an uncalibrated car from
      # nagging with refuse alerts — the attempt simply repeats once the blocker clears. ET.ENABLE
      # skips it when upstream already asked (its own pcmEnable on the same frame).
      #
      # A soft disable is deliberately not latched: upstream's own state machine already treats it
      # as recoverable, returning to enabled when the condition clears inside SOFT_DISABLE_TIME,
      # and a transient one — an EPS temp fault, a door, a gear — should cost the state for as
      # long as it lasts, not for the rest of the drive. The NO_ENTRY that every SOFT_DISABLE
      # event but bigModelFailed also carries is what keeps the request from re-firing while one
      # holds; SOFT_DISABLE is excluded from the request itself so that a type without one cannot
      # engage and soft-disable in a loop.
      events.add(EventName.buttonEnable)

    if self._blocked:
      # The one state a driver cannot read off the road. The feature is on, nobody suppressed their
      # switch, and the car does not steer: the suppressors that keep a brake tap from disengaging
      # are also what make this latch silent, since the events upstream would have explained it with
      # are exactly the ones filtered out above. A permanent banner says so until the driver re-arms.
      #
      # Deliberately *not* the broader "armed and not steering": a NO_ENTRY hold (a standstill with
      # the brake, a model still loading) already carries upstream's own alert, and a soft disable is
      # upstream's recoverable one, so a banner there would double up on a state the driver is being
      # told about and can do nothing extra for. The latch is the one that says nothing by itself.
      #
      # It also covers the safety layer failing, which is the failure this feature cannot see
      # otherwise: a panda that grants nothing -- an rx check that never settles, a config the
      # forklift never flagged, a firmware without the health bit -- eventually raises
      # controlsMismatch, which is an IMMEDIATE_DISABLE, so it lands *here*, as the latch. There is
      # no reason string for it because there cannot be: the grant is cleared whenever openpilot is
      # not engaged, so a latched car and an ungranted one are indistinguishable from the outside.
      events.add(EventName.lateralEngageOff)
