"""moonpilot's half-engaged policy: the safety param it asks for, and the events it changes.

The load-bearing test is the scripted one at the end: it runs the module's events through
upstream's own `StateMachine`, so "half-engaged" is asserted as the (state, enabled, active)
triple selfdrived would publish, not as a set of events the module happened to produce.
"""

import unittest
from typing import cast

from opendbc.car.structs import car
from opendbc.car.honda.values import HondaSafetyFlags
from opendbc.car.toyota.values import ToyotaFlags, ToyotaSafetyFlags
from opendbc.car.volkswagen.values import VolkswagenFlags, VolkswagenSafetyFlags
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.selfdrived.events import ET, EVENTS, Events
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, StateMachine

from moonpilot.features import LATERAL_ENGAGE, SLAM
from moonpilot.engage import (LateralEngage, car_unavailable_reason, moonpilot_actuator_gate,
                              moonpilot_engage, moonpilot_engage_safety_param)

ButtonType = car.CarState.ButtonEvent.Type
EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store."""

  def __init__(self, on=True):
    self._on = on

  def get(self, key, return_default=False):
    return self._on


def _params(on=True) -> Params:
  return cast(Params, FakeParams(on))


def _cp(brand='toyota', pcm_cruise=True, flags=0, safety_param=73, safety_configs=1, passive=False):
  """A real CarParams message, so the safety-param write is the one card actually does."""
  cp = car.CarParams.new_message()
  cp.brand = brand
  cp.pcmCruise = pcm_cruise
  cp.passive = passive
  cp.flags = int(flags)
  configs = cp.init('safetyConfigs', safety_configs)
  for c in configs:
    c.safetyParam = safety_param
  return cp


def _cs(available=False, enabled=False, buttons=()):
  cs = car.CarState.new_message()
  cs.cruiseState.available = available
  cs.cruiseState.enabled = enabled
  cs.buttonEvents = [{'type': t, 'pressed': p} for t, p in buttons]
  return cs


def _ps(controls_allowed=False, controls_allowed_lateral=False):
  ps = log.PandaState.new_message()
  ps.controlsAllowed = controls_allowed
  ps.controlsAllowedLateral = controls_allowed_lateral
  return ps


LKAS_PRESS = ((ButtonType.lkas, True), (ButtonType.lkas, False))


class TestEngageSafetyParam(unittest.TestCase):
  # One row per brand the gate accepts, with the bit its own mode header reads. A flag that did not
  # match the C would be a param the car never enables, so the two are tested together.
  BRAND_FLAGS = (('toyota', ToyotaSafetyFlags.LATERAL_ENGAGE, 0xFF),
                 ('honda', HondaSafetyFlags.LATERAL_ENGAGE, None),
                 ('volkswagen', VolkswagenSafetyFlags.LATERAL_ENGAGE, None))

  def test_available_car_gets_the_flag(self):
    for brand, flag, mask in self.BRAND_FLAGS:
      with self.subTest(brand=brand):
        cp = _cp(brand=brand)
        moonpilot_engage_safety_param(cp, _params(on=True))

        param = cp.safetyConfigs[0].safetyParam
        self.assertEqual(param, 73 | int(flag))
        # Nothing else of the fork's: the flag is the only bit added over the car's own param.
        self.assertEqual(param & ~(mask or 0xFFFF), int(flag) & ~(mask or 0xFFFF))

        # Idempotent, so a second call cannot corrupt the param
        moonpilot_engage_safety_param(cp, _params(on=True))
        self.assertEqual(cp.safetyConfigs[0].safetyParam, param)

  def test_param_off_leaves_carparams_alone(self):
    cp = _cp()
    moonpilot_engage_safety_param(cp, _params(on=False))
    self.assertEqual(cp.safetyConfigs[0].safetyParam, 73)

  def test_out_of_scope_cars_are_untouched(self):
    for cp in (
      _cp(brand='hyundai'),
      _cp(pcm_cruise=False),
      # Card replaces safetyConfigs with a noOutput config before the seam runs, so a
      # passive car's config must stay untouched however Toyota-shaped the params look
      _cp(passive=True),
      _cp(flags=ToyotaFlags.UNSUPPORTED_DSU),
      _cp(flags=ToyotaFlags.TSS2 | ToyotaFlags.UNSUPPORTED_DSU),
      # Volkswagen's other two platforms: their safety hooks never set acc_main_on in the
      # stock-ACC configuration this feature is for
      _cp(brand='volkswagen', flags=VolkswagenFlags.PQ),
      _cp(brand='volkswagen', flags=VolkswagenFlags.MLB),
    ):
      moonpilot_engage_safety_param(cp, _params(on=True))
      self.assertEqual(cp.safetyConfigs[0].safetyParam, 73, cp.flags)

  def test_multiple_safety_configs_are_untouched(self):
    cp = _cp(safety_configs=2)
    moonpilot_engage_safety_param(cp, _params(on=True))
    for config in cp.safetyConfigs:
      self.assertEqual(config.safetyParam, 73)


class TestEngageScope(unittest.TestCase):
  def test_inert_when_out_of_scope_or_off(self):
    for cp, params in (
      (_cp(), _params(on=False)),
      (_cp(brand='hyundai'), _params(on=True)),
      (_cp(pcm_cruise=False), _params(on=True)),
      (_cp(passive=True), _params(on=True)),
      (_cp(flags=ToyotaFlags.UNSUPPORTED_DSU), _params(on=True)),
      (_cp(brand='volkswagen', flags=VolkswagenFlags.PQ), _params(on=True)),
    ):
      engage = moonpilot_engage(cp, params)
      self.assertFalse(engage.enabled)

      # An inert instance changes nothing: the disengage events survive, nothing engages,
      # and the panda cross-check is upstream's.
      events = Events()
      events.add(EventName.pcmDisable)
      events.add(EventName.pedalPressed)
      engage.update(_cs(available=True), events, enabled=False)
      self.assertEqual(events.events, [EventName.pedalPressed, EventName.pcmDisable])
      self.assertFalse(engage.controls_allowed(_ps(controls_allowed_lateral=True)))

  def test_enabled_for_a_stock_acc_car(self):
    # Per-brand flags: the two brands' numbers overlap, so a Toyota flag on a Volkswagen is the
    # MLA platform as far as that gate is concerned.
    # Volkswagen MQB is the platform with no platform flag set; PQ and MLB are the ones excluded.
    for brand, flags in (('toyota', ToyotaFlags.TSS2), ('honda', 0), ('volkswagen', 0)):
      with self.subTest(brand=brand):
        self.assertTrue(moonpilot_engage(_cp(brand=brand, flags=flags), _params(on=True)).enabled)


class TestActuatorGate(unittest.TestCase):
  """The gate controlsd puts upstream's two actuator permissions through.

  A frame the safety layer rejects is silent to the sender, so what this pins is the direction of
  the failure: out of scope it must be upstream exactly, and in scope it must be the panda's word.
  """

  def test_inert_when_out_of_scope_or_off(self):
    # Upstream's behavior for every car the feature is not enabled on: both permissions pass
    # through whatever the panda says, including saying nothing at all.
    for cp, params in (
      (_cp(), _params(on=False)),
      (_cp(brand='hyundai'), _params(on=True)),
      (_cp(pcm_cruise=False), _params(on=True)),
      (_cp(passive=True), _params(on=True)),
      (_cp(flags=ToyotaFlags.UNSUPPORTED_DSU), _params(on=True)),
      (_cp(brand='volkswagen', flags=VolkswagenFlags.PQ), _params(on=True)),
    ):
      gate = moonpilot_actuator_gate(cp, params)
      for states in ([], [_ps()], [_ps(controls_allowed=True)], [_ps(controls_allowed_lateral=True)]):
        self.assertTrue(gate.lateral(states), states)
        self.assertTrue(gate.longitudinal(states), states)

  def test_half_engaged_follows_the_panda(self):
    gate = moonpilot_actuator_gate(_cp(), _params(on=True))

    # The half-engaged panda: steering armed, nothing longitudinal. That is the state of a car the
    # driver has switched on without setting ACC, and its speed stays the car's own ACC's business.
    half = [_ps(controls_allowed=False, controls_allowed_lateral=True)]
    self.assertTrue(gate.lateral(half))
    self.assertFalse(gate.longitudinal(half))

    # Setting ACC is what raises the panda's longitudinal grant, and with it the full engagement
    full = [_ps(controls_allowed=True, controls_allowed_lateral=True)]
    self.assertTrue(gate.lateral(full))
    self.assertTrue(gate.longitudinal(full))

    # Nothing granted, including a pandaState that has not arrived yet: openpilot commands nothing
    # rather than something the panda would refuse
    for states in ([], [_ps()]):
      self.assertFalse(gate.lateral(states), states)
      self.assertFalse(gate.longitudinal(states), states)


class TestEngagePolicy(unittest.TestCase):
  def setUp(self):
    self.engage = moonpilot_engage(_cp(), _params(on=True))
    self.assertTrue(self.engage.enabled)

  def _run(self, cs, enabled=False, extra_events=()):
    events = Events()
    for name in extra_events:
      events.add(name)
    self.engage.update(cs, events, enabled)
    return events

  def test_arms_when_the_main_switch_comes_on(self):
    events = self._run(_cs(available=True))
    self.assertEqual(events.events, [EventName.buttonEnable])
    self.assertTrue(events.contains(ET.ENABLE))

  def test_no_engage_request_while_a_no_entry_is_present(self):
    events = self._run(_cs(available=True), extra_events=(EventName.carNotReady,))
    self.assertTrue(EventName.buttonEnable not in events.events)
    # The blocker itself is untouched, so upstream still alerts on it
    self.assertTrue(EventName.carNotReady in events.events)

  def test_brake_and_gas_no_longer_disengage(self):
    events = self._run(_cs(available=True, enabled=True), extra_events=(EventName.pcmDisable, EventName.pedalPressed, EventName.gasPressedOverride))
    self.assertTrue(EventName.pcmDisable not in events.events)
    self.assertTrue(EventName.pedalPressed not in events.events)
    # Gas keeps its softer path: an override is not a disable, and steering continues through it
    self.assertTrue(EventName.gasPressedOverride in events.events)
    self.assertFalse(events.contains(ET.USER_DISABLE))

  def test_an_authoritative_disable_latches_until_rearmed(self):
    # Cancel from the wheel: USER_DISABLE, and the engage request stops
    events = self._run(_cs(available=True), extra_events=(EventName.buttonCancel,))
    self.assertTrue(EventName.buttonEnable not in events.events)

    # It stays latched across cycles, so the disable lasts longer than the frame it arrived in
    events = self._run(_cs(available=True))
    self.assertTrue(EventName.buttonEnable not in events.events)

    # Setting ACC re-arms, which is the driver asking for the full stack
    events = self._run(_cs(available=True, enabled=True))
    self.assertTrue(EventName.buttonEnable in events.events)

  def test_lkas_button_toggles(self):
    # Press with the release in the same frame: one toggle, not two
    events = self._run(_cs(available=True, enabled=True, buttons=LKAS_PRESS))
    self.assertTrue(EventName.buttonCancel in events.events)
    self.assertTrue(EventName.buttonEnable not in events.events)

    # Pressed again: the block clears and the engage request comes back
    events = self._run(_cs(available=True, buttons=LKAS_PRESS))
    self.assertTrue(EventName.buttonEnable in events.events)

  def test_lkas_off_holds_while_acc_stays_set(self):
    """The latch is released on ACC's rising edge, not while ACC is set.

    Clearing it on the level would disengage for one frame and re-engage on the next, so the
    driver's own off switch would not do anything while cruising with ACC set.
    """
    self._run(_cs(available=True, enabled=True))  # ACC set: nothing blocked
    events = self._run(_cs(available=True, enabled=True, buttons=LKAS_PRESS))
    self.assertTrue(EventName.buttonCancel in events.events)

    # ACC still set, no button, and the driver's state is still off — for as many frames as it takes
    for _ in range(5):
      events = self._run(_cs(available=True, enabled=True), enabled=False)
      self.assertTrue(EventName.buttonEnable not in events.events)

    # ACC cancel then re-set is the re-arm, because the transition is what counts
    self._run(_cs(available=True), enabled=False, extra_events=(EventName.pcmDisable,))
    self.assertTrue(EventName.buttonEnable not in self._run(_cs(available=True), enabled=False).events)
    events = self._run(_cs(available=True, enabled=True), extra_events=(EventName.pcmEnable,))
    self.assertTrue(EventName.buttonEnable not in events.events)  # upstream's own pcmEnable is the ask

  def test_an_authoritative_disable_latches(self):
    # One representative event per authoritative disable type, each asserted to really carry that
    # type, so the test cannot go vacuous if upstream moves the events around
    for event_type, event in (
      (ET.USER_DISABLE, EventName.steerDisengage),
      (ET.IMMEDIATE_DISABLE, EventName.steerUnavailable),
    ):
      engage = LateralEngage(True)
      events = Events()
      events.add(event)
      self.assertTrue(events.contains(event_type), event)
      engage.update(_cs(available=True), events, enabled=True)
      self.assertTrue(EventName.buttonEnable not in events.events, event_type)

      # And it stays latched on the next cycle, without the event: the disable lasts
      events = Events()
      engage.update(_cs(available=True), events, enabled=False)
      self.assertTrue(EventName.buttonEnable not in events.events, event_type)

  def test_a_soft_disable_is_not_latched(self):
    """Upstream recovers from a soft disable by itself, so half-engagement must not outlast it.

    Latching one would cost the state for the rest of the drive on a condition upstream rides
    out — an EPS temp fault, a door, a gear — which is what sunnypilot's own MADS "paused" state
    exists to avoid.
    """
    engage = LateralEngage(True)
    events = Events()
    events.add(EventName.espDisabled)
    self.assertTrue(events.contains(ET.SOFT_DISABLE))
    engage.update(_cs(available=True), events, enabled=True)
    self.assertTrue(EventName.buttonEnable not in events.events)

    # The condition clears and upstream has disabled: the engage request comes straight back, no
    # driver gesture needed
    events = Events()
    engage.update(_cs(available=True), events, enabled=False)
    self.assertTrue(EventName.buttonEnable in events.events)

  def test_a_soft_disable_that_stays_does_not_nag(self):
    """The request is gated on the soft disable itself, so a type without a NO_ENTRY cannot loop"""
    engage = LateralEngage(True)
    # bigModelFailed is the one SOFT_DISABLE event upstream raises without a NO_ENTRY
    for _ in range(10):
      events = Events()
      events.add(EventName.bigModelFailed)
      self.assertTrue(events.contains(ET.SOFT_DISABLE) and not events.contains(ET.NO_ENTRY))
      engage.update(_cs(available=True), events, enabled=False)
      self.assertTrue(EventName.buttonEnable not in events.events)

  def test_main_switch_off_clears_and_defers_to_upstream(self):
    self._run(_cs(available=True, enabled=True, buttons=LKAS_PRESS))  # block it
    events = self._run(_cs(available=False))
    self.assertTrue(EventName.buttonEnable not in events.events)

    # The main switch is upstream's own disengage, and it also un-blocks
    events = self._run(_cs(available=True, enabled=True), extra_events=(EventName.wrongCarMode,))
    self.assertTrue(EventName.buttonEnable not in events.events)
    self.assertTrue(events.contains(ET.USER_DISABLE))

  def test_controls_allowed_accepts_the_panda_lateral_flag(self):
    self.assertTrue(self.engage.controls_allowed(_ps(controls_allowed=True)))
    self.assertTrue(self.engage.controls_allowed(_ps(controls_allowed_lateral=True)))
    self.assertFalse(self.engage.controls_allowed(_ps()))

    inert = LateralEngage(False)
    self.assertFalse(inert.controls_allowed(_ps(controls_allowed_lateral=True)))
    self.assertTrue(inert.controls_allowed(_ps(controls_allowed=True)))


class TestScriptedDrive(unittest.TestCase):
  """The behavioral proof: the module's events drive upstream's state machine as intended.

  Each step replays what selfdrived does around the seam — data_sample (the state machine's
  output from the previous cycle), update_events (upstream's own sources for this car state,
  then the fork's update), then the state machine itself.
  """

  def setUp(self):
    self.engage = moonpilot_engage(_cp(flags=ToyotaFlags.TSS2), _params(on=True))
    self.events = Events()
    self.sm = StateMachine()
    self.enabled = False
    self.active = False
    self.prev = _cs()
    self.acc_set = False

  def _step(self, cs, brake=False, gas=False, acc_set=False, main_switch=True, buttons=(), fault=None):
    # Upstream's car events for this car state, the ones the fork's scope touches. pcmEnable and
    # pcmDisable are rising/falling edges in car_events, not levels, so the harness emits them the
    # same way: a driver's ACC cancel is one frame of pcmDisable, however long ACC stays off.
    self.events.clear()
    if not main_switch:
      self.events.add(EventName.wrongCarMode)  # not cruiseState.available
    if gas:
      self.events.add(EventName.gasPressedOverride)
    if brake:
      self.events.add(EventName.pedalPressed)
    if fault is not None:
      self.events.add(fault)  # a level-triggered condition, e.g. an EPS temp fault
    if acc_set and not self.acc_set:
      self.events.add(EventName.pcmEnable)
    elif self.acc_set and not acc_set:
      self.events.add(EventName.pcmDisable)
    self.acc_set = acc_set

    cs.cruiseState.available = main_switch
    cs.cruiseState.enabled = acc_set
    cs.buttonEvents = [{'type': t, 'pressed': p} for t, p in buttons]
    self.engage.update(cs, self.events, self.enabled)
    self.enabled, self.active = self.sm.update(self.events)
    self.prev = cs
    return self.sm.state, self.enabled, self.active

  def test_half_engaged_through_a_drive(self):
    # Main switch on: enabled and actuating, with no ACC set
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))

    # A brake tap: still driving, which is the point of the feature
    self.assertEqual(self._step(_cs(available=True), brake=True), (State.enabled, True, True))

    # ACC set on top: upstream's own path, unchanged
    self.assertEqual(self._step(_cs(available=True), acc_set=True), (State.enabled, True, True))

    # The wheel's LKAS button while cruising with ACC set: the driver's off switch has to hold.
    # Its own disable event rides the press frame only, so a latch cleared on ACC's level would
    # re-engage on the very next frame and the button would appear to do nothing.
    self.assertEqual(self._step(_cs(available=True), acc_set=True, buttons=LKAS_PRESS), (State.disabled, False, False))
    self.assertEqual(self._step(_cs(available=True), acc_set=True), (State.disabled, False, False))

    # Pressed again: back on, ACC set the whole time
    self.assertEqual(self._step(_cs(available=True), acc_set=True, buttons=LKAS_PRESS), (State.enabled, True, True))

    # ACC canceled: back to half-engaged, not disengaged
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))

    # Gas tap: an override, so still actuating
    self.assertEqual(self._step(_cs(available=True), gas=True), (State.overriding, True, True))

    # LKAS button: the driver's explicit off, with the press and release in one frame
    self.assertEqual(self._step(_cs(available=True), buttons=LKAS_PRESS), (State.disabled, False, False))

    # LKAS button again: engaged, no restart or ACC press needed
    self.assertEqual(self._step(_cs(available=True), buttons=LKAS_PRESS), (State.enabled, True, True))

    # Cruise main switch off: upstream's own disengage
    self.assertEqual(self._step(_cs(available=False), main_switch=False), (State.disabled, False, False))

    # And the main switch back on re-engages, which is the arming gesture
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))

  def test_a_soft_disable_recovers_by_itself(self):
    """A transient fault costs the state while it lasts, not the rest of the drive.

    Upstream's soft-disable timer is the whole mechanism, and the module stays out of its way: a
    fault that clears inside SOFT_DISABLE_TIME comes back on its own, and one that outlasts it
    comes back the frame the condition does. Neither takes the LKAS button or the main switch.
    """
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))

    # A short EPS temp fault: soft-disabling, still actuating, and enabled again the frame it
    # clears — the case a latch on SOFT_DISABLE would have ended for the drive
    self.assertEqual(self._step(_cs(available=True), fault=EventName.steerTempUnavailable), (State.softDisabling, True, True))
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))

    # A door open longer than the timer reaches disabled, and holds there while it stays open
    for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) + 2):
      self._step(_cs(available=True), fault=EventName.doorOpen)
    self.assertEqual(self._step(_cs(available=True), fault=EventName.doorOpen), (State.disabled, False, False))

    # Closed again: engaged, with no driver gesture
    self.assertEqual(self._step(_cs(available=True)), (State.enabled, True, True))


class TestCarGate(unittest.TestCase):
  """The car-side gate, which is also what the settings panels show a driver."""

  def test_in_scope_cars_have_no_reason(self):
    for brand in ('toyota', 'honda', 'volkswagen'):
      with self.subTest(brand=brand):
        self.assertIsNone(car_unavailable_reason(LATERAL_ENGAGE, _cp(brand=brand)))

  def test_each_out_of_scope_car_says_why(self):
    self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, _cp(brand='hyundai')),
                     "Toyota, Lexus, Honda or Volkswagen only")
    self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, _cp(passive=True)), "not in dashcam mode")
    self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, _cp(pcm_cruise=False)), "stock ACC only")
    self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, _cp(flags=ToyotaFlags.UNSUPPORTED_DSU)),
                     "not supported on this car")
    # Volkswagen's two platforms without a stock-ACC main switch in the safety layer
    for flags in (VolkswagenFlags.PQ, VolkswagenFlags.MLB):
      self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, _cp(brand='volkswagen', flags=flags)),
                       "not supported on this car")

  def test_the_panel_reason_and_the_behavior_are_the_same_gate(self):
    """A row that says a car cannot do it while the feature runs on it would be worse than silence."""
    for cp in (_cp(), _cp(brand='honda'), _cp(brand='volkswagen'), _cp(brand='hyundai'), _cp(passive=True),
               _cp(pcm_cruise=False), _cp(flags=ToyotaFlags.UNSUPPORTED_DSU),
               _cp(brand='volkswagen', flags=VolkswagenFlags.PQ)):
      self.assertEqual(car_unavailable_reason(LATERAL_ENGAGE, cp) is None,
                       moonpilot_engage(cp, _params(on=True)).enabled)

  def test_features_without_a_car_requirement_are_unaffected(self):
    self.assertIsNone(car_unavailable_reason(SLAM, _cp(brand='hyundai')))

  def test_no_carparams_yet_is_not_a_reason(self):
    """The panel renders before carParams lands, and an unloaded CP is not a verdict on the car."""
    self.assertIsNone(car_unavailable_reason(LATERAL_ENGAGE, None))


class TestHalfEngagementBanner(unittest.TestCase):
  """The banner for a car that is armed and not steering.

  It exists because the suppressors are also what make the latch silent: `pcmDisable` and
  `pedalPressed` are the events upstream would have explained a disengagement with, so a latched
  car looks exactly like a car with nothing switched on.
  """

  def setUp(self):
    self.engage = moonpilot_engage(_cp(), _params(on=True))

  def _run(self, cs, enabled=False, extra_events=()):
    events = Events()
    for name in extra_events:
      events.add(name)
    self.engage.update(cs, events, enabled)
    return events

  def _latch(self):
    # controlsMismatch is an IMMEDIATE_DISABLE, and it is what a panda that never grants steering
    # raises -- the failure this banner exists for. A steerOverride is not a disable at all.
    return self._run(_cs(available=True), enabled=True, extra_events=(EventName.controlsMismatch,))

  def test_the_banner_appears_once_the_latch_is_set(self):
    events = self._latch()
    self.assertTrue(self.engage._blocked)
    self.assertTrue(EventName.lateralEngageOff in events.events)

  def test_no_banner_while_it_works(self):
    events = self._run(_cs(available=True), enabled=True)
    self.assertFalse(self.engage._blocked)
    self.assertNotIn(EventName.lateralEngageOff, events.events)

  def test_the_banner_clears_on_re_arm(self):
    self._latch()
    events = self._run(_cs(available=True, buttons=((ButtonType.lkas, True),)))
    self.assertFalse(self.engage._blocked)
    self.assertNotIn(EventName.lateralEngageOff, events.events)

  def test_no_banner_for_a_hold_upstream_already_explains(self):
    """Narrow on purpose: a standstill or a loading model carries its own alert already."""
    events = self._run(_cs(available=True), extra_events=(EventName.carNotReady,))
    self.assertFalse(self.engage._blocked)
    self.assertNotIn(EventName.lateralEngageOff, events.events)

  def test_the_banner_cannot_feed_the_latch(self):
    """It must carry no type the state machine acts on, or it would hold its own cause on."""
    self._latch()
    # A clean cycle with the latch still set: the banner is added, and nothing else is.
    events = self._run(_cs(available=True))
    self.assertTrue(EventName.lateralEngageOff in events.events)
    for et in (ET.USER_DISABLE, ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE, ET.NO_ENTRY, ET.ENABLE):
      self.assertFalse(events.contains(et), f"the banner must not carry {et}")

  def test_the_event_has_an_alert(self):
    """`create_alerts` does `EVENTS[e].keys()`, so a name with no entry is a selfdrived crash."""
    self.assertTrue(EventName.lateralEngageOff in EVENTS)
    self.assertTrue(ET.PERMANENT in EVENTS[EventName.lateralEngageOff])

  def test_it_is_debounced_before_it_shows(self):
    """A second of the condition before anything appears, so the frame or two an engage takes --
    or a panda granting a frame late -- cannot flash the banner."""
    engage = moonpilot_engage(_cp(), _params(on=True))
    events = Events()

    def cycle(enabled):
      events.clear()
      engage.update(_cs(available=True), events, enabled)
      return events.create_alerts([ET.PERMANENT])

    # Latch it, then hold the condition and watch the counter do the work
    events.clear()
    events.add(EventName.controlsMismatch)
    engage.update(_cs(available=True), events, True)
    self.assertTrue(engage._blocked)
    self.assertEqual(cycle(False), [], "the first cycle must not flash it")

    # Halfway through the delay it is still down...
    for _ in range(int(0.5 / DT_CTRL)):
      self.assertEqual(cycle(False), [])
    # ...and a full second of the condition is enough to put it up, where it stays.
    for _ in range(int(0.5 / DT_CTRL) + 2):
      shown = cycle(False)
    self.assertTrue(shown, "the banner should be up after a second of the condition")
    self.assertEqual(shown[0].alert_text_1, "Lateral Engagement Off")
    self.assertTrue(cycle(False), "and it stays up while the latch does")

  def test_the_alert_is_permanent_so_it_stays_up(self):
    """ET.PERMANENT is never in `clear_event_types`, unlike WARNING and NO_ENTRY."""
    self.assertTrue(ET.PERMANENT in EVENTS[EventName.lateralEngageOff])


if __name__ == "__main__":
  unittest.main()
