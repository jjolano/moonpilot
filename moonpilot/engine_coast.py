"""Engine-braking coast, planner side (`MoonpilotEngineCoast`, off by default; a test toggle).

On this car the planner's lamp-free authority is the brake band's -0.10 coast: any deeper request lets
the PCM choose the friction brakes (PERMIT_BRAKING), and between about -0.15 and -0.35 it lights the
lamps on and off. The driver's own lift-off is -0.33 at speed with no brakes at all. With the toggle on,
the forked Toyota carcontroller (`opendbc/car/moonpilot/engine_coast.py`) holds PERMIT_BRAKING off while
the request stays at that engine coast, and the planner's brake band shifts its thresholds down by the
difference, so a lead law ask down to that depth is coasted on the engine. Off, or on any other car, the
planner is bit-identical to the default.
"""
from opendbc.car.moonpilot.engine_coast import ENGINE_COAST_MIN_SPEED, engine_coast_floor
from opendbc.car.toyota.values import ToyotaFlags


def engine_floor(CP, v_ego: float) -> float | None:
  """The coast this car makes without its brakes at `v_ego`, or None where the default applies."""
  if v_ego <= ENGINE_COAST_MIN_SPEED or CP.brand != "toyota" or not CP.flags & ToyotaFlags.MOONPILOT_ENGINE_COAST:
    return None
  return engine_coast_floor(v_ego)
