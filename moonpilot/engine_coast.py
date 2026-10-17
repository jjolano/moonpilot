"""Engine-braking coast, planner side (`MoonpilotEngineCoast`, off by default; a test toggle).

On this car the planner's lamp-free authority is the brake band's -0.10 coast: any deeper request lets
the PCM choose the friction brakes (PERMIT_BRAKING), and between about -0.15 and -0.35 it lights the
lamps on and off. The driver's own lift-off is -0.33 at speed with no brakes at all. With the toggle on,
the forked Toyota carcontroller (`opendbc/car/moonpilot/engine_coast.py`) holds PERMIT_BRAKING off while
the request stays at that engine coast, and the planner coasts to it: the brake band's thresholds shift
down by the difference, the follow cushion targets it, and a lead ask down to the approach decel is
raised to it whenever coasting there provably keeps the floor headway until speeds match
(`coast_budget`). Off, or on any other car, the planner is bit-identical to the default.

Measured on the PC, assuming the car meets the engine coast without brakes — which only a drive can
show: steady follows (114 recorded, flown) 3.7 -> 0.6 lamp presses a minute, mild lead slowdowns
(57 recorded) lamp-free in 22 rather than 3, with the lowest gap, TTC and headway of 137 recorded
slowdowns unchanged. The budget only pays at this depth: at the -0.10 coast it split presses
(91 -> 110 on the mild slowdowns), which is why the default mode does not use it.
"""
from opendbc.car.moonpilot.engine_coast import ENGINE_COAST_MIN_SPEED, engine_coast_floor
from opendbc.car.toyota.values import ToyotaFlags

MOONPILOT_ENGINE_COAST_LEAD_T = 1.0  # s a lead's current braking is assumed to hold in `coast_budget`


def engine_floor(CP, v_ego: float) -> float | None:
  """The coast this car makes without its brakes at `v_ego`, or None where the default applies."""
  if v_ego <= ENGINE_COAST_MIN_SPEED or CP.brand != "toyota" or not CP.flags & ToyotaFlags.MOONPILOT_ENGINE_COAST:
    return None
  return engine_coast_floor(v_ego)


def coast_margin(v_ego, gap, v_lead, a_lead, coast, headway_t, stop_distance) -> float:
  """Meters a coast at `coast` leaves above max(headway_t * v, stop_distance) once speeds match,
  with the lead's braking held for `MOONPILOT_ENGINE_COAST_LEAD_T` and constant after; negative
  means the floor is crossed. Closed form, two phases."""
  decel = -coast
  a_l = min(float(a_lead), 0.0)
  t1 = MOONPILOT_ENGINE_COAST_LEAD_T if v_lead + a_l * MOONPILOT_ENGINE_COAST_LEAD_T >= 0.0 else v_lead / -a_l
  ve1 = max(v_ego - decel * t1, 0.0)
  vl1 = max(v_lead + a_l * t1, 0.0)
  g1 = gap - ((v_ego - v_lead) * t1 + 0.5 * (-decel - a_l) * t1 * t1)
  margin = g1 - max(headway_t * ve1, stop_distance)
  closing = ve1 - vl1
  if closing <= 0.0:
    return margin
  return min(margin, g1 - closing * closing / (2.0 * decel) - max(headway_t * vl1, stop_distance))


def coast_budget(ask, v_ego, gap, v_lead, a_lead, coast, approach_decel, headway_t, stop_distance) -> float:
  """A lead ask between the approach decel and the coast, raised to the coast when it provably fits."""
  if -approach_decel <= ask < coast and coast_margin(v_ego, max(gap, 0.0), max(v_lead, 0.0), a_lead, coast, headway_t, stop_distance) >= 0.0:
    return coast
  return ask
