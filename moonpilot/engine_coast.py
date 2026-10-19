"""Engine-braking coast, planner side (`MoonpilotEngineCoast`, off by default; a test toggle).

On this car the planner's lamp-free authority is the brake band's -0.10 coast: any deeper request lets
the PCM choose the friction brakes (PERMIT_BRAKING), and between about -0.15 and -0.35 it lights the
lamps on and off. The driver's own lift-off is -0.33 at speed with no brakes at all. With the toggle on,
the forked Toyota carcontroller (`opendbc/car/moonpilot/engine_coast.py`) holds PERMIT_BRAKING off while
the request stays at that engine coast, and the planner coasts to it: the brake band's thresholds shift
down by the difference, the follow cushion targets it, a lead ask down to the approach decel is
raised to it whenever coasting there provably keeps the floor headway until speeds match
(`coast_budget`), and a far arrival holds speed until its ask passes 0.45 of it
(`MOONPILOT_ARRIVAL_HOLD_*` in `moonpilot/longitudinal.py`). Off, or on any other car, the planner is
bit-identical to the default.

Measured on the PC, assuming the car meets the engine coast without brakes — which only a drive can
show: steady follows (114 recorded, flown) 3.7 -> 0.3 lamp presses a minute, mild lead slowdowns
(57 recorded) lamp-free in 24 rather than 3, with the lowest gap, TTC and headway of 137 recorded
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


def _quad_min(c0, c1, c2, T) -> float:
  """Minimum of c0 + c1 t + c2 t^2 over [0, T]: the end points, and the vertex when it is inside."""
  m = min(c0, c0 + c1 * T + c2 * T * T)
  if c2 > 0.0 and 0.0 < -c1 / (2.0 * c2) < T:
    m = min(m, c0 - c1 * c1 / (4.0 * c2))
  return m


def coast_margin(v_ego, gap, v_lead, a_lead, coast, headway_t, stop_distance) -> float:
  """The least meters a coast at `coast` keeps above max(headway_t * v_ego, stop_distance) at any
  moment until speeds match — now included — with the lead's braking held for
  `MOONPILOT_ENGINE_COAST_LEAD_T` and constant after; negative means the floor is crossed. In each
  phase the gap is quadratic and the floor linear in time, so the minimum is exact."""
  decel = -coast
  a_l = min(float(a_lead), 0.0)
  t1 = MOONPILOT_ENGINE_COAST_LEAD_T if v_lead + a_l * MOONPILOT_ENGINE_COAST_LEAD_T >= 0.0 else v_lead / -a_l
  r0 = v_ego - v_lead
  # phase 1: gap(t) = gap - r0 t + (a_l + decel) t^2 / 2, ego speed v_ego - decel t
  c2 = 0.5 * (a_l + decel)
  margin = min(_quad_min(gap - headway_t * v_ego, -r0 + headway_t * decel, c2, t1), _quad_min(gap - stop_distance, -r0, c2, t1))
  g1 = gap - r0 * t1 + c2 * t1 * t1
  ve1 = v_ego - decel * t1
  closing = ve1 - max(v_lead + a_l * t1, 0.0)
  if closing <= 0.0:
    return margin
  # phase 2, lead at constant speed until the ego matches it: gap(s) = g1 - closing s + decel s^2 / 2
  t2 = closing / decel
  return min(margin, _quad_min(g1 - headway_t * ve1, -closing + headway_t * decel, 0.5 * decel, t2),
             _quad_min(g1 - stop_distance, -closing, 0.5 * decel, t2))


def coast_budget(ask, v_ego, gap, v_lead, a_lead, coast, approach_decel, headway_t, stop_distance) -> float:
  """A lead ask between the approach decel and the coast, raised to the coast when it provably fits."""
  if -approach_decel <= ask < coast and coast_margin(v_ego, max(gap, 0.0), max(v_lead, 0.0), a_lead, coast, headway_t, stop_distance) >= 0.0:
    return coast
  return ask
