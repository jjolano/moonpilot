"""How upstream's acados longitudinal MPC (`long_mpc.py`) is solved: the seam in its `run()`.

Upstream takes one SQP_RTI step per frame and `reset()`s on any failure. Both were measured on upstream's maneuver
suite against a second solver iterated to convergence on the same inputs, and both change here:

- **RTI steps until the plan settles**, at most `MOONPILOT_MPC_MAX_STEPS`. One step is the converged plan to 0.001 m/s^2
  at p99 of frames, but not on a step input: the frame a stopped lead appears 60 m ahead at 20 m/s, the warm start is
  the no-lead plan and one step still rose to +1.2 m/s^2 where the converged plan brakes (4.7 m/s^2 apart over the
  plan). Full Gauss-Newton steps oscillate there before settling — 6.9 / 0.55 / 2.2 / 0.15 m/s^2 off after 2 / 3 / 4 / 6
  steps, 0.001 after 10 — so the cap is a bound, not the expected count: over the suite a frame took 2.03 steps on
  average, 3 at p99, 5 at most, and the accel the planner reads off the plan came within 0.005 m/s^2 of the converged
  plan's on every frame, against 0.52 for one step. Settling is judged step against step, so a settled frame costs two:
  acados time p50 0.34 -> 0.55 ms on a dev PC.
- **Why steps, not codegen'd SQP.** SQP capped at 3 iterations was tried: the prebuilt libacados prints
  `ocp_nlp_sqp: maximum iterations reached` to stdout on every capped frame, and with the NLP tolerances at their 1e-6
  default under a 1e-3 QP tolerance every frame is capped. Iterating here also leaves the generated solver alone.
- **A failed solve is retried once from a cold seed** — every node at the current state, the seed `set_cur_state`
  already uses after a speed jump — before upstream's reset, which zeroes the plan and `a_prev`: `get_accel_from_plan`
  read the failed frame as exactly 0 m/s^2 and `A_CHANGE_COST` then held the next plans near zero, so one forced failure
  mid-stop took the lead candidate from -1.95 to 0.00, and 0.45 s later it was back only to -1.34. `solver.reset()`
  clears the iterate alone; parameters, costs and the x0 bound are solver inputs and survive it.
- **A retry that fails too holds the last plan**, shifted one frame along its own horizon, for up to
  `MOONPILOT_MPC_HOLD_T`. Upstream's `run()` reads the plan and derives `a_prev` from whatever the solver holds, so writing
  the shifted plan back is what keeps both. Only a failure that outlasts the hold reaches upstream's reset, and it keeps
  reaching it until a solve succeeds: a plan that old is a guess about leads it has not seen.

Imported by `long_mpc.py`, never the other way round: that module loads the compiled solver.
"""

import numpy as np

from openpilot.common.swaglog import cloudlog

ACADOS_SUCCESS = 0  # acados/utils/types.h

MOONPILOT_MPC_MAX_STEPS = 10  # RTI steps per frame at most
MOONPILOT_MPC_STEP_TOL = 1e-2  # m/s and m/s^2: a step that moves no planned speed or accel more than this ends the frame
# ponytail: no recorded failure to size it on (none in upstream's maneuver suite); measure hold runs on device if they appear
MOONPILOT_MPC_HOLD_T = 0.5  # s a failing solver may run on the last plan before upstream's reset


def _plan(solver, n):
  return np.array([solver.get(i, 'x')[1:] for i in range(n + 1)])


def _iterate(solver, n):
  status = solver.solve()
  solve_time = float(solver.get_stats('time_tot')[0])
  prev = _plan(solver, n)
  for _ in range(MOONPILOT_MPC_MAX_STEPS - 1):
    if status != ACADOS_SUCCESS:
      break
    status = solver.solve()
    solve_time += float(solver.get_stats('time_tot')[0])
    plan = _plan(solver, n)
    if np.max(np.abs(plan - prev)) < MOONPILOT_MPC_STEP_TOL:
      break
    prev = plan
  return status, solve_time


class MoonpilotSolve:
  """One per `LongitudinalMpc`: the hold is a run of frames, so it is state."""

  def __init__(self, t_idxs, dt):
    self.t_idxs, self.dt = t_idxs, dt
    self.hold_frames = round(MOONPILOT_MPC_HOLD_T / dt)
    self.held = 0  # consecutive frames on a held plan

  def __call__(self, solver, x0, x_sol, u_sol):
    """`x_sol`/`u_sol` are the plan the MPC last used. Returns (status, acados time summed over every step taken); status
    is ACADOS_SUCCESS for a usable plan."""
    n = len(self.t_idxs) - 1
    status, solve_time = _iterate(solver, n)
    if status != ACADOS_SUCCESS:
      solver.reset()
      for i in range(n + 1):
        solver.set(i, 'x', x0)
      status, retry_time = _iterate(solver, n)
      solve_time += retry_time
    if status == ACADOS_SUCCESS:
      self.held = 0
    elif self.held < self.hold_frames:
      if self.held == 0:
        cloudlog.warning(f"Long mpc holding its last plan, solution_status: {status}")
      self.held += 1
      self._hold(solver, x_sol, u_sol, n)
      status = ACADOS_SUCCESS
    return status, solve_time

  def _hold(self, solver, x_sol, u_sol, n):
    t = self.t_idxs + self.dt
    x = np.column_stack([np.interp(t, self.t_idxs, x_sol[:, k]) for k in range(x_sol.shape[1])])
    x[:, 0] -= np.interp(self.dt, self.t_idxs, x_sol[:, 0])  # positions from where the ego is now
    u = np.interp(self.t_idxs[:-1] + self.dt, self.t_idxs[:-1], u_sol[:, 0])
    for i in range(n + 1):
      solver.set(i, 'x', x[i])
    for i in range(n):
      solver.set(i, 'u', u[i:i + 1])
