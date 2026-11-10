import unittest
from types import SimpleNamespace

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS, LongitudinalMpc
from moonpilot.long_mpc import MOONPILOT_MPC_HOLD_T

ACADOS_QP_FAILURE = 4


def _radar(d_rel, v_lead, present=True):
  lead = SimpleNamespace(present=present, dRel=d_rel, vLead=v_lead, aLeadK=0.0, aLeadTau=1.5, modelProb=1.0)
  return SimpleNamespace(leadOne=lead, leadTwo=lead)


def _step(mpc, v, a, radar):
  mpc.set_weights()
  mpc.set_cur_state(v, a)
  mpc.update(radar)


class _FailNext:
  """The compiled solver, with its next `fail` solve() calls reported as QP failures after they ran."""

  def __init__(self, real):
    self.real, self.fail = real, 0

  def solve(self):
    status = self.real.solve()
    if self.fail > 0:
      self.fail -= 1
      return ACADOS_QP_FAILURE
    return status

  def __getattr__(self, name):
    return getattr(self.real, name)


def _fly_stop(test, fail_at, fails, check):
  """15 m/s onto a stopped lead 40 m out, the state advanced along a clean MPC's plan; a twin is fed the same state and
  fails `fails` solve() calls from frame `fail_at`. `check(frame, clean, faulted)` runs from that frame on."""
  clean, faulted = LongitudinalMpc(), LongitudinalMpc()
  faulted.solver = _FailNext(faulted.solver)
  v, a, gap = 15.0, 0.0, 40.0
  for frame in range(fail_at + 20):
    if frame == fail_at:
      faulted.solver.fail = fails
    for mpc in (clean, faulted):
      _step(mpc, v, a, _radar(gap, 0.0))
    if frame >= fail_at:
      test.assertLess(clean.a_solution[1], -1.0)  # still a real stop while it fails
      check(frame, clean, faulted)
    a_next = float(np.interp(DT_MDL, T_IDXS, clean.a_solution))
    v_next = v + DT_MDL * (a + a_next) / 2
    gap -= DT_MDL * (v + v_next) / 2
    v, a = v_next, a_next


class TestLongMpcSolve(unittest.TestCase):
  def test_one_failed_solve_keeps_the_braking_plan(self):
    """Upstream's reset zeroed the plan and a_prev on any failure: the lead candidate read 0 m/s^2 that frame and
    A_CHANGE_COST held the next plans near zero. One failed solve mid-stop must leave the plan where it was."""
    def check(frame, clean, faulted):
      np.testing.assert_allclose(faulted.a_solution, clean.a_solution, atol=0.05, err_msg=f"frame {frame}")
    _fly_stop(self, 30, 1, check)

  def test_a_failure_the_retry_cannot_clear_holds_the_last_plan_then_resets(self):
    """Every solve failing: the plan the car was following keeps braking through `MOONPILOT_MPC_HOLD_T`, and past it the
    MPC is upstream's reset, a zero plan, rather than a stale plan held indefinitely."""
    hold = round(MOONPILOT_MPC_HOLD_T / DT_MDL)

    def check(frame, clean, faulted):
      if frame - 30 < hold:
        self.assertLess(faulted.a_solution[1], -1.0, f"frame {frame}")
        if frame == 30:
          np.testing.assert_allclose(faulted.a_solution, clean.a_solution, atol=0.05)
      else:
        self.assertFalse(np.any(faulted.a_solution), f"frame {frame}")
    _fly_stop(self, 30, 10**6, check)

  def test_a_lead_that_appears_is_braked_for_in_its_first_frame(self):
    """SQP_RTI takes one step per frame from the last plan, which with no lead was accelerating: the frame a stopped
    lead appeared 60 m ahead at 20 m/s, its plan still rose to +1.2 m/s^2 over the first second where the converged
    plan brakes at -0.8 to -1.3. The frame a lead appears, the plan is the converged one."""
    mpc = LongitudinalMpc()
    for _ in range(40):
      _step(mpc, 20.0, 0.0, _radar(200.0, 0.0, present=False))
    _step(mpc, 20.0, 0.0, _radar(60.0, 0.0))
    first = mpc.a_solution.copy()
    for _ in range(60):
      mpc.run()
    self.assertLess(mpc.a_solution[1], -0.5)  # the converged plan brakes for it
    np.testing.assert_allclose(first, mpc.a_solution, atol=0.05)


if __name__ == "__main__":
  unittest.main()
