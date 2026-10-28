import unittest

import numpy as np

from moonpilot.lab.bursts import _transition

N, LEAD_IN, K = 100, 40, 300  # a burst second, the delay lead-in and the burst start, in 100 Hz frames


def _chain(active_from=None, active_until=None, gas_until=None):
  """A chain of 600 frames, active from `active_from` (frame) to `active_until`, the gas pressed until `gas_until`."""
  n = 600
  active, gas = np.ones(n), np.zeros(n)
  if active_from is not None:
    active[:active_from] = 0
  if active_until is not None:
    active[active_until:] = 0
  if gas_until is not None:
    gas[:gas_until] = 1
  return {"long_active": active, "gas": gas, "brake": np.zeros(n)}


class TestControlTransition(unittest.TestCase):
  """`burst-chain` names a pedal hand-back only for a real inactive-to-active edge before the burst with a pedal just
  before it; any other inactive frame in the window is a transition of some other kind, never the pedal's."""

  def test_releasing_the_gas_just_before_the_burst_is_a_hand_back(self):
    self.assertEqual(_transition(_chain(active_from=K - 20, gas_until=K - 20), K, LEAD_IN, N), 2)

  def test_an_engagement_with_no_pedal_is_another_transition(self):
    self.assertEqual(_transition(_chain(active_from=K - 20), K, LEAD_IN, N), 1)

  def test_a_disengagement_after_the_burst_is_another_transition(self):
    self.assertEqual(_transition(_chain(active_until=K + 50, gas_until=10), K, LEAD_IN, N), 1)

  def test_active_throughout_is_no_transition(self):
    self.assertEqual(_transition(_chain(gas_until=10), K, LEAD_IN, N), 0)


if __name__ == "__main__":
  unittest.main()
