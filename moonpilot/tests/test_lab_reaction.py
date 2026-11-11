import unittest

import numpy as np

from openpilot.common.realtime import DT_MDL

from moonpilot.lab.reaction import BEFORE, _changed, lead_events


def lead_speed(accel, at=12.0, hold=2.0, seconds=30.0, noise=0.03, seed=0):
  """A lead holding 15 m/s that changes speed at `accel` from `at` for `hold` s, read with radar-sized speed noise."""
  t = np.arange(round(seconds / DT_MDL)) * DT_MDL
  v = 15.0 + accel * np.clip(t - at, 0.0, hold)
  return v + np.random.default_rng(seed).normal(0.0, noise, len(t))


class TestLeadEvents(unittest.TestCase):
  """Every time `lead-reaction` prints is measured from these onsets: a late or missed one moves the whole table."""

  def test_a_speed_change_is_found_at_its_onset_with_its_sign_and_size(self):
    for accel in (-1.0, 0.7):
      with self.subTest(accel=accel):
        events = lead_events(lead_speed(accel))
        self.assertEqual(len(events), 1)
        k, sign, size = events[0]
        self.assertLessEqual(abs(k * DT_MDL - 12.0), 0.15)
        self.assertEqual(sign, np.sign(accel))
        self.assertAlmostEqual(size, abs(accel), delta=0.15)

  def test_a_steady_lead_and_a_wander_under_the_threshold_are_not_events(self):
    self.assertEqual(lead_events(lead_speed(0.0)), [])
    self.assertEqual(lead_events(lead_speed(-0.2)), [])

  def test_a_change_without_five_seconds_after_it_is_left_out(self):
    """The response window must fit in the stretch, or a late event reads as 'never answered'."""
    self.assertEqual(lead_events(lead_speed(-1.0, at=26.5)), [])


class TestChanged(unittest.TestCase):
  def test_a_pedal_change_is_timed_from_the_onset_and_may_precede_it(self):
    state = np.zeros(400, dtype=bool)
    state[190:] = True  # half a second before an onset at frame 200
    self.assertAlmostEqual(_changed(state, 200), -0.5)

  def test_a_state_already_held_two_seconds_before_is_not_a_change(self):
    state = np.ones(400, dtype=bool)
    self.assertIsNone(_changed(state, 200))
    self.assertIsNone(_changed(np.zeros(400, dtype=bool), BEFORE - 1))  # no two seconds of stretch before it

  def test_a_state_never_reached_is_nan_not_a_time(self):
    self.assertTrue(np.isnan(_changed(np.zeros(400, dtype=bool), 200)))
