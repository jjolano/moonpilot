import unittest

import numpy as np

from moonpilot.lab.model_braking import episodes


class TestEpisodes(unittest.TestCase):
  def test_runs_within_merge_join_and_farther_runs_split(self):
    t = np.arange(40) * 0.05
    mask = np.zeros(40, dtype=bool)
    mask[2:5] = mask[10:12] = True  # 0.25 s apart: one episode
    mask[30:32] = True  # 0.9 s after the last masked frame: still within 1 s
    self.assertEqual(episodes(mask, t, np.zeros(40), merge_s=0.5), [slice(2, 12), slice(30, 32)])
    self.assertEqual(episodes(mask, t, np.zeros(40), merge_s=1.0), [slice(2, 32)])

  def test_never_spans_two_drives(self):
    t = np.arange(10) * 0.05
    mask = np.ones(10, dtype=bool)
    group = np.array([0] * 5 + [1] * 5)
    self.assertEqual(episodes(mask, t, group), [slice(0, 5), slice(5, 10)])


if __name__ == "__main__":
  unittest.main()
