"""The localization-validity ride-through: a transient dropout is ridden, a sustained one is not.

The failure this exists for is measured, not hypothetical. On route 000003f8 one invalid
`cameraOdometry` message inside a camera-stream desync made locationd publish `inputsOK=False` for
exactly one 20 Hz frame -- 59 ms, 1 of 78,029 deviceMotion samples in the drive -- and upstream's
`locationdTemporaryError` soft-disabled openpilot off that single frame.

Both directions are pinned here, because the whole feature is the claim that a bounded window can
be taken out of a safety gate and nothing else: ridden for a transient, still raised for anything
that outlives the window, and the latch never starved by a flapping signal.

The window edge is asserted to the frame, not to an exact index. A monotonic clock in the thousands
of seconds cannot represent 1000.3 exactly, so the elapsed time read at the nominal boundary is
0.29999999999995 and the comparison closes one frame later -- 46 ns against a 0.3 s window, below
anything the fork can act on and not worth pinning as an exact index.
"""

import unittest

from moonpilot import locationd
from moonpilot.locationd import MOONPILOT_LOCATION_RIDE_THROUGH_S, moonpilot_locationd_ok

DT = 0.01  # selfdrived runs at 100 Hz
WINDOW_FRAMES = 30  # the 0.3 s ride-through; fixed to bound test work if the production value regresses
START = 1000.0


class Frames:
  """A continuous frame clock, so successive `feed` calls keep advancing the way selfdrived's own
  loop does -- a scenario is one uninterrupted run, not a sequence that restarts at zero.

  True in the returned list means "do not raise the event", which is what keeps
  `locationdTemporaryError` off the bus.
  """

  def __init__(self) -> None:
    self.frame: int = 0

  @property
  def now(self) -> float:
    return START + self.frame * DT

  def feed(self, samples: list[bool]) -> list[bool]:
    out: list[bool] = []
    for ok in samples:
      out.append(moonpilot_locationd_ok(ok, self.now))
      self.frame += 1
    return out


def _first_raise(decisions: list[bool]) -> int:
  for i, keep_quiet in enumerate(decisions):
    if not keep_quiet:
      return i
  raise AssertionError("the window never closed")


class TestLocationdRideThrough(unittest.TestCase):
  def setUp(self):
    locationd.reset()

  def assert_window_edge(self, decisions: list[bool], expected: int, msg: str = "") -> int:
    """The window closes at its own length to within a frame, and stays closed after it.

    `expected` is the nominal edge in this list's own indices: for a run of consecutive false
    samples that is `WINDOW_FRAMES`, and for a bad sample every other frame it is half that.
    """
    i = _first_raise(decisions)
    self.assertGreaterEqual(i, expected, f"{msg}first raise at {i}, expected at least {expected}")
    self.assertLessEqual(i, expected + 1, f"{msg}first raise at {i}, expected at most {expected + 1}")
    self.assertTrue(all(decisions[:i]), f"{msg}everything before the edge must be ridden")
    self.assertTrue(not any(decisions[i:]), f"{msg}it must keep raising")
    return i

  def test_window_length_matches_production_duration(self):
    self.assertAlmostEqual(MOONPILOT_LOCATION_RIDE_THROUGH_S, WINDOW_FRAMES * DT, places=9)

  def test_a_single_false_sample_raises_nothing(self):
    # The measured drive: deviceMotion publishes at 20 Hz, so one bad sample stays visible to
    # selfdrived's 100 Hz loop for five frames before the good one replaces it.
    f = Frames()
    held = f.feed([False] * 5)
    self.assertTrue(all(held), held)
    self.assertTrue(all(f.feed([True] * 20)))

  def test_a_dropout_inside_the_window_is_ridden(self):
    f = Frames()
    ridden = f.feed([False] * (WINDOW_FRAMES - 2))
    self.assertTrue(all(ridden), "a sub-window dropout must be ridden")
    self.assertTrue(all(f.feed([True] * 10)))

  def test_a_sustained_dropout_still_raises(self):
    # The safety direction. Past the window the seam hands upstream's answer back, so the event
    # raises at the window edge and keeps raising for as long as localization stays bad.
    f = Frames()
    self.assert_window_edge(f.feed([False] * (WINDOW_FRAMES + 50)), WINDOW_FRAMES, "sustained: ")

  def test_the_window_does_not_rearm_while_inputs_stay_bad(self):
    # A gate that reopened on a good frame inside a long failure would let it through, so every
    # frame from the edge on is a raise and there is no second quiet stretch anywhere.
    f = Frames()
    decisions = f.feed([False] * (WINDOW_FRAMES * 3))
    raised = [i for i, d in enumerate(decisions) if not d]
    self.assert_window_edge(decisions, WINDOW_FRAMES)
    self.assertEqual(len(raised), len(decisions) - raised[0], "no quiet stretch may reopen")

  def test_recovery_for_a_full_window_rearms_the_ride_through(self):
    f = Frames()
    f.feed([False] * 3)
    f.feed([True] * (WINDOW_FRAMES + 5))  # a clean stretch as long as the window clears the latch
    later = f.feed([False] * 5)
    self.assertTrue(all(later), "a later transient must be ridden again")

  def test_a_short_recovery_does_not_rearm(self):
    # Recovery shorter than the window is noise on a flapping signal, not a clean stretch, so the
    # original false run keeps ageing and the event still arrives inside this window.
    f = Frames()
    f.feed([False] * 3)
    f.feed([True] * (WINDOW_FRAMES - 5))
    self.assertTrue(any(f.feed([False] * WINDOW_FRAMES)), "the first run must not be restarted")

  def test_a_flapping_signal_is_not_starved(self):
    # A bad sample every other frame never accumulates the clean stretch that closes the latch, so
    # the false run keeps ageing and the event fires instead of being held off forever. Good frames
    # never raise on their own -- upstream raises on a false read, and a good read is upstream's
    # answer whatever the latch is doing.
    f = Frames()
    pairs = WINDOW_FRAMES + 10
    decisions = f.feed([ok for _ in range(pairs) for ok in (False, True)])
    self.assert_window_edge(decisions[0::2], WINDOW_FRAMES // 2, "flapping: ")
    self.assertTrue(all(decisions[1::2]), "a good read is never itself a reason to raise")

  def test_good_inputs_are_upstreams_answer(self):
    # With no dropout in progress the seam is upstream's predicate verbatim: good in, True out.
    f = Frames()
    self.assertTrue(all(f.feed([True] * 20)))
    self.assertTrue(moonpilot_locationd_ok(True, 5000.0))

  def test_reset_clears_the_latch(self):
    f = Frames()
    f.feed([False] * (WINDOW_FRAMES + 1))
    locationd.reset()
    self.assertTrue(all(f.feed([False] * 5)), "after a reset the next dropout starts a new window")


if __name__ == "__main__":
  unittest.main()
