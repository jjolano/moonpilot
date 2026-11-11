"""Retry delay shared by supervised installers."""


class Backoff:
  def __init__(self, start: float = 30.0, maximum: float = 1800.0):
    self.start = start
    self.maximum = maximum
    self.current = start

  def reset(self) -> None:
    self.current = self.start

  def next(self) -> float:
    delay = self.current
    self.current = min(self.current * 2, self.maximum)
    return delay
