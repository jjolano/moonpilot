"""What the fork's learned values share: how often they persist, and how a block history resumes.

Two estimators keep their evidence in upstream's `BlockAverage` ring (`moonpilot/latency.py`'s longitudinal lag and
`moonpilot/lagd.py`'s lateral one). `BlockAverage.__init__` would tile one value into every slot, which flattens the
history each boot and zeroes the spread `MAX_LAG_STD` checks, so both resume the completed blocks themselves.
"""

from openpilot.selfdrive.locationd.lagd import BlockAverage

PERSIST_EVERY = 1200  # frames between param writes


def completed_blocks(avg: BlockAverage) -> list[float]:
  """The finished blocks, oldest first: exactly the ones `BlockAverage.get` averages. Once the ring has
  wrapped that is every slot but the one being written."""
  if avg.valid_blocks < avg.num_blocks:
    order = range(avg.valid_blocks)
  else:
    order = [(avg.block_idx + i) % avg.num_blocks for i in range(1, avg.num_blocks)]
  return [float(avg.values[i, 0]) for i in order]


def restore_blocks(avg: BlockAverage, blocks: list[float]) -> None:
  """Resume `completed_blocks` exactly: the next block goes after the newest and, once the ring is full,
  retires the oldest."""
  avg.values[:len(blocks), 0] = blocks
  avg.valid_blocks = len(blocks)
  avg.block_idx = len(blocks) % avg.num_blocks
  avg.idx = 0
