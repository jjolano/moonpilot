from typing import Any, cast

from openpilot.common.params import Params


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store.

  `on` is the answer for every key, which is what keeps a test that only cares about one feature
  readable; `overrides` names a single key when a test needs one feature off and the rest on. `puts`
  records what the code under test persisted."""

  def __init__(self, on=True, overrides=None):
    self._on = on
    self._overrides = dict(overrides or {})
    self.puts: dict[str, Any] = {}

  def get(self, key, return_default=False):
    return self._overrides.get(key, self._on)

  def put(self, key, value, block=False):
    self.puts[key] = value


def _params(on=True, overrides=None) -> Params:
  return cast(Params, FakeParams(on, overrides))
