from typing import Any, cast

from openpilot.common.params import ParamKeyType, Params

from moonpilot.features import Feature

ABSENT = "moonpilot_no_such_package_anywhere"
PRESENT = "json"


def feature(requires=(), choices=(), car=None) -> Feature:
  return Feature(key="MoonpilotLeadLateral", title="test", description="test description",
                 requires=requires, choices=choices, car=car)

_DECLARED = Params()  # the compiled key table: types and declared defaults; never written


class FakeParams:
  """Duck-typed stand-in for Params, so the tests never touch the real param store.

  `on` answers BOOL rows only; overrides answer named keys, and writes take precedence. An undeclared key raises,
  as Params does. `puts` records writes."""

  def __init__(self, on=True, overrides=None):
    self._on = on
    # Off unless asked: on for every key would put every planner test in experimental mode, which swaps the
    # planner's whole candidate set rather than adding one feature.
    self._overrides = {"MoonpilotExperimentalMode": False, **dict(overrides or {})}
    self.puts: dict[str, Any] = {}
    self.values = self.puts

  def get(self, key, block=False, return_default=False):
    _DECLARED.check_key(key)
    if key in self.puts:
      return self.puts[key]
    if key in self._overrides:
      return self._overrides[key]
    if _DECLARED.get_type(key) == ParamKeyType.BOOL:
      return self._on
    return _DECLARED.get_default_value(key) if return_default else None

  def get_bool(self, key, block=False):
    value = self.get(key, block=block)
    return value == "1" if isinstance(value, str) else bool(value)

  def put(self, key, value, block=False):
    _DECLARED.check_key(key)
    self.puts[key] = value

  def put_bool(self, key, value, block=False):
    _DECLARED.check_key(key)
    self.puts[key] = "1" if value else "0"

  def remove(self, key, block=False):
    _DECLARED.check_key(key)
    self.puts.pop(key, None)
    self._overrides.pop(key, None)


def params(on=True, overrides=None) -> Params:
  return cast(Params, FakeParams(on, overrides))
