"""Shared status-param line encoding."""


def put(params, key: str, state: str, detail: str = "") -> None:
  value = f"{state} {detail}".strip()
  if params.get(key) != value:
    params.put(key, value)


def read(params, key: str) -> tuple[str, str]:
  state, _, detail = (params.get(key) or "").partition(" ")
  return state, detail
