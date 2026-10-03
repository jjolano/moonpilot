"""Sound policy delegated from openpilot's soundd seam."""


def moonpilot_alert_volume(start_volume: float) -> float:
  """Keep warning-alert volume at its cabin-derived opening level."""
  return start_volume
