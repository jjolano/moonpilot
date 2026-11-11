def moonpilot_state(sm):
  """The published `moonpilotState`, or None unless it is both valid and alive: `alive` is what covers a publisher that
  died with a fresh-looking message left in the socket.

  getattr, not `sm.valid`: upstream's longitudinal maneuver plant hands the planner a plain dict as `sm`."""
  if getattr(sm, "valid", {}).get("moonpilotState", False) and getattr(sm, "alive", {}).get("moonpilotState", False):
    return sm["moonpilotState"]
  return None
