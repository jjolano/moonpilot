from openpilot.system.manager.process import DaemonProcess, NativeProcess, PythonProcess

from moonpilot import deps, features, modelstatus


def _only_onroad(started: bool, params, CP) -> bool:
  # Same predicate as process_config.only_onroad, and local for the same reason every predicate here is:
  # `process_config` imports this table, so importing back would be circular. `ensure_running` does not
  # restart an exited child while `should_run` remains true; the manager retries only after a later evaluation.
  return started


def _deps_missing(started: bool, params, CP) -> bool:
  # Onroad and offroad: a package can only be fetched where there is a network, which is usually parked.
  return len(deps.missing()) > 0


def _tailscale_wanted(started: bool, params, CP) -> bool:
  # Onroad and offroad: remote access is wanted while driving too, and the toggle defaults off.
  return features.enabled(features.TAILSCALE, params)



# moonpilot processes, merged into upstream's procs by moonpilot_procs in openpilot/system/manager/process_config.py.
MOONPILOT_PROCS: list[DaemonProcess | NativeProcess | PythonProcess] = [
  # One publisher carries both lead trajectories and the rolling-window ego correction.
  PythonProcess("leadd", "moonpilot.leadd", _only_onroad),
  # Installs the external Python packages a feature declares, once the device has a network.
  PythonProcess("depsd", "moonpilot.depsd", _deps_missing),
  # Installs and supervises tailscaled, publishing MoonpilotTailscaleStatus for the panels.
  PythonProcess("tailscaled", "moonpilot.tailscaled", _tailscale_wanted),
  # The marketplace worker fetches and compiles requests, then publishes MoonpilotModelsStatus.
  PythonProcess("modelsd", "moonpilot.modelsd", lambda _started, params, _CP: modelstatus.worker_wanted(params)),
]


def moonpilot_procs(procs: list[DaemonProcess | NativeProcess | PythonProcess]) -> list[DaemonProcess | NativeProcess | PythonProcess]:
  """Upstream's process table with the fork's: lagd's row runs the fork's estimator under upstream's name and
  predicate, so lateralDelay keeps one publisher and every consumer reads it unchanged (moonpilot/lagd.py);
  MOONPILOT_PROCS is appended."""
  return [PythonProcess("lagd", "moonpilot.lagd", p.should_run) if p.name == "lagd" and isinstance(p, PythonProcess) else p
          for p in procs] + MOONPILOT_PROCS
