from openpilot.system.manager.process import DaemonProcess, NativeProcess, PythonProcess

from moonpilot import deps, features


def _only_onroad(started: bool, params, CP) -> bool:
  # Same predicate as process_config.only_onroad. Defined here rather than imported: that module
  # imports MOONPILOT_PROCS from this one, so importing back would be circular.
  return started


def _deps_missing(started: bool, params, CP) -> bool:
  # Onroad and offroad: a package can only be fetched where there is a network, which is usually
  # parked. Top-level `moonpilot.deps` import is fine — it is stdlib-only and imports nothing that
  # reaches back here, so neither the init-path rule nor _only_onroad's circularity applies.
  return len(deps.missing()) > 0


def _tailscale_wanted(started: bool, params, CP) -> bool:
  # Onroad and offroad: remote access is wanted while driving too, and the toggle defaults off.
  # Same top-level-import reasoning as _deps_missing, for `moonpilot.features`.
  return features.enabled(features.TAILSCALE, params)


# moonpilot processes, appended to upstream's procs in openpilot/system/manager/process_config.py.
# Constructors: NativeProcess(name, cwd, cmdline, should_run, enabled=True) /
#               PythonProcess(name, module, should_run, enabled=True) / DaemonProcess(name, module, param_name)
# Typed as the same union upstream's literal list infers, so `procs += MOONPILOT_PROCS` stays well-typed.
MOONPILOT_PROCS: list[DaemonProcess | NativeProcess | PythonProcess] = [
  # Normalizes the model's lead trajectories onto moonpilotState. Observation only, so no
  # config_realtime_process; see moonpilot/lead.py.
  PythonProcess("leadd", "moonpilot.leadd", _only_onroad),
  # Installs the external Python packages a feature declares, once the device has a network.
  # Stopped by the manager as soon as nothing is missing; see moonpilot/depsd.py.
  PythonProcess("depsd", "moonpilot.depsd", _deps_missing),
  # Installs and supervises tailscaled, and publishes MoonpilotTailscaleStatus for the panels.
  # Started and stopped by the toggle; see moonpilot/tailscaled.py.
  PythonProcess("tailscaled", "moonpilot.tailscaled", _tailscale_wanted),
]
