from openpilot.system.manager.process import DaemonProcess, NativeProcess, PythonProcess


def _only_onroad(started: bool, params, CP) -> bool:
  # Same predicate as process_config.only_onroad. Defined here rather than imported: that module
  # imports MOONPILOT_PROCS from this one, so importing back would be circular.
  return started

# moonpilot processes, appended to upstream's procs in openpilot/system/manager/process_config.py.
# Constructors: NativeProcess(name, cwd, cmdline, should_run, enabled=True) /
#               PythonProcess(name, module, should_run, enabled=True) / DaemonProcess(name, module, param_name)
# Typed as the same union upstream's literal list infers, so `procs += MOONPILOT_PROCS` stays well-typed.
MOONPILOT_PROCS: list[DaemonProcess | NativeProcess | PythonProcess] = [
  # Normalizes the model's lead trajectories onto moonpilotState. Observation only, so no
  # config_realtime_process; see moonpilot/lead.py.
  PythonProcess("leadd", "moonpilot.leadd", _only_onroad),
]
