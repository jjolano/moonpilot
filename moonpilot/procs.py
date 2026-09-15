from openpilot.system.manager.process import DaemonProcess, NativeProcess, PythonProcess

# moonpilot processes, appended to upstream's procs in openpilot/system/manager/process_config.py.
# Constructors: NativeProcess(name, cwd, cmdline, should_run, enabled=True) /
#               PythonProcess(name, module, should_run, enabled=True) / DaemonProcess(name, module, param_name)
# Typed as the same union upstream's literal list infers, so `procs += MOONPILOT_PROCS` stays well-typed.
MOONPILOT_PROCS: list[DaemonProcess | NativeProcess | PythonProcess] = []
