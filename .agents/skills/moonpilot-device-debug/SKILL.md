---
name: moonpilot-device-debug
description: Diagnoses a live moonpilot comma device over SSH - which manager processes are up and why, why a fork toggle appears inert, whether the flashed panda firmware is the fork's, and what swaglog says. Use when the user reports a moonpilot feature "does nothing" or "isn't working", a process missing or dying, half-engagement or steering not engaging, a "Communication Issue Between Processes" / "Process Not Running" alert, or asks to check panda health, msgq, params or device logs on the car.
---

# Debugging a running moonpilot device

Every command below runs **inside** `ssh moonpilot`. Read-only: nothing here writes a param, stops
a process, or reboots. Deploying is `moonpilot-device-deploy`; the PC loop is `moonpilot-develop`;
route logs are `moonpilot-log-analysis`. Cereal field reads, the `alive`-vs-`valid` distinction, the
msgq one-publisher rule and the swaglog/jq recipes are in [SIGNALS.md](SIGNALS.md).

## Start from the snapshot

From the repo on the PC, one round trip for identity, version, HEAD, submodules, params and a
process summary:

```bash
bash .agents/skills/moonpilot-device-deploy/scripts/device-status.sh
```

Everything below answers a question that snapshot leaves open.

## Is a process up, and should it be?

```bash
PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c '
from openpilot.cereal import messaging
sm = messaging.SubMaster(["managerState"])
for _ in range(10):
  sm.update(1000)
  if sm.updated["managerState"]: break
else: raise SystemExit("managerState never arrived - the manager is not running")
for p in sm["managerState"].processes:
  print(f"{p.name:22} running={p.running} shouldBeRunning={p.shouldBeRunning} exitCode={p.exitCode}")'
```

- `running=False shouldBeRunning=False` is **normal**: gated off right now. The predicates are in
  `openpilot/system/manager/process_config.py` and, for fork processes, `moonpilot/procs.py` -
  `updated` is `only_offroad`, the control stack `only_onroad`, `depsd` only while a declared
  dependency is missing.
- `shouldBeRunning` is `self.proc is not None and not self.shutting_down`
  (`openpilot/system/manager/process.py:126`) - it reports that the manager holds a process object,
  *not* what the predicate says. `True` with `running=False` means the child exited and nothing will
  bring it back before a reboot: `start()` returns early while `self.proc` is set
  (`process.py:142-147,166-171`), and only the `should_run`-false path clears it.
- **But a deliberate exit looks identical**, and this device has one. `updated` sits at
  `shouldBeRunning=True, running=False` permanently because `DisableUpdates=1` makes it log and
  `exit(0)` (`openpilot/system/updated/updated.py:418`). Check that param before calling a stopped
  `updated` a fault; for any other process, go to its swaglog lines (`SIGNALS.md`).
- `exitCode` is `self.proc.exitcode or 0`, so `0` also means "no exit code yet". Judge by `running`.

## Why is a fork feature doing nothing?

Take these in order and stop at the first that answers.

```bash
PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c '
from openpilot.common.params import Params
from moonpilot import features
p = Params()
for f in features.FEATURES:
  print(f"{f.key:24} declared={p.get(f.key, return_default=True)!s:5} get_bool={p.get_bool(f.key)!s:5}"
        f" available={features.available(f)!s:5} enabled={features.enabled(f, p)}")'
```

1. `declared` disagreeing with `get_bool` means the param row was never seeded - the read in the
   code must be `params.get(key, return_default=True)`. See AGENTS.md, **Features**.
2. `available=False` means a module in the feature's `requires` is not importable yet; that is
   `depsd`'s job, so check its row in the process table above and its swaglog lines.
3. The car may not be able to run it at all:

```bash
PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c '
from opendbc.car.structs import car
from openpilot.common.params import Params
raw = Params().get("CarParams")
if raw is None: raise SystemExit("CarParams unset - no car has been seen since boot")
with car.CarParams.from_bytes(raw) as CP:
  print(CP.carFingerprint, "| pcmCruise", CP.pcmCruise, "| opLong", CP.openpilotLongitudinalControl,
        "| lat", CP.lateralTuning.which(), "| passive", CP.passive)
  for c in CP.safetyConfigs: print("  safety", c.safetyModel, hex(c.safetyParam))'
```

   `lat` other than `torque` rules out the fork steering controller; `opLong False` rules out the
   fork longitudinal; lateral engagement additionally needs `pcmCruise` — the car's own ACC
   owning speed — and not `passive`. It is not brand-limited any more: where a mode decodes no
   cruise main switch the panda arms on openpilot's own engaged heartbeat, so a car outside
   Toyota/Honda/Volkswagen is in scope and a missing `controlsAllowedLateral` there means the
   heartbeat, not the car, is the thing to look at.
   AGENTS.md, **The lateral controller** and **Lateral-only engagement**, hold the exact ceilings.
   `safetyParam` is a bitfield: `0x1049` on this RAV4 is the Toyota TSS2 flag plus the fork's
   `LATERAL_ENGAGE` (`16 << 8`), so the fork's safety param is present on the car.
4. If the param looks right and the car qualifies, the behaviour is picked once at construction,
   so the toggle needs a restart. Flipping it onroad changes nothing.

## Is the panda side live?

```bash
PYTHONPATH=/data/openpilot:/data/openpilot/panda /usr/local/venv/bin/python3 -c '
import subprocess
from panda import Panda
fw = Panda().get_version()
pin = subprocess.check_output(["git","-C","/data/openpilot","rev-parse","HEAD:panda"], text=True).strip()
print("flashed", fw, "| pin", pin[:8], "|",
      "MATCH" if fw.startswith("DEV-") and pin.startswith(fw.split("-")[1]) else "MISMATCH")'
```

A fork build reports `DEV-<short-sha>-DEBUG`, the sha being panda's own HEAD when the firmware was
built. A sha that is not the superproject's pin means the board is running someone else's safety
layer: `controls_allowed_lateral` is never set, so `controlsAllowedLateral` stays false and
half-engagement is inert no matter what the params say. (The status script reads
`panda/board/obj/gitversion.h` - what was *built*; this reads what is *flashed*.)

## Do not

- Do not reboot or restart the manager to "see if it helps" without asking. The fault state - which
  children died, what the panda thinks - is gone the moment you do.
- Do not write params while investigating a symptom. It changes the thing you are measuring, and
  most fork behaviour is latched at construction anyway.
- Do not kill a manager child. Nothing brings it back before the next boot.
- Do not touch `/data/continue.sh` or `/data/moonpilot/ssh/` - they carry the boot-time SSH key
  restore, and losing that loses the device.
