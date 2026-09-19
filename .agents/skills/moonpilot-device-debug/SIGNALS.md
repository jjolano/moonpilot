# Reading device signals and logs

Companion to `SKILL.md`. Load this once you know *what* to look at — process state and feature
gating live in the skill; the wire format and the log layout live here.

## Reading cereal on the device

```bash
PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 -c '
from openpilot.cereal import messaging
sm = messaging.SubMaster(["pandaStates","selfdriveState","onroadEvents","moonpilotState"])
sm.update(2000)
print("alive", sm.alive); print("valid", sm.valid); print("all_checks", sm.all_checks())
for ps in sm["pandaStates"]:
  print(ps.pandaType, "safety", ps.safetyModel, "ignition", ps.ignitionLine,
        "ca", ps.controlsAllowed, "caLateral", ps.controlsAllowedLateral)
print("selfdriveState", sm["selfdriveState"].state, sm["selfdriveState"].enabled,
      repr(sm["selfdriveState"].alertText1))
print("onroadEvents", [e.name for e in sm["onroadEvents"]])'
```

`sm.alive` is "a publisher is sending"; a message's own `valid` is the publisher's verdict on its
own inputs. They fail independently - a daemon that died leaves a fresh-looking `valid=True` message
in the socket, which is why readers check both. Offroad only a handful of services publish, so
`alive=False` for `selfdriveState`, `onroadEvents` and `moonpilotState` is expected there, and the
panda reports `safetyModel noOutput` until the car is awake.

msgq allows **one publisher per service**. A second `PubMaster` on the same service takes the write
uid, prints `Killing old publisher: <service>`, and the original's next `send` raises
`MultiplePublishersError` - which kills that process, and by the rule in `SKILL.md` the manager will
not restart it. A whole subsystem disappearing after a "small" new publisher is this.

## What do the logs say?

```bash
L=$(ls -t /data/log/swaglog.* | head -40 | tac)
jq -rc 'select(.levelnum>=30) | [(.created|floor|strflocaltime("%m-%d %H:%M:%S")), .ctx.daemon,
        (.["msg$s"] // (.msg|tostring))] | @tsv' $L | tail -40
```

The string payload is `msg$s` and the structured one is `msg`, so a filter must take both. Swap the
`select` for `.ctx.daemon=="selfdrived"` to follow one process, or pull the structured events that
name failing services directly:

```bash
jq -c 'select((.msg|type)=="object")
       | select(.msg["event$s"]|test("initialized|commIssue|not_running")) | .msg' $L | tail -5
```

`selfdrived.initialized` carries `invalid`, `not_alive` and `not_freq_ok` arrays - the exact service
names behind a communication alert. `/data/log` holds thousands of rotated files, so always slice
with `ls -t`. `/data/manager.log` is **not** the live log: it is append-only across installs and on
this device is months stale from a previous fork (`stat -c %y /data/manager.log`) - never quote a
traceback from it without checking that date.
