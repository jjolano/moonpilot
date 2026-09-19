---
name: moonpilot-device-deploy
description: Gets moonpilot onto the comma 3X and proves it landed — reaching the device, pushing the superproject and the forked submodules, updating or replacing /data/openpilot, and verifying tree, params, processes and remote access afterwards. Use when the user says deploy, push to the device, update the car, put this on the 3X, switch the device off another fork, or reinstall a wedged checkout; when ssh to the device stopped working after an IP or hostname change; or when they want the deployed state checked. Not for diagnosing a device that boots but misbehaves (moonpilot-device-debug), the PC build/test loop (moonpilot-develop), or drive logs and routes (moonpilot-log-analysis).
---

# Deploying to the device

## Reach it

```bash
ssh moonpilot            # tailnet 100.94.10.12 — verified, use this first
ssh moonpilot-home       # 10.0.1.205   } private LAN, DHCP: both are examples,
ssh moonpilot-hotspot    # 172.20.10.10 } not facts about where it is now
ssh moonpilot-mdns       # comma-487ad0.local, same LAN only
tailscale status | grep moonpilot   # shows the node's current direct path, i.e. its live LAN IP
```

The tailnet node is **`moonpilot`**; the device's own `hostname` is still `comma-487ad0`, so
anything addressing the tailnet as `comma-487ad0` is stale — the fork's tailscaled renamed it.
Only `ssh moonpilot` is verified from this container, which is on neither private LAN and has no
mDNS resolver: `moonpilot-hotspot` times out and `moonpilot-mdns` fails to resolve *here*, which
says nothing about a host that is on the LAN. Out of band: `docs/how-to/connect-to-comma.md` and
`tools/op.sh som-debug` both point at `panda/scripts/som_debug.sh`, which does not exist at this
panda pin (upstream deleted it in panda `8b5d328f`, "remove UART"). Treat the OBD-C serial console
as gone. With no network, what is left is ADB (enabled in device settings) or re-provisioning.

## Push the submodules too

The device resolves `.gitmodules`' relative URLs against its own origin and fetches the default
refspec, so a superproject pin whose commit was never pushed to the fork submodule's remote — or
sits on a side branch (AGENTS.md, *Forked opendbc and panda*) — makes `git submodule update` fail
on the device. This has bitten `opendbc_repo` once already.

```bash
for s in panda opendbc_repo; do
  git -C $s merge-base --is-ancestor "$(git -C $s rev-parse HEAD)" origin/master \
    && echo "$s pushed" || echo "$s UNPUSHED"
done
git -C opendbc_repo push origin master   # fix, then push the superproject
git push origin master
```

## Update an already-deployed device

```bash
SHA=$(git rev-parse HEAD)
ssh moonpilot "cd /data/openpilot && git fetch origin && git reset --hard $SHA \
  && git submodule sync --recursive && git submodule update --init --recursive && rm -f .overlay_init"
ssh moonpilot 'sudo reboot'
```

That is `op_switch` (`tools/op.sh:373-402`) minus the branch checkout and the `clean -df`, which
would delete `/data/openpilot` scratch you may still want. Three boot-time facts make it work:

- `.overlay_init` absent means the launcher skips overlay-update activation
  (`launch_chffrplus.sh:46`), so the updater cannot swap a staged tree over a hand-deployed one.
- `prebuilt` absent means the launcher runs `build.py` itself (`launch_chffrplus.sh:93`). Let it.
  A hand-run `build.py` dies `No module named 'opendbc'` without the package symlinks the launcher
  makes first (`launch_chffrplus.sh:77-81`).
- If the tree's `launch_env.sh` `AGNOS_VERSION` differs from `/VERSION`, launch flashes AGNOS and
  reboots (`launch_chffrplus.sh:20-30`) — several minutes and more than one reboot before the UI
  comes back. Not a hang.

## Replace the checkout (other fork, or wedged tree)

```bash
ssh moonpilot 'rm -rf /data/openpilot && git clone https://github.com/jjolano/moonpilot.git /data/openpilot \
  && cd /data/openpilot && git submodule update --init --recursive'
ssh moonpilot 'sudo reboot'
```

Clone over **https**, never the `git@github.com:` URL this PC pushes with: the device has no GitHub
SSH key, and the relative submodule URLs inherit the origin's scheme. Budget the space — the
checkout plus its build is 2.4 G today against 8.9 G free on `/data` (89 G total), and a fork whose
`launch_env.sh` pins a newer AGNOS than `/VERSION` spends the first boot flashing it.

## Verify

```bash
bash .agents/skills/moonpilot-device-deploy/scripts/device-status.sh [ssh-target]
```

Pass: `head` is the sha you deployed, tracking `origin/master`, `dirty 0`; every submodule at its
recorded pin with 0 dirty files; `agnos` equal to what `launch_env` wants; the process block present
at all (it needs a live `managerState`, so it proves the manager is up) with `ui` in the running
list. Read that block carefully: manager fills `running` and `shouldBeRunning` only for a process it
has already spawned (`openpilot/system/manager/process.py:121-129`), so a never-started row — every
`DaemonProcess`, e.g. `manage_athenad` — reports both false and is not a fault. `shouldBeRunning but
stopped` means the manager holds a process object that is not alive and will not be restarted before
a reboot (`start()` returns early while `self.proc` is set) — but that is also what a **deliberate**
exit looks like, and this device has one: `DisableUpdates=1` makes `updated` log and `exit(0)`
(`openpilot/system/updated/updated.py:418`), so its row sits there permanently and is not a
regression. Anything failing these is `moonpilot-device-debug`'s problem, not a redeploy.

## Keep remote access across the deploy

`MoonpilotTailscale` defaults **off** (`moonpilot/params_keys.h:16`), so deploying over an install
whose tailscale came from anywhere else silently ends remote access at the reboot. Before rebooting,
turn the param on and copy the outgoing daemon's state file to
`/data/moonpilot/tailscale/tailscaled.state` (root-owned, mode 600) — that file carries the node
identity, so the tailnet name and IP survive. The fork's supervisor installs its own pinned client
and restarts the daemon from there; the mechanism is `moonpilot/tailscale.py` and AGENTS.md,
*A binary the device doesn't ship*. `device-status.sh` prints the state file and
`MoonpilotTailscaleStatus`, which is how you confirm it reconnected.
