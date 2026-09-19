#!/usr/bin/env bash
# Read-only post-deploy report for a moonpilot device. One ssh round trip, writes nothing.
set -eu
exec ssh -o ConnectTimeout=15 "${1:-moonpilot}" bash -s <<'REMOTE'
set -u
OP=/data/openpilot
P=/data/params/d

echo "== device"
echo "hostname     $(hostname)"
echo "dongle id    $(cat $P/DongleId 2>/dev/null)"
echo "agnos        $(cat /VERSION 2>/dev/null)  (launch_env wants: $(sed -n 's/.*AGNOS_VERSION="\([^"]*\)".*/\1/p' $OP/launch_env.sh))"
echo "version      $(cat $P/Version 2>/dev/null)"

echo "== git"
cd "$OP" || exit 1
echo "head         $(git rev-parse HEAD)"
echo "tracking     $(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || echo '(none)') -> $(git rev-parse '@{u}' 2>/dev/null || echo -)"
echo "dirty        $(git status --porcelain | wc -l) file(s)"
echo "overlay_init $([ -e .overlay_init ] && echo present || echo absent)   prebuilt $([ -e prebuilt ] && echo present || echo absent)"
git submodule status --recursive | sed 's/^/  /'
echo "  -- dirty files per submodule"
git submodule foreach --recursive --quiet 'echo "  $displaypath $(git status --porcelain | wc -l)"'

echo "== panda"
sed -n 's/.*gitversion\[[0-9]*\] = "\(.*\)".*/firmware     \1/p' "$OP/panda/board/obj/gitversion.h" 2>/dev/null || echo "firmware     (not built)"

echo "== params"
for f in "$P"/Moonpilot*; do
  [ -e "$f" ] || continue
  printf '%-28s %s\n' "$(basename "$f")" "$(head -c 200 "$f" | tr -d '\0' | tr '\n' ' ')"
done

echo "== tailscale"
sudo -n ls -l /data/moonpilot/tailscale/tailscaled.state 2>&1 | sed 's/^/  /'

echo "== processes"
PYTHONPATH=$OP timeout 25 /usr/local/venv/bin/python3 - <<'PY' 2>&1 | sed 's/^/  /'
import openpilot.cereal.messaging as messaging
sm = messaging.SubMaster(["managerState"])
for _ in range(50):
  sm.update(200)
  if sm.updated["managerState"]:
    break
if not sm.alive["managerState"]:
  print("managerState not alive - manager is not running")
  raise SystemExit
procs = sm["managerState"].processes
run = sorted(p.name for p in procs if p.running)
want = sorted(p.name for p in procs if p.shouldBeRunning and not p.running)
extra = sorted(p.name for p in procs if p.running and not p.shouldBeRunning)
print(f"running {len(run)}/{len(procs)}: {' '.join(run)}")
print(f"shouldBeRunning but stopped: {' '.join(want) or 'none'}")
print(f"running but not wanted: {' '.join(extra) or 'none'}")
PY
REMOTE
