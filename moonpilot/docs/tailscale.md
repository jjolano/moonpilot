# A binary the device doesn't ship
Some capabilities are native binaries rather than Python packages; tailscale is the one so far, in `moonpilot/tailscale.py`. `moonpilot/fetch.py` owns the shared download and extraction path.

The supervisor checks the stable track every 6 hours while online and offroad; it installs and restarts on a newer version, leaving the current daemon running if an upgrade fails.

The state file follows the platform contract: on the current device, `tailscale.state_path()` uses `<data_root>/tailscale/tailscaled.state` (`/data/moonpilot/tailscale/tailscaled.state`) for node identity, login session and prefs. This deliberately trades factory-reset survival for a canonical data-root path, so a factory reset may log the node out; no `/persist` migration or persistence error is involved on device. On PC, the existing `persist_root()` migration behavior remains. The binaries, control socket and `--statedir` stay under `data_root()`.

Device processes are not root, so a privileged child goes through `sudo -n` — `tailscale.sudo()` prefixes `daemon_args()`, `cli_args()`, `up_args()` and the kill in the supervisor. The supervisor kills its child on exit and scopes that kill to its own `--socket` path rather than the binary, because `binaries()` prefers a pair the OS ships and the binary path would take a distro's daemon down with ours. For the reason the manager does not restart an exited child, see `moonpilot/procs.py`. For the tailscaled `--statedir` requirement, see `moonpilot/tailscale.py`.

Daemon → UI state travels through one `CLEAR_ON_MANAGER_START` string param, `MoonpilotTailscaleStatus`, decoded by `moonpilot/tailscale.py`, so neither panel ever shells out and a reboot cannot leave a stale `running 100.x` on screen. Both trees' QR sign-in dialogs — `moonpilot/ui/tailscale_qr.py` and `moonpilot/ui/tailscale_qr_mici.py` — are fork-owned on purpose and share the texture handling in `tailscale_qr.py`'s `QrTexture`: they import `make_texture` from upstream's qrcode module but subclass no upstream dialog, because a silently-unused override of a renamed upstream method is exactly the failure a merge cannot see.

