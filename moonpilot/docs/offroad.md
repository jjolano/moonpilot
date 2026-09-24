# Offroad mode
`moonpilot/offroad.py` holds the device offroad while the car's ignition stays on, so the work that
is offroad-only — the settings rows gated on `ui_state.is_offroad()`, uploads, a software update, the
SSH and tailscale sessions — is reachable without turning the car off and waiting for the device to
power down. Upstream already has the whole decision in one place: `hardwared`'s `onroad_conditions`,
where every member is a reason the device *may* be onroad (`hardwared.py:206-210`) and
`should_start = all(onroad_conditions.values())` (`:386`) is what reads them, so the mechanism there
is one dict member and the line that refreshes it. Everything downstream follows
`deviceState.started` with no fork edit: the manager stops `card`, `controlsd`, `selfdrived` and
`loggerd`, `updated` starts, `IsOffroad` flips, and both UI trees switch to their offroad layout.

- **The mode is a param, not a process state.** `MoonpilotOffroad`, read with `return_default=True`
  (the fork's rule: `get_bool` ignores the declared default), declared
  `CLEAR_ON_MANAGER_START | CLEAR_ON_IGNITION_ON`. Both flags are cleared by upstream's manager, not
  by fork code — `manager_init()` for the first, and the ignition **rising** edge its 1 Hz loop sees
  for the second (`manager.py:30,33,135`) — which is what makes a reboot and a new ignition cycle the
  two automatic ends of the mode. The row deliberately does **not** carry
  `CLEAR_ON_OFFROAD_TRANSITION`: entering the mode *is* an offroad transition, so that flag would
  clear the request one frame after it was written. `moonpilot/tests/test_offroad.py` pins the flags.
- **The seam is the dict member and the refresh.** `onroad_conditions["moonpilot_onroad"]` is
  `onroad_condition(params)` — the inverse of the request — and the refresh sits above `ign_edge`, so
  a flip of the param is published on that tick rather than at the next 2 Hz one. No subscription is
  added: the entry gate is read in the UI, which is where the driver's hand is.
- **Two entries, one gate.** The moonpilot panel's row and a hold on the driving view both reach
  `can_enter` through their tree's own `parked()`: the car is on (`ui_state.started`, which is
  `deviceState.started` *and* the ignition), it is not being driven (`v_ego` under
  `MOONPILOT_OFFROAD_SPEED`, the fork's standstill convention, pinned equal to the planner's own two),
  and openpilot is not engaged. The row is a `button_item` in `moonpilot/ui/settings.py` and a pushed
  `BigButton` in `moonpilot/ui/settings_mici.py`; the gesture is tizi's
  `moonpilot/ui/offroad_mode.py` and mici's `moonpilot/ui/offroad_mode_mici.py`, each with its own
  dialog and its own widgets — self-contained per tree. The gesture
  offers **entering only**, so a hold can never take a device that is already offroad back onroad.
- **Leaving is never gated.** The mode's own state is `started` false, so a gate on the way out would
  strand the device on a screen with nothing to press: `enabled()` is `requested or parked`, and the
  row is live whenever the request is set.
- **The row is not in `FEATURES`.** It is not a swap between two behaviors but a device action with a
  live label — `TURN ON`, `TURN OFF`, `OFFROAD` (already offroad, not because of this mode) or
  `PARKED ONLY` (onroad and moving, or engaged) — and mici's value line has to be pushed from
  `_update_state` where tizi's row re-resolves its callables every render. That is why it sits beside
  the tailscale row in both panels rather than going through the feature table, and why the manager
  clearing the param under the panel's feet is visible at all.
- **The hold also toggles the sidebar in tizi**, which is upstream's own click handling and not
  something the fork can avoid: this view fires its click callback on the *press*
  (`_handle_mouse_press`), so holding the screen does both, with the confirm dialog over the sidebar
  afterwards. In mici the click fires on release and the base class clears the press tracking before
  a long press, so a hold does not also scroll to the home layout. While driving the gesture is inert
  — `parked()` is false — so a resting hand only does what it already does today.
- **What it costs on the road**, which is why the flags matter: while the mode is set there is no
  assist and no camera logging, and the car moving does not end it — the exits are the row, the
  ignition cycle and a reboot. A driver who parks, switches it on and then drives away gets a device
  that is watching nothing, and the recovery is the row, which is reachable from the offroad UI the
  mode produces. The manager will not power the device down while the ignition is on either
  (`power_monitor.should_shutdown` ends with `&= not ignition`), so the mode does not become a
  shutdown timer.
- **The panda is safe in this state with nothing fork-owned added**: `pandad` runs offroad by design
  and `panda_safety.configureSafetyMode(is_onroad=False)` puts it in NO_OUTPUT whenever
  `deviceState.started` is false, so a parked car with the ignition on and no onroad processes cannot
  be commanded.

`moonpilot/tests/test_offroad.py` pins the policy, the four labels, both directions of the gate, the
speed against the planner's own constants, and the two literals in `hardwared.py` — nothing in this
tree imports that file, so a merge that drops them deletes the feature silently.

