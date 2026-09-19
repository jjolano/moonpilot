#!/usr/bin/env bash

recover() {
  local action="${1:-}"
  local staging_root="${2:-}"
  local dir="${3:-}"
  local old_openpilot="${staging_root}/old_openpilot"
  local failed_openpilot="${staging_root}/failed_openpilot"
  local swap_marker="${staging_root}/moonpilot_swap"
  local boot_ok="${staging_root}/moonpilot_boot_ok"
  local rollback="${staging_root}/moonpilot_rollback"
  local boot_id_path="${MOONPILOT_BOOT_ID_PATH:-/proc/sys/kernel/random/boot_id}"
  local current_boot_id swap_token boot_ok_token

  if [ "$action" != "recover" ] || [ -z "$staging_root" ] || [ -z "$dir" ]; then
    echo none
    return
  fi

  if [ ! -e "$dir" ] && [ -e "$old_openpilot" ]; then
    mv "$old_openpilot" "$dir"
    rm -f "$swap_marker"
    echo restart
    return
  fi

  if [ ! -r "$boot_id_path" ]; then
    echo none
    return
  fi
  current_boot_id="$(< "$boot_id_path")"
  if [ -z "$current_boot_id" ]; then
    echo none
    return
  fi

  if [ ! -e "$old_openpilot" ] || [ ! -e "$swap_marker" ]; then
    echo none
    return
  fi

  if [ ! -r "$swap_marker" ]; then
    echo none
    return
  fi
  swap_token="$(< "$swap_marker")"
  if [ -z "$swap_token" ]; then
    echo none
    return
  fi
  if [ "$swap_token" = "$current_boot_id" ]; then
    echo none
    return
  fi

  if [ -r "$boot_ok" ]; then
    boot_ok_token="$(< "$boot_ok")"
    if [ "$boot_ok_token" = "$swap_token" ]; then
      rm -f "$swap_marker"
      echo cleaned
      return
    fi
  fi

  rm -rf "$failed_openpilot"
  mv "$dir" "$failed_openpilot"
  mv "$old_openpilot" "$dir"
  printf 'update rolled back at %s: the new tree never reached a healthy manager\n' "$(date '+%Y-%m-%d %H:%M:%S')" > "$rollback"
  rm -f "$swap_marker"
  echo restart
}

recover "$@"
