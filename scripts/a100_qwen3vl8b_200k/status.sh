#!/bin/bash
# Refresh $VELA_TASK/STATUS.md (contract section 34); with --watch, every 10 minutes until every row ended.
set -Eeuo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${VELA_TASK:?}" "${TP:?}"
while true; do
  "$TP" "$HERE/status.py" --task "$VELA_TASK" > /dev/null
  [ "${1:-}" = "--watch" ] || { cat "$VELA_TASK/STATUS.md"; exit 0; }
  ended=0
  for m in dflash dfly dflash2 dspark dflare domino jetspec; do
    grep -q "EXIT_CODE=" "$VELA_TASK/logs/$m/stdout.log" 2>/dev/null && ended=$((ended + 1))
  done
  [ "$ended" -eq 7 ] && exit 0
  sleep 600
done
