#!/bin/bash
# Formal launcher of the a100_qwen3vl8b_200k_s42 matrix (contract sections 4, 25, 24.5).
#
#   bash scripts/a100_qwen3vl8b_200k/run_all.sh 2>&1 | tee $VELA_TASK/logs/launcher.$(date +%Y%m%d_%H%M%S).log
#
# Discovers this node's visible A100-class GPUs (VELA_GPUS restricts them on a
# shared node, e.g. VELA_GPUS=6,7), forms split-device worker slots of two GPUs
# each (max workers = floor(n / 2)) and fills free slots in the frozen queue
# order DFlash -> DFly -> DFlash2 -> DSpark -> DFlare -> Domino -> JetSpec.
# A slot is free when neither GPU's physical-UUID lock exists. Each method runs
# as its own run_one.sh (own PID, run dir, logs); one method failing never stops
# the others. Launchers on several nodes may share the queue: a row counts as
# running while its shared-disk row lock has a fresh heartbeat. Re-running the
# launcher resumes unfinished rows from their durable checkpoints and skips
# rows whose log ends with EXIT_CODE=0 (done) or EXIT_CODE=1 (blocked; append
# a line or delete its stdout.log to retry). Exits when no row is pending or
# running.
set -Eeuo pipefail
trap 'echo "[$(date "+%F %T")] launcher abort line=$LINENO cmd=$BASH_COMMAND exit=$?"' ERR
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}"
QUEUE=(dflash dfly dflash2 dspark dflare domino jetspec)
say() { echo "[$(date '+%F %T')] $*"; }

mapfile -t VISIBLE < <(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '$2 ~ /A100|A800/ {print $1}')
if [ -n "${VELA_GPUS:-}" ]; then
  IFS=, read -ra WANT <<< "$VELA_GPUS"
  GPUS_OK=()
  for g in "${WANT[@]}"; do printf '%s\n' "${VISIBLE[@]}" | grep -qx "$g" && GPUS_OK+=("$g"); done
else
  GPUS_OK=("${VISIBLE[@]}")
fi
SLOTS=()
for ((i = 0; i + 1 < ${#GPUS_OK[@]}; i += 2)); do SLOTS+=("${GPUS_OK[$i]},${GPUS_OK[$((i + 1))]}"); done
[ "${#SLOTS[@]}" -gt 0 ] || { say "no GPU pair available (visible: ${VISIBLE[*]:-none})"; exit 1; }
say "launcher host=$(hostname -s) commit=$(git rev-parse --short HEAD) slots=${SLOTS[*]} queue=${QUEUE[*]}"

state() {  # pending | running | done | blocked
  # Running = the row lock on the shared disk exists with a heartbeat younger
  # than 15 minutes (run_one refreshes it every minute on whichever node runs it).
  local out=$VELA_TASK/logs/$1/stdout.log lock=$VELA_TASK/locks/row_$1
  if [ -f "$out" ] && grep -q "EXIT_CODE=0" "$out"; then echo done
  elif [ -f "$out" ] && tail -n 1 "$out" | grep -q "EXIT_CODE=1"; then echo blocked
  elif [ -n "$(find "$lock/heartbeat" -mmin -15 2>/dev/null)" ]; then echo running
  else echo pending; fi
}
slot_free() {
  local g uuid
  for g in ${1//,/ }; do
    uuid=$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)
    # A GPU lock counts only while its heartbeat is fresh (run_one refreshes it every minute).
    [ -n "$(find "$VELA_TASK/locks/$uuid/heartbeat" -mmin -15 2>/dev/null)" ] && return 1
  done
  return 0
}

while true; do
  pending=() running=0
  for m in "${QUEUE[@]}"; do
    case "$(state "$m")" in
      pending) pending+=("$m") ;;
      running) running=$((running + 1)) ;;
    esac
  done
  if [ "${#pending[@]}" -eq 0 ] && [ "$running" -eq 0 ]; then
    say "all rows ended: $(for m in "${QUEUE[@]}"; do printf '%s=%s ' "$m" "$(state "$m")"; done)"
    exit 0
  fi
  for slot in "${SLOTS[@]}"; do
    [ "${#pending[@]}" -gt 0 ] || break
    slot_free "$slot" || continue
    m=${pending[0]}
    pending=("${pending[@]:1}")
    mkdir -p "$VELA_TASK/logs/$m"
    setsid bash "$HERE/run_one.sh" "$m" "$slot" > "$VELA_TASK/logs/$m/run_one.console" 2>&1 < /dev/null &
    say "launched $m on GPUs $slot (pid $!)"
    sleep 30  # let run_one take its locks before the next slot check
  done
  sleep 300
done
