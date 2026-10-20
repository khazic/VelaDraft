#!/bin/bash
# One formal row of the a100_qwen3vl8b_200k_s42 matrix on one GPU pair (contract sections 4, 15, 24.5):
#
#   bash scripts/a100_qwen3vl8b_200k/run_one.sh <method> <gpu_a>,<gpu_b>
#
# Logical cuda:0 (first GPU) trains the drafter, cuda:1 holds the frozen target.
# The pair is locked by physical GPU UUID under $VELA_TASK/locks/ for the whole
# run, and the row itself by $VELA_TASK/locks/row_<method> (shared disk, so a
# launcher on any node sees it), refreshed by a heartbeat every minute. The trainer restarts with --resume from runs/<method>/train/ckpt_last.pt
# after any failure (up to MAX_ATTEMPTS). Logs: $VELA_TASK/logs/<method>/
# (stdout.log, stderr.log, metrics.jsonl -> train_log.jsonl, system.jsonl).
# Ends with "EXIT_CODE=0 <method> done" or "EXIT_CODE=1 <method> ..." in stdout.log.
#
# Optional VELA_GPU_COMPANION_CMD: a command run as "$VELA_GPU_COMPANION_CMD <gpu>"
# per GPU for the duration of the row (e.g. a light utilization keeper that a
# shared cluster requires); it is stopped when the row ends.
set -Eeuo pipefail
M=${1:?method} GPUS=${2:?gpu pair, e.g. 0,1}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}" "${TP:?}"
CORPUS=${VELA_CORPUS:-$VELA_TASK/corpus}
TOTAL=75000
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
CFG=configs/paper_v1/a100_qwen3vl8b_200k/${M}_s42.yaml
RUN=$VELA_TASK/runs/$M
LOGS=$VELA_TASK/logs/$M
mkdir -p "$RUN" "$LOGS" "$VELA_TASK/locks"
OUT=$LOGS/stdout.log ERR=$LOGS/stderr.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }
trap 'say "run_one abort line=$LINENO cmd=$BASH_COMMAND exit=$?"' ERR

[ -f "$CFG" ] || { say "EXIT_CODE=1 $M unknown method (no $CFG)"; exit 1; }
[ -f "$CORPUS/regen/checksums.txt" ] || { say "EXIT_CODE=1 $M corpus not frozen ($CORPUS)"; exit 1; }
IFS=, read -ra PAIR <<< "$GPUS"
[ "${#PAIR[@]}" -eq 2 ] || { say "EXIT_CODE=1 $M needs exactly two GPUs, got $GPUS"; exit 1; }

# Row lock + physical-UUID locks (mkdir is atomic); released on exit.
# A lock whose heartbeat is older than 15 minutes belongs to a dead run (e.g. a lost
# node) and is taken over; the rename makes the takeover race-free between launchers.
# Lock conflicts exit 3 without an EXIT_CODE line, so they never mark the row blocked.
ROW_LOCK=$VELA_TASK/locks/row_$M
if [ -d "$ROW_LOCK" ] && [ -z "$(find "$ROW_LOCK/heartbeat" -mmin -15 2>/dev/null)" ]; then
  mv "$ROW_LOCK" "$ROW_LOCK.stale.$$" 2>/dev/null && say "took over a stale row lock: $(cat "$ROW_LOCK.stale.$$/owner" 2>/dev/null)"
fi
if ! mkdir "$ROW_LOCK" 2>/dev/null; then
  echo "[$(date '+%F %T')] $M already running: $(cat "$ROW_LOCK/owner" 2>/dev/null)"
  exit 3
fi
echo "$M pid=$$ host=$(hostname -s) gpus=$GPUS since=$(date -Is)" > "$ROW_LOCK/owner"
touch "$ROW_LOCK/heartbeat"
UUIDS=()
for g in "${PAIR[@]}"; do UUIDS+=("$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)"); done
LOCKED=()
release() {
  rm -rf "$ROW_LOCK"
  for uuid in "${LOCKED[@]}"; do rm -rf "$VELA_TASK/locks/$uuid"; done
  for pid in ${COMPANIONS:-}; do kill "$pid" 2>/dev/null || true; done
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true
}
trap release EXIT
for uuid in "${UUIDS[@]}"; do
  lock=$VELA_TASK/locks/$uuid
  if [ -d "$lock" ] && [ -z "$(find "$lock/heartbeat" -mmin -15 2>/dev/null)" ]; then
    mv "$lock" "$lock.stale.$$" 2>/dev/null && say "took over a stale GPU lock: $(cat "$lock.stale.$$/owner" 2>/dev/null)"
  fi
  if mkdir "$lock" 2>/dev/null; then
    echo "$M pid=$$ host=$(hostname -s) gpus=$GPUS since=$(date -Is)" > "$lock/owner"
    touch "$lock/heartbeat"
    LOCKED+=("$uuid")
  else
    echo "[$(date '+%F %T')] $M GPU $uuid is locked by: $(cat "$VELA_TASK/locks/$uuid/owner" 2>/dev/null)"
    exit 3
  fi
done

export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8
PROVENANCE=$("$TP" -c 'import vlm_dflash; print(vlm_dflash.__file__)')
[ "${PROVENANCE#"$PWD"/}" != "$PROVENANCE" ] || { say "EXIT_CODE=1 $M imports $PROVENANCE, not $PWD"; exit 1; }
ln -sfn "$CORPUS/data" "$RUN/data"
ln -sfn "$CORPUS/regen" "$RUN/regen"
ln -sfn "$RUN/train/train_log.jsonl" "$LOGS/metrics.jsonl"

COMPANIONS=""
if [ -n "${VELA_GPU_COMPANION_CMD:-}" ]; then
  for g in "${PAIR[@]}"; do
    setsid $VELA_GPU_COMPANION_CMD "$g" > "$LOGS/companion_gpu$g.log" 2>&1 < /dev/null &
    COMPANIONS="$COMPANIONS $!"
  done
fi
( while true; do
    printf '{"time": %s, "gpus": [%s], "host_mem_used_gb": %s}\n' "$(date +%s)" \
      "$(nvidia-smi -i "$GPUS" --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '{printf "%s{\"index\": %s, \"uuid\": \"%s\", \"util\": %s, \"mem_mib\": %s}", (NR>1?", ":""), $1, $2, $3, $4}')" \
      "$(free -g | awk '/^Mem/ {print $3}')" >> "$LOGS/system.jsonl"
    touch "$ROW_LOCK/heartbeat"
    for uuid in "${LOCKED[@]}"; do touch "$VELA_TASK/locks/$uuid/heartbeat"; done
    sleep 60
  done ) &
SAMPLER=$!

say "formal $M gpus=$GPUS uuids=${UUIDS[*]} host=$(hostname -s) commit=$(git rev-parse --short HEAD) code=$PROVENANCE"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  args=(-m vlm_dflash train --config "$CFG" --run-dir "$RUN" --mode online)
  [ -e "$RUN/train/ckpt_last.pt" ] && args+=(--resume)
  say "attempt $attempt args=${args[*]}"
  set +e
  "$TP" "${args[@]}" >> "$OUT" 2>> "$ERR"
  code=$?
  set -e
  if tail -n 200 "$OUT" | grep -q "train: done with steps=$TOTAL"; then
    say "EXIT_CODE=0 $M done"
    exit 0
  fi
  say "attempt $attempt exited $code; last error: $(grep -E '^[A-Za-z]+Error' "$ERR" | tail -1 | cut -c1-200)"
  sleep 60
done
say "EXIT_CODE=1 $M gave up after $MAX_ATTEMPTS attempts"
exit 1
