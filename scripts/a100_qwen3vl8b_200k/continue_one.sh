#!/bin/bash
# Continuation of one finished row of the a100_qwen3vl8b_200k_s42 matrix on one GPU pair:
#
#   bash scripts/a100_qwen3vl8b_200k/continue_one.sh <method> <gpu_a>,<gpu_b> [smoke]
#
# A new run directory $VELA_CONT/runs/<method> starts from the finished row's
# $VELA_TASK/runs/<method>/train/ckpt_last.pt (weights, fp32 masters, AdamW moments, RNG,
# data position; the row's ckpt_final.pt must hold the same weights) and trains
# VELA_CONT_ADD_STEPS more updates (default 50,000 = two epochs) at a constant VELA_CONT_LR
# (default 1e-5, the row's final rate, no warmup), with the row's schedule-driven objective
# settings (recipe stage, Domino lambda_base) held at its last update. Same config, corpus,
# online feature mode and drafter/target GPU split as run_one.sh; after a failure the trainer
# restarts with --resume from the continuation's own ckpt_last.pt (up to MAX_ATTEMPTS).
# "smoke" runs one update into $VELA_CONT/smoke/<method> (one attempt) and gates it with
# continue_smoke_check.py.
# Logs: $VELA_CONT/logs/<method>/ (smoke: logs/smoke_<method>/), stdout.log ends with
# "EXIT_CODE=0 <method> done" or "EXIT_CODE=1 <method> ...".
set -Eeuo pipefail
M=${1:?method} GPUS=${2:?gpu pair, e.g. 0,1} MODE=${3:-formal}
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}" "${VELA_CONT:?}" "${TP:?}"
CORPUS=${VELA_CORPUS:-$VELA_TASK/corpus}
SOURCE_STEPS=75000
LR=${VELA_CONT_LR:-1e-5}
case $MODE in
  formal) ADD=${VELA_CONT_ADD_STEPS:-50000} RUN=$VELA_CONT/runs/$M LOGS=$VELA_CONT/logs/$M NAME=$M
          MAX_ATTEMPTS=${MAX_ATTEMPTS:-20} ;;
  smoke) ADD=1 RUN=$VELA_CONT/smoke/$M LOGS=$VELA_CONT/logs/smoke_$M NAME=smoke_$M MAX_ATTEMPTS=1 ;;
  *) echo "unknown mode $MODE (formal|smoke)" >&2; exit 2 ;;
esac
TOTAL=$((SOURCE_STEPS + ADD))
CFG=configs/paper_v1/a100_qwen3vl8b_200k/${M}_s42.yaml
SRC=$VELA_TASK/runs/$M/train
mkdir -p "$RUN" "$LOGS" "$VELA_CONT/locks"
OUT=$LOGS/stdout.log ERR=$LOGS/stderr.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }
trap 'say "continue_one abort line=$LINENO cmd=$BASH_COMMAND exit=$?"' ERR

[ -f "$CFG" ] || { say "EXIT_CODE=1 $M unknown method (no $CFG)"; exit 1; }
[ -f "$CORPUS/regen/checksums.txt" ] || { say "EXIT_CODE=1 $M corpus not frozen ($CORPUS)"; exit 1; }
for f in ckpt_last.pt ckpt_final.pt; do
  [ -f "$SRC/$f" ] || { say "EXIT_CODE=1 $M source row has no $SRC/$f"; exit 1; }
done
IFS=, read -ra PAIR <<< "$GPUS"
[ "${#PAIR[@]}" -eq 2 ] || { say "EXIT_CODE=1 $M needs exactly two GPUs, got $GPUS"; exit 1; }

# Row lock + physical-UUID locks under $VELA_CONT/locks (same protocol as run_one.sh: atomic
# mkdir, a heartbeat older than 15 minutes marks a dead owner, conflicts exit 3).
ROW_LOCK=$VELA_CONT/locks/row_$NAME
if [ -d "$ROW_LOCK" ] && [ -z "$(find "$ROW_LOCK/heartbeat" -mmin -15 2>/dev/null)" ]; then
  mv "$ROW_LOCK" "$ROW_LOCK.stale.$$" 2>/dev/null && say "took over a stale row lock: $(cat "$ROW_LOCK.stale.$$/owner" 2>/dev/null)"
fi
if ! mkdir "$ROW_LOCK" 2>/dev/null; then
  echo "[$(date '+%F %T')] $NAME already running: $(cat "$ROW_LOCK/owner" 2>/dev/null)"
  exit 3
fi
echo "$NAME pid=$$ host=$(hostname -s) gpus=$GPUS since=$(date -Is)" > "$ROW_LOCK/owner"
touch "$ROW_LOCK/heartbeat"
UUIDS=()
for g in "${PAIR[@]}"; do UUIDS+=("$(nvidia-smi -i "$g" --query-gpu=uuid --format=csv,noheader)"); done
LOCKED=()
release() {
  rm -rf "$ROW_LOCK"
  for uuid in "${LOCKED[@]}"; do rm -rf "$VELA_CONT/locks/$uuid"; done
  [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true
}
trap release EXIT
for uuid in "${UUIDS[@]}"; do
  lock=$VELA_CONT/locks/$uuid
  if [ -d "$lock" ] && [ -z "$(find "$lock/heartbeat" -mmin -15 2>/dev/null)" ]; then
    mv "$lock" "$lock.stale.$$" 2>/dev/null && say "took over a stale GPU lock: $(cat "$lock.stale.$$/owner" 2>/dev/null)"
  fi
  if mkdir "$lock" 2>/dev/null; then
    echo "$NAME pid=$$ host=$(hostname -s) gpus=$GPUS since=$(date -Is)" > "$lock/owner"
    touch "$lock/heartbeat"
    LOCKED+=("$uuid")
  else
    echo "[$(date '+%F %T')] $NAME GPU $uuid is locked by: $(cat "$VELA_CONT/locks/$uuid/owner" 2>/dev/null)"
    exit 3
  fi
done

export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8
PROVENANCE=$("$TP" -c 'import vlm_dflash; print(vlm_dflash.__file__)')
[ "${PROVENANCE#"$PWD"/}" != "$PROVENANCE" ] || { say "EXIT_CODE=1 $M imports $PROVENANCE, not $PWD"; exit 1; }
ln -sfn "$CORPUS/data" "$RUN/data"
ln -sfn "$CORPUS/regen" "$RUN/regen"
ln -sfn "$RUN/train/train_log.jsonl" "$LOGS/metrics.jsonl"

( while true; do
    printf '{"time": %s, "gpus": [%s], "host_mem_used_gb": %s}\n' "$(date +%s)" \
      "$(nvidia-smi -i "$GPUS" --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '{printf "%s{\"index\": %s, \"uuid\": \"%s\", \"util\": %s, \"mem_mib\": %s}", (NR>1?", ":""), $1, $2, $3, $4}')" \
      "$(free -g | awk '/^Mem/ {print $3}')" >> "$LOGS/system.jsonl"
    touch "$ROW_LOCK/heartbeat"
    for uuid in "${LOCKED[@]}"; do touch "$VELA_CONT/locks/$uuid/heartbeat"; done
    sleep 60
  done ) &
SAMPLER=$!

say "continuation $MODE $M gpus=$GPUS uuids=${UUIDS[*]} host=$(hostname -s) commit=$(git rev-parse --short HEAD) code=$PROVENANCE source=$SRC steps=$SOURCE_STEPS+$ADD lr=$LR"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  args=(-m vlm_dflash train --config "$CFG" --run-dir "$RUN" --mode online
        --set "continuation.source_state=$SRC/ckpt_last.pt" --set "continuation.source_final=$SRC/ckpt_final.pt"
        --set "continuation.add_steps=$ADD" --set "continuation.lr=$LR")
  [ -e "$RUN/train/ckpt_last.pt" ] && args+=(--resume)
  say "attempt $attempt args=${args[*]}"
  set +e
  "$TP" "${args[@]}" >> "$OUT" 2>> "$ERR"
  code=$?
  set -e
  if tail -n 200 "$OUT" | grep -q "train: done with steps=$TOTAL"; then
    if [ "$MODE" = smoke ] && ! "$TP" scripts/a100_qwen3vl8b_200k/continue_smoke_check.py "$RUN" "$SRC" --lr "$LR" >> "$OUT" 2>> "$ERR"; then
      say "EXIT_CODE=1 $M smoke check failed (see $RUN/smoke_check.json)"
      exit 1
    fi
    say "EXIT_CODE=0 $M done"
    exit 0
  fi
  say "attempt $attempt exited $code; last error: $(grep -E '^[A-Za-z]+Error' "$ERR" | tail -1 | cut -c1-200)"
  [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep 60
done
say "EXIT_CODE=1 $M gave up after $MAX_ATTEMPTS attempts"
exit 1
