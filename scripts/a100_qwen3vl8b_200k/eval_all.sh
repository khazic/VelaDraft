#!/bin/bash
# Evaluation of the a100_qwen3vl8b_200k_s42 matrix (contract sections 26-30) on one GPU pair:
#
#   bash scripts/a100_qwen3vl8b_200k/eval_all.sh <gpu_a>,<gpu_b> [method ...]
#
# Jobs (each on one GPU: target + drafter at batch 1; the two GPUs run two jobs at once):
#   - the shared AR baseline of every evaluation set (once, eval/ar/);
#   - per method, the final checkpoint on every set with sequential verification
#     (correctness reference) and strict_parallel verification (speed path);
#   - per method, the epoch-1 / epoch-2 snapshots (updates 25,000 / 50,000) on
#     heldout_test and mmspec, sequential (matched-budget tables).
# A job whose summary exists is skipped, so the script resumes. Writes
# eval/<method>/DONE when all of a method's jobs finished. Needs VELA_TASK, TP.
set -Eeuo pipefail
trap 'echo "EVAL_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
GPUS=${1:?gpu pair}
shift
METHODS=("$@")
[ "${#METHODS[@]}" -gt 0 ] || METHODS=(dflash dfly dflash2 dspark dflare domino jetspec)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}" "${TP:?}"
export PYTHONPATH=$PWD
SETS=(mmspec textvqa docvqa chartqa long32 heldout_dev heldout_test)
MATCHED_SETS=(heldout_test mmspec)
CONTRACT=$VELA_TASK/runs/dflash/contract.json
IFS=, read -ra GPU_LIST <<< "$GPUS"
say() { echo "[$(date '+%F %T')] $*"; }

jobs_file=$(mktemp)
for set in "${SETS[@]}"; do
  [ -f "$VELA_TASK/eval/ar/$set.ar.summary.json" ] || echo "ar $set" >> "$jobs_file"
done
for m in "${METHODS[@]}"; do
  final=$VELA_TASK/runs/$m/train/ckpt_final.pt
  [ -f "$final" ] || { say "skip $m: no final checkpoint yet"; continue; }
  for set in "${SETS[@]}"; do
    for verification in sequential strict_parallel; do
      [ -f "$VELA_TASK/eval/$m/$set.ckpt_final.$verification.summary.json" ] || echo "sd $set $m $final $verification" >> "$jobs_file"
    done
  done
  for step in 0025000 0050000; do
    snap=$VELA_TASK/runs/$m/train/ckpt_step$step.pt
    [ -f "$snap" ] || continue
    for set in "${MATCHED_SETS[@]}"; do
      [ -f "$VELA_TASK/eval/$m/$set.ckpt_step$step.sequential.summary.json" ] || echo "sd $set $m $snap sequential" >> "$jobs_file"
    done
  done
done
say "eval gpus=$GPUS commit=$(git rev-parse --short HEAD) jobs=$(wc -l < "$jobs_file")"

run_job() {  # run_job GPU KIND SET [METHOD CKPT VERIFICATION]
  local gpu=$1 kind=$2 set=$3
  if [ "$kind" = ar ]; then
    CUDA_VISIBLE_DEVICES=$gpu "$TP" scripts/a100_qwen3vl8b_200k/eval_matrix.py ar --set "$set" \
      --sets "$VELA_TASK/eval_sets" --contract "$CONTRACT" --out "$VELA_TASK/eval/ar" \
      > "$VELA_TASK/logs/eval_ar_$set.log" 2>&1
  else
    local m=$4 ckpt=$5 verification=$6
    until [ -f "$VELA_TASK/eval/ar/$set.ar.summary.json" ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=$gpu "$TP" scripts/a100_qwen3vl8b_200k/eval_matrix.py sd --set "$set" \
      --sets "$VELA_TASK/eval_sets" --contract "$CONTRACT" --method "$m" --checkpoint "$ckpt" \
      --verification "$verification" --ar "$VELA_TASK/eval/ar" --out "$VELA_TASK/eval/$m" \
      > "$VELA_TASK/logs/eval_${m}_${set}_$(basename "$ckpt" .pt)_$verification.log" 2>&1
  fi
}

# One worker per GPU pulls the next job; AR jobs are listed first.
worker() {
  local gpu=$1 line
  while true; do
    line=$(flock "$jobs_file.lock" bash -c 'head -1 "$1" && sed -i 1d "$1"' _ "$jobs_file")
    [ -n "$line" ] || return 0
    say "gpu $gpu: $line"
    # shellcheck disable=SC2086
    run_job "$gpu" $line || say "gpu $gpu: FAILED $line"
  done
}
for gpu in "${GPU_LIST[@]}"; do worker "$gpu" & done
wait
rm -f "$jobs_file" "$jobs_file.lock"
for m in "${METHODS[@]}"; do
  n=$(ls "$VELA_TASK/eval/$m"/*.ckpt_final.*.summary.json 2>/dev/null | wc -l)
  [ "$n" -eq $((2 * ${#SETS[@]})) ] && touch "$VELA_TASK/eval/$m/DONE" && say "$m evaluated"
done
say "EVAL_ALL_DONE"
