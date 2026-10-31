#!/bin/bash
# Evaluation of the continuation, one worker per listed GPU (every job runs on one GPU):
#
#   bash scripts/a100_qwen3vl8b_200k/continue_eval.sh <gpu>[,<gpu>...] [method ...]
#
# Checkpoints per method: orig_final (the source row's ckpt_final.pt; its ckpt_last.pt train
# state is scored by the objective too, as the step-0 equivalence check), cont_ep1 (the
# continuation's ckpt_step0100000.pt) and cont_ep2 (its ckpt_final.pt once the row wrote
# train_summary.json). Jobs: the heldout_dev AR baseline (once); per checkpoint the validation
# objective on the frozen dev split (continue_objective.py, position 75000 = the held schedule)
# and speculative decoding of heldout_dev with sequential (exactness reference) and
# strict_parallel (speed path) verification (eval_matrix.py, greedy, batch 1).
# Results: $VELA_CONT/eval/<method>/<label>/. Jobs are claimed by atomic mkdir under
# $VELA_CONT/eval/claims, so several invocations (nodes) share one queue; a failed job drops
# its claim and is retried on a later pass. Passes repeat every VELA_EVAL_POLL seconds
# (default 600) until every job of every method is done.
# Needs VELA_TASK, VELA_CONT, VELA_DEVTEST (staged dev/test rows with target answers), TP.
set -Eeuo pipefail
GPUS=${1:?gpu list}
shift
METHODS=("$@")
[ "${#METHODS[@]}" -gt 0 ] || METHODS=(dflash2 dfly dflash dspark dflare domino jetspec)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}" "${VELA_CONT:?}" "${VELA_DEVTEST:?}" "${TP:?}"
export PYTHONPATH=$PWD
EVAL=$VELA_CONT/eval CLAIMS=$VELA_CONT/eval/claims LOGS=$VELA_CONT/logs/eval
SET=heldout_dev POSITION=75000 POLL=${VELA_EVAL_POLL:-600}
mkdir -p "$CLAIMS" "$LOGS" "$EVAL/ar"
say() { echo "[$(date '+%F %T')] $*"; }

checkpoint() {  # checkpoint METHOD LABEL -> path, empty when not available yet
  case $2 in
    orig_final) echo "$VELA_TASK/runs/$1/train/ckpt_final.pt" ;;
    cont_ep1) [ -f "$VELA_CONT/runs/$1/train/ckpt_step0100000.pt" ] && echo "$VELA_CONT/runs/$1/train/ckpt_step0100000.pt" ;;
    cont_ep2) [ -f "$VELA_CONT/runs/$1/train/train_summary.json" ] && echo "$VELA_CONT/runs/$1/train/ckpt_final.pt" ;;
  esac
  return 0
}

jobs() {  # one line per job: KEY DONE_FILE KIND ARGS...
  echo "ar $EVAL/ar/$SET.ar.summary.json ar"
  local m label ckpt out verification
  for m in "${METHODS[@]}"; do
    for label in orig_final cont_ep1 cont_ep2; do
      ckpt=$(checkpoint "$m" "$label")
      out=$EVAL/$m/$label
      if [ -z "$ckpt" ]; then echo "pending.$m.$label - pending"; continue; fi
      echo "obj.$m.$label $out/objective.json obj $m $label $ckpt"
      for verification in sequential strict_parallel; do
        echo "sd.$m.$label.$verification $out/$SET.$(basename "$ckpt" .pt).$verification.summary.json sd $m $label $ckpt $verification"
      done
    done
  done
}

run_job() {  # run_job GPU KIND ARGS...
  local gpu=$1 kind=$2
  shift 2
  case $kind in
    ar)
      CUDA_VISIBLE_DEVICES=$gpu "$TP" scripts/a100_qwen3vl8b_200k/eval_matrix.py ar --set "$SET" \
        --sets "$VELA_TASK/eval_sets" --contract "$VELA_TASK/runs/dflash/contract.json" --out "$EVAL/ar" \
        > "$LOGS/ar_$SET.log" 2>&1 ;;
    obj)
      local m=$1 label=$2 ckpt=$3 out=$EVAL/$1/$2 extra=()
      mkdir -p "$out"
      if [ "$label" = orig_final ]; then
        extra=(--checkpoint "$VELA_TASK/runs/$m/train/ckpt_last.pt" --label orig_train_state
               --out "$out/objective.train_state.json")
      fi
      CUDA_VISIBLE_DEVICES=$gpu "$TP" scripts/a100_qwen3vl8b_200k/continue_objective.py --single-gpu \
        --config "$VELA_TASK/runs/$m/config.json" --contract "$VELA_TASK/runs/$m/contract.json" \
        --rows-dir "$VELA_DEVTEST" --split "$VELA_DEVTEST/data/split_dev.jsonl" --position "$POSITION" \
        "${extra[@]}" --checkpoint "$ckpt" --label "$label" --out "$out/objective.json" \
        > "$LOGS/obj_${m}_$label.log" 2>&1 ;;
    sd)
      local m=$1 label=$2 ckpt=$3 verification=$4
      CUDA_VISIBLE_DEVICES=$gpu "$TP" scripts/a100_qwen3vl8b_200k/eval_matrix.py sd --set "$SET" \
        --sets "$VELA_TASK/eval_sets" --contract "$VELA_TASK/runs/dflash/contract.json" --method "$m" \
        --checkpoint "$ckpt" --verification "$verification" --ar "$EVAL/ar" --out "$EVAL/$m/$label" \
        > "$LOGS/sd_${m}_${label}_$verification.log" 2>&1 ;;
  esac
}

worker() {  # one pass over the queue on one GPU; prints the number of unfinished jobs
  local gpu=$1 key done_file kind left=0
  while read -r key done_file kind rest; do
    [ "$kind" = pending ] && { left=$((left + 1)); continue; }
    [ -f "$done_file" ] && continue
    left=$((left + 1))
    if [ "$kind" = sd ] && [ ! -f "$EVAL/ar/$SET.ar.summary.json" ]; then continue; fi
    mkdir "$CLAIMS/$key" 2>/dev/null || continue
    echo "$(hostname -s) gpu=$gpu since=$(date -Is)" > "$CLAIMS/$key/owner"
    say "gpu $gpu: start $key" >&2
    # shellcheck disable=SC2086
    if run_job "$gpu" "$kind" $rest && [ -f "$done_file" ]; then
      say "gpu $gpu: done $key" >&2
      left=$((left - 1))
    else
      say "gpu $gpu: FAILED $key (claim dropped, retried next pass)" >&2
      rm -rf "$CLAIMS/$key"
    fi
  done < <(jobs)
  echo "$left"
}

say "continue_eval gpus=$GPUS methods=${METHODS[*]} commit=$(git rev-parse --short HEAD)"
IFS=, read -ra GPU_LIST <<< "$GPUS"
while true; do
  tmp=$(mktemp -d)
  for gpu in "${GPU_LIST[@]}"; do worker "$gpu" > "$tmp/$gpu" & done
  wait
  left=$(cat "$tmp"/* | sort -n | tail -1)
  rm -rf "$tmp"
  [ "$left" -eq 0 ] && { say "EVAL_ALL_DONE"; exit 0; }
  say "pass finished, $left job(s) left; next pass in ${POLL}s"
  sleep "$POLL"
done
