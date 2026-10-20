#!/bin/bash
# Method smoke gate of the a100_qwen3vl8b_200k_s42 matrix (contract sections 22-24), on one GPU pair:
#
#   SMOKE_CORPUS=$VELA_TASK/pilot bash scripts/a100_qwen3vl8b_200k/smoke.sh dflash 6,7
#
#   run A  overfit sanity (4 samples x 200 updates) + 110 updates uninterrupted,
#          the 100-update calibration window (stage timing, peaks, GPU utilization);
#   run B  the same config, SIGKILLed right after the step-8 resumable checkpoint,
#          then --resume until update 40;
#   gate   resume_gate.py: steps, LR, sample-stream digests and recipe stage must be
#          identical update by update, losses within BF16 tolerance.
#
# Both runs use the formal config with the smoke corpus' dataset/regen overrides,
# in their own run dirs under $VELA_TASK/smoke/<method>/. Prints SMOKE_PASS / SMOKE_FAIL.
set -Eeuo pipefail
trap 'echo "SMOKE_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
M=${1:?method} GPUS=${2:?gpu pair, e.g. 6,7}
: "${SMOKE_CORPUS:?set SMOKE_CORPUS to a finished regen run dir (data/ + regen/)}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
CFG=configs/paper_v1/a100_qwen3vl8b_200k/${M}_s42.yaml
OUT=$VELA_TASK/smoke/$M
CALIB_STEPS=${CALIB_STEPS:-110} RESUME_STEPS=${RESUME_STEPS:-40} KILL_STEP=${KILL_STEP:-8}
mkdir -p "$OUT"
# The smoke corpus was regenerated from its own pool slice; the same overrides
# keep the data/regen stage digests equal so training reuses them as-is.
CORPUS_SET=(--set "dataset.path=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["path"])' "$SMOKE_CORPUS/config.json")"
            --set "dataset.slice=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"]["slice"])' "$SMOKE_CORPUS/config.json")"
            --set regen.select=null)
say() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/smoke.log"; }

# train_args RUN_DIR STEPS [extra args] -> ARGS. The trainer is started directly
# (never through a backgrounded shell function), so $! is the python process.
train_args() {
  local run=$1 steps=$2
  shift 2
  mkdir -p "$run"
  ln -sfn "$SMOKE_CORPUS/data" "$run/data"
  ln -sfn "$SMOKE_CORPUS/regen" "$run/regen"
  ARGS=(-m vlm_dflash train --config "$CFG" --run-dir "$run" --mode online --steps "$steps" "${CORPUS_SET[@]}"
        --set train.save_every_steps="$KILL_STEP" "$@")
}
export CUDA_VISIBLE_DEVICES=$GPUS OMP_NUM_THREADS=8

# Code provenance: the env also has another VelaDraft checkout installed; this
# repo must be the one imported (PYTHONPATH), or the smoke tests the wrong code.
PROVENANCE=$(PYTHONPATH=$PWD "$TP" -c 'import vlm_dflash; print(vlm_dflash.__file__)')
[ "${PROVENANCE#"$PWD"/}" != "$PROVENANCE" ] || { echo "SMOKE_FAIL $M imports $PROVENANCE, not $PWD"; exit 1; }
export PYTHONPATH=$PWD
say "smoke $M gpus=$GPUS commit=$(git rev-parse --short HEAD) corpus=$SMOKE_CORPUS code=$PROVENANCE"
rm -rf "$OUT/run_a" "$OUT/run_b"
( while true; do nvidia-smi -i "$GPUS" --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader,nounits \
    | sed "s/^/$(date +%s), /" >> "$OUT/gpu_samples.csv"; sleep 5; done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

say "run A: overfit sanity + $CALIB_STEPS updates"
start=$(date +%s)
train_args "$OUT/run_a" "$CALIB_STEPS"
"$TP" "${ARGS[@]}" > "$OUT/run_a.log" 2>&1
echo "$start $(date +%s)" > "$OUT/run_a.window"
grep -q "train: done with steps=$CALIB_STEPS" "$OUT/run_a.log"
say "run A done"

say "run B: kill after the step-$KILL_STEP checkpoint, resume to $RESUME_STEPS"
train_args "$OUT/run_b" "$CALIB_STEPS"
"$TP" "${ARGS[@]}" > "$OUT/run_b1.log" 2>&1 &
PID=$!
until grep -q "wrote resumable checkpoint (step $KILL_STEP)" "$OUT/run_b1.log"; do
  kill -0 "$PID" 2>/dev/null || { say "run B exited before step $KILL_STEP"; exit 1; }
  sleep 2
done
kill -9 "$PID"
wait "$PID" 2>/dev/null || true
say "run B killed after step $KILL_STEP"
train_args "$OUT/run_b" "$CALIB_STEPS" --resume
"$TP" "${ARGS[@]}" > "$OUT/run_b2.log" 2>&1 &
PID=$!
until [ "$(wc -l < "$OUT/run_b/train/train_log.jsonl" 2>/dev/null || echo 0)" -ge "$RESUME_STEPS" ]; do
  kill -0 "$PID" 2>/dev/null || { say "resumed run B exited early"; exit 1; }
  sleep 2
done
kill -9 "$PID"
wait "$PID" 2>/dev/null || true
say "run B resumed to $RESUME_STEPS"

"$TP" scripts/a100_qwen3vl8b_200k/resume_gate.py --method "$M" --run-a "$OUT/run_a" --run-b "$OUT/run_b" \
  --steps "$RESUME_STEPS" --calib-from 11 --gpu-samples "$OUT/gpu_samples.csv" --out "$OUT/smoke_report.json" \
  | tee -a "$OUT/smoke.log"
grep -q "^SMOKE_PASS" "$OUT/smoke.log"
