#!/bin/bash
# Regen stage of the a100_qwen3vl8b_200k_s42 matrix (contract sections 11-12):
# prepare-data -> requests -> vLLM generate (one TP=1 shard per GPU) -> finalize.
#
#   RUN=$VELA_TASK/corpus POOL=$VELA_POOL/pool.jsonl GPUS=0,1,2,3,4,5,6,7 \
#     bash scripts/a100_qwen3vl8b_200k/regen.sh [--set key=value ...]
#
# Needs TP (training-env python), VP (vLLM-env python), VELA_SHM (node-local
# scratch: the run's images live there, linked from <run>/data/images) and the
# config variables (VLM_DFLASH_TARGET_QWEN3VL, VELA_POOL). Every step is
# resumable: prepare/finalize reuse matching stage manifests, generate skips
# rows already written. Ends with REGEN_DONE or a non-zero exit.
set -Eeuo pipefail
trap 'echo "REGEN_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
: "${RUN:?set RUN to the regen run dir}" "${POOL:?set POOL to the pool jsonl}" "${GPUS:?set GPUS, e.g. 0,1}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
# The dataset and regen blocks are identical in all seven configs (one shared corpus).
CFG=configs/paper_v1/a100_qwen3vl8b_200k/dflash_s42.yaml
SET=(--set "dataset.path=$POOL" "$@")
IMAGES=$VELA_SHM/$(basename "$RUN")_images
mkdir -p "$RUN/data" "$RUN/logs" "$IMAGES"
# Pool media live on node-local tmpfs and are hard-linked into the data stage,
# so the stage's images dir must be on the same filesystem.
[ -e "$RUN/data/images" ] || ln -s "$IMAGES" "$RUN/data/images"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/logs/regen.log"; }
say "regen start run=$RUN pool=$POOL gpus=$GPUS commit=$(git rev-parse --short HEAD)"
OMP_NUM_THREADS=1 "$TP" scripts/regen_vllm.py prepare-data --config "$CFG" --run-dir "$RUN" "${SET[@]}" \
  2>&1 | tee -a "$RUN/logs/prepare.log" | grep -E "DATA_STAGE_DONE|prepare_data:" || true
grep -q DATA_STAGE_DONE "$RUN/logs/prepare.log"
say "data stage ready"
if [ ! -e "$RUN/regen_vllm/REQUESTS_READY" ]; then
  OMP_NUM_THREADS=1 "$TP" scripts/regen_vllm.py requests --config "$CFG" --run-dir "$RUN" --workers "${WORKERS:-32}" \
    "${SET[@]}" 2>&1 | tee -a "$RUN/logs/requests.log" | tail -2
  grep -q REQUESTS_DONE "$RUN/logs/requests.log"
  touch "$RUN/regen_vllm/REQUESTS_READY"
fi
say "requests ready"
IFS=, read -ra GPU_LIST <<< "$GPUS"
pids=()
for shard in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES=${GPU_LIST[$shard]} OMP_NUM_THREADS=8 "$VP" scripts/regen_vllm.py generate --run-dir "$RUN" \
    --shard "$shard" --num-shards "${#GPU_LIST[@]}" --tp 1 --chunk "${CHUNK:-1024}" \
    >> "$RUN/logs/generate_$(printf %03d "$shard").log" 2>&1 &
  pids+=($!)
done
failed=0
for shard in "${!pids[@]}"; do
  wait "${pids[$shard]}" || failed=1
  grep -q "GENERATE_DONE shard=$shard" "$RUN/logs/generate_$(printf %03d "$shard").log" || failed=1
done
[ "$failed" -eq 0 ] || { say "generate failed"; exit 1; }
say "all ${#GPU_LIST[@]} shards done"
OMP_NUM_THREADS=1 "$TP" scripts/regen_vllm.py finalize --config "$CFG" --run-dir "$RUN" "${SET[@]}" \
  2>&1 | tee -a "$RUN/logs/finalize.log" | tail -2
grep -q FINALIZE_DONE "$RUN/logs/finalize.log"
say "REGEN_DONE"
