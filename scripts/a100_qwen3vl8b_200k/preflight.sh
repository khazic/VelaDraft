#!/bin/bash
# Node + GPU-pair preflight of the a100_qwen3vl8b_200k_s42 matrix (contract sections 3 and 5).
#
#   GPUS=0,1 bash scripts/a100_qwen3vl8b_200k/preflight.sh
#
# Needs the environment the matrix scripts share: VELA_TASK (task root),
# VLM_DFLASH_TARGET_QWEN3VL, ONEVISION_ROOT, VELA_SHM (node-local scratch),
# TP (training-env python). Records the repo state, nvidia-smi, the topology and
# the GPU UUIDs under $VELA_TASK/preflight/<host>-gpu<pair>/, then runs
# preflight_gpu.py on the pair. Exits non-zero when a mandatory gate fails.
set -Eeuo pipefail
trap 'echo "PREFLIGHT_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
: "${GPUS:?set GPUS to one GPU pair, e.g. GPUS=0,1}"
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT=$VELA_TASK/preflight/$(hostname -s)-gpu${GPUS//,/_}
mkdir -p "$OUT"
{
  echo "date: $(date -Is)"
  echo "repo: $REPO"
  echo "branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  echo "head: $(git -C "$REPO" rev-parse HEAD)"
  echo "dirty:"; git -C "$REPO" status --porcelain
  echo "python: $("$TP" -c 'import sys; print(sys.version.split()[0])')"
} > "$OUT/repo.txt"
nvidia-smi > "$OUT/nvidia-smi.txt"
nvidia-smi topo -m > "$OUT/nvidia-smi-topo.txt"
nvidia-smi -L > "$OUT/nvidia-smi-L.txt"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total --format=csv > "$OUT/gpus.csv"
{ free -g; df -h /dev/shm "$VELA_TASK"; } > "$OUT/host.txt"
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPUS "$TP" scripts/a100_qwen3vl8b_200k/preflight_gpu.py \
  --model "$VLM_DFLASH_TARGET_QWEN3VL" --onevision-root "$ONEVISION_ROOT" \
  --scratch "$VELA_SHM/preflight_${GPUS//,/_}" --out "$OUT" 2>&1 | tee "$OUT/preflight_gpu.log"
grep -q "^PREFLIGHT_GPU_PASS" "$OUT/preflight_gpu.log"
