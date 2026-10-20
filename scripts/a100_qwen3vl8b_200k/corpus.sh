#!/bin/bash
# Build and freeze the shared 200K corpus of the a100_qwen3vl8b_200k_s42 matrix (contract sections 11-12).
#
#   GPUS=0,1,2,3,4,5,6,7 OVERSAMPLE=general=1.61,ocr=1.45,chart=2.5,reasoning=1.98 \
#     bash scripts/a100_qwen3vl8b_200k/corpus.sh
#
# 1. derive the formal pool from the frozen source pool (VELA_SRC_POOL) with the
#    per-group oversample measured by the regen pilot; dev/test are unchanged;
# 2. link + verify its media from VELA_MEDIA_SOURCE (a prepared-data image dir);
# 3. regen.sh over GPUS (vLLM, one shard per GPU; resumable);
# 4. freeze: checksums, read-only files, the contract-layout manifests and the
#    regen report. Ends with CORPUS_FROZEN.
set -Eeuo pipefail
trap 'echo "CORPUS_ABORT line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
: "${VELA_TASK:?}" "${VELA_SRC_POOL:?}" "${VELA_MEDIA_SOURCE:?}" "${VELA_SHM:?}" "${TP:?}" "${GPUS:?}" "${OVERSAMPLE:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
POOL=${VELA_POOL:-$VELA_TASK/data/pool}
CORPUS=$VELA_TASK/corpus
QUOTAS=general=70000,ocr=60000,chart=30000,reasoning=40000
say() { echo "[$(date '+%F %T')] $*" | tee -a "$VELA_TASK/logs/corpus.log"; }
mkdir -p "$VELA_TASK/logs"
say "corpus start commit=$(git rev-parse --short HEAD) oversample=$OVERSAMPLE gpus=$GPUS"
if [ ! -s "$POOL/checksums.txt" ]; then
  "$TP" scripts/a100_qwen3vl8b_200k/derive_pool.py --src "$VELA_SRC_POOL" --out "$POOL" --quotas "$QUOTAS" \
    --oversample "$OVERSAMPLE" --media-root "${VELA_SRC_MEDIA_ROOT:?}=$VELA_SHM/pool_media" | tee -a "$VELA_TASK/logs/corpus.log"
fi
( cd "$POOL" && sha256sum -c checksums.txt --quiet ) && say "pool checksums ok"
"$TP" scripts/a100_qwen3vl8b_200k/link_media.py --pool "$POOL/pool.jsonl" --from "$VELA_MEDIA_SOURCE" --workers 32 \
  | tail -2 | tee -a "$VELA_TASK/logs/corpus.log"
RUN=$CORPUS POOL=$POOL/pool.jsonl GPUS=$GPUS bash scripts/a100_qwen3vl8b_200k/regen.sh
"$TP" scripts/a100_qwen3vl8b_200k/freeze_corpus.py --task "$VELA_TASK" --pool "$POOL" --corpus "$CORPUS" \
  | tee -a "$VELA_TASK/logs/corpus.log"
"$TP" scripts/a100_qwen3vl8b_200k/regen_report.py --run "$CORPUS" --meta "$POOL/pool_meta.jsonl" > /dev/null
say "CORPUS_FROZEN"
