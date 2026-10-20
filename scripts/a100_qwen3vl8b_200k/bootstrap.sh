#!/bin/bash
# Bootstrap of the a100_qwen3vl8b_200k_s42 matrix (contract section 24.5). Run once per node before run_all.sh:
#
#   bash scripts/a100_qwen3vl8b_200k/bootstrap.sh 2>&1 | tee $VELA_TASK/logs/bootstrap.$(date +%Y%m%d_%H%M%S).log
#
# Mandatory gates, in order (any failure exits non-zero and nothing is launched):
#   1. environment: training env (torch / transformers / triton) and vLLM env versions;
#   2. target checkpoint: present (downloaded from the pinned HF revision when missing),
#      small-file git hashes equal to that revision;
#   3. data: OneVision root and the frozen source pool checksums;
#   4. repo: HEAD, branch, dirty files; the seven configs match their generator;
#   5. CPU regression tests: tokenizer non-roundtrip / prompt-drift capture tests,
#      Qwen3-VL contract tests, matrix config tests;
#   6. GPU preflight on the first slot (real images, hidden routing, final hidden,
#      MRoPE, mask, split-device transfer, FlexAttention parity, seven-method
#      forward/backward, checkpoint save/load);
#   7. method smokes (overfit sanity, exact resume, calibration) on the first slot,
#      skipped for methods that already passed at this commit;
#   8. resolved configs: the seven effective configs and their sha256 under $VELA_TASK/configs/.
set -Eeuo pipefail
trap 'echo "BOOTSTRAP_FAIL line=$LINENO cmd=$BASH_COMMAND exit=$?" >&2' ERR
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE/../.."
: "${VELA_TASK:?}" "${VLM_DFLASH_TARGET_QWEN3VL:?}" "${ONEVISION_ROOT:?}" "${VELA_SHM:?}" "${TP:?}" "${VP:?}"
TARGET_REPO=Qwen/Qwen3-VL-8B-Instruct
TARGET_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
# git blob hashes of the small files at TARGET_REVISION (from the hub's paths-info).
declare -A TARGET_BLOBS=(
  [config.json]=0e2df614c5fe1ad2b6b540c8625bebf2d7ec43ca
  [generation_config.json]=e116347d7ef42621c7a5957f82c015c38d939c1f
  [chat_template.json]=1081bacf1af7c7c6de4a585ce02cd0fd34e382da
  [tokenizer.json]=c6cc1014128b19d1fc46b1d30a23e3b1d35db421
)
QUEUE=(dflash dfly dflash2 dspark dflare domino jetspec)
export PYTHONPATH=$PWD
OUT=$VELA_TASK/preflight
mkdir -p "$OUT" "$VELA_TASK/logs" "$VELA_TASK/configs"
say() { echo "[$(date '+%F %T')] $*"; }

say "1/8 environment"
"$TP" -c 'import sys, torch, transformers, triton; print("train env", sys.version.split()[0], torch.__version__, torch.version.cuda, transformers.__version__, "triton", triton.__version__)' | tee "$OUT/env.txt"
"$VP" -c 'import vllm, torch, transformers; print("vllm env", vllm.__version__, torch.__version__, transformers.__version__)' 2>/dev/null | tail -1 | tee -a "$OUT/env.txt"
PROVENANCE=$("$TP" -c 'import vlm_dflash; print(vlm_dflash.__file__)')
[ "${PROVENANCE#"$PWD"/}" != "$PROVENANCE" ] || { echo "BOOTSTRAP_FAIL imports $PROVENANCE, not $PWD"; exit 1; }

say "2/8 target checkpoint $TARGET_REPO@$TARGET_REVISION"
if [ ! -f "$VLM_DFLASH_TARGET_QWEN3VL/config.json" ]; then
  hf download "$TARGET_REPO" --revision "$TARGET_REVISION" --local-dir "$VLM_DFLASH_TARGET_QWEN3VL" 2>&1 | tail -2
fi
for name in "${!TARGET_BLOBS[@]}"; do
  got=$(git hash-object "$VLM_DFLASH_TARGET_QWEN3VL/$name")
  [ "$got" = "${TARGET_BLOBS[$name]}" ] || { echo "BOOTSTRAP_FAIL $name is $got, not revision $TARGET_REVISION"; exit 1; }
done
ls "$VLM_DFLASH_TARGET_QWEN3VL"/model-0000[1-4]-of-00004.safetensors > /dev/null
echo "target revision $TARGET_REVISION (blob hashes verified)" | tee -a "$OUT/env.txt"

say "3/8 data"
[ -d "$ONEVISION_ROOT" ] || { echo "BOOTSTRAP_FAIL no OneVision root $ONEVISION_ROOT"; exit 1; }
if [ -n "${VELA_SRC_POOL:-}" ]; then ( cd "$VELA_SRC_POOL" && sha256sum -c checksums.txt --quiet ) && echo "source pool checksums ok"; fi

say "4/8 repo"
{ echo "head $(git rev-parse HEAD)"; echo "branch $(git rev-parse --abbrev-ref HEAD)"; echo "dirty:"; git status --porcelain; } | tee "$OUT/repo_head.txt"
"$TP" scripts/a100_qwen3vl8b_200k/matrix.py check

say "5/8 CPU regression tests"
CUDA_VISIBLE_DEVICES="" "$TP" -m pytest -q tests/test_wp2_multiturn.py tests/test_turn_start_positions.py \
  tests/test_qwen3vl_target.py tests/test_paper_v1_qwen3vl8b_configs.py tests/test_qwen3vl8b_pool_tools.py \
  "tests/test_online_mode.py::test_online_capture_rejects_prompt_drift_and_token_count_mismatch" \
  2>&1 | tail -3 | tee "$OUT/cpu_tests.txt"
grep -qE "passed" "$OUT/cpu_tests.txt" && ! grep -qE "failed|error" "$OUT/cpu_tests.txt"

mapfile -t SLOTS < <(VELA_GPUS=${VELA_GPUS:-} bash -c '
  mapfile -t g < <(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F", " "\$2 ~ /A100|A800/ {print \$1}")
  if [ -n "$VELA_GPUS" ]; then IFS=, read -ra g <<< "$VELA_GPUS"; fi
  for ((i = 0; i + 1 < ${#g[@]}; i += 2)); do echo "${g[$i]},${g[$((i + 1))]}"; done')
[ "${#SLOTS[@]}" -gt 0 ] || { echo "BOOTSTRAP_FAIL no GPU pair"; exit 1; }
SLOT=${SLOTS[0]}

say "6/8 GPU preflight on $SLOT"
GPUS=$SLOT bash scripts/a100_qwen3vl8b_200k/preflight.sh > "$VELA_TASK/logs/preflight_bootstrap.log" 2>&1
grep "^PREFLIGHT_GPU" "$VELA_TASK/logs/preflight_bootstrap.log"

say "7/8 method smokes on $SLOT (corpus ${SMOKE_CORPUS:=$VELA_TASK/pilot})"
COMMIT=$(git rev-parse --short HEAD)
for m in "${QUEUE[@]}"; do
  log=$VELA_TASK/smoke/$m/smoke.log
  if [ -f "$log" ] && grep -q "commit=$COMMIT " "$log" && grep -q "^SMOKE_PASS" "$log"; then
    say "smoke $m already passed at $COMMIT"
    continue
  fi
  SMOKE_CORPUS=$SMOKE_CORPUS bash scripts/a100_qwen3vl8b_200k/smoke.sh "$m" "$SLOT" > "$VELA_TASK/logs/smoke_$m.log" 2>&1
  grep -E "^SMOKE_(PASS|FAIL)" "$log"
done

say "8/8 resolved configs"
"$TP" - "$VELA_TASK/configs" <<'PY'
import hashlib, json, sys
from argparse import Namespace
from pathlib import Path

from vlm_dflash.cli import build_train_config

out = Path(sys.argv[1])
hashes = {}
for method in ("dflash", "dfly", "dflash2", "dspark", "dflare", "domino", "jetspec"):
    path = Path(f"configs/paper_v1/a100_qwen3vl8b_200k/{method}_s42.yaml")
    cfg = build_train_config(Namespace(
        preset="single-gpu-48g", config=str(path), target=None, dataset=None, hf_dataset=None, hf_split=None,
        dataset_source=None, annotation=None, images_dir=None, limit=None, algorithm=None, set_list=[],
        feature_mode=None))
    (out / f"{method}_s42.yaml").write_text(path.read_text())
    (out / f"{method}_s42.effective.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    hashes[method] = {"yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "effective_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
(out / "config_hashes.json").write_text(json.dumps(hashes, indent=2))
print(json.dumps(hashes, indent=2))
PY
say "BOOTSTRAP_PASS commit=$COMMIT"
