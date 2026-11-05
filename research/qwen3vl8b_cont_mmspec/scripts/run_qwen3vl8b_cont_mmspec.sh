#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_qwen3vl8b_cont_mmspec.sh [options]

Options:
  --preflight             Check paths and all requested method exports only.
  --gpus LIST             Physical GPU ids, e.g. 0,1,2,3 (default: 0).
  --methods LIST          Comma list or all (default: all).
  --suite smoke|formal|all (default: all).
  --resume                Skip an already valid report; archive invalid old runs.
  --publish               Upload method directories, summary and manifest to HF.
  --campaign-id ID        Reuse a campaign id instead of generating one.
  --help                  Show this help.
USAGE
}

PRE_FLIGHT=0
GPUS="0"
METHODS="all"
SUITE="all"
RESUME=0
PUBLISH=0
CAMPAIGN_ID=""

while (($#)); do
  case "$1" in
    --preflight) PRE_FLIGHT=1; shift ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --publish) PUBLISH=1; shift ;;
    --campaign-id) CAMPAIGN_ID="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SUITE" in smoke|formal|all) ;; *) echo "invalid --suite: $SUITE" >&2; exit 2 ;; esac

# Machine paths come from a site env file kept outside the repository (Q3VL8B_SITE), defining
# Q3VL8B_ROOT (task root holding exports/), TARGET, SOURCE, IMAGES, PYTHON, RESULT_ROOT and
# optionally SHM (node-local scratch), HF_HOME and Q3VL8B_PRELAUNCH (a hook given one GPU index
# that may release site placeholder processes on that card; the protocol never stops a foreign
# process and refuses a busy card).
if [[ -n "${Q3VL8B_SITE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$Q3VL8B_SITE"
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${Q3VL8B_ROOT:?set Q3VL8B_ROOT (or Q3VL8B_SITE)}"
: "${TARGET:?set TARGET}" "${SOURCE:?set SOURCE}" "${IMAGES:?set IMAGES}" "${PYTHON:?set PYTHON}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
SHM="${SHM:-/dev/shm/${USER:-root}_q3vl8b_cont}"
HF_REPO_ID="${HF_REPO_ID:-khazic/veladraft-qwen3vl8b-200k-cont-dspark}"
HF_PREFIX="${HF_PREFIX:-eval/qwen3vl8b_mmspec600}"

if [[ -z "$CAMPAIGN_ID" ]]; then
  CAMPAIGN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo nogit)"
fi
CAMPAIGN_DIR="$RESULT_ROOT/$CAMPAIGN_ID"
LOG_DIR="$CAMPAIGN_DIR/logs"
mkdir -p "$LOG_DIR"

# Every method uses the A800 port in research/qwen3vl8b_cont_mmspec/native: one CPU export and
# one runtime manifest per method, all at K=7 over the trained block of 8 (see README.md).
ALL_METHODS=(dflash dspark domino jetspec dflash2 dflare dfly)
declare -A DRAFTS MANIFESTS KS PYTHONS
for method in "${ALL_METHODS[@]}"; do
  DRAFTS[$method]="$Q3VL8B_ROOT/exports/$method"
  MANIFESTS[$method]="$Q3VL8B_ROOT/exports/${method}_runtime.json"
  KS[$method]=7
  PYTHONS[$method]="$PYTHON"
done

# Native-eval runtime: offline hub, node-local compile caches, count-mode GPU ownership (the pod
# PID namespace hides host PIDs from nvidia-smi), one client context + one engine core per card.
mkdir -p "$SHM/gpu_locks" "$SHM/triton" "$SHM/xdg"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$SHM/triton" XDG_CACHE_HOME="$SHM/xdg"
export QWEN8B_DIGEST_CACHE="$SHM/digests.sqlite"
export VELADRAFT_GPU_LOCK_DIR="$SHM/gpu_locks"
export VELADRAFT_GPU_OWNERSHIP="${VELADRAFT_GPU_OWNERSHIP:-count}" VELADRAFT_GPU_MAX_OWN_APPS="${VELADRAFT_GPU_MAX_OWN_APPS:-2}"
cd "$REPO"

die() { echo "ERROR: $*" >&2; exit 1; }
log() { echo "[$(date -Is)] $*" | tee -a "$LOG_DIR/launcher.log"; }

wait_for_gpu() {
  local gpu="$1" used pids
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required for native GPU evaluation"
  while :; do
    if [[ -n "${Q3VL8B_PRELAUNCH:-}" ]]; then "$Q3VL8B_PRELAUNCH" "$gpu" || true; fi
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)"
    pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "$used" =~ ^[0-9]+$ && "$used" -lt 500 && -z "$pids" ]]; then
      return 0
    fi
    log "WAIT_GPU gpu=$gpu used=${used:-unknown} pids=${pids:-none}"
    sleep 30
  done
}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
((${#GPU_LIST[@]} > 0)) || die "at least one GPU is required"
for gpu in "${GPU_LIST[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU id: $gpu"; done

if [[ "$METHODS" == all ]]; then
  SELECTED=("${ALL_METHODS[@]}")
else
  IFS=',' read -r -a SELECTED <<< "$METHODS"
fi
for method in "${SELECTED[@]}"; do
  [[ -n "${DRAFTS[$method]+x}" ]] || die "unknown method: $method"
done

preflight_method() {
  local method="$1" py="${PYTHONS[$method]}"
  [[ -d "${DRAFTS[$method]}" ]] || die "$method draft export missing: ${DRAFTS[$method]}"
  [[ -f "${MANIFESTS[$method]}" ]] || die "$method runtime manifest missing: ${MANIFESTS[$method]}"
  [[ -f "$TARGET/config.json" ]] || die "target config missing: $TARGET/config.json"
  [[ -f "$SOURCE" ]] || die "MMSpec source missing: $SOURCE"
  [[ -d "$IMAGES" ]] || die "MMSpec images directory missing: $IMAGES"
  [[ -x "$py" ]] || die "Python runtime missing/executable bit absent: $py"
  "$py" - <<'PY'
import importlib.util
for name in ("torch", "vllm"):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f"{name} is not importable in the selected runtime")
PY
  log "PREFLIGHT_OK method=$method draft=${DRAFTS[$method]} manifest=${MANIFESTS[$method]} K=${KS[$method]} python=$py"
}

for method in "${SELECTED[@]}"; do preflight_method "$method"; done
if ((PRE_FLIGHT)); then
  log "PREFLIGHT_COMPLETE methods=${SELECTED[*]}"
  exit 0
fi

mkdir -p "$CAMPAIGN_DIR"
cat > "$CAMPAIGN_DIR/run_manifest.json" <<EOF_MANIFEST
{
  "campaign_id": "${CAMPAIGN_ID}",
  "target": "${TARGET}",
  "source": "${SOURCE}",
  "images": "${IMAGES}",
  "methods": [$(printf '"%s",' "${SELECTED[@]}" | sed 's/,$//')],
  "dataset": "MMSpec full test",
  "sessions": 600,
  "turns": 723,
  "repeats": 3,
  "max_new_tokens": 1024,
  "gpu_memory_utilization": 0.78,
  "speculative_tokens": 7,
  "code_commit": "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)",
  "gpu_name": "$(nvidia-smi -i "${GPU_LIST[0]}" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)",
  "exports_and_manifests": "${Q3VL8B_ROOT}/exports (export_receipt.json per method)",
  "protocol_notes": "research/qwen3vl8b_cont_mmspec/README.md",
  "hf_repo_id": "${HF_REPO_ID}",
  "hf_prefix": "${HF_PREFIX}/${CAMPAIGN_ID}"
}
EOF_MANIFEST

valid_formal() {
  "${PYTHONS[$1]}" - "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ok=(d.get("status")=="complete" and d.get("resources_verified") is True
    and d.get("timed_pairs")==2169 and not d.get("smoke",False))
raise SystemExit(0 if ok else 1)
PY
}
valid_smoke() {
  "${PYTHONS[$1]}" - "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
ok=(d.get("status")=="complete" and d.get("resources_verified") is True
    and d.get("smoke",False))
raise SystemExit(0 if ok else 1)
PY
}

archive_invalid() {
  local dir="$1"
  if [[ -e "$dir" ]]; then
    mv "$dir" "${dir}.invalid.$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "$dir"
}

upload_method() {
  local method="$1"
  ((PUBLISH)) || return 0
  command -v hf >/dev/null 2>&1 || die "--publish requires the hf CLI"
  hf upload "$HF_REPO_ID" "$CAMPAIGN_DIR/$method" "$HF_PREFIX/$CAMPAIGN_ID/$method" \
    --repo-type model --commit-message "eval: Qwen3-VL-8B CONT ${method} MMSpec ${CAMPAIGN_ID}"
}

run_stage() {
  local method="$1" stage="$2" gpu="$3" out="$4"; shift 4
  local py="${PYTHONS[$method]}" stage_dir="$(dirname "$out")"
  wait_for_gpu "$gpu"
  mkdir -p "$stage_dir"
  local -a cmd=(
    "$py" -u -m vlm_dflash.cli native-eval
    --target "$TARGET" --draft "${DRAFTS[$method]}"
    --source "$SOURCE" --images-dir "$IMAGES"
    --runtime-manifest "${MANIFESTS[$method]}" --method "$method" --gpu "$gpu"
    --gpu-memory-utilization .78 --speculative-tokens "${KS[$method]}"
    --min-pixels 65536 --max-pixels 1048576 --max-model-len 8192
    --chat-template-kwargs '{"enable_thinking":false}'
    --system-prompt 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human questions.'
    "$@" --out "$out"
  )
  printf '%q ' "${cmd[@]}" > "$stage_dir/command.txt"
  printf '\n' >> "$stage_dir/command.txt"
  "${cmd[@]}" > "$stage_dir/stdout.log" 2>&1
}

run_method() {
  local method="$1" gpu="$2"
  local dir="$CAMPAIGN_DIR/$method"
  mkdir -p "$dir"
  log "START method=$method gpu=$gpu K=${KS[$method]}"

  if [[ "$SUITE" == smoke || "$SUITE" == all ]]; then
    local smoke_dir="$dir/smoke" smoke="$smoke_dir/report.json"
    if ((RESUME)) && [[ -f "$smoke" ]] && valid_smoke "$method" "$smoke"; then
      log "SKIP_VALID_SMOKE method=$method"
    else
      [[ -f "$smoke" ]] && archive_invalid "$smoke_dir"
      log "RUN_SMOKE method=$method gpu=$gpu"
      if ! run_stage "$method" smoke "$gpu" "$smoke" --num-prompts 2 --max-new-tokens 128 --repeats 1; then
        echo "smoke failed; inspect $smoke.evidence" > "$dir/FAILED_SMOKE.txt"
        log "SMOKE_FAILED method=$method"
        upload_method "$method"
        return 0
      fi
      if ! valid_smoke "$method" "$smoke"; then
        echo "smoke report invalid; inspect $smoke.evidence" > "$dir/FAILED_SMOKE.txt"
        log "SMOKE_INVALID method=$method"
        upload_method "$method"
        return 0
      fi
    fi
  fi

  if [[ "$SUITE" == formal || "$SUITE" == all ]]; then
    local formal_dir="$dir/formal" formal="$formal_dir/report.json"
    if ((RESUME)) && [[ -f "$formal" ]] && valid_formal "$method" "$formal"; then
      log "SKIP_VALID_FORMAL method=$method"
    else
      [[ -f "$formal" ]] && archive_invalid "$formal_dir"
      log "RUN_FORMAL method=$method gpu=$gpu"
      if ! run_stage "$method" formal "$gpu" "$formal" --num-prompts 600 --max-new-tokens 1024 --repeats 3; then
        echo "formal failed; inspect $formal.evidence" > "$dir/FAILED_FORMAL.txt"
        log "FORMAL_FAILED method=$method"
        upload_method "$method"
        return 0
      fi
      if ! valid_formal "$method" "$formal"; then
        echo "formal report invalid or resource evidence contaminated; inspect $formal.evidence" > "$dir/FAILED_FORMAL.txt"
        log "FORMAL_INVALID method=$method"
        upload_method "$method"
        return 0
      fi
    fi
  fi
  upload_method "$method"
  log "COMPLETE method=$method gpu=$gpu"
}

log "CAMPAIGN_START id=$CAMPAIGN_ID gpus=$GPUS methods=${SELECTED[*]} suite=$SUITE"
declare -A WORKER_ARGS
for i in "${!SELECTED[@]}"; do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  WORKER_ARGS[$gpu]="${WORKER_ARGS[$gpu]-} ${SELECTED[$i]}"
done
for gpu in "${GPU_LIST[@]}"; do
  [[ -n "${WORKER_ARGS[$gpu]-}" ]] || continue
  # shellcheck disable=SC2086
  (for method in ${WORKER_ARGS[$gpu]}; do run_method "$method" "$gpu"; done) > "$LOG_DIR/gpu${gpu}.log" 2>&1 &
done
wait

"$PYTHON" - "$CAMPAIGN_DIR" > "$CAMPAIGN_DIR/summary.csv" <<'PY'
import csv,glob,json,os,sys
root=sys.argv[1]
fields=["method","status","sessions","turns","repeats","timed_pairs","speedup_sum_ar_over_sd","ci_low","ci_high","ar_seconds","sd_seconds","error"]
rows=[]
manifest=json.load(open(os.path.join(root,"run_manifest.json")))
for method in manifest["methods"]:
    p=os.path.join(root,method,"formal","report.json")
    if os.path.isfile(p):
        d=json.load(open(p)); ci=d.get("session_cluster_CI95") or d.get("bootstrap_ci95") or {}
        if isinstance(ci,(list,tuple)) and len(ci)>=2: low,high=ci[0],ci[1]
        elif isinstance(ci,dict): low,high=ci.get("low"),ci.get("high")
        else: low=high=None
        rows.append({"method":method,"status":d.get("status"),"sessions":d.get("sessions"),"turns":d.get("turns"),"repeats":d.get("repeats"),"timed_pairs":d.get("timed_pairs"),"speedup_sum_ar_over_sd":d.get("speedup_sum_ar_over_sd"),"ci_low":low,"ci_high":high,"ar_seconds":d.get("ar_seconds"),"sd_seconds":d.get("sd_seconds"),"error":d.get("error")})
    else:
        failed=os.path.join(root,method,"FAILED_FORMAL.txt")
        rows.append({"method":method,"status":"failed_or_missing","sessions":None,"turns":None,"repeats":None,"timed_pairs":None,"speedup_sum_ar_over_sd":None,"ci_low":None,"ci_high":None,"ar_seconds":None,"sd_seconds":None,"error":open(failed).read().strip() if os.path.isfile(failed) else "formal report missing"})
w=csv.DictWriter(sys.stdout,fieldnames=fields); w.writeheader(); w.writerows(rows)
PY

if ((PUBLISH)); then
  command -v hf >/dev/null 2>&1 || die "--publish requires the hf CLI"
  hf upload "$HF_REPO_ID" "$CAMPAIGN_DIR/summary.csv" "$HF_PREFIX/$CAMPAIGN_ID/summary.csv" --repo-type model --commit-message "eval: summary ${CAMPAIGN_ID}"
  hf upload "$HF_REPO_ID" "$CAMPAIGN_DIR/run_manifest.json" "$HF_PREFIX/$CAMPAIGN_ID/run_manifest.json" --repo-type model --commit-message "eval: manifest ${CAMPAIGN_ID}"
fi
log "CAMPAIGN_COMPLETE id=$CAMPAIGN_ID results=$CAMPAIGN_DIR"
