#!/usr/bin/env python3
"""Write REPORT.md and the HF bundle templates of the a100_qwen3vl8b_200k_s42 matrix (contract sections 35, 42).

    python scripts/a100_qwen3vl8b_200k/report.py --task "$VELA_TASK" --repo-id khazic/VelaDraft-Qwen3VL-8B-200K

Inputs: preflight records, shared/corpus_stats.json, the regen reports, the smoke
reports, results/*.csv (results.py), status.json, HF_UPLOAD_STATUS.md and
codebase_integration.md. Outputs: ``REPORT.md`` (internal, may name internal
paths) and ``hf_templates/{README.md, LICENSE_OR_NOTICE.md,
experiment_manifest.json, preflight_public.json}`` (public, no internal paths;
build_hf_bundle.py scans them again).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

QUEUE = ["dflash", "dspark", "domino", "jetspec", "dflash2", "dflare", "dfly"]
ROWS = {"dflash": "M1 DFlash", "dspark": "M2 DSpark", "domino": "M3 Domino", "jetspec": "M4 JetSpec",
        "dflash2": "M5 DFlash2", "dflare": "M6 DFlare", "dfly": "M7 DFly"}
MATURITY = {
    "dflash": "validated", "dspark": "validated (upstream_dspark recipe)", "domino": "validated",
    "jetspec": "trained with forward KL; evaluated as a block proposal reference (tree runtime not ported)",
    "dflash2": "PARTIAL: experimental objective reconstruction (upstream objective undisclosed)",
    "dflare": "validated", "dfly": "validated (upstream_two_stage recipe)",
}


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _table(rows: list[dict], columns: list[str]) -> list[str]:
    if not rows:
        return ["(not available yet)"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines += ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    task = args.task
    stats = _json(task / "shared" / "corpus_stats.json") or {}
    regen = _json(task / "corpus" / "regen_report.json") or {}
    pilot = _json(task / "pilot" / "regen_report.json") or {}
    status = {row["method"]: row for row in (_json(task / "status.json") or [])}
    smokes = {m: _json(task / "smoke" / m / "smoke_report.json") or {} for m in QUEUE}
    final = _csv(task / "results" / "final.csv")
    epoch1, epoch2 = _csv(task / "results" / "epoch1.csv"), _csv(task / "results" / "epoch2.csv")
    speed = _csv(task / "results" / "speed_quality.csv")
    cost = _csv(task / "results" / "cost_correctness.csv")
    hf = (task / "HF_UPLOAD_STATUS.md").read_text() if (task / "HF_UPLOAD_STATUS.md").exists() else "Status: NOT_STARTED"
    eval_cols = ["method", "set", "AL_draft", "AL_progress", "sequence_exact_rate", "quality", "ar_quality"]

    sections = [
        "# REPORT: a100_qwen3vl8b_200k_s42", "",
        "## 1. Experiment contract", "",
        "Qwen/Qwen3-VL-8B-Instruct (BF16) x seven drafters, 200,000 OneVision requests x 3 epochs (75,000 updates at "
        "global batch 8), seed 42, online frozen-teacher features, split-device workers (drafter cuda:0, target "
        "cuda:1). Full mapping and every decision: codebase_integration.md.", "",
        "## 2-3. Hardware and environment", "",
        "See preflight/ (nvidia-smi, topology, GPU UUIDs, versions) and the per-node bootstrap logs.", "",
        "## 4. Dataset provenance", "",
        f"Train / dev / test requests: {json.dumps({k: (stats.get(k) or {}).get('valid_requests') for k in ('train', 'val', 'test')})}; "
        f"unique media: {json.dumps({k: (stats.get(k) or {}).get('unique_media') for k in ('train', 'val', 'test')})}; "
        f"media overlap across splits: {json.dumps(stats.get('media_overlap'))}.", "",
        "## 5. Regen statistics", "",
        f"Pilot: valid rate {pilot.get('valid_rate')}, keep per group "
        f"{json.dumps({g: v['keep_rate'] for g, v in (pilot.get('per_group') or {}).items()})}.",
        f"Corpus: generated {regen.get('generated')}, kept {regen.get('kept')}, skip reasons "
        f"{json.dumps(regen.get('skip_reasons'))}, answers {json.dumps(regen.get('answer_tokens_all_generated'))}.", "",
        "## 6. Method profiles", "",
        *[f"- {ROWS[m]}: {MATURITY[m]}" for m in QUEUE], "",
        "## 7. Training curves", "",
        "Per-update logs: runs/<method>/train/train_log.jsonl (loss, accuracy, accept_len, lr, grad_norm, stage "
        "timing, peaks).", "",
        "## 8. Matched-budget tables (sequential verification)", "",
        "### Epoch 1 (25,000 updates)", "", *_table(epoch1, eval_cols), "",
        "### Epoch 2 (50,000 updates)", "", *_table(epoch2, eval_cols), "",
        "### Final (75,000 updates)", "", *_table(final, eval_cols), "",
        "## 9-13. Quality, acceptance, fidelity, latency, speedup (final checkpoints)", "",
        *_table(speed, ["method", "set", "verification", "quality", "ar_quality", "speedup", "speedup_mean_of_ratios",
                        "AL_draft", "AL_progress", "sequence_exact_rate", "token_exact"]), "",
        "## 14. GPU-hours and cost", "",
        *_table(cost, ["method", "status", "train_gpu_hours", "sec_per_update_calibration", "train_peak_gb_calibration",
                       "target_peak_gb_calibration", "final_loss", "sequential_token_exact_all_sets"]), "",
        "## 15. Failed / blocked rows", "",
        *[f"- {m}: {status[m]['state']} ({status[m]['failure']})" for m in QUEUE
          if m in status and status[m]["state"] not in ("done", "done (evaluated)")], "",
        "## 16. Limitations", "",
        "No DDP / TP and no producer-consumer overlap; JetSpec is evaluated with the block-parallel decoder; the "
        "DFlash2 objective is an experimental reconstruction; parallel verification is reported as experimental "
        "unless its exactness gate passed; latency is single-GPU batch 1.", "",
        "## 17. Hugging Face artifact", "", hf, "",
        "## Smoke gates", "",
        *[f"- {m}: overfit passed={(smokes[m].get('overfit_sanity') or {}).get('passed')}, resume mismatches="
          f"{json.dumps((smokes[m].get('resume') or {}).get('exact_mismatches'))}, max loss drift="
          f"{(smokes[m].get('resume') or {}).get('max_loss_rel_diff')}, sec/update="
          f"{(smokes[m].get('calibration') or {}).get('sec_per_update_mean')}" for m in QUEUE],
    ]
    (task / "REPORT.md").write_text("\n".join(sections) + "\n")

    templates = task / "hf_templates"
    templates.mkdir(parents=True, exist_ok=True)
    readme = [
        "---", "license: apache-2.0", "base_model: Qwen/Qwen3-VL-8B-Instruct", "tags:", "- speculative-decoding",
        "- vision-language", "- drafter", "---", "",
        "# VelaDraft: Qwen3-VL-8B-Instruct, 200K x 3 epochs, seven speculative drafters", "",
        "Medium-scale benchmark of seven block drafters for Qwen/Qwen3-VL-8B-Instruct trained under one matched "
        "protocol with [VelaDraft](https://github.com/LUOseu/VelaDraft): 200,000 LLaVA-OneVision requests, 3 epochs "
        "(75,000 updates, global batch 8), seed 42, online frozen-teacher features (teacher hidden states extracted "
        "live from the frozen target every epoch), single A100-class node training.", "",
        "## Methods", "", "| row | method | maturity |", "| --- | --- | --- |",
        *[f"| {ROWS[m].split()[0]} | {ROWS[m].split()[1]} | {MATURITY[m]} |" for m in QUEUE], "",
        "## Training", "",
        "3-layer dense Qwen3 drafter (hidden 4096, intermediate 12288, 32/8 heads, head_dim 128), block 8, 512 anchors, "
        "teacher layers [1, 17, 33] + final hidden, sequence length 4096, AdamW (0.9, 0.95), lr 1e-4 cosine to 1e-5 "
        "with 3,000 warmup updates, BF16 drafter with fp32 masters. Exact budget and knobs: "
        "`provenance/training_plan.json`, `configs/`.", "",
        "## Evaluation", "",
        "MMSpec (600 sessions, by topic), TextVQA (256 fixed validation), DocVQA (256 fixed validation), ChartQA "
        "(256 fixed test), a 32-session long-response diagnostic (2048 tokens), held-out dev512 + test512. Batch 1, "
        "greedy, target and drafter on one GPU; sequential verification is the exactness reference; parallel "
        "verification is reported as experimental unless its exactness gate passes. Protocol: "
        "`evaluation/protocol.json`.", "",
        "## Results", "", "See `results/final.csv`, `results/benchmark_results.csv`, `results/speed_quality.csv`.", "",
        *_table(speed[:28], ["method", "set", "verification", "AL_draft", "AL_progress", "speedup", "token_exact"]), "",
        "## Fidelity caveat", "",
        "Parallel / strict-parallel verification can flip BF16 near-ties; those numbers are an experimental throughput "
        "diagnostic and are not claimed lossless unless `token_exact` is true for that row.", "",
        "## Data provenance", "",
        "Training requests come from lmms-lab/LLaVA-OneVision-Data (revision b31b1dad33f0f916b7ccd9cf8eaec32df98c6914); "
        "answers were regenerated by the target (sampled, T 0.7, top_p 0.8, top_k 20). This repository ships only ids, "
        "source rows and image hashes (`provenance/dataset_manifest_public.json`), not images or regenerated text.", "",
        "## Code and reproduction", "",
        "Code: see `provenance/git_commit.txt`. Minimal reproduction: build the pool and corpus with "
        "`scripts/a100_qwen3vl8b_200k/corpus.sh`, then `bootstrap.sh` and `run_all.sh` from the same directory.",
    ]
    (templates / "README.md").write_text("\n".join(readme) + "\n")
    (templates / "LICENSE_OR_NOTICE.md").write_text(
        "# License and notice\n\nDrafter weights and code-derived artifacts follow the VelaDraft repository license "
        "(Apache-2.0). The target model Qwen/Qwen3-VL-8B-Instruct is Apache-2.0. Training data is not redistributed: "
        "LLaVA-OneVision-Data subsets carry their own licenses; MMSpec, TextVQA, DocVQA and ChartQA are referenced by "
        "id only.\n")
    (templates / "experiment_manifest.json").write_text(json.dumps({
        "experiment": "a100_qwen3vl8b_200k_s42", "target": {"repo": "Qwen/Qwen3-VL-8B-Instruct",
                                                            "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"},
        "methods": QUEUE, "train_requests": 200000, "epochs": 3, "updates": 75000, "seed": 42,
        "hf_repo": args.repo_id,
    }, indent=2))
    contract = _json(task / "runs" / "dflash" / "contract.json") or {}
    (templates / "preflight_public.json").write_text(json.dumps({
        "target_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "contract": {key: contract.get(key) for key in ("text", "layers", "rope", "mask", "image", "tokens", "head")},
        "environment": (task / "preflight" / "env.txt").read_text().splitlines()
        if (task / "preflight" / "env.txt").exists() else [],
    }, indent=2))
    print("REPORT_WRITTEN")


if __name__ == "__main__":
    main()
