#!/usr/bin/env python3
"""Build the result tables of the a100_qwen3vl8b_200k_s42 matrix (contract sections 26-31, 33).

    python scripts/a100_qwen3vl8b_200k/results.py --task "$VELA_TASK"

Reads ``eval/<method>/*.summary.json``, ``eval/ar/*.summary.json``, the formal
train logs and the smoke reports, and writes under ``results/``:

- ``epoch1.csv`` / ``epoch2.csv`` / ``final.csv``: matched-budget rows (updates
  25,000 / 50,000 / 75,000; sequential verification): AL_draft, AL_progress and
  exactness on heldout_test and mmspec (final: every set);
- ``benchmark_results.csv``: every (method, checkpoint, verification, set) summary;
- ``speed_quality.csv``: final checkpoints, per set: task quality (AR vs SD),
  speedup sum/sum and mean of ratios, exactness, verification path;
- ``cost_correctness.csv``: per method: GPU-hours (smoke, training), sec/update,
  peaks, final loss, overfit sanity, resume gate, fidelity across sets, status label.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

QUEUE = ["dflash", "dfly", "dflash2", "dspark", "dflare", "domino", "jetspec"]
LABELS = {"jetspec": "JetSpec-trained block proposal reference", "dflash2": "PARTIAL (experimental objective)"}
STEPS = {"epoch1": "ckpt_step0025000", "epoch2": "ckpt_step0050000", "final": "ckpt_final"}


def _summaries(task: Path) -> list[dict]:
    out = []
    for path in sorted((task / "eval").glob("*/*.summary.json")):
        summary = json.loads(path.read_text())
        summary["method"] = summary.get("method") or ("ar" if path.parent.name == "ar" else path.parent.name)
        summary["checkpoint_name"] = Path(summary["checkpoint"]).stem if summary.get("checkpoint") else None
        out.append(summary)
    return out


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _r(value, digits: int = 4):
    return round(value, digits) if isinstance(value, float) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    task = args.task
    summaries = _summaries(task)
    ar = {summary["set"]: summary for summary in summaries if summary["mode"] == "ar"}
    sd = [summary for summary in summaries if summary["mode"] == "sd"]

    benchmark = [{
        "method": s["method"], "label": LABELS.get(s["method"], ""), "checkpoint": s["checkpoint_name"],
        "verification": s["verification"], "set": s["set"], "turns": s["turns"], "AL_draft": _r(s.get("AL_draft")),
        "AL_progress": _r(s.get("AL_progress")), "acceptance_rate": _r(s.get("acceptance_rate")),
        "rounds": s.get("rounds"), "target_calls": s.get("target_calls"), "drafter_calls": s.get("drafter_calls"),
        "mean_new_tokens": _r(s.get("mean_new_tokens"), 1), "quality": _r(s.get("quality")),
        "ar_quality": _r(ar.get(s["set"], {}).get("quality")), "sequence_exact_rate": _r(s.get("sequence_exact_rate")),
        "stopping_exact_rate": _r(s.get("stopping_exact_rate")), "token_exact": s.get("token_exact"),
        "speedup": _r(s.get("speedup"), 3), "speedup_mean_of_ratios": _r(s.get("speedup_mean_of_ratios"), 3),
        "strict_fallback_rounds": s.get("strict_fallback_rounds"), "wall_seconds_total": _r(s.get("wall_seconds_total"), 1),
    } for s in sd]
    _write(task / "results" / "benchmark_results.csv", benchmark)

    for name, stem in STEPS.items():
        rows = [row for row in benchmark if row["checkpoint"] == stem and row["verification"] == "sequential"]
        if name != "final":
            rows = [row for row in rows if row["set"] in ("heldout_test", "mmspec")]
        _write(task / "results" / f"{name}.csv", rows)

    speed = [{key: row[key] for key in ("method", "label", "set", "verification", "quality", "ar_quality", "speedup",
                                        "speedup_mean_of_ratios", "AL_draft", "AL_progress", "sequence_exact_rate",
                                        "token_exact", "strict_fallback_rounds")}
             for row in benchmark if row["checkpoint"] == "ckpt_final"]
    _write(task / "results" / "speed_quality.csv", speed)

    cost = []
    for method in QUEUE:
        smoke_path = task / "smoke" / method / "smoke_report.json"
        smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        summary_path = task / "runs" / method / "train" / "train_summary.json"
        train = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        finals = [row for row in benchmark if row["method"] == method and row["checkpoint"] == "ckpt_final"
                  and row["verification"] == "sequential"]
        calib = smoke.get("calibration", {})
        wall = train.get("wall_seconds")
        cost.append({
            "method": method, "label": LABELS.get(method, ""),
            "status": "done" if (task / "runs" / method / "train" / "ckpt_final.pt").exists() else "not finished",
            "train_updates": train.get("steps"), "train_wall_hours": _r(wall / 3600, 1) if wall else None,
            "train_gpu_hours": _r(2 * wall / 3600, 1) if wall else None,
            "sec_per_update_calibration": calib.get("sec_per_update_mean"),
            "train_peak_gb_calibration": calib.get("train_peak_gb"), "target_peak_gb_calibration": calib.get("target_peak_gb"),
            "final_loss": train.get("final_loss"), "overfit_sanity_passed": (smoke.get("overfit_sanity") or {}).get("passed"),
            "resume_exact_mismatches": json.dumps((smoke.get("resume") or {}).get("exact_mismatches")),
            "resume_max_loss_rel_diff": (smoke.get("resume") or {}).get("max_loss_rel_diff"),
            "sequential_token_exact_all_sets": all(row["token_exact"] for row in finals) if finals else None,
        })
    _write(task / "results" / "cost_correctness.csv", cost)
    print(f"RESULTS_WRITTEN {len(benchmark)} eval rows")


if __name__ == "__main__":
    main()
