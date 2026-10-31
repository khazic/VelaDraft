#!/usr/bin/env python3
"""Status and results of the Qwen3-VL-8B 200K continuation (seven rows).

    python scripts/a100_qwen3vl8b_200k/continue_report.py --task TASK --cont CONT

Per method: smoke gate, training progress (added / cumulative updates, last loss, AL,
gradient norm, non-finite rows), and for each evaluated checkpoint (orig_final, cont_ep1,
cont_ep2) the dev validation objective plus heldout_dev speculative decoding (AL, speedup
over AR, exactness against AR). The best checkpoint is the one with the lowest dev
objective among those evaluated; ties and missing results are stated, never filled in.
Writes CONT/REPORT.md and CONT/results.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METHODS = ("dflash2", "dfly", "dflash", "dspark", "dflare", "domino", "jetspec")
LABELS = ("orig_final", "cont_ep1", "cont_ep2")
SOURCE_STEPS = 75000
STEPS_PER_EPOCH = 25000


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def train_progress(run: Path) -> dict:
    log_path = run / "train" / "train_log.jsonl"
    if not log_path.exists():
        return {"state": "not started"}
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if not rows:
        return {"state": "starting"}
    last = rows[-1]
    window = rows[-200:]
    non_finite = [row["step"] for row in rows if not all(math.isfinite(row.get(key, 0.0)) for key in ("loss", "grad_norm"))]
    done = (run / "train" / "train_summary.json").exists()
    return {
        "state": "done" if done else "training",
        "step": last["step"],
        "added_steps": last["step"] - SOURCE_STEPS,
        "lr": last["lr"],
        "loss_last200": sum(row["loss"] for row in window) / len(window),
        "accept_len_last200": sum(row["accept_len"] for row in window) / len(window),
        "grad_norm_last200": sum(row["grad_norm"] for row in window) / len(window),
        "grad_norm_max": max(row["grad_norm"] for row in rows),
        "non_finite_steps": non_finite[:20],
    }


def checkpoint_results(cont: Path, method: str, label: str) -> dict:
    out = cont / "eval" / method / label
    result: dict = {}
    objective = read_json(out / "objective.json")
    if objective:
        result["objective"] = {
            key: objective.get(key)
            for key in ("objective_mean", "objective_weighted", "accuracy", "accept_len", "rows_with_target_answer",
                        "checkpoint_sha256", "recipe_stage", "lambda_base")
        }
    state = read_json(out / "objective.train_state.json")
    if state and objective:
        result["step0_equivalence"] = {
            "train_state_objective": state.get("objective_mean"),
            "final_objective": objective.get("objective_mean"),
            "identical": state.get("objective_mean") == objective.get("objective_mean")
            and state.get("accept_len") == objective.get("accept_len"),
        }
    for verification in ("sequential", "strict_parallel"):
        summaries = sorted(out.glob(f"heldout_dev.*.{verification}.summary.json"))
        if summaries:
            summary = read_json(summaries[0]) or {}
            result[verification] = {
                key: summary.get(key)
                for key in ("turns", "AL_draft", "AL_progress", "acceptance_rate", "speedup", "speedup_mean_of_ratios",
                            "sequence_exact_rate", "token_exact", "stopping_exact_rate", "quality", "device")
            }
    return result


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=Path, required=True, help="source matrix task root")
    parser.add_argument("--cont", type=Path, required=True, help="continuation task root")
    args = parser.parse_args()
    results = {}
    for method in METHODS:
        run = args.cont / "runs" / method
        entry = {
            "smoke": read_json(args.cont / "smoke" / method / "smoke_check.json"),
            "train": train_progress(run),
            "checkpoints": {label: checkpoint_results(args.cont, method, label) for label in LABELS},
        }
        scored = {
            label: data["objective"]["objective_mean"]
            for label, data in entry["checkpoints"].items()
            if data.get("objective", {}).get("objective_mean") is not None
        }
        entry["best_by_dev_objective"] = min(scored, key=scored.get) if scored else None
        entry["evaluated"] = sorted(scored)
        results[method] = entry
    (args.cont / "results.json").write_text(json.dumps(results, indent=2))

    lines = ["# Qwen3-VL-8B 200K continuation", ""]
    lines += ["## Training", "", "| method | smoke | state | cumulative step | added | lr | loss (last 200) | AL (last 200) | grad norm (last 200 / max) | non-finite |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for method, entry in results.items():
        train, smoke = entry["train"], entry["smoke"]
        lines.append(
            f"| {method} | {fmt(smoke.get('passed')) if smoke else '-'} | {train.get('state')} | {fmt(train.get('step'))} "
            f"| {fmt(train.get('added_steps'))} | {fmt(train.get('lr'), 6)} | {fmt(train.get('loss_last200'))} "
            f"| {fmt(train.get('accept_len_last200'), 3)} | {fmt(train.get('grad_norm_last200'), 3)} / {fmt(train.get('grad_norm_max'), 3)} "
            f"| {len(train.get('non_finite_steps') or [])} |"
        )
    lines += ["", "## Evaluation (dev objective on the frozen split; heldout_dev speculative decoding, greedy, batch 1)", "",
              "| method | checkpoint | dev objective | dev AL | SD AL (seq) | exact vs AR (seq) | SD AL (strict) | speedup (strict) | exact vs AR (strict) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for method, entry in results.items():
        for label, data in entry["checkpoints"].items():
            objective = data.get("objective", {})
            sequential = data.get("sequential", {})
            strict = data.get("strict_parallel", {})
            mark = " (best)" if entry["best_by_dev_objective"] == label else ""
            lines.append(
                f"| {method} | {label}{mark} | {fmt(objective.get('objective_mean'))} | {fmt(objective.get('accept_len'), 3)} "
                f"| {fmt(sequential.get('AL_progress'), 3)} | {fmt(sequential.get('sequence_exact_rate'), 3)} "
                f"| {fmt(strict.get('AL_progress'), 3)} | {fmt(strict.get('speedup'), 3)} | {fmt(strict.get('sequence_exact_rate'), 3)} |"
            )
    lines += ["", "## Step-0 equivalence (source train state vs source final, dev objective)", ""]
    for method, entry in results.items():
        check = entry["checkpoints"]["orig_final"].get("step0_equivalence")
        lines.append(f"- {method}: {fmt(check.get('identical')) if check else 'pending'}"
                     + (f" ({check['final_objective']} vs {check['train_state_objective']})" if check else ""))
    (args.cont / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
