#!/usr/bin/env python3
"""Gate a one-update continuation smoke before the formal continuation starts.

    python scripts/a100_qwen3vl8b_200k/continue_smoke_check.py SMOKE_RUN SOURCE_TRAIN_DIR --lr 1e-5

Checks the smoke run (``continue_one.sh <method> <pair> smoke``) against its source row:
one logged update at step source+1 with the constant rate and finite loss / gradient
norm; the objective schedule held at the source's last update (DFly in its second
recipe stage, Domino at ``lambda_base`` 0); AdamW moments continued (every
parameter's optimizer step count is source+1, not 1); finite weights that moved
away from the source's final weights. Prints a JSON report; exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("smoke_run", type=Path)
    parser.add_argument("source_train", type=Path)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    train = args.smoke_run / "train"
    failures: list[str] = []

    source_final = torch.load(args.source_train / "ckpt_final.pt", map_location="cpu", weights_only=False)
    source_step = int(source_final["step"])
    algorithm = source_final["algorithm"]
    rows = [json.loads(line) for line in open(train / "train_log.jsonl", encoding="utf-8") if line.strip()]
    if [row["step"] for row in rows] != [source_step + 1]:
        failures.append(f"logged steps {[row['step'] for row in rows]}, expected [{source_step + 1}]")
    for row in rows:
        if not math.isclose(row["lr"], args.lr, rel_tol=1e-6):
            failures.append(f"step {row['step']} lr {row['lr']}, expected {args.lr}")
        if not (math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"])):
            failures.append(f"step {row['step']} loss {row['loss']} grad_norm {row['grad_norm']}")

    summary = json.loads((train / "train_summary.json").read_text())
    held = summary.get("continuation") or {}
    if held.get("objective_schedule_held_at") != [source_step, source_step]:
        failures.append(f"objective schedule held at {held.get('objective_schedule_held_at')}")
    if algorithm == "dfly" and held.get("recipe_stage") != 1:
        failures.append(f"dfly recipe stage {held.get('recipe_stage')}, expected 1 (second stage)")
    if algorithm == "domino" and held.get("lambda_base") != 0.0:
        failures.append(f"domino lambda_base {held.get('lambda_base')}, expected 0")

    state = torch.load(train / "ckpt_last.pt", map_location="cpu", weights_only=False)
    adam_steps = sorted({int(entry["step"]) for entry in state["optimizer"]["state"].values()})
    if adam_steps != [source_step + 1]:
        failures.append(f"AdamW step counts {adam_steps}, expected [{source_step + 1}]")
    if sorted({float(group["lr"]) for group in state["optimizer"]["param_groups"]}) != [args.lr]:
        failures.append("optimizer param-group rate is not the continuation rate")

    final = torch.load(train / "ckpt_final.pt", map_location="cpu", weights_only=False)
    moved, max_delta = 0, 0.0
    for name, tensor in final["model"].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            failures.append(f"non-finite weights in {name}")
        reference = source_final["model"][name]
        if not torch.equal(tensor, reference):
            moved += 1
            max_delta = max(max_delta, float((tensor.float() - reference.float()).abs().max()))
    if moved == 0:
        failures.append("no weight moved after the update")

    report = {
        "algorithm": algorithm,
        "source_step": source_step,
        "row": rows[0] if rows else None,
        "held": held,
        "adam_steps": adam_steps,
        "tensors_moved": moved,
        "tensors_total": len(final["model"]),
        "max_abs_weight_delta": max_delta,
        "failures": failures,
        "passed": not failures,
    }
    (args.smoke_run / "smoke_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
