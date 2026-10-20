#!/usr/bin/env python3
"""Exact-resume gate and 100-update calibration of one method smoke (contract sections 22-24).

    python scripts/a100_qwen3vl8b_200k/resume_gate.py --method dflash --run-a A --run-b B --steps 40 \\
        --calib-from 11 --gpu-samples gpu_samples.csv --out smoke_report.json

Run A trained uninterrupted; run B was killed after an early resumable
checkpoint and resumed. Update by update (1..steps) the step number, learning
rate, sample-stream digest (``batch_digest``) and recipe stage must be
identical; losses may differ only by BF16 non-determinism (the bound is
recorded). Run A's updates from ``--calib-from`` on are the calibration window:
seconds per update and per stage, train/target peaks, CPU RSS, GPU utilization
(from ``--gpu-samples``, ``epoch, index, uuid, util, mem``), checkpoint sizes.
Prints ``SMOKE_PASS <method>`` or ``SMOKE_FAIL <method> <reasons>``.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

#: Losses of the resumed run must stay within this relative bound of run A.
LOSS_REL_TOL = 5e-2


def _rows(run: Path) -> dict[int, dict]:
    rows = {}
    with open(run / "train" / "train_log.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[int(row["step"])] = row
    return rows


def _gpu_window(path: Path | None, window: tuple[float, float] | None) -> dict:
    if path is None or not path.exists():
        return {}
    util: dict[str, list[float]] = {}
    mem: dict[str, list[float]] = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            stamp, index = float(row[0]), row[1].strip()
            if window and not window[0] <= stamp <= window[1]:
                continue
            util.setdefault(index, []).append(float(row[3]))
            mem.setdefault(index, []).append(float(row[4]))
    return {index: {"util_mean": round(statistics.mean(values), 1), "util_p10": sorted(values)[len(values) // 10],
                    "mem_max_mib": max(mem[index]), "samples": len(values)} for index, values in util.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", required=True)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--calib-from", type=int, default=11)
    parser.add_argument("--gpu-samples", type=Path, default=None)
    parser.add_argument("--save-every", type=int, default=8,
                        help="the smoke's checkpoint cadence: the update after each save also carries the save's wall")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = _rows(args.run_a), _rows(args.run_b)
    failures = []
    exact = {"step": 0, "lr": 0, "batch_digest": 0, "recipe_stage": 0}
    loss_rel = []
    for step in range(1, args.steps + 1):
        if step not in a or step not in b:
            failures.append(f"update {step} missing (a={step in a}, b={step in b})")
            continue
        left, right = a[step], b[step]
        for key in exact:
            if left.get(key) != right.get(key):
                exact[key] += 1
        loss_rel.append(abs(left["loss"] - right["loss"]) / max(1e-9, abs(left["loss"])))
    failures += [f"{count} update(s) differ in {key}" for key, count in exact.items() if count]
    if loss_rel and max(loss_rel) > LOSS_REL_TOL:
        failures.append(f"loss differs by {max(loss_rel):.3g} (> {LOSS_REL_TOL})")
    if not all("batch_digest" in row for row in a.values()):
        failures.append("run A logged no batch digests (train.stage_timing off?)")

    overfit = json.loads((args.run_a / "train" / "overfit_sanity.json").read_text())
    summary_a = json.loads((args.run_a / "train" / "train_summary.json").read_text())
    calib = [row for step, row in sorted(a.items()) if step >= args.calib_from]
    # A smoke checkpoints every few updates (a formal row every 1,250): the state write
    # lands in the next update's loader time, so those updates are left out of the rates.
    steady = [row for row in calib if (int(row["step"]) - 1) % args.save_every != 0] or calib
    stages = sorted({key for row in steady for key in row if key.startswith("time_")})
    timing = {key: round(statistics.mean(row.get(key, 0.0) for row in steady), 4) for key in stages}
    step_times = sorted(row["time_step"] for row in steady if "time_step" in row)
    with_saves = [row["time_step"] for row in calib if "time_step" in row]
    window = None
    window_file = args.run_a.parent / "run_a.window"
    if window_file.exists():
        start, end = (float(value) for value in window_file.read_text().split())
        window = (start, end)
    train_dir_b = args.run_b / "train"
    sizes = {name: path.stat().st_size for name in ("ckpt_last.pt", "ckpt_init.pt")
             if (path := train_dir_b / name).exists()}
    report = {
        "method": args.method,
        "resume": {
            "compared_updates": args.steps, "killed_after_checkpoint": True, "exact_mismatches": exact,
            "max_loss_rel_diff": max(loss_rel) if loss_rel else None, "loss_rel_tolerance": LOSS_REL_TOL,
            "bitwise_loss_equal_updates": sum(1 for value in loss_rel if value == 0.0),
        },
        "overfit_sanity": overfit,
        "calibration": {
            "updates": len(calib), "from_update": args.calib_from, "steady_updates": len(steady),
            "excluded": f"the {len(calib) - len(steady)} update(s) right after a checkpoint write (save every {args.save_every})",
            "sec_per_update_mean_including_saves": round(statistics.mean(with_saves), 3) if with_saves else None,
            "sec_per_update_mean": round(statistics.mean(step_times), 3) if step_times else None,
            "sec_per_update_p50": step_times[len(step_times) // 2] if step_times else None,
            "sec_per_update_p90": step_times[int(len(step_times) * 0.9)] if step_times else None,
            "stage_seconds_mean": timing,
            "train_peak_gb": max((row.get("train_peak_gb", 0.0) for row in calib), default=None),
            "target_peak_gb": max((row.get("target_peak_gb", 0.0) for row in calib), default=None),
            "cpu_rss_gb_max": max((row.get("cpu_rss_gb", 0.0) for row in calib), default=None),
            "h2d_bytes_per_update": round(statistics.mean(row.get("h2d_bytes", 0) for row in calib)),
            "d2h_bytes_per_update": round(statistics.mean(row.get("d2h_bytes", 0) for row in calib)),
            "gpu": _gpu_window(args.gpu_samples, window),
            "checkpoint_bytes": sizes,
            "summary_peak_gpu_mem_gb": summary_a.get("peak_gpu_mem_gb"),
        },
    }
    if step_times:
        per_update = statistics.mean(step_times)
        report["calibration"]["eta_75000_updates_hours"] = round(per_update * 75000 / 3600, 1)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"SMOKE_PASS {args.method}" if not failures else f"SMOKE_FAIL {args.method} {'; '.join(failures)}")


if __name__ == "__main__":
    main()
