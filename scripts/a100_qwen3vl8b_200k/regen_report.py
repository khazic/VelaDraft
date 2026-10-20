#!/usr/bin/env python3
"""Regen statistics of a finished regen run (contract section 12.1 pilot report, also used for the full corpus).

    python scripts/a100_qwen3vl8b_200k/regen_report.py --run RUN --meta POOL/pool_meta.jsonl [--gpu-log FILE]

Reads ``regen_vllm/{requests,outputs_*}.jsonl`` (raw vLLM outputs, every
attempted row), ``regen/{target_data,skipped}.jsonl`` (finalize's verdict) and
the pool meta (row -> group). Writes ``<run>/regen_report.json`` with valid
rate, answer-length percentiles, cap-hit and natural-EOS rates, empty or
invalid answers, per-group lengths and keep rates, generation throughput and
(optionally) the peak GPU memory sampled during generation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(q: float) -> int:
        return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]

    return {"n": len(ordered), "mean": round(sum(ordered) / len(ordered), 1), "p50": at(0.5), "p90": at(0.9),
            "p95": at(0.95), "p99": at(0.99), "max": ordered[-1]}


def _throughput(run: Path, outputs: list[dict]) -> dict:
    """Rows/s and generated tokens/s per shard from the generate logs' first and last progress lines."""
    stamp = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]")
    shards = {}
    for log in sorted(glob.glob(str(run / "logs" / "generate_*.log"))):
        rates = [float(match.group(1)) for match in re.finditer(r"\(([\d.]+) rows/s\)", Path(log).read_text())]
        if rates:
            shards[Path(log).stem] = rates[-1]
    regen_log = run / "logs" / "regen.log"
    wall = None
    if regen_log.exists():
        times = {}
        for line in regen_log.read_text().splitlines():
            match = stamp.match(line)
            if match and ("requests ready" in line or "shards done" in line):
                times["start" if "requests ready" in line else "end"] = datetime.fromisoformat(match.group(1))
        if {"start", "end"} <= set(times):
            wall = (times["end"] - times["start"]).total_seconds()
    tokens = sum(len(row["answer_ids"]) for row in outputs)
    return {"rows_per_s_per_shard": shards, "generate_wall_seconds": wall, "generated_tokens": tokens,
            "generated_tokens_per_s": round(tokens / wall, 1) if wall else None,
            "rows_per_s": round(len(outputs) / wall, 2) if wall else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, default=None, help="nvidia-smi csv sampled during generate")
    args = parser.parse_args()
    run = args.run
    group_of = {row["id"]: row["group"] for row in _jsonl(args.meta)}
    sampling = json.loads((run / "regen_vllm" / "sampling.json").read_text())
    requests = {row["id"]: row for row in _jsonl(run / "regen_vllm" / "requests.jsonl")}
    outputs = [row for path in sorted(glob.glob(str(run / "regen_vllm" / "outputs_*.jsonl"))) for row in _jsonl(Path(path))]
    kept = list(_jsonl(run / "regen" / "target_data.jsonl"))
    skipped = list(_jsonl(run / "regen" / "skipped.jsonl")) if (run / "regen" / "skipped.jsonl").exists() else []
    stop_ids = set(sampling["stop_token_ids"])

    finish = Counter(row["finish_reason"] for row in outputs)
    natural_eos = sum(1 for row in outputs if row["answer_ids"] and row["answer_ids"][-1] in stop_ids)
    answer_len = [len(row["answer_ids"]) for row in outputs]
    per_group_len, per_group_kept, per_group_attempted = defaultdict(list), Counter(), Counter()
    for row in outputs:
        per_group_attempted[group_of.get(row["id"], "?")] += 1
    for record in kept:
        group = group_of.get(str(record["id"]), "?")
        per_group_kept[group] += 1
        regen = record["regen"]
        ids = regen["turns"][-1]["answer_ids"] if "turns" in regen else regen["answer_ids"]
        per_group_len[group].append(len(ids))
    report = {
        "run": str(run),
        "sampling": {key: sampling.get(key) for key in ("temperature", "top_p", "top_k", "presence_penalty",
                                                         "repetition_penalty", "max_new_tokens", "seed",
                                                         "stop_token_ids", "image_min_pixels", "image_max_pixels")},
        "rows_in_pool_slice": len(requests),
        "skipped_before_generate": dict(Counter(row.get("skip") for row in requests.values() if row.get("skip"))),
        "generated": len(outputs),
        "prompt_match_rate": round(sum(row["prompt_match"] for row in outputs) / max(1, len(outputs)), 4),
        "finish_reason": dict(finish),
        "cap_hit_rate": round(finish.get("length", 0) / max(1, len(outputs)), 4),
        "natural_eos_rate": round(natural_eos / max(1, len(outputs)), 4),
        "empty_answers": sum(1 for row in outputs if not row["answer_ids"]),
        "answer_tokens_all_generated": _percentiles(answer_len),
        "kept": len(kept),
        "valid_rate": round(len(kept) / max(1, len(requests)), 4),
        "skip_reasons": dict(Counter(row.get("reason") for row in skipped)),
        "per_group": {
            group: {"attempted": per_group_attempted[group], "kept": per_group_kept[group],
                    "keep_rate": round(per_group_kept[group] / max(1, per_group_attempted[group]), 4),
                    "answer_tokens": _percentiles(per_group_len[group])}
            for group in sorted(per_group_attempted)
        },
        "throughput": _throughput(run, outputs),
    }
    if args.gpu_log and args.gpu_log.exists():
        peaks: dict[str, int] = {}
        with open(args.gpu_log, newline="") as handle:
            for row in csv.reader(handle):
                if len(row) >= 3 and row[1].strip().isdigit():
                    peaks[row[0].strip()] = max(peaks.get(row[0].strip(), 0), int(row[1]))
        report["peak_gpu_memory_mib"] = peaks
    (run / "regen_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("REGEN_REPORT_DONE")


if __name__ == "__main__":
    main()
