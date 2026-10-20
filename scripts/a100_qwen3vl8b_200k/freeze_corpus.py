#!/usr/bin/env python3
"""Freeze the shared corpus and lay out the contract manifests (sections 11, 12 and 33).

    python scripts/a100_qwen3vl8b_200k/freeze_corpus.py --task TASK --pool POOL --corpus TASK/corpus

Checks that the regenerated corpus holds exactly the frozen quotas (200,000
train requests: 70K / 60K / 30K / 40K) and the 512 / 512 dev / test members,
that no media is shared across splits, then writes ``checksums.txt`` in
``corpus/data`` and ``corpus/regen``, makes both read-only, and writes
``shared/{train,dev,test}_manifest.json``, ``shared/corpus_stats.json``,
``shared/checksums.txt`` and ``regen/`` (resolved generation config, links to
the corpus regen files). Prints ``FREEZE_DONE`` or exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from pathlib import Path

QUOTAS = {"general": 70_000, "ocr": 60_000, "chart": 30_000, "reasoning": 40_000}
DEV = TEST = 512
DATA_FILES = ("data.jsonl", "val.jsonl", "test.jsonl", "manifest.json", "meta.json", "stage_manifest.json")
REGEN_FILES = ("target_data.jsonl", "regen_stats.json", "skipped.jsonl", "surplus.jsonl", "stage_manifest.json")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_checksums(directory: Path, names) -> str:
    present = [name for name in names if (directory / name).exists()]
    text = "".join(f"{_sha(directory / name)}  {name}\n" for name in present)
    (directory / "checksums.txt").write_text(text)
    return text


def _read_only(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    meta = {record["id"]: record for record in _jsonl(args.pool / "pool_meta.jsonl")}
    kept = [str(row["id"]) for row in _jsonl(args.corpus / "regen" / "target_data.jsonl")]
    manifest = json.loads((args.corpus / "data" / "manifest.json").read_text())
    failures = []
    per_group = Counter(meta[row_id]["group"] for row_id in kept)
    if len(kept) != sum(QUOTAS.values()) or dict(per_group) != QUOTAS:
        failures.append(f"train corpus has {len(kept)} rows {dict(per_group)}, want {QUOTAS}")
    if len(set(kept)) != len(kept):
        failures.append("duplicate row ids in target_data.jsonl")
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    kept_set = set(kept)
    for row in manifest["rows"]:
        info = meta.get(str(row["id"]), {})
        if row["split"] == "train" and str(row["id"]) not in kept_set:
            continue  # candidate regen dropped or past its quota
        by_split[row["split"]].append({
            "id": row["id"], "group": info.get("group"), "subset": info.get("subset"),
            "source_file": info.get("source_file"), "source_row": info.get("source_row"),
            "pool_rank": info.get("pool_rank"),
            "image_sha256": row.get("image_sha256") or (row.get("image_sha256_list") or [None])[0],
        })
    if len(by_split["val"]) != DEV or len(by_split["test"]) != TEST:
        failures.append(f"dev/test have {len(by_split['val'])}/{len(by_split['test'])} rows, want {DEV}/{TEST}")
    media = {split: {row["image_sha256"] for row in rows} for split, rows in by_split.items()}
    overlap = {"train_dev": len(media["train"] & media["val"]), "train_test": len(media["train"] & media["test"]),
               "dev_test": len(media["val"] & media["test"])}
    if any(overlap.values()):
        failures.append(f"media shared across splits: {overlap}")
    if failures:
        print("\n".join(failures))
        sys.exit(1)

    shared = args.task / "shared"
    regen = args.task / "regen"
    shared.mkdir(parents=True, exist_ok=True)
    regen.mkdir(parents=True, exist_ok=True)
    names = {"train": "train_manifest.json", "val": "dev_manifest.json", "test": "test_manifest.json"}
    for split, rows in by_split.items():
        (shared / names[split]).write_text(json.dumps({"split": split, "count": len(rows), "rows": rows}))
    regen_stats = json.loads((args.corpus / "regen" / "regen_stats.json").read_text())
    stats = {
        split: {"valid_requests": len(rows), "unique_media": len(media[split]),
                "per_group": dict(Counter(row["group"] for row in rows)),
                "per_subset": dict(Counter(row["subset"] for row in rows))}
        for split, rows in by_split.items()
    }
    stats["media_overlap"] = overlap
    stats["pool"] = json.loads((args.pool / "pool_stats.json").read_text())
    stats["regen"] = {key: regen_stats.get(key) for key in (
        "num_input", "num_valid", "num_kept", "num_surplus", "skip_reasons", "kept_per_group", "surplus_per_group",
        "answer_token_histogram", "prompt_token_histogram")}
    (shared / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    data_sums = _write_checksums(args.corpus / "data", DATA_FILES)
    regen_sums = _write_checksums(args.corpus / "regen", REGEN_FILES)
    _write_checksums(shared, (*names.values(), "corpus_stats.json"))
    (regen / "resolved_generation_config.json").write_text(json.dumps(regen_stats.get("generation_protocol"), indent=2))
    for name in ("target_data.jsonl", "regen_stats.json", "checksums.txt"):
        link = regen / name
        if not link.exists():
            os.symlink(args.corpus / "regen" / name, link)
    for directory, files in ((args.corpus / "data", DATA_FILES), (args.corpus / "regen", REGEN_FILES)):
        for name in (*files, "checksums.txt"):
            if (directory / name).exists():
                _read_only(directory / name)
    print(json.dumps({split: stats[split]["valid_requests"] for split in by_split}), overlap)
    print(data_sums + regen_sums, end="")
    print("FREEZE_DONE")


if __name__ == "__main__":
    main()
