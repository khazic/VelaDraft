#!/usr/bin/env python3
"""Derive a smaller-quota request pool from a frozen ``build_onevision_pool.py`` pool.

    python scripts/a100_qwen3vl8b_200k/derive_pool.py --src SRC_POOL --out OUT_POOL \\
        --quotas general=70000,ocr=60000,chart=30000,reasoning=40000 \\
        --oversample general=1.6,ocr=1.45,chart=2.1,reasoning=1.7 \\
        --media-root /old/pool_media=/new/pool_media

The source pool fixes the dev/test members (frozen first, one row per image,
media-disjoint from train) and a seeded per-group train order (``pool_rank``).
The derived pool keeps every dev/test row unchanged and takes, per group, the
train rows with ``pool_rank < ceil(quota * oversample)`` as regen candidates;
the next ranks become the reserve that replaces rows regen drops. Selection
never reshuffles: a derived pool is a prefix of the source order per group.
Rows are copied verbatim except for the image path prefix (``--media-root``).

``--pilot N_general,N_ocr,...`` additionally writes ``pilot_pool.jsonl``: the
lowest-rank train candidates per group, for the ~1K regen pilot.

``--exclude-ids FILE`` drops train rows (e.g. images shared with an evaluation
set) before the cut: each group still gets ``ceil(quota * oversample)``
candidates, taken from the next ranks. An excluded dev/test row is an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _pairs(text: str, cast) -> dict:
    out = {}
    for item in text.split(","):
        key, value = item.split("=", 1)
        out[key.strip()] = cast(value)
    return out


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _rebase(row: dict, old: str | None, new: str | None) -> dict:
    if old is None:
        return row
    row = dict(row)
    for key in ("image", "images"):
        value = row.get(key)
        if isinstance(value, str) and value.startswith(old):
            row[key] = new + value[len(old) :]
        elif isinstance(value, list):
            row[key] = [new + item[len(old) :] if item.startswith(old) else item for item in value]
    return row


def derive(src: Path, out: Path, quotas: dict, oversample: dict, media_root: tuple | None, pilot: dict | None,
           exclude: set[str] | None = None) -> dict:
    meta = {record["id"]: record for record in _read_jsonl(src / "pool_meta.jsonl")}
    exclude = set(exclude or ())
    held_out = sorted(row_id for row_id in exclude if row_id in meta and meta[row_id]["split"] != "train")
    if held_out:
        raise SystemExit(f"--exclude-ids names dev/test rows {held_out[:5]}; the held-out splits are frozen.")
    # Rank among the group's non-excluded train rows: excluded rows never count.
    by_group: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row_id, record in meta.items():
        if record["split"] == "train" and row_id not in exclude:
            by_group[record["group"]].append((int(record["pool_rank"]), row_id))
    effective_rank = {row_id: rank for rows in by_group.values() for rank, (_, row_id) in enumerate(sorted(rows))}
    groups = {record["group"] for record in meta.values()}
    if set(quotas) != groups or set(oversample) != groups:
        raise SystemExit(f"quota/oversample groups must match the pool's groups {sorted(groups)}.")
    take = {group: math.ceil(quotas[group] * oversample[group]) for group in quotas}
    available = Counter(meta[row_id]["group"] for row_id in effective_rank)
    short = {group: (take[group], available[group]) for group in take if available[group] < take[group]}
    if short:
        raise SystemExit(f"source pool too small for (needed, available): {short}")
    old, new = media_root if media_root else (None, None)

    def role(record: dict) -> str:
        if record["split"] != "train":
            return record["split"]
        if record["id"] in exclude:
            return "excluded"
        return "candidate" if effective_rank[record["id"]] < take[record["group"]] else "reserve"

    out.mkdir(parents=True, exist_ok=True)
    train_rows, held_rows, reserve_rows = [], [], []
    pilot_rows: dict[str, list] = defaultdict(list)
    # Source order: pool.jsonl (train candidates by group/rank, then dev/test),
    # then reserve.jsonl (the source's own reserve, by group/rank).
    for name in ("pool.jsonl", "reserve.jsonl"):
        for row in _read_jsonl(src / name):
            record = meta[row["id"]]
            kind = role(record)
            rebased = _rebase(row, old, new)
            if kind == "candidate":
                train_rows.append(rebased)
                if pilot and effective_rank[record["id"]] < pilot[record["group"]]:
                    pilot_rows[record["group"]].append(rebased)
            elif kind == "reserve":
                reserve_rows.append((record["group"], effective_rank[record["id"]], rebased))
            elif kind != "excluded":
                held_rows.append(rebased)
    reserve_rows.sort(key=lambda item: (list(quotas).index(item[0]), item[1]))

    def write_jsonl(path: Path, rows) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out / "pool.jsonl", train_rows + held_rows)
    write_jsonl(out / "reserve.jsonl", [row for _, _, row in reserve_rows])
    write_jsonl(out / "pool_meta.jsonl", ({**record, "source_role": record.get("role"), "role": role(record)}
                                          for record in meta.values()))
    if pilot:
        write_jsonl(out / "pilot_pool.jsonl", [row for group in quotas for row in pilot_rows[group]])
    counts = Counter((meta[row["id"]]["group"], meta[row["id"]]["split"]) for row in train_rows + held_rows)
    stats = {
        "derived_from": str(src),
        "source_checksums": (src / "checksums.txt").read_text() if (src / "checksums.txt").exists() else None,
        "quotas": quotas,
        "oversample": oversample,
        "train_candidates_per_group": take,
        "rule": "per group, the first ceil(quota * oversample) non-excluded train rows by pool_rank; dev/test unchanged",
        "media_root": {"old": old, "new": new},
        "counts": {f"{group}/{split}": value for (group, split), value in sorted(counts.items())},
        "num_train_candidates": len(train_rows),
        "num_dev": sum(1 for row in held_rows if meta[row["id"]]["split"] == "dev"),
        "num_test": sum(1 for row in held_rows if meta[row["id"]]["split"] == "test"),
        "num_reserve": len(reserve_rows),
        "num_pilot": sum(len(rows) for rows in pilot_rows.values()),
        "num_excluded": sum(1 for row_id in exclude if row_id in meta),
    }
    (out / "pool_stats.json").write_text(json.dumps(stats, indent=2))
    names = ["pool.jsonl", "reserve.jsonl", "pool_meta.jsonl", "pool_stats.json"] + (["pilot_pool.jsonl"] if pilot else [])
    (out / "checksums.txt").write_text("".join(f"{_sha(out / name)}  {name}\n" for name in names))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quotas", required=True, help="group=rows,...")
    parser.add_argument("--oversample", required=True, help="group=factor,... (regen candidates per quota row)")
    parser.add_argument("--media-root", default=None, help="OLD=NEW image path prefix rewrite")
    parser.add_argument("--pilot", default=None, help="group=rows,... lowest-rank candidates for the regen pilot")
    parser.add_argument("--exclude-ids", type=Path, default=None, help="file of train row ids to drop before the cut")
    args = parser.parse_args()
    media_root = tuple(args.media_root.split("=", 1)) if args.media_root else None
    stats = derive(
        args.src, args.out, _pairs(args.quotas, int), _pairs(args.oversample, float), media_root,
        _pairs(args.pilot, int) if args.pilot else None,
        set(args.exclude_ids.read_text().split()) if args.exclude_ids else None,
    )
    print(json.dumps({key: stats[key] for key in ("train_candidates_per_group", "num_train_candidates", "num_dev",
                                                  "num_test", "num_reserve", "num_pilot", "num_excluded")}))
    print("DERIVE_POOL_DONE")


if __name__ == "__main__":
    main()
