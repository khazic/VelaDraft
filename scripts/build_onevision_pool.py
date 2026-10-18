#!/usr/bin/env python3
"""Build a grouped, deduplicated LLaVA-OneVision-Data request pool for VelaDraft.

    python scripts/build_onevision_pool.py \
        --root /data/LLaVA-OneVision-Data --groups configs/a100_qwen36/onevision_groups.yaml \
        --out /data/pool --media-dir /dev/shm/vela/pool_media \
        --exclude-media eval_media_sha256.txt

Per coverage group (see the groups YAML) the script draws candidate rows from
the group's parquet subsets (weights ``rows ** subset_weight_power``), keeps
rows with exactly one decodable image and a human/assistant conversation,
drops duplicates by (image bytes, normalized request) and every image listed in
``--exclude-media`` (evaluation media), then freezes the splits in this order:

1. test, then dev: stratified by group quota, one row per image, so the
   frozen evaluation members never share media with training;
2. train candidates: every other row whose image is not in dev/test, in a
   seeded per-group order recorded as ``pool_rank``. The first
   ``ceil(quota * oversample)`` per group are the regen candidates; the rest are
   written as a reserve so overflow (rows regen drops) is replaced from the
   same ordered pool instead of shrinking the corpus.

Outputs: ``pool.jsonl`` (train candidates + dev + test, ``split`` field for
``dataset.split_field``), ``reserve.jsonl``, ``pool_meta.jsonl`` (id -> group,
subset, pool_rank, source row), ``pool_stats.json`` and ``checksums.txt``.
Images are written once per content hash under ``--media-dir``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

_HUMAN = {"human", "user"}
_GPT = {"gpt", "assistant"}
_EXT = {"JPEG": ".jpg", "PNG": ".png", "GIF": ".gif", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tif"}
#: Qwen-VL processors refuse aspect ratios above 200 (smart_resize).
_MAX_ASPECT = 200.0


def dhash(image) -> str:
    """64-bit difference hash: catches re-encoded / resized copies of an image."""
    from PIL import Image

    small = image.convert("L").resize((9, 8), Image.BILINEAR)
    pixels = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return f"{bits:016x}"


def _subset_files(root: Path, subset: str) -> list[str]:
    return sorted(glob.glob(str(root / glob.escape(subset) / "**" / "*.parquet"), recursive=True))


def _subset_rows(files: list[str]) -> list[int]:
    import pyarrow.parquet as pq

    return [pq.ParquetFile(path).metadata.num_rows for path in files]


def _allocate(total: int, sizes: dict[str, int], power: float) -> dict[str, int]:
    """Split ``total`` draws over subsets by ``size ** power``, capped at each size (water-filling)."""
    alloc = {name: 0 for name in sizes}
    open_names = [name for name, size in sizes.items() if size > 0]
    remaining = min(total, sum(sizes.values()))
    while remaining > 0 and open_names:
        weights = {name: sizes[name] ** power for name in open_names}
        norm = sum(weights.values())
        shares = {name: remaining * weights[name] / norm for name in open_names}
        floors = {name: int(share) for name, share in shares.items()}
        leftover = remaining - sum(floors.values())
        for name in sorted(open_names, key=lambda n: (-(shares[n] - floors[n]), n))[:leftover]:
            floors[name] += 1
        progressed = 0
        for name in list(open_names):
            take = min(floors[name], sizes[name] - alloc[name])
            alloc[name] += take
            progressed += take
            if alloc[name] >= sizes[name]:
                open_names.remove(name)
        remaining -= progressed
        if progressed == 0:
            break
    return alloc


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("<image>", " ")).strip().lower()


def _clean_turns(conversations) -> tuple[list[dict] | None, str]:
    """Validated human/assistant turns with image placeholders removed, or (None, reason)."""
    if not conversations:
        return None, "no_conversation"
    turns = []
    placeholders = 0
    for index, turn in enumerate(conversations):
        role = str(turn.get("from", "")).lower()
        value = str(turn.get("value") or "")
        placeholders += value.count("<image>")
        expected = _HUMAN if index % 2 == 0 else _GPT
        if role not in expected:
            return None, "bad_role_order"
        text = value.replace("<image>", "").strip()
        if not text:
            return None, "empty_turn"
        turns.append({"from": "human" if role in _HUMAN else "gpt", "value": text})
    if len(turns) % 2 or len(turns) < 2:
        return None, "no_final_answer"
    if placeholders > 1:
        return None, "multi_image_placeholder"
    return turns, ""


def _process_file(task: dict) -> dict:
    """Read the selected rows of one parquet file; write their media; return candidates."""
    import pyarrow.parquet as pq
    from PIL import Image

    media_dir = Path(task["media_dir"])
    parquet = pq.ParquetFile(task["path"])
    wanted = sorted(task["rows"])
    out, rejects = [], Counter()
    offset = 0
    cursor = 0
    for group_index in range(parquet.num_row_groups):
        size = parquet.metadata.row_group(group_index).num_rows
        local = []
        while cursor < len(wanted) and wanted[cursor] < offset + size:
            local.append(wanted[cursor] - offset)
            cursor += 1
        if local:
            table = parquet.read_row_group(group_index, columns=["id", "image", "conversations"])
            for local_row, row in zip(local, table.take(local).to_pylist()):
                source_row = offset + local_row
                image = row.get("image") or {}
                data = image.get("bytes") if isinstance(image, dict) else None
                if not data:
                    rejects["no_image"] += 1
                    continue
                turns, reason = _clean_turns(row.get("conversations"))
                if turns is None:
                    rejects[reason] += 1
                    continue
                try:
                    with Image.open(io.BytesIO(data)) as probe:
                        probe.verify()
                    with Image.open(io.BytesIO(data)) as decoded:
                        decoded.load()
                        width, height, fmt = decoded.width, decoded.height, decoded.format
                        perceptual = dhash(decoded)
                except Exception:
                    rejects["corrupt_image"] += 1
                    continue
                if min(width, height) < 1 or max(width, height) / min(width, height) > _MAX_ASPECT:
                    rejects["extreme_aspect"] += 1
                    continue
                digest = hashlib.sha256(data).hexdigest()
                ext = _EXT.get(str(fmt).upper(), f".{str(fmt).lower()}")
                rel = f"{digest[:2]}/{digest}{ext}"
                dst = media_dir / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                    tmp.write_bytes(data)
                    os.replace(tmp, dst)
                request = " || ".join(_normalize(turn["value"]) for turn in turns[:-1])
                out.append(
                    {
                        "group": task["group"],
                        "subset": task["subset"],
                        "source_file": os.path.relpath(task["path"], task["root"]),
                        "source_row": source_row,
                        "ov_id": row.get("id"),
                        "media_sha256": digest,
                        "media_dhash": perceptual,
                        "media": rel,
                        "width": width,
                        "height": height,
                        "turns": turns,
                        "request_sha256": hashlib.sha256(request.encode("utf-8")).hexdigest(),
                    }
                )
        offset += size
    return {"candidates": out, "rejects": dict(rejects), "subset": task["subset"]}


def _split_quota(total: int, quotas: dict[str, int]) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` rows over group quotas."""
    whole = sum(quotas.values())
    shares = {name: total * quota / whole for name, quota in quotas.items()}
    out = {name: int(share) for name, share in shares.items()}
    for name in sorted(shares, key=lambda n: (-(shares[n] - out[n]), n))[: total - sum(out.values())]:
        out[name] += 1
    return out


def _row_record(candidate: dict, media_dir: Path, split: str, rank: int | None) -> dict:
    """Repo JSONL row: v1 for one question/answer pair, schema v2 for multi-turn."""
    image = str(media_dir / candidate["media"])
    row_id = f"ov-{candidate['media_sha256'][:16]}-{candidate['request_sha256'][:8]}"
    turns = candidate["turns"]
    if len(turns) == 2:
        row = {"id": row_id, "image": image, "question": turns[0]["value"], "answer": turns[1]["value"]}
    else:
        conversations = [dict(turn) for turn in turns]
        conversations[0]["value"] = "<image>\n" + conversations[0]["value"]
        row = {
            "id": row_id,
            "images": [image],
            "image_turns": [0],
            "conversations": conversations,
            "schema": "vlm-dflash-data-v2",
        }
    row["split"] = split
    meta = {
        "id": row_id,
        "split": split,
        "group": candidate["group"],
        "subset": candidate["subset"],
        "pool_rank": rank,
        "n_turns": len(turns) // 2,
        "source_file": candidate["source_file"],
        "source_row": candidate["source_row"],
        "ov_id": candidate["ov_id"],
        "media_sha256": candidate["media_sha256"],
        "width": candidate["width"],
        "height": candidate["height"],
    }
    return row, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="local LLaVA-OneVision-Data directory")
    parser.add_argument("--groups", required=True, help="group mapping YAML (quotas + subsets)")
    parser.add_argument("--out", required=True, help="output directory for pool.jsonl and stats")
    parser.add_argument("--media-dir", required=True, help="directory receiving one file per image hash")
    parser.add_argument("--exclude-media", action="append", default=[], help="file of image sha256 to drop")
    parser.add_argument("--exclude-dhash", action="append", default=[], help="file of image dHash values to drop")
    parser.add_argument("--oversample", type=float, default=1.12, help="train candidates per quota row")
    parser.add_argument("--draw-margin", type=float, default=1.3, help="raw draws per needed row")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root, out_dir, media_dir = Path(args.root), Path(args.out), Path(args.media_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(Path(args.groups).read_text(encoding="utf-8"))
    seed = int(spec.get("seed", 42))
    power = float(spec.get("subset_weight_power", 0.5))
    quotas = {name: int(group["quota"]) for name, group in spec["groups"].items()}
    dev_quota = _split_quota(int(spec.get("dev_size", 512)), quotas)
    test_quota = _split_quota(int(spec.get("test_size", 512)), quotas)
    train_quota = {name: math.ceil(quota * args.oversample) for name, quota in quotas.items()}

    def read_set(paths: list[str]) -> set[str]:
        return {line.strip() for path in paths for line in Path(path).read_text().splitlines() if line.strip()}

    excluded, excluded_dhash = read_set(args.exclude_media), read_set(args.exclude_dhash)

    # ---- draw plan ---------------------------------------------------------
    tasks, plan = [], {}
    for name, group in spec["groups"].items():
        files = {subset: _subset_files(root, subset) for subset in group["subsets"]}
        missing = [subset for subset, paths in files.items() if not paths]
        if missing:
            raise SystemExit(f"group {name}: subsets without parquet files under {root}: {missing}")
        rows = {subset: _subset_rows(paths) for subset, paths in files.items()}
        sizes = {subset: sum(counts) for subset, counts in rows.items()}
        needed = train_quota[name] + dev_quota[name] + test_quota[name]
        alloc = _allocate(math.ceil(needed * args.draw_margin), sizes, power)
        plan[name] = {"needed": needed, "available": sum(sizes.values()), "draws": alloc}
        for subset, draws in alloc.items():
            picks = sorted(random.Random(f"{seed}:{subset}").sample(range(sizes[subset]), draws))
            start = 0
            for path, count in zip(files[subset], rows[subset]):
                local = [pick - start for pick in picks if start <= pick < start + count]
                if local:
                    tasks.append(
                        {"group": name, "subset": subset, "path": path, "rows": local,
                         "root": str(root), "media_dir": str(media_dir)}
                    )
                start += count
    print(f"draw plan: {sum(len(t['rows']) for t in tasks)} rows over {len(tasks)} files", flush=True)

    # ---- read + validate ---------------------------------------------------
    candidates, rejects = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, result in enumerate(pool.map(_process_file, tasks, chunksize=1), start=1):
            candidates.extend(result["candidates"])
            rejects.update(result["rejects"])
            if done % 50 == 0 or done == len(tasks):
                print(f"read {done}/{len(tasks)} files, {len(candidates)} candidates", flush=True)
    candidates.sort(key=lambda c: (c["group"], c["subset"], c["source_file"], c["source_row"]))

    # ---- dedupe (media + normalized request) and evaluation exclusion ------
    seen, unique = set(), []
    for candidate in candidates:
        key = (candidate["media_sha256"], candidate["request_sha256"])
        if candidate["media_sha256"] in excluded:
            rejects["eval_media_overlap"] += 1
        elif candidate["media_dhash"] in excluded_dhash:
            rejects["eval_media_dhash_overlap"] += 1
        elif key in seen:
            rejects["duplicate_request"] += 1
        else:
            seen.add(key)
            unique.append(candidate)

    # ---- freeze test, then dev (one row per image, stratified) -------------
    by_media: dict[str, list[int]] = defaultdict(list)
    for index, candidate in enumerate(unique):
        by_media[candidate["media_sha256"]].append(index)
    media_group = {media: unique[indices[0]]["group"] for media, indices in by_media.items()}
    held: dict[str, dict[str, list[int]]] = {"test": {}, "dev": {}}
    used_media: set[str] = set()
    for split, split_quota in (("test", test_quota), ("dev", dev_quota)):
        for name in quotas:
            medias = sorted(media for media, group in media_group.items() if group == name and media not in used_media)
            random.Random(f"{seed}:{split}:{name}").shuffle(medias)
            chosen = medias[: split_quota[name]]
            if len(chosen) < split_quota[name]:
                raise SystemExit(f"group {name}: only {len(chosen)} images left for {split} ({split_quota[name]} needed)")
            used_media.update(chosen)
            held[split][name] = [by_media[media][0] for media in chosen]
    siblings_dropped = sum(len(by_media[media]) - 1 for media in used_media)

    # ---- train candidates in seeded per-group order + reserve --------------
    train_rows, reserve_rows, metas = [], [], []
    stats_groups = {}
    for name in quotas:
        pool_indices = [
            index for index, candidate in enumerate(unique)
            if candidate["group"] == name and candidate["media_sha256"] not in used_media
        ]
        random.Random(f"{seed}:train:{name}").shuffle(pool_indices)
        if len(pool_indices) < quotas[name]:
            raise SystemExit(f"group {name}: {len(pool_indices)} train rows < quota {quotas[name]}")
        for rank, index in enumerate(pool_indices):
            row, meta = _row_record(unique[index], media_dir, "train", rank)
            (train_rows if rank < train_quota[name] else reserve_rows).append(row)
            metas.append({**meta, "role": "candidate" if rank < train_quota[name] else "reserve"})
        stats_groups[name] = {
            "quota": quotas[name],
            "train_candidates": min(len(pool_indices), train_quota[name]),
            "reserve": max(0, len(pool_indices) - train_quota[name]),
            "dev": dev_quota[name],
            "test": test_quota[name],
            "subsets": dict(Counter(unique[index]["subset"] for index in pool_indices[: train_quota[name]])),
        }
    held_rows = []
    for split in ("dev", "test"):
        for name, indices in held[split].items():
            for index in indices:
                row, meta = _row_record(unique[index], media_dir, split, None)
                held_rows.append(row)
                metas.append({**meta, "role": split})

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "pool.jsonl", train_rows + held_rows)
    write_jsonl(out_dir / "reserve.jsonl", reserve_rows)
    write_jsonl(out_dir / "pool_meta.jsonl", metas)
    stats = {
        "seed": seed,
        "subset_weight_power": power,
        "oversample": args.oversample,
        "draw_margin": args.draw_margin,
        "plan": plan,
        "num_drawn": sum(len(t["rows"]) for t in tasks),
        "num_valid": len(candidates),
        "num_unique": len(unique),
        "rejects": dict(rejects),
        "excluded_media": len(excluded),
        "excluded_dhash": len(excluded_dhash),
        "dev_test_sibling_rows_dropped": siblings_dropped,
        "unique_media": {
            "train_candidates": len({row.get("image") or row["images"][0] for row in train_rows}),
            "dev": len(set().union(*[{unique[i]["media_sha256"] for i in v} for v in held["dev"].values()])),
            "test": len(set().union(*[{unique[i]["media_sha256"] for i in v} for v in held["test"].values()])),
        },
        "groups": stats_groups,
    }
    (out_dir / "pool_stats.json").write_text(json.dumps(stats, indent=2))
    with (out_dir / "checksums.txt").open("w") as handle:
        for name in ("pool.jsonl", "reserve.jsonl", "pool_meta.jsonl", "pool_stats.json"):
            handle.write(f"{hashlib.sha256((out_dir / name).read_bytes()).hexdigest()}  {name}\n")
    print(json.dumps({k: stats[k] for k in ("num_drawn", "num_valid", "num_unique", "rejects", "unique_media")}))
    print("POOL_DONE")


if __name__ == "__main__":
    sys.exit(main())
