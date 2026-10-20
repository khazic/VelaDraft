#!/usr/bin/env python3
"""Pool rows whose image is also an evaluation image (sha256 or 64-bit dHash), as an exclusion list.

    python scripts/a100_qwen3vl8b_200k/media_overlap.py --eval-images 'MMSpec/test/images/*' \\
        --pool-meta SRC_POOL/pool_meta.jsonl --media-source PREPARED_IMAGES --out OUT_DIR --workers 48

The dHash is the pool builder's own (``scripts/build_onevision_pool.py``), so a
re-encoded or resized copy of an evaluation image is caught as well as a
byte-identical one. Pool images are read from a prepared-data media dir
(``<row id>_<sha256[:16]><ext>``). Writes ``exclude_ids.txt`` (one row id per
line), ``eval_media_sha256.txt``, ``eval_media_dhash.txt`` and
``overlap_report.json`` (counts per split and group, the matched pairs).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_BUILDER = Path(__file__).resolve().parents[1] / "build_onevision_pool.py"
_spec = importlib.util.spec_from_file_location("build_onevision_pool", _BUILDER)
_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_builder)


def _hashes(path: str) -> tuple[str, str, str | None]:
    """``(path, sha256, dhash or None when the image cannot be decoded)``."""
    from PIL import Image

    data = Path(path).read_bytes()
    try:
        with Image.open(path) as image:
            image.draft("L", (64, 64))
            dhash = _builder.dhash(image)
    except Exception:
        dhash = None
    return path, hashlib.sha256(data).hexdigest(), dhash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-images", action="append", required=True, help="glob of evaluation images (repeatable)")
    parser.add_argument("--pool-meta", type=Path, required=True)
    parser.add_argument("--media-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    eval_paths = sorted({path for pattern in args.eval_images for path in glob.glob(pattern)})
    meta = [json.loads(line) for line in open(args.pool_meta, encoding="utf-8") if line.strip()]
    suffix = {}
    for path in args.media_source.iterdir():
        stem, ext = path.stem, path.suffix
        suffix[stem] = ext
    pool_paths = {}
    for record in meta:
        key = f"{record['id']}_{record['media_sha256'][:16]}"
        if key in suffix:
            pool_paths[record["id"]] = str(args.media_source / f"{key}{suffix[key]}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        eval_hashes = list(pool.map(_hashes, eval_paths, chunksize=16))
        pool_hashes = dict(
            (row_id, (sha, dhash))
            for row_id, (_, sha, dhash) in zip(pool_paths, pool.map(_hashes, pool_paths.values(), chunksize=256))
        )
    eval_sha = {sha: path for path, sha, _ in eval_hashes}
    eval_dhash = {dhash: path for path, _, dhash in eval_hashes if dhash}
    (args.out / "eval_media_sha256.txt").write_text("".join(f"{sha}\n" for sha in sorted(eval_sha)))
    (args.out / "eval_media_dhash.txt").write_text("".join(f"{dhash}\n" for dhash in sorted(eval_dhash)))
    by_id = {record["id"]: record for record in meta}
    matches = []
    for row_id, (sha, dhash) in pool_hashes.items():
        kind = "sha256" if sha in eval_sha else ("dhash" if dhash in eval_dhash else None)
        if kind:
            record = by_id[row_id]
            matches.append({"id": row_id, "split": record["split"], "group": record["group"], "match": kind,
                            "eval_image": Path(eval_sha.get(sha) or eval_dhash[dhash]).name})
    (args.out / "exclude_ids.txt").write_text("".join(f"{match['id']}\n" for match in matches))
    report = {
        "eval_images": len(eval_paths), "eval_unique_sha256": len(eval_sha), "eval_unique_dhash": len(eval_dhash),
        "pool_rows": len(meta), "pool_rows_with_media": len(pool_paths),
        "pool_rows_without_media": len(meta) - len(pool_paths),
        "overlap_rows": len(matches),
        "overlap_by_split": dict(Counter(match["split"] for match in matches)),
        "overlap_by_group": dict(Counter(match["group"] for match in matches)),
        "overlap_by_match": dict(Counter(match["match"] for match in matches)),
        "matches": matches,
    }
    (args.out / "overlap_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "matches"}))
    print("MEDIA_OVERLAP_DONE")


if __name__ == "__main__":
    main()
