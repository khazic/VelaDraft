#!/usr/bin/env python3
"""Materialize a pool's image files by hard-linking verified copies from a prepared-data media dir.

    python scripts/a100_qwen3vl8b_200k/link_media.py --pool OUT_POOL/pool.jsonl \\
        --from /dev/shm/.../corpus_images --workers 32

Pool rows name their image by content (``<sha256[:2]>/<sha256>.<ext>``). A
prepare stage stores the same bytes as ``<row id>_<sha256[:16]><ext>``. Every
linked file's full sha256 is checked against the name it is linked to, so the
pool's media are byte-identical to the frozen source pool's. Prints
``LINK_MEDIA_DONE`` or exits non-zero listing what is missing or corrupt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _sha(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(item: tuple[str, str]) -> tuple[str, str | None]:
    """``(dst, error or None)``: verify the source bytes, then hard-link it to ``dst``."""
    src, dst = item
    want = Path(dst).stem
    if os.path.exists(dst):
        return dst, None if _sha(dst) == want else f"existing file has a different sha256: {dst}"
    if not os.path.exists(src):
        return dst, f"missing source {src}"
    if _sha(src) != want:
        return dst, f"source bytes do not match {want}: {src}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except FileExistsError:
        pass
    return dst, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, required=True, help="pool jsonl whose image paths to materialize")
    parser.add_argument("--from", dest="source", type=Path, required=True, help="prepared-data media dir")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    items = {}
    with open(args.pool, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            images = row.get("images") or [row["image"]]
            for path in images:
                sha, ext = Path(path).stem, Path(path).suffix
                items[path] = str(args.source / f"{row['id']}_{sha[:16]}{ext}")
    work = [(src, dst) for dst, src in items.items()]
    errors = []
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    results = pool.map(_link, work, chunksize=256) if pool is not None else map(_link, work)
    for done, (_dst, error) in enumerate(results, start=1):
        if error:
            errors.append(error)
        if done % 50000 == 0:
            print(f"linked {done}/{len(work)}", flush=True)
    if pool is not None:
        pool.shutdown()
    print(json.dumps({"files": len(work), "errors": len(errors), "first_errors": errors[:5]}))
    if errors:
        sys.exit(1)
    print("LINK_MEDIA_DONE")


if __name__ == "__main__":
    main()
