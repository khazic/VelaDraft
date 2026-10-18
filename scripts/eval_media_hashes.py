#!/usr/bin/env python3
"""Image sha256 + dHash lists of evaluation parquet splits (train/eval media dedupe).

    python scripts/eval_media_hashes.py --out eval_media \
        "/data/lmms-lab__textvqa/data/validation-*.parquet" \
        "/data/lmms-lab__DocVQA/DocVQA/validation-*.parquet" \
        "/data/lmms-lab__ChartQA/data/test-*.parquet"

Writes ``<out>_sha256.txt`` and ``<out>_dhash.txt`` (one value per line) for
``build_onevision_pool.py --exclude-media / --exclude-dhash``, plus a JSON
summary with the per-pattern image counts.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_onevision_pool import dhash  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--column", default="image", help="image column (struct with a bytes field)")
    parser.add_argument("patterns", nargs="+", help="parquet glob patterns")
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    sha_values, dhash_values, summary = set(), set(), {}
    for pattern in args.patterns:
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"no parquet files match {pattern}")
        count = 0
        for path in files:
            parquet = pq.ParquetFile(path)
            for index in range(parquet.num_row_groups):
                for image in parquet.read_row_group(index, columns=[args.column]).column(0).to_pylist():
                    data = image.get("bytes") if isinstance(image, dict) else None
                    if not data:
                        continue
                    sha_values.add(hashlib.sha256(data).hexdigest())
                    with Image.open(io.BytesIO(data)) as decoded:
                        dhash_values.add(dhash(decoded))
                    count += 1
        summary[pattern] = {"files": len(files), "images": count}
        print(f"{pattern}: {count} images", flush=True)
    Path(f"{args.out}_sha256.txt").write_text("".join(f"{value}\n" for value in sorted(sha_values)))
    Path(f"{args.out}_dhash.txt").write_text("".join(f"{value}\n" for value in sorted(dhash_values)))
    summary["unique_sha256"], summary["unique_dhash"] = len(sha_values), len(dhash_values)
    Path(f"{args.out}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
