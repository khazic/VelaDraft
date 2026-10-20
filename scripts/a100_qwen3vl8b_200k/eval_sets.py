#!/usr/bin/env python3
"""Freeze the evaluation members of the a100_qwen3vl8b_200k_s42 matrix (contract sections 11 and 26.1).

    python scripts/a100_qwen3vl8b_200k/eval_sets.py --out TASK/eval_sets --pool POOL \\
        --mmspec MMSPEC_REPO --textvqa 'TEXTVQA/validation-*.parquet' --docvqa 'DOCVQA/validation-*.parquet' \\
        --chartqa 'CHARTQA/test-*.parquet' --members TASK/shared/eval_members.json

Sets (one ``<set>.jsonl`` each, rows ``{id, image, system, turns, answers,
max_new_tokens, topic}``; images extracted under ``media/<set>/``):

- ``mmspec``: the official 600 sessions (6 topics x 100; multi-turn sessions
  keep all their user turns), MMSpec's system prompt, 1024 new tokens per turn;
- ``textvqa`` / ``docvqa`` / ``chartqa``: 256 fixed members each (seed 42 over
  the sorted question ids of TextVQA validation, DocVQA validation, ChartQA
  test), "Answer the question using a single word or phrase.", 128 new tokens;
- ``long32``: the 32 frozen dev rows with the longest dataset answers, 2048 new
  tokens (long-response diagnostic);
- ``heldout_dev`` / ``heldout_test``: the 512 + 512 frozen dev / test rows of
  the pool, 128 new tokens.

``eval_members.json`` records per set the source, revision, ids and image
sha256 so the members are fixed before the training corpus is frozen.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import random
from pathlib import Path

SHORT_ANSWER = "Answer the question using a single word or phrase."
MMSPEC_SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, "
    "detailed, and polite answers to the human's questions."
)


def _parquet_rows(pattern: str, columns: list[str]):
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(pattern)):
        table = pq.ParquetFile(path)
        for group in range(table.num_row_groups):
            yield from table.read_row_group(group, columns=columns).to_pylist()


def _save_image(blob: dict, out: Path) -> tuple[str, str]:
    from PIL import Image

    data = blob["bytes"]
    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    path = out.with_suffix(".jpg" if fmt in ("jpeg", "jpg") else f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path), hashlib.sha256(data).hexdigest()


def _answers(value) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except ValueError:
            pass
        return [value]
    return [str(item) for item in value]


def _benchmark(name: str, pattern: str, id_key: str | None, answer_key: str, out: Path, count: int = 256) -> list[dict]:
    """Fixed ``count`` members: seed 42 over the sorted ids, then one pass to extract only those rows.

    ``id_key=None`` (ChartQA has no id column) uses the row's position in the
    sorted parquet files, zero-padded so it sorts like an integer.
    """
    def key(position: int, row: dict) -> str:
        return str(row[id_key]) if id_key else f"{position:06d}"

    columns = [id_key] if id_key else ["question"]
    ids = sorted(key(position, row) for position, row in enumerate(_parquet_rows(pattern, columns)))
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{name}: duplicate {id_key} values")
    chosen = set(random.Random(42).sample(ids, count))
    rows = []
    read = [column for column in (id_key, "question", answer_key, "image") if column]
    for position, row in enumerate(_parquet_rows(pattern, read)):
        row_id = key(position, row)
        if row_id not in chosen:
            continue
        image, sha = _save_image(row["image"], out / "media" / name / row_id)
        rows.append({"id": f"{name}-{row_id}", "image": image, "image_sha256": sha, "system": None,
                     "turns": [f"{row['question'].strip()}\n{SHORT_ANSWER}"], "answers": _answers(row[answer_key]),
                     "max_new_tokens": 128, "topic": name})
    return sorted(rows, key=lambda row: row["id"])


def _mmspec(repo: Path) -> list[dict]:
    base = repo / "dataset" / "MMSpec" / "test"
    rows = []
    for line in open(base / "mmspec.jsonl", encoding="utf-8"):
        record = json.loads(line)
        image = base / "images" / record["image"]
        rows.append({"id": record["id"], "image": str(image),
                     "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "system": MMSPEC_SYSTEM,
                     "turns": list(record["turns"]), "answers": None, "max_new_tokens": 1024,
                     "topic": record["topic"], "category": record.get("category")})
    if len(rows) != 600:
        raise SystemExit(f"MMSpec test has {len(rows)} sessions, expected 600")
    return rows


def _pool_rows(pool: Path, split: str) -> list[dict]:
    meta = {json.loads(line)["id"]: json.loads(line) for line in open(pool / "pool_meta.jsonl", encoding="utf-8")}
    rows = []
    for line in open(pool / "pool.jsonl", encoding="utf-8"):
        row = json.loads(line)
        if row.get("split") != split:
            continue
        if "conversations" in row:
            turns = [turn["value"] for turn in row["conversations"] if turn["from"] in ("human", "user")]
            answer = next(turn["value"] for turn in reversed(row["conversations"]) if turn["from"] in ("gpt", "assistant"))
        else:
            turns, answer = [row["question"]], row["answer"]
        rows.append({"id": row["id"], "image": row.get("image") or row["images"][0],
                     "image_sha256": meta[row["id"]]["media_sha256"], "system": None,
                     "turns": [turn.replace("<image>\n", "").replace("<image>", "").strip() for turn in turns],
                     "history": row.get("conversations"), "answers": [answer], "max_new_tokens": 128,
                     "topic": meta[row["id"]]["group"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--mmspec", type=Path, required=True)
    parser.add_argument("--textvqa", required=True)
    parser.add_argument("--docvqa", required=True)
    parser.add_argument("--chartqa", required=True)
    parser.add_argument("--members", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import subprocess

    mmspec_commit = subprocess.run(["git", "-C", str(args.mmspec), "rev-parse", "HEAD"], capture_output=True,
                                   text=True).stdout.strip()
    dev, test = _pool_rows(args.pool, "dev"), _pool_rows(args.pool, "test")
    long32 = [dict(row, id=f"long-{row['id']}", max_new_tokens=2048)
              for row in sorted(dev, key=lambda row: (-len(row["answers"][0]), row["id"]))[:32]]
    sets = {
        "mmspec": (_mmspec(args.mmspec), f"github.com/killthefullmoon/MMSpec@{mmspec_commit} dataset/MMSpec/test"),
        "textvqa": (_benchmark("textvqa", args.textvqa, "question_id", "answers", args.out),
                    "lmms-lab/textvqa validation, 256 by random.Random(42).sample over sorted question_id"),
        "docvqa": (_benchmark("docvqa", args.docvqa, "questionId", "answers", args.out),
                   "lmms-lab/DocVQA DocVQA validation, 256 by random.Random(42).sample over sorted questionId"),
        "chartqa": (_benchmark("chartqa", args.chartqa, None, "answer", args.out),
                    "lmms-lab/ChartQA test, 256 by random.Random(42).sample over row positions"),
        "long32": (long32, "frozen dev rows with the 32 longest dataset answers"),
        "heldout_dev": (dev, "frozen pool dev split"),
        "heldout_test": (test, "frozen pool test split"),
    }
    members = {}
    for name, (rows, source) in sets.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        members[name] = {"source": source, "count": len(rows), "ids": [row["id"] for row in rows],
                         "image_sha256": sorted({row["image_sha256"] for row in rows})}
    args.members.parent.mkdir(parents=True, exist_ok=True)
    args.members.write_text(json.dumps(members, indent=2))
    print(json.dumps({name: entry["count"] for name, entry in members.items()}))
    print("EVAL_SETS_FROZEN")


if __name__ == "__main__":
    main()
