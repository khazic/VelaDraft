"""CPU tests of the a100_qwen3vl8b_200k pool derivation and media linking scripts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "a100_qwen3vl8b_200k"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"q3vl_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


derive_pool = _load("derive_pool")
link_media = _load("link_media")


def _write(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _source_pool(tmp_path: Path, media: Path) -> Path:
    """Two groups, ranks 0..5 each (0-3 candidates, 4-5 reserve), one dev and one test row per group."""
    src = tmp_path / "src"
    src.mkdir()
    pool, reserve, meta = [], [], []
    for group in ("general", "ocr"):
        for rank in range(6):
            row_id = f"{group}-{rank}"
            row = {"id": row_id, "image": f"{media}/{rank:02d}/{group}{rank}.png", "question": "q", "answer": "a",
                   "split": "train"}
            (pool if rank < 4 else reserve).append(row)
            meta.append({"id": row_id, "split": "train", "group": group, "pool_rank": rank,
                         "role": "candidate" if rank < 4 else "reserve"})
    for group in ("general", "ocr"):
        for split in ("dev", "test"):
            row_id = f"{group}-{split}"
            pool.append({"id": row_id, "image": f"{media}/xx/{row_id}.png", "question": "q", "answer": "a",
                         "split": split})
            meta.append({"id": row_id, "split": split, "group": group, "pool_rank": None, "role": split})
    _write(src / "pool.jsonl", pool)
    _write(src / "reserve.jsonl", reserve)
    _write(src / "pool_meta.jsonl", meta)
    (src / "checksums.txt").write_text("abc  pool.jsonl\n")
    return src


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def test_derive_takes_rank_prefixes_and_keeps_dev_test(tmp_path):
    src = _source_pool(tmp_path, "/old/media")
    out = tmp_path / "out"
    stats = derive_pool.derive(src, out, {"general": 2, "ocr": 1}, {"general": 1.5, "ocr": 1.0},
                               ("/old/media", "/new/media"), {"general": 1, "ocr": 1})
    assert stats["train_candidates_per_group"] == {"general": 3, "ocr": 1}
    assert _ids(out / "pool.jsonl") == ["general-0", "general-1", "general-2", "ocr-0", "general-dev",
                                        "general-test", "ocr-dev", "ocr-test"]
    # The reserve continues each group's order: source candidates past the new cut, then the source reserve.
    assert _ids(out / "reserve.jsonl") == ["general-3", "general-4", "general-5", "ocr-1", "ocr-2", "ocr-3",
                                           "ocr-4", "ocr-5"]
    assert _ids(out / "pilot_pool.jsonl") == ["general-0", "ocr-0"]
    first = json.loads((out / "pool.jsonl").read_text().splitlines()[0])
    assert first["image"] == "/new/media/00/general0.png" and first["split"] == "train"
    roles = {row["id"]: row["role"] for row in map(json.loads, (out / "pool_meta.jsonl").read_text().splitlines())}
    assert roles["general-2"] == "candidate" and roles["general-3"] == "reserve" and roles["ocr-dev"] == "dev"
    assert (stats["num_dev"], stats["num_test"], stats["num_reserve"]) == (2, 2, 8)
    for line in (out / "checksums.txt").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_derive_excluded_rows_do_not_count(tmp_path):
    src = _source_pool(tmp_path, "/old/media")
    stats = derive_pool.derive(src, tmp_path / "out", {"general": 2, "ocr": 1}, {"general": 1.0, "ocr": 1.0}, None,
                               None, {"general-0", "ocr-1"})
    # general-0 is skipped, so general takes ranks 1 and 2; ocr-1 leaves the reserve.
    assert _ids(tmp_path / "out" / "pool.jsonl")[:3] == ["general-1", "general-2", "ocr-0"]
    assert "ocr-1" not in _ids(tmp_path / "out" / "reserve.jsonl")
    assert stats["num_excluded"] == 2
    with pytest.raises(SystemExit, match="held-out splits are frozen"):
        derive_pool.derive(src, tmp_path / "out2", {"general": 1, "ocr": 1}, {"general": 1.0, "ocr": 1.0}, None,
                           None, {"ocr-dev"})


def test_derive_refuses_a_too_small_source(tmp_path):
    src = _source_pool(tmp_path, "/old/media")
    with pytest.raises(SystemExit, match="too small"):
        derive_pool.derive(src, tmp_path / "out", {"general": 10, "ocr": 1}, {"general": 1.0, "ocr": 1.0}, None, None)
    with pytest.raises(SystemExit, match="groups must match"):
        derive_pool.derive(src, tmp_path / "out", {"general": 1}, {"general": 1.0}, None, None)


def test_link_media_verifies_bytes(tmp_path, monkeypatch, capsys):
    source = tmp_path / "corpus"
    source.mkdir()
    payload = b"image-bytes"
    sha = hashlib.sha256(payload).hexdigest()
    (source / f"row-1_{sha[:16]}.png").write_bytes(payload)
    target = tmp_path / "media" / sha[:2] / f"{sha}.png"
    pool = tmp_path / "pool.jsonl"
    _write(pool, [{"id": "row-1", "image": str(target)}])
    monkeypatch.setattr("sys.argv", ["link_media.py", "--pool", str(pool), "--from", str(source), "--workers", "1"])
    link_media.main()
    assert target.read_bytes() == payload
    assert "LINK_MEDIA_DONE" in capsys.readouterr().out
    # Corrupt source bytes are refused.
    (source / f"row-2_{'0' * 16}.png").write_bytes(b"other")
    _write(pool, [{"id": "row-2", "image": str(tmp_path / "media" / "00" / f"{'0' * 64}.png")}])
    with pytest.raises(SystemExit):
        link_media.main()
