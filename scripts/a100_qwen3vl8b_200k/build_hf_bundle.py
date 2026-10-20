#!/usr/bin/env python3
"""Build the public Hugging Face bundle of the a100_qwen3vl8b_200k_s42 matrix (contract sections 37-45).

    python scripts/a100_qwen3vl8b_200k/build_hf_bundle.py --task "$VELA_TASK" --repo-id khazic/VelaDraft-Qwen3VL-8B-200K \\
        --replace "$VLM_DFLASH_TARGET_QWEN3VL=Qwen/Qwen3-VL-8B-Instruct" --replace "$VELA_TASK=<task_root>" ...

The bundle (``<task>/hf_upload_bundle/``) holds only artifacts meant to be
public: the seven configs, per method the exported final drafter (safetensors +
config, sanitized) with its metrics and training summary (or a STATUS.md for a
blocked row), the result tables, evaluation protocol / summaries / compressed
per-turn records, and public provenance (commit, environment, dataset and
evaluation manifests as ids + hashes, training plan). Regenerated answers and
raw images are not included. Every text file is rewritten with ``--replace``
pairs, then scanned: any internal path, host, address, credential or
organization token left fails the build. Writes ``bundle_manifest.json``
(file, size, sha256, source, decision) and ``checksums.sha256``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

QUEUE = ["dflash", "dfly", "dflash2", "dspark", "dflare", "domino", "jetspec"]
LABELS = {"jetspec": "JetSpec-trained block proposal reference (the upstream tree runtime is not ported)",
          "dflash2": "PARTIAL: the repository's experimental reconstruction of the DFlash2 objective"}
TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".csv", ".jsonl"}
#: Patterns that must never reach the public bundle.
FORBIDDEN = [
    re.compile(pattern, re.I) for pattern in (
        r"/llm-align", r"/dev/shm", r"/root/", r"/home/", r"liuchonghan", r"duyimin", r"h-liuchonghan",
        r"10\.178\.", r"10\.177\.", r"qihoo", r"360\.cn", r"geelib", r"zhinao", r"dev05", r"public-proxy",
        r"hf_[A-Za-z0-9]{20,}", r"ghp_[A-Za-z0-9]{20,}", r"api\.360",
    )
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


class Sanitizer:
    def __init__(self, pairs: list[tuple[str, str]]):
        # Longest prefix first, so a nested path is rewritten before its parent.
        self.pairs = sorted(pairs, key=lambda pair: -len(pair[0]))

    def text(self, value: str) -> str:
        for old, new in self.pairs:
            value = value.replace(old, new)
        return value

    def obj(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.obj(item) for item in value]
        if isinstance(value, dict):
            return {self.text(str(key)): self.obj(item) for key, item in value.items()}
        return value


def scan(bundle: Path) -> list[str]:
    """Forbidden tokens left in text files and in safetensors headers."""
    hits = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES:
            text = path.read_text(errors="ignore")
        elif path.suffix == ".gz":
            text = gzip.decompress(path.read_bytes()).decode(errors="ignore")
        elif path.suffix == ".safetensors":
            with open(path, "rb") as handle:
                size = int.from_bytes(handle.read(8), "little")
                text = handle.read(size).decode(errors="ignore")
        else:
            continue
        for pattern in FORBIDDEN:
            match = pattern.search(text)
            if match:
                hits.append(f"{path.relative_to(bundle)}: {pattern.pattern} -> {text[max(0, match.start() - 40):match.end() + 40]!r}")
    return hits


def _write_json(path: Path, value, sanitizer: Sanitizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitizer.obj(value), indent=2, ensure_ascii=False))


def export_method(task: Path, method: str, out: Path, sanitizer: Sanitizer) -> bool:
    """Exported final drafter of one row (sanitized); False when the row has no final checkpoint."""
    from vlm_dflash.export.hf import export_run

    run = task / "runs" / method
    if not (run / "train" / "ckpt_final.pt").exists():
        return False
    staging = out.parent / f".export_{method}"
    if staging.exists():
        shutil.rmtree(staging)
    export_run(run, checkpoint="ckpt_final.pt", out_dir=staging)
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    for path in staging.iterdir():
        if path.name == "checksums.json":
            continue
        if path.suffix == ".json":
            _write_json(final / path.name, json.loads(path.read_text()), sanitizer)
        elif path.suffix in TEXT_SUFFIXES:
            (final / path.name).write_text(sanitizer.text(path.read_text()))
        else:
            shutil.copy2(path, final / path.name)
    (final / "checksums.json").write_text(json.dumps({p.name: _sha(p) for p in sorted(final.iterdir())}, indent=2))
    shutil.rmtree(staging)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--replace", action="append", default=[], help="OLD=NEW text rewrite (repeatable)")
    parser.add_argument("--code-url", default="https://github.com/LUOseu/VelaDraft")
    args = parser.parse_args()
    task = args.task
    bundle = task / "hf_upload_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    sanitizer = Sanitizer([tuple(item.split("=", 1)) for item in args.replace])
    manifest = []

    def record(path: Path, source: str, decision: str = "public") -> None:
        manifest.append({"file": str(path.relative_to(bundle)), "size": path.stat().st_size, "sha256": _sha(path),
                         "source": sanitizer.text(source), "decision": decision})

    for method in QUEUE:
        cfg = REPO / "configs" / "paper_v1" / "a100_qwen3vl8b_200k" / f"{method}_s42.yaml"
        dst = bundle / "configs" / cfg.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, dst)
        record(dst, f"repo:{cfg.relative_to(REPO)}")
        out = bundle / "methods" / method
        out.mkdir(parents=True, exist_ok=True)
        if export_method(task, method, out, sanitizer):
            for path in sorted((out / "final").iterdir()):
                record(path, f"export_run(runs/{method}, ckpt_final.pt)")
            metrics = [json.loads(p.read_text()) for p in sorted((task / "eval" / method).glob("*.summary.json"))]
            _write_json(out / "metrics.json", {"label": LABELS.get(method, ""), "evaluations": metrics}, sanitizer)
            record(out / "metrics.json", f"eval/{method}/*.summary.json")
            summary = task / "runs" / method / "train" / "train_summary.json"
            smoke = task / "smoke" / method / "smoke_report.json"
            _write_json(out / "training_summary.json", {
                "train_summary": json.loads(summary.read_text()) if summary.exists() else None,
                "smoke_gate": json.loads(smoke.read_text()) if smoke.exists() else None,
            }, sanitizer)
            record(out / "training_summary.json", f"runs/{method}/train/train_summary.json + smoke report")
        else:
            status = task / "status.json"
            rows = {row["method"]: row for row in json.loads(status.read_text())} if status.exists() else {}
            row = rows.get(method, {})
            (out / "STATUS.md").write_text(sanitizer.text(
                f"# {method}: no final checkpoint\n\nState: {row.get('state', 'unknown')}\n\n"
                f"Last update: {row.get('update', '-')}\n\nBlocker: {row.get('failure', '-')}\n"))
            record(out / "STATUS.md", "status.json")

    for path in sorted((task / "results").glob("*.csv")):
        dst = bundle / "results" / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(sanitizer.text(path.read_text()))
        record(dst, f"results/{path.name}")

    evaluation = bundle / "evaluation"
    summaries = [json.loads(p.read_text()) for p in sorted((task / "eval").glob("*/*.summary.json"))]
    _write_json(evaluation / "summary.json", summaries, sanitizer)
    record(evaluation / "summary.json", "eval/*/*.summary.json")
    _write_json(evaluation / "protocol.json", {
        "layout": "target + drafter on one GPU, batch 1, BF16, greedy",
        "timing_boundary": "processor + vision + host-to-device + prefill + draft + verify + output; image decode "
                           "and model load excluded; 2 warm-up sessions; CUDA synchronize on both sides",
        "AL_draft": "accepted draft tokens / verification rounds",
        "AL_progress": "emitted output tokens / verification rounds",
        "speedup": "sum(paired AR wall) / sum(paired SD wall); mean of per-turn ratios reported separately",
        "verification": {"sequential": "correctness reference (one target-verified token per call, no target-side "
                                       "speedup by construction)",
                         "strict_parallel": "block verification with a near-tie fallback to the sequential path"},
        "sets": json.loads((task / "shared" / "eval_members.json").read_text()),
    }, sanitizer)
    record(evaluation / "protocol.json", "eval_matrix.py + shared/eval_members.json")
    raw = evaluation / "raw_records"
    raw.mkdir(parents=True, exist_ok=True)
    for path in sorted((task / "eval").glob("*/*.jsonl")):
        dst = raw / f"{path.parent.name}.{path.name}.gz"
        dst.write_bytes(gzip.compress(sanitizer.text(path.read_text()).encode()))
        record(dst, f"eval/{path.parent.name}/{path.name}")

    provenance = bundle / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    commit = (task / "FORMAL_COMMIT").read_text().strip() if (task / "FORMAL_COMMIT").exists() else "unknown"
    (provenance / "git_commit.txt").write_text(f"{args.code_url} (fork branch khazic/feat/qwen3vl8b-200k)\n{commit}\n")
    record(provenance / "git_commit.txt", "FORMAL_COMMIT")
    env_lines = (task / "preflight" / "env.txt").read_text() if (task / "preflight" / "env.txt").exists() else ""
    (provenance / "environment.txt").write_text(sanitizer.text(env_lines))
    record(provenance / "environment.txt", "preflight/env.txt")
    members = json.loads((task / "shared" / "eval_members.json").read_text())
    _write_json(provenance / "eval_members_public.json", members, sanitizer)
    record(provenance / "eval_members_public.json", "shared/eval_members.json")
    dataset = {}
    for split, name in (("train", "train_manifest.json"), ("dev", "dev_manifest.json"), ("test", "test_manifest.json")):
        rows = json.loads((task / "shared" / name).read_text())["rows"]
        dataset[split] = [{key: row.get(key) for key in ("id", "group", "subset", "source_file", "source_row",
                                                         "image_sha256")} for row in rows]
    _write_json(provenance / "dataset_manifest_public.json", {
        "source": "lmms-lab/LLaVA-OneVision-Data", "revision": "b31b1dad33f0f916b7ccd9cf8eaec32df98c6914",
        "note": "ids, OneVision source rows and image hashes only; rebuild the requests from the public dataset "
                "with scripts/build_onevision_pool.py + scripts/a100_qwen3vl8b_200k/derive_pool.py",
        "splits": dataset,
        "regen_generation_config": json.loads((task / "regen" / "resolved_generation_config.json").read_text()),
    }, sanitizer)
    record(provenance / "dataset_manifest_public.json", "shared/*_manifest.json + regen config")
    _write_json(provenance / "training_plan.json", {
        "updates_per_epoch": 25000, "epochs": 3, "total_updates": 75000, "global_batch": 8, "micro_batch": 2,
        "grad_accumulation": 4, "warmup_updates": 3000, "lr": 1e-4, "min_lr": 1e-5, "betas": [0.9, 0.95],
        "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42, "sequence_length": 4096, "block_size": 8,
        "anchors": 512, "capture_layers": [1, 17, 33], "online_batch_size": 1,
    }, sanitizer)
    record(provenance / "training_plan.json", "matrix.py")

    for name in ("README.md", "LICENSE_OR_NOTICE.md", "experiment_manifest.json", "preflight_public.json"):
        source = task / "hf_templates" / name
        if source.exists():
            (bundle / name).write_text(sanitizer.text(source.read_text()))
            record(bundle / name, f"hf_templates/{name}")

    hits = scan(bundle)
    if hits:
        print("\n".join(hits[:50]))
        raise SystemExit(f"BUNDLE_FAIL {len(hits)} forbidden token(s) left; nothing may be uploaded")
    (bundle / "bundle_manifest.json").write_text(json.dumps({"repo_id": args.repo_id, "files": manifest}, indent=2))
    with open(bundle / "checksums.sha256", "w", encoding="utf-8") as handle:
        for path in sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "checksums.sha256"):
            handle.write(f"{_sha(path)}  {path.relative_to(bundle)}\n")
    print(f"BUNDLE_READY {len(manifest)} files in {bundle}")


if __name__ == "__main__":
    main()
