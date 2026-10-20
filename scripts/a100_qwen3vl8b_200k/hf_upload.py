#!/usr/bin/env python3
"""Upload and verify the Hugging Face bundle (contract sections 37, 43, 44, 46).

    python scripts/a100_qwen3vl8b_200k/hf_upload.py --task "$VELA_TASK" --repo-id "$HF_REPO_ID" [--dry-run]

Reads the token from the environment (``HF_TOKEN`` or the hub's stored login),
never prints it. ``--dry-run`` writes the file manifest that would be uploaded
and stops (status PREPARED). Otherwise: create the public model repo (exist_ok),
``upload_large_folder`` the bundle, then verify remotely: every bundle file is
listed, sizes match, LFS files' sha256 match ``checksums.sha256`` and small text
files are re-downloaded and compared byte for byte. The remote revision is
written to ``HF_UPLOAD_STATUS.md``; status VERIFIED only when every check
passes, else FAILED (or UPLOAD_BLOCKED when credentials / network are missing).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

KEY_FILES = ("README.md", "checksums.sha256", "results/final.csv", "results/benchmark_results.csv")


def _status(task: Path, status: str, lines: list[str]) -> None:
    text = ["# HF upload status: a100_qwen3vl8b_200k_s42", "", f"Status: {status}", "",
            f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", "", *lines]
    (task / "HF_UPLOAD_STATUS.md").write_text("\n".join(text) + "\n")
    print(f"HF_UPLOAD_{status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    bundle = args.task / "hf_upload_bundle"
    sums = dict(reversed(line.split("  ", 1)) for line in (bundle / "checksums.sha256").read_text().splitlines())
    files = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file())
    if args.dry_run:
        (args.task / "hf_upload_file_list.txt").write_text("\n".join(files) + "\n")
        _status(args.task, "PREPARED", [f"Repo: {args.repo_id}", f"Files: {len(files)} (see hf_upload_file_list.txt)"])
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()["name"]
    except Exception as error:  # no credentials or no network
        _status(args.task, "UPLOAD_BLOCKED", [f"Repo: {args.repo_id}", f"Reason: {type(error).__name__}"])
        return
    _status(args.task, "UPLOADING", [f"Repo: {args.repo_id}", f"As: {user}", f"Files: {len(files)}"])
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="model", folder_path=str(bundle))
    info = api.model_info(args.repo_id, files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    problems = []
    for name in files:
        local = bundle / name
        sibling = remote.get(name)
        if sibling is None:
            problems.append(f"missing remotely: {name}")
            continue
        if sibling.size is not None and sibling.size != local.stat().st_size:
            problems.append(f"size differs: {name}")
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None and name in sums and lfs.sha256 != sums[name]:
            problems.append(f"sha256 differs: {name}")
    for name in KEY_FILES + tuple(name for name in files if name.startswith("configs/")):
        if name not in remote:
            continue
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(args.repo_id, name, revision=info.sha)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sums.get(name):
            problems.append(f"downloaded copy differs: {name}")
    lines = [f"Repo: https://huggingface.co/{args.repo_id}", f"Remote revision: {info.sha}", f"Files: {len(files)}",
             f"Checked: listing + sizes for all files, LFS sha256 where present, byte compare of "
             f"{', '.join(KEY_FILES)} and the configs"]
    if problems:
        _status(args.task, "FAILED", lines + ["", "Problems:", *[f"- {problem}" for problem in problems[:50]]])
    else:
        _status(args.task, "VERIFIED", lines)
    (args.task / "hf_upload_result.json").write_text(json.dumps({"repo": args.repo_id, "revision": info.sha,
                                                                 "problems": problems}, indent=2))


if __name__ == "__main__":
    main()
