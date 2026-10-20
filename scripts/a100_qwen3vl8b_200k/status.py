#!/usr/bin/env python3
"""Write ``$VELA_TASK/STATUS.md``: one row per method (contract section 34).

    python scripts/a100_qwen3vl8b_200k/status.py --task "$VELA_TASK"

Reads every row's run_one logs, train_log.jsonl (with stage timing), the smoke
reports, eval outputs and the HF upload status; never writes anything but
STATUS.md (and status.json, the same table machine-readable).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

QUEUE = ["dflash", "dfly", "dflash2", "dspark", "dflare", "domino", "jetspec"]
TOTAL, PER_EPOCH, DFLY_BOUNDARY = 75_000, 25_000, 18_750
#: Rows whose formal result carries a maturity label (contract sections 18 M4/M5).
LABELS = {"jetspec": "JetSpec-trained block proposal reference", "dflash2": "PARTIAL (experimental objective reconstruction)"}


def _tail_rows(path: Path, window: int = 200) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "rb") as handle:
        handle.seek(max(0, path.stat().st_size - 400_000))
        lines = handle.read().decode(errors="ignore").splitlines()[-window:]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _first_attempt_time(out: Path) -> float | None:
    if not out.exists():
        return None
    match = re.search(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] formal ", out.read_text(errors="ignore"), re.M)
    return datetime.fromisoformat(match.group(1)).timestamp() if match else None


def _hf_status(path: Path) -> str:
    match = re.search(r"Status: (\w+)", path.read_text()) if path.exists() else None
    return match.group(1) if match else "NOT_STARTED"


def method_row(task: Path, method: str) -> dict:
    logs, run = task / "logs" / method, task / "runs" / method
    out = logs / "stdout.log"
    text = out.read_text(errors="ignore")[-50_000:] if out.exists() else ""
    rows = _tail_rows(run / "train" / "train_log.jsonl")
    step = int(rows[-1]["step"]) if rows else 0
    smoke_log = task / "smoke" / method / "smoke.log"
    smoke_text = smoke_log.read_text(errors="ignore") if smoke_log.exists() else ""
    smoke = "pass" if "SMOKE_PASS" in smoke_text else ("fail" if "SMOKE_FAIL" in smoke_text else "pending")
    if "EXIT_CODE=0" in text:
        state = "done"
    elif "EXIT_CODE=1" in text:
        state = "blocked"
    elif text:
        state = "training"
    else:
        state = "not started"
    eval_dir = task / "eval" / method
    if (eval_dir / "DONE").exists():
        state = "done (evaluated)" if state == "done" else state
    timed = [row["time_step"] for row in rows[-100:] if "time_step" in row]
    sec_per_update = sum(timed) / len(timed) if timed else None
    started = _first_attempt_time(out)
    wall_hours = (time.time() - started) / 3600 if started and state == "training" else None
    snapshots = sorted(path.name for path in (run / "train").glob("ckpt_step*.pt")) if (run / "train").exists() else []
    failure = ""
    errors = re.findall(r"last error: (.+)$", text, re.M)
    if errors and state != "done":
        failure = errors[-1].strip()
    blocked = re.findall(r"EXIT_CODE=1 \S+ (.+)$", text, re.M)
    if blocked:
        failure = blocked[-1].strip()
    stage = "-"
    if method == "dfly" and rows:
        stage = "stage1 LK/D-PACE" if step < DFLY_BOUNDARY else "stage2 e2e-TV"
    hf = task / "HF_UPLOAD_STATUS.md"
    return {
        "method": method,
        "label": LABELS.get(method, ""),
        "cpu_ready": "yes",
        "data_ready": "yes" if (task / "corpus" / "regen" / "checksums.txt").exists() else "no",
        "smoke": smoke,
        "state": state,
        "update": step,
        "epoch": min(3, step // PER_EPOCH + 1) if step else "-",
        "stage": stage,
        "sec_per_update": round(sec_per_update, 2) if sec_per_update else None,
        "eta_hours": round((TOTAL - step) * sec_per_update / 3600, 1) if sec_per_update and state == "training" else None,
        "gpu_hours": round(2 * wall_hours, 1) if wall_hours else None,
        "target_peak_gb": max((row.get("target_peak_gb", 0) for row in rows), default=None) or None,
        "train_peak_gb": max((row.get("train_peak_gb", 0) for row in rows), default=None) or None,
        "latest_checkpoint": "ckpt_last.pt" if (run / "train" / "ckpt_last.pt").exists() else "-",
        "snapshots": snapshots[-1] if snapshots else "-",
        "final_checkpoint": "ckpt_final.pt" if (run / "train" / "ckpt_final.pt").exists() else "-",
        "eval": "done" if (eval_dir / "DONE").exists() else ("running" if eval_dir.exists() else "pending"),
        "hf_upload": _hf_status(hf),
        "failure": failure.replace("|", "/")[:160] or "-",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args()
    rows = [method_row(args.task, method) for method in QUEUE]
    columns = ["method", "cpu_ready", "data_ready", "smoke", "state", "update", "epoch", "stage", "sec_per_update",
               "eta_hours", "gpu_hours", "target_peak_gb", "train_peak_gb", "latest_checkpoint", "snapshots",
               "final_checkpoint", "eval", "hf_upload", "failure"]
    lines = [
        "# STATUS: a100_qwen3vl8b_200k_s42",
        "",
        f"Updated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Budget: 200,000 train / 512 dev / 512 test requests, "
        f"3 epochs, {TOTAL:,} updates per row at global batch 8.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [f"{row['method']} ({row['label']})" if row["label"] and column == "method" else row[column]
                 for column in columns]
        if columns[5] == "update":
            cells[5] = f"{row['update']:,} / {TOTAL:,}"
        lines.append("| " + " | ".join("-" if cell in (None, "") else str(cell) for cell in cells) + " |")
    (args.task / "STATUS.md").write_text("\n".join(lines) + "\n")
    (args.task / "status.json").write_text(json.dumps(rows, indent=2))
    print("STATUS_WRITTEN")


if __name__ == "__main__":
    main()
