#!/usr/bin/env python3
"""Targeted replay of real historical render-drift rows under the non-roundtrip rule (contract section 19.5).

    python scripts/a100_qwen3vl8b_200k/replay_render_drift.py --run OLD_REGEN_RUN --out replay.json

``OLD_REGEN_RUN`` is a finished vLLM regen run whose finalize dropped rows as
``render_drift`` under the old token-prefix rule (``regen/skipped.jsonl``). Each
such row is rebuilt exactly as finalize built it (prompt ids + sampled answer
ids, the answer text decoded from them) and re-checked with the current
``_verify_full_render``: rows whose decoded text equals the render are now
accepted (``same_text``); rows whose text differs must still fail. The report
lists both, with the first differing decoded characters of every rejected row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from vlm_dflash.data.sharegpt import load_sharegpt
    from vlm_dflash.target.load import load_processor
    from vlm_dflash.target.regen_vllm import _verify_last_turn_render

    run = args.run
    cfg = json.loads((run / "config.json").read_text())
    contract = json.loads((run / "contract.json").read_text())
    processor = load_processor(cfg, contract)
    drift_ids = {str(entry["id"]) for entry in _jsonl(run / "regen" / "skipped.jsonl")
                 if entry.get("reason") == "render_drift"}
    rows, media_dir = load_sharegpt(run / "data" / "data.jsonl")
    rows = {str(row["id"]): row for row in rows if str(row["id"]) in drift_ids}
    requests = {str(r["id"]): r for r in _jsonl(run / "regen_vllm" / "requests.jsonl") if str(r["id"]) in drift_ids}
    outputs = {}
    for path in sorted((run / "regen_vllm").glob("outputs_*.jsonl")):
        for record in _jsonl(path):
            if str(record["id"]) in drift_ids:
                outputs[str(record["id"])] = record
    verdicts: Counter = Counter()
    rejected = []
    for row_id in sorted(drift_ids):
        request, output, row = requests[row_id], outputs[row_id], rows[row_id]
        prompt_ids = [int(token) for token in request["prompt_ids"]]
        answer_ids = [int(token) for token in output["answer_ids"]]
        answer_text = processor.tokenizer.decode(answer_ids[:-1], skip_special_tokens=False)
        try:
            kind = _verify_last_turn_render(processor, row, media_dir, prompt_ids + answer_ids,
                                            int(request["target_index"]), answer_text)
        except RuntimeError as error:
            verdicts["still_rejected"] += 1
            rejected.append({"id": row_id, "error": str(error)[:200]})
            continue
        verdicts[kind] += 1
    # Negative control on the same real rows: one answer token changed to another
    # text token must be rejected (changed decoded text is drift, not a split).
    controls: Counter = Counter()
    for row_id in sorted(drift_ids)[:20]:
        request, output, row = requests[row_id], outputs[row_id], rows[row_id]
        prompt_ids = [int(token) for token in request["prompt_ids"]]
        answer_ids = [int(token) for token in output["answer_ids"]]
        mutated = list(answer_ids)
        mutated[len(mutated) // 2] = processor.tokenizer.convert_tokens_to_ids("Z")
        answer_text = processor.tokenizer.decode(answer_ids[:-1], skip_special_tokens=False)
        try:
            _verify_last_turn_render(processor, row, media_dir, prompt_ids + mutated, int(request["target_index"]),
                                     answer_text)
            controls["wrongly_accepted"] += 1
        except RuntimeError:
            controls["rejected"] += 1
    report = {"run": str(run), "historical_render_drift_rows": len(drift_ids), "verdicts": dict(verdicts),
              "rejected": rejected, "negative_control_mutated_answers": dict(controls)}
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("historical_render_drift_rows", "verdicts",
                                                   "negative_control_mutated_answers")}))
    print("REPLAY_DONE")


if __name__ == "__main__":
    main()
