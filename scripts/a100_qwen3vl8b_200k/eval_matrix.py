#!/usr/bin/env python3
"""Evaluation of the a100_qwen3vl8b_200k_s42 matrix on one GPU (contract sections 26-30).

    # shared autoregressive baseline of one evaluation set (run once, reused by every method)
    CUDA_VISIBLE_DEVICES=0 python scripts/a100_qwen3vl8b_200k/eval_matrix.py ar --set mmspec \\
        --sets TASK/eval_sets --contract TASK/runs/dflash/contract.json --out TASK/eval/ar
    # speculative decoding of one checkpoint against that baseline
    CUDA_VISIBLE_DEVICES=0 python scripts/a100_qwen3vl8b_200k/eval_matrix.py sd --set mmspec \\
        --sets TASK/eval_sets --contract TASK/runs/dflash/contract.json --method dflash \\
        --checkpoint TASK/runs/dflash/train/ckpt_final.pt --verification sequential --ar TASK/eval/ar \\
        --out TASK/eval/dflash

Target and drafter share one GPU at batch 1 (the canonical single-A100 serving
layout of section 29), BF16, greedy. Every session turn is timed end to end
(processor, vision, host-to-device, prefill, draft, verify, output) with a CUDA
synchronize on both sides; image decoding from disk and model loading happen
before the clock starts. Multi-turn sessions (MMSpec) feed the model's own
earlier answers back as history, like MMSpec's harness. Two warm-up sessions
run first and are not recorded.

Outputs per run: ``<set>.<label>.jsonl`` (one record per session turn) and
``<set>.<label>.summary.json``: generated length, latency, task quality
(TextVQA VQA accuracy, DocVQA ANLS, ChartQA relaxed accuracy), and for SD
AL_draft (accepted draft tokens / verification rounds), AL_progress (emitted
tokens / rounds), rounds, target and drafter calls, token / sequence / stopping
exactness against the AR baseline, and the speedup sum(AR wall) / sum(SD wall)
over paired turns (the mean of per-turn ratios is reported separately).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

# --------------------------------------------------------------------------- #
# task-quality scorers
# --------------------------------------------------------------------------- #

_NUMBER_WORDS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!*#%^&$|~:']")


def normalize_vqa(text: str) -> str:
    """The VQA evaluation's answer normalization (punctuation, number words, articles)."""
    text = text.lower().replace("\n", " ").replace("\t", " ").strip()
    text = _PUNCT.sub(lambda match: "" if match.group(0) in "'" else " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    words = [_NUMBER_WORDS.get(word, word) for word in text.split() if word not in _ARTICLES]
    return " ".join(words)


def vqa_accuracy(prediction: str, answers: list[str]) -> float:
    """Official VQA accuracy: mean over leave-one-out annotator subsets of min(1, matches / 3)."""
    pred = normalize_vqa(prediction)
    gts = [normalize_vqa(answer) for answer in answers]
    scores = []
    for index in range(len(gts)):
        others = gts[:index] + gts[index + 1:]
        scores.append(min(1.0, sum(1 for gt in others if gt == pred) / 3.0))
    return sum(scores) / len(scores) if scores else 0.0


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str], threshold: float = 0.5) -> float:
    """Average normalized Levenshtein similarity (DocVQA), 0 below the threshold."""
    pred = " ".join(prediction.strip().lower().split())
    best = 0.0
    for answer in answers:
        gt = " ".join(answer.strip().lower().split())
        if not gt and not pred:
            return 1.0
        distance = _levenshtein(pred, gt) / max(len(pred), len(gt), 1)
        best = max(best, 1.0 - distance if distance < threshold else 0.0)
    return best


def relaxed_accuracy(prediction: str, answers: list[str], tolerance: float = 0.05) -> float:
    """ChartQA relaxed accuracy: numbers within 5% relative error, otherwise exact (case-insensitive) match."""
    pred = prediction.strip().rstrip(".").strip()

    def number(text: str) -> float | None:
        text = text.replace(",", "").replace("%", "").strip()
        try:
            return float(text)
        except ValueError:
            return None

    for answer in answers:
        target, value = number(answer), number(pred)
        if target is not None and value is not None:
            if target == 0.0 and value == 0.0:
                return 1.0
            if target != 0.0 and abs(value - target) / abs(target) <= tolerance:
                return 1.0
        elif pred.lower() == answer.strip().lower():
            return 1.0
    return 0.0


SCORERS = {"textvqa": vqa_accuracy, "docvqa": anls, "chartqa": relaxed_accuracy}

# --------------------------------------------------------------------------- #
# sessions
# --------------------------------------------------------------------------- #


def build_messages(row: dict, turn_index: int, history: list[tuple[str, str]]) -> list[dict]:
    """Chat messages for one turn: optional system, image on the first user turn, earlier turns as history."""
    messages = []
    if row.get("system"):
        messages.append({"role": "system", "content": [{"type": "text", "text": row["system"]}]})
    dataset_history = row.get("history")
    if turn_index == 0 and dataset_history and len(dataset_history) > 2:
        # Held-out multi-turn rows: the dataset's earlier turns verbatim, generate the last answer.
        for index, turn in enumerate(dataset_history[:-1]):
            text = str(turn["value"]).replace("<image>\n", "").replace("<image>", "").strip()
            if turn["from"] in ("human", "user"):
                content = ([{"type": "image"}] if index == 0 else []) + [{"type": "text", "text": text}]
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        return messages
    for index, (user, assistant) in enumerate(history):
        content = ([{"type": "image"}] if index == 0 else []) + [{"type": "text", "text": user}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})
    content = ([{"type": "image"}] if not history else []) + [{"type": "text", "text": row["turns"][turn_index]}]
    messages.append({"role": "user", "content": content})
    return messages


def stopped_naturally(tokens: list[int], stop_ids: set[int], cap: int) -> bool:
    return bool(tokens) and tokens[-1] in stop_ids and len(tokens) <= cap


class Runner:
    def __init__(self, args) -> None:
        from vlm_dflash.adapters import adapter_for_model
        from vlm_dflash.target.load import load_target

        self.contract = json.loads(Path(args.contract).read_text())
        cfg = {"target": {"dtype": "bfloat16", "attn_implementation": "sdpa", "device": "cuda:0",
                          "local_files_only": True}}
        started = time.perf_counter()
        self.model, self.processor = load_target(cfg, self.contract)
        self.load_seconds = time.perf_counter() - started
        self.adapter = adapter_for_model(self.model)
        stops = self.model.generation_config.eos_token_id
        self.stop_ids = set(stops if isinstance(stops, list) else [stops])
        self.device = next(self.model.parameters()).device
        self.draft = None

    def load_draft(self, method: str, checkpoint: str) -> None:
        from vlm_dflash.algorithms import get_algorithm

        algorithm = get_algorithm(method)
        # "parity": verification logits use the generate() SDPA backend (bf16 near-ties otherwise flip).
        algorithm.configure_verify_backend(False)
        started = time.perf_counter()
        draft, meta = algorithm.load_decoder(checkpoint)
        self.draft = draft.to(device=self.device, dtype=torch.bfloat16).eval()
        self.draft_load_seconds = time.perf_counter() - started
        self.draft_meta = {"step": meta.get("step"), "algorithm": meta.get("algorithm")} if isinstance(meta, dict) else {}

    def generate(self, messages: list[dict], image, cap: int, mode: str, verification: str | None) -> dict:
        from vlm_dflash.target.prompts import render_conversation

        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        text = render_conversation(self.processor, messages, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        prompt_len = int(inputs["input_ids"].shape[1])
        stats = {}
        with torch.inference_mode():
            if mode == "ar":
                output = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False, repetition_penalty=1.0)
            else:
                output, stats = self.draft.spec_generate(
                    target=self.model, input_ids=inputs["input_ids"], max_new_tokens=cap,
                    stop_token_ids=sorted(self.stop_ids), temperature=0.0,
                    target_kwargs=self.adapter.forward_kwargs_from_inputs(inputs), return_stats=True,
                    sequential_target_verification=verification == "sequential",
                    strict_parallel_margin=2.0 if verification == "strict_parallel" else None,
                )
        torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - started
        tokens = output[0, prompt_len:].tolist()
        record = {"prompt_tokens": prompt_len, "tokens": tokens, "new_tokens": len(tokens), "wall_seconds": wall,
                  "text": self.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                  "natural_stop": stopped_naturally(tokens, self.stop_ids, cap)}
        if stats:
            record.update({
                "rounds": int(stats["verify_steps"]), "accepted_draft_tokens": int(stats["accepted_tokens"]),
                "proposed_draft_tokens": int(stats["draft_tokens"]),
                "target_calls": int(stats.get("target_forward_calls", 0)),
                # The block-parallel decoder runs one draft forward per verification round.
                "drafter_calls": int(stats["verify_steps"]),
                "strict_rounds": int(stats.get("strict_parallel_rounds", 0) or 0),
                "strict_fallback_rounds": int(stats.get("strict_fallback_rounds", 0) or 0),
                **{key: float(value) for key, value in stats.items() if key.endswith("_seconds")},
            })
        return record


def run_set(runner: Runner, rows: list[dict], mode: str, verification: str | None, ar: dict | None) -> list[dict]:
    from PIL import Image

    out = []
    for warm in rows[:2]:  # warm-up, not recorded
        runner.generate(build_messages(warm, 0, []), Image.open(warm["image"]).convert("RGB"), 16, mode, verification)
    for row in rows:
        image = Image.open(row["image"]).convert("RGB")  # disk IO outside the timed boundary
        history: list[tuple[str, str]] = []
        turns = 1 if row.get("history") and len(row["history"]) > 2 else len(row["turns"])
        for turn_index in range(turns):
            record = runner.generate(build_messages(row, turn_index, history), image, int(row["max_new_tokens"]),
                                     mode, verification)
            record.update({"id": row["id"], "turn": turn_index, "topic": row.get("topic")})
            if row.get("answers") and row["topic"] in SCORERS:
                record["quality"] = SCORERS[row["topic"]](record["text"], row["answers"])
            if ar is not None:
                reference = ar.get((row["id"], turn_index))
                if reference is not None:
                    tokens, ref = record["tokens"], reference["tokens"]
                    record["token_exact_prefix"] = next((i for i, (a, b) in enumerate(zip(tokens, ref)) if a != b),
                                                        min(len(tokens), len(ref)))
                    record["sequence_exact"] = tokens == ref
                    record["stopping_exact"] = record["natural_stop"] == reference["natural_stop"] and len(tokens) == len(ref)
                    record["ar_wall_seconds"] = reference["wall_seconds"]
            history.append((row["turns"][turn_index], record["text"]))
            out.append(record)
    return out


def summarize(records: list[dict], mode: str) -> dict:
    summary = {"turns": len(records), "sessions": len({record["id"] for record in records}),
               "generated_tokens": sum(record["new_tokens"] for record in records),
               "mean_new_tokens": statistics.mean(record["new_tokens"] for record in records),
               "wall_seconds_total": sum(record["wall_seconds"] for record in records),
               "natural_stop_rate": statistics.mean(1.0 if record["natural_stop"] else 0.0 for record in records)}
    scored = [record["quality"] for record in records if "quality" in record]
    if scored:
        summary["quality"] = statistics.mean(scored)
    if mode == "sd":
        rounds = sum(record["rounds"] for record in records)
        summary.update({
            "rounds": rounds,
            "AL_draft": sum(record["accepted_draft_tokens"] for record in records) / max(1, rounds),
            "AL_progress": sum(record["new_tokens"] for record in records) / max(1, rounds),
            "acceptance_rate": sum(record["accepted_draft_tokens"] for record in records)
            / max(1, sum(record["proposed_draft_tokens"] for record in records)),
            "target_calls": sum(record["target_calls"] for record in records),
            "drafter_calls": sum(record["drafter_calls"] for record in records),
            "strict_fallback_rounds": sum(record["strict_fallback_rounds"] for record in records),
        })
        paired = [record for record in records if "ar_wall_seconds" in record]
        if paired:
            summary.update({
                "paired_turns": len(paired),
                "sequence_exact_rate": statistics.mean(1.0 if record["sequence_exact"] else 0.0 for record in paired),
                "token_exact": all(record["sequence_exact"] for record in paired),
                "stopping_exact_rate": statistics.mean(1.0 if record["stopping_exact"] else 0.0 for record in paired),
                "speedup": sum(record["ar_wall_seconds"] for record in paired)
                / max(1e-9, sum(record["wall_seconds"] for record in paired)),
                "speedup_mean_of_ratios": statistics.mean(record["ar_wall_seconds"] / record["wall_seconds"]
                                                          for record in paired),
                "failed_ids": sorted({record["id"] for record in paired if not record["sequence_exact"]})[:50],
            })
    by_topic = {}
    for topic in sorted({record.get("topic") for record in records if record.get("topic")}):
        subset = [record for record in records if record.get("topic") == topic]
        by_topic[topic] = summarize(subset, mode) if len({record.get("topic") for record in records}) > 1 else None
    if any(by_topic.values()):
        summary["by_topic"] = by_topic
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("ar", "sd"))
    parser.add_argument("--set", required=True)
    parser.add_argument("--sets", type=Path, required=True, help="eval_sets dir (eval_sets.py output)")
    parser.add_argument("--contract", required=True, help="a formal run's contract.json (target + processor pin)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method")
    parser.add_argument("--checkpoint")
    parser.add_argument("--verification", choices=("sequential", "parallel", "strict_parallel"), default="sequential")
    parser.add_argument("--ar", type=Path, help="dir of the shared AR baseline records")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open(args.sets / f"{args.set}.jsonl", encoding="utf-8")]
    rows = rows[: args.limit] if args.limit else rows
    runner = Runner(args)
    label = "ar" if args.mode == "ar" else f"{Path(args.checkpoint).stem}.{args.verification}"
    ar = None
    if args.mode == "sd":
        runner.load_draft(args.method, args.checkpoint)
        ar_path = args.ar / f"{args.set}.ar.jsonl"
        ar = {(record["id"], record["turn"]): record for record in map(json.loads, open(ar_path, encoding="utf-8"))}
    records = run_set(runner, rows, args.mode, args.verification if args.mode == "sd" else None, ar)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / f"{args.set}.{label}.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = summarize(records, args.mode)
    summary.update({"set": args.set, "mode": args.mode, "label": label, "method": args.method,
                    "checkpoint": args.checkpoint, "verification": args.verification if args.mode == "sd" else None,
                    "target_load_seconds": runner.load_seconds,
                    "draft_load_seconds": getattr(runner, "draft_load_seconds", None),
                    "draft_meta": getattr(runner, "draft_meta", None),
                    "device": torch.cuda.get_device_name(0), "batch_size": 1, "dtype": "bfloat16",
                    "timing_boundary": "processor + vision + H2D + prefill + draft + verify + output; image decode and "
                                       "model load excluded; 2 warm-up sessions"})
    (args.out / f"{args.set}.{label}.summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary.get(key) for key in ("set", "label", "turns", "quality", "AL_draft", "AL_progress",
                                                         "speedup", "token_exact", "sequence_exact_rate")}))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
