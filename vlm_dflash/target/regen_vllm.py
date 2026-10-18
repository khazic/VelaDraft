"""Target-native regeneration through vLLM (``regen.backend: vllm``).

vLLM runs in its own environment, so the stage is split in three steps that
communicate through ``<run>/regen_vllm/``:

1. :func:`build_requests` (training env): renders every prompt with the same
   prompts.py path, processor and chat-template kwargs that capture/online
   extraction use, and stores the exact prompt token ids.
2. ``scripts/regen_vllm.py generate`` (vLLM env): samples answers for a shard
   of requests and records, per row, whether vLLM's own prompt token ids equal
   the stored ones.
3. :func:`finalize_vllm_regen` (training env, called by ``run_regen``): turns
   the outputs into the regular ``regen/target_data.jsonl`` records. Rows whose
   prompt render differs, whose answer hit the length cap (never converted into
   a natural-EOS example), that are too short, or whose prompt exceeds
   ``regen.max_prompt_tokens`` are dropped with a reason; with
   ``regen.select`` the kept rows are then cut to exact per-group quotas in the
   pool's frozen order (rows past a quota are recorded as surplus).

Supervision follows ``last_turn_only`` for multi-turn rows: the dataset's
earlier turns stay as context and only the final answer is regenerated.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ..data.sharegpt import (
    load_sharegpt,
    row_image_path,
    row_image_paths,
    row_image_turns,
    row_is_multi_turn,
    row_question,
)
from ..provenance import identity_from_contract, record_stage
from ..utils import ensure_dir, log, read_json, read_jsonl, write_json, write_jsonl
from .prompts import messages_from_row, render_conversation, render_prompt

#: Sampling knobs forwarded to vLLM's ``SamplingParams`` (recorded verbatim).
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty")


def request_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "regen_vllm"


def target_stop_ids(model_path: str | Path) -> list[int]:
    """EOS ids from the target's generation config (first entry = primary stop)."""
    eos = read_json(Path(model_path) / "generation_config.json").get("eos_token_id")
    return [int(token) for token in (eos if isinstance(eos, list) else [eos]) if token is not None]


def _last_assistant_index(row: dict) -> int:
    indices = [
        index
        for index, turn in enumerate(row.get("conversations") or [])
        if str(turn.get("from", "")).lower() in ("gpt", "assistant", "model")
    ]
    if not indices:
        raise ValueError(f"Row {row.get('id')!r} has no assistant turn.")
    return indices[-1]


def _render_request(processor, row: dict, media_dir: Path) -> dict:
    """Prompt text, image paths and HF prompt token ids for one row."""
    if row_is_multi_turn(row):
        target_index = _last_assistant_index(row)
        messages, image_indices = messages_from_row(row, upto=target_index)
        text = render_conversation(processor, messages, add_generation_prompt=True)
        paths = [row_image_paths(row, media_dir)[position] for position in image_indices]
        kind = "v2"
    else:
        target_index = None
        text = render_prompt(processor, row_question(row))
        paths = [row_image_path(row, media_dir)]
        kind = "v1"
    images = [Image.open(path).convert("RGB") for path in paths]
    ids = processor(text=[text], images=images, return_tensors="pt")["input_ids"][0].tolist()
    return {
        "id": row.get("id"),
        "kind": kind,
        "target_index": target_index,
        "prompt": text,
        "images": [str(path) for path in paths],
        "prompt_ids": ids,
        "prompt_tokens": len(ids),
    }


_WORKER: dict = {}


def _worker_init(cfg: dict, contract: dict, media_dir: str) -> None:
    from .load import load_processor

    _WORKER.update(processor=load_processor(cfg, contract), media_dir=Path(media_dir))


def _worker_render(row: dict) -> dict:
    return _render_request(_WORKER["processor"], row, _WORKER["media_dir"])


def build_requests(cfg: dict, run_dir: str | Path, contract: dict, *, workers: int = 16) -> dict:
    """Render every ``data/data.jsonl`` row into ``regen_vllm/requests.jsonl``."""
    from concurrent.futures import ProcessPoolExecutor

    run_dir = Path(run_dir)
    regen_cfg = cfg.get("regen", {}) or {}
    max_prompt = int(regen_cfg.get("max_prompt_tokens", 2048))
    image_token_id = int(contract["tokens"]["image_token_id"])
    max_image_tokens = contract["image"].get("max_tokens")
    rows, media_dir = load_sharegpt(run_dir / "data" / "data.jsonl")
    out_dir = ensure_dir(request_dir(run_dir))
    counts: Counter = Counter()
    with (out_dir / "requests.jsonl").open("w", encoding="utf-8") as handle, ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(cfg, contract, str(media_dir))
    ) as pool:
        for done, request in enumerate(pool.map(_worker_render, rows, chunksize=64), start=1):
            image_tokens = sum(1 for token in request["prompt_ids"] if token == image_token_id)
            # One image per row: a count above the contract bound means the pixel
            # bounds were not applied, which would desynchronize every stage.
            if max_image_tokens is not None and image_tokens > int(max_image_tokens) * len(request["images"]):
                raise RuntimeError(
                    f"Row {request['id']!r} renders {image_tokens} image tokens > contract bound "
                    f"{max_image_tokens}; the processor ignored the pinned pixel bounds."
                )
            if request["prompt_tokens"] > max_prompt:
                request["skip"] = "prompt_too_long"
            counts[request.get("skip", "ok")] += 1
            handle.write(json.dumps(request) + "\n")
            if done % 20000 == 0:
                log(f"regen_vllm: rendered {done}/{len(rows)} requests")
    stats = {"num_rows": len(rows), "max_prompt_tokens": max_prompt, "counts": dict(counts)}
    write_json(out_dir / "requests_stats.json", stats)
    # Everything the vLLM-side generator needs, so it never imports this package.
    write_json(
        out_dir / "sampling.json",
        {
            **{key: regen_cfg.get(key) for key in SAMPLING_KEYS},
            "max_new_tokens": int(regen_cfg.get("max_new_tokens", 2048)),
            "window_tokens": int(regen_cfg.get("window_tokens", 4096)),
            "seed": int(regen_cfg.get("seed", 42)),
            "stop_token_ids": target_stop_ids(contract["target"]["model_path"]),
            "model_path": contract["target"]["model_path"],
            "image_min_pixels": int(contract["image"]["min_pixels"]),
            "image_max_pixels": int(contract["image"]["max_pixels"]),
        },
    )
    return stats


def _load_outputs(out_dir: Path) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for path in sorted(out_dir.glob("outputs_*.jsonl")):
        for record in read_jsonl(path):
            outputs[str(record["id"])] = record
    return outputs


def _verify_last_turn_render(processor, row: dict, media_dir: Path, full_ids: list[int], target_index: int,
                             answer_text: str) -> None:
    from .regen import _verify_full_render

    images = [Image.open(path).convert("RGB") for path in row_image_paths(row, media_dir)]
    _verify_full_render(processor, row, images, full_ids, {target_index: answer_text})


def finalize_vllm_regen(cfg: dict, run_dir: str | Path, contract: dict, limit: int | None = None) -> dict:
    """Build ``regen/`` from vLLM outputs (the ``run_regen`` body for ``backend: vllm``)."""
    from .load import load_processor
    from .regen import _length_summary

    run_dir = Path(run_dir)
    regen_cfg = cfg.get("regen", {}) or {}
    min_response_tokens = int(regen_cfg.get("min_response_tokens", 8))
    max_new_tokens = int(regen_cfg.get("max_new_tokens", 2048))
    rows, media_dir = load_sharegpt(run_dir / "data" / "data.jsonl")
    if limit is not None:
        rows = rows[:limit]
    in_dir = request_dir(run_dir)
    requests = {str(record["id"]): record for record in read_jsonl(in_dir / "requests.jsonl")}
    outputs = _load_outputs(in_dir)
    pending = [str(row["id"]) for row in rows if not requests.get(str(row["id"]), {}).get("skip")
               and str(row["id"]) not in outputs]
    if pending:
        raise RuntimeError(
            f"regen.backend=vllm: {len(pending)} row(s) have no vLLM output (first: {pending[:3]}); "
            "finish `scripts/regen_vllm.py generate` for every shard before finalizing."
        )
    processor = load_processor(cfg, contract)
    stop_ids = target_stop_ids(contract["target"]["model_path"])
    stop_set = set(stop_ids)

    kept, skipped = [], []
    for row in rows:
        row_id = str(row["id"])
        request = requests[row_id]
        if request.get("skip"):
            skipped.append({"id": row_id, "reason": request["skip"], "prompt_tokens": request["prompt_tokens"]})
            continue
        output = outputs[row_id]
        answer_ids = [int(token) for token in output["answer_ids"]]
        reason = None
        if not output.get("prompt_match"):
            reason = "prompt_render_mismatch"
        elif output.get("finish_reason") != "stop":
            reason = "length_capped"
        else:
            if not answer_ids or answer_ids[-1] not in stop_set:
                stop = output.get("stop_reason")
                answer_ids.append(int(stop) if isinstance(stop, int) and stop in stop_set else stop_ids[0])
            if len(answer_ids) <= 1:
                reason = "empty_answer"
            elif len(answer_ids) < min_response_tokens:
                reason = "too_short"
        if reason is not None:
            skipped.append({"id": row_id, "reason": reason, "answer_tokens": len(answer_ids)})
            continue
        prompt_ids = [int(token) for token in request["prompt_ids"]]
        answer_text = processor.tokenizer.decode(answer_ids[:-1], skip_special_tokens=False)
        if request["kind"] == "v1":
            record = {
                "id": row_id,
                "conversations": [
                    {"from": "human", "value": row_question(row)},
                    {"from": "gpt", "value": answer_text},
                ],
                "images": row["images"],
                "regen": {"prompt_ids": prompt_ids, "answer_ids": answer_ids, "stop_token_id": answer_ids[-1]},
            }
        else:
            target_index = int(request["target_index"])
            full_ids = prompt_ids + answer_ids
            _verify_last_turn_render(processor, row, media_dir, full_ids, target_index, answer_text)
            record = {
                "id": row_id,
                "conversations": row["conversations"],
                "images": row["images"],
                "image_turns": row_image_turns(row),
                "schema": "vlm-dflash-data-v2",
                "regen": {
                    "mode": "last_turn_only",
                    "assistant_indices": [target_index],
                    "turns": [
                        {
                            "prompt_ids": prompt_ids,
                            "answer_ids": answer_ids,
                            "answer_text": answer_text,
                            "stop_token_id": answer_ids[-1],
                            "forced_stop": False,
                        }
                    ],
                    "full_ids": full_ids,
                    "answer_spans": [[len(prompt_ids), len(full_ids)]],
                    "n_turns": 1,
                    "forced_stop": 0,
                },
            }
        kept.append(record)

    selection, surplus, group_of = _select_quotas(regen_cfg.get("select"), kept)
    if selection is not None:
        kept_ids = {record["id"] for record in selection}
        kept = [record for record in kept if record["id"] in kept_ids]

    def answer_len(record: dict) -> int:
        regen = record["regen"]
        return len(regen["turns"][0]["answer_ids"]) if "turns" in regen else len(regen["answer_ids"])

    def prompt_len(record: dict) -> int:
        regen = record["regen"]
        return len(regen["turns"][0]["prompt_ids"]) if "turns" in regen else len(regen["prompt_ids"])

    regen_dir = ensure_dir(run_dir / "regen")
    write_jsonl(regen_dir / "target_data.jsonl", kept)
    write_jsonl(regen_dir / "skipped.jsonl", skipped)
    write_jsonl(regen_dir / "surplus.jsonl", [{"id": record["id"]} for record in surplus])
    vllm_meta = [read_json(path) for path in sorted(in_dir.glob("generate_meta_*.json"))]
    stats = {
        "backend": "vllm",
        "num_input": len(rows),
        "num_valid": len(kept) + len(surplus),
        "num_kept": len(kept),
        "num_surplus": len(surplus),
        "num_skipped": len(skipped),
        "skip_reasons": dict(Counter(entry["reason"] for entry in skipped)),
        "max_new_tokens": max_new_tokens,
        "min_response_tokens": min_response_tokens,
        "max_prompt_tokens": int(regen_cfg.get("max_prompt_tokens", 2048)),
        "greedy": False,
        "generation_protocol": {
            "backend": "vllm",
            "do_sample": True,
            "num_beams": 1,
            **{key: regen_cfg.get(key) for key in SAMPLING_KEYS},
            "max_new_tokens": max_new_tokens,
            "window_tokens": int(regen_cfg.get("window_tokens", 4096)),
            "eos_token_ids": sorted(stop_set),
            "seed": int(regen_cfg.get("seed", 42)),
            "seed_policy": "per-row: sha256('<seed>:<row id>')[:8]",
            "chat_template_kwargs": (contract.get("prompt") or {}).get("chat_template_kwargs", {}),
            "vllm": vllm_meta,
        },
        "multi_turn_mode": "last_turn_only",
        "answer_token_histogram": _length_summary([answer_len(r) for r in kept], cap=max_new_tokens),
        "prompt_token_histogram": _length_summary([prompt_len(r) for r in kept], cap=None),
        "selection": regen_cfg.get("select"),
    }
    if selection is not None:
        stats["kept_per_group"] = dict(Counter(group_of[record["id"]] for record in kept))
        stats["surplus_per_group"] = dict(Counter(group_of[record["id"]] for record in surplus))
    write_json(regen_dir / "regen_stats.json", stats)
    record_stage(
        run_dir,
        "regen",
        {"identity": identity_from_contract(contract), "stats": stats, "code": "vlm_dflash.target.regen_vllm"},
    )
    if not kept:
        raise RuntimeError("Regen kept zero samples; check the vLLM outputs and filters.")
    log(f"regen_vllm: kept {len(kept)} rows, surplus {len(surplus)}, skipped {dict(stats['skip_reasons'])}")
    return stats


def _select_quotas(
    select: dict | None, kept: list[dict]
) -> tuple[list[dict] | None, list[dict], dict[str, str]]:
    """Exact per-group quotas in the pool's frozen ``pool_rank`` order (``regen.select``).

    Returns ``(selected or None, surplus, row id -> group)``.
    """
    if not select:
        return None, [], {}
    quotas = {str(name): int(value) for name, value in select["quotas"].items()}
    meta = {str(record["id"]): record for record in read_jsonl(select["meta"])}
    group_of: dict[str, str] = {}
    by_group: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for record in kept:
        info = meta.get(str(record["id"]))
        if info is None:
            raise RuntimeError(f"regen.select: row {record['id']!r} is missing from {select['meta']}.")
        group_of[record["id"]] = str(info["group"])
        by_group[str(info["group"])].append((int(info["pool_rank"]), record))
    selected, surplus = [], []
    short = {}
    for name, quota in quotas.items():
        ordered = [record for _, record in sorted(by_group.get(name, []), key=lambda item: item[0])]
        if len(ordered) < quota:
            short[name] = {"valid": len(ordered), "quota": quota}
        selected.extend(ordered[:quota])
        surplus.extend(ordered[quota:])
    if short:
        raise RuntimeError(
            f"regen.select: groups below quota {short}; regenerate additional rows from the pool reserve "
            "before finalizing (the corpus must not shrink)."
        )
    return selected, surplus, group_of
