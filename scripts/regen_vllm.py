#!/usr/bin/env python3
"""vLLM regeneration for ``regen.backend: vllm`` (see vlm_dflash/target/regen_vllm.py).

Training env (the env ``vlm-dflash train`` uses):

    python scripts/regen_vllm.py prepare-data --config CFG --run-dir RUN
    python scripts/regen_vllm.py requests     --config CFG --run-dir RUN --workers 32

vLLM env, one process per tensor-parallel GPU group (resumable; shards are
disjoint by row index):

    python scripts/regen_vllm.py generate --run-dir RUN --shard 0 --num-shards 8 --tp 2

Training env again, once every shard reports GENERATE_DONE:

    python scripts/regen_vllm.py finalize --config CFG --run-dir RUN

``prepare-data`` and ``finalize`` write the same stage manifests (inputs
digests) as ``vlm-dflash train``, so a training run whose ``dataset`` and
``regen`` blocks match reuses ``data/`` and ``regen/`` without rebuilding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _train_cfg(args) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from vlm_dflash.cli import build_train_config

    namespace = Namespace(
        preset=args.preset, config=args.config, target=None, dataset=None, hf_dataset=None, hf_split=None,
        dataset_source=None, annotation=None, images_dir=None, limit=None, algorithm=None, set_list=args.set_list,
        feature_mode=None,
    )
    return build_train_config(namespace)


def cmd_prepare_data(args) -> None:
    from vlm_dflash.cli import stage_digest, stage_digest_payload, stage_status, write_stage_manifest
    from vlm_dflash.contract import ensure_contract
    from vlm_dflash.data.prepare import build_dataset
    from vlm_dflash.config import dump_effective_config

    cfg = _train_cfg(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_effective_config(cfg, run_dir / "config.json")
    contract = ensure_contract(cfg, run_dir)
    digest = stage_digest(cfg, contract, "data", None)
    if stage_status(run_dir, "data", "data/data.jsonl", digest, args.force) == "run":
        build_dataset(cfg, run_dir / "data")
        write_stage_manifest(run_dir, "data", stage_digest_payload(cfg, contract, "data", None), digest)
    print(f"DATA_STAGE_DONE digest={digest[:12]}")


def cmd_requests(args) -> None:
    from vlm_dflash.contract import ensure_contract
    from vlm_dflash.target.regen_vllm import build_requests

    cfg = _train_cfg(args)
    contract = ensure_contract(cfg, Path(args.run_dir))
    print(json.dumps(build_requests(cfg, args.run_dir, contract, workers=args.workers)))
    print("REQUESTS_DONE")


def cmd_finalize(args) -> None:
    from vlm_dflash.cli import stage_digest, stage_digest_payload, write_stage_manifest
    from vlm_dflash.contract import ensure_contract
    from vlm_dflash.target.regen import run_regen

    cfg = _train_cfg(args)
    run_dir = Path(args.run_dir)
    contract = ensure_contract(cfg, run_dir)
    if str((cfg.get("regen") or {}).get("backend", "hf")).lower() != "vllm":
        raise SystemExit("finalize: the config's regen.backend is not 'vllm'.")
    data_digest = stage_digest(cfg, contract, "data", None)
    stats = run_regen(cfg, run_dir, contract)
    write_stage_manifest(
        run_dir, "regen", stage_digest_payload(cfg, contract, "regen", data_digest),
        stage_digest(cfg, contract, "regen", data_digest),
    )
    print(json.dumps({k: stats[k] for k in ("num_input", "num_kept", "num_surplus", "skip_reasons")}))
    print("FINALIZE_DONE")


def _request_seed(seed: int, row_id: str) -> int:
    # Recorded as the seed policy in regen_stats.json; this side never imports the package.
    return int(hashlib.sha256(f"{seed}:{row_id}".encode()).hexdigest()[:8], 16)


def cmd_generate(args) -> None:
    import vllm
    from PIL import Image
    from vllm import LLM, SamplingParams

    run_dir = Path(args.run_dir)
    in_dir = run_dir / "regen_vllm"
    sampling = json.loads((in_dir / "sampling.json").read_text())
    out_path = in_dir / f"outputs_{args.shard:03d}.jsonl"
    done: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(str(json.loads(line)["id"]))
            except (ValueError, KeyError):
                continue  # torn last line from an interrupted run; the row is redone
    todo = []
    with (in_dir / "requests.jsonl").open() as handle:
        for index, line in enumerate(handle):
            if index % args.num_shards != args.shard:
                continue
            request = json.loads(line)
            if request.get("skip") or str(request["id"]) in done:
                continue
            todo.append(request)
    print(f"shard {args.shard}/{args.num_shards}: {len(todo)} to generate, {len(done)} already done", flush=True)

    window = int(sampling["window_tokens"])
    llm = LLM(
        model=sampling["model_path"],
        tensor_parallel_size=args.tp,
        dtype="bfloat16",
        max_model_len=window,
        seed=int(sampling["seed"]),
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs={
            "size": {"shortest_edge": sampling["image_min_pixels"], "longest_edge": sampling["image_max_pixels"]}
        },
        max_num_seqs=args.max_num_seqs,
    )
    meta = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "tensor_parallel_size": args.tp,
        "vllm": vllm.__version__,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_seqs": args.max_num_seqs,
    }
    try:
        import torch
        import transformers

        meta.update(torch=torch.__version__, transformers=transformers.__version__,
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except Exception:  # metadata only
        pass
    (in_dir / f"generate_meta_{args.shard:03d}.json").write_text(json.dumps(meta, indent=2))

    started, produced = time.time(), 0
    with out_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start : start + args.chunk]
            prompts, params = [], []
            for request in chunk:
                images = [Image.open(path).convert("RGB") for path in request["images"]]
                prompts.append({"prompt": request["prompt"], "multi_modal_data": {"image": images[0]}})
                params.append(
                    SamplingParams(
                        n=1,
                        temperature=float(sampling["temperature"]),
                        top_p=float(sampling["top_p"]),
                        top_k=int(sampling["top_k"]),
                        presence_penalty=float(sampling["presence_penalty"]),
                        repetition_penalty=float(sampling["repetition_penalty"]),
                        max_tokens=min(int(sampling["max_new_tokens"]), window - int(request["prompt_tokens"])),
                        seed=_request_seed(int(sampling["seed"]), str(request["id"])),
                        stop_token_ids=list(sampling["stop_token_ids"]),
                        skip_special_tokens=False,
                        detokenize=False,
                    )
                )
            results = llm.generate(prompts, params, use_tqdm=False)
            for request, result in zip(chunk, results):
                completion = result.outputs[0]
                vllm_ids = list(result.prompt_token_ids or [])
                record = {
                    "id": request["id"],
                    "prompt_match": vllm_ids == request["prompt_ids"],
                    "vllm_prompt_tokens": len(vllm_ids),
                    "answer_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                }
                if not record["prompt_match"]:
                    record["vllm_prompt_ids"] = vllm_ids
                handle.write(json.dumps(record) + "\n")
                produced += 1
            handle.flush()
            rate = produced / max(1e-6, time.time() - started)
            print(f"shard {args.shard}: {produced}/{len(todo)} done ({rate:.1f} rows/s)", flush=True)
    print(f"GENERATE_DONE shard={args.shard}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "requests", "finalize"):
        step = sub.add_parser(name)
        step.add_argument("--config", required=True)
        step.add_argument("--preset", default="single-gpu-48g")
        step.add_argument("--run-dir", required=True)
        step.add_argument("--set", dest="set_list", action="append", default=[], help="dotted key=value override")
        if name == "prepare-data":
            step.add_argument("--force", action="store_true")
        if name == "requests":
            step.add_argument("--workers", type=int, default=16)
    gen = sub.add_parser("generate")
    gen.add_argument("--run-dir", required=True)
    gen.add_argument("--shard", type=int, required=True)
    gen.add_argument("--num-shards", type=int, required=True)
    gen.add_argument("--tp", type=int, default=2)
    gen.add_argument("--chunk", type=int, default=2048, help="requests per llm.generate call")
    gen.add_argument("--max-num-seqs", type=int, default=256)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args()
    {"prepare-data": cmd_prepare_data, "requests": cmd_requests, "generate": cmd_generate,
     "finalize": cmd_finalize}[args.command](args)


if __name__ == "__main__":
    main()
