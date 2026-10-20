#!/usr/bin/env python3
"""GPU preflight of one split-device worker for the Qwen3-VL-8B matrix (contract sections 3, 6-10, 14, 16, 20-22).

Run on one GPU pair (``CUDA_VISIBLE_DEVICES=a,b``): logical ``cuda:1`` holds the
frozen target, ``cuda:0`` the drafter, exactly like a formal row.

    python scripts/a100_qwen3vl8b_200k/preflight_gpu.py \\
        --model "$VLM_DFLASH_TARGET_QWEN3VL" --onevision-root "$ONEVISION_ROOT" \\
        --scratch "$SCRATCH" --out "$TASK_ROOT/preflight/gpu"

Everything goes through the repo's own load / render / capture / trainer code.
Writes ``preflight_gpu.json`` and prints ``PREFLIGHT_GPU_PASS`` or
``PREFLIGHT_GPU_FAIL <gates>``; a failed mandatory gate exits non-zero.
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import matrix  # noqa: E402
from vlm_dflash.algorithms import get_algorithm  # noqa: E402
from vlm_dflash.contract import contract_defaults_from_config, resolve_contract  # noqa: E402
from vlm_dflash.dflash.features import collate_features  # noqa: E402
from vlm_dflash.target.load import load_target  # noqa: E402
from vlm_dflash.target.online import OnlineFeatureDataset  # noqa: E402
from vlm_dflash.target.prompts import encode_prompt  # noqa: E402
from vlm_dflash.target.sample_capture import (  # noqa: E402
    compute_multimodal_positions,
    extract_target_head,
    finalize_prepared_records,
    prepare_sample_record,
)

#: Expected text contract (section 6.2); the downloaded config is the truth.
EXPECTED_TEXT = {
    "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36, "num_attention_heads": 32,
    "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
}
LAYER_IDS = [1, 17, 33]
#: OneVision subsets for the real-image samples (one per coverage group).
SAMPLE_SUBSETS = {
    "chart": "chartqa(cauldron,llava_format)",
    "doc": "infographic_vqa",
    "scene": "sharegpt4v(coco)",
    "reasoning": "geo170k(qa)",
}

report: dict = {"gates": {}}


def gate(name: str, ok: bool, detail=None, mandatory: bool = True) -> bool:
    report["gates"][name] = {"pass": bool(ok), "mandatory": mandatory, "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {json.dumps(detail, default=str)[:400]}", flush=True)
    return ok


def sync() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def first_row(root: str, subset: str) -> dict:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(f"{root}/{glob.escape(subset)}/**/*.parquet", recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files for OneVision subset {subset!r}")
    return pq.ParquetFile(files[0]).read_row_group(0).slice(0, 1).to_pylist()[0]


def clean_question(text: str) -> str:
    return text.replace("<image>\n", "").replace("<image>", "").strip()


def make_row(processor, name: str, image, question: str, answer: str, media_dir: Path) -> dict:
    """A v1 regen row (what regen/target_data.jsonl holds) for one real image."""
    image_name = f"{name}.png"
    image.save(media_dir / image_name)
    prompt_ids = encode_prompt(processor, question, image)["input_ids"][0].tolist()
    answer_ids = processor.tokenizer(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return {
        "id": name,
        "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
        "images": [image_name],
        "regen": {"prompt_ids": prompt_ids, "answer_ids": answer_ids},
    }


def capture(model, processor, rows: list[dict], media_dir: Path, contract: dict) -> list[dict]:
    """The online source's own path: render, one (batched) target forward, normalize."""
    prepared = [prepare_sample_record(processor, row, media_dir, seq_length=4096) for row in rows]
    records = finalize_prepared_records(
        model, prepared, layer_ids=LAYER_IDS, image_token_id=int(contract["tokens"]["image_token_id"]),
        merge_size=contract["image"].get("merge_size"), dtype=torch.bfloat16, store_final_hidden=True,
        tokenizer=processor.tokenizer,
    )
    return [OnlineFeatureDataset._normalize(record) for record in records]


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def env_section() -> None:
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    caps = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    peer = torch.cuda.device_count() > 1 and torch.cuda.can_device_access_peer(1, 0)
    a = torch.randn(2048, 2048, device="cuda:0")
    b = torch.randn(2048, 2048, device="cuda:0")
    ref = a @ b
    bf16 = (a.bfloat16() @ b.bfloat16()).float()
    rel = float((bf16 - ref).norm() / ref.norm())
    import transformers

    report["env"] = {
        "python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__, "triton": _version("triton"), "devices": names,
        "capability": caps, "p2p_cuda1_to_cuda0": bool(peer), "bf16_matmul_rel_error": rel,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    gate("cuda_pair", torch.cuda.device_count() == 2 and all("A100" in name or "A800" in name for name in names),
         {"devices": names})
    gate("bf16_matmul", rel < 1e-2, {"rel_error": rel})


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception as error:  # recorded, not fatal
        return f"unavailable: {error.__class__.__name__}"


def contract_section(args) -> dict:
    cfg = matrix.build("dflash")
    cfg["target"]["model"] = args.model
    defaults = contract_defaults_from_config(cfg)
    contract = resolve_contract(args.model, defaults, out_path=Path(args.out) / "contract.json")
    from transformers import AutoConfig

    text = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    actual = {key: int(getattr(text, key)) for key in EXPECTED_TEXT}
    mismatch = {key: [EXPECTED_TEXT[key], actual[key]] for key in EXPECTED_TEXT if actual[key] != EXPECTED_TEXT[key]}
    report["contract"] = {
        "model_type": contract["target"]["model_type"], "text_actual": actual, "text_mismatch": mismatch,
        "layers": contract["layers"], "rope": contract["rope"], "mask": contract["mask"], "image": contract["image"],
        "tokens": contract["tokens"], "head": contract["head"], "prompt": contract["prompt"],
    }
    gate("target_family", contract["target"]["model_type"] == "qwen3_vl", contract["target"]["model_type"])
    gate("text_contract", not mismatch, mismatch or actual)
    gate("layer_routing", contract["layers"]["draft_layer_ids"] == LAYER_IDS, contract["layers"]["draft_layer_ids"])
    gate("mrope", contract["rope"]["sections"] == [24, 20, 20] and contract["rope"]["interleaved"] is True,
         {key: contract["rope"][key] for key in ("sections", "interleaved", "mode")})
    generation = json.loads((Path(args.model) / "generation_config.json").read_text())
    special = {
        "eos": generation.get("eos_token_id"), "pad": generation.get("pad_token_id"),
        "bos": generation.get("bos_token_id"), "image": contract["tokens"]["image_token_id"],
    }
    flat = {int(v) for value in special.values() for v in (value if isinstance(value, list) else [value]) if v is not None}
    mask_id = int(contract["mask"]["token_id"])
    gate("mask_token", mask_id == 151669 and mask_id not in flat and contract["mask"]["token_str"] is None,
         {"mask": mask_id, "tokenizer_len": contract["mask"]["tokenizer_vocab_size"], "special": special})
    return contract


def target_section(args, contract: dict):
    cfg = matrix.build("dflash")
    cfg["target"]["model"] = args.model
    started = time.perf_counter()
    model, processor = load_target(cfg, contract)
    load_seconds = time.perf_counter() - started
    devices = sorted({str(p.device) for p in model.parameters()})
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    report["target"] = {
        "class": type(model).__name__, "load_seconds": round(load_seconds, 1), "devices": devices,
        "params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 3), "training": model.training,
        "trainable_tensors": len(trainable), "mem_gb_cuda1": round(torch.cuda.memory_allocated(1) / 2**30, 2),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    }
    gate("target_load", type(model).__name__ == "Qwen3VLForConditionalGeneration" and devices == ["cuda:1"],
         report["target"])
    gate("target_frozen", not trainable and not model.training, {"trainable": trainable[:3]})
    embed = model.get_input_embeddings().weight
    norms = embed.float().norm(dim=-1)
    report["mask_row"] = {"norm": float(norms[151669]), "median_vocab_norm": float(norms[:151643].median()),
                          "rows": int(embed.shape[0])}
    return model, processor


def real_image_section(args, model, processor, contract: dict, media_dir: Path) -> list[dict]:
    from PIL import Image

    rows = []
    for name, subset in SAMPLE_SUBSETS.items():
        raw = first_row(args.onevision_root, subset)
        image = Image.open(io.BytesIO(raw["image"]["bytes"])).convert("RGB")
        conv = raw["conversations"]
        rows.append(make_row(processor, name, image, clean_question(conv[0]["value"]), conv[1]["value"], media_dir))
    image_token = int(contract["tokens"]["image_token_id"])
    details = {}
    for row in rows:
        image = Image.open(media_dir / row["images"][0]).convert("RGB")
        inputs = encode_prompt(processor, row["conversations"][0]["value"], image)
        ids = inputs["input_ids"][0]
        grid = inputs["image_grid_thw"]
        merge = int(contract["image"]["merge_size"])
        expected = int(grid.prod(-1).sum()) // (merge * merge)
        positions = compute_multimodal_positions(
            model, inputs["input_ids"].to("cuda:1"), inputs["attention_mask"].to("cuda:1"), grid.to("cuda:1"),
            mm_token_type_ids=inputs.get("mm_token_type_ids").to("cuda:1") if "mm_token_type_ids" in inputs else None,
        )
        details[row["id"]] = {
            "image_size": list(image.size), "prompt_tokens": int(ids.numel()),
            "image_tokens": int((ids == image_token).sum()), "expected_image_tokens": expected,
            "grid_thw": grid.tolist(), "pixel_values": list(inputs["pixel_values"].shape),
            "has_mm_token_type_ids": "mm_token_type_ids" in inputs, "position_ids": list(positions.shape),
            "positions_differ_from_1d": bool((positions[1] != positions[0]).any()),
        }
    report["real_images"] = details
    report["chat_template_head"] = processor.tokenizer.decode(rows[0]["regen"]["prompt_ids"][:12])
    gate("real_image_processor", all(
        d["image_tokens"] == d["expected_image_tokens"] > 0 and d["has_mm_token_type_ids"] and d["image_tokens"] <= 1024
        and d["positions_differ_from_1d"] for d in details.values()), details)
    return rows


def teacher_section(args, model, processor, contract: dict, rows: list[dict], media_dir: Path) -> list[dict]:
    records = capture(model, processor, rows[:1], media_dir, contract)
    record = records[0]
    prepared = prepare_sample_record(processor, rows[0], media_dir, seq_length=4096)
    inputs = {key: value.to("cuda:1") if torch.is_tensor(value) else value for key, value in prepared["inputs"].items()}
    full_ids = prepared["full_ids"].unsqueeze(0).to("cuda:1")
    with torch.inference_mode():
        mm = inputs["mm_token_type_ids"]
        mm_full = torch.cat([mm, torch.zeros(1, full_ids.shape[1] - mm.shape[1], dtype=mm.dtype, device=mm.device)], 1)
        out = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mm_full, output_hidden_states=True,
                    use_cache=False)
    hs = out.hidden_states
    width = int(contract["text"]["hidden_size"])
    mapping = {
        str(layer): max_abs(record["hidden"][:, i * width:(i + 1) * width], hs[layer + 1][0].cpu())
        for i, layer in enumerate(LAYER_IDS)
    }
    # Same-shape recompute: the forward applies lm_head to the whole h[-1], so
    # the identical GEMM must reproduce the logits; a different shape differs
    # only by bf16 rounding (recorded) with the same greedy token.
    head = model.lm_head(hs[-1])
    final_vs_logits = max_abs(head, out.logits)
    top1_agree = float((head.argmax(-1) == out.logits.argmax(-1)).float().mean())
    partial_vs_logits = max_abs(model.lm_head(hs[-1][0, -4:]), out.logits[0, -4:])
    final_vs_record = max_abs(record["final_hidden"], hs[-1][0].cpu())
    report["teacher"] = {
        "num_hidden_states": len(hs), "resolved_hidden_state_indices": [layer + 1 for layer in LAYER_IDS],
        "record_hidden": [list(record["hidden"].shape), str(record["hidden"].dtype), str(record["hidden"].device)],
        "record_final_hidden": [list(record["final_hidden"].shape), str(record["final_hidden"].dtype)],
        "layer_mapping_max_abs": mapping, "lm_head_final_vs_logits_max_abs": final_vs_logits,
        "lm_head_final_vs_logits_top1_agreement": top1_agree,
        "lm_head_last4_positions_vs_logits_max_abs": partial_vs_logits,
        "final_hidden_vs_record_max_abs": final_vs_record, "requires_grad": bool(record["hidden"].requires_grad),
    }
    (Path(args.out) / "hidden_layer_mapping.json").write_text(json.dumps({
        "semantic_layer_ids": LAYER_IDS, "resolved_hidden_state_indices": [layer + 1 for layer in LAYER_IDS],
        "store_final_hidden": True, "num_hidden_states": len(hs),
    }, indent=2))
    gate("hidden_layer_mapping", len(hs) == 37 and max(mapping.values()) == 0.0, mapping)
    gate("final_hidden_semantics", final_vs_logits == 0.0 and top1_agree == 1.0 and final_vs_record == 0.0,
         {"lm_head(h[-1]) vs logits": final_vs_logits, "top1_agreement": top1_agree,
          "last4_recompute_max_abs (bf16 rounding)": partial_vs_logits, "record vs h[-1]": final_vs_record})
    gate("teacher_no_grad", not record["hidden"].requires_grad and record["hidden"].dtype == torch.bfloat16)
    return records


def visual_section(model, processor, contract: dict, rows: list[dict], media_dir: Path) -> None:
    """Different image, same text: the text-position features must change."""
    from PIL import Image

    pair = []
    for name, source in (("swap_a", rows[0]), ("swap_b", rows[2])):
        image = Image.open(media_dir / source["images"][0]).convert("RGB").resize((448, 448))
        pair.append(make_row(processor, name, image, "Describe the image briefly.", "It shows several objects.",
                             media_dir))
    same_ids = pair[0]["regen"]["prompt_ids"] == pair[1]["regen"]["prompt_ids"]
    a, b = capture(model, processor, pair, media_dir, contract)
    width = int(contract["text"]["hidden_size"])
    tail = slice(-8, None)
    diffs = {str(layer): float((a["hidden"][tail, i * width:(i + 1) * width].float()
                                 - b["hidden"][tail, i * width:(i + 1) * width].float()).abs().mean())
             for i, layer in enumerate(LAYER_IDS)}
    report["different_image_same_text"] = {"same_input_ids": same_ids, "text_tail_mean_abs_diff": diffs}
    gate("visual_path_changes_features", same_ids and min(diffs.values()) > 0.0, diffs)


def _feature_drift(left: torch.Tensor, right: torch.Tensor, width: int) -> dict:
    """Per captured layer: relative L2 drift, cosine, max abs diff and the layer's own max |value|."""
    out = {}
    for i, layer in enumerate(LAYER_IDS):
        a = left[:, i * width:(i + 1) * width].float()
        b = right[:, i * width:(i + 1) * width].float()
        out[str(layer)] = {
            "rel_l2": float((a - b).norm() / a.norm()),
            "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
            "max_abs": float((a - b).abs().max()), "max_value": float(a.abs().max()),
        }
    return out


def batch_parity_section(model, processor, contract: dict, rows: list[dict], media_dir: Path) -> dict:
    """online_batch_size 1 vs 2 (and 2 vs 4) on fixed inputs (section 16).

    Padding changes the bf16 kernels, so features can only match up to rounding
    (relative drift is gated, ids / masks / positions must be identical); the
    loss, valid anchors and greedy-token comparisons follow in loss_parity_section.
    """
    width = int(contract["text"]["hidden_size"])
    single = [capture(model, processor, [row], media_dir, contract)[0] for row in rows]
    batched_by_size = {1: single}
    out = {}
    for size in (2, 4):
        batched = []
        for start in range(0, len(rows), size):
            batched += capture(model, processor, rows[start:start + size], media_dir, contract)
        batched_by_size[size] = batched
        out[f"batch{size}_vs_batch1"] = {
            row["id"]: {
                "hidden": _feature_drift(s["hidden"], b["hidden"], width),
                "final_hidden_rel_l2": float((s["final_hidden"].float() - b["final_hidden"].float()).norm()
                                             / s["final_hidden"].float().norm()),
                "ids_equal": bool(torch.equal(s["input_ids"], b["input_ids"])),
                "loss_mask_equal": bool(torch.equal(s["loss_mask"], b["loss_mask"])),
                "position3_equal": bool(torch.equal(s["position3"], b["position3"])),
            }
            for row, s, b in zip(rows, single, batched)
        }
    report["online_batch_parity"] = out
    exact = all(v["ids_equal"] and v["loss_mask_equal"] and v["position3_equal"] for d in out.values() for v in d.values())
    worst = max(layer["rel_l2"] for d in out.values() for v in d.values() for layer in v["hidden"].values())
    worst_final = max(v["final_hidden_rel_l2"] for d in out.values() for v in d.values())
    gate("online_batch_feature_parity", exact and worst < 2e-2 and worst_final < 2e-2,
         {"ids_masks_positions_identical": exact, "worst_rel_l2_hidden": worst, "worst_rel_l2_final": worst_final})
    return batched_by_size


def loss_parity_section(contract: dict, model, args, batched_by_size: dict) -> None:
    """Same drafter, same anchors: loss / valid anchors / greedy tokens per online batch size."""
    image_token = int(contract["tokens"]["image_token_id"])
    module, _ = drafter_module("dflash", contract, model, args)
    module.eval()
    results = {}
    for size, records in batched_by_size.items():
        batch = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                 for key, value in collate_features(records, image_token).items() if key != "lengths"}
        torch.manual_seed(4321)
        with torch.no_grad():
            metrics = forward(module, batch)
        greedy = []
        for record in records:
            answer = record["loss_mask"].bool()
            logits = model.lm_head(record["final_hidden"][answer].to("cuda:1"))
            greedy.append(logits.argmax(-1).cpu())
        results[size] = {"loss": float(metrics.loss), "valid_blocks": int(metrics.valid_blocks),
                         "valid_tokens": int(metrics.valid_tokens), "greedy": greedy}
    base = results[1]
    detail = {}
    for size in (2, 4):
        other = results[size]
        agree = torch.cat([(a == b).float() for a, b in zip(base["greedy"], other["greedy"])]).mean()
        detail[f"batch{size}_vs_batch1"] = {
            "loss": [base["loss"], other["loss"]], "loss_rel_diff": abs(other["loss"] - base["loss"]) / abs(base["loss"]),
            "valid_blocks": [base["valid_blocks"], other["valid_blocks"]],
            "valid_tokens": [base["valid_tokens"], other["valid_tokens"]], "greedy_agreement": float(agree),
        }
    report["online_batch_loss_parity"] = detail
    gate("online_batch_loss_parity", all(
        d["valid_blocks"][0] == d["valid_blocks"][1] and d["valid_tokens"][0] == d["valid_tokens"][1]
        and d["loss_rel_diff"] < 1e-2 and d["greedy_agreement"] >= 0.99 for d in detail.values()), detail)
    del module
    torch.cuda.empty_cache()


def transfer_section(records: list[dict]) -> dict:
    """Split-device boundary: target GPU -> host -> drafter GPU, as the online loader does."""
    record = records[0]
    on_target = {key: value.to("cuda:1") for key, value in record.items() if torch.is_tensor(value)}
    nbytes = sum(value.numel() * value.element_size() for value in on_target.values())
    sync()
    started = time.perf_counter()
    host = {key: value.cpu() for key, value in on_target.items()}
    sync()
    d2h = time.perf_counter() - started
    started = time.perf_counter()
    _ = {key: value.to("cuda:0") for key, value in host.items()}
    sync()
    h2d = time.perf_counter() - started
    started = time.perf_counter()
    _ = {key: value.to("cuda:0") for key, value in on_target.items()}
    sync()
    direct = time.perf_counter() - started
    info = {"bytes_per_record": int(nbytes), "dtype": str(record["hidden"].dtype), "source": "cuda:1",
            "destination": "cuda:0", "path": "host staging (record .cpu() then batch .to(train device))",
            "d2h_seconds": d2h, "h2d_seconds": h2d, "direct_device_copy_seconds": direct,
            "p2p_available": report["env"]["p2p_cuda1_to_cuda0"]}
    report["feature_transfer"] = info
    gate("split_device_transfer", nbytes > 0, info)
    return info


def drafter_module(method: str, contract: dict, model, args):
    cfg = matrix.build(method)
    cfg["target"]["model"] = args.model
    algorithm = get_algorithm(cfg["algorithm"])
    algorithm.validate_config(cfg)
    embed, lm_head, _tied = extract_target_head(model, contract)
    draft_config = algorithm.build_draft_config(contract, cfg)
    module = algorithm.build_trainer_module(draft_config, contract, cfg, embed.to(torch.bfloat16).cpu(),
                                            lm_head_weight=None if lm_head is None else lm_head.to(torch.bfloat16).cpu())
    module.configure_feature_layout(LAYER_IDS, int(contract["text"]["hidden_size"]))
    module.to("cuda:0")
    module.lm_head.to("cuda:0")
    module.embed_tokens.to("cuda:0")
    module.draft_model.to(torch.bfloat16)
    module.train()
    return module, draft_config


def forward(module, batch: dict, step: int = 0, total: int = matrix.TOTAL_UPDATES):
    hook = getattr(module, "set_training_progress", None)
    if hook is not None:
        hook(step, total)
    return module(input_ids=batch["input_ids"], hidden_states=batch["hidden"], loss_mask=batch["loss_mask"],
                  image_mask=batch["image_mask"], multimodal_position_ids=batch["position3"],
                  teacher_hidden=batch.get("final_hidden"))


def grads(module) -> dict[str, torch.Tensor]:
    return {name: p.grad.detach().float().clone() for name, p in module.draft_model.named_parameters()
            if p.grad is not None}


def flex_parity_section(contract: dict, model, args, batch: dict) -> None:
    """Reference dense mask (sdpa) vs FlexAttention block mask on the same anchors (section 21)."""
    from torch.nn.attention.flex_attention import create_mask

    from vlm_dflash.dflash.mask import build_dflash_mask_mod, create_dflash_sdpa_mask

    module, _ = drafter_module("dflash", contract, model, args)
    results = {}
    for backend in ("flex_attention", "sdpa"):
        module.attention_backend = backend
        module.draft_model.config._attn_implementation = backend
        module.zero_grad(set_to_none=True)
        torch.manual_seed(1234)
        metrics = forward(module, batch)
        metrics.loss.backward()
        results[backend] = {"loss": float(metrics.loss), "accept_len": float(metrics.accept_len), "grads": grads(module)}
    flex, ref = results["flex_attention"], results["sdpa"]
    names = sorted(set(flex["grads"]) & set(ref["grads"]))
    rel = max(float((flex["grads"][n] - ref["grads"][n]).norm() / (ref["grads"][n].norm() + 1e-12)) for n in names)
    finite = all(torch.isfinite(flex["grads"][n]).all() for n in names)
    # Allowed positions: the flex mask_mod materialized vs the dense additive mask.
    anchors = torch.tensor([[3, 9, 20]], device="cuda:0")
    keep = torch.tensor([[True, True, False]], device="cuda:0")
    dense = create_dflash_sdpa_mask(anchors, keep, 32, 8, "cuda:0", torch.float32) == 0
    mask_mod = build_dflash_mask_mod(anchors, keep, 32, 8, 3)
    flex_mask = create_mask(mask_mod, 1, 1, 24, 32 + 24, device="cuda:0")
    kept_rows = keep.repeat_interleave(8, dim=1)[0]
    positions_equal = bool(torch.equal(dense[0, 0][kept_rows], flex_mask[0, 0][kept_rows]))
    detail = {"loss_flex": flex["loss"], "loss_sdpa": ref["loss"], "accept_len": [flex["accept_len"], ref["accept_len"]],
              "max_rel_grad_diff": rel, "grads_finite": bool(finite), "allowed_positions_equal": positions_equal}
    report["flex_vs_reference_mask"] = detail
    gate("flex_vs_reference_mask", positions_equal and finite and abs(flex["loss"] - ref["loss"]) < 2e-2 * abs(ref["loss"])
         and rel < 0.1, detail)
    del module
    torch.cuda.empty_cache()


def methods_section(contract: dict, model, args, batch: dict) -> None:
    """One real-image forward/backward per method (section 20) + method-semantics probes (section 22)."""
    out = {}
    for method in matrix.QUEUE:
        torch.cuda.reset_peak_memory_stats(0)
        module, _ = drafter_module(method, contract, model, args)
        # Gradient coverage at steps where every head is live: Domino's correction
        # head starts after its curriculum (lambda_base < 1), DFly spans both
        # stages. Zero-initialized residual branches (DFly's hidden correction
        # down_proj) pass no gradient to their inputs on the first update, so the
        # audit runs on the second backward, after one optimizer step.
        probe_steps = {"domino": [matrix.TOTAL_UPDATES // 10], "dfly": [0, matrix.DFLY_BOUNDARY]}.get(method, [0])
        optimizer = torch.optim.AdamW([p for p in module.draft_model.parameters() if p.requires_grad], lr=1e-4)
        groups: dict[str, float] = {}
        for probe in probe_steps:
            for update in range(2):
                module.zero_grad(set_to_none=True)
                torch.manual_seed(0)
                metrics = forward(module, batch, step=probe + update)
                metrics.loss.backward()
                if update == 0:
                    optimizer.step()
            got = grads(module)
            for name, grad in got.items():
                key = ".".join(name.split(".")[:2])
                groups[key] = groups.get(key, 0.0) + float(grad.norm())
        del optimizer
        trainable = {".".join(n.split(".")[:2]) for n, p in module.draft_model.named_parameters() if p.requires_grad}
        silent = sorted(key for key in trainable if groups.get(key, 0.0) == 0.0)
        extra = {key: float(value) for key, value in (getattr(metrics, "extra_metrics", None) or {}).items()
                 if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1)}
        entry = {"probe_steps": probe_steps, "loss": float(metrics.loss), "finite_loss": bool(torch.isfinite(metrics.loss)),
                 "finite_grads": all(bool(torch.isfinite(g).all()) for g in got.values()),
                 "params_m": round(sum(p.numel() for p in module.draft_model.parameters()) / 1e6, 1),
                 "modules_without_grad": silent, "extra_metrics": extra,
                 "peak_gb_cuda0": round(torch.cuda.max_memory_allocated(0) / 2**30, 2)}
        if method == "dfly":
            entry["stage_index"] = [module.train_recipe.stage_index(step, matrix.TOTAL_UPDATES)
                                    for step in (0, matrix.DFLY_BOUNDARY - 1, matrix.DFLY_BOUNDARY)]
        if method == "domino":
            lambdas = []
            for step in (0, matrix.TOTAL_UPDATES // 2, matrix.TOTAL_UPDATES - 1):
                module.set_training_progress(step, matrix.TOTAL_UPDATES)
                lambdas.append(float(module.current_lambda_base))
            entry["lambda_base_at_0_half_last"] = lambdas
        out[method] = entry
        gate(f"method_{method}_forward_backward", entry["finite_loss"] and entry["finite_grads"] and not silent, entry)
        del module
        torch.cuda.empty_cache()
    report["methods"] = out


def checkpoint_section(contract: dict, model, args, batch: dict, scratch: Path) -> None:
    module, _ = drafter_module("dflash", contract, model, args)
    optimizer = torch.optim.AdamW([p for p in module.draft_model.parameters() if p.requires_grad], lr=1e-4)
    torch.manual_seed(0)
    forward(module, batch).loss.backward()
    optimizer.step()
    path = scratch / "preflight_ckpt.pt"
    torch.save({"model": module.draft_model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    state = torch.load(path, map_location="cpu")
    fresh, _ = drafter_module("dflash", contract, model, args)
    fresh.draft_model.load_state_dict(state["model"])
    same = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(module.draft_model.state_dict().values(),
                                                              fresh.draft_model.state_dict().values()))
    gate("checkpoint_save_load", same, {"bytes": path.stat().st_size})
    path.unlink()
    del module, fresh
    torch.cuda.empty_cache()


def timing_section(model, processor, contract: dict, rows: list[dict], media_dir: Path) -> None:
    """Teacher forward wall at batch 1 / 2 for the real rows (seconds, synchronized)."""
    out = {}
    for size in (1, 2):
        capture(model, processor, rows[:size], media_dir, contract)  # warm-up
        torch.cuda.reset_peak_memory_stats(1)
        sync()
        started = time.perf_counter()
        capture(model, processor, rows[:size], media_dir, contract)
        sync()
        out[f"batch{size}"] = {"seconds": round(time.perf_counter() - started, 3),
                               "peak_gb_cuda1": round(torch.cuda.max_memory_allocated(1) / 2**30, 2)}
    report["teacher_timing"] = out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--onevision-root", required=True)
    parser.add_argument("--scratch", required=True, help="node-local scratch dir (images, temp checkpoint)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    media_dir = Path(args.scratch) / "preflight_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    try:
        env_section()
        contract = contract_section(args)
        model, processor = target_section(args, contract)
        rows = real_image_section(args, model, processor, contract, media_dir)
        records = teacher_section(args, model, processor, contract, rows, media_dir)
        visual_section(model, processor, contract, rows, media_dir)
        batched_by_size = batch_parity_section(model, processor, contract, rows, media_dir)
        loss_parity_section(contract, model, args, batched_by_size)
        transfer_section(records)
        batch = {key: value.to("cuda:0") if torch.is_tensor(value) else value
                 for key, value in collate_features(capture(model, processor, rows[:2], media_dir, contract),
                                                    int(contract["tokens"]["image_token_id"])).items()
                 if key != "lengths"}
        flex_parity_section(contract, model, args, batch)
        methods_section(contract, model, args, batch)
        checkpoint_section(contract, model, args, batch, Path(args.scratch))
        timing_section(model, processor, contract, rows, media_dir)
    except Exception as error:  # every failure lands in the report
        import traceback

        gate("preflight_completed", False, f"{error.__class__.__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    failed = [name for name, entry in report["gates"].items() if entry["mandatory"] and not entry["pass"]]
    report["status"] = "PASS" if not failed else "FAIL"
    (out / "preflight_gpu.json").write_text(json.dumps(copy.deepcopy(report), indent=2, default=str))
    print("PREFLIGHT_GPU_PASS" if not failed else f"PREFLIGHT_GPU_FAIL {' '.join(failed)}", flush=True)
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
