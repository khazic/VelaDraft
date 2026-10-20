#!/usr/bin/env python3
"""Diagnose batched vs per-row online capture drift on the real Qwen3-VL target (one GPU pair).

    CUDA_VISIBLE_DEVICES=a,b python scripts/a100_qwen3vl8b_200k/diag_batch_capture.py \\
        --model "$VLM_DFLASH_TARGET_QWEN3VL" --onevision-root "$ONEVISION_ROOT" --scratch DIR --out OUT.json

Separates the candidate causes of a batch-size dependence of teacher features:
the same row twice (batch size alone, no padding), a short row padded next to a
long one (padding), the model's own MRoPE positions and the vision features
for the batch versus each row alone, and the share of answer positions whose
greedy token changes. Drift is reported per captured layer and per region
(image tokens, text before the image span, text after it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import preflight_gpu as pf  # noqa: E402
from vlm_dflash.target.sample_capture import extend_mm_token_type_ids, prepare_sample_record  # noqa: E402


def regions(record: dict, image_token: int) -> dict[str, torch.Tensor]:
    ids = record["input_ids"].long()
    image = ids == image_token
    positions = torch.arange(ids.numel())
    first, last = int(positions[image].min()), int(positions[image].max())
    return {"all": torch.ones_like(image), "image": image, "text_before": positions < first,
            "text_after": positions > last}


def drift(ref: dict, other: dict, width: int, image_token: int) -> dict:
    out = {}
    masks = regions(ref, image_token)
    for i, layer in enumerate(pf.LAYER_IDS):
        a = ref["hidden"][:, i * width:(i + 1) * width].float()
        b = other["hidden"][:, i * width:(i + 1) * width].float()
        out[f"layer{layer}"] = {name: round(float((a[m] - b[m]).norm() / a[m].norm()), 5) for name, m in masks.items()
                                if m.any()}
    a, b = ref["final_hidden"].float(), other["final_hidden"].float()
    out["final"] = {name: round(float((a[m] - b[m]).norm() / a[m].norm()), 5) for name, m in masks.items() if m.any()}
    per_position = ((a - b).norm(dim=-1) / a.norm(dim=-1)).tolist()
    worst = sorted(range(len(per_position)), key=lambda index: -per_position[index])[:5]
    out["final_worst_positions"] = [(index, round(per_position[index], 4)) for index in worst]
    return out


def greedy_flips(model, ref: dict, other: dict) -> float:
    answer = ref["loss_mask"].bool()
    device = model.lm_head.weight.device
    with torch.no_grad():
        left = model.lm_head(ref["final_hidden"][answer].to(device)).argmax(-1)
        right = model.lm_head(other["final_hidden"][answer].to(device)).argmax(-1)
    return round(float((left != right).float().mean()), 5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--onevision-root", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    media = Path(args.scratch)
    media.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(Path(args.out).parent)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    import matrix
    from vlm_dflash.contract import contract_defaults_from_config, resolve_contract
    from vlm_dflash.target.load import load_target

    # Same placement as a formal worker: the target on logical cuda:1 of the pair.
    cfg = matrix.build("dflash")
    cfg["target"]["model"] = args.model
    contract = resolve_contract(args.model, contract_defaults_from_config(cfg))
    model, processor = load_target(cfg, contract)
    image_token = int(contract["tokens"]["image_token_id"])
    width = int(contract["text"]["hidden_size"])
    rows = pf.real_image_section(args, model, processor, contract, media)
    dev = next(model.parameters()).device

    def cap(batch):
        return pf.capture(model, processor, batch, media, contract)

    single = {row["id"]: cap([row])[0] for row in rows}
    report: dict = {"rows": {row["id"]: int(single[row["id"]]["input_ids"].numel()) for row in rows}}
    chart, doc = rows[0], rows[1]
    # 1. Batch size alone: the same row twice (no padding anywhere).
    twice = cap([doc, doc])
    report["same_row_twice"] = {"drift": drift(single["doc"], twice[0], width, image_token),
                                "greedy_flip_rate": greedy_flips(model, single["doc"], twice[0])}
    # 2. Padding: the short chart row padded next to the long doc row.
    padded = cap([chart, doc])
    report["short_padded_next_to_long"] = {
        "short_row": {"drift": drift(single["chart"], padded[0], width, image_token),
                      "greedy_flip_rate": greedy_flips(model, single["chart"], padded[0])},
        "long_row_unpadded": {"drift": drift(single["doc"], padded[1], width, image_token),
                              "greedy_flip_rate": greedy_flips(model, single["doc"], padded[1])},
    }
    # 3. The model's own MRoPE positions and the vision features, batch vs rows.
    prepared = [prepare_sample_record(processor, row, media, seq_length=4096) for row in (chart, doc)]
    lengths = [int(item["full_ids"].numel()) for item in prepared]
    width_max = max(lengths)
    ids = torch.zeros(2, width_max, dtype=torch.long)
    mask = torch.zeros(2, width_max, dtype=torch.long)
    mm = torch.zeros(2, width_max, dtype=torch.long)
    for index, item in enumerate(prepared):
        ids[index, :lengths[index]] = item["full_ids"]
        mask[index, :lengths[index]] = 1
        mm[index, :lengths[index]] = extend_mm_token_type_ids(item["inputs"]["mm_token_type_ids"], lengths[index])[0]
    grid = torch.cat([item["inputs"]["image_grid_thw"] for item in prepared])
    base = model.model
    with torch.no_grad():
        batch_pos, _ = base.get_rope_index(ids.to(dev), mm.to(dev), image_grid_thw=grid.to(dev),
                                           attention_mask=mask.to(dev))
        same = []
        for index, item in enumerate(prepared):
            row_pos, _ = base.get_rope_index(item["full_ids"][None].to(dev), mm[index:index + 1, :lengths[index]].to(dev),
                                             image_grid_thw=item["inputs"]["image_grid_thw"].to(dev))
            same.append(bool(torch.equal(row_pos[:, 0], batch_pos[:, index, :lengths[index]])))
        pixels = torch.cat([item["inputs"]["pixel_values"] for item in prepared]).to(dev)
        batch_vis = base.get_image_features(pixels, grid.to(dev))
        vis_rows = [base.get_image_features(item["inputs"]["pixel_values"].to(dev), item["inputs"]["image_grid_thw"].to(dev))
                    for item in prepared]
    embeds_batch = torch.cat(list(batch_vis.pooler_output)) if hasattr(batch_vis, "pooler_output") else None
    report["mrope_positions_equal_per_row"] = same
    if embeds_batch is not None:
        embeds_rows = torch.cat([torch.cat(list(v.pooler_output)) for v in vis_rows])
        report["vision_embeds_rel_l2"] = round(float((embeds_batch.float() - embeds_rows.float()).norm()
                                                     / embeds_rows.float().norm()), 6)
        deep_b = [d.float() for d in batch_vis.deepstack_features]
        deep_r = [torch.cat([v.deepstack_features[k] for v in vis_rows]).float() for k in range(len(deep_b))]
        report["deepstack_rel_l2"] = [round(float((b - r).norm() / r.norm()), 6) for b, r in zip(deep_b, deep_r)]
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("DIAG_DONE")


if __name__ == "__main__":
    main()
