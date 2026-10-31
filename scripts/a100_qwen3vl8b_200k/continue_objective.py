#!/usr/bin/env python3
"""Validation objective of drafter checkpoints on a frozen held-out split (continuation report).

    CUDA_VISIBLE_DEVICES=6,7 python scripts/a100_qwen3vl8b_200k/continue_objective.py \\
        --config TASK/runs/dflash/config.json --contract TASK/runs/dflash/contract.json \\
        --rows-dir DEVTEST --split DEVTEST/data/split_dev.jsonl --position 75000 \\
        --checkpoint TASK/runs/dflash/train/ckpt_final.pt --label orig_final \\
        --out CONT/eval/dflash/objective.orig_final.json

The objective is the drafter's own training objective (the training forward and
loss), computed under ``no_grad`` in training mode, with the anchors of batch
``i`` drawn from ``--seed + i``: every checkpoint is scored on the same rows,
features and anchors. ``--rows-dir`` is a staged run layout
(``regen/target_data.jsonl`` + ``data/meta.json``) whose rows carry the target
answers; ``--split`` lists the row ids to score (rows without a target answer
are counted and skipped). Schedule-driven settings (recipe stage, Domino's
``lambda_base``) are those at update ``--position``. Several
``--checkpoint``/``--label`` pairs share one target load and one extraction;
a train state (``ckpt_last.pt``) is scored through its ``model`` weights.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch

from vlm_dflash.algorithms import get_algorithm
from vlm_dflash.dflash.features import capture_layer_layout, collate_features
from vlm_dflash.dflash.trainer_core import NoValidAnchorsError
from vlm_dflash.target.online import OnlineFeatureDataset, OnlineFeatureSource
from vlm_dflash.utils import log, read_json, read_jsonl, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Accumulator:
    """Token- and block-weighted sums over batches (per-batch means would bias short batches)."""

    def __init__(self) -> None:
        self.batches = 0
        self.skipped = 0
        self.samples = 0
        self.loss_sum = 0.0
        self.weighted_loss = 0.0
        self.weight = 0.0
        self.valid = 0
        self.correct = 0
        self.accept_sum = 0.0
        self.blocks = 0
        self.extra: dict[str, float] = {}

    def add(self, metrics, samples: int) -> None:
        loss = float(metrics.loss.detach())
        weight = float(metrics.loss_weight)
        if not torch.isfinite(metrics.loss.detach()):
            raise RuntimeError(f"non-finite validation objective {loss} in batch {self.batches}")
        self.batches += 1
        self.samples += samples
        self.loss_sum += loss
        self.weighted_loss += loss * weight
        self.weight += weight
        self.valid += int(metrics.valid_tokens)
        self.correct += int(metrics.correct_tokens)
        self.accept_sum += float(metrics.accept_len_sum)
        self.blocks += int(metrics.valid_blocks)
        for key, value in (metrics.extra_metrics or {}).items():
            self.extra[key] = self.extra.get(key, 0.0) + float(value)

    def result(self) -> dict:
        batches = max(self.batches, 1)
        return {
            "objective_mean": self.loss_sum / batches,
            "objective_weighted": self.weighted_loss / self.weight if self.weight else None,
            "components_mean": {key: value / batches for key, value in self.extra.items()},
            "accuracy": self.correct / self.valid if self.valid else 0.0,
            "accept_len": self.accept_sum / self.blocks if self.blocks else 0.0,
            "supervised_positions": self.valid,
            "batches": self.batches,
            "skipped_batches": self.skipped,
            "samples": self.samples,
        }


def score(module, records: list[dict], *, device, image_token_id: int, batch_size: int, seed: int, spatial: bool) -> dict:
    acc = Accumulator()
    with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        for batch_index, start in enumerate(range(0, len(records), batch_size)):
            chunk = records[start : start + batch_size]
            batch = collate_features(chunk, image_token_id)
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
                if key != "lengths"
            }
            torch.manual_seed(seed + batch_index)
            try:
                metrics = module(
                    input_ids=batch["input_ids"],
                    hidden_states=batch["hidden"],
                    loss_mask=batch["loss_mask"],
                    image_mask=batch["image_mask"],
                    multimodal_position_ids=batch["position3"] if spatial else None,
                    teacher_hidden=batch.get("final_hidden"),
                )
            except NoValidAnchorsError:
                acc.skipped += 1
                continue
            acc.add(metrics, len(chunk))
    return acc.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the run's effective config.json")
    parser.add_argument("--contract", type=Path, required=True, help="the run's contract.json")
    parser.add_argument("--rows-dir", type=Path, required=True, help="staged layout with regen/target_data.jsonl")
    parser.add_argument("--split", type=Path, required=True, help="jsonl of the row ids to score")
    parser.add_argument("--position", type=int, required=True, help="update whose schedule settings apply")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--out", type=Path, action="append", required=True, help="one JSON per checkpoint")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--single-gpu", action="store_true", help="target and drafter both on cuda:0")
    args = parser.parse_args()
    if not len(args.checkpoint) == len(args.label) == len(args.out):
        raise SystemExit("--checkpoint, --label and --out must be given the same number of times.")

    cfg = read_json(args.config)
    contract = read_json(args.contract)
    if args.single_gpu:
        cfg.setdefault("target", {})["device"] = "cuda:0"
        cfg.setdefault("train", {})["device"] = "cuda:0"
    algorithm = get_algorithm(cfg.get("algorithm"))
    algorithm.validate_config(cfg)
    train_cfg = cfg.get("train") or {}
    device = torch.device(train_cfg.get("device", "cuda:0"))
    dtype = torch.bfloat16 if str(train_cfg.get("draft_dtype", "bfloat16")) == "bfloat16" else torch.float32

    started = time.time()
    source = OnlineFeatureSource(cfg, contract, args.rows_dir)
    try:
        split_ids = [str(record["id"]) for record in read_jsonl(args.split)]
        index_of = {str(row["id"]): index for index, row in enumerate(source.rows)}
        indices = [index_of[row_id] for row_id in split_ids if row_id in index_of]
        source.set_phase("validation")
        records = [OnlineFeatureDataset._normalize(source.extract(index)) for index in indices]
        source.set_phase("train")
        extract_seconds = time.time() - started
        log(f"objective: {len(records)}/{len(split_ids)} split rows carry a target answer ({extract_seconds:.0f}s)")

        draft_config = algorithm.build_draft_config(contract, cfg)
        module = algorithm.build_trainer_module(
            draft_config, contract, cfg, source.embed_weight, lm_head_weight=source.lm_head_weight
        )
        capture, hidden_size = capture_layer_layout(
            source.manifest(), contract_hidden_size=int(contract["text"]["hidden_size"])
        )
        module.configure_feature_layout(capture, hidden_size)
        module.to(device)
        module.lm_head.to(device)
        module.embed_tokens.to(device)
        module.draft_model.to(dtype)
        module.train()
        progress_hook = getattr(module, "set_training_progress", None)
        if progress_hook is not None:
            progress_hook(args.position, args.position)
        spatial = bool(
            getattr(module, "uses_multimodal_position_ids", getattr(module.draft_model, "spatial_rope_enabled", False))
        )
        common = {
            "algorithm": algorithm.name,
            "split": str(args.split),
            "split_sha256": sha256_file(args.split),
            "rows_sha256": sha256_file(args.rows_dir / "regen" / "target_data.jsonl"),
            "split_rows": len(split_ids),
            "rows_with_target_answer": len(records),
            "position": args.position,
            "recipe_stage": getattr(module, "recipe_stage_index", None),
            "lambda_base": getattr(module, "current_lambda_base", None),
            "batch_size": args.batch_size,
            "anchor_seed": args.seed,
            "target_model_fingerprint": contract["target"].get("model_fingerprint"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        for checkpoint, label, out in zip(args.checkpoint, args.label, args.out):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("algorithm") != algorithm.name:
                raise SystemExit(f"{checkpoint} is algorithm {payload.get('algorithm')!r}, not {algorithm.name!r}.")
            module.draft_model.load_state_dict(payload["model"])
            module.draft_model.to(dtype)
            del payload
            began = time.time()
            result = score(
                module,
                records,
                device=device,
                image_token_id=int(contract["tokens"]["image_token_id"]),
                batch_size=args.batch_size,
                seed=args.seed,
                spatial=spatial,
            )
            record = {
                **common,
                **result,
                "label": label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "score_seconds": time.time() - began,
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            write_json(out, record)
            log(
                f"objective {algorithm.name} {label}: objective={record['objective_mean']:.5f} "
                f"weighted={record['objective_weighted']} acc={record['accuracy']:.4f} "
                f"al={record['accept_len']:.3f} -> {out}"
            )
    finally:
        source.close()


if __name__ == "__main__":
    main()
