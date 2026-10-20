#!/usr/bin/env python3
"""The a100_qwen3vl8b_200k_s42 matrix: one frozen base config plus seven method overlays.

Single source of truth for the seven formal configs in
``configs/paper_v1/a100_qwen3vl8b_200k/``:

    python scripts/a100_qwen3vl8b_200k/matrix.py write    # (re)generate the YAML files
    python scripts/a100_qwen3vl8b_200k/matrix.py check    # fail if a file drifted or a key is unknown

Every emitted key is checked against ``KNOWN_KEYS`` (keys the training,
regen and eval code actually reads), so an unknown or misspelled field fails
here instead of being silently ignored by the trainer. Paths stay environment
variables (``VLM_DFLASH_TARGET_QWEN3VL``, ``VELA_POOL``); nothing machine
specific is written into the repository.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs" / "paper_v1" / "a100_qwen3vl8b_200k"
EXPERIMENT = "a100_qwen3vl8b_200k_s42"

#: Formal budget (contract section 1).
TRAIN_ROWS = 200_000
QUOTAS = {"general": 70_000, "ocr": 60_000, "chart": 30_000, "reasoning": 40_000}
MICRO_BATCH, GRAD_ACCUM, EPOCHS = 2, 4, 3
UPDATES_PER_EPOCH = 25_000
TOTAL_UPDATES = 75_000
WARMUP_UPDATES = 3_000
SNAPSHOT_EVERY = 12_500
DFLY_BOUNDARY = 18_750

#: Launch priority (contract sections 4 and 25).
QUEUE = ["dflash", "dfly", "dflash2", "dspark", "dflare", "domino", "jetspec"]
#: Contract row ids (section 2).
ROWS = {"dflash": "M1", "dspark": "M2", "domino": "M3", "jetspec": "M4", "dflash2": "M5", "dflare": "M6", "dfly": "M7"}

BASE: dict = {
    "target": {
        "model": "${VLM_DFLASH_TARGET_QWEN3VL}",
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        # Split-device worker: logical cuda:1 holds the frozen target, cuda:0 the drafter.
        "device": "cuda:1",
        "local_files_only": True,
        # 64..1024 visual tokens per image (patch 16, merge 2); the processor default
        # (16M pixels) cannot keep a whole visual span inside the 4096-token window.
        "image_min_pixels": 65536,
        "image_max_pixels": 1048576,
        # No-op for this template (it never reads the flag); kept so the frozen
        # prompt protocol states enable_thinking=false explicitly.
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "drafter": {
        "num_hidden_layers": 3,
        # Inherited from the target text config; stated so a mismatch fails loud.
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "block_size": 8,
        "num_anchors": 512,
        "attention_backend": "flex_attention",
        # First embedding row past the tokenizer (151669 tokens); never emitted.
        "mask_token_id": 151669,
        "spatial_rope_enabled": True,
        "spatial_rope_mode": "replace",
        "spatial_rope_interleaved": True,
        "spatial_rope_sections": [24, 20, 20],
    },
    "dataset": {
        "source": "jsonl",
        "path": "${VELA_POOL}/pool.jsonl",
        "split_field": "split",
        # Every train candidate of the pool; regen.select cuts them to the quotas.
        "slice": None,
        "seed": 42,
        "image_link_mode": "hardlink",
    },
    "regen": {
        "backend": "vllm",
        # Qwen3-VL-8B-Instruct generation_config.json: sampled, T=0.7, top_p 0.8,
        # top_k 20, repetition 1.0, no presence penalty field.
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "min_response_tokens": 8,
        # Longer prompts are dropped so prompt + answer fit window_tokens with a
        # natural EOS (length-capped answers are dropped, never relabelled).
        "max_prompt_tokens": 2048,
        "window_tokens": 4096,
        "seed": 42,
        "multi_turn_mode": "last_turn_only",
        "select": {"meta": "${VELA_POOL}/pool_meta.jsonl", "quotas": dict(QUOTAS)},
    },
    "capture": {
        "seq_length": 4096,
        "dtype": "bfloat16",
        "store_final_hidden": True,
        "visual_path_check": True,
        "visual_check_samples": 4,
    },
    "train": {
        "device": "cuda:0",
        "feature_mode": "online",
        # One target forward per row: batched capture makes BF16 vision features
        # depend on the batch's other images (up to 7% for a small image, fp32
        # agrees to 3e-5), while batch 1 matches the batch-1 inference prefill.
        "online_batch_size": 1,
        "online_feature_cache_gb": 0,
        "num_workers": 0,
        "micro_batch_size": MICRO_BATCH,
        "grad_accumulation_steps": GRAD_ACCUM,
        "num_epochs": EPOCHS,
        "lr": 1.0e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.04,
        # Floor 1e-5 = lr * min_lr_ratio.
        "min_lr_ratio": 0.1,
        "seed": 42,
        "log_every_steps": 10,
        "draft_dtype": "bfloat16",
        # Resumable state every 1,250 updates; eval snapshots at the formal
        # 12,500 multiples (plus the DFly stage boundary, written by the recipe).
        "save_every_steps": 1250,
        "milestone_snapshots": True,
        "snapshot_every_steps": SNAPSHOT_EVERY,
        # Per-step stage wall times, peaks and transfer bytes in train_log.jsonl.
        "stage_timing": True,
        "overfit_sanity": {"enabled": True, "num_samples": 4, "steps": 200, "ce_drop": 0.6, "slot0_acc": 0.9},
    },
    "eval": {
        "verification": "sequential",
        "sequential_target_verification": True,
        "verify_sdpa_backend": "parity",
        "max_new_tokens": 128,
    },
}

#: Method overlays (contract section 18), merged over BASE.
METHODS: dict[str, dict] = {
    "dflash": {
        "algorithm": {"name": "dflash"},
        "drafter": {"loss_decay_gamma": 7.0},
    },
    "dspark": {
        "algorithm": {
            "name": "dspark",
            "dspark": {
                "markov_rank": 256,
                "markov_head_type": "vanilla",
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
                "confidence_head_alpha": 1.0,
                "sample_from_anchor": True,
            },
            # confidence_coef exists only in the upstream DSpark objective.
            "train_recipe": {"name": "upstream_dspark", "confidence_coef": 1.0},
        },
        "drafter": {"loss_decay_gamma": 4.0},
    },
    "domino": {
        "algorithm": {
            "name": "domino",
            "domino": {
                "emb_dim": 256,
                "gru_hidden_dim": 1024,
                "pure_draft_prefix_len": 1,
                "shift_label": True,
                "enable_correction_head": True,
                # lambda_base 1 -> 0 linearly over the whole run (75,000 updates).
                "lambda_base_start": 1.0,
                "lambda_base_decay_ratio": 1.0,
            },
        },
        "drafter": {"loss_decay_gamma": 7.0},
    },
    "jetspec": {
        # Causal in-block attention and T^2 scaling are fixed behavior (no keys).
        "algorithm": {"name": "jetspec", "jetspec": {"kd_temperature": 1.0, "kd_chunk_size": 1024}},
        "drafter": {"loss_decay_gamma": None},
    },
    "dflash2": {
        # Unary forward KL is fixed behavior; the objective is the repo's
        # experimental reconstruction (formal row status PARTIAL).
        "algorithm": {
            "name": "dflash2",
            "dflash2": {
                "conv_kernel_size": 2,
                "conv_group_size": 16,
                "selector_rank": 256,
                "selector_top_k": 16,
                "selector_loss_alpha": 1.0,
                "kd_chunk_size": 1024,
            },
        },
        "drafter": {"loss_decay_gamma": 7.0},
    },
    "dflare": {
        "algorithm": {
            "name": "dflare",
            "dflare": {
                "context_layer_ids": [1, 17, 33],
                "separate_context_kv": True,
                "fusion_trainable": True,
                "fusion_init": "distance",
            },
        },
        "drafter": {"loss_decay_gamma": 7.0},
    },
    "dfly": {
        "algorithm": {
            "name": "dfly",
            "dfly": {
                "context_layer_ids": [1, 17, 33],
                "fusion_trainable": True,
                "fusion_init": "distance",
                "enable_hidden_correction": True,
                "hidden_correction_intermediate_size": None,
            },
            # Stage 1 LK/D-PACE until 25% of the run (update 18,750), stage 2 e2e-TV.
            "train_recipe": {"name": "upstream_two_stage", "cold_start_fraction": 0.25},
        },
        "drafter": {"loss_decay_gamma": 7.0},
    },
}

#: Keys the code reads, per block (see codebase_integration.md for where).
KNOWN_KEYS: dict[str, set[str]] = {
    "run": {"name", "dir"},
    "target": {
        "model", "dtype", "attn_implementation", "device", "device_map", "max_memory", "local_files_only",
        "image_min_pixels", "image_max_pixels", "chat_template_kwargs",
    },
    "drafter": {
        "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "block_size", "num_anchors", "attention_backend", "mask_token_id", "spatial_rope_enabled",
        "spatial_rope_mode", "spatial_rope_interleaved", "spatial_rope_sections", "capture_layer_ids",
        "loss_decay_gamma",
    },
    "dataset": {"source", "path", "split_field", "slice", "seed", "image_link_mode", "val_size", "test_size"},
    "regen": {
        "backend", "temperature", "top_p", "top_k", "presence_penalty", "repetition_penalty", "max_new_tokens",
        "min_response_tokens", "max_prompt_tokens", "window_tokens", "seed", "multi_turn_mode", "select",
        "finalize_workers",
    },
    "regen.select": {"meta", "quotas"},
    "capture": {"seq_length", "dtype", "store_final_hidden", "shard_size", "visual_path_check", "visual_check_samples"},
    "train": {
        "device", "feature_mode", "online_batch_size", "online_feature_cache_gb", "num_workers", "micro_batch_size",
        "grad_accumulation_steps", "num_epochs", "lr", "betas", "weight_decay", "max_grad_norm", "warmup_ratio",
        "min_lr_ratio", "seed", "log_every_steps", "draft_dtype", "save_every_steps", "milestone_snapshots",
        "snapshot_every_steps", "stage_timing", "overfit_sanity",
    },
    "train.overfit_sanity": {"enabled", "num_samples", "steps", "ce_drop", "slot0_acc"},
    "eval": {
        "verification", "sequential_target_verification", "verify_sdpa_backend", "max_new_tokens", "num_prompts",
        "strict_parallel_margin", "parallel_diagnostic_prompts", "checkpoints",
    },
    "algorithm": {"name", "train_recipe", *QUEUE},
}


def _merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build(method: str) -> dict:
    """The full config of one row (``run`` + ``algorithm`` first, then BASE order)."""
    merged = _merge(BASE, METHODS[method])
    ordered = {"run": {"name": f"{EXPERIMENT}_{method}"}, "algorithm": merged.pop("algorithm")}
    ordered.update(merged)
    return ordered


def unknown_keys(cfg: dict) -> list[str]:
    """Dotted names of keys outside ``KNOWN_KEYS`` and the algorithm factories' knobs."""
    from vlm_dflash.objectives import _RECIPE_ALLOWED_OPTIONS

    knobs = _algorithm_knobs()
    problems = []
    for block, value in cfg.items():
        if block not in KNOWN_KEYS:
            problems.append(block)
            continue
        if not isinstance(value, dict):
            continue
        for key, inner in value.items():
            if key not in KNOWN_KEYS[block]:
                problems.append(f"{block}.{key}")
            nested = f"{block}.{key}"
            if nested in KNOWN_KEYS and isinstance(inner, dict):
                problems.extend(f"{nested}.{name}" for name in inner if name not in KNOWN_KEYS[nested])
    algorithm = cfg.get("algorithm", {})
    for name, block in algorithm.items():
        if name in knobs and isinstance(block, dict):
            problems.extend(f"algorithm.{name}.{key}" for key in block if key not in knobs[name])
    recipe = algorithm.get("train_recipe")
    if isinstance(recipe, dict):
        allowed = _RECIPE_ALLOWED_OPTIONS.get(recipe.get("name"), ())
        problems.extend(f"algorithm.train_recipe.{key}" for key in recipe if key != "name" and key not in allowed)
    return problems


def _algorithm_knobs() -> dict[str, set[str]]:
    from vlm_dflash.dflare.factory import DEFAULTS as DFLARE
    from vlm_dflash.dflash2.factory import DEFAULTS as DFLASH2
    from vlm_dflash.dfly.factory import DEFAULTS as DFLY
    from vlm_dflash.domino.factory import DEFAULTS as DOMINO
    from vlm_dflash.dspark.factory import DEFAULTS as DSPARK
    from vlm_dflash.jetspec.factory import DEFAULTS as JETSPEC

    return {"dspark": set(DSPARK), "domino": set(DOMINO), "jetspec": set(JETSPEC), "dflash2": set(DFLASH2),
            "dflare": set(DFLARE), "dfly": set(DFLY)}


def budget_errors(cfg: dict) -> list[str]:
    """The formal budget the trainer will derive from this config (contract section 1)."""
    train = cfg["train"]
    micro_batches = math.ceil(TRAIN_ROWS / int(train["micro_batch_size"]))
    per_epoch = math.ceil(micro_batches / int(train["grad_accumulation_steps"]))
    total = per_epoch * int(train["num_epochs"])
    warmup = max(1, int(float(train["warmup_ratio"]) * total))  # the trainer's own rounding
    errors = []
    if int(train["micro_batch_size"]) * int(train["grad_accumulation_steps"]) != 8:
        errors.append("global batch is not 8")
    if per_epoch != UPDATES_PER_EPOCH or total != TOTAL_UPDATES:
        errors.append(f"updates {per_epoch}/epoch, {total} total (want {UPDATES_PER_EPOCH}, {TOTAL_UPDATES})")
    if warmup != WARMUP_UPDATES:
        errors.append(f"warmup {warmup} updates (want {WARMUP_UPDATES})")
    if abs(float(train["lr"]) * float(train["min_lr_ratio"]) - 1e-5) > 1e-12:
        errors.append("min lr is not 1e-5")
    if sum(cfg["regen"]["select"]["quotas"].values()) != TRAIN_ROWS:
        errors.append("quotas do not sum to the train budget")
    return errors


HEADER = (
    "# GENERATED by scripts/a100_qwen3vl8b_200k/matrix.py; do not edit by hand.\n"
    "# Row {row} ({method}) of the a100_qwen3vl8b_200k_s42 matrix: Qwen3-VL-8B-Instruct target, 200K\n"
    "# OneVision requests, online frozen-teacher features, 3 epochs x 25,000 updates, seed 42.\n"
)


def render(method: str) -> str:
    body = yaml.safe_dump(build(method), sort_keys=False, default_flow_style=None, width=118)
    return HEADER.format(row=ROWS[method], method=method) + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("write", "check"))
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    failures = []
    for method in QUEUE:
        cfg = build(method)
        failures += [f"{method}: unknown key {key}" for key in unknown_keys(cfg)]
        failures += [f"{method}: {error}" for error in budget_errors(cfg)]
        path = CONFIG_DIR / f"{method}_s42.yaml"
        if args.mode == "write":
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(render(method))
        elif not path.exists() or path.read_text() != render(method):
            failures.append(f"{method}: {path.relative_to(REPO)} differs from the generator")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"MATRIX_OK {args.mode} {len(QUEUE)} configs in {CONFIG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
