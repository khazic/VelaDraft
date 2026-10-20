"""The a100_qwen3vl8b_200k_s42 configs: generated, known keys only, frozen budget, one shared corpus."""

from __future__ import annotations

import importlib.util
import re
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("qwen3vl8b_matrix", REPO / "scripts" / "a100_qwen3vl8b_200k" / "matrix.py")
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)

#: Keys the default ``single-gpu-48g`` preset adds that the formal configs do
#: not set, and why none of them reaches this run.
INERT_PRESET_KEYS = {
    "dataset.val_size": "the predefined split_field sizes dev/test",
    "dataset.min_answer_chars": "read only by the llava_instruct and hf sources (this run is jsonl)",
    "regen.batch_size": "HF generate backend only (this run uses regen.backend=vllm)",
    "capture.shard_size": "offline feature stores only (this run is online)",
    "eval.num_prompts": "the matrix eval scripts pass their own prompt sets",
    "eval.parallel_diagnostic_prompts": "diagnostic only",
    "eval.checkpoints": "the matrix eval scripts name checkpoints explicitly",
}


@pytest.mark.parametrize("method", matrix.QUEUE)
def test_config_file_matches_generator(method):
    path = matrix.CONFIG_DIR / f"{method}_s42.yaml"
    assert path.read_text() == matrix.render(method), "run scripts/a100_qwen3vl8b_200k/matrix.py write"


@pytest.mark.parametrize("method", matrix.QUEUE)
def test_config_has_only_known_keys_and_the_frozen_budget(method):
    cfg = matrix.build(method)
    assert matrix.unknown_keys(cfg) == []
    assert matrix.budget_errors(cfg) == []


def test_unknown_keys_are_reported():
    cfg = matrix.build("dspark")
    cfg["drafter"]["hidden_sizee"] = 4096
    cfg["train"]["warmup_updates"] = 3000
    cfg["algorithm"]["dspark"]["markov_rnk"] = 256
    cfg["algorithm"]["train_recipe"]["cold_start_fraction"] = 0.25
    assert sorted(matrix.unknown_keys(cfg)) == [
        "algorithm.dspark.markov_rnk",
        "algorithm.train_recipe.cold_start_fraction",
        "drafter.hidden_sizee",
        "train.warmup_updates",
    ]


def test_budget_errors_catch_changed_geometry():
    cfg = matrix.build("dflash")
    cfg["train"]["grad_accumulation_steps"] = 8
    cfg["train"]["warmup_ratio"] = 0.03
    errors = matrix.budget_errors(cfg)
    assert any("global batch" in error for error in errors)
    assert any("warmup" in error for error in errors)


def test_every_known_key_is_read_by_the_code():
    sources = "\n".join(path.read_text() for path in (REPO / "vlm_dflash").rglob("*.py"))
    sources += (REPO / "scripts" / "regen_vllm.py").read_text()
    for block, keys in matrix.KNOWN_KEYS.items():
        if block in ("algorithm", "run"):
            continue
        for key in keys:
            assert re.search(rf"[\"']{key}[\"']", sources), f"{block}.{key} is never read"


def test_shared_blocks_are_identical_across_rows():
    shared = ("target", "dataset", "regen", "capture", "train", "eval")
    reference = matrix.build("dflash")
    for method in matrix.QUEUE:
        cfg = matrix.build(method)
        for block in shared:
            assert cfg[block] == reference[block], (method, block)
        drafter = {key: value for key, value in cfg["drafter"].items() if key != "loss_decay_gamma"}
        assert drafter == {key: value for key, value in reference["drafter"].items() if key != "loss_decay_gamma"}


def test_method_profiles_follow_the_contract():
    gamma = {method: matrix.build(method)["drafter"]["loss_decay_gamma"] for method in matrix.QUEUE}
    assert gamma == {"dflash": 7.0, "dfly": 7.0, "dflash2": 7.0, "dspark": 4.0, "dflare": 7.0, "domino": 7.0,
                     "jetspec": None}
    assert matrix.build("dfly")["algorithm"]["train_recipe"] == {"name": "upstream_two_stage", "cold_start_fraction": 0.25}
    assert matrix.build("dspark")["algorithm"]["train_recipe"] == {"name": "upstream_dspark", "confidence_coef": 1.0}
    for method in ("dflare", "dfly"):
        assert matrix.build(method)["algorithm"][method]["context_layer_ids"] == [1, 17, 33]


@pytest.mark.parametrize("method", matrix.QUEUE)
def test_effective_config_validates_and_preset_leftovers_are_inert(method, monkeypatch):
    from vlm_dflash.cli import build_train_config, load_preset
    from vlm_dflash.config import load_config

    monkeypatch.setenv("VLM_DFLASH_TARGET_QWEN3VL", "/models/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("VELA_POOL", "/data/pool")
    args = Namespace(
        preset="single-gpu-48g", config=str(matrix.CONFIG_DIR / f"{method}_s42.yaml"), target=None, dataset=None,
        hf_dataset=None, hf_split=None, dataset_source=None, annotation=None, images_dir=None, limit=None,
        algorithm=None, set_list=[], feature_mode=None,
    )
    effective = build_train_config(args)  # runs the algorithm's validate_config
    assert effective["train"]["feature_mode"] == "online"
    formal = load_config(matrix.CONFIG_DIR / f"{method}_s42.yaml")
    preset = load_preset("single-gpu-48g")
    leftovers = [
        f"{block}.{key}"
        for block, values in preset.items()
        if isinstance(values, dict)
        for key in values
        if key not in (formal.get(block) or {})
    ]
    assert sorted(leftovers) == sorted(INERT_PRESET_KEYS)
    # Every formal value survives the preset merge unchanged.
    for block, values in formal.items():
        if isinstance(values, dict):
            for key, value in values.items():
                assert effective[block][key] == value, f"{block}.{key}"
