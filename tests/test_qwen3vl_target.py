"""Qwen3-VL-8B target contract on CPU: layer routing, inherited draft shape, MRoPE, mask token."""

from __future__ import annotations

import json

import pytest

from vlm_dflash.adapters import get_adapter
from vlm_dflash.contract import build_target_layer_ids
from vlm_dflash.dflash.factory import build_draft_config

#: The text/vision blocks of Qwen/Qwen3-VL-8B-Instruct ``config.json`` (revision 0c351dd).
QWEN3VL_8B_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


def _contract(tmp_path) -> dict:
    (tmp_path / "config.json").write_text(json.dumps(QWEN3VL_8B_CONFIG))
    return {
        "target": {"model_path": str(tmp_path)},
        "layers": {"num_target_layers": 36, "draft_layer_ids": build_target_layer_ids(36, 3)},
        "text": {"hidden_size": 4096},
        "rope": {"sections": [24, 20, 20], "interleaved": True, "partial_rotary_factor": None},
    }


def _drafter(**extra) -> dict:
    return {"drafter": {"num_hidden_layers": 3, "block_size": 8, "mask_token_id": 151669, **extra}}


def test_qwen3vl_hidden_layer_mapping():
    # DFlash routing over 36 target layers: start at 1, end at 36 - 3, uniform.
    assert build_target_layer_ids(36, 3) == [1, 17, 33]


def test_qwen3vl_draft_inherits_target_shape(tmp_path):
    config = build_draft_config(_contract(tmp_path), _drafter())
    assert (config.hidden_size, config.intermediate_size) == (4096, 12288)
    assert (config.num_attention_heads, config.num_key_value_heads, config.head_dim) == (32, 8, 128)
    assert config.num_hidden_layers == 3
    assert config.vocab_size == 151936
    assert config.dflash_config["target_layer_ids"] == [1, 17, 33]
    assert config.dflash_config["mask_token_id"] == 151669
    # MRoPE moves to the explicit spatial settings; the draft rope is plain 1-D.
    assert config.dflash_config["spatial_rope_sections"] == [24, 20, 20]
    assert config.dflash_config["spatial_rope_interleaved"] is True
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 5000000.0}


def test_qwen3vl_draft_accepts_stated_matching_shape(tmp_path):
    stated = {"hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
    config = build_draft_config(_contract(tmp_path), _drafter(intermediate_size=12288, **stated))
    assert (config.hidden_size, config.head_dim) == (4096, 128)


@pytest.mark.parametrize(
    ("key", "value"), [("hidden_size", 2048), ("num_attention_heads", 16), ("num_key_value_heads", 4), ("head_dim", 64)]
)
def test_qwen3vl_draft_rejects_stated_mismatching_shape(tmp_path, key, value):
    with pytest.raises(ValueError, match=f"drafter.{key}={value} disagrees"):
        build_draft_config(_contract(tmp_path), _drafter(**{key: value}))


def test_qwen3vl_mrope_contract():
    from transformers import AutoConfig

    adapter = get_adapter("qwen3_vl")
    assert adapter.info.model_class == "Qwen3VLForConditionalGeneration"
    config = AutoConfig.for_model(**QWEN3VL_8B_CONFIG)
    rope = adapter.resolve_rope(config, config.text_config, 128, {"spatial_rope_sections": [24, 20, 20]})
    assert rope["mode"] == "mrope"
    assert rope["sections"] == [24, 20, 20]
    assert rope["interleaved"] is True
