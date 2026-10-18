"""Qwen3.5/3.6-MoE adapter (hybrid Gated DeltaNet + full-attention MoE target).

Shares the Qwen3-VL processor and staged-capture interface
(``compute_3d_position_ids`` + ``mm_token_type_ids``). The family differences
the pipeline must honor are all resolved from the target config at runtime:

* partial rotary (``partial_rotary_factor`` 0.25: 64 of 256 head channels
  rotate) with interleaved MRoPE sections ``[11, 11, 10]``;
* patch size 16 (not 14) and pixel bounds exposed as ``size.shortest_edge`` /
  ``size.longest_edge``;
* a text config without a dense ``intermediate_size``: the draft MLP width
  must be set explicitly in the drafter block.

The target's built-in MTP head is not loaded (``mtp.*`` weights are ignored on
load), so the target-only baseline is plain autoregressive decoding.
"""

from __future__ import annotations

from .base import AdapterInfo, VLMAdapter


class Qwen35MoeAdapter(VLMAdapter):
    info = AdapterInfo(
        name="qwen3_5_moe",
        family="qwen-vl",
        model_types=("qwen3_5_moe",),
        model_class="Qwen3_5MoeForConditionalGeneration",
        notes=(
            "Hybrid Gated DeltaNet / full-attention MoE target with partial rotary (factor 0.25) and "
            "interleaved MRoPE [11, 11, 10]. Install flash-linear-attention for the GDN chunk kernel; "
            "the torch fallback is too slow for training. Pin image_max_pixels explicitly: the processor "
            "default (16M pixels) allows up to 16384 visual tokens per image."
        ),
    )
