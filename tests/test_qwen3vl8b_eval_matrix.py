"""CPU tests of the a100_qwen3vl8b_200k evaluation helpers: scorers, session messages, summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "q3vl_eval_matrix", Path(__file__).resolve().parents[1] / "scripts" / "a100_qwen3vl8b_200k" / "eval_matrix.py"
)
em = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(em)


def test_vqa_accuracy_is_the_leave_one_out_mean():
    answers = ["dakota"] * 3 + ["dakota digital"] * 7
    assert em.vqa_accuracy("Dakota.", answers) == pytest.approx(1.0 * 7 / 10 + (2 / 3) * 3 / 10)
    assert em.vqa_accuracy("two", ["2"] * 10) == 1.0  # number words normalize
    assert em.vqa_accuracy("the cat", ["cat"] * 10) == 1.0  # articles drop
    assert em.vqa_accuracy("dog", ["cat"] * 10) == 0.0


def test_anls_threshold_and_case():
    assert em.anls("Paris", ["paris"]) == 1.0
    assert em.anls("pariss", ["paris"]) == pytest.approx(1 - 1 / 6)
    assert em.anls("london", ["paris"]) == 0.0
    assert em.anls("x", ["y", "x"]) == 1.0


def test_relaxed_accuracy_numbers_and_strings():
    assert em.relaxed_accuracy("104", ["100"]) == 1.0
    assert em.relaxed_accuracy("106", ["100"]) == 0.0
    assert em.relaxed_accuracy("1,050.", ["1000"]) == 1.0
    assert em.relaxed_accuracy("Yes", ["yes"]) == 1.0
    assert em.relaxed_accuracy("0", ["0"]) == 1.0


def test_build_messages_single_multi_turn_and_dataset_history():
    row = {"system": "sys", "turns": ["q1", "q2"]}
    first = em.build_messages(row, 0, [])
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[1]["content"][0] == {"type": "image"}
    second = em.build_messages(row, 1, [("q1", "a1")])
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"][0] == {"type": "image"} and second[3]["content"] == [{"type": "text", "text": "q2"}]
    held = {"turns": ["h1", "h2"], "history": [{"from": "human", "value": "<image>\nh1"}, {"from": "gpt", "value": "g1"},
                                                {"from": "human", "value": "h2"}, {"from": "gpt", "value": "g2"}]}
    messages = em.build_messages(held, 0, [])
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == [{"type": "image"}, {"type": "text", "text": "h1"}]


def _sd(record_id, topic, new, rounds, accepted, wall, ar_wall, exact):
    return {"id": record_id, "topic": topic, "new_tokens": new, "wall_seconds": wall, "natural_stop": True,
            "rounds": rounds, "accepted_draft_tokens": accepted, "proposed_draft_tokens": rounds * 7,
            "target_calls": rounds + 1, "drafter_calls": rounds, "strict_fallback_rounds": 0,
            "ar_wall_seconds": ar_wall, "sequence_exact": exact, "stopping_exact": exact}


def test_summarize_sd_speedup_is_sum_over_sum_and_al_definitions():
    records = [_sd("a", "x", 10, 5, 5, 1.0, 3.0, True), _sd("b", "y", 30, 10, 20, 3.0, 3.0, False)]
    summary = em.summarize(records, "sd")
    assert summary["AL_draft"] == pytest.approx(25 / 15)
    assert summary["AL_progress"] == pytest.approx(40 / 15)
    assert summary["speedup"] == pytest.approx(6.0 / 4.0)
    assert summary["speedup_mean_of_ratios"] == pytest.approx((3.0 + 1.0) / 2)
    assert summary["token_exact"] is False and summary["failed_ids"] == ["b"]
    assert set(summary["by_topic"]) == {"x", "y"}
    assert summary["by_topic"]["x"]["AL_draft"] == pytest.approx(1.0)


def test_stopped_naturally():
    assert em.stopped_naturally([5, 151645], {151645}, 128)
    assert not em.stopped_naturally([5, 6], {151645}, 2)
