"""Continuation of a finished run: constant rate, held objective schedule, exact own resume.

Runs the real training loop on the tiny CPU draft of ``test_resume``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_resume import _CFG, _CONTRACT, _tiny_module, _TinyDataset
from vlm_dflash.dflash.train import (
    _check_continuation_record,
    _continuation_record,
    continuation_settings,
    load_continuation_source,
    run_training_loop,
)

_LR = 2e-4


def _run(module, *, total_steps, resume_state=None, sink=None, continuation=None, save_every=2):
    return run_training_loop(
        module,
        _TinyDataset(),
        _CFG,
        _CONTRACT,
        torch.device("cpu"),
        total_steps=total_steps,
        seed=11,
        shuffle=True,
        log_path=None,
        resume_state=resume_state,
        on_checkpoint=sink,
        save_every=save_every,
        continuation=continuation,
    )


def _finished_source() -> dict:
    """The completed-run state a 4-step run writes after its last update."""
    states: list[dict] = []
    _run(_tiny_module(), total_steps=4, sink=states.append)
    assert int(states[-1]["step"]) == 4
    return states[-1]


def _continue(source: dict, *, sink=None, module=None):
    # A loaded optimizer keeps the state's step tensors and advances them in
    # place, so every run gets its own copy (on disk a state is read once).
    return _run(
        module or _tiny_module(),
        total_steps=8,
        resume_state=copy.deepcopy(source),
        sink=sink,
        continuation={"source_steps": 4, "lr": _LR, "fresh": True},
    )


def test_continuation_runs_a_constant_rate_on_the_source_moments():
    source = _finished_source()
    source_lr = source["optimizer"]["param_groups"][0]["lr"]
    assert source_lr != pytest.approx(_LR)
    output = _continue(source)
    rows = output["history"]
    assert [row["step"] for row in rows] == list(range(1, 9))
    assert rows[:4] == source["history"]
    assert [row["lr"] for row in rows[4:]] == pytest.approx([_LR] * 4)
    assert output["summary"]["resumed_from_step"] == 4
    assert output["summary"]["continuation"]["lr"] == _LR
    assert output["summary"]["continuation"]["objective_schedule_held_at"] == [4, 4]


def test_continuation_keeps_adamw_state_and_counts_steps_on():
    source = _finished_source()
    states: list[dict] = []
    _continue(source, sink=states.append)
    final = states[-1]
    assert int(final["step"]) == 8
    assert int(final["steps_total_planned"]) == 8
    source_steps = {int(entry["step"]) for entry in source["optimizer"]["state"].values()}
    final_steps = {int(entry["step"]) for entry in final["optimizer"]["state"].values()}
    assert source_steps == {4}
    assert final_steps == {8}
    assert [group["lr"] for group in final["optimizer"]["param_groups"]] == pytest.approx([_LR])


def test_continuation_holds_the_progress_hook_at_the_source_end():
    calls: list[tuple[int, int]] = []
    module = _tiny_module()
    module.set_training_progress = lambda step, total: calls.append((int(step), int(total)))
    _continue(_finished_source(), module=module)
    assert calls and set(calls) == {(4, 4)}


def test_continuation_resume_reproduces_the_uninterrupted_continuation():
    source = _finished_source()
    uninterrupted = [row["loss"] for row in _continue(source)["history"]]
    mid: dict = {}
    _continue(source, sink=lambda state: mid.setdefault("state", state) if int(state["step"]) == 6 else None)
    assert int(mid["state"]["step"]) == 6
    resumed = _run(
        _tiny_module(),
        total_steps=8,
        resume_state=mid["state"],
        continuation={"source_steps": 4, "lr": _LR, "fresh": False},
    )
    rows = resumed["history"]
    assert [row["lr"] for row in rows[6:]] == pytest.approx([_LR] * 2)
    for first, second in zip([row["loss"] for row in rows], uninterrupted):
        assert first == pytest.approx(second, abs=1e-9)


def test_continuation_needs_a_state():
    with pytest.raises(ValueError, match="needs the source"):
        _run(_tiny_module(), total_steps=8, continuation={"source_steps": 4, "lr": _LR, "fresh": True})


def test_continuation_settings_validation():
    assert continuation_settings({}) is None
    cont = continuation_settings({"continuation": {"source_state": "a.pt", "add_steps": 5, "lr": "1e-5"}})
    assert cont == {
        "source_state": "a.pt",
        "source_final": None,
        "add_steps": 5,
        "lr": 1e-5,
        "lr_schedule": "constant",
    }
    with pytest.raises(ValueError, match="source_state"):
        continuation_settings({"continuation": {"add_steps": 5, "lr": 1e-5}})
    with pytest.raises(ValueError, match="add_steps"):
        continuation_settings({"continuation": {"source_state": "a.pt", "lr": 1e-5}})
    with pytest.raises(ValueError, match="lr"):
        continuation_settings({"continuation": {"source_state": "a.pt", "add_steps": 5}})


def test_continuation_record_checks():
    cont = continuation_settings({"continuation": {"source_state": "a.pt", "add_steps": 5, "lr": 1e-5}})
    record = _continuation_record(cont, 4)
    assert record["source_steps"] == 4
    assert _continuation_record(None, 4) is None
    _check_continuation_record({"continuation": dict(record)}, record)
    _check_continuation_record({}, None)
    with pytest.raises(ValueError, match="written by a continuation"):
        _check_continuation_record({"continuation": dict(record)}, None)
    with pytest.raises(ValueError, match="not a continuation state"):
        _check_continuation_record({}, record)
    with pytest.raises(ValueError, match="continues"):
        _check_continuation_record({"continuation": {**record, "lr": 2e-5}}, record)


_GATE = {
    "algorithm_name": "dflash",
    "mode": "online",
    "feature_source_digest": "src",
    "contract": {},
    "plan_version": "vlm-dflash-training-plan-v1",
    "plan_digest": "plan",
    "cfg": {},
}


def _gated(state: dict, *, step: int = 4) -> dict:
    return {
        **state,
        "algorithm": "dflash",
        "feature_mode": "online",
        "feature_source_digest": "src",
        "training_plan_version": "vlm-dflash-training-plan-v1",
        "training_plan_digest": "plan",
        "step": step,
    }


def _write(tmp_path: Path, name: str, payload: dict) -> str:
    path = tmp_path / name
    torch.save(payload, path)
    return str(path)


def test_load_continuation_source_verifies_state_and_final(tmp_path):
    state = _gated(_finished_source())
    final = {"algorithm": "dflash", "step": 4, "model": state["model"]}
    cont = {
        "source_state": _write(tmp_path, "last.pt", state),
        "source_final": _write(tmp_path, "final.pt", final),
    }
    loaded = load_continuation_source(cont, source_steps=4, **_GATE)
    assert int(loaded["step"]) == 4

    with pytest.raises(ValueError, match="only a finished run"):
        unfinished = _gated(state, step=2)
        load_continuation_source(
            {"source_state": _write(tmp_path, "mid.pt", unfinished)}, source_steps=4, **_GATE
        )

    name = next(iter(state["master"]))
    drifted = dict(state)
    drifted["master"] = {**state["master"], name: state["master"][name] + 1.0}
    with pytest.raises(RuntimeError, match="does not round"):
        load_continuation_source({"source_state": _write(tmp_path, "drift.pt", drifted)}, source_steps=4, **_GATE)

    other = {**final, "model": {**final["model"], name: final["model"][name] + 1.0}}
    with pytest.raises(RuntimeError, match="differs"):
        load_continuation_source(
            {"source_state": cont["source_state"], "source_final": _write(tmp_path, "other.pt", other)},
            source_steps=4,
            **_GATE,
        )

    with pytest.raises(ValueError, match="itself a continuation"):
        chained = {**state, "continuation": {"source_steps": 4}}
        load_continuation_source({"source_state": _write(tmp_path, "chained.pt", chained)}, source_steps=4, **_GATE)

    with pytest.raises(ValueError, match="expected 'dflash' at step 4"):
        wrong = {**final, "step": 3}
        load_continuation_source(
            {"source_state": cont["source_state"], "source_final": _write(tmp_path, "wrong.pt", wrong)},
            source_steps=4,
            **_GATE,
        )
