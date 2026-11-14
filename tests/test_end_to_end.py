"""End-to-end: rollout -> two-view teacher scoring -> consensus loss -> update -> evaluation."""

from __future__ import annotations

import dataclasses
import math

import pytest

from vdopd.demo import DemoConfig, build_toy_pair, build_toy_trainer, format_results, run_demo
from vdopd.diagnostics import position_statistics

SMALL = DemoConfig(
    max_operand=4,
    teacher_hidden=48,
    teacher_layers=2,
    student_hidden=32,
    student_layers=1,
    teacher_pretrain_steps=60,
    student_pretrain_steps=10,
    pretrain_batch_size=16,
    iterations=6,
    rollout_batch_size=8,
    eval_questions=6,
    eval_samples=2,
)


@pytest.mark.parametrize("objective", ["vdopd", "eopd_2v", "gkd_rkl"])
def test_demo_pipeline_runs_for_several_objectives(objective):
    results = run_demo(dataclasses.replace(SMALL, objective=objective))
    for key in ("teacher", "initial_student", "distilled_student"):
        row = results[key]
        assert 0.0 <= row["worst_view"] <= row["worst_template"] <= row["mean_view"] <= 1.0
        assert 0.0 <= row["avg_at_k"] <= row["pass_at_k"] <= 1.0
    training = results["training"]
    assert training["iterations"] == 6 and math.isfinite(training["last_loss"])
    assert training["teacher_scoring_flops"] > 0 and training["total_flops"] > 0
    if objective == "gkd_rkl":
        assert training["mean_mass"] is None  # a single teacher view has no consensus mass
    else:
        assert 0.0 < training["mean_mass"] <= 1.0 + 1e-6
    assert "distilled student" in format_results(results, SMALL.eval_samples)


def test_demo_is_deterministic():
    first = run_demo(SMALL)
    second = run_demo(SMALL)
    assert first["training"]["last_loss"] == second["training"]["last_loss"]
    assert first["distilled_student"] == second["distilled_student"]


def test_training_improves_the_consensus_fit_on_held_fixed_rollouts():
    """VD-OPD updates lower the consensus cross-entropy at fixed pre-update prefixes."""
    pair = build_toy_pair(SMALL)
    cfg = dataclasses.replace(SMALL, iterations=25, learning_rate=3e-3)
    trainer = build_toy_trainer(pair, cfg, pair.train_problems)
    probe = trainer.collect_rollouts(10_000)  # a rollout batch that is never trained on
    before = position_statistics(pair.student, pair.teacher, probe)
    history = trainer.train()
    after = position_statistics(pair.student, pair.teacher, probe)
    assert all(step["status"] == "ok" and step["stepped"] for step in history)
    assert after["consensus_fit_loss"].mean() < before["consensus_fit_loss"].mean()
    assert history[-1]["flops_total"] > history[0]["flops_total"]
