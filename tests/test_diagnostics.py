"""Section 3, Section 5.5 and Appendix B.7/B.8 diagnostics."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from conftest import make_trainer
from vdopd.consensus import normalized_consensus
from vdopd.diagnostics import (
    binned_utility_summary,
    candidate_logits,
    consensus_fit_loss,
    entropy_quartile_summary,
    matched_branch_updates,
    paired_view_statistics,
    position_statistics,
)
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.scoring import score_responses


def test_entropy_quartile_summary():
    entropy = np.arange(100, dtype=float)
    js = np.concatenate([np.full(25, 0.1), np.full(25, 0.2), np.full(25, 0.3), np.full(25, 0.4)])
    summary = entropy_quartile_summary(entropy[::-1].copy(), js[::-1].copy())
    assert [row["count"] for row in summary] == [25, 25, 25, 25]
    assert [row["js_median"] for row in summary] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert summary[1]["entropy_min"] == 25 and summary[1]["entropy_max"] == 49
    assert all(row["js_p5"] <= row["js_median"] <= row["js_p95"] for row in summary)


def test_binned_utility_summary_follows_appendix_b7():
    # Three populated bins out of 24: [0, 1/24), [1/24, 2/24) and the last one.
    mass = np.array([0.01, 0.02, 0.03, 0.05, 0.06, 0.99, 1.0])
    utility = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 5.0, 7.0])
    correct = np.array([1, 0, 0, 1, 1, 1, 0], dtype=bool)
    summary = binned_utility_summary(mass, utility, correct, num_bins=24, window=3)
    assert summary["count"].tolist() == [3, 2] + [0] * 21 + [2]
    assert summary["bin_centers"][0] == pytest.approx(1 / 48)
    # The three-bin moving average acts on the per-bin summaries; empty bins stay empty.
    assert summary["median"][0] == pytest.approx((2.0 + 15.0) / 2)
    assert summary["median"][1] == pytest.approx((2.0 + 15.0) / 2)
    assert summary["median"][23] == pytest.approx(6.0)  # Z = 1 falls into the last bin
    assert np.isnan(summary["median"][2:23]).all()
    assert summary["correctness"][23] == pytest.approx(0.5)
    assert summary["correctness"][0] == pytest.approx((1 / 3 + 1.0) / 2)
    unsmoothed = binned_utility_summary(mass, utility, correct, window=1)
    assert unsmoothed["median"][0] == 2.0 and unsmoothed["p90"][1] == pytest.approx(19.0)


def test_position_statistics_match_direct_computation(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    stats = position_statistics(
        trainer.student, trainer.teacher, batch, block_size=4, microbatch_size=2
    )
    original = [pair.original_ids for pair in batch.pairs]
    wrapped = [pair.view2_ids for pair in batch.pairs]
    log_p1 = score_responses(trainer.teacher, original, batch.responses)
    log_p2 = score_responses(trainer.teacher, wrapped, batch.responses)
    log_q = score_responses(trainer.student, original, batch.responses)
    target, mass = normalized_consensus(log_p1, log_p2)
    assert stats["mass"].shape == (batch.num_positions,)
    assert torch.allclose(stats["mass"], mass, atol=1e-5)
    assert torch.allclose(stats["consensus_fit_loss"], consensus_fit_loss(log_q, target), atol=1e-5)
    owners = [i for i, response in enumerate(batch.responses) for _ in response]
    assert stats["response_index"].tolist() == owners
    paired = paired_view_statistics(trainer.teacher, original, wrapped, batch.responses)
    assert torch.allclose(paired["entropy_p1"], stats["entropy_p1"], atol=1e-5)
    assert torch.allclose(paired["js_divergence"], stats["js_divergence"], atol=1e-5)


def test_matched_branches_share_state_and_measure_utility(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    state_before = copy.deepcopy(trainer.student.model.state_dict())
    flops_before = trainer.flops.total
    calls = []

    def evaluate() -> float:
        calls.append(1)
        return float(len(calls))

    names = ("arithmetic", "weighted_arithmetic", "geometric", "vdopd", "shuffled_weight")
    objectives = {name: build_objective(ObjectiveConfig(name)) for name in names}
    result = matched_branch_updates(trainer, batch, objectives, evaluate)
    assert set(result["branches"]) == set(names) and len(calls) == 1 + len(names)
    assert 0.0 < result["pre_update"]["mass"] <= 1.0
    for branch in result["branches"].values():
        assert branch["status"] == "ok" and branch["stepped"]
        assert branch["utility"].shape == (batch.num_positions,)
        assert branch["utility"].mean() > 0  # one update improves the consensus fit on average
    # The trainer is restored: same weights, empty optimizer state, unchanged FLOPs meter.
    for name, value in trainer.student.model.state_dict().items():
        assert torch.equal(value, state_before[name])
    assert trainer.optimizer.state_dict()["state"] == {}
    assert trainer.flops.total == flops_before


def test_candidate_logits_are_length_normalised_log_likelihoods(make_lm):
    """Equation 15: z_k = (1 / L_k) sum_l log pi(a_{k,l} | state, a_{k,<l})."""
    model = make_lm(seed=9)
    state = [7, 8, 9, 10]
    candidates = [[20], [21, 22, 23], [24, 25]]
    values = candidate_logits(model, state, candidates)
    for value, candidate in zip(values, candidates, strict=True):
        total = 0.0
        for position, token in enumerate(candidate):
            prefix = torch.tensor([state + candidate[:position]])
            logits = model.logits(prefix, torch.ones_like(prefix))[0, -1].float()
            total += torch.log_softmax(logits, -1)[token].item()
        assert value == pytest.approx(total / len(candidate), abs=1e-5)
