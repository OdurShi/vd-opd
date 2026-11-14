"""Supplementary E09 (pair gradients), E16 (diagnostics by Z) and E17 (matched updates)."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import PrefixSumLM, make_trainer, random_log_probs
from vdopd import theory as th
from vdopd.consensus import geometric_consensus, squared_hellinger
from vdopd.data.problems import build_messages
from vdopd.data.synthetic import SYNTHETIC_TEMPLATE, make_arithmetic_problems
from vdopd.diagnostics import (
    AGREEMENT_ROWS,
    OUTCOMES,
    Z_STRATA,
    AnalysisState,
    agreement_summary,
    agreement_table,
    build_analysis_states,
    candidate_distributions,
    candidate_logits,
    continuation_outcome,
    hellinger_tail_split,
    matched_branch_updates,
    oracle_kl,
    oracle_kl_decrease,
    pair_gradient_cosines,
    pair_outcomes,
    position_statistics,
    quartile_labels,
    sample_final_answers,
    sequence_log_likelihoods,
    shared_wrong_answers,
    state_agreement,
    stratum_means,
)
from vdopd.eval.evaluator import EvalConfig
from vdopd.objectives import ObjectiveConfig, build_objective
from vdopd.trainer import RolloutBatch, RolloutConfig
from vdopd.views import TRAIN_WRAPPERS, select_views

REPOSITORY = Path(__file__).resolve().parents[1]
PROBLEMS = make_arithmetic_problems(max_operand=4)[:6]
PROMPTS = [build_messages(p.question, SYNTHETIC_TEMPLATE) for p in PROBLEMS]
ROLLOUT = RolloutConfig(batch_size=6, max_response_tokens=7, generation_batch_size=4)
# Supplementary E17: the five matched updates.
E17_BRANCHES = {
    "EOPD(full)": "eopd_full",
    "Geometric-1": "geometric",
    "Batch-mean-Z": "batch_mean_weight",
    "Shuffled-Z": "shuffled_weight",
    "VD-OPD": "vdopd",
}


def flat_gradient(trainer) -> torch.Tensor:
    return torch.cat([p.grad.detach().flatten().double() for p in trainer.parameters])


# ------------------------------------------------------------------ E09
def test_pair_gradient_cosines_match_a_direct_computation(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    weights_before = copy.deepcopy(trainer.student.model.state_dict())
    flops_before = trainer.flops.state_dict()
    result = pair_gradient_cosines(trainer, batch)
    assert result["views"] == [w.name for w in TRAIN_WRAPPERS]
    assert result["num_positions"] == batch.num_positions
    # The diagnostic leaves the trainer untouched.
    assert all(p.grad is None for p in trainer.parameters)
    assert trainer.flops.state_dict() == flops_before
    for name, value in trainer.student.model.state_dict().items():
        assert torch.equal(value, weights_before[name])

    gradients = []
    for wrapper in TRAIN_WRAPPERS:  # every response paired with the same wrapper
        pairs = [
            select_views(
                trainer.prompts[index],
                codec,
                max_response_tokens=7,
                teacher_context=trainer.teacher.max_context,
                student_context=trainer.student.max_context,
                rng=np.random.default_rng(0),
                wrappers=(wrapper,),
            )
            for index in batch.prompt_indices
        ]
        assert all(pair.wrapper == wrapper for pair in pairs)
        trainer.accumulate_gradients(RolloutBatch(0, batch.prompt_indices, pairs, batch.responses))
        gradients.append(flat_gradient(trainer))
    trainer.optimizer.zero_grad(set_to_none=True)
    mean = torch.stack(gradients).mean(0)
    for wrapper, gradient in zip(TRAIN_WRAPPERS, gradients, strict=True):
        cosine = torch.dot(gradient, mean) / (gradient.norm() * mean.norm())
        assert result["cosine_to_mean"][wrapper.name] == pytest.approx(cosine.item(), abs=1e-6)
        assert result["gradient_norm"][wrapper.name] == pytest.approx(gradient.norm().item())
        assert -1.0 <= result["cosine_to_mean"][wrapper.name] <= 1.0
    assert result["mean_gradient_norm"] == pytest.approx(mean.norm().item(), rel=1e-6)
    pairwise = np.array(result["pairwise_cosine"])
    assert np.allclose(np.diag(pairwise), 1.0) and np.allclose(pairwise, pairwise.T)


def test_identical_pairs_have_cosine_one(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(1)
    same = [TRAIN_WRAPPERS[0]] * 3
    result = pair_gradient_cosines(trainer, batch, views=same)
    assert list(result["cosine_to_mean"].values()) == pytest.approx([1.0], abs=1e-9)
    assert np.allclose(result["pairwise_cosine"], 1.0)
    with pytest.raises(ValueError):
        pair_gradient_cosines(trainer, batch, views=())


def test_pair_gradient_script_table_layout():
    """The script writes the two columns of the E09 table with the paper's view labels."""
    text = (REPOSITORY / "scripts" / "pair_gradient_cosine.py").read_text()
    assert 'COLUMNS = ("paired_view", "cosine_to_mean_gradient")' in text
    for label in ("Instruction 1", "Instruction 2", "Layout wrapper"):
        assert label in text


# ------------------------------------------------------------------ E16
def test_hellinger_tail_split_hand_examples():
    """Share of 0.5 ||sqrt(p1) - sqrt(p2)||^2 outside the union of the top-k tokens."""
    log = lambda values: torch.tensor(values, dtype=torch.float64).log().float()  # noqa: E731
    # The two views disagree only on their top tokens: nothing lies outside the union.
    outside, total = hellinger_tail_split(
        log([0.7, 0.1, 0.1, 0.1]), log([0.1, 0.7, 0.1, 0.1]), top_k=1
    )
    assert outside.item() == pytest.approx(0.0, abs=1e-7)
    assert total.item() == pytest.approx((math.sqrt(0.7) - math.sqrt(0.1)) ** 2, abs=1e-6)
    # The views share their top token and disagree on the rest: all of it lies outside.
    outside, total = hellinger_tail_split(log([0.5, 0.3, 0.2]), log([0.5, 0.2, 0.3]), top_k=1)
    assert outside.item() == pytest.approx(total.item(), abs=1e-7) and total.item() > 0
    # In general: total = 1 - Z, the outside part shrinks as k grows and vanishes at k = d.
    log_p1, log_p2 = random_log_probs((20, 30), 1), random_log_probs((20, 30), 2)
    previous = None
    for k in (1, 5, 10, 30, 99):
        outside, total = hellinger_tail_split(log_p1, log_p2, top_k=k)
        assert torch.allclose(total, squared_hellinger(log_p1, log_p2), atol=1e-6)
        assert torch.allclose(total, 1 - geometric_consensus(log_p1, log_p2)[1], atol=1e-5)
        assert (outside >= 0).all() and (outside <= total + 1e-7).all()
        if previous is not None:
            assert (outside <= previous + 1e-7).all()
        previous = outside
    assert torch.allclose(previous, torch.zeros(20), atol=1e-7)


def test_quartile_labels_are_equal_sized_rank_groups():
    values = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4])
    assert quartile_labels(values).tolist() == [4, 1, 3, 2, 3, 1, 4, 2]
    labels = quartile_labels(np.random.default_rng(0).random(1000))
    assert np.bincount(labels)[1:].tolist() == [250, 250, 250, 250]
    assert Z_STRATA == (("Low", 1), ("High", 4))


def test_continuation_outcome_categories():
    assert continuation_outcome("5", "5.0", "5") == "both_correct"
    assert continuation_outcome("5", "7", "5") == "one_correct"
    assert continuation_outcome(None, "5", "5") == "one_correct"
    assert continuation_outcome("7", "7.0", "5") == "same_wrong"  # equivalent wrong answers
    assert continuation_outcome("7", "8", "5") == "different_wrong"
    assert continuation_outcome(None, None, "5") == "different_wrong"  # no shared answer
    assert continuation_outcome("7", None, "5") == "different_wrong"
    outcomes = pair_outcomes(
        [["5", "7", "7"], ["1", None, "2"]], [["5", "7", "8"], ["3", "3", "2"]], ["5", "3"]
    )
    names = [[OUTCOMES[code] for code in row] for row in outcomes]
    assert names == [
        ["both_correct", "same_wrong", "different_wrong"],
        ["one_correct", "one_correct", "same_wrong"],
    ]
    shared = shared_wrong_answers(
        [["7", "7.0", "9", "5"], ["3", "3", "3", "3"], ["8", "6", "8", "6"]],
        [["7", "7", "9", "5"], ["3", "1", "2", "3"], ["8", "6", "8", "6"]],
        ["5", "3", "5"],
    )
    assert shared == ["7", None, "8"]  # majority shared wrong answer; ties keep the first


def test_agreement_summary_and_table():
    mass = np.array([0.1, 0.2, 0.9, 1.0, 0.5, 0.6, 0.3, 0.8])
    agreement = {
        "mass": mass,
        "same_top1": np.array([0, 1, 1, 1, 0, 1, 0, 1], dtype=bool),
        "hellinger_total": 1.0 - mass,
        "hellinger_outside": np.array([0.3, 0.1, 0.05, 0.0, 0.1, 0.1, 0.2, 0.1]),
    }
    outcomes = np.array([[0, 1], [2, 3], [0, 0], [0, 0], [1, 1], [2, 2], [3, 3], [0, 2]])
    summary = agreement_summary(agreement, outcomes)
    assert [row["quartile"] for row in summary] == [1, 2, 3, 4]
    low, high = summary[0], summary[3]  # states with Z in {0.1, 0.2} and {0.9, 1.0}
    assert (low["states"], low["pairs"], high["states"], high["pairs"]) == (2, 4, 2, 4)
    assert low["mean_mass"] == pytest.approx(0.15) and high["mean_mass"] == pytest.approx(0.95)
    assert low["same_top1"] == 0.5 and high["same_top1"] == 1.0
    # Share of the stratum's total Hellinger disagreement, not a mean of per-state shares.
    assert low["tail_share"] == pytest.approx((0.3 + 0.1) / (0.9 + 0.8))
    assert high["tail_share"] == pytest.approx(0.05 / 0.1)
    assert [low[name] for name in OUTCOMES] == [0.25, 0.25, 0.25, 0.25]  # micro-average
    assert [high[name] for name in OUTCOMES] == [1.0, 0.0, 0.0, 0.0]
    for row in summary:
        assert sum(row[name] for name in OUTCOMES) == pytest.approx(1.0)
    table = agreement_table(summary)
    assert [row["diagnostic"] for row in table] == [label for label, *_ in AGREEMENT_ROWS]
    assert table[0] == {"diagnostic": "States", "low_z_q1": "2", "high_z_q4": "2"}
    assert table[2] == {"diagnostic": "Mean Z", "low_z_q1": "0.15", "high_z_q4": "0.95"}
    assert table[3]["low_z_q1"] == "50" and table[5]["high_z_q4"] == "100"
    without = agreement_table(agreement_summary(agreement))
    assert [row["diagnostic"] for row in without] == [
        "States",
        "Mean Z",
        "Same top-1 next token (%)",
        "Disagreement outside union of top-10 tokens (%)",
    ]


def test_agreement_table_uses_the_layout_of_the_reported_table():
    path = (
        REPOSITORY / "results/supplementary/06_agreement_analysis/E16_teacher_diagnostics_by_Z.csv"
    )
    if not path.is_file():
        pytest.skip("the reported table is not present")
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["diagnostic", "low_z_q1", "high_z_q4"]
    assert [row["diagnostic"] for row in rows] == [label for label, *_ in AGREEMENT_ROWS]


def test_analysis_states_are_fixed_prefixes_of_student_responses(make_lm, codec):
    student = make_lm(seed=1)
    kwargs = {"rollout": ROLLOUT, "seed": 11, "num_prefixes": 4, "teacher_context": 256}
    states = build_analysis_states(student, codec, PROMPTS, **kwargs)
    assert len(states) == 4 * len(PROMPTS)  # four prefixes per question
    assert [s.question for s in states] == [q for q in range(len(PROMPTS)) for _ in range(4)]
    assert states == build_analysis_states(student, codec, PROMPTS, **kwargs)  # seeded
    assert states != build_analysis_states(student, codec, PROMPTS, **{**kwargs, "seed": 22})
    eos = codec.tokenizer.eos_token_id
    wrappers = {w.name for w in TRAIN_WRAPPERS}
    for state in states:
        assert list(state.original_ids) == codec.encode(PROMPTS[state.question])
        assert state.wrapper in wrappers and len(state.view2_ids) > len(state.original_ids)
        assert len(state.prefix) < ROLLOUT.max_response_tokens and eos not in state.prefix
    # One alternative view per question, shared by its four states.
    assert all(len({s.view2_ids for s in states if s.question == q}) == 1 for q in range(6))
    assert len({len(s.prefix) for s in states}) > 1  # cut positions vary


def test_state_agreement_matches_the_teacher_next_token_distributions(make_lm, codec):
    student, teacher = make_lm(seed=1), make_lm(seed=2, hidden_size=48)
    states = build_analysis_states(student, codec, PROMPTS, rollout=ROLLOUT, seed=11)
    agreement = state_agreement(teacher, states, block_size=5, microbatch_size=3, top_k=10)
    assert all(len(values) == len(states) for values in agreement.values())
    for index in (0, 7, len(states) - 1):
        state = states[index]
        log_probs = []
        for prompt in (state.original_ids, state.view2_ids):
            ids = torch.tensor([list(prompt) + list(state.prefix)])
            logits = teacher.logits(ids, torch.ones_like(ids))[0, -1].float()
            log_probs.append(torch.log_softmax(logits, -1))
        mass = geometric_consensus(*log_probs)[1].item()
        outside, total = hellinger_tail_split(*log_probs, top_k=10)
        assert agreement["mass"][index] == pytest.approx(mass, abs=1e-5)
        assert agreement["hellinger_total"][index] == pytest.approx(1 - mass, abs=1e-5)
        assert agreement["hellinger_outside"][index] == pytest.approx(outside.item(), abs=1e-5)
        assert agreement["same_top1"][index] == (log_probs[0].argmax() == log_probs[1].argmax())
    again = state_agreement(teacher, states, block_size=64, microbatch_size=64)
    assert np.allclose(again["mass"], agreement["mass"], atol=1e-5)
    assert all(len(v) == 0 for v in state_agreement(teacher, []).values())


class ContinuationModel:
    """Continues every state with a fixed token sequence and records the calls."""

    max_context = None

    def __init__(self, codec, continuation: str) -> None:
        self.tokens = codec.encode_response(continuation)
        self.calls: list[dict] = []

    def generate(self, prompts, *, max_new_tokens, temperature, top_p, seed):
        self.calls.append(
            {
                "prompts": [list(p) for p in prompts],
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
            }
        )
        return [self.tokens[:max_new_tokens] for _ in prompts]


def test_sampled_continuations_complete_the_fixed_prefix(codec):
    cfg = EvalConfig(num_samples=8, max_response_tokens=12, batch_size=4, seed=3)
    prompts = [codec.encode(PROMPTS[0]), codec.encode(PROMPTS[1])]
    prefixes = [codec.tokenizer.encode_text("\\boxed{1"), []]
    model = ContinuationModel(codec, "7}")
    answers = sample_final_answers(model, codec, prompts, prefixes, cfg, num_samples=3, seed=5)
    # The final answer is read from the complete response: prefix followed by continuation.
    assert answers == [["17"] * 3, [None] * 3]
    assert [len(call["prompts"]) for call in model.calls] == [4, 2]  # six jobs in batches of 4
    assert model.calls[0]["prompts"][0] == prompts[0] + prefixes[0]
    assert all((c["temperature"], c["top_p"]) == (1.0, 0.8) for c in model.calls)
    # The cap covers the whole response: a state with an 8-token prefix may add 4 tokens.
    alone = ContinuationModel(codec, "7}")
    sample_final_answers(alone, codec, prompts[:1], prefixes[:1], cfg, num_samples=2, seed=5)
    assert len(prefixes[0]) == 8 and alone.calls[0]["max_new_tokens"] == 12 - 8
    assert model.calls[0]["max_new_tokens"] == 12  # the batch also holds an empty prefix
    # Same arguments, same seeds: paired streams for two views or for before/after an update.
    other = ContinuationModel(codec, "7}")
    sample_final_answers(other, codec, prompts[::-1], prefixes, cfg, num_samples=3, seed=5)
    assert [c["seed"] for c in other.calls] == [c["seed"] for c in model.calls]
    stream = ContinuationModel(codec, "7}")
    sample_final_answers(stream, codec, prompts, prefixes, cfg, num_samples=3, seed=5, stream=1)
    assert [c["seed"] for c in stream.calls] != [c["seed"] for c in model.calls]
    # A continuation is cut where the response cap is reached.
    tight = EvalConfig(max_response_tokens=9, batch_size=4)
    capped = sample_final_answers(
        model, codec, prompts[:1], prefixes[:1], tight, num_samples=1, seed=5
    )
    assert capped == [[None]]  # "\boxed{1" + "7" has no closing brace
    assert model.calls[-1]["max_new_tokens"] == 1
    full = EvalConfig(max_response_tokens=8, batch_size=4)  # the prefix already fills the cap
    calls = len(model.calls)
    assert sample_final_answers(
        model, codec, prompts[:1], prefixes[:1], full, num_samples=1, seed=5
    ) == [[None]]
    assert len(model.calls) == calls  # nothing is generated
    with pytest.raises(ValueError):
        sample_final_answers(model, codec, prompts, prefixes[:1], cfg, num_samples=1, seed=5)


# ------------------------------------------------------------------ E17
def test_candidate_distributions_are_softmaxes_of_the_candidate_logits(make_lm):
    model = make_lm(seed=9)
    states = [[7, 8, 9, 10], [11, 12], [13, 14, 15]]
    candidates = [[[20], [21, 22, 23], [24, 25], [26]], [[30, 31], [32]], [[40], [41], [42], [43]]]
    distributions = candidate_distributions(model, states, candidates, batch_size=2, block_size=3)
    for state, options, distribution in zip(states, candidates, distributions, strict=True):
        logits = candidate_logits(model, state, options)
        assert np.allclose(distribution, th.softmax(logits), atol=1e-9)
        assert distribution.sum() == pytest.approx(1.0) and len(distribution) == len(options)
    single = candidate_distributions(model, states, candidates, batch_size=1)
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(single, distributions, strict=True))
    likelihoods = sequence_log_likelihoods(model, [states[0]] * 2, [[20], [21, 22, 23]])
    assert np.allclose(likelihoods, candidate_logits(model, states[0], candidates[0])[:2])
    with pytest.raises(ValueError):
        candidate_distributions(model, states, candidates[:2])
    with pytest.raises(ValueError):
        sequence_log_likelihoods(model, [states[0]], [[]])


def test_oracle_kl_uses_the_fixed_four_candidate_target():
    """p* = (0.99, 0.01/3, 0.01/3, 0.01/3); positive utility = the oracle KL decreased."""
    target = th.smoothed_reference(0)
    assert target.tolist() == pytest.approx([0.99, 0.01 / 3, 0.01 / 3, 0.01 / 3])
    q = np.array([0.4, 0.3, 0.2, 0.1])
    assert oracle_kl(q) == pytest.approx(th.kl(target, q))
    assert oracle_kl(target) == pytest.approx(0.0, abs=1e-12)
    better, worse = np.array([0.7, 0.1, 0.1, 0.1]), np.array([0.1, 0.7, 0.1, 0.1])
    utility = oracle_kl_decrease([q, q, q], [better, worse, q])
    assert utility[0] > 0 > utility[1] and utility[2] == 0.0
    assert utility[0] == pytest.approx(th.kl(target, q) - th.kl(target, better))
    second = oracle_kl_decrease([q], [worse], correct_index=1)
    assert second[0] == pytest.approx(
        th.kl(th.smoothed_reference(1), q) - th.kl(th.smoothed_reference(1), worse)
    )
    assert second[0] > 0


def test_stratum_means_report_low_and_high_z_with_question_bootstrap():
    rng = np.random.default_rng(0)
    questions = np.repeat(np.arange(25), 4)  # four states per question
    mass = rng.random(100)
    quartiles = quartile_labels(mass)
    values = np.where(quartiles == 1, -1.0, 2.0) + 0.1 * rng.normal(size=100)
    result = stratum_means(values, questions, quartiles, num_resamples=500, seed=1)
    assert set(result) == {"Low", "High"}
    assert result["Low"].estimate == pytest.approx(values[quartiles == 1].mean())
    assert result["High"].estimate == pytest.approx(values[quartiles == 4].mean())
    for ci in result.values():
        assert ci.low <= ci.estimate <= ci.high and ci.confidence == 0.95
    assert result["Low"].high < 0 < result["High"].low
    # Shared-wrong quantities use only the eligible states of a stratum.
    eligible = rng.random(100) < 0.3
    masked = np.where(eligible, values, np.nan)
    restricted = stratum_means(masked, questions, quartiles, eligible=eligible, num_resamples=200)
    assert restricted["Low"].estimate == pytest.approx(values[eligible & (quartiles == 1)].mean())
    empty = stratum_means(values, questions, quartiles, eligible=np.zeros(100, dtype=bool))
    assert empty == {"Low": None, "High": None}


def test_cached_teacher_states_reproduce_the_position_statistics(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    kwargs = {"block_size": 5, "microbatch_size": trainer.cfg.microbatch_size}
    direct = position_statistics(trainer.student, trainer.teacher, batch, **kwargs)
    cache = trainer.teacher_hidden_states(batch, 2)
    cached = position_statistics(
        trainer.student, trainer.teacher, batch, teacher_hidden=cache, **kwargs
    )
    for key, value in direct.items():
        assert torch.allclose(cached[key].double(), value.double(), atol=1e-6), key
    reference = trainer.accumulate_gradients(batch)
    reused = trainer.accumulate_gradients(batch, teacher_hidden=cache)
    assert reused["loss"] == pytest.approx(reference["loss"], rel=1e-6)
    assert reused["mass_mean"] == pytest.approx(reference["mass_mean"], rel=1e-6)
    with pytest.raises(ValueError):
        trainer.accumulate_gradients(batch, teacher_hidden=cache[:1])
    with pytest.raises(ValueError):  # a different number of microbatches
        position_statistics(
            trainer.student, trainer.teacher, batch, teacher_hidden=cache[:1], **kwargs
        )
    with pytest.raises(ValueError):  # the same number, but other microbatch boundaries
        position_statistics(trainer.student, trainer.teacher, batch, teacher_hidden=cache[:1])
    other = make_trainer(make_lm, codec, microbatch_size=2)
    with pytest.raises(ValueError):
        other.accumulate_gradients(batch, teacher_hidden=[cache[0], cache[0], cache[1]])


def test_matched_updates_share_checkpoint_rollouts_teacher_and_positions(make_lm, codec):
    trainer = make_trainer(make_lm, codec)
    batch = trainer.collect_rollouts(0)
    state_before = copy.deepcopy(trainer.student.model.state_dict())
    teacher_passes = []
    hidden_states = trainer.teacher.hidden_states

    def counting(input_ids, attention_mask):
        teacher_passes.append(1)
        return hidden_states(input_ids, attention_mask)

    trainer.teacher.hidden_states = counting
    measurements = []

    def measure() -> dict:
        probe = torch.stack([p.detach().double().sum() for p in trainer.student.parameters()])
        measurements.append(probe)
        return {"probe": probe}

    objectives = {
        label: build_objective(ObjectiveConfig(name)) for label, name in E17_BRANCHES.items()
    }
    result = matched_branch_updates(trainer, batch, objectives, measure)
    assert list(result["branches"]) == list(E17_BRANCHES)
    # The teacher is run once per microbatch and view for all five branches together.
    microbatches = math.ceil(len(batch.responses) / trainer.cfg.microbatch_size)
    assert len(teacher_passes) == 2 * microbatches
    # Same number of update positions, same starting point, one step each.
    assert {b["num_positions"] for b in result["branches"].values()} == {batch.num_positions}
    assert all(b["status"] == "ok" and b["stepped"] for b in result["branches"].values())
    assert len(measurements) == 1 + len(E17_BRANCHES)
    assert torch.equal(result["before_score"]["probe"], measurements[0])
    for branch in result["branches"].values():
        assert not torch.equal(branch["after_score"]["probe"], measurements[0])  # updated
        assert branch["utility"].shape == (batch.num_positions,)
    for name, value in trainer.student.model.state_dict().items():
        assert torch.equal(value, state_before[name])  # restored afterwards
    assert trainer.optimizer.state_dict()["state"] == {}


def test_matched_updates_with_a_deterministic_teacher(codec, make_lm):
    """Cached teacher states also serve a teacher without parameters (stub)."""
    teacher = PrefixSumLM(len(codec.tokenizer))
    trainer = make_trainer(make_lm, codec, teacher=teacher)
    batch = trainer.collect_rollouts(0)
    objectives = {name: build_objective(ObjectiveConfig(name)) for name in ("eopd_full", "vdopd")}
    result = matched_branch_updates(trainer, batch, objectives)
    assert result["before_score"] is None
    assert all(branch["after_score"] is None for branch in result["branches"].values())
    assert 0.0 < result["pre_update"]["mass"] <= 1.0


def test_e17_script_columns_match_the_reported_table():
    text = (REPOSITORY / "scripts" / "local_update_diagnostic.py").read_text()
    for label in E17_BRANCHES:
        assert f'"{label}"' in text
    path = REPOSITORY / "results/supplementary/06_agreement_analysis/E17_local_update_utility.csv"
    if not path.is_file():
        pytest.skip("the reported table is not present")
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for column in reader.fieldnames:
        assert f'"{column}"' in text
    assert {row["update"] for row in rows} == set(E17_BRANCHES)
    assert {row["stratum"] for row in rows} == {label for label, _ in Z_STRATA}


def test_analysis_state_is_a_plain_record():
    state = AnalysisState(0, (1, 2), (1, 9, 2), "layout", (3,), 4)
    assert state.original_ids + state.prefix == (1, 2, 3) and state.next_token == 4
