"""Supplementary E07 (target x weight factorial), E08 (K teacher views) and E18 (two tokens)."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import make_trainer, random_log_probs
from vdopd import theory as th
from vdopd.config import load_config
from vdopd.consensus import (
    arithmetic_mean,
    geometric_consensus,
    normalized_consensus,
)
from vdopd.objectives import (
    ConsensusObjective,
    ObjectiveConfig,
    PositionBlock,
    assign_weights,
    build_objective,
)
from vdopd.scoring import pack_prompt_and_response, response_hidden_states, score_responses
from vdopd.seeding import derive_seed
from vdopd.views import ORIGINAL_VIEW, TRAIN_WRAPPERS, apply_wrapper, select_views

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIGS = REPOSITORY / "configs" / "qwen3_1p7b"
D = 7
MESSAGES = [{"role": "user", "content": "What is 2+3? Answer in \\boxed{}."}]

# Table 4 / supplementary E07: (config file, objective name, target, weight).
FACTORIAL = [
    ("ablation_arithmetic", "arithmetic", "arithmetic", "unit"),
    ("ablation_weighted_arithmetic", "weighted_arithmetic", "arithmetic", "mass"),
    ("ablation_geometric", "geometric", "geometric", "unit"),
    ("ablation_shuffled_weight", "shuffled_weight", "geometric", "shuffled"),
    ("ablation_batch_mean_weight", "batch_mean_weight", "geometric", "batch_mean"),
    ("vdopd", "vdopd", "geometric", "mass"),
]


def views(count: int, n: int = 5, seed: int = 0) -> list[torch.Tensor]:
    return [random_log_probs((n, D), seed + m) for m in range(count)]


# ------------------------------------------------------------------ E08: K views
def test_two_views_are_computed_exactly_as_before():
    """K = 2 keeps the operations of Equations 4-5: c = exp{(log p1 + log p2) / 2}."""
    log_p1, log_p2 = views(2, n=64, seed=10)
    c, mass = geometric_consensus(log_p1, log_p2)
    reference = (0.5 * (log_p1 + log_p2)).exp()
    assert torch.equal(c, reference) and torch.equal(mass, reference.sum(-1))
    assert torch.equal(arithmetic_mean(log_p1, log_p2), 0.5 * (log_p1.exp() + log_p2.exp()))
    r, _ = normalized_consensus(log_p1, log_p2)
    assert torch.allclose(r.sum(-1), torch.ones(64), atol=1e-5)


def test_k_view_consensus_is_the_geometric_mean_of_all_views():
    """c_K(a) = (prod_m p_m(a))^(1/K) and Z_K = sum_a c_K(a) (E08)."""
    log_views = views(4, n=6, seed=20)
    c, mass = geometric_consensus(*log_views)
    probabilities = np.stack([v.double().exp().numpy() for v in log_views])
    reference = np.prod(probabilities, axis=0) ** 0.25
    assert np.allclose(c.numpy(), reference, atol=1e-6)
    assert np.allclose(mass.numpy(), reference.sum(-1), atol=1e-6)
    assert c.dtype == torch.float32 and not c.requires_grad
    # Hand example with K = 4: (0.5 * 0.5 * 0.125 * 0.5)^(1/4) = 0.3536 on the first token.
    hand = [torch.tensor(p).log() for p in ([0.5, 0.5], [0.5, 0.5], [0.125, 0.875], [0.5, 0.5])]
    c, mass = geometric_consensus(*hand)
    expected = [(0.5**3 * 0.125) ** 0.25, (0.5**3 * 0.875) ** 0.25]
    assert c.tolist() == pytest.approx(expected, abs=1e-6)
    assert mass.item() == pytest.approx(sum(expected), abs=1e-6)
    # The NumPy reference implementation agrees and three views work as well.
    assert th.consensus_mass(*probabilities[:, 0]) == pytest.approx(reference[0].sum(), abs=1e-12)
    c3, _ = geometric_consensus(*log_views[:3])
    assert np.allclose(c3.numpy(), np.prod(probabilities[:3], axis=0) ** (1 / 3), atol=1e-6)
    assert torch.allclose(
        arithmetic_mean(*log_views), torch.as_tensor(probabilities.mean(0)).float(), atol=1e-6
    )


def test_k_view_mass_limits_and_bounds():
    log_p = random_log_probs((5, D), 30)
    c, mass = geometric_consensus(log_p, log_p, log_p, log_p)
    assert torch.allclose(c, log_p.exp(), atol=1e-6)  # identical views keep the teacher
    assert torch.allclose(mass, torch.ones(5), atol=1e-5)
    # Repeating a pair of views leaves the pooled target unchanged.
    log_p1, log_p2 = views(2, seed=31)
    pair, _ = geometric_consensus(log_p1, log_p2)
    assert torch.allclose(geometric_consensus(log_p1, log_p2, log_p1, log_p2)[0], pair, atol=1e-6)
    for seed in range(5):  # 0 <= Z_K <= 1 (generalised Hoelder inequality)
        mass = geometric_consensus(*views(4, n=100, seed=40 + 10 * seed))[1]
        assert (mass >= 0).all() and (mass <= 1 + 1e-6).all()
    # One view that excludes a token removes it from the joint target; disjoint supports
    # of any two views give zero mass without NaN.
    excluded = [torch.tensor(p).log() for p in ([0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.5, 0.5])]
    assert geometric_consensus(*excluded)[0][1].item() == 0.0
    disjoint = [torch.tensor(p).log() for p in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, 0.5])]
    c, mass = geometric_consensus(*disjoint)
    assert torch.equal(c, torch.zeros(2)) and mass.item() == 0.0
    r, _ = normalized_consensus(*disjoint)
    assert torch.equal(r, torch.zeros(2))


def test_products_are_formed_in_log_space():
    """A product of four small probabilities underflows float32; the log-space mean does not."""
    log_p = torch.tensor([-30.0, math.log1p(-math.exp(-30.0))])
    c, _ = geometric_consensus(log_p, log_p, log_p, log_p)
    assert c[0].item() == pytest.approx(math.exp(-30.0), rel=1e-5)
    naive = (log_p.exp() ** 4) ** 0.25  # exp(-120) is below the float32 range
    assert naive[0].item() == 0.0
    # The float64 reference keeps tiny masses as well (a plain product would return 0).
    tiny = np.array([1e-200, 1.0 - 1e-200])
    assert th.geometric_pool(tiny, tiny)[0][0] == pytest.approx(1e-200, rel=1e-9)
    assert np.sqrt(tiny * tiny)[0] == 0.0


def test_k_view_objective_loss_gradient_and_view_count():
    log_views = views(4, seed=50)
    logits = torch.randn(5, D, generator=torch.Generator().manual_seed(51)).requires_grad_(True)
    tokens = torch.zeros(5, dtype=torch.long)
    objective = build_objective(ObjectiveConfig("vdopd", num_views=4))
    assert objective.num_views == 4 and objective.uses_consensus_mass
    out = objective(PositionBlock(logits, tuple(log_views), tokens))
    c, mass = geometric_consensus(*log_views)
    log_q = F.log_softmax(logits, -1)
    assert torch.allclose(out.loss, -(c * log_q).sum(-1), atol=1e-6)
    assert torch.equal(out.weight, mass) and torch.equal(out.mass, mass)
    (gradient,) = torch.autograd.grad(out.loss.sum(), logits)
    assert torch.allclose(gradient, mass.unsqueeze(-1) * log_q.exp() - c, atol=1e-6)
    assert (gradient.abs().sum(-1) <= 2 * mass + 1e-6).all()
    # The other targets and weights extend to K views in the same way.
    unit = ConsensusObjective("arithmetic", "unit", num_views=4)(
        PositionBlock(logits, tuple(log_views), tokens)
    )
    assert torch.allclose(unit.loss, -(arithmetic_mean(*log_views) * log_q).sum(-1), atol=1e-6)
    with pytest.raises(ValueError):  # four views requested, two supplied
        objective(PositionBlock(logits, tuple(log_views[:2]), tokens))
    with pytest.raises(ValueError):
        ConsensusObjective(num_views=1)
    # K is a property of the consensus family; it cannot be set for other objectives.
    assert build_objective(ObjectiveConfig("vdopd")).num_views == 2
    assert build_objective(ObjectiveConfig("eopd_2v", num_views=2)).num_views == 2
    with pytest.raises(ValueError):
        build_objective(ObjectiveConfig("eopd_full", num_views=2))


def legacy_alternative(messages, codec, rng):
    """View selection as implemented before K views existed (one uniform draw)."""
    valid = [(w, codec.encode(apply_wrapper(messages, w))) for w in TRAIN_WRAPPERS]
    return valid[int(rng.integers(len(valid)))]


def test_view_selection_for_two_views_is_unchanged(codec):
    kwargs = {"max_response_tokens": 8, "teacher_context": None, "student_context": None}
    new_rng, old_rng = np.random.default_rng(123), np.random.default_rng(123)
    for _ in range(200):
        pair = select_views(MESSAGES, codec, rng=new_rng, **kwargs)
        wrapper, ids = legacy_alternative(MESSAGES, codec, old_rng)
        assert pair.wrapper == wrapper and list(pair.view2_ids) == ids
        assert pair.extra_view_ids == () and pair.extra_wrappers == ()
        assert pair.student_ids is None and pair.student_prompt_ids == pair.original_ids
        assert pair.teacher_view_ids == (pair.original_ids, pair.view2_ids)
    assert new_rng.integers(1 << 30) == old_rng.integers(1 << 30)  # same stream position
    # A single-view objective still samples its (unused) alternative from the same stream.
    one = select_views(MESSAGES, codec, rng=np.random.default_rng(5), num_views=1, **kwargs)
    two = select_views(MESSAGES, codec, rng=np.random.default_rng(5), num_views=2, **kwargs)
    assert one == two


def test_four_views_use_the_original_prompt_and_all_three_wrappers(codec):
    kwargs = {"max_response_tokens": 8, "teacher_context": None, "student_context": None}
    pair = select_views(MESSAGES, codec, rng=np.random.default_rng(0), num_views=4, **kwargs)
    assert pair.wrappers == TRAIN_WRAPPERS  # all three, in the order of Table 5
    assert len(pair.teacher_view_ids) == 4 and list(pair.teacher_view_ids[0]) == codec.encode(
        MESSAGES
    )
    for wrapper, ids in zip(TRAIN_WRAPPERS, pair.teacher_view_ids[1:], strict=True):
        assert list(ids) == codec.encode(apply_wrapper(MESSAGES, wrapper))
    # K = 3 samples two different wrappers without replacement, kept in pool order.
    seen = set()
    rng = np.random.default_rng(1)
    for _ in range(60):
        pair = select_views(MESSAGES, codec, rng=rng, num_views=3, **kwargs)
        names = [w.name for w in pair.wrappers]
        assert len(set(names)) == 2
        assert names == [w.name for w in TRAIN_WRAPPERS if w.name in names]
        seen.add(tuple(names))
    assert len(seen) == 3
    # Views that do not fit the teacher context fall back to the identity transformation.
    length = len(codec.encode(MESSAGES))
    pair = select_views(
        MESSAGES,
        codec,
        rng=np.random.default_rng(2),
        num_views=4,
        max_response_tokens=8,
        teacher_context=length + 8 + 10,  # only "Problem:" + two newlines fits
        student_context=None,
    )
    assert [w.name for w in pair.wrappers] == ["layout", ORIGINAL_VIEW.name, ORIGINAL_VIEW.name]
    assert pair.teacher_view_ids[2] == pair.teacher_view_ids[3] == pair.original_ids


def direct_k_view_loss(trainer, batch) -> torch.Tensor:
    """-(1/N_k) sum c_K^T log q with ordinary autograd (reference implementation)."""
    count = trainer.objective.num_views
    log_views = [
        score_responses(
            trainer.teacher, [pair.teacher_view_ids[m] for pair in batch.pairs], batch.responses
        )
        for m in range(count)
    ]
    original = [pair.original_ids for pair in batch.pairs]
    packed = pack_prompt_and_response(original, batch.responses, trainer.student.pad_token_id)
    logits = trainer.student.project(response_hidden_states(trainer.student, packed)).float()
    c, _ = geometric_consensus(*log_views)
    return -(c * F.log_softmax(logits, -1)).sum() / batch.num_positions


def test_trainer_scores_four_teacher_views(make_lm, codec):
    trainer = make_trainer(make_lm, codec, num_views=4)
    batch = trainer.collect_rollouts(0)
    assert all(len(pair.teacher_view_ids) == 4 for pair in batch.pairs)
    assert all(pair.wrappers == TRAIN_WRAPPERS for pair in batch.pairs)
    stats = trainer.accumulate_gradients(batch)
    blockwise = [p.grad.clone() for p in trainer.parameters]
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = direct_k_view_loss(trainer, batch)
    direct = torch.autograd.grad(loss, trainer.parameters)
    assert stats["status"] == "ok" and stats["loss"] == pytest.approx(loss.item(), rel=1e-5)
    for a, b in zip(blockwise, direct, strict=True):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-4)
    assert 0.0 < stats["mass_mean"] <= 1.0 + 1e-6  # Z_4 of the four views
    assert stats["weight_mean"] == pytest.approx(stats["mass_mean"], rel=1e-5)
    # A rollout batch collected for two views cannot serve a four-view objective.
    two = make_trainer(make_lm, codec)
    with pytest.raises(ValueError):
        two.accumulate_gradients(two.collect_rollouts(0), trainer.objective)


def test_teacher_flops_scale_with_the_number_of_views(make_lm, codec):
    four = make_trainer(make_lm, codec, num_views=4)
    two = make_trainer(make_lm, codec)
    batch = four.collect_rollouts(0)
    four.accumulate_gradients(batch)
    params = four.teacher.num_flop_parameters()
    tokens = sum(
        len(ids) + len(response)
        for pair, response in zip(batch.pairs, batch.responses, strict=True)
        for ids in pair.teacher_view_ids
    )
    assert four.flops.teacher_scoring == pytest.approx(2.0 * params * tokens)
    two_batch = two.collect_rollouts(0)
    two.accumulate_gradients(two_batch)
    assert two_batch.responses == batch.responses  # same rollouts, same student cost
    assert four.flops.student_training == two.flops.student_training
    assert four.flops.student_generation == two.flops.student_generation
    assert four.flops.teacher_scoring > 1.9 * two.flops.teacher_scoring


def test_four_view_run_stops_at_the_flops_budget_of_the_two_view_run(make_lm, codec):
    """'K=4, same FLOPs': the existing total-FLOPs budget shortens the four-view run."""
    reference = make_trainer(make_lm, codec, iterations=6)
    reference.train()
    budget = reference.flops.total
    same_updates = make_trainer(make_lm, codec, num_views=4, iterations=6)
    same_updates.train()
    same_flops = make_trainer(make_lm, codec, num_views=4, iterations=6, max_total_flops=budget)
    same_flops.train()
    assert same_updates.iteration == reference.iteration == 6
    assert same_updates.flops.total > budget
    assert same_flops.iteration < 6
    # The run stops at the first iteration boundary at which the budget is spent.
    assert same_flops.history[-2]["flops_total"] < budget <= same_flops.flops.total
    assert same_flops.summary()["num_teacher_views"] == 4


def test_compute_and_view_matched_controls(make_lm, codec):
    """Supplementary E06 (Table 2): EOPD-2V has the teacher access and update budget of
    VD-OPD; VD-OPD-Budget is VD-OPD stopped at the total FLOPs of EOPD (full)."""
    two_view, vd = (
        make_trainer(make_lm, codec, name, iterations=5) for name in ("eopd_2v", "vdopd")
    )
    assert two_view.objective.num_views == vd.objective.num_views == 2
    assert two_view.total_iterations == vd.total_iterations == 5  # same update budget
    first, second = two_view.collect_rollouts(0), vd.collect_rollouts(0)
    # Same source questions, same sampled teacher views, same (initial) student rollouts.
    assert first.prompt_indices == second.prompt_indices and first.pairs == second.pairs
    assert first.responses == second.responses
    two_view.accumulate_gradients(first)
    vd.accumulate_gradients(second)
    assert two_view.flops.state_dict() == vd.flops.state_dict()  # two scoring passes each
    for iteration in range(1, 5):  # the view stream does not depend on the objective
        a = [pair.wrapper for pair in two_view.collect_rollouts(iteration).pairs]
        assert a == [pair.wrapper for pair in vd.collect_rollouts(iteration).pairs]

    single = make_trainer(make_lm, codec, "eopd_full", iterations=6)
    single.train()
    budget = single.flops.total
    budgeted = make_trainer(make_lm, codec, "vdopd", iterations=6, max_total_flops=budget)
    budgeted.train()
    full = make_trainer(make_lm, codec, "vdopd", iterations=6)
    full.train()
    assert single.iteration == full.iteration == 6 and full.flops.total > budget
    assert budgeted.iteration < 6  # training is shortened; two teacher views are retained
    assert budgeted.history[-2]["flops_total"] < budget <= budgeted.flops.total
    assert budgeted.flops.total - budget < budgeted.flops.total / budgeted.iteration * 1.5
    cfg = load_config(CONFIGS / "vdopd_budget.yaml")
    assert cfg.objective.name == "vdopd" and cfg.train.max_total_flops == 4.7e18
    assert build_objective(load_config(CONFIGS / "eopd_2v.yaml").objective).num_views == 2


def test_view_count_configs():
    same_updates = load_config(CONFIGS / "vdopd_k4.yaml")
    same_flops = load_config(CONFIGS / "vdopd_k4_budget.yaml")
    default = load_config(CONFIGS / "vdopd.yaml")
    for cfg in (same_updates, same_flops):
        assert cfg.objective.name == "vdopd" and cfg.objective.num_views == 4
        assert build_objective(cfg.objective).num_views == 4
        assert cfg.train.passes == default.train.passes and cfg.rollout == default.rollout
    assert same_updates.train.max_total_flops is None
    assert same_flops.train.max_total_flops == 5.8e18
    assert default.objective.num_views is None and build_objective(default.objective).num_views == 2


# ------------------------------------------------------------------ E07: factorial
@pytest.mark.parametrize(("file", "name", "target", "weight"), FACTORIAL)
def test_factorial_cells_exist_as_configs_and_objectives(file, name, target, weight):
    cfg = load_config(CONFIGS / f"{file}.yaml")
    objective = build_objective(cfg.objective)
    assert cfg.objective.name == name
    assert (objective.target, objective.weight, objective.num_views) == (target, weight, 2)


@pytest.mark.parametrize(("file", "name", "target", "weight"), FACTORIAL)
def test_factorial_logit_gradient_and_bound(file, name, target, weight):
    """l = -w s^T log q gives grad_z l = w (q - s) and ||grad_z l||_1 <= 2 w.

    For VD-OPD (s = r, w = Z) this is grad_z l = Z (q - r) and ||grad_z l||_1 <= 2 Z for
    l = -c^T log q (Equation 8, Appendix D.3).
    """
    log_p1, log_p2 = views(2, n=6, seed=60)
    logits = torch.randn(6, D, generator=torch.Generator().manual_seed(61)).requires_grad_(True)
    c, mass = geometric_consensus(log_p1, log_p2)
    r = c / mass.unsqueeze(-1)
    generator = torch.Generator().manual_seed(62)
    assigned = (
        assign_weights(mass, weight, generator) if weight in ("shuffled", "batch_mean") else None
    )
    block = PositionBlock(logits, (log_p1, log_p2), torch.zeros(6, dtype=torch.long), assigned)
    out = build_objective(ObjectiveConfig(name))(block)
    s = r if target == "geometric" else 0.5 * (log_p1.exp() + log_p2.exp())
    w = {"unit": torch.ones_like(mass), "mass": mass}.get(weight, assigned)
    q = torch.softmax(logits, -1)
    (gradient,) = torch.autograd.grad(out.loss.sum(), logits)
    assert torch.allclose(gradient, w.unsqueeze(-1) * (q - s), atol=1e-6)
    assert (gradient.abs().sum(-1) <= 2 * w + 1e-6).all()
    assert torch.allclose(out.weight, w, atol=1e-6) and torch.allclose(out.mass, mass, atol=1e-6)
    assert not out.weight.requires_grad  # the weight is a constant of the update


@pytest.mark.parametrize(("file", "name", "target", "weight"), FACTORIAL)
def test_factorial_loss_is_averaged_over_the_unweighted_token_count(
    make_lm, codec, file, name, target, weight
):
    """Every cell divides by N_k, not by the sum of the weights, so Z attenuates the loss."""
    trainer = make_trainer(make_lm, codec, name)
    batch = trainer.collect_rollouts(0)
    stats = trainer.accumulate_gradients(batch)
    original = [pair.original_ids for pair in batch.pairs]
    log_p1 = score_responses(trainer.teacher, original, batch.responses)
    log_p2 = score_responses(
        trainer.teacher, [pair.view2_ids for pair in batch.pairs], batch.responses
    )
    log_q = score_responses(trainer.student, original, batch.responses)
    c, mass = geometric_consensus(log_p1, log_p2)
    s = c / mass.unsqueeze(-1) if target == "geometric" else arithmetic_mean(log_p1, log_p2)
    if weight in ("shuffled", "batch_mean"):
        generator = torch.Generator().manual_seed(derive_seed(11, "weights", 0))
        w = assign_weights(mass, weight, generator)
    else:
        w = mass if weight == "mass" else torch.ones_like(mass)
    n_k = batch.num_positions
    expected = float(-(w * (s * log_q).sum(-1)).double().sum()) / n_k
    assert stats["num_positions"] == n_k == len(mass)
    assert stats["loss"] == pytest.approx(expected, rel=1e-5)
    assert stats["weight_mean"] == pytest.approx(float(w.double().mean()), rel=1e-5)
    if weight != "unit":  # normalising by the weights instead would give a larger loss
        assert float(w.sum()) < n_k
        assert abs(stats["loss"]) < abs(expected) * n_k / float(w.sum())


def test_shared_student_log_probabilities_are_computed_once(monkeypatch):
    """EOPD-2V evaluates both terms on one log-softmax of the student logits."""
    calls = []
    original = F.log_softmax

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "log_softmax", counting)
    log_p1, log_p2 = views(2, n=4, seed=70)
    logits = torch.randn(4, D, requires_grad=True)
    block = PositionBlock(logits, (log_p1, log_p2), torch.zeros(4, dtype=torch.long))
    build_objective(ObjectiveConfig("eopd_2v"))(block)
    assert len(calls) == 1
    assert block.student_log_probs is block.student_log_probs


# ------------------------------------------------------------------ E18: two tokens
def test_two_token_example():
    """p1 = p2 = (0.5, 0.5) gives r = (0.5, 0.5), Z = 1; opposite confident views give the
    same r with Z = 0.199 (three decimals)."""
    shared, contradictory = th.two_token_example()
    assert shared["p1"] == shared["p2"] == (0.5, 0.5)
    assert shared["r"] == pytest.approx((0.5, 0.5)) and round(shared["z"], 3) == 1.0
    assert (contradictory["p1"], contradictory["p2"]) == ((0.99, 0.01), (0.01, 0.99))
    assert contradictory["r"] == pytest.approx((0.5, 0.5))
    assert round(contradictory["z"], 3) == 0.199
    assert contradictory["z"] == pytest.approx(2 * math.sqrt(0.99 * 0.01), abs=1e-12)
    # The training code gives the same target and mass.
    for row in (shared, contradictory):
        log_p1, log_p2 = (torch.tensor(row[key], dtype=torch.float64).log() for key in ("p1", "p2"))
        r, mass = normalized_consensus(log_p1, log_p2)
        assert r.tolist() == pytest.approx([0.5, 0.5], abs=1e-6)
        assert mass.item() == pytest.approx(row["z"], abs=1e-6)


def test_two_token_example_script(tmp_path, capsys):
    path = REPOSITORY / "examples" / "two_token_example.py"
    spec = importlib.util.spec_from_file_location("two_token_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.COLUMNS == ("case", "p1", "p2", "r", "z")
    assert [tuple(row.values()) for row in module.table()] == [
        ("Shared uncertainty", "(0.5, 0.5)", "(0.5, 0.5)", "(0.5, 0.5)", "1"),
        ("Contradictory confidence", "(0.99, 0.01)", "(0.01, 0.99)", "(0.5, 0.5)", "0.199"),
    ]
    written = tmp_path / "E18_two_token_example.csv"
    module.main(["--csv", str(written)])
    assert "Contradictory confidence" in capsys.readouterr().out
    reported = REPOSITORY / "results/supplementary/06_agreement_analysis" / written.name
    if reported.is_file():  # the closed-form example reproduces the reported table exactly
        tables = []
        for path in (written, reported):
            with open(path, encoding="utf-8", newline="") as handle:
                tables.append(list(csv.reader(handle)))
        assert tables[0] == tables[1]
