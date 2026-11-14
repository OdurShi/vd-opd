"""Equation 7, 8, 13, 25 and Table 4: objectives, limiting cases and gradient checks."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import random_log_probs
from vdopd.consensus import entropy, geometric_consensus, kl_divergence
from vdopd.objectives import (
    EOPD,
    UNSPECIFIED_BASELINES,
    BlockLoss,
    ConsensusObjective,
    ForwardKL,
    GeneralizedJSD,
    ObjectiveConfig,
    PositionBlock,
    ReverseKL,
    SampledTokenOPD,
    assign_weights,
    build_objective,
    register_objective,
    truncated_forward_kl,
)

D = 7


def block(logits, lp1, lp2=None, tokens=None, weights=None) -> PositionBlock:
    views = (lp1,) if lp2 is None else (lp1, lp2)
    if tokens is None:
        tokens = torch.zeros(logits.shape[0], dtype=torch.long)
    return PositionBlock(logits, views, tokens, weights)


def student_logits(n: int = 4, seed: int = 0) -> torch.Tensor:
    return torch.randn(n, D, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)


def test_vdopd_loss_matches_hand_computation():
    """l = -c^T log q with c = (0.48, 0.48) and q = (0.8, 0.2)."""
    lp1 = torch.tensor([[0.64, 0.36]]).log()
    lp2 = torch.tensor([[0.36, 0.64]]).log()
    logits = torch.tensor([[math.log(0.8), math.log(0.2)]], requires_grad=True)
    out = ConsensusObjective()(block(logits, lp1, lp2))
    expected = -0.48 * (math.log(0.8) + math.log(0.2))
    assert out.loss.item() == pytest.approx(expected, abs=1e-6)
    assert out.weight.item() == pytest.approx(0.96, abs=1e-6)


def test_loss_decomposition_and_logit_gradient():
    """l = Z [KL(r || q) + H(r)] and grad_z l = Z q - c (Equation 8)."""
    lp1, lp2 = random_log_probs((4, D), 1), random_log_probs((4, D), 2)
    logits = student_logits()
    out = ConsensusObjective()(block(logits, lp1, lp2))
    c, mass = geometric_consensus(lp1, lp2)
    r = c / mass.unsqueeze(-1)
    log_q = F.log_softmax(logits, -1)
    decomposition = mass * (kl_divergence(r.log(), log_q) + entropy(r.log()))
    assert torch.allclose(out.loss, decomposition, atol=1e-5)
    (gradient,) = torch.autograd.grad(out.loss.sum(), logits)
    assert torch.allclose(gradient, mass.unsqueeze(-1) * log_q.exp() - c, atol=1e-6)
    # ||grad_z l||_1 <= 2 Z (Equation 25)
    assert (gradient.abs().sum(-1) <= 2 * mass + 1e-6).all()


def test_logit_hessian_is_mass_times_categorical_covariance():
    """Hessian = Z [diag(q) - q q^T] (Equation 25)."""
    lp1, lp2 = random_log_probs((1, D), 3), random_log_probs((1, D), 4)
    z0 = torch.randn(D, generator=torch.Generator().manual_seed(5))

    def loss(z: torch.Tensor) -> torch.Tensor:
        return ConsensusObjective()(block(z.unsqueeze(0), lp1, lp2)).loss.sum()

    hessian = torch.autograd.functional.hessian(loss, z0)
    q = torch.softmax(z0, -1)
    mass = geometric_consensus(lp1, lp2)[1]
    assert torch.allclose(hessian, mass * (torch.diag(q) - torch.outer(q, q)), atol=1e-5)
    assert torch.linalg.eigvalsh(hessian).min() > -1e-6  # positive semidefinite


def test_identical_views_reduce_to_single_view_forward_kl_with_full_weight():
    lp = random_log_probs((4, D), 6)
    logits = student_logits(seed=7)
    vd = ConsensusObjective()(block(logits, lp, lp))
    fkl = ForwardKL()(block(logits, lp))
    assert torch.allclose(vd.weight, torch.ones(4), atol=1e-5)
    grad_vd = torch.autograd.grad(vd.loss.sum(), logits, retain_graph=True)[0]
    grad_fkl = torch.autograd.grad(fkl.loss.sum(), logits)[0]
    assert torch.allclose(grad_vd, grad_fkl, atol=1e-6)
    # The two losses differ only by the student-independent entropy H(p).
    assert torch.allclose(vd.loss - fkl.loss, entropy(lp), atol=1e-5)


def test_disjoint_supports_give_zero_loss_and_gradient():
    lp1 = torch.tensor([[0.5, 0.5, 0.0, 0.0]]).log()
    lp2 = torch.tensor([[0.0, 0.0, 0.3, 0.7]]).log()
    logits = torch.randn(1, 4, requires_grad=True)
    out = ConsensusObjective()(block(logits, lp1, lp2))
    out.loss.sum().backward()
    assert out.loss.item() == 0.0 and out.weight.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_disagreement_attenuates_loss_and_gradient_by_the_mass_ratio():
    """Appendix D.1: the conflicting case contributes 0.199 times the agreeing case."""
    logits = torch.tensor([[0.3, -0.2]], requires_grad=True)
    half = torch.tensor([[0.5, 0.5]]).log()
    agree = ConsensusObjective()(block(logits, half, half))
    conflict = ConsensusObjective()(
        block(logits, torch.tensor([[0.99, 0.01]]).log(), torch.tensor([[0.01, 0.99]]).log())
    )
    ratio = 2 * math.sqrt(0.0099)
    assert conflict.loss.item() == pytest.approx(ratio * agree.loss.item(), rel=1e-5)
    g_agree = torch.autograd.grad(agree.loss.sum(), logits, retain_graph=True)[0]
    g_conflict = torch.autograd.grad(conflict.loss.sum(), logits)[0]
    assert torch.allclose(g_conflict, ratio * g_agree, atol=1e-6)


def test_teacher_receives_no_gradient_and_weight_is_not_differentiated():
    lp1 = random_log_probs((3, D), 8).requires_grad_(True)
    lp2 = random_log_probs((3, D), 9).requires_grad_(True)
    logits = student_logits(3, seed=10)
    for objective in (
        ConsensusObjective(),
        ConsensusObjective("arithmetic", "mass"),
        ConsensusObjective("geometric", "unit"),
        build_objective(ObjectiveConfig("eopd_2v")),
        ForwardKL(),
        ReverseKL(),
        GeneralizedJSD(),
        SampledTokenOPD(),
    ):
        out = objective(block(logits, lp1, lp2))
        assert not out.weight.requires_grad
        gradients = torch.autograd.grad(
            out.loss.sum(), [logits, lp1, lp2], allow_unused=True, retain_graph=True
        )
        assert gradients[0] is not None
        assert gradients[1] is None and gradients[2] is None


@pytest.mark.parametrize(
    ("name", "target", "weight"),
    [
        ("arithmetic", "arithmetic", "unit"),
        ("weighted_arithmetic", "arithmetic", "mass"),
        ("geometric", "geometric", "unit"),
        ("vdopd", "geometric", "mass"),
    ],
)
def test_table4_target_weight_combinations(name, target, weight):
    lp1, lp2 = random_log_probs((5, D), 11), random_log_probs((5, D), 12)
    logits = student_logits(5, seed=13)
    objective = build_objective(ObjectiveConfig(name))
    assert (objective.target, objective.weight) == (target, weight)
    out = objective(block(logits, lp1, lp2))
    c, mass = geometric_consensus(lp1, lp2)
    s = c / mass.unsqueeze(-1) if target == "geometric" else 0.5 * (lp1.exp() + lp2.exp())
    w = mass if weight == "mass" else torch.ones_like(mass)
    expected = -w * (s * F.log_softmax(logits, -1)).sum(-1)
    assert torch.allclose(out.loss, expected, atol=1e-5)
    assert torch.allclose(out.weight, w, atol=1e-6)
    assert torch.allclose(s.sum(-1), torch.ones(5), atol=1e-5)  # normalised target


def test_weight_assignment_controls():
    """Shuffling preserves the weight multiset; batch-mean preserves the total weight."""
    mass = torch.rand(50, generator=torch.Generator().manual_seed(14))
    shuffled = assign_weights(mass, "shuffled", torch.Generator().manual_seed(0))
    assert torch.equal(shuffled.sort().values, mass.sort().values)
    assert not torch.equal(shuffled, mass)
    again = assign_weights(mass, "shuffled", torch.Generator().manual_seed(0))
    assert torch.equal(shuffled, again)  # seeded permutation
    mean = assign_weights(mass, "batch_mean")
    assert torch.allclose(mean, torch.full((50,), mass.mean().item()))
    assert mean.sum().item() == pytest.approx(mass.sum().item(), rel=1e-5)


def test_weight_controls_keep_each_positions_geometric_target():
    lp1, lp2 = random_log_probs((6, D), 15), random_log_probs((6, D), 16)
    logits = student_logits(6, seed=17)
    mass = geometric_consensus(lp1, lp2)[1]
    vd = ConsensusObjective()(block(logits, lp1, lp2))
    # Assigning each position its own mass reproduces VD-OPD exactly.
    same = ConsensusObjective("geometric", "shuffled")(block(logits, lp1, lp2, weights=mass))
    assert torch.allclose(same.loss, vd.loss, atol=1e-6)
    unit = ConsensusObjective("geometric", "unit")(block(logits, lp1, lp2))
    mean = ConsensusObjective("geometric", "batch_mean")(
        block(logits, lp1, lp2, weights=assign_weights(mass, "batch_mean"))
    )
    assert torch.allclose(mean.loss, mass.mean() * unit.loss, atol=1e-6)
    with pytest.raises(ValueError):
        ConsensusObjective("geometric", "shuffled")(block(logits, lp1, lp2))


def test_gkd_divergences_match_definitions_and_vanish_at_the_teacher():
    lp = random_log_probs((4, D), 18)
    logits = student_logits(seed=19)
    log_q = F.log_softmax(logits, -1)
    p, q = lp.exp(), log_q.exp()
    assert torch.allclose(
        ForwardKL()(block(logits, lp)).loss, (p * (lp - log_q)).sum(-1), atol=1e-6
    )
    assert torch.allclose(
        ReverseKL()(block(logits, lp)).loss, (q * (log_q - lp)).sum(-1), atol=1e-6
    )
    beta = 0.9
    m = beta * p + (1 - beta) * q
    jsd = beta * (p * (p / m).log()).sum(-1) + (1 - beta) * (q * (q / m).log()).sum(-1)
    assert torch.allclose(GeneralizedJSD(beta)(block(logits, lp)).loss, jsd, atol=1e-6)
    at_teacher = lp.clone().requires_grad_(True)
    for objective in (ForwardKL(), ReverseKL(), GeneralizedJSD()):
        out = objective(block(at_teacher, lp))
        assert torch.allclose(out.loss, torch.zeros(4), atol=1e-6)
        (gradient,) = torch.autograd.grad(out.loss.sum(), at_teacher)
        assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-6)


def test_sampled_token_opd_has_the_reverse_kl_gradient_in_expectation():
    lp = random_log_probs((1, D), 20)
    logits = student_logits(1, seed=21)
    q = torch.softmax(logits, -1).detach()
    expected = torch.zeros(1, D)
    for token in range(D):  # exact expectation over y ~ q
        out = SampledTokenOPD()(block(logits, lp, tokens=torch.tensor([token])))
        (gradient,) = torch.autograd.grad(out.loss.sum(), logits)
        expected += q[0, token] * gradient
    (reverse_kl,) = torch.autograd.grad(ReverseKL()(block(logits, lp)).loss.sum(), logits)
    assert torch.allclose(expected, reverse_kl, atol=1e-6)


def test_eopd_gate_threshold_and_components():
    """EOPD = sampled-token OPD + 1[H(p) > 0.8] * FKL (Appendix B.3.1, B.5.3)."""
    low = torch.tensor([[0.97, 0.01, 0.01, 0.01]]).log()  # H = 0.168 nats
    high = torch.tensor([[0.25, 0.25, 0.25, 0.25]]).log()  # H = 1.386 nats
    assert entropy(low).item() < 0.8 < entropy(high).item()
    logits = torch.randn(1, 4, generator=torch.Generator().manual_seed(22)).requires_grad_(True)
    tokens = torch.tensor([2])
    for lp, gate in ((low, 0.0), (high, 1.0)):
        out = EOPD()(block(logits, lp, tokens=tokens))
        opd = SampledTokenOPD()(block(logits, lp, tokens=tokens)).loss
        fkl = ForwardKL()(block(logits, lp)).loss
        assert out.weight.item() == gate
        assert torch.allclose(out.loss, opd + gate * fkl, atol=1e-6)
    weighted = EOPD(fkl_weight=0.5)(block(logits, high, tokens=tokens))
    assert torch.allclose(
        weighted.loss,
        SampledTokenOPD()(block(logits, high, tokens=tokens)).loss
        + 0.5 * ForwardKL()(block(logits, high)).loss,
        atol=1e-6,
    )


def test_truncated_forward_kl():
    lp, lq = random_log_probs((3, D), 23), random_log_probs((3, D), 24)
    assert torch.allclose(truncated_forward_kl(lp, lq, None), kl_divergence(lp, lq))
    assert torch.allclose(truncated_forward_kl(lp, lq, D), kl_divergence(lp, lq))
    top, index = lp.topk(3, dim=-1)
    manual = (top.exp() * (top - lq.gather(-1, index))).sum(-1)
    assert torch.allclose(truncated_forward_kl(lp, lq, 3), manual, atol=1e-6)
    renormalised = truncated_forward_kl(lp, lq, 3, renormalize=True)
    assert (renormalised >= -1e-6).all()  # a proper KL between renormalised distributions
    assert build_objective(ObjectiveConfig("eopd_top16")).top_k == 16
    assert build_objective(ObjectiveConfig("eopd_full")).top_k is None


def test_eopd_2v_averages_the_single_view_objective():
    """L_EOPD-2V = 0.5 [L_EOPD(p1, q) + L_EOPD(p2, q)] with per-view entropy gates (Eq. 13)."""
    lp1 = torch.tensor([[0.97, 0.01, 0.01, 0.01], [0.25, 0.25, 0.25, 0.25]]).log()
    lp2 = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]]).log()
    logits = torch.randn(2, 4, generator=torch.Generator().manual_seed(25)).requires_grad_(True)
    tokens = torch.tensor([1, 3])
    objective = build_objective(ObjectiveConfig("eopd_2v"))
    assert objective.num_views == 2
    out = objective(block(logits, lp1, lp2, tokens=tokens))
    first = EOPD(view=0)(block(logits, lp1, lp2, tokens=tokens))
    second = EOPD(view=1)(block(logits, lp1, lp2, tokens=tokens))
    assert torch.allclose(out.loss, 0.5 * (first.loss + second.loss), atol=1e-6)
    # Position 0: only the second view is above the entropy threshold.
    assert first.weight.tolist() == [0.0, 1.0] and second.weight.tolist() == [1.0, 1.0]
    assert out.weight.tolist() == [0.5, 1.0]


def test_view_count_and_mass_flags():
    assert build_objective(ObjectiveConfig("vdopd")).num_views == 2
    assert build_objective(ObjectiveConfig("eopd_full")).num_views == 1
    assert build_objective(ObjectiveConfig("gkd_fkl")).num_views == 1
    assert build_objective(ObjectiveConfig("vdopd")).uses_consensus_mass
    assert not build_objective(ObjectiveConfig("geometric")).uses_consensus_mass
    assert build_objective(ObjectiveConfig("shuffled_weight")).weight_assignment == "shuffled"
    assert build_objective(ObjectiveConfig("batch_mean_weight")).weight_assignment == "batch_mean"
    assert build_objective(ObjectiveConfig("vdopd")).weight_assignment is None
    with pytest.raises(ValueError):
        ForwardKL()(PositionBlock(torch.zeros(1, 2), (), torch.zeros(1, dtype=torch.long)))


def test_unspecified_baselines_are_hooks_and_custom_objectives_can_be_registered():
    for name in UNSPECIFIED_BASELINES:
        with pytest.raises(NotImplementedError):
            build_objective(ObjectiveConfig(name))
    with pytest.raises(ValueError):
        build_objective(ObjectiveConfig("no_such_objective"))

    class Constant:
        num_views = 1
        weight_assignment = None
        uses_consensus_mass = False

        def __call__(self, block: PositionBlock) -> BlockLoss:
            value = block.student_logits.sum(-1) * 0.0 + 1.0
            return BlockLoss(loss=value, weight=torch.ones_like(value))

    register_objective("constant_for_test", lambda cfg: Constant())
    out = build_objective(ObjectiveConfig("constant_for_test"))(
        block(student_logits(), random_log_probs((4, D), 26))
    )
    assert out.loss.tolist() == [1.0] * 4
