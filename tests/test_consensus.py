"""Equations 1 and 3-6, Appendix D.1 and D.2: geometry of the consensus target."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import random_log_probs
from vdopd.consensus import (
    arithmetic_mean,
    bhattacharyya_affinity,
    entropy,
    generalized_kl,
    geometric_consensus,
    js_divergence,
    kl_divergence,
    normalized_consensus,
    squared_hellinger,
    view_decomposition,
)


def log(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).log().float()


def test_hand_computed_consensus_and_mass():
    # sqrt(0.64 * 0.36) = 0.48 on both tokens, so c = (0.48, 0.48) and Z = 0.96.
    c, mass = geometric_consensus(log([0.64, 0.36]), log([0.36, 0.64]))
    assert torch.allclose(c, torch.tensor([0.48, 0.48]), atol=1e-6)
    assert mass.item() == pytest.approx(0.96, abs=1e-6)
    # sqrt(0.25 * 1.0) = 0.5 and sqrt(0.75 * 0.0) = 0.
    c, mass = geometric_consensus(log([0.25, 0.75]), log([1.0, 0.0]))
    assert torch.allclose(c, torch.tensor([0.5, 0.0]), atol=1e-6)
    assert mass.item() == pytest.approx(0.5, abs=1e-6)


def test_two_token_illustration_of_appendix_d1():
    """Both cases share the arithmetic mean and normalised target; masses are 1 and 0.199."""
    agree = (log([0.5, 0.5]), log([0.5, 0.5]))
    conflict = (log([0.99, 0.01]), log([0.01, 0.99]))
    for views in (agree, conflict):
        assert torch.allclose(arithmetic_mean(*views), torch.tensor([0.5, 0.5]), atol=1e-6)
        assert torch.allclose(normalized_consensus(*views)[0], torch.tensor([0.5, 0.5]), atol=1e-6)
    assert bhattacharyya_affinity(*agree).item() == pytest.approx(1.0, abs=1e-6)
    assert bhattacharyya_affinity(*conflict).item() == pytest.approx(
        2 * math.sqrt(0.0099), abs=1e-6
    )
    assert 2 * math.sqrt(0.0099) == pytest.approx(0.199, abs=5e-4)


def test_identical_views_preserve_the_teacher_distribution():
    log_p = random_log_probs((5, 11), seed=1)
    c, mass = geometric_consensus(log_p, log_p)
    assert torch.allclose(c, log_p.exp(), atol=1e-6)
    assert torch.allclose(mass, torch.ones(5), atol=1e-5)


def test_disjoint_supports_give_zero_mass_without_nan():
    log_p1 = log([0.5, 0.5, 0.0, 0.0])
    log_p2 = log([0.0, 0.0, 0.3, 0.7])
    c, mass = geometric_consensus(log_p1, log_p2)
    assert torch.equal(c, torch.zeros(4))
    assert mass.item() == 0.0
    assert js_divergence(log_p1, log_p2).item() == pytest.approx(math.log(2), abs=1e-6)
    assert squared_hellinger(log_p1, log_p2).item() == pytest.approx(1.0, abs=1e-6)


def test_square_root_decomposition_identities():
    """c = p_bar - u_minus^2, Z = 1 - ||u_minus||^2, <u_plus, u_minus> = 0 (Eq. 3-5)."""
    log_p1, log_p2 = random_log_probs((64, 17), seed=2), random_log_probs((64, 17), seed=3)
    u_plus, u_minus = view_decomposition(log_p1, log_p2)
    c, mass = geometric_consensus(log_p1, log_p2)
    p_bar = arithmetic_mean(log_p1, log_p2)
    assert torch.allclose(c, p_bar - u_minus.square(), atol=1e-6)
    assert torch.allclose(c, (u_plus.square() - u_minus.square()) / 2, atol=1e-6)
    assert torch.allclose(p_bar, (u_plus.square() + u_minus.square()) / 2, atol=1e-6)
    assert torch.allclose(mass, 1 - u_minus.square().sum(-1), atol=1e-5)
    assert torch.allclose((u_plus * u_minus).sum(-1), torch.zeros(64), atol=1e-5)
    assert torch.allclose(squared_hellinger(log_p1, log_p2), 1 - mass, atol=1e-5)


def test_bounds_on_target_and_mass():
    """0 <= c(a) <= p_bar(a) for every token and 0 <= Z <= 1 (Section 4.2)."""
    for seed in range(5):
        log_p1 = random_log_probs((200, 9), seed=seed, scale=4.0)
        log_p2 = random_log_probs((200, 9), seed=seed + 100, scale=4.0)
        c, mass = geometric_consensus(log_p1, log_p2)
        assert (c >= 0).all()
        assert (c <= arithmetic_mean(log_p1, log_p2) + 1e-7).all()
        assert (mass >= 0).all() and (mass <= 1 + 1e-6).all()
        js = js_divergence(log_p1, log_p2)
        assert (js >= -1e-7).all() and (js <= math.log(2) + 1e-6).all()


def test_consensus_is_the_unnormalised_barycenter():
    """F(v) = 0.5 sum_m D_I(v || p_m) = D_I(v || c) + 1 - Z, minimised at v = c (Eq. 6, 23)."""
    log_p1, log_p2 = random_log_probs((13,), seed=4), random_log_probs((13,), seed=5)
    p1, p2 = log_p1.exp(), log_p2.exp()
    c, mass = geometric_consensus(log_p1, log_p2)

    def objective(v: torch.Tensor) -> torch.Tensor:
        return 0.5 * (generalized_kl(v, p1) + generalized_kl(v, p2))

    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        v = torch.rand(13, generator=generator) * 0.3
        assert objective(v).item() == pytest.approx(
            (generalized_kl(v, c) + 1 - mass).item(), abs=1e-5
        )
        assert objective(v) >= objective(c) - 1e-6
    assert objective(c).item() == pytest.approx(1 - mass.item(), abs=1e-5)


def test_normalised_pool_identity_on_the_simplex():
    """0.5 sum_m KL(v || p_m) = KL(v || r) - log Z for probability vectors v (Eq. 24)."""
    log_p1, log_p2 = random_log_probs((7,), seed=6), random_log_probs((7,), seed=7)
    r, mass = normalized_consensus(log_p1, log_p2)
    log_v = random_log_probs((7,), seed=8)
    left = 0.5 * (kl_divergence(log_v, log_p1) + kl_divergence(log_v, log_p2))
    right = kl_divergence(log_v, r.log()) - mass.log()
    assert left.item() == pytest.approx(right.item(), abs=1e-5)


def test_softmax_of_averaged_logits_normalises_the_mass_away():
    """Averaging log-probabilities keeps Z; softmax of averaged logits equals r = c / Z."""
    z1, z2 = torch.randn(9, generator=torch.Generator().manual_seed(9)), torch.randn(9)
    log_p1, log_p2 = torch.log_softmax(z1, -1), torch.log_softmax(z2, -1)
    c, mass = geometric_consensus(log_p1, log_p2)
    pooled = torch.softmax((z1 + z2) / 2, dim=-1)
    assert mass.item() < 1.0
    assert pooled.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert torch.allclose(pooled, c / mass, atol=1e-6)
    assert not torch.allclose(pooled, c, atol=1e-3)


def test_target_is_detached_and_float32():
    log_p1 = random_log_probs((3, 5), seed=10).double().requires_grad_(True)
    log_p2 = random_log_probs((3, 5), seed=11).double().requires_grad_(True)
    c, mass = geometric_consensus(log_p1, log_p2)
    assert not c.requires_grad and not mass.requires_grad
    assert c.dtype == torch.float32 and mass.dtype == torch.float32


def test_entropy_and_js_reference_values():
    assert entropy(log([0.5, 0.5])).item() == pytest.approx(math.log(2), abs=1e-6)
    assert entropy(log([1.0, 0.0])).item() == 0.0
    log_p = random_log_probs((4, 6), seed=12)
    assert torch.allclose(js_divergence(log_p, log_p), torch.zeros(4), atol=1e-6)
    assert torch.allclose(
        js_divergence(log_p, log_p.flip(-1)), js_divergence(log_p.flip(-1), log_p), atol=1e-6
    )
