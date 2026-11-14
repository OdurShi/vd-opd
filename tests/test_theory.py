"""Numerical checks of the theoretical statements (Section 4.4, Appendix D)."""

from __future__ import annotations

import numpy as np
import pytest

from vdopd import theory as th


def random_state(rng, d=6, scale=1.5):
    z0 = scale * rng.normal(size=d)
    p_star = th.softmax(scale * rng.normal(size=d))
    return z0, p_star


def test_theorem1_exact_improvement_identity():
    """E E(r) - E(p0) = C - G with r = softmax(z0 + delta / 2)."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        z0, p_star = random_state(rng)
        deltas = th.center(rng.normal(size=(5, 6)))
        weights = rng.random(5)
        p0 = th.softmax(z0)
        mu = np.average(deltas, axis=0, weights=weights)
        gain = th.expected_pooling_gain(p_star, z0, deltas, weights)
        cost = th.curvature_cost(z0, deltas, weights)
        correction = th.mean_correction(p_star, p0, mu)
        assert gain == pytest.approx(cost - correction, abs=1e-12)
        assert (gain < 0) == (correction > cost)  # pooling improves iff G > C
        assert cost >= 0


def test_normalised_geometric_pool_is_softmax_of_average_logits():
    rng = np.random.default_rng(1)
    z0, delta = rng.normal(size=5), th.center(rng.normal(size=5))
    p0, pt = th.softmax(z0), th.softmax(z0 + delta)
    c = np.sqrt(p0 * pt)
    assert np.allclose(c / c.sum(), th.pooled_target(z0, delta))
    assert np.allclose(th.wrapper_shift(np.log(p0), np.log(pt)), delta)
    # Equation 53: log Z = A(z0 + w) - [A(z0) + A(z0 + 2w)] / 2 with w = delta / 2.
    log_partition = lambda z: np.log(np.exp(z).sum())  # noqa: E731
    expected = log_partition(z0 + delta / 2) - 0.5 * (log_partition(z0) + log_partition(z0 + delta))
    assert np.log(th.consensus_mass(p0, pt)) == pytest.approx(expected, abs=1e-12)
    # D_KL(p0 || r) = K_{p0}(delta / 2)
    assert th.kl(p0, th.pooled_target(z0, delta)) == pytest.approx(
        th.tilt_gap(p0, delta / 2), abs=1e-12
    )


def test_corollary5_centered_noise_cannot_improve_the_teacher():
    rng = np.random.default_rng(2)
    z0, p_star = random_state(rng)
    noise = rng.normal(size=(8, 6))
    deltas = th.center(noise - noise.mean(axis=0))  # mu = 0
    assert np.allclose(deltas.mean(axis=0), 0, atol=1e-12)
    gain = th.expected_pooling_gain(p_star, z0, deltas)
    assert gain == pytest.approx(th.curvature_cost(z0, deltas), abs=1e-12)
    assert gain > 0  # strictly worse whenever delta != 0 with positive probability
    # Equation 57 with a non-zero mean: C = KL(p0 || p_mu) + E KL(p_mu || r).
    deltas = deltas + th.center(rng.normal(size=6))
    mu = deltas.mean(axis=0)
    p0, p_mu = th.softmax(z0), th.pooled_target(z0, mu)
    spread = np.mean([th.kl(p_mu, th.pooled_target(z0, d)) for d in deltas])
    assert th.curvature_cost(z0, deltas) == pytest.approx(th.kl(p0, p_mu) + spread, abs=1e-12)
    pooled_error = np.mean([th.kl(p_star, th.pooled_target(z0, d)) for d in deltas])
    assert pooled_error == pytest.approx(th.kl(p_star, p_mu) + spread, abs=1e-12)


def test_lemma3_tilted_variance_representation_and_bounds():
    rng = np.random.default_rng(3)
    for _ in range(20):
        p = th.softmax(rng.normal(size=5))
        w = 2.0 * rng.normal(size=5)
        grid = np.linspace(0.0, 1.0, 4001)
        tilted = [th.softmax(np.log(p) + t * w) for t in grid]
        variances = np.array([th.variance(q, w) for q in tilted])
        integrand = (1 - grid) * variances
        integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid)))
        gap = th.tilt_gap(p, w)
        assert gap == pytest.approx(integral, abs=1e-6)
        assert 0 <= gap <= th.oscillation(w) ** 2 / 8 + 1e-12
        assert th.oscillation(w) ** 2 / 8 <= np.sum(th.center(w) ** 2) / 4 + 1e-12
    assert th.tilt_gap(p, np.full(5, 3.7)) == pytest.approx(0.0, abs=1e-12)  # K = 0 iff Pw = 0


def test_lemma4_range_variance_envelopes():
    rng = np.random.default_rng(4)
    for _ in range(200):
        p = th.softmax(rng.normal(size=4))
        w = 3.0 * rng.normal(size=4)
        r, s = th.oscillation(w), th.variance(p, w)
        lower, upper = th.range_variance_envelopes(r, s)
        gap = th.tilt_gap(p, w)
        assert lower - 1e-10 <= gap <= upper + 1e-10 <= th.range_envelope(r) + 2e-10
    # Sharpness (Equation 52): two-point perturbations attain both envelopes.
    r, s = 2.0, 0.6
    root = np.sqrt(1 - 4 * s / r**2)
    thetas = ((1 + root) / 2, (1 - root) / 2)
    for theta, bound in zip(thetas, th.range_variance_envelopes(r, s), strict=True):
        p, w = np.array([1 - theta, theta]), np.array([0.0, r])
        assert th.variance(p, w) == pytest.approx(s) and th.tilt_gap(p, w) == pytest.approx(bound)
    # Psi(R) = R^2 / 8 + O(R^4), nondecreasing, and Psi(R) <= R^2 / 8.
    values = [th.range_envelope(x) for x in np.linspace(0.0, 6.0, 61)]
    assert all(a <= b + 1e-12 for a, b in zip(values[:-1], values[1:], strict=True))
    assert th.range_envelope(0.05) == pytest.approx(0.05**2 / 8, rel=1e-3)
    assert all(th.range_envelope(x) <= x * x / 8 + 1e-12 for x in np.linspace(0.01, 6.0, 50))


def test_corollary6_distribution_sensitive_bounds():
    rng = np.random.default_rng(5)
    for _ in range(50):
        z0 = rng.normal(size=5)
        deltas = th.center(rng.normal(size=(6, 5)))
        p0 = th.softmax(z0)
        length = max(th.oscillation(d / 2) for d in deltas)
        w_bar = np.mean([th.variance(p0, d) for d in deltas])
        cost = th.curvature_cost(z0, deltas)
        assert th.phi_minus(length) * w_bar / 4 - 1e-12 <= cost
        assert cost <= th.phi_plus(length) * w_bar / 4 + 1e-12
        assert abs(cost - w_bar / 8) <= length * np.exp(length) * w_bar / 24 + 1e-12
        # W = mu^T H0 mu + tr(H0 Sigma) (Equation 66)
        h0 = np.diag(p0) - np.outer(p0, p0)
        mu = deltas.mean(axis=0)
        sigma = (deltas - mu).T @ (deltas - mu) / len(deltas)
        assert w_bar == pytest.approx(mu @ h0 @ mu + np.trace(h0 @ sigma), abs=1e-12)
    assert th.phi_plus(0.0) == th.phi_minus(0.0) == 0.5
    assert th.phi_minus(1.0) < 0.5 < th.phi_plus(1.0)


def test_proposition7_local_fisher_expansion():
    rng = np.random.default_rng(6)
    z0, p_star = random_state(rng)
    directions = th.center(rng.normal(size=(4, 6)))
    p0 = th.softmax(z0)
    mean_variance = np.mean([th.variance(p0, x) for x in directions])
    for t in (1e-2, 1e-3):
        cost = th.curvature_cost(z0, t * directions)
        assert cost == pytest.approx(t**2 / 8 * mean_variance, rel=5 * t)
        gain = th.expected_pooling_gain(p_star, z0, t * directions)
        linear = t / 2 * np.dot(p0 - p_star, directions.mean(axis=0))
        assert gain == pytest.approx(linear + t**2 / 8 * mean_variance, abs=5 * t**3)


def bernoulli_kl(p: float, q: float) -> float:
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def test_theorem8_binary_minimax_risk():
    sigmoid = lambda s: 1 / (1 + np.exp(-s))  # noqa: E731
    grid = np.linspace(1e-4, 1 - 1e-4, 20001)
    for bound in (0.5, 1.0, 2.5):
        radius = th.binary_minimax_risk(bound)

        def information_radius(t: float, bound=bound) -> float:
            low, high = sigmoid(t - bound), sigmoid(t + bound)
            return float(np.min(np.maximum(bernoulli_kl(low, grid), bernoulli_kl(high, grid))))

        # The worst observed teacher is uniform (t = 0), where the radius equals R(B).
        assert information_radius(0.0) == pytest.approx(radius, abs=1e-6)
        assert radius == pytest.approx(bernoulli_kl(sigmoid(bound), 0.5), abs=1e-12)
        for t in (0.5, 1.0, 3.0, -2.0):
            assert information_radius(t) <= radius + 1e-9
    assert th.binary_minimax_risk(0.0) == 0.0
    assert th.binary_minimax_risk(0.1) == pytest.approx(0.1**2 / 8 - 0.1**4 / 64, abs=1e-8)


def test_proposition9_agreement_is_not_a_correctness_certificate():
    rng = np.random.default_rng(7)
    p_star, p0, pt = (th.softmax(rng.normal(size=5)) for _ in range(3))
    c = np.sqrt(p0 * pt)
    r, mass = c / c.sum(), c.sum()
    assert th.kl(p_star, r) == pytest.approx(
        0.5 * (th.kl(p_star, p0) + th.kl(p_star, pt)) + np.log(mass), abs=1e-12
    )
    eps = 1e-3
    # Z = 1 with a large target error: both views share the same wrong belief.
    wrong = np.array([eps, 1 - eps])
    assert th.consensus_mass(wrong, wrong) == pytest.approx(1.0)
    assert th.kl(np.array([1 - eps, eps]), wrong) == pytest.approx(
        (1 - 2 * eps) * np.log((1 - eps) / eps)
    )
    # Tiny Z with zero target error: opposite confident views pool to the uniform target.
    a, b = np.array([1 - eps, eps]), np.array([eps, 1 - eps])
    assert th.consensus_mass(a, b) == pytest.approx(2 * np.sqrt(eps * (1 - eps)))
    pooled = np.sqrt(a * b) / np.sqrt(a * b).sum()
    assert th.kl(np.array([0.5, 0.5]), pooled) == pytest.approx(0.0, abs=1e-12)


def test_target_improvement_diagnostic_quantities():
    """Appendix B.8: Equations 17-20 and 22."""
    p_star = th.smoothed_reference(2)
    assert p_star.tolist() == pytest.approx([0.01 / 3, 0.01 / 3, 0.99, 0.01 / 3])
    rng = np.random.default_rng(8)
    z0 = rng.normal(size=4)
    z_wrappers = z0 + rng.normal(size=(4, 4)) + np.array([0.0, 0.0, 0.8, 0.0])
    records = th.cross_fit_margins(p_star, z0, z_wrappers)
    assert len(records) == 4
    p0 = th.softmax(z0)
    deltas = th.center(z_wrappers - z0)
    fit = deltas[1:]
    assert records[0]["g_hat"] == pytest.approx(0.5 * np.dot(p_star - p0, fit.mean(axis=0)))
    assert records[0]["c_hat"] == pytest.approx(
        np.mean([th.kl(p0, th.pooled_target(z0, d)) for d in fit])
    )
    assert records[0]["margin"] == pytest.approx(records[0]["g_hat"] - records[0]["c_hat"])
    held = np.sqrt(p0 * th.softmax(z_wrappers[0]))
    assert records[0]["delta_kl_held"] == pytest.approx(
        th.kl(p_star, p0) - th.kl(p_star, held / held.sum())
    )
    # delta_beta = beta * mu + (delta - mu): the mean correction scales linearly with beta.
    mu = deltas.mean(axis=0)
    for beta in (0.0, 0.5, 2.0):
        rescaled = th.rescale_mean_shift(deltas, beta)
        assert np.allclose(rescaled.mean(axis=0), beta * mu)
        assert np.allclose(rescaled - rescaled.mean(axis=0), deltas - mu)
        assert th.mean_correction(p_star, p0, rescaled.mean(axis=0)) == pytest.approx(
            beta * th.mean_correction(p_star, p0, mu)
        )
