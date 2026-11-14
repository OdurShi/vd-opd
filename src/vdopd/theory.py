"""Fixed-prefix analysis of the normalised consensus target (Section 4.4, Appendix B.8, D).

Small float64 NumPy implementations of the quantities in Theorem 1 and its corollaries, used
by the numerical checks in ``tests/test_theory.py`` and by the target-improvement
diagnostic of Appendix B.8. Nothing here enters the training objective.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def softmax(z: np.ndarray) -> np.ndarray:
    return np.exp(log_softmax(z))


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """``D_KL(p || q)`` for probability vectors with ``q > 0`` wherever ``p > 0``."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    support = p > 0
    return float(np.sum(p[support] * (np.log(p[support]) - np.log(q[support]))))


def center(v: np.ndarray) -> np.ndarray:
    """Apply ``P = I_d - 11^T / d``: remove the additive constant of a logit vector."""
    v = np.asarray(v, dtype=np.float64)
    return v - v.mean(axis=-1, keepdims=True)


def wrapper_shift(log_p0: np.ndarray, log_pt: np.ndarray) -> np.ndarray:
    """Centered shift ``delta = P (log p_T - log p_0)`` (Section 4.4, Equation 18)."""
    return center(np.asarray(log_pt, dtype=np.float64) - np.asarray(log_p0, dtype=np.float64))


def pooled_target(z0: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Normalised geometric pool ``r = softmax(z0 + delta / 2)`` (Theorem 1)."""
    return softmax(np.asarray(z0, dtype=np.float64) + 0.5 * np.asarray(delta, dtype=np.float64))


def geometric_pool(*views: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geometric consensus of ``K >= 1`` probability vectors: ``(c, Z, r)``.

    ``c(a) = (prod_m p_m(a))^(1/K)`` (Equation 4 for ``K = 2``; supplementary experiment
    E08 for ``K = 4``), ``Z = sum_a c(a)`` (Equation 5) and ``r = c / Z``. The product is
    formed as a mean of log-probabilities, so it neither underflows for small
    probabilities nor loses the exact zero of a token that one view excludes. Views may
    carry leading batch dimensions. At ``Z = 0`` the normalised target is undefined and
    ``r`` is returned as zeros.
    """
    if not views:
        raise ValueError("at least one view is required")
    stacked = np.stack([np.asarray(view, dtype=np.float64) for view in views])
    with np.errstate(divide="ignore"):
        c = np.exp(np.log(stacked).mean(axis=0))
    mass = c.sum(axis=-1)
    denominator = np.expand_dims(mass, -1)
    r = np.divide(c, denominator, out=np.zeros_like(c), where=denominator > 0)
    return c, mass, r


def consensus_mass(p0: np.ndarray, pt: np.ndarray, *more_views: np.ndarray) -> float:
    """``Z = sum_a sqrt(p0(a) pT(a))`` (Equation 5); with further views, ``Z_K`` of E08."""
    return float(geometric_pool(p0, pt, *more_views)[1])


def two_token_example() -> list[dict[str, Any]]:
    """Two-token illustration of Appendix D.1 (supplementary experiment E18).

    Two pairs of teacher views over a two-token vocabulary share the normalised target
    ``r = (1/2, 1/2)`` but differ in the retained mass: identical uncertain views keep
    ``Z = 1``, confident views that select opposite tokens keep
    ``Z = 2 sqrt(0.99 * 0.01) = 0.199`` (three decimals). Each row holds ``case``, the two
    views ``p1`` and ``p2``, the normalised consensus ``r`` and the mass ``z``.
    """
    cases = (
        ("Shared uncertainty", (0.5, 0.5), (0.5, 0.5)),
        ("Contradictory confidence", (0.99, 0.01), (0.01, 0.99)),
    )
    rows = []
    for name, p1, p2 in cases:
        _, mass, r = geometric_pool(p1, p2)
        rows.append(
            {"case": name, "p1": p1, "p2": p2, "r": tuple(float(v) for v in r), "z": float(mass)}
        )
    return rows


def tilt_gap(p: np.ndarray, w: np.ndarray) -> float:
    """``K_p(w) = log sum_a p_a e^{w_a} - sum_a p_a w_a`` (Equation 26)."""
    p, w = np.asarray(p, dtype=np.float64), np.asarray(w, dtype=np.float64)
    shift = w.max()
    return float(np.log(np.sum(p * np.exp(w - shift))) + shift - np.sum(p * w))


def oscillation(w: np.ndarray) -> float:
    """``osc(w) = max_a w_a - min_a w_a``."""
    return float(np.max(w) - np.min(w))


def variance(p: np.ndarray, w: np.ndarray) -> float:
    """``Var_p(w) = sum_a p_a (w_a - sum_j p_j w_j)^2``."""
    p, w = np.asarray(p, dtype=np.float64), np.asarray(w, dtype=np.float64)
    return float(np.sum(p * (w - np.sum(p * w)) ** 2))


def mean_correction(p_star: np.ndarray, p0: np.ndarray, mu: np.ndarray) -> float:
    """``G = 0.5 <p* - p0, mu>`` (Theorem 1)."""
    return float(0.5 * np.dot(np.asarray(p_star) - np.asarray(p0), np.asarray(mu)))


def curvature_cost(
    z0: np.ndarray, deltas: np.ndarray, weights: Sequence[float] | None = None
) -> float:
    """``C = E D_KL(p0 || r)`` over the wrapper shifts ``deltas`` (rows), Theorem 1."""
    deltas = np.atleast_2d(np.asarray(deltas, dtype=np.float64))
    probabilities = _weights(len(deltas), weights)
    p0 = softmax(z0)
    return float(
        sum(w * kl(p0, pooled_target(z0, d)) for w, d in zip(probabilities, deltas, strict=True))
    )


def expected_pooling_gain(
    p_star: np.ndarray, z0: np.ndarray, deltas: np.ndarray, weights: Sequence[float] | None = None
) -> float:
    """``E E(r) - E(p0)`` with ``E(q) = D_KL(p* || q)``; Theorem 1 states it equals ``C - G``."""
    deltas = np.atleast_2d(np.asarray(deltas, dtype=np.float64))
    probabilities = _weights(len(deltas), weights)
    pooled = sum(
        w * kl(p_star, pooled_target(z0, d)) for w, d in zip(probabilities, deltas, strict=True)
    )
    return float(pooled - kl(p_star, softmax(z0)))


def _weights(count: int, weights: Sequence[float] | None) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count)
    array = np.asarray(weights, dtype=np.float64)
    return array / array.sum()


# ----------------------------------------------------------------- envelopes (Appendix D)
def range_envelope(r: float) -> float:
    """``Psi(R) = log((e^R - 1) / R) - 1 + R / (e^R - 1)`` with ``Psi(0) = 0`` (Equation 33)."""
    if r < 1e-6:
        return r * r / 8.0
    return float(np.log(np.expm1(r) / r) - 1.0 + r / np.expm1(r))


def _g(r: float, u: float) -> float:
    return float(np.log1p(u * np.expm1(r)) - r * u)


def range_variance_envelopes(r: float, s: float) -> tuple[float, float]:
    """``(F_-(R, S), F_+(R, S))`` of Equation 33 for ``0 <= S <= R^2 / 4``."""
    if r <= 0:
        return 0.0, 0.0
    root = np.sqrt(max(0.0, 1.0 - 4.0 * s / (r * r)))
    return _g(r, (1.0 + root) / 2.0), _g(r, (1.0 - root) / 2.0)


def phi_plus(length: float) -> float:
    """``phi_+(L) = (e^L - 1 - L) / L^2`` with ``phi_+(0) = 1/2`` (Corollary 6)."""
    return 0.5 if length < 1e-4 else float((np.expm1(length) - length) / length**2)


def phi_minus(length: float) -> float:
    """``phi_-(L) = (L - 1 + e^{-L}) / L^2`` with ``phi_-(0) = 1/2`` (Corollary 6)."""
    return 0.5 if length < 1e-4 else float((length + np.expm1(-length)) / length**2)


def binary_minimax_risk(bound: float) -> float:
    """``R(B) = (B/2) tanh(B/2) - log cosh(B/2)`` (Theorem 8)."""
    x = bound / 2.0
    return float(x * np.tanh(x) - np.log(np.cosh(x)))


# ------------------------------------------- target-improvement diagnostic (Appendix B.8)
def smoothed_reference(
    correct_index: int, num_candidates: int = 4, epsilon: float = 0.01
) -> np.ndarray:
    """Smoothed ground-truth distribution over candidates (Equation 17)."""
    p_star = np.full(num_candidates, epsilon / (num_candidates - 1))
    p_star[correct_index] = 1.0 - epsilon
    return p_star


def cross_fit_margins(
    p_star: np.ndarray, z0: np.ndarray, z_wrappers: np.ndarray
) -> list[dict[str, float]]:
    """Leave-one-wrapper-out cross-fitting at one fixed prefix (Equations 18-20).

    Args:
        p_star: reference distribution over the candidates.
        z0: candidate logits of the original teacher view.
        z_wrappers: ``[V, K]`` candidate logits under ``V`` unseen wrappers.

    Returns:
        One record per held-out wrapper ``h`` with the fitted correction ``G_hat``, curvature
        cost ``C_hat``, margin ``M_hat = G_hat - C_hat`` (all estimated from the other
        wrappers) and the held-out improvement
        ``delta_kl_held = D_KL(p* || p0) - D_KL(p* || r_held)``.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    z_wrappers = np.atleast_2d(np.asarray(z_wrappers, dtype=np.float64))
    p0 = softmax(z0)
    deltas = center(z_wrappers - z0)  # Equation 18
    records = []
    for held in range(len(deltas)):
        fit = np.delete(deltas, held, axis=0)
        g_hat = mean_correction(p_star, p0, fit.mean(axis=0))
        c_hat = float(np.mean([kl(p0, pooled_target(z0, d)) for d in fit]))
        r_held = pooled_target(z0, deltas[held])
        records.append(
            {
                "g_hat": g_hat,
                "c_hat": c_hat,
                "margin": g_hat - c_hat,
                "delta_kl_held": kl(p_star, p0) - kl(p_star, r_held),
            }
        )
    return records


def rescale_mean_shift(deltas: np.ndarray, beta: float) -> np.ndarray:
    """Mean-correction intervention ``delta_beta = beta * mu + (delta - mu)`` (Equation 22)."""
    deltas = np.atleast_2d(np.asarray(deltas, dtype=np.float64))
    mu = deltas.mean(axis=0, keepdims=True)
    return beta * mu + (deltas - mu)
