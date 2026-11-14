"""Square-root geometry of two teacher views and the geometric consensus target.

Implements Equations 1 and 3-6 of the paper together with the identities of Appendix D.1
and D.2. All functions take *log-probabilities* over the last dimension, work on arbitrary
leading dimensions, use natural logarithms, and reduce in float32 (Appendix A.2 and B.4).
Zero probabilities are represented by ``-inf`` log-probabilities and handled with the
convention ``0 log 0 = 0``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_SQRT_HALF = 1.0 / math.sqrt(2.0)


def _xlogy_from_log(log_p: Tensor, log_ratio: Tensor) -> Tensor:
    """Return ``p * log_ratio`` with the convention ``0 * (anything) = 0``."""
    p = log_p.exp()
    return torch.where(p > 0, p * log_ratio, torch.zeros_like(p))


def entropy(log_p: Tensor) -> Tensor:
    """Shannon entropy ``H(p) = -sum_a p(a) log p(a)`` in nats (Section 3)."""
    log_p = log_p.float()
    return -_xlogy_from_log(log_p, log_p).sum(-1)


def kl_divergence(log_p: Tensor, log_q: Tensor) -> Tensor:
    """``D_KL(p || q) = sum_a p(a) log[p(a) / q(a)]`` in nats (Section 3)."""
    log_p, log_q = log_p.float(), log_q.float()
    return _xlogy_from_log(log_p, log_p - log_q).sum(-1)


def js_divergence(log_p1: Tensor, log_p2: Tensor) -> Tensor:
    """Jensen-Shannon divergence between two views, Equation 1.

    ``D_JS(p1, p2) = 0.5 KL(p1 || m) + 0.5 KL(p2 || m)`` with ``m = (p1 + p2) / 2``.
    The result lies in ``[0, ln 2]``.
    """
    log_p1, log_p2 = log_p1.float(), log_p2.float()
    log_m = torch.logaddexp(log_p1, log_p2) - math.log(2.0)
    return 0.5 * kl_divergence(log_p1, log_m) + 0.5 * kl_divergence(log_p2, log_m)


def view_decomposition(log_p1: Tensor, log_p2: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric and differential square-root coordinates, Equation 3.

    ``[u_plus, u_minus] = [sqrt(p1), sqrt(p2)] R`` with the orthogonal matrix
    ``R = (1/sqrt 2) [[1, 1], [1, -1]]``, i.e. ``u_pm = (sqrt(p1) +- sqrt(p2)) / sqrt(2)``.
    """
    s1 = (0.5 * log_p1.float()).exp()
    s2 = (0.5 * log_p2.float()).exp()
    return (s1 + s2) * _SQRT_HALF, (s1 - s2) * _SQRT_HALF


def arithmetic_mean(log_p1: Tensor, log_p2: Tensor) -> Tensor:
    """Arithmetic pool ``p_bar = (p1 + p2) / 2`` (Section 4.2, Table 4)."""
    return 0.5 * (log_p1.float().exp() + log_p2.float().exp())


def geometric_consensus(log_p1: Tensor, log_p2: Tensor) -> tuple[Tensor, Tensor]:
    """Unnormalised geometric consensus target and its retained mass (Equations 4 and 5).

    The target is formed exactly as in Appendix A.2, by averaging full-vocabulary
    log-probabilities, ``c = exp{(log p1 + log p2) / 2} = sqrt(p1 * p2)``. Averaging
    log-probabilities (rather than applying a softmax to averaged logits) is what retains
    the mass ``Z = 1^T c = <sqrt(p1), sqrt(p2)>`` (the Bhattacharyya affinity) in ``[0, 1]``.

    Args:
        log_p1: log-probabilities of the first (original-prompt) teacher view, ``[..., d]``.
        log_p2: log-probabilities of the second (wrapped-prompt) teacher view, ``[..., d]``.

    Returns:
        ``(c, Z)`` with shapes ``[..., d]`` and ``[...]``, both float32 and detached from
        the teacher graph.
    """
    log_c = 0.5 * (log_p1.detach().float() + log_p2.detach().float())
    c = log_c.exp()
    return c, c.sum(-1)


def bhattacharyya_affinity(log_p1: Tensor, log_p2: Tensor) -> Tensor:
    """``Z = <sqrt(p1), sqrt(p2)>`` (Equation 5); equals ``geometric_consensus(...)[1]``."""
    return geometric_consensus(log_p1, log_p2)[1]


def squared_hellinger(log_p1: Tensor, log_p2: Tensor) -> Tensor:
    """Squared Hellinger distance ``0.5 ||sqrt(p1) - sqrt(p2)||^2 = 1 - Z`` (Section 4.2)."""
    _, u_minus = view_decomposition(log_p1, log_p2)
    return u_minus.square().sum(-1)


def generalized_kl(v: Tensor, w: Tensor) -> Tensor:
    """Generalised (I-)divergence ``D_I(v || w) = sum_a [v log(v / w) - v + w]``.

    Defined for non-negative ``v`` and ``w`` (Section 4.2), with ``0 log(0 / w) = 0`` and,
    following Appendix D.2, ``v log(v / 0) = +inf`` for ``v > 0``.
    """
    v, w = v.float(), w.float()
    ratio_term = torch.where(v > 0, v * (v.log() - w.log()), torch.zeros_like(v))
    return (ratio_term - v + w).sum(-1)


def normalized_consensus(log_p1: Tensor, log_p2: Tensor) -> tuple[Tensor, Tensor]:
    """Normalised geometric target ``r = c / Z`` (the logarithmic opinion pool) and ``Z``.

    At ``Z = 0`` the normalised target is undefined; this function then returns zeros for
    ``r`` so that the corresponding position contributes nothing downstream.
    """
    c, mass = geometric_consensus(log_p1, log_p2)
    safe = mass.clamp_min(torch.finfo(c.dtype).tiny).unsqueeze(-1)
    return torch.where(mass.unsqueeze(-1) > 0, c / safe, torch.zeros_like(c)), mass
